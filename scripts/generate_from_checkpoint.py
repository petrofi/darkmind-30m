from pathlib import Path
import argparse
from datetime import datetime
import json
import random
import re
import sys

import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from model.gpt import GPTConfig, GPTLanguageModel, count_parameters


DEFAULT_CHECKPOINT = ROOT_DIR / "models" / "darkmind-30m-10k.pt"
DEFAULT_CONFIG = ROOT_DIR / "configs" / "darkmind_30m_1000step.json"
TOKENIZER_DIR = ROOT_DIR / "tokenizer" / "darkmind-tokenizer"
SPECIAL_STOP_TOKENS = ["</s>", "<s>", "<pad>", "<unk>", "<mask>", "<|end|>"]
TURN_STOP_MARKERS = [
    "\n\nKullanıcı:",
    "\nKullanıcı:",
    "\n\nSen:",
    "\nSen:",
    "\n\nSoru:",
    "\nSoru:",
    "<|end|>",
]
SAFE_FALLBACK = "Bu konuda yeterli bilgiye sahip olmayabilirim."


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def set_seed(seed: int, device: str) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def clean_special_tokens(output_text: str) -> str:
    for stop_token in SPECIAL_STOP_TOKENS:
        if stop_token in output_text:
            output_text = output_text.split(stop_token)[0]

    return output_text.strip()


def trim_at_turn_marker(text: str) -> str:
    for marker in TURN_STOP_MARKERS:
        if marker in text:
            text = text.split(marker)[0]
            break

    return text


def clean_generated_text(text: str) -> str:
    text = clean_special_tokens(text)
    text = trim_at_turn_marker(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(Asistan:)(\S)", r"\1 \2", text)
    text = re.sub(r"(Kullanıcı:)(\S)", r"\1 \2", text)
    return text.strip()


def fix_dialogue_spacing(output_text: str) -> str:
    return clean_generated_text(output_text)


def extract_answer(output_text: str, prompt_text: str) -> str:
    output_text = clean_special_tokens(output_text)

    if output_text.startswith(prompt_text):
        answer = output_text[len(prompt_text):]
    else:
        answer = output_text

    answer = clean_generated_text(answer)

    if answer.startswith("Asistan:"):
        answer = answer[len("Asistan:"):].strip()

    return answer


def stop_after_first_answer(output_text: str, prompt_text: str) -> str:
    answer = extract_answer(output_text, prompt_text)
    return f"{prompt_text}{answer}".strip()


def load_tokenizer(tokenizer_dir: Path = TOKENIZER_DIR) -> ByteLevelBPETokenizer:
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"

    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(f"Tokenizer files not found in {tokenizer_dir}")

    return ByteLevelBPETokenizer(str(vocab_path), str(merges_path))


def build_config_from_file(config_path: Path, vocab_size: int) -> GPTConfig:
    cfg = load_json(config_path)
    model_cfg = cfg["model"]

    return GPTConfig(
        vocab_size=vocab_size,
        block_size=model_cfg["block_size"],
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        dropout=model_cfg["dropout"],
    )


def load_checkpoint_model(
    checkpoint_path: Path,
    device: str,
    config_path: Path | None = None,
    vocab_size: int | None = None,
) -> tuple[GPTLanguageModel, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_dict = checkpoint if isinstance(checkpoint, dict) else {}
    state_dict = checkpoint_dict.get("model_state_dict", checkpoint)

    if "config" in checkpoint_dict:
        config = GPTConfig(**checkpoint_dict["config"])
        config_source = "checkpoint"
    else:
        if config_path is None or vocab_size is None:
            raise ValueError(
                "Checkpoint has no embedded config. Provide --config and --tokenizer."
            )

        config = build_config_from_file(config_path, vocab_size)
        config_source = str(config_path)

    model = GPTLanguageModel(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    checkpoint_dict["resolved_config_source"] = config_source
    return model, checkpoint_dict


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty <= 1.0:
        return logits

    for token_id in set(generated_ids[0].tolist()):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= repetition_penalty
        else:
            logits[0, token_id] /= repetition_penalty

    return logits


def apply_no_repeat_ngram_blocking(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    ngram_size: int,
) -> torch.Tensor:
    if ngram_size <= 0:
        return logits

    token_ids = generated_ids[0].tolist()

    if len(token_ids) + 1 < ngram_size:
        return logits

    prefix_size = ngram_size - 1
    current_prefix = tuple(token_ids[-prefix_size:]) if prefix_size > 0 else tuple()
    banned_tokens = set()

    for index in range(0, len(token_ids) - ngram_size + 1):
        ngram = tuple(token_ids[index:index + ngram_size])

        if ngram[:-1] == current_prefix:
            banned_tokens.add(ngram[-1])

    for token_id in banned_tokens:
        logits[0, token_id] = -float("inf")

    return logits


def apply_top_p_filtering(logits: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return logits

    if top_p <= 0:
        raise ValueError("top_p must be greater than 0")

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_indices_to_remove,
    )

    return logits.masked_fill(indices_to_remove, -float("inf"))


@torch.no_grad()
def generate_with_controls(
    model: GPTLanguageModel,
    tokenizer: ByteLevelBPETokenizer,
    prompt_text: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    stop_on_user_marker: bool = True,
    top_p: float | None = 0.9,
    no_repeat_ngram_size: int = 3,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be greater than 0 and less than or equal to 1")

    if no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size must be at least 0")

    encoded = tokenizer.encode(prompt_text)
    idx = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    top_k_value = None if top_k <= 0 else min(top_k, model.config.vocab_size)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        logits = apply_repetition_penalty(logits, idx, repetition_penalty)
        logits = apply_no_repeat_ngram_blocking(
            logits,
            idx,
            no_repeat_ngram_size,
        )

        if top_k_value is not None:
            values, _ = torch.topk(logits, top_k_value)
            logits[logits < values[:, [-1]]] = -float("inf")

        logits = apply_top_p_filtering(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        if stop_on_user_marker:
            decoded_text = tokenizer.decode(idx[0].tolist())
            generated_part = decoded_text[len(prompt_text):] if decoded_text.startswith(prompt_text) else decoded_text

            if any(marker in generated_part for marker in TURN_STOP_MARKERS):
                break

    return tokenizer.decode(idx[0].tolist())


def write_samples_jsonl(
    output_path: Path,
    prompt: str,
    settings: dict,
    samples: list[dict[str, str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for index, sample in enumerate(samples, start=1):
            row = {
                "sample_index": index,
                "prompt": prompt,
                "settings": settings,
                "raw_output": sample["raw_output"],
                "answer": sample["answer"],
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate text from a DarkMind checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT.relative_to(ROOT_DIR)),
        help="Checkpoint path. Example: checkpoints/darkmind_30m.pt",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG.relative_to(ROOT_DIR)),
        help="DarkMind config used when the checkpoint has no embedded config.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=str(TOKENIZER_DIR.relative_to(ROOT_DIR)),
        help="Tokenizer directory.",
    )
    parser.add_argument("--prompt", type=str, default="DarkMind")
    parser.add_argument(
        "--dialogue",
        type=str,
        default=None,
        help="Dialogue mode. Example: --dialogue 'DarkMind hazır bir model mi?'",
    )
    parser.add_argument(
        "--chat_format",
        action="store_true",
        help="Wrap --prompt as 'Kullanıcı: ...\\nAsistan:' before generation.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stop_at_next_user",
        action="store_true",
        help="If enabled, stop generation when the next dialogue turn starts.",
    )
    parser.add_argument(
        "--stop_on_chat_turn",
        dest="stop_on_chat_turn",
        action="store_true",
        default=True,
        help="When chat format is enabled, stop when a new user turn starts.",
    )
    parser.add_argument(
        "--no_stop_on_chat_turn",
        dest="stop_on_chat_turn",
        action="store_false",
        help="Disable automatic chat-turn stopping for --chat_format.",
    )
    args = parser.parse_args()

    num_return_sequences = (
        args.num_samples
        if args.num_samples is not None
        else args.num_return_sequences
    )

    if num_return_sequences < 1:
        raise ValueError("--num_return_sequences must be at least 1")

    if not 0 < args.top_p <= 1:
        raise ValueError("--top_p must be greater than 0 and less than or equal to 1")

    if args.no_repeat_ngram_size < 0:
        raise ValueError("--no_repeat_ngram_size must be at least 0")

    if args.dialogue is not None:
        args.prompt = f"Kullanıcı: {args.dialogue.strip()}\nAsistan:"
        args.stop_at_next_user = True
    elif args.chat_format:
        args.prompt = f"Kullanıcı: {args.prompt.strip()}\nAsistan:"
        if args.stop_on_chat_turn:
            args.stop_at_next_user = True

    checkpoint_path = resolve_path(args.checkpoint)
    config_path = resolve_path(args.config)
    tokenizer_dir = resolve_path(args.tokenizer)
    output_path = resolve_path(args.output_path) if args.output_path else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(tokenizer_dir)
    vocab_size = len(load_json(tokenizer_dir / "vocab.json"))
    model, checkpoint = load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        device=device,
        config_path=config_path,
        vocab_size=vocab_size,
    )

    print("=" * 70)
    print("DarkMind checkpoint generation")
    print("=" * 70)
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Tokenizer: {tokenizer_dir}")
    print(f"Model config source: {checkpoint.get('resolved_config_source', 'unknown')}")
    print(f"Run name: {checkpoint.get('run_name', 'unknown')}")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Vocab size: {model.config.vocab_size}")
    print(f"Block size: {model.config.block_size}")

    settings = {
        "checkpoint": str(checkpoint_path),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "num_return_sequences": num_return_sequences,
        "seed": args.seed,
        "stop_at_next_user": args.stop_at_next_user,
        "stop_on_chat_turn": args.stop_on_chat_turn,
        "chat_format": args.chat_format,
    }

    samples = []

    for sample_index in range(num_return_sequences):
        set_seed(args.seed + sample_index, device)
        raw_output = generate_with_controls(
            model=model,
            tokenizer=tokenizer,
            prompt_text=args.prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop_on_user_marker=args.stop_at_next_user,
            top_p=args.top_p,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        answer = extract_answer(raw_output, args.prompt)
        samples.append({
            "raw_output": raw_output.strip() or SAFE_FALLBACK,
            "answer": answer or SAFE_FALLBACK,
        })

    if output_path:
        write_samples_jsonl(output_path, args.prompt, settings, samples)

    print("=" * 70)
    print("PROMPT:")
    print(args.prompt)

    for index, sample in enumerate(samples, start=1):
        print("=" * 70)
        print(f"RAW OUTPUT {index}:")
        print(sample["raw_output"])
        print("=" * 70)
        print(f"ASSISTANT ANSWER {index}:")
        print(sample["answer"])

    if output_path:
        print("=" * 70)
        print(f"Samples saved: {output_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
