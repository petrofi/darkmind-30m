import shutil

import pytest

from darkmind_v2.tokenizer.load_frozen_tokenizer import (
    DEFAULT_FROZEN_DIR,
    EXPECTED_HASHES,
    FrozenTokenizer,
    sha256_file,
    verify_frozen_tokenizer,
)


def test_frozen_tokenizer_hashes_and_special_ids() -> None:
    manifest = verify_frozen_tokenizer()
    assert manifest["vocab_size"] == 24000
    assert sha256_file(DEFAULT_FROZEN_DIR / "tokenizer.model") == EXPECTED_HASHES["tokenizer.model"]
    assert sha256_file(DEFAULT_FROZEN_DIR / "tokenizer.vocab") == EXPECTED_HASHES["tokenizer.vocab"]
    tokenizer = FrozenTokenizer()
    assert tokenizer.vocab_size == 24000
    assert [tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id] == [0, 1, 2, 3]


def test_roundtrip_and_document_boundaries() -> None:
    tokenizer = FrozenTokenizer()
    text = "Turkce ve English metin 123."
    token_ids = tokenizer.encode(text)
    assert tokenizer.decode(token_ids) == text
    document = tokenizer.encode_document(text, add_bos=True)
    assert document[0] == tokenizer.bos_token_id
    assert document[-1] == tokenizer.eos_token_id


def test_modified_frozen_artifact_is_rejected(tmp_path) -> None:
    copied = tmp_path / "darkmind_v2_sp_bpe24k_v1"
    shutil.copytree(DEFAULT_FROZEN_DIR, copied)
    with (copied / "tokenizer.vocab").open("a", encoding="utf-8") as handle:
        handle.write("\nmodified")
    with pytest.raises(ValueError, match="immutable tokenizer artifact changed"):
        FrozenTokenizer(copied)


def test_dynamic_vocabulary_operations_are_forbidden() -> None:
    tokenizer = FrozenTokenizer()
    with pytest.raises(RuntimeError):
        tokenizer.add_tokens(["new-token"])
    with pytest.raises(RuntimeError):
        tokenizer.train("anything")
