# Teacher Generation Prompt Template

Generate {count} original {language_name} instruction-response examples for:

- category={category}
- difficulty={difficulty}
- language={language_code}

Return only a valid JSON array. Do not wrap in markdown.

Each item must include:

```json
{
  "prompt": "...",
  "response": "...",
  "source": "qwen3_vl_30b_teacher",
  "category": "{category}",
  "language": "{language_code}",
  "difficulty": "{difficulty}"
}
```

Rules:

- Responses must be 1-4 sentences.
- Keep answers practical and accurate.
- Use the requested language exactly.
- No OpenAI, ChatGPT, or Qwen identity claims.
- DarkMind identity only in identity examples.
- No private data, secrets, API keys, or token-looking strings.
- No unsafe cyber/offensive instructions.
- No medical diagnosis or therapy claims.
