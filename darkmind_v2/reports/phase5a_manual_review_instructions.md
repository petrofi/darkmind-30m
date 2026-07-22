# Phase 5A manual review instructions

## Review artifact

- Packet: `C:\DarkMindRuntime\phase5a\manual_review\phase5a_manual_review_packet_v2.md`
- Blank score schema: `C:\DarkMindRuntime\phase5a\manual_review\phase5a_manual_review_schema.json`
- Checkpoint: `step_011972_tokens_098074624`
- Model SHA-256: `458816257836a60d804a373c17c617642c99e413c6c190d4fd1e2f73b95fd993`
- Prompt manifest SHA-256: `c82db48e4276d4a9a4d90ea1752956a55848869c55ea0e2ce590358eb39f9197`

The packet contains 150 blinded base-continuation outputs: 78 greedy and 72 seeded-sampling results, with 12 or 13 items from each category. It tests continuation behavior, not chat or instruction following.

## Scoring

Read the prompt and raw continuation before entering any score. Automatic semantic scores are intentionally absent.

| Field | Scale | Meaning |
| --- | ---: | --- |
| grammatical structure | 0-4 | sentence-level syntax and readability |
| topical consistency | 0-4 | continuation remains connected to the prompt |
| language consistency | 0-4 | expected language is maintained |
| completion quality | 0-4 | continuation develops or closes the passage coherently |
| repetition severity, reversed | 0-4 | 0 is severe repetition; 4 is no meaningful repetition |
| factual reliability, where applicable | 0-4 or N/A | claims remain supported by the supplied context |
| technical usefulness, where applicable | 0-4 or N/A | code or technical continuation is structurally useful |
| overall usability | 0-4 | practical base-continuation quality |
| reviewer note | free text | concise explanation of the main success or failure |

Use `N/A` only for the two conditional fields. Do not infer truth from fluent wording, and do not reward a response for behaving like a chatbot. Repetition, loop, and EOS flags are mechanical warnings, not semantic scores.

## Integrity rules

- Keep the raw prompt and continuation unchanged.
- Fill only the blank human-score fields and reviewer note.
- Review both decoding modes by the same standard.
- Record uncertainty in the note instead of guessing.
- Preserve the original packet and schema; create a separately named scored copy for completed review.

No manual scores were populated during Phase 5A. Therefore the automatic report can support health diagnosis, but it cannot establish factual reliability, semantic quality, or SFT readiness by itself.
