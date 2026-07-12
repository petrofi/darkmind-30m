# Phase 2B.2 Staging Review

## Status

- Branch: `codex/darkmind-v2-phase2a-tiny-smoke`
- Staging area: empty
- Runtime data, runs, checkpoints, exports, profiling, caches, and logs: ignored
- Frozen tokenizer artifact/stat changes: deliberately excluded
- `WORKTREE_MAP.md`: deliberately excluded

## Recommended Explicit Commands

Run these only after user approval. Do not replace them with `git add .`.

```powershell
git add -- .gitattributes .gitignore darkmind_v2/README.md

git add -- darkmind_v2/config/base_quality_gates.json darkmind_v2/config/corpus_attribution_summary.json darkmind_v2/config/model_tiny_smoke.json darkmind_v2/config/public_research_preview_gates.json darkmind_v2/config/train_tiny_smoke.json darkmind_v2/config/train_tiny_stage1.json darkmind_v2/config/train_tiny_stage1_r2.json

git add -- darkmind_v2/data_pipeline/tokenize_corpus.py darkmind_v2/data_pipeline/tokenize_phase1b_corpus.py darkmind_v2/data_pipeline/tokenized_manifest.py darkmind_v2/data_pipeline/validate_full_tokenized_corpus.py darkmind_v2/data_pipeline/validate_tokenized_shards.py

git add -- darkmind_v2/eval/public_preview_prompts.jsonl darkmind_v2/evaluation/audit_public_preview.py darkmind_v2/evaluation/diagnose_initial_generation.py darkmind_v2/evaluation/diagnose_seeded_sampling_bytes.py darkmind_v2/evaluation/evaluate_base_model.py darkmind_v2/evaluation/evaluate_stage1_run.py darkmind_v2/evaluation/generate_fixed_prompts.py darkmind_v2/evaluation/trace_byte_fallback.py darkmind_v2/evaluation/validate_generation_health.py darkmind_v2/evaluation/validate_stage1_artifacts.py

git add -- darkmind_v2/export/LICENSE_INFORMATION.md darkmind_v2/export/MODEL_CARD_STAGE1.md darkmind_v2/export/README.md darkmind_v2/export/export_huggingface.py darkmind_v2/export/export_stage1_huggingface.py darkmind_v2/export/tokenization_darkmind_v2.py darkmind_v2/export/validate_huggingface_export.py darkmind_v2/export/validate_public_preview_package.py darkmind_v2/export/validate_stage1_huggingface.py

git add -- darkmind_v2/modeling/configuration_darkmind_v2.py darkmind_v2/modeling/model_io.py darkmind_v2/modeling/modeling_darkmind_v2.py darkmind_v2/modeling/validate_model_config.py darkmind_v2/tokenizer/load_frozen_tokenizer.py

git add -- darkmind_v2/training/calibrate_tiny_stage1.py darkmind_v2/training/checkpointing.py darkmind_v2/training/probe_training_environment.py darkmind_v2/training/token_shard_dataset.py darkmind_v2/training/train_tiny_smoke.py darkmind_v2/training/train_tiny_stage1.py darkmind_v2/training/training_state.py darkmind_v2/training/validate_stage1_config.py darkmind_v2/training/validate_training_config.py

git add -- darkmind_v2/tests/fixtures/phase2a_tiny_corpus.jsonl darkmind_v2/tests/test_phase2a_model.py darkmind_v2/tests/test_phase2a_tokenized_shards.py darkmind_v2/tests/test_phase2a_tokenizer.py darkmind_v2/tests/test_phase2a_training_evaluation_export.py darkmind_v2/tests/test_phase2b1_generation_policy.py darkmind_v2/tests/test_phase2b1_tokenization_config.py darkmind_v2/tests/test_phase2b2_public_preview.py darkmind_v2/tests/test_phase2b2_release_package.py

git add -- darkmind_v2/reports/phase2a_hardware_budget.md darkmind_v2/reports/phase2a_huggingface_release_plan.md darkmind_v2/reports/phase2a_tiny_architecture.md darkmind_v2/reports/phase2a_tokenization_plan.md darkmind_v2/reports/phase2b1_checkpoint_comparison.md darkmind_v2/reports/phase2b1_evaluation.md darkmind_v2/reports/phase2b1_huggingface_export.md darkmind_v2/reports/phase2b1_initial_generation_diagnosis.md darkmind_v2/reports/phase2b1_seeded_sampling_byte_diagnosis.md darkmind_v2/reports/phase2b1_training_run.md darkmind_v2/reports/phase2b2_generation_samples.md darkmind_v2/reports/phase2b2_public_preview_audit.md darkmind_v2/reports/phase2b2_release_package_validation.md darkmind_v2/reports/phase2b2_staging_review.md
```

## Post-Staging Review

After explicit staging, inspect with:

```powershell
git status --short
git diff --cached --stat
git diff --cached --check
git diff --cached --name-only
```

Do not commit or push until the staged list is reviewed separately.
