# DarkMind Roadmap

DarkMind is a Turkish-focused small LLM research and learning project, not a production-grade assistant. The roadmap is ambitious, but progress should be measured honestly with datasets, configs, checkpoints, eval runs, and documented limitations.

## Stage 0 - Working Tiny GPT

- Tiny GPT works
- Tokenizer works
- Local training works
- Checkpoint generation works
- Basic text generation works

## Stage 1 - Structured Small Dataset

- 100k-250k character Turkish dataset
- Basic chat identity
- Basic coding examples
- `darkmind_eval_v02`
- Dataset quality checks
- Human-readable docs for the training loop

## Stage 2 - Better Turkish/Code Dataset

- 1M+ character clean Turkish/code dataset
- Stronger code eval
- Improved correction candidates
- Deduplication
- Source metadata
- Manual approval flow for self-improvement examples
- Allowlisted web data pipeline

## Stage 3 - Larger Measurable Training

- 10M+ token Turkish dataset
- Better tokenizer
- Larger config experiments
- Train/validation/test split
- Benchmark dashboard
- Eval comparison reports
- Experiment registry discipline

## Stage 4 - Instruction And Retrieval Track

- Instruction tuning
- Coding assistant track
- RAG document helper
- Model card releases
- Safer factual update workflow through retrieval instead of constant retraining

## Stage 5 - Turkish-Focused Release Candidate

- Turkish-focused LLM release candidate
- Reproducible training
- Dataset license report
- Public demo
- Clear model limitations
- No fake claims about general intelligence
