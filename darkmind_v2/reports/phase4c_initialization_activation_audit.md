# DarkMind v2 Phase 4C Initialization and Activation Audit

Device/dtype: `cuda` / `torch.bfloat16`

Base V1 initializes every Linear and Embedding matrix with normal std 0.02. It does not apply GPT-style depth-aware scaling to attention or MLP residual-output projections.

Real batch final/early residual RMS ratio: `56.8328`.
Synthetic batch final/early residual RMS ratio: `57.6704`.
Predeclared >2x residual alert: **TRIGGERED**.
Monotonic real-batch depth growth: **YES**.
Initial logit-scale alert: **NO**.
Non-finite activation count present: **NO**.

Optional versioned initialization arm justified: **YES**.

Per-layer residual, attention, MLP, normalization, entropy, logit, and parameter statistics are retained in the ignored runtime JSON.
