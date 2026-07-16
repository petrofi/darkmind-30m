# DarkMind v2 Phase 4B Corrected Training Policy

Stable factorial policy found: **NO**

No policy is frozen unless it passes integrity, final validation/eval improvement, rebound, sustained-divergence, and generation-health requirements.

| Arm | Order | Peak LR | Final validation | Final eval | Rebound val/eval | Stability | Final generation |
|---|---|---:|---:|---:|---:|---|---|
| arm_a_legacy_lr3e4 | legacy_order_v1 | 0.00030 | 10.801898 | 10.647525 | 29.864% / 28.615% | unstable | 135/200 repeated, 104/200 loops; 100/500 sampled repeated, 61/500 sampled loops |
| arm_b_legacy_lr15e5 | legacy_order_v1 | 0.00015 | 8.985372 | 8.879882 | 11.027% / 10.453% | unstable | 183/200 repeated, 170/200 loops; 135/500 sampled repeated, 120/500 sampled loops |
| arm_c_stratified_lr3e4 | deterministic_stratified_v1 | 0.00030 | 8.333518 | 8.218347 | 10.404% / 9.710% | unstable | 200/200 repeated, 200/200 loops; 258/500 sampled repeated, 249/500 sampled loops |
| arm_d_stratified_lr15e5 | deterministic_stratified_v1 | 0.00015 | 7.618146 | 7.542179 | 3.566% / 3.366% | unstable | 200/200 repeated, 200/200 loops; 340/500 sampled repeated, 328/500 sampled loops |

The lower learning rate and stratified order both improve final loss, but every arm remains unstable. The combined arm has only a 3-4% loss rebound, yet it shows three consecutive worsening evaluations and complete greedy loop collapse.

The predeclared optional 0.0002 trigger did not fire because no 0.00015 arm was stable and neither lower-LR arm had its best checkpoint at step 610 with monotonic final improvement.

`train_base_v1_production_100m_v2.json` was not created. No final confirmation run or future resume checkpoint is authorized.

Recommended next diagnosis: optimizer dynamics, schedule shape/warmup transition, precision behavior, initialization scale, and architecture/training-policy assumptions. Frozen Base V1, tokenizer, and Corpus V3 remain unmodified.
