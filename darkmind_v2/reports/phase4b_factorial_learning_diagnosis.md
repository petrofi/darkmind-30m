# DarkMind v2 Phase 4B Factorial Learning Diagnosis

Initialization hash shared by all arms: `f1da070885650b70dc999f22b6ef8a438bb47fe7479020dc46ebdf68ae3d9c6b`

| Arm | Order | Peak LR | Final validation | Final eval | Validation rebound | Eval rebound | Stability |
|---|---|---:|---:|---:|---:|---:|---|
| arm_a_legacy_lr3e4 | legacy_order_v1 | 0.00030 | 10.801898 | 10.647525 | 29.864% | 28.615% | unstable |
| arm_b_legacy_lr15e5 | legacy_order_v1 | 0.00015 | 8.985372 | 8.879882 | 11.027% | 10.453% | unstable |
| arm_c_stratified_lr3e4 | deterministic_stratified_v1 | 0.00030 | 8.333518 | 8.218347 | 10.404% | 9.710% | unstable |
| arm_d_stratified_lr15e5 | deterministic_stratified_v1 | 0.00015 | 7.618146 | 7.542179 | 3.566% | 3.366% | unstable |

## Factor effects

Lower-LR main effect, validation loss reduction: 1.265949
Lower-LR main effect, eval loss reduction: 1.221906
Stratified-order main effect, validation loss reduction: 1.917803
Stratified-order main effect, eval loss reduction: 1.883440
Interaction effect, validation: 1.101155
Interaction effect, eval: 1.091476

Stable policy found: **NO**
Selected exploratory arm: **none**
Optional 0.0002 trigger: **NO**

## Decision questions

1. Did the lower learning rate prevent late-stage deterioration? **NO.** Both 0.00015 arms still worsened after their best checkpoint.
2. Did deterministic stratification prevent deterioration caused by sequence-order shift? **NO.** It improved final loss, but both stratified arms remained unstable.
3. Were both changes sufficient together? **NO.** Their effects were beneficial but non-additive, and the combined arm still diverged late and collapsed into repeated generation loops.
4. Did the Phase 4A baseline failure reproduce outside OneDrive? **YES.** The legacy-order 0.0003 arm reproduced the late loss rebound on the external runtime.
5. Was OneDrive responsible for the learning failure? **NO.** Runtime integrity remained stable outside OneDrive while the learning failure persisted.
6. Is another optimizer, schedule, precision, initialization, or architecture factor unresolved? **YES.** No arm met the predeclared stable-policy gate.
