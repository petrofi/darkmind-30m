# Checkpoint Weight Delta Audit

Base: `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-30m\models\darkmind-30m-10k-step15000.pt`
Student: `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-30m\models\darkmind-30m-qwen-distill-pilot500-tr-en-v2.pt`
Missing keys in student: `[]`
Shape mismatches: `[]`

## Focus Modules

| Module | Abs Delta Norm | Relative Delta Norm | Max Abs Change | Base Norm |
|---|---:|---:|---:|---:|
| token_embedding | 2.999642 | 0.024026 | 0.004553 | 124.847733 |
| position_embedding | 0.135992 | 0.010643 | 0.002111 | 12.777032 |
| first_transformer_block | 1.000950 | 0.005068 | 0.003297 | 197.486141 |
| final_transformer_block | 1.267133 | 0.006221 | 0.003471 | 203.677568 |
| final_layer_norm | 0.036755 | 0.001437 | 0.002479 | 25.579914 |
| lm_head | 2.999642 | 0.024026 | 0.004553 | 124.847733 |

## All Module Groups

| Module | Abs Delta Norm | Relative Delta Norm | Max Abs Change |
|---|---:|---:|---:|
| blocks.1 | 0.982526 | 0.004971 | 0.003451 |
| blocks.2 | 0.993410 | 0.004951 | 0.003745 |
| blocks.3 | 0.988622 | 0.004914 | 0.003072 |
| blocks.4 | 1.018311 | 0.005030 | 0.003343 |
| blocks.5 | 1.088008 | 0.005344 | 0.003701 |
| blocks.6 | 1.161140 | 0.005693 | 0.003750 |
| final_layer_norm | 0.036755 | 0.001437 | 0.002479 |
| final_transformer_block | 1.267133 | 0.006221 | 0.003471 |
| first_transformer_block | 1.000950 | 0.005068 | 0.003297 |
| lm_head | 2.999642 | 0.024026 | 0.004553 |
| position_embedding | 0.135992 | 0.010643 | 0.002111 |
| token_embedding | 2.999642 | 0.024026 | 0.004553 |
