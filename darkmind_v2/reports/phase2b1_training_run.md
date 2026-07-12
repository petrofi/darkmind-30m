# Phase 2B.1 Stage-1 Training Run

## Completed Run

- Run: `tiny_stage1_seed20260712_r2`
- Optimizer steps: `256`
- Consumed tokens: `1,048,576`
- Effective tokens per optimizer step: `4,096`
- Initial raw loss: `10.116636`
- Midpoint raw loss: `7.119819`
- Final raw loss: `7.168717`
- Initial full validation loss: `10.123761`
- Midpoint full validation loss: `7.391306`
- Final full validation loss: `7.096028`
- Final full eval loss: `7.081271`
- Selected checkpoint: final step 256

The original zero-token aborted run remains unchanged. Stage-1 training was not rerun during post-training diagnosis or evaluation.

## Resume And Integrity

All five checkpoints contain matching model config, tokenizer and corpus provenance; expected optimizer, scheduler, RNG, and data-position state; and valid model hashes. Scheduler epochs are `0, 64, 128, 192, 256`, matching checkpoint steps. Data positions are `0, 262144, 524288, 786432, 1048576`, proving that the midpoint resume did not reset scheduler or data state.

## Throughput Audit

- Recorded end-to-end segment wall time: `31.940078 s`
- Active optimizer-step training time: `17.836148 s`
- Validation/generation/checkpoint/reload overhead: `14.103930 s`
- Canonical full-wall throughput: `32,829.48 effective tokens/s`
- Active training throughput: `58,789.38 effective tokens/s`
- Mean per-step reported throughput: `61,224.02 effective tokens/s`
- Full-wall optimizer rate: `8.015 steps/s`
- Cold calibration: `10,013.07 effective tokens/s` for one `0.409065 s` step

The old 61k figure is the arithmetic mean of warmed per-step rates and excludes periodic evaluation/checkpoint work. The 10k calibration is a single cold CUDA step that includes first-step warm-up. The canonical metric uses both complete recorded training segments, including periodic overhead, and is therefore 32.8k effective tokens/s.

## Limitations

The loss improvement establishes pipeline function, not language quality. Repetition, incoherence, early sampled invalid byte sequences, and limited token exposure remain unresolved. Further base pretraining is the recommended next phase; SFT or teacher-data generation should not begin from a mistaken quality claim.
