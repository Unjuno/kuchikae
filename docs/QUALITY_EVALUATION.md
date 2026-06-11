# Quality Evaluation

Kuchikae evaluates quality along three axes.

## Text preservation

- numeric token preservation
- negation preservation
- named entity preservation

## Audio readability

- generated file exists
- audio can be read by `soundfile`
- output length is non-zero
- peak amplitude stays within clipping bounds

## Latency

- `p50` and `p90` latency for batch and streaming paths
- time to first partial transcript
- time to first committed text
- time to first audio segment

## Real-time factor

- compare audio duration to end-to-end processing time
- record separately for STT, text transform, and voice output
