# Leaky Integrator / Accumulator — Summary

## Overview

Implemented a Leaky Integrator (Accumulator) to stabilize BCI command triggering by accumulating prediction confidence over time and applying a configurable decay (leak). Commands are only emitted when a bucket (per class) exceeds a trigger threshold.

## Files changed

- CODE/npg_inference.py
  - Added `LeakyAccumulator` class
  - Added `predict_with_accumulator()` and `get_accumulator_status()`
  - Increased defaults: `confidence_threshold` → 0.65, `smoothing_window` → 8
  - Added accumulator statistics and reset support
- CODE/npg_realtime_bci.py
  - Use accumulator for epoch processing by default
  - New CLI options: `--no-accumulator`, `--acc-threshold`, `--acc-decay`, `--neutral-low`, `--neutral-high`
  - Periodic status/logging now shows bucket fill %

## Key behavior

- Neutral zone: predictions with confidence between 0.45 and 0.55 are treated as "uncertain" and do not add to buckets.
- Accumulation: predictions outside the neutral zone add a scaled contribution to the corresponding class bucket.
- Decay (leak): all buckets are multiplied by (1 - decay_rate) on each update (default decay_rate=0.15).
- Triggering: when a bucket >= `trigger_threshold` (default 2.0) and cooldown passed, a command is emitted and that bucket is partially reduced.
- Cooldown prevents immediate re-triggering for a few updates.

## Defaults (configurable)

- `confidence_threshold`: 0.65 (used when accumulator disabled)
- `smoothing_window`: 8
- `accumulator_threshold`: 2.0
- `accumulator_decay`: 0.15
- `neutral_zone`: (0.45, 0.55)

## Usage examples

Run normally (accumulator enabled):

```bash
python CODE/npg_realtime_bci.py --simulate
```

Disable accumulator (fallback to smoothed thresholding):

```bash
python CODE/npg_realtime_bci.py --simulate --no-accumulator --confidence 0.7
```

Adjust accumulator sensitivity:

```bash
python CODE/npg_realtime_bci.py --simulate --acc-threshold 1.5 --acc-decay 0.2
```

Tune neutral zone bounds (widen to ignore more uncertain predictions):

```bash
python CODE/npg_realtime_bci.py --simulate --neutral-low 0.40 --neutral-high 0.60
```

## Notes

- The accumulator improves robustness by requiring sustained evidence before issuing commands and reduces spurious single-epoch triggers.
- If you see no triggers, try lowering the `--acc-threshold`, reducing `--acc-decay`, or increasing model confidence via preprocessing/model improvements.
- The summary and examples are intentionally concise; see `CODE/npg_inference.py` and `CODE/npg_realtime_bci.py` for implementation details and exact parameter names.
