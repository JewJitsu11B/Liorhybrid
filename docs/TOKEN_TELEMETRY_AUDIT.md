# Training & Inference Token Flow + Telemetry Audit

## Scope
- Verify token ordering through training (trainer2) and inference (chat generation).
- Validate training telemetry equations and relevance of logged metrics.

## Training Pipeline: Token Flow
- **Batch ingress**: `trainer2.entrypoint` moves dataloader batches to device (`to_device`) before any mutation, preserving ordering (`training/trainer2.py`, lines ~445-459).
- **Window processing**: Each window calls `run_two_phase_and_update` (free+nudge) or `run_window` (free only) with the batch untouched except for device move, ensuring tokens stay in dataloader order (lines ~462-497).
- **Target alignment**: Nudge targets use `input_ids` shifted right when `nudge_use_shifted_target=True`, padding with the last token to maintain sequence length; otherwise uses the unshifted tokens. This guarantees next-token targets follow the original order (`build_nudge_signal`, lines ~2084-2103).
- **Coordinate mapping**: Target embeddings are trimmed/padded to coordinate dim but not re-ordered; attention masks optionally weight averages, retaining positional semantics (lines ~2107-2130).

## Inference Pipeline: Token Flow
- **Encoding**: `InferenceEngine._encode_text_to_ids` uses the training tokenizer with special tokens, enforcing consistent prefix and ordering (`inference/inference.py`, lines ~198-205).
- **Generation loop**: `generate` appends each sampled `next_id` to `generated_ids` in step order, stopping on EOS; prompt + continuation are decoded in the same order they were produced (lines ~230-293).
- **Field updates**: After each token, `field.evolve_step()` is called, so state progression matches token progression without reordering (lines ~287-291).

## Telemetry Equations & Relevance (Training)
- **Per-step metrics**: `run_window` accumulates `lior_acc`, `R_acc`, `spd_acc`; `lior_step` computes `lior = R_sc * spd` where `spd` is a quadratic form of velocity (`quad_form_batch`) with metric `g0` (`training/trainer2.py`, lines ~723-745, ~1990-2020).
- **Window means**: `WindowMetrics` stores means by dividing accumulators by step count (`inv_steps`) before returning from `run_window` (lines ~2015-2020).
- **Buffered logging**: `_log_metrics_buffered` writes a rolling buffer on GPU and, at `log_every_windows`, averages lior/R/spd, synchronizes via side stream, and prints the window summary (lines ~2874-2924). This avoids per-step host syncs while preserving numerical correctness.
- **Telemetry JSONL**: `maybe_log_metrics` writes a record containing `lior_mean`, `R_mean`, `spd_mean`, `window_ms`, `mem_norm`, and `tbptt_window_steps`, tagged with `epoch`, `window`, `batch`, and a `config_hash` for reproducibility (`training/trainer2.py`, lines ~2927-2964). Files are scoped under `run_dir/run_name` and prefixed with a `run_meta` record (`_ensure_jsonl`, lines ~2820-2839).

## Findings
- **Token order** is preserved in both training and inference: no reshuffling beyond device transfer; nudge targets explicitly align to next-token order.
- **Telemetry equations** correctly compute window-level means and log physically meaningful quantities (lior, R, spd) along with timing/memory context, making the telemetry relevant for monitoring training dynamics.
