# Changelog

All notable changes to the Observe.AI inference-optimization & benchmarking framework.
Format loosely follows Keep a Changelog; dates are ISO-8601.

---

## [Unreleased] — 2026-08-20 — Trainium (Neuron) single-instance testing (Flow B)

Added the Trainium path for gpt-oss-20b, following the validated vLLM Neuron recipe
(Neuron SDK 2.31). Single instance only; **disaggregated inference (DI) is out of
scope** (documented as a phase-2 peak-throughput topology). Same OpenAI API on
:8000, so the shared benchmark/quality harness runs against it unchanged.

### Added
- **`cluster/neuron-nodepool.yaml`** — Karpenter NodePool for `trn2`/`trn3`,
  `aws.amazon.com/neuron` taint (requires the AWS Neuron device plugin).
- **`vllm/models/neuron/`** — single-instance gpt-oss-20b: `deployment.yaml`
  (TP8, recipe `neuron_config` buckets, compile timeouts, 60-min `startupProbe`
  for cold JIT compile, persistent NEFF/weights cache), `pvc.yaml` (150Gi cache),
  `service.yaml`, `model.env`, `deploy.sh --hw`, `stop.sh`, `README.md`.
- **Matrix cells** — `configs/manifests/gpt-oss-20b-trn2-bf16.yaml` and
  `gpt-oss-20b-trn3-mxfp4.yaml` (`source: neuron`, TP8).
- **`config/config.env`** — `NEURON_VLLM_IMAGE` (blank, fail-fast guarded).
- **`run-benchmark.sh`** — `--hw trn*` routes to the `-neuron` service and picks
  quant by generation (trn3 → mxfp4, trn2 → bf16).

### Notes / decisions
- **Like-for-like = MXFP4 = Trn3.** gpt-oss ships MXFP4; g5/g6/g6e serve MXFP4
  (dequantized to BF16 for compute on Ada); Trn3 runs MXFP4 natively. Trn2 has no
  FP4 → it inflates the *same* MXFP4 values to BF16 (~40 GB): **same quality, 3×
  memory**, not higher quality. Trn2 is a valid "no-FP4 tax" data point, not a
  like-for-like MXFP4 comparison.
- **Concurrency cap:** the recipe compiles `num_seqs_buckets=[4]` → the realtime
  profile is representative; the batch profile queues rather than scales.
- **Cost caveat:** Trainium instances are whole multi-chip nodes (trn2.48xlarge =
  16 chips / 1.5 TB HBM; Trn3 mostly UltraServers). A 20B model uses a sliver, so
  single-instance cost/token is over-provisioned by construction — report Trn
  perf/quality confidently, annotate cost.
- **Placeholders to fill before running** are consolidated in the README
  ("Placeholders to fill") and `vllm/models/neuron/README.md`.

## [Unreleased] — 2026-08-20 — Quality evaluation stage (accuracy / F1)

Added a **separate** quality stage (model correctness), distinct from the
performance harness. Runs the customer AutoQA rubric (binary Yes/No) against a
deployed vLLM endpoint and reports accuracy / precision / recall / F1.

### Added
- **`inference/quality-eval.py`** — deterministic (temp 0, fixed seed) evaluator
  over the same OpenAI API. Confusion matrix + precision/recall/F1 (+ macro-F1,
  per-question F1). Reads `content`, falling back to `reasoning`/`reasoning_content`
  (reasoning models leave `content` empty). A guided-decoding capability probe
  degrades to `free_parse` if `guided_choice` isn't accepted.
- **`configs/quality/autoqa_v1.yaml`** — frozen eval config (label set, decoding
  mode, fields, reasoning effort).
- **`inference/run-quality.sh`** — stages the dataset to S3 (`--stage-dataset`),
  runs the eval Job in-cluster, uploads results. Accuracy is hardware-independent
  → run once per (model, quantization), not per GPU family.

### Fixed / hardened
- **Reasoning-model parsing** — gpt-oss returns the verdict in `reasoning`; a run
  scored 0 until we read that field and gave it a real token budget (`free_parse`,
  `reasoning_max_tokens`, `reasoning_effort: low`). guided_choice with tiny
  max_tokens truncated it mid-thought.
- **Gold-label guard** — fails loudly (with sample values) if the `answer` column
  isn't in the label set — catches CSV column/quoting shifts that otherwise yield
  a silent all-zero confusion matrix.
- **Unparseable-as-correct edge** — an `UNKNOWN` prediction on a negative row is no
  longer miscounted as a true negative.

### Data handling (per shellkode-security)
- The customer AutoQA dataset (synthetic, shared for assessment) lives in **S3
  only** — pulled to ephemeral pod storage at run time, never committed to git
  (`.gitignore`) or baked into a ConfigMap. Transcripts are never logged (only
  `data_id` + label + metrics).

## [Unreleased] — 2026-08-20 — Benchmark matrix restructure (3 models × 4 hardware)

Reframed the framework into a customer-deliverable **benchmark matrix** per the SoW:
the agreed models benchmarked across GPU families (and a Trainium flow), driven entirely
by config — no per-model or per-hardware hard-coding in framework code.

### Locked model set (corrected)
- **gpt-oss-20b** (`openai/gpt-oss-20b`) — MoE, MXFP4.
- **Qwen3.5-4B** (`Qwen/Qwen3.5-4B`) — dense multimodal, bf16.
- **Gemma-4-26B-A4B** (`google/gemma-4-26B-A4B-it`) — MoE (~3.8B active), 4-bit W4A16.

  **Correction of an earlier assumption:** a prior pass renamed the CSV's
  "Qwen3.5-4B" / "Qwen3.6-35B-A3B" to `Qwen3-4B` / `Qwen3-30B-A3B`, assuming they were
  typos. They are **not** — Qwen3.5-4B and Gemma-4 are real 2026 releases (verified on
  Hugging Face). The mistaken `qwen3-4b` and `qwen3-30b-a3b` model dirs + manifests were
  removed and replaced with the correct models above.

### Why Gemma-4-26B-A4B (MoE) instead of the 31B dense
- As an MoE with ~3.8B active params, its **4-bit footprint (~14.4 GB) fits a single 24 GB
  card** (g5/g6) with real KV headroom, and it decodes fast — so every matrix cell is
  **single-GPU** (no tensor parallelism).
- The 31B dense would need **TP=4** for bf16/fp8; on L40S/A10G/L4 (**no NVLink**, PCIe
  only) TP=4 scales to ~2–2.5×, not 4× → worse $/1M-tokens and broken cost parity. Kept as
  an optional quality-ceiling variant, not the baseline. See `docs/benchmark-matrix.md`.

### Added
- **Hardware dimension via templating** (no new NodePools needed — the `gpu-inf` NodePool
  already spans g5/g6/g6e). Each model's `deployment.yaml` now has a `nodeSelector` on
  `node.kubernetes.io/instance-type` and templated `--tensor-parallel-size`, GPU count,
  model folder and served name. `deploy.sh` takes `--hw <instance-type>` and `--tp <n>`,
  so one model runs on any family without editing YAML.
- **New model dirs**: `vllm/models/qwen3.5-4b/` and `vllm/models/gemma-4-26b-a4b-it/`
  (deployment/pvc/service/model.env/deploy.sh/stop.sh) + matching `model-download/` jobs.
- **Per-cell manifests** (`configs/manifests/<model>-<hw>-<quant>.yaml`): gpt-oss-20b ×
  {g5,g6,g6e} (MXFP4), Qwen3.5-4B × {g5,g6,g6e} (bf16), Gemma-4-26B-A4B × {g5,g6,g6e}
  (W4A16) + an optional `gemma-4-26b-a4b-g6e-fp8` higher-precision cell.
- **`docs/benchmark-matrix.md`** — feasibility grid, memory math, precision policy,
  the 12xlarge/TP tradeoff, and per-cell run commands.
- **Neuron / Trainium Flow B scaffold** (`vllm/models/neuron/`: README, compile-job,
  serve deployment) + `cluster/neuron-nodepool.yaml`. Compile stage → NEFF cache in S3;
  benchmark/observability/teardown shared with the GPU flow. Marked **pending per-model
  validation** (gpt-oss MoE-MXFP4, Qwen3.5 hybrid-attention VLM, and Gemma-4 on Neuron are
  all unconfirmed — the scaffold fails loudly rather than silently producing nothing).

### Changed
- gpt-oss-20b and qwen-2.5-0.5b deployments/deploy.sh **retrofitted** with the same
  hardware templating so all four models share one mechanism.
- Kept `qwen-2.5-0.5b` as the **smoke test** and `gpt-oss-20b-opt1.yaml` as a historical
  tuning experiment.
- **Fixed `inference/run-benchmark.sh`** (the one-command runner) to match the reworked
  harness — this was blocking any real run:
  - Runs the Job from **`${VLLM_IMAGE}`** (vLLM DLC) instead of `python:3.11-slim`, since
    the harness now shells out to `vllm bench serve` (absent from the slim image). Only
    `boto3` is pip-installed (to `/tmp`) for the S3 upload.
  - Points at the **per-cell manifest** `<model>-<hw>-<quant>.yaml` (the old hardcoded
    `gpt-oss-20b-baseline.yaml` was deleted); added `--hw` and `--manifest` flags and
    fail-fast checks when a manifest/profile file is missing.
  - Dropped the now-dead `qa_eval_v1` dataset upload/download path (profiles use
    `vllm bench serve`'s built-in `random` dataset) and the deleted `qwen3-35b-nvfp4` entry.
- Verified the gpt-oss-20b path end-to-end: `envsubst` renders `deployment.yaml` with zero
  unresolved placeholders; `bash -n` clean on run/deploy/stop scripts; manifest cell resolves.

### Notes / risks flagged
- Qwen3.5-4B and Gemma-4 are **multimodal**; we benchmark the **text path only**.
- Gemma-4-26B-A4B needs a **vLLM-loadable 4-bit checkpoint** (compressed-tensors W4A16
  preferred; AWQ with `--quantization awq`). The download job **requires** `QUANT_HF_ID`
  rather than guessing. If none exists, quantize the bf16 base with llm-compressor offline.
- Manifest `cost.instance_hourly_usd` values are approximate us-east-2 on-demand — **verify
  before each run**.

---

## [Unreleased] — 2026-09-10 — Benchmark harness rewrite (bench-serve orchestrator)

The round that fixed the core measurement bug behind the bad gpt-oss-20b numbers.

### Fixed
- **Throughput/cost collapse (~0.6 tok/s, ~$127k/1M).** The old custom async harness
  counted output tokens by word-splitting `delta.content`, which **dropped
  `reasoning_content`** for reasoning models like gpt-oss — so token counts (and thus
  throughput) were near-zero and the cost math divided by ~0.
- **Rewrote `inference/load-test.py`** as an orchestrator around **`vllm bench serve`**
  (the official, tokenizer-accurate load generator): correct TTFT / TPOT / ITL / E2E
  percentiles and tokenizer-counted throughput (reasoning-aware). It scrapes vLLM
  `/metrics` (KV-cache %, queue depth, prefix-cache hit rate) and an optional
  `DCGM_METRICS_URL` for GPU util/mem/power, then derives normalized cost with a
  **zero-token guard** (emits `null`, never garbage). Adds `tokens_per_usd` and
  `energy_per_1m_tokens_wh`.
- **Accuracy** is set to `None` in the perf row — model quality is a **separate**
  lm-eval-harness stage (MMLU-Pro/GPQA/DROP/IFEval on public data), per SoW ownership.

### Changed
- **`inference/benchmark-job.yaml`** now runs from the **vLLM DLC image** (so the
  `vllm bench serve` CLI + tokenizers are present), CPU-only, with `boto3` pip-installed to
  a writable path (read-only rootfs). Runs off the GPU node (podAntiAffinity) to avoid
  contaminating latency.
- **Workload profiles** (`realtime_v1`, `batch_v1`) aligned to the bench-serve contract
  (`input.dataset_name`/`random_input_len`/`ignore_eos`, `num_prompts_factor`/`min_prompts`,
  `bench_extra_args`) while preserving the frozen semantics (token lengths, concurrency, SLO).

---

## [Unreleased] — 2026-09-10 — gpt-oss-20b vLLM config fixes

Applied the official gpt-oss recipe to `vllm/models/gpt-oss-20b/deployment.yaml`.

### Fixed / Changed
- **Removed `--trust-remote-code`** — not needed for gpt-oss; avoids RCE risk from a
  poisoned HF repo.
- **Removed `--enable-prefix-caching`** → **`--no-enable-prefix-caching`** for consistent,
  comparable benchmark numbers.
- **Added `--async-scheduling`** (recipe optimization).
- **`--gpu-memory-utilization` 0.80 → 0.90** (reconciled with the comment that claimed 0.95).
- **`--max-model-len` 8192 → 16384** — gpt-oss is a reasoning model; 8192 truncated answers
  after long chain-of-thought. MXFP4 (~13 GB, not the previously-claimed 41.6 GB) leaves
  ample HBM.
- **`--max-num-batched-tokens` 2048 → 8192**, **`--max-num-seqs` → 256**.
- **Deliberately did NOT set `--stream-interval`** — the recipe's `stream-interval=20` (for
  max-throughput serving) emits one chunk per 20 tokens and would distort the ITL that
  `vllm bench serve` measures. Kept per-token ITL accurate.
- Fixed misleading comments (weights size, memory utilization).

### Notes
- L40S/Ada is **not** gpt-oss's optimal-kernel tier (FlashInfer MXFP4 is Hopper/Blackwell);
  vLLM falls back to the Triton/Marlin path here → lower throughput than H100 recipe
  figures. Expected, documented — not a bug.
