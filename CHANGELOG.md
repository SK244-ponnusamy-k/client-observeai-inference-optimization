# Changelog

All notable changes to the Observe.AI inference-optimization & benchmarking framework.
Format loosely follows Keep a Changelog; dates are ISO-8601.

---

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

---

## [Unreleased] — 2026-09-25 — Optimization round 2: trustworthy telemetry + Gemma-4-31B KV/batch tuning

Follow-up to the 24-Sep weekly sweep (gpt-oss-20b / Qwen3.5-4B / Gemma-4-31B ×
g5 / g6 / g6e / g7e). Before tuning further we had to fix the harness: two of the
columns on the result sheet were not measuring what they claimed.

### Fixed — benchmark telemetry (`inference/load-test.py`)
- **KV-cache usage read 0 % on every row.** `/metrics` was scraped once *after*
  `vllm bench serve` returned — by then the queue has drained and the KV cache is
  empty. A new `_TelemetrySampler` polls vLLM `/metrics` and DCGM every 2 s **while
  the level runs** and records the peak (`kv_cache_utilization_pct`) and mean
  (`kv_cache_utilization_mean_pct`) KV usage, and the peak waiting-queue depth.
- **Impossible GPU util / memory (e.g. 400 %, 164 GB on a single 24 GB L4).** The
  in-cluster `dcgm-exporter` Service load-balances across one exporter pod *per GPU
  node*, and the harness summed every GPU on whichever node answered. GPU samples are
  now **scoped to the vLLM pod under test** via DCGM's `pod`/`namespace`/`container`
  labels (exact `<deployment>-<rs>-<pod>` match, so a tagged parallel deploy is not
  mixed in); scrapes that land on another node are discarded. Across a pod's GPUs:
  util = mean, memory = sum, power = sum. Over the run: util/power = mean, memory = peak.
  If no sample matched, the GPU fields are `null` (with a warning), never a wrong number.
- Removed `vllm:gpu_memory_utilization` from the KV-usage aliases — it is the static
  `--gpu-memory-utilization` setting, not KV usage. `vllm:kv_cache_usage_perc` (V1) is
  now tried first.
- `energy_per_1m_tokens_wh` now uses **mean in-run power** instead of one post-run
  (idle) reading.

### Added
- Result rows gain `kv_cache_utilization_mean_pct`, `telemetry_samples` and
  `gpu_telemetry_samples` (0 ⇒ GPU columns are null, not measured). Additive — existing
  consumers are unaffected.
- `OAI_VLLM_DEPLOYMENT` / `OAI_VLLM_NAMESPACE` env on the benchmark Job
  (`inference/run-benchmark.sh` sets them from the resolved Service name;
  `inference/benchmark-job.yaml` derives them from `MODEL_NAME`).
- **`tests/test_telemetry.py`** — stdlib `unittest` regression tests with fake vLLM /
  DCGM endpoints covering both bugs (`python -m unittest discover -s tests`).
- **`catalog/models/gemma-4-31b-opt2.yaml`** + generated files
  (`vllm/models/gemma-4-31b-opt2/`, `configs/manifests/gemma-4-31b-opt2-gpu.yaml`,
  `model-download/gemma-4-31b-opt2/`) — an A/B **candidate**, deployed side-by-side with
  the baseline under its own names and sharing the same S3 weights (no re-download).

  | Knob | Baseline | opt2 | Why (from the 24-Sep data) |
  |---|---|---|---|
  | `--gpu-memory-utilization` | 0.85 | **0.92** | g5.12xlarge: throughput flat at ~130 tok/s from C=16→64 with constant ITL while TTFT climbs 24 s→208 s — the running batch can't grow. After ~15.8 GB/GPU of weights, 0.85 leaves only ~2–3 GB KV per 24 GB GPU; 0.92 gives ~1.7× the KV blocks. |
  | `--max-num-seqs` | 128 | **256** | g6e.12xlarge: 484 tok/s at both C=128 and C=256 (SLO ≥ 500) with TTFT p95 173 s at C=256 — half the requests queue behind a 128-seq cap while ~100 GB of KV is free. |
  | prefix caching | on | **off** | Same policy as gpt-oss/Qwen so rows are comparable (random dataset ⇒ ~0 hits). |

### Findings recorded (no config change)
- **gpt-oss-20b on g5.2xlarge / g6.2xlarge cannot meet the batch SLO (≥ 500 tok/s)
  by tuning.** Throughput plateaus at ~297 / ~229 tok/s from C=128 up — a
  memory-bandwidth ceiling of A10G / L4 on the Marlin/Triton MXFP4 fallback path, not
  a scheduler limit. g6e.2xlarge (≥ C=64) and g7e.2xlarge (≥ C=32) pass; recommend
  g6e/g7e for gpt-oss batch and keep g5/g6 for realtime only (they pass realtime).
- Historic GPU Util / GPU Mem / KV Cache columns in the 24-Sep sheet are **not valid**
  and should be re-collected with this harness; latency, throughput and cost columns
  come from `vllm bench serve` and are unaffected.

### Not yet done / next
- **opt2 is untested** — run it on g6e.12xlarge and g5.12xlarge
  (`oai deploy gemma-4-31b-opt2 --hw g6e.12xlarge --benchmark`), compare with
  `gemma-4-31b`, and fold the winning knobs back into `gemma-4-31b.yaml`.
- Separate A/Bs, deliberately left out of opt2 so each effect is attributable:
  `--kv-cache-dtype fp8` on g6e/g7e only (Ampere/g5 has no FP8), and
  `--async-scheduling` for Gemma4 at TP=4.
- `prefix_cache_hit_rate` is still a cumulative counter ratio since server start,
  not a per-level delta.
