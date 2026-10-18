# gpt-oss on vLLM-Neuron — non-DI optimization report

Status: review / recommendations. Owner: genai-platform@shellkode. Updated: 2026-10-18.

Scope: tuning the **single-instance (non-disaggregated)** gpt-oss-20b deployment
on Trainium2 that this repo currently ships
(`vllm/models/gpt-oss-20b-neuron/`). Compares the live config against AWS's
tested vLLM-Neuron recipe (`trn.md`) and lists concrete, low-risk changes.

> **Headline caveat — read first.** AWS's own recipe is explicit that
> single-instance serving is "adequate for simple tasks" and that **only
> disaggregated inference (DI) delivers the full throughput and latency of
> gpt-oss on Neuron.** Everything below optimizes the *non-DI* path we run today.
> We have **not** benchmarked DI. Expect DI (separate prefill/decode instances
> over EFA/NIXL, expert-parallel decode) to be **materially faster** — it is the
> recommended next workstream once non-DI is tuned and baselined.

## 1. Current deployment (what we run today)

From `vllm/models/gpt-oss-20b-neuron/deployment.yaml` + `model.env`:

| Setting | Current value |
| --- | --- |
| Instance | `trn2.3xlarge` (1 chip) |
| LNC | 2 (fuse 2 physical cores → 1 logical) |
| Tensor parallel | `--tensor-parallel-size 4` (= `NEURON_CORES`) |
| Precision | BF16 (`--hf-overrides '{"quantization_config": {}}'` strips MXFP4) |
| `max-model-len` | 8192 |
| `max-num-batched-tokens` | 4096 |
| `max-num-seqs` | 32 |
| Prefix caching | on (`--enable-prefix-caching`) |
| Hybrid KV cache mgr | on (`--no-disable-hybrid-kv-cache-manager`) |
| `block-size` | 8 |
| Compiled buckets | `num_batched_tokens_buckets:[4096]`, `num_seqs_buckets:[1,2,4,8,16,32]` |
| Timeouts | compile/exec 1200s |

This is a **functional-validation** configuration, not a performance-tuned one.

## 2. AWS tested non-DI recipe (`trn.md`) for comparison

| Setting | `trn.md` single-instance |
| --- | --- |
| Tensor parallel | `--tensor-parallel-size 8` |
| `max-model-len` | 16384 |
| `max-num-batched-tokens` | 8192 |
| `max-num-seqs` | 4 |
| Precision | MXFP4 on Trn3 (auto-selected); BF16 on Trn2 |
| Prefix caching | on |
| Hybrid KV cache mgr | on |
| Compiled buckets | single bucket per dim (`[8192]`, `[4]`) |

Note the recipe targets a full node (TP8). On `trn2.3xlarge` (1 chip, 4 logical
cores at LNC=2) TP8 is not available — our TP4 is the correct ceiling for this
instance. That is the main reason our numbers will trail the recipe until we
move to a larger Trn2/Trn3 node or to DI.

## 3. Findings and recommended changes (non-DI)

Ordered by expected impact / lowest risk first. Each requires a **recompile**
unless noted (compile-time buckets and TP are baked into the NEFF).

### 3.1 Precision: evaluate MXFP4 instead of forcing BF16  — HIGH impact

We pass `--hf-overrides '{"quantization_config": {}}'`, which **strips the
published MXFP4 quantization** and serves BF16. gpt-oss ships MXFP4 weights; the
AWS recipe serves MXFP4 (auto on Trn3, and selectable via
`neuron_config.quantization: "mxfp4"`). BF16 roughly doubles weight HBM and
memory-bandwidth pressure per token versus MXFP4.

- **Action:** on Trn3, drop the `quantization_config: {}` override and let MXFP4
  load; on Trn2, test `neuron_config.quantization: "mxfp4"` per the recipe.
  Measure quality on the AutoQA eval set before/after — MXFP4 vs BF16 can shift
  output quality, so this is a quality + perf tradeoff, not a free win.
- **Risk:** medium (accuracy). Gate on the quality harness.

### 3.2 Right-size `max-num-seqs` / batched-tokens to the workload  — HIGH impact

`max-num-seqs 32` with a single `num_batched_tokens_buckets:[4096]` is a broad
batch ceiling compiled against a small prefill segment. For the **realtime**
(latency-bound) profile a large batch ceiling adds decode latency; for the
**batch** (throughput-bound) profile 4096 batched tokens throttles prefill.

- **Action:** compile **two artifacts** matched to the two frozen profiles
  instead of one compromise:
  - realtime: `max-num-seqs 4`, `num_seqs_buckets:[1,2,4]`,
    `num_batched_tokens_buckets:[8192]`
  - batch: `max-num-seqs` 16–32, `num_batched_tokens_buckets:[8192]` (raise to
    the largest batch that fits HBM at `max-model-len`)
- **Risk:** low. Standard vLLM tuning; gate on HBM headroom at compile.

### 3.3 Collapse decode bucket spread  — MEDIUM impact

`num_seqs_buckets:[1,2,4,8,16,32]` compiles six decode graphs. The AWS recipe
uses a **single** bucket per dimension for a predictable, well-optimized graph.
Multiple buckets help variable traffic but cost compile time and can fragment
the scheduler.

- **Action:** for a fixed benchmark profile, compile a single `num_seqs` bucket
  equal to the profile's concurrency. Keep the multi-bucket build only for a
  mixed-traffic deployment.
- **Risk:** low.

### 3.4 Raise `max-num-batched-tokens` to 8192  — MEDIUM impact

We use 4096; the recipe uses 8192, which lets prefill process a bigger segment
per step (fewer prefill iterations for long AutoQA transcripts). Must match the
compiled `num_batched_tokens_buckets`.

- **Action:** set `--max-num-batched-tokens 8192` and compile
  `num_batched_tokens_buckets:[8192]`. Confirm it fits HBM at the chosen
  `max-model-len`.
- **Risk:** low–medium (HBM).

### 3.5 `max-model-len` to the real workload  — depends on profile

The recipe uses 16384; we use 8192. Set this to the frozen workload profile's
p95 input + max output, not higher — a larger context grows the KV cache and
reduces the batch that fits. Right-size rather than maximize.

- **Action:** derive from the AutoQA workload profile (transcript p95 + QA-form
  output length) and set explicitly. Recompile.
- **Risk:** low.

### 3.6 Decode-only: disable logprobs  — LOW/MEDIUM impact

`trn.md` sets `--max-logprobs 0` on the decode role so the graph skips gathering
the full logits tensor before sampling (saves decode time + memory). In our
single-instance server, if the benchmark does not request logprobs, the same
applies.

- **Action:** add `--max-logprobs 0` when the workload does not need logprobs.
- **Risk:** low (only affects logprob-consuming clients).

### 3.7 On-device sampling for deterministic benchmarking  — correctness

For reproducible benchmark runs, set
`on_device_sampling_config: {all_greedy: true}` (top-k=1). The framework's
determinism rule (fixed seed/profile) wants this for comparable numbers.

- **Action:** enable greedy on-device sampling for benchmark manifests; leave
  sampling as-is for quality runs that intentionally sample.
- **Risk:** none for benchmarking.

## 4. Recommended non-DI serve command (Trn2, after recompile)

```bash
# Realtime profile artifact (latency-bound)
NEURON_SKIP_EFA_AFFINITY=1 vllm serve /model \
  --served-model-name gpt-oss-20b \
  --tensor-parallel-size 4 \
  --max-model-len <profile_p95> \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 4 \
  --max-logprobs 0 \
  --enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --additional-config '{"neuron_config": {
      "quantization": "mxfp4",
      "num_batched_tokens_buckets": [8192],
      "num_seqs_buckets": [1,2,4],
      "on_device_sampling_config": {"all_greedy": true}
  }}' \
  --port 8000
```

(Drop `"quantization": "mxfp4"` and keep BF16 only if the quality eval regresses
unacceptably. Build a second artifact with a larger `num_seqs` bucket for the
batch profile.)

## 5. What this does NOT fix — and why DI is the real next step

Non-DI co-locates prefill and decode on the same cores, so a long prefill stalls
decode and vice-versa; a single TP4 server also leaves most of a larger node
idle. `trn.md` resolves both with **disaggregated inference**: dedicated prefill
instances (TP4, `ep_degree 2`) feeding an expert-parallel decode instance
(TP8 DP×, EP32/EP64) over NIXL/EFA, scaling prefill independently (up to 16P1D).

**We have not tested DI.** Based on the AWS guidance it should deliver
substantially higher throughput and lower tail latency than any non-DI tuning
above. Recommended sequence:

1. Apply §3 changes, recompile, re-baseline non-DI on the frozen profiles.
2. Stand up the DI topology from `trn.md` (needs 2 instances + EFA) and
   benchmark against the non-DI baseline.
3. Record both in the results store with the standard schema for an apples-to-
   apples non-DI vs DI comparison.

## References

- `trn.md` — AWS tested vLLM-Neuron recipe (single-instance + DI) for gpt-oss.
- `vllm/models/gpt-oss-20b-neuron/` — the current deployment this report reviews.
- `ml-pipeline-standards.md` — frozen workload profile + cost/benchmark rules.
