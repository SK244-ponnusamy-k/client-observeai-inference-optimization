# Trainium (Neuron) — single-instance gpt-oss-20b

Flow B (Trainium) for the benchmark matrix. This directory deploys **gpt-oss-20b
single-instance (non-DI)** on AWS Trainium via the vLLM Neuron plugin, following
the validated recipe (Neuron SDK 2.31). It exposes the **same OpenAI API on
:8000**, so the shared performance and quality harnesses run against it
unchanged — only the deploy differs.

```
HF weights ─▶ [vllm serve on Trn (JIT-compiles NEFFs → cached on PVC)] ─▶ [benchmark] ─▶ [teardown]
              └───────────── Flow-B-only ─────────────┘                  └── shared with GPU flow ──┘
```

## Scope

- **In scope:** single instance, one `vllm serve --tensor-parallel-size 8`, whole
  model on one Trn node. The apples-to-apples comparable to the GPU single-node
  cells.
- **Out of scope (phase 2):** **disaggregated inference (DI)** — prefill/decode
  split across instances + proxy router + EFA/NIXL. That's the peak-throughput
  production topology; it's a different (multi-instance) cost class and a large
  lift. Not built here.
- **Model:** gpt-oss-20b only (it has a tested Neuron recipe). Qwen3.5-4B and
  Gemma-4 on Neuron are unvalidated — do not assume they compile/serve yet.

## Hardware

- **Trn3** → serves gpt-oss in **MXFP4** (native, auto-selected). Best match.
- **Trn2** → serves gpt-oss in **BF16**.
- Same `vllm serve` command works on both; the Neuron backend picks weights by
  hardware. Only the instance type changes.

## Files

| File | Purpose |
|---|---|
| `deployment.yaml` | single-instance TP8 serve (templated: image, instance type, device count, model ref) |
| `pvc.yaml` | persistent cache for HF weights (`--download-dir`) + compiled NEFFs |
| `service.yaml` | ClusterIP `oai-infopt-vllm-gpt-oss-20b-neuron:8000` |
| `model.env` | names + hardware defaults |
| `deploy.sh` / `stop.sh` | deploy (with `--hw`) / teardown |

## Run

```bash
# 0. once: Neuron NodePool + Neuron device plugin installed on the cluster
kubectl apply -f cluster/neuron-nodepool.yaml
# (install the AWS Neuron k8s device plugin separately — exposes aws.amazon.com/neuron)

# 1. set NEURON_VLLM_IMAGE in config/config.env  (vLLM Neuron plugin, SDK 2.31)

# 2. deploy (first launch JIT-compiles NEFFs — can take 20-60 min; cached after)
bash vllm/models/neuron/deploy.sh --hw trn2.48xlarge      # BF16 on Trn2
#   or                              --hw trn3.<size>       # MXFP4 on Trn3

# 3. benchmark / quality — SHARED harness, just pick the trn hardware:
bash inference/run-benchmark.sh --model gpt-oss-20b --hw trn2 --profile realtime
bash inference/run-quality.sh   --model gpt-oss-20b --hw trn2 --quant bf16   # after the quality merge

# 4. teardown (releases the Trn node; keeps the cache PVC)
bash vllm/models/neuron/stop.sh
```

## Important caveats

- **Concurrency cap.** The tested recipe compiles `num_seqs_buckets=[4]`, so the
  server handles ~4 concurrent sequences. The **realtime** profile (conc 1/2/4)
  is representative; the **batch** profile (conc 16+) will *queue*, not
  parallelize, unless you recompile with larger `num_seqs_buckets`. Maximizing
  Neuron throughput is precisely what DI is for (out of scope). Read Trn batch
  numbers with this in mind.
- **Cold-start compilation.** First launch compiles graphs (minutes → tens of
  minutes). The `startupProbe` allows a 60-min budget; NEFFs persist on the cache
  PVC so later pods start fast. Don't delete the PVC between runs unless you want
  a clean recompile.
- **Quality determinism.** For reproducible Yes/No in the quality eval, add
  `on_device_sampling_config: {all_greedy: true}` to `neuron_config` (Neuron
  defaults to top-k sampling).
- **Observability.** DCGM is NVIDIA-only. Device utilisation on Trn comes from
  **neuron-monitor** (different endpoint/metric names) — vLLM `/metrics`
  (throughput, KV) still works, so latency/throughput/cost are unaffected; only
  device-util telemetry needs the neuron-monitor path (or mark it N/A for Trn).

## Open decisions to confirm (before promoting results)

1. **Trn2 vs Trn3** — which generation, and is Trn3 available in-region + on the
   EKS Neuron AMI (SDK 2.31)? (Trn analog of the G7-driver caveat.)
2. **`NEURON_VLLM_IMAGE`** — pin the exact AWS Neuron vLLM-plugin image + digest.
3. **`NEURON_DEVICE_COUNT`** — TP8 = 8 NeuronCores; the device plugin counts chips
   (2 cores each) → likely `4`. Confirm against your plugin build (some expose
   `aws.amazon.com/neuroncore`).
4. **Weight source** — HF `--download-dir` (tested path, current default) vs the
   S3 copy (only if runai_streamer is confirmed on Neuron — it isn't the tested path).
5. **Instance $/hr** — replace the placeholder Trn rates in the manifests with real
   on-demand pricing for a meaningful cost/1M.
6. **Prefix caching** — the recipe leaves it on; the GPU cells turned it off for
   consistency. Decide which for the cross-hardware comparison.
