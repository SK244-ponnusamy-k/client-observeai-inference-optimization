# Catalog entry schema

A catalog entry is a YAML file at `catalog/models/<id>.yaml`. Fields marked
**required** must be present. Everything else has a sensible default, so a
minimal entry is short.

The `oai model validate <id>` command checks all of this for you before
anything runs.

---

## `id` (required)

Short, lowercase, dash-separated name. This becomes the folder name and every
Kubernetes resource name. Example: `qwen3-4b`, `gpt-oss-20b`.

Rules: lowercase letters, digits, and dashes only. Must start with a letter.

## `display_name`

Human-friendly name shown in `oai model list`. Example: `Qwen3 4B Instruct`.

## `hardware` (required)

Which kind of accelerator this model runs on. One of:

| value    | meaning                                            |
|----------|----------------------------------------------------|
| `gpu`    | NVIDIA GPU (g5 / g6 / g6e / g7 families)            |
| `neuron` | AWS Trainium / Inferentia (trn1 / trn2 / inf2)     |

This single choice decides which template set is used (GPU vLLM DLC vs
vLLM-Neuron + compile step).

---

## `source` (required)

Where the weights come from.

```yaml
source:
  hf_id: openai/gpt-oss-20b     # required — the HuggingFace repo id
  gated: true                   # true if you must accept a license / use a token
  size_gb: 13                   # approximate on-disk size in GB (used for storage + instance sizing)
  ignore_patterns:              # optional — file globs to skip on download
    - "*.msgpack"
    - "*.h5"
```

- `gated: true` means the download job will require the HuggingFace token
  (stored in AWS Secrets Manager, wired via External Secrets Operator). The
  wizard reminds you to accept the license first.

## `serving`

How the model is served. All fields optional with safe defaults.

```yaml
serving:
  served_name: gpt-oss-20b      # the model name clients call; defaults to `id`
  tokenizer: openai/gpt-oss-20b # defaults to source.hf_id
  quantization: mxfp4           # mxfp4 | fp8 | w4a16 | bf16 | none  (documentation + manifest)
  max_model_len: 16384          # context window; default 8192
  gpu_memory_utilization: 0.90  # GPU only; default 0.90
  max_num_seqs: 256             # scheduler ceiling; default 256
  max_num_batched_tokens: 8192  # default 8192
  extra_args:                   # optional raw vLLM flags appended verbatim
    - "--async-scheduling"
    - "--no-enable-prefix-caching"
```

## `resources`

Container requests/limits. Optional; defaults are derived from `source.size_gb`.

```yaml
resources:
  cpu_request: "4"
  cpu_limit: "8"
  memory_request: "24Gi"
  memory_limit: "30Gi"
  vram_gb_required: 16          # used by dynamic instance selection
```

---

## `instances` (required for real deployments)

Drives **dynamic instance selection**. List the accelerator families this model
is allowed to run on, cheapest / most-preferred first. The tool picks the first
family that is actually available in your region and account, and falls back
down the list automatically.

```yaml
instances:
  tensor_parallel_size: 1       # GPUs (or NeuronCores) per replica
  candidates:                   # ordered by preference (first = preferred)
    - g6e.2xlarge
    - g6.2xlarge
    - g5.2xlarge
```

For Neuron:

```yaml
instances:
  neuron_cores: 8               # = tensor-parallel degree; must match compile time
  candidates:
    - trn2.48xlarge
    - trn1.32xlarge
  node_group: trn2-48xl-ngc     # optional managed node group fallback name
```

## `neuron` (Neuron models only)

Compile-time parameters. These are **baked into the compiled artifact** — the
tool warns you that changing them later forces a recompile.

```yaml
neuron:
  source_folder: gpt-oss-20b        # S3 folder with the bf16 weights (compile input)
  neff_folder: gpt-oss-20b-neuron   # S3 folder for compiled artifacts (defaults to <source>-neuron)
  sequence_length: 16384
  batch_size: 8
  auto_cast_type: bf16              # Neuron supports fp16/bf16 only (no 4-bit)
  instance_family: trn2             # nodeSelector value for the Karpenter path
```

---

## `benchmark`

Controls benchmarking. Fully optional — every field has a default.

```yaml
benchmark:
  auto: false                       # if true, `oai deploy` runs a benchmark right after
  profiles:                         # which workload profiles to run
    - configs/workload_profiles/realtime_v1.yaml
    - configs/workload_profiles/batch_v1.yaml
  instance_hourly_usd: 2.24         # $/hr for cost math (VERIFY for your region)
  dataset: null                     # optional custom dataset (s3://... or a key)
  skip_batch: false                 # only run the realtime profile if true
```

You can also override any of this at the command line:

```
oai deploy my-model --benchmark --profile realtime_v1
oai benchmark my-model --dataset s3://my-bucket/my-prompts.jsonl
```

## `tags`

Optional key/value pairs merged into the standard resource tags.

```yaml
tags:
  Owner: jane@company.com
  CostCenter: "1234"
```
