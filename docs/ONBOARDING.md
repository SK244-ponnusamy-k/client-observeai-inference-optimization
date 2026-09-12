# Onboarding a model — the simple guide

This guide is for anyone who wants to add a model and run it, without needing to
know Kubernetes, YAML, or shell scripting. If you can fill in a short form, you
can onboard a model.

Everything is driven by one command: `oai`.

---

## Before you start (one-time cluster + monitoring setup)

If your environment is already running, skip this — a regular user does not touch
it. If you are setting up from scratch, `oai cluster` does the whole thing, and
every command reuses what already exists instead of recreating it:

```
oai cluster bootstrap                 # create the EKS cluster + S3 + IAM + node pools
oai cluster setup-monitoring          # install Prometheus + Grafana + GPU metrics (DCGM)
oai cluster setup-neuron              # (Trainium only) create the trn2 node group
oai cluster status                    # one view: cluster + monitoring + node group health
```

Notes:
- `oai cluster bootstrap` reuses the cluster if it already exists (pass `--force`
  to reconcile). Use `--compute gpu`, `--compute gpu,g7`, `--compute neuron`, or
  `--compute all` to choose which accelerator pools to enable.
- `oai cluster setup-monitoring` is what powers the Grafana dashboards and the
  GPU-utilization column in your benchmark results. It reuses an existing
  install.
- To tear everything down later: `oai cluster teardown` (keeps your S3 weights
  and IAM roles; asks you to type the cluster name to confirm).

You also need the basic tools installed: AWS CLI v2 (configured), `kubectl`,
`eksctl`, `helm`, a real `bash` (Git Bash on Windows), and Python 3.11+. And
`config/config.env` filled in (bucket, region, cluster name).

To use `oai`, put the launcher on your PATH or call it from the project folder:

- Windows PowerShell: `.\oai.ps1 <command>`
- Git Bash / WSL / macOS / Linux: `./oai <command>`

Throughout this guide we write `oai <command>`.

---

## The whole flow in five commands

```
oai model new             # 1. describe the model (guided questions)
oai model generate <id>   # 2. build the deployment files for it
oai download <id>         # 3. copy the weights from HuggingFace into S3
oai deploy <id>           # 4. run it (an instance is auto-picked)
oai benchmark <id>        # 5. measure it (optional)
```

When you are done:

```
oai stop <id>             # frees the expensive GPU/Trainium node
```

That's it. The rest of this document explains each step and what to do if
something goes wrong.

---

## Step 1 — Describe the model

Run the wizard:

```
oai model new
```

It asks plain questions:

- **HuggingFace id** — e.g. `mistralai/Mistral-7B-Instruct-v0.3`
- **Short name** — the folder/resource name, e.g. `mistral-7b` (a good default
  is suggested)
- **GPU or Trainium** — pick `gpu` unless you specifically want AWS Trainium
- **Gated?** — say yes if the model needs a license or token on HuggingFace
- **Which instances** — accept the suggested list, or type your own preferred
  ones (cheapest/most-preferred first)
- **Quantization, context length, cost/hr** — accept the defaults if unsure
- **Auto-benchmark after deploy?** — yes if you want numbers automatically

It writes a small file at `catalog/models/<id>.yaml` and checks it for mistakes.

> Prefer not to use the wizard? Copy `catalog/TEMPLATE.yaml` to
> `catalog/models/<id>.yaml` and edit the fields. Every field is explained
> inline and in `catalog/SCHEMA.md`.

Check it any time (safe, read-only):

```
oai model validate <id>
```

It tells you, in plain language, if anything is off — for example if the model
is too big for the instances you chose, or a Trainium model was given a 4-bit
quantization it can't use.

---

## Step 2 — Build the deployment files

```
oai model generate <id>
```

This turns your one small file into all the Kubernetes and script files needed
to run the model — with the platform's security settings already applied. You
never edit those generated files; if you need a change, edit the catalog entry
and re-run generate.

> Safety: generate never overwrites files it didn't create. If you point it at a
> model that already has hand-written files, it leaves them alone and tells you.

---

## Step 3 — Download the weights

```
oai download <id>
```

This copies the model from HuggingFace into your S3 bucket (once). It skips the
download if the model is already there.

If the model is **gated**, you must first:

1. Accept the license on the model's HuggingFace page.
2. Make sure the `hf-token` secret exists in the cluster (the platform team
   usually sets this up once via AWS Secrets Manager).

`oai` reminds you of both and points to the exact page.

---

## Step 4 — Deploy

```
oai deploy <id>
```

What happens:

1. `oai` re-checks the catalog entry.
2. It **picks an instance for you**: it goes down your preferred list and
   chooses the first type that is actually available in your region and within
   your account's quota. If your first choice isn't available, it moves to the
   next and tells you why.
3. It runs the model and waits until it is healthy.

Useful variations:

```
oai deploy <id> --benchmark        # deploy AND benchmark right away
oai deploy <id> --hw g5.2xlarge    # force a specific instance
oai deploy <id> --validate         # run a quick smoke test after it's ready
```

If nothing is available, `oai` does not fail silently. It explains every option:
request more quota, add more instance types to your list, try another region,
use the managed node-group fallback (Trainium), or simply wait for capacity.

### Trainium (Neuron) models

Trainium needs two extra things compared to GPU: a **one-time node-group setup**
and a **one-time compile** the first time you deploy a given model.

**1. One-time: make sure the Trainium node group exists.**

```
oai cluster setup-neuron
```

This is idempotent and safe to run any time:

- If a trn2 node group already exists, it is **reused** — nothing is created.
- If it does not exist, it creates it once (CloudFormation stack, launch
  template, EFA security group, the managed node group, and the Neuron device
  plugin), then leaves it **idle at size 0 so it costs nothing** until you
  deploy.

Useful variants:

```
oai cluster setup-neuron --dry-run          # check capacity/AZ, create nothing
oai cluster setup-neuron --az us-east-2b    # pin an availability zone
oai cluster setup-neuron --on-demand        # use on-demand instead of spot
oai cluster status                          # show the node group + whether it's billing
```

The node group name comes from `TRN2_NODEGROUP_NAME` in `config/config.env`
(default `trn2-48xl-ngc`). Keep the `node_group` field in your Neuron catalog
entry set to the same value.

**2. First deploy compiles the model; later deploys are fast.**

```
oai deploy <id> --compile --managed-ng     # first time (compiles, can be slow)
oai deploy <id> --managed-ng               # later deploys reuse the compiled cache
```

`oai deploy` scales the node group up for you before serving, and `oai stop`
scales it back to 0 to stop billing. If you try to deploy with `--managed-ng`
before the node group exists, `oai` stops and tells you to run
`oai cluster setup-neuron` first — it never creates that infrastructure
silently during a deploy.

> If your account can instead auto-provision trn2 through Karpenter, you can skip
> the node group entirely: leave `node_group` out of the catalog entry and deploy
> without `--managed-ng`.

---

## Step 5 — Benchmark (optional)

```
oai benchmark <id>                       # runs the profiles from the catalog
oai benchmark <id> --profile realtime    # just the low-latency profile
oai benchmark <id> --profile batch       # just the high-throughput profile
oai benchmark <id> --dataset s3://bucket/my-prompts.jsonl   # your own prompts
```

Results (latency, throughput, and cost) are written to the results S3 bucket and
pushed to the Grafana dashboards. Cost math uses the `$/hr` you set in the
catalog — set it correctly for your region to get real numbers.

To benchmark automatically on every deploy, set `benchmark.auto: true` in the
catalog entry (or use `--benchmark`).

---

## Seeing what's running and stopping it

```
oai status        # which models are running + which accelerator nodes are billed
oai stop <id>     # stop a model and free its node (stops the biggest cost)
```

`oai stop` leaves your S3 weights and the monitoring stack in place, so you can
redeploy quickly later without re-downloading.

---

## When something goes wrong

Every `oai` error ends with a `->` line telling you what to do next. The most
common cases:

| Symptom | What it usually means | Fix |
|---|---|---|
| "No candidate instances are usable" | No accelerator capacity/quota right now | Request quota, add instance types, change region, or wait |
| "Model not found in S3" | You skipped the download | `oai download <id>` |
| "HuggingFace token secret not found" | Gated model, no token | Accept the license; ensure the `hf-token` secret exists |
| "kubectl ... could not reach a cluster" | kubeconfig not set | `aws eks update-kubeconfig --name <cluster> --region <region>` |
| Deploy never becomes ready | Node still provisioning, or a config issue | Re-run; or check `kubectl describe pod ...` (the exact command is printed) |

You do not need to memorize these — `oai` prints the relevant one for you.

---

## Quick reference

```
oai model new                 Create a catalog entry (guided)
oai model list                List onboarded models
oai model validate <id>       Check an entry for problems
oai model generate <id>       Build deployment files
oai download <id>             Weights -> S3
oai deploy <id>               Run it (auto-picks instance)
oai deploy <id> --benchmark   Run it and benchmark
oai stop <id>                 Stop and free the node
oai status                    What's running + cost
oai benchmark <id>            Benchmark a running model

# One-time cluster + monitoring setup (idempotent — reuses what exists)
oai cluster bootstrap         Create the EKS cluster + S3 + IAM + node pools
oai cluster setup-monitoring  Install Prometheus + Grafana + DCGM (GPU metrics)
oai cluster setup-neuron      Create (or reuse) the trn2 node group (Trainium)
oai cluster status            Cluster + monitoring + node-group health in one view
oai cluster teardown          Delete the cluster (keeps S3 weights + IAM)
```
