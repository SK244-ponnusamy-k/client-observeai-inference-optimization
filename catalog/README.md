# Model Catalog — onboard a new model in one file

This folder is the **only place you need to touch to add a new model**. You do
not need to know Kubernetes, YAML hardening, IAM, or shell scripting.

## What is a catalog entry?

Every model the framework can deploy is described by **one small file**:

```
catalog/models/<your-model-id>.yaml
```

You fill in plain fields (model name, where to download it from, how big it is,
which chips it should run on). The `oai` tool reads that file and **generates
all the complex Kubernetes and deployment files for you** — with the exact same
security hardening the platform team already reviewed.

## The three ways to add a model

Pick whichever fits you. They all end up at the same place.

### 1. Guided wizard (recommended for non-technical users)

```
oai model new
```

It asks you plain questions ("What is the HuggingFace ID?", "Is it a gated
model?", "GPU or Trainium?") and writes the catalog file for you. You never open
a YAML editor.

### 2. Copy the template

```
cp catalog/TEMPLATE.yaml catalog/models/my-model.yaml
# edit the handful of fields, every one is explained inline
```

### 3. Copy an existing example

Look in `catalog/models/` for a model similar to yours (a GPU model or a
Trainium/Neuron model) and copy it.

## After you have a catalog entry

```
oai model validate my-model     # checks your entry for mistakes (safe, read-only)
oai model generate my-model     # writes all the deployment files
oai download my-model           # pulls weights from HuggingFace into S3
oai deploy   my-model           # runs it on the cluster (auto-picks an instance)
oai deploy   my-model --benchmark   # deploy AND benchmark automatically
oai status                      # see everything that is running + cost
oai stop     my-model           # stop it and free the expensive GPU/Trainium node
```

Every command prints plain-language progress and, if something goes wrong,
tells you exactly what to do next. Nothing is silent.

## What you do NOT have to understand

The `oai model generate` step produces these files for you from your one
catalog entry — you never write them by hand:

- `vllm/models/<id>/deployment.yaml`, `service.yaml`, `pvc.yaml`, `model.env`
- `vllm/models/<id>/deploy.sh`, `stop.sh`
- `model-download/<id>/download.sh`, `job.yaml`
- `configs/manifests/<id>-<hardware>.yaml` (the benchmark run manifest)
- for Trainium models: `compile-job.yaml` and the Neuron deployment variant

All of them keep the platform's security posture: non-root, read-only root
filesystem, dropped Linux capabilities, least-privilege IAM, pinned images,
no public load balancers.

## Field reference

See `catalog/SCHEMA.md` for every field, whether it is required, and what it
does.
