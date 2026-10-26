# Qwen3.5-9B (DeltaNet + GQA) on Trainium2 — Bring-up Issues & Resolutions

**Model:** Qwen3.5-9B — hybrid architecture (DeltaNet linear-attention layers + GQA full-attention layers, 32 layers, interleave pattern 3×linear : 1×full).
**Hardware:** AWS Trainium2 (`trn2.48xlarge`), EKS cluster `ai-eks-cluster` (us-east-2), namespace `oai-infopt`.
**Serving path:** Custom NxDI (NeuronX Distributed Inference) modeling code + custom NKI DeltaNet kernels, behind an OpenAI-compatible FastAPI shim (`server.py`). vLLM-Neuron does not support this architecture out of the box.
**Final config:** TP=4, seq_len=2048, batch=4, bf16, LNC=2, text-only compiled artifact.

---

## 1. HBM overflow during compile

- **Symptom:** `neuronx-cc` compile failed with HBM allocation error (NCC_EVRF009), ~54 GB/core requested vs 24 GB available.
- **Cause:** Prefill graph compiled at the full `[batch, seq_len, hidden]`; `batch=16` was too large for Trainium2 HBM.
- **Resolution:** Reduced `QWEN35_MAX_BATCH_SIZE` from 16 → 4.

## 2. Host-RAM OOM during compile

- **Symptom:** Compile job pod OOM-killed even at 96 Gi.
- **Cause:** Compiling the monolithic prefill graph (4.8M-instruction module) with the fused DeltaNet NKI kernel across all 24 DeltaNet layers re-materializes at once; exceeds host RAM.
- **Resolution:** Raised compile-job memory 96 Gi → 240 Gi; reduced `seq_len` 8192 → 4096 → 2048 to bound the module size (bucketing was unavailable — see issue 7).

## 3. emptyDir evictions (pod disk pressure)

- **Symptom:** Compile/serving pods evicted mid-run with disk-pressure.
- **Cause:** Default `emptyDir` size limits too small for the 778 MB artifact + sharded checkpoints + source weights.
- **Resolution:** Increased `emptyDir` sizes in **both** `compile-job.yaml` and `deployment.yaml`:
  - `compiled` 60 Gi → 200 Gi
  - `model-weights` → 120 Gi
  - `app` 16 Mi → 2 Gi

## 4. On-device DeltaNet kernel hang (TP-independent)

- **Symptom:** Model compiled + loaded, but every forward pass hung on-device (`NRT_TIMEOUT`, status=101), warmup never completed. First suspected a TP=16 ALLGATHER collective hang; dropping to TP=4 showed the **same** hang, proving it was TP-independent.
- **Cause:** The fused single-kernel DeltaNet CTE path ran a 32-chunk sequential in-kernel loop at seq_len=4096 that OOMed/hung the kernel.
- **Resolution:**
  - Switched to the per-chunk kernel via `USE_NKI_CHUNKED=1` (smaller per-call kernel launches).
  - Dropped `seq_len` 4096 → 2048 (16 chunks instead of 32).
  - Set TP=4 (reference-validated degree).
  - Result: warmup clean, DeltaNet kernel executes on-device.

## 5. VL-signature serving bug — `forward() is missing value for argument_9` (HTTP 503)

- **Symptom:** Model compiled, loaded, warmed up, but **every** generation request returned HTTP 503 with:
  `forward() is missing value for argument 'argument_9'` (24-arg traced signature).
- **Cause:** The custom modeling code is a vision-language (VL) model. Its `ModelWrapper.input_generator` padded the traced inputs to **24 arguments** (14 empty placeholders + 3 VL tensors: mrope / vision_embeddings / vision_mask). The installed NxDI SDK (2.31.1) generation dispatch only supplies the standard ~7 model inputs, so trailing args (`argument_9` = `current_length`, etc.) were never filled.
- **Resolution — text-only compile:** Added a `QWEN35_TEXT_ONLY=1` gate so the VL machinery is bypassed:
  - `_text_only()` helper in `modeling_qwen35.py`.
  - `input_generator` / `pad_inputs` return the stock base (7-input) set.
  - `get_required_kwargs` returns `[]` (no `llava_args`).
  - `_get_model_outputs` delegates to the stock `super()._get_model_outputs(...)`.
  - `server.py` `load()` skips the VL `_prepare_with_llava` monkeypatch and uses the stock `HuggingFaceGenerationAdapter` + plain `generate()`.
  - The original VL artifact is preserved; the text-only build is written to a **separate** S3 path (`qwen3-5-9b-nxdi-textonly-traced/`).
- **mRoPE/vision drop verified safe:** decoder falls back to standard partial RoPE from 1D `position_ids`; vision inputs are None-guarded.

## 6. Env var not wired into pod containers (the "silent" recompile failure)

- **Symptom:** After the first text-only recompile (~70 min), serving **still** failed with `argument_9`. The trace had still produced a 24-arg graph.
- **Cause (root):** `QWEN35_TEXT_ONLY=1` was set in `model.env` but **never injected into the compile/serve pod container env**. Verified directly: the completed compile pod's env had every `QWEN35_*` var **except** `QWEN35_TEXT_ONLY`. So `_text_only()` evaluated False at trace time and the VL padding path ran.
- **Resolution:** Wired the var end-to-end:
  - Added `QWEN35_TEXT_ONLY` env entry to `compile-job.yaml` and `deployment.yaml`.
  - Added `${QWEN35_TEXT_ONLY}` to **both** `envsubst` allowlists in `deploy.sh` (envsubst uses an explicit allowlist — an unlisted var would be left as a literal `${...}` string and break the YAML) plus the export list.
- **Verification that it worked this time:** The trace's "unused tensor / will be ignored" warnings appeared at **input indices 0–6** (7-arg text-only graph) instead of **0–23** (24-arg VL graph) in the prior run. Note: the `input example shape = torch.Size([4, 2048])` log line is identical for both and is NOT a reliable discriminator — the **index range of the ignored-tensor warnings** is the real signal.

## 7. Sequence bucketing unavailable in installed SDK

- **Symptom:** Attempts to compile with bucketing kwargs (`enable_bucketing`, `context_encoding_buckets`, …) were rejected by the installed `NeuronConfig`.
- **Cause:** SDK 2.31.1's `NeuronConfig` does not accept the bucket-list kwargs; `compile.py` probes the signature and falls back to `seq_len = min(seq_len, 4096)`.
- **Resolution/impact:** Accepted a single monolithic graph at `seq_len=2048` (no bucketing). Adequate for the benchmark/quality workloads (~1.5k tokens/request). Raising seq_len would need a bucketing-capable image.

## 8. Kubernetes Job template immutability on redeploy

- **Symptom:** `./oai deploy --compile` failed with `The Job "oai-infopt-nxdi-compile-qwen3-5-9b-nxdi" is invalid: spec.template: Invalid value: ...` after a code/env change.
- **Cause:** A Kubernetes **Job's `spec.template` is immutable**; `kubectl apply` cannot patch the pod template of an existing (completed) Job.
- **Resolution:** Delete the old compile Job before re-applying:
  `kubectl delete job oai-infopt-nxdi-compile-qwen3-5-9b-nxdi -n oai-infopt`.

## 9. Compile-cache skip on reused S3 path

- **Symptom:** A recompile exited 0 immediately without actually recompiling.
- **Cause:** The compile job's S3 pre-check (and `compile.py`'s own idempotency check) skips compilation if the `COMPILED_FOLDER` S3 prefix is non-empty.
- **Resolution:** Clear the artifact before forcing a recompile:
  `aws s3 rm s3://shellkode-ai-models/qwen3-5-9b-nxdi-textonly-traced/ --recursive --region us-east-2`.

## 10. Chat endpoint returns empty content

- **Symptom:** `/v1/chat/completions` returned HTTP 200 but empty `content` (generated tokens decoded to empty string); `/v1/completions` returned coherent text.
- **Cause:** Chat template + greedy decoding (`top_k=1`) on the base completion model produced an immediate stop / special tokens stripped by `skip_special_tokens`. A prompt-format/sampling behavior, not a serving fault.
- **Status:** Benchmark uses `/v1/chat/completions` with `--ignore-eos`, which forces the full output length, so throughput/latency are still measured correctly. (A proper chat template + sampling tweak would fix content, no recompile needed.)

## 11. False-pass benchmark — `No such file or directory: 'vllm'`

- **Symptom:** `./oai deploy --validate` reported "Batch benchmark: PASSED" but produced **zero data** — every concurrency level (16/32/64/128/256/500) failed with `[Errno 2] No such file or directory: 'vllm'`.
- **Cause:** The `--validate` flow (`post-deploy-validate.sh`) hardcodes `image: python:3.11-slim` for its benchmark Jobs. That image has no `vllm` CLI; the in-pod `pip install vllm` fallback failed / installed the console script to `/tmp/pip-user/bin` (off PATH). The job still exited 0, so deploy.sh marked it "PASSED" (misleading).
- **Resolution:** Run the **real** benchmark via `inference/run-benchmark.sh`, which uses `$VLLM_IMAGE` (the vLLM DLC that ships the `vllm` CLI). `vllm bench serve` is a pure HTTP load generator against the OpenAI endpoint — no GPU/Neuron needed in the benchmark pod.
- **Recommended follow-up:** patch `post-deploy-validate.sh` to use the DLC image and to fail loudly when `vllm` is missing.

## 12. On-device out-of-bounds in decode graph under sustained/batched load (status=1006)

- **Symptom:** After the text-only fix, **single** requests generated correct text (verified: a clean 480-token essay). But after several sequential/batched requests, the decode graph crashed:
  `status=1006 Execution Out-Of-Bounds Memory Access`, `scatter/gather (indirect memory copy via scalar DGE) out-of-bound access`, `model=token_generation_model`, `engine=GPSIMD`.
- **Cause (root):** The decode-path KV-cache scatter uses `position_ids` **directly and unclamped** as the sequence-axis index into the `seq_len=2048` KV buffer (SDK `kv_cache_manager.py`: `index = scatter_index if self.is_medusa else position_ids`). The DeltaNet (linear_attention) layers also emit dummy zero KV that still goes through `kv_mgr.update_cache` at `position_ids`. With batched, left-padded requests, the stock NxDI decode position (`amax(position_ids)+1`) climbs; across sustained batches the index eventually reaches/exceeds 2048 → out-of-bounds scatter.
- **Resolution:** Clamp the decode position in `get_model_output` (decode path only), right after `position_ids` is finalized, before it feeds the decoder layers and `kv_mgr.update_cache`:
  ```python
  if not is_for_context_encoding:
      position_ids = position_ids.clamp(max=self.n_positions - 1)
  ```
  Prefill left untouched (legitimately spans 0..seq_len-1). Static op → compiles cleanly. For workloads under 2048 tokens the clamp is a no-op during normal operation and acts as a hard safety bound. **Requires recompile** (the indexing is baked into the traced graph).

## 13. ~34 s "TTFT" on the 2048 build (and the 10k / 1-500 concurrency rebuild)

- **Symptom:** TTFT ~34 s under concurrency on the running TP=4 / 2048 / batch-4 build.
- **Causes (all serving/shape, not raw prefill speed):**
  1. *Fake streaming* — the shim sent the whole completion as one SSE chunk, so the benchmark's TTFT was really end-to-end latency (512 tokens x ~20 ms).
  2. *Static batch of 4, one replica* — later requests queued behind whole batches; only 1 of 16 chips (4 of 64 logical cores) was used.
  3. *No bucketing* — every prefill padded to the full compiled length.
  4. *Left padding* — NxDI expects right padding; mixed-length batches corrupted KV for every row but the longest.
- **Issue 7 correction:** bucketing was never unavailable. `NeuronConfig.__init__` takes `**kwargs`, so the `inspect.signature` probe in compile.py always failed and silently forced `seq_len=min(seq_len, 4096)` with bucketing off. compile.py now passes the kwargs and verifies them on the constructed object.
- **Resolution (10k build, `qwen3-5-9b-nxdi-textonly-10k-cb8-traced`):**
  - `SEQ_LEN=10240`, `MAX_CONTEXT_LEN=8192`, prefill buckets `1024,2048,4096,8192`, decode bucket `10240`.
  - Prefill batch 1 + continuous batching, decode batch 8. The largest prefill graph is 1x8192 = the same token-slot count as the validated 4x2048 graph (16x4096 is what overflowed HBM).
  - DeltaNet state writes go to the `seq_ids` slot when prefill batch < allocated batch (`_write_state_rows`, masked `torch.where`, see #14); the old `rows + buffer*0` broadcast would have overwritten every slot. The static-batch path is unchanged.
  - TP=4, LNC=2, `USE_NKI_CHUNKED=1`, `QWEN35_TEXT_ONLY=1`: unchanged from the running build.
  - Serving: real per-token SSE, right padding, per-request max_tokens/EOS, HTTP 400 for prompts > 8192, `/metrics`, and `router.py` running 16 TP4 replicas (one per chip) = 128 decode slots.
- **If compile hits NCC_EVRF009 (HBM):** set `QWEN35_MAX_BATCH_SIZE=4` and recompile; nothing else changes.

---

## 14. 10k CB build: every decode fails with `vector DGE` out-of-bound (status=1006)

- **Symptom:** `qwen3-5-9b-nxdi-textonly-10k-cb8-traced` loads and prefill runs, but `token_generation_model` fails with `scatter/gather (indirect memory copy via vector DGE) out-of-bound access`, then `status=1006`; the worker logs `batch processing failed`.
- **What it is not:** #12 was the KV scatter (`scalar DGE`, GPSIMD) and is clamped. Traced through NxDI 0.10 (`model_wrapper._forward_with_pad`, `kv_cache_manager`, `hf_adapter`): host-side `seq_ids` (sorted 0..7), `position_ids` (≤ 10239) and the KV scatter all stay in range.
- **Most likely cause:** the decode graph's only device-derived gather index is `input_ids`, the token sampled by the previous step. An id outside `[0, vocab)` (for example argmax over NaN prefill logits) makes the embedding gather go out of bounds. Upstream contrib Qwen3.5 also avoids `index_put` for DeltaNet state writes and marks continuous batching as unvalidated.
- **No-recompile mitigation (server.py):** `_StepTracker` checks every sampled id. An invalid id is swapped in place for the pad id before the next decode, the row ends with `finish_reason="error"`, and `invalid sampled token id` is logged with the id. The device no longer crashes, and the log confirms or rules out this cause.
- **Recompile fix (modeling_qwen35.py, next compile):** `_write_state_rows` now uses a masked `torch.where` (no `index_put`, int32 compare). Decode clamps `input_ids` into the vocab before `embed_tokens`. DeltaNet clamps `seq_ids` before the state `index_select`.
- **Confirmed (QWEN35_DEBUG_IO=1, DP=1):** CTE in/out is correct (`[760,6511,314,9338,369]` -> `11751` " Paris"). TKG input is correct too (`input_ids=[11751]`, `pos=[5]`, `seq_ids=[0]`, mask sum 5). TKG `out[0]` (int32 token) = `1001471217` = `0x3BB13CF1`, which is the bit pattern of small floats (~0.005 / ~0.03), not NaN. So the logits are finite, and the token output of the greedy sampler is corrupt.
- **Fix (v2 build):** greedy on-device sampling runs `nxd_argmax`, which uses the NKI argmax kernel unless `disable_argmax_kernel=True`. That kernel worked at batch 4 / 2048 and is the prime suspect at batch 8 / 10240 / CB. `QWEN35_DISABLE_ARGMAX_KERNEL=1` (compile.py -> NeuronConfig) switches to torch argmax plus a TP all-gather. Compiled to the new folder `qwen3-5-9b-nxdi-textonly-10k-cb8-v2-traced`. Verify with the same DEBUG_IO run before benchmarking.
- **Fallback:** comment BUILD A and uncomment BUILD B in `model.env`, then redeploy without `--compile`.

## Key operational notes

- **Deploy (reuse artifact):** `./oai deploy qwen3-5-9b-nxdi`
- **Deploy (force recompile ~70 min):** `./oai deploy qwen3-5-9b-nxdi --compile`
- **Before a recompile:** delete the old compile Job AND clear the S3 `COMPILED_FOLDER`, or the compile will be skipped / fail on Job immutability.
- **Do NOT trust `--validate`'s "PASSED"** for real perf — use `inference/run-benchmark.sh` (DLC image) for actual benchmark data.
- **compile and serve env must match** (seq_len, batch, TP, LNC, `USE_NKI_CHUNKED`, `QWEN35_TEXT_ONLY`) or the traced artifact won't load / will behave differently.
- **Text-only trace confirmation signal:** ignored-tensor warning indices 0–6 (7-arg) vs 0–23 (24-arg VL). The input-shape log line alone is not a discriminator.
- **Capacity:** trn2.48xlarge Capacity Block; recompiles are ~70 min each — budget accordingly.

## Security follow-up (outstanding)

- A Hugging Face token is present in `config/.env` — **rotate it**; it should not be committed.
