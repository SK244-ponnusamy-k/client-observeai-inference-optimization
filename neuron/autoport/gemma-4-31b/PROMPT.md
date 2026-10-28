# AutoPort prompt — Gemma 4 31B → vLLM Neuron (trn2.48xlarge)

Paste everything in the box below into Claude Code on the AutoPort box, from
`~/autoport/gemma4/vllm-neuron` (set up by `neuron/autoport/setup-box.sh`).
`--review` makes the agent stop after the architecture analysis so you can check
its TP choice and reference model before it generates code.

```text
Port the model gemma4 from google/gemma-4-31B-it to vLLM Neuron. --review

Context for this port:

Weights are already local at /home/ubuntu/models/gemma-4-31B-it (bf16, 62.5 GB).
Use them; do not download from Hugging Face (the model is gated and there is no
HF token on this box).

Text only. The checkpoint's architecture is Gemma4ForConditionalGeneration
(language model + vision + audio towers). Port the language model and register it
under the exact architecture string "Gemma4ForConditionalGeneration", so that
`vllm serve` on this checkpoint picks it up; load only the language-model weights
and skip the vision/audio towers and their projector weights.

Architecture facts from text_config (verify them against the modeling source):
- 60 decoder layers; layer_types repeats 5 x sliding_attention then 1 x
  full_attention (10 full-attention layers).
- Sliding layers: window 1024, 32 query heads, 16 KV heads, head_dim 256,
  RoPE theta 10000.
- Full-attention layers have a DIFFERENT shape: global_head_dim 512,
  num_global_key_value_heads 4, attention_k_eq_v true (keys and values share
  one projection), partial_rotary_factor 0.25 with rope_type "proportional",
  RoPE theta 1e6.
- final_logit_softcapping 30.0; hidden_activation gelu_pytorch_tanh;
  intermediate_size 21504; hidden_size 5376; tie_word_embeddings true;
  vocab_size 262144. Also check what hidden_size_per_layer_input and
  num_kv_shared_layers mean for the 31B model (they may be unused at this size).

Neuron SDK 2.33's NKI library ships kernels tuned for Gemma4 in its models/
folder; look there before writing new kernels.

Serving target (used for the benchmark afterwards):
- one trn2.48xlarge, max_model_len 8192, bf16;
- prefer a TP size that leaves room for several data-parallel replicas per node
  (e.g. TP8 x 8 replicas). Note that 4 global KV heads cannot split evenly past
  TP4, so say how you handle KV heads for the full-attention layers at the TP you
  choose;
- report the recommended TP, max batch size and the KV-cache budget per replica.

Acceptance: logit validation sigma < 5 at top-5 and top-50 against the CPU
reference, and Equivalence skill components at R < 1.2. If a level fails, report
it and the likely cause instead of hiding it.

When validation is done, write a short HANDOFF.md next to the model package with:
the TP / max batch / max_model_len you validated, the exact `vllm serve` flags,
the env vars it needs, and `git -C ~/autoport/gemma4/vllm-neuron diff --stat`
against the base commit in ~/autoport/gemma4/BASE_COMMIT.
```

## What you get back

| Path (in `~/autoport/gemma4/vllm-neuron`) | What |
|---|---|
| `vllm_neuron/model/gemma4/` | the port: `model.py`, `config.py`, `factory.py`, `README.md` |
| `vllm_neuron/model/registry.py` | modified: `Gemma4ForConditionalGeneration` registered |
| `examples/vllm_neuron/models/gemma4/results/results.md` | offline + online serving results |
| `test/vllm_neuron/model/gemma4/bf16/e2e/test_logits.py` | logit validation |
| `vllm_neuron/model/gemma4/HANDOFF.md` | what `oai` needs to serve it |

The patched `vllm_neuron` package is the bundle to serve on EKS: overlay it on
the SDK 2.33 vLLM image and run the `vllm serve` flags from `HANDOFF.md`.
