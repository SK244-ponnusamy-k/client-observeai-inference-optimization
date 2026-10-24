"""
File: server.py
Purpose: OpenAI-compatible HTTP shim serving the Qwen3.5 NxDI custom model on Trainium.
Owner: genai-platform@shellkode
Created: 2026-10-18 | Deps: fastapi, uvicorn, torch, transformers, neuronx_distributed_inference
Jira: client-observeai-inference-optimization (Qwen3.5 nxdi bring-up)

Why this exists
---------------
Qwen3.5 is a hybrid DeltaNet + GQA decoder. vLLM-Neuron does NOT support this
architecture out of the box, so we cannot `vllm serve` it. The reference
NeuronX Distributed Inference (NxDI) contrib implementation
(`src/modeling_qwen35.py` + custom NKI DeltaNet kernels) exposes only a Python
API: build config -> compile -> load -> generate via HuggingFaceGenerationAdapter.

This module wraps that Python API in a thin FastAPI server that implements the
subset of the OpenAI REST API the benchmark/quality harness depends on
(`/health`, `/v1/completions`, `/v1/chat/completions`, `/v1/models`). The HTTP
contract matches the vLLM serving path on :8000, so every downstream stage
(benchmark, observability, teardown) is unchanged — only this deploy differs,
per the framework's "shared tail" rule.

Configuration is env-only (Pydantic settings); nothing is hardcoded. The model
is compiled on first boot if a traced/NEFF artifact is not already present,
otherwise the cached artifact is loaded.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass
class _BatchItem:
    prompt: str
    max_new_tokens: int
    correlation_id: str
    future: asyncio.Future
    ignore_eos: bool = False

# ──────────────────────────────────────────────────────────────────────────────
# Structured JSON logging (correlation id per request)
# ──────────────────────────────────────────────────────────────────────────────


class _JsonLogFormatter(logging.Formatter):
    """Minimal structured JSON log line. No PII — we log token counts, not text."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "service": "qwen35-nxdi-shim",
            "msg": record.getMessage(),
        }
        corr = getattr(record, "correlation_id", None)
        if corr:
            payload["correlation_id"] = corr
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


_handler = logging.StreamHandler(stream=sys.stdout)
_handler.setFormatter(_JsonLogFormatter())
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
logger = logging.getLogger("qwen35_nxdi_server")


def _log(level: int, msg: str, correlation_id: Optional[str] = None, **fields: Any) -> None:
    logger.log(level, msg, extra={"correlation_id": correlation_id, "fields": fields})


# ──────────────────────────────────────────────────────────────────────────────
# Settings — env only, validated by Pydantic. No hardcoded paths/endpoints.
# ──────────────────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Runtime configuration. Every value is overridable via environment."""

    model_config = SettingsConfigDict(env_prefix="QWEN35_", extra="ignore")

    # Filesystem paths (populated by the init container that syncs S3 -> local).
    model_path: str = Field(default="/model", description="HF weights dir (config.json + safetensors)")
    compiled_path: str = Field(default="/compiled", description="Traced/NEFF artifact dir")

    # NxDI / Neuron compile-time parameters. These are BAKED into the compiled
    # artifact — changing them forces a recompile.
    tp_degree: int = Field(default=4, description="Tensor-parallel degree = NeuronCores")
    seq_len: int = Field(default=160, description="Max sequence length (reference-validated: 160)")
    max_batch_size: int = Field(default=1, description="Max concurrent sequences")
    logical_nc_config: int = Field(default=2, description="LNC: fuse N physical cores into 1 logical (trn2=2)")
    torch_dtype: str = Field(default="bfloat16", description="bfloat16 | float16 | float32")
    top_k_on_device: int = Field(default=1, description="On-device sampling top-k (1 = greedy)")

    # Server. Binds all interfaces so the in-cluster Service can reach the pod;
    # exposure is controlled by k8s Service + NetworkPolicy, not by the bind addr.
    host: str = Field(default="0.0.0.0", description="Bind host")  # nosec B104
    port: int = Field(default=8000, description="Bind port — matches the vLLM serving contract")
    served_model_name: str = Field(default="Qwen3.5-9B", description="Name clients call in the `model` field")

    # Generation defaults / ceilings
    default_max_tokens: int = Field(default=256, description="max_new_tokens when a request omits it")

    def resolved_dtype(self) -> torch.dtype:
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        key = self.torch_dtype.lower()
        if key not in mapping:
            raise ValueError(
                f"Unsupported QWEN35_TORCH_DTYPE={self.torch_dtype!r}; "
                f"expected one of {sorted(mapping)}"
            )
        return mapping[key]


# ──────────────────────────────────────────────────────────────────────────────
# Model engine — loads the NxDI model once, serializes generate() calls.
# ──────────────────────────────────────────────────────────────────────────────


class Qwen35Engine:
    """Owns the compiled NxDI model + tokenizer and runs generation.

    The compiled Neuron graph is single-tenant: concurrent generate() calls on
    one engine are not safe. Concurrency is achieved via dynamic micro-batching
    up to `max_batch_size` in a non-blocking background worker.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = Lock()
        self._model: Any = None
        self._gen_model: Any = None
        self._tokenizer: Any = None
        self._ready = False
        self._queue: Optional[asyncio.Queue[_BatchItem]] = None
        self._worker_task: Optional[asyncio.Task] = None

    @property
    def ready(self) -> bool:
        return self._ready

    def load(self) -> None:
        """Build config, compile if needed, load the model and tokenizer.

        Raises:
            RuntimeError: if the reference modeling package or weights are missing.
        """
        s = self._settings
        # Ensure the vendored reference package (src/) is importable.
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)

        try:
            from transformers import AutoTokenizer  # noqa: WPS433 (lazy heavy import)
            from neuronx_distributed_inference.models.config import (  # noqa: WPS433
                NeuronConfig,
                OnDeviceSamplingConfig,
            )
            from neuronx_distributed_inference.utils.hf_adapter import (  # noqa: WPS433
                HuggingFaceGenerationAdapter,
            )
            from src.modeling_qwen35 import (  # noqa: WPS433
                Qwen35InferenceConfig,
                NeuronQwen35ForCausalLM,
            )
        except ImportError as exc:  # pragma: no cover - env-specific
            raise RuntimeError(
                "NxDI runtime not available. This server must run inside the "
                "NeuronX Distributed Inference environment (SDK 2.29+, NKI 0.3). "
                f"Import failed: {exc}"
            ) from exc

        config_json = os.path.join(s.model_path, "config.json")
        if not os.path.isfile(config_json):
            raise RuntimeError(f"Weights not found: missing {config_json}")

        _log(logging.INFO, "Building NeuronConfig", tp_degree=s.tp_degree, seq_len=s.seq_len,
             max_batch_size=s.max_batch_size, dtype=s.torch_dtype, lnc=s.logical_nc_config)

        neuron_config = NeuronConfig(
            tp_degree=s.tp_degree,
            batch_size=s.max_batch_size,
            ctx_batch_size=s.max_batch_size,
            tkg_batch_size=s.max_batch_size,
            max_batch_size=s.max_batch_size,
            seq_len=s.seq_len,
            torch_dtype=s.resolved_dtype(),
            logical_nc_config=s.logical_nc_config,
            enable_bucketing=False,
            flash_decoding_enabled=False,
            on_device_sampling_config=OnDeviceSamplingConfig(top_k=s.top_k_on_device),
            save_sharded_checkpoint=True,
        )

        with open(config_json, encoding="utf-8") as handle:
            hf_config = json.load(handle)
        text_config = hf_config.get("text_config", hf_config)
        config_dict = dict(text_config)
        config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
        config_dict["tie_word_embeddings"] = hf_config.get(
            "tie_word_embeddings",
            text_config.get("tie_word_embeddings", False),
        )
        if "rope_parameters" in text_config:
            config_dict["rope_theta"] = text_config["rope_parameters"].get("rope_theta", 10000000)

        config = Qwen35InferenceConfig(neuron_config=neuron_config, **config_dict)

        compiled_ready = os.path.isdir(s.compiled_path) and any(
            os.scandir(s.compiled_path)
        )
        if not compiled_ready:
            _log(logging.INFO, "No compiled artifact found — compiling (first boot)",
                 compiled_path=s.compiled_path)
            os.makedirs(s.compiled_path, exist_ok=True)
            model = NeuronQwen35ForCausalLM(s.model_path, config)
            model.compile(s.compiled_path)
            _log(logging.INFO, "Compile complete", compiled_path=s.compiled_path)
        else:
            _log(logging.INFO, "Reusing cached compiled artifact", compiled_path=s.compiled_path)

        model = NeuronQwen35ForCausalLM(s.compiled_path)
        model.load(s.compiled_path)

        # model_path is a LOCAL dir synced from our own S3 bucket (not a Hub
        # repo), so there is no remote download/revision to pin here.
        # Left-padding is required for causal LM batch generation.
        tokenizer = AutoTokenizer.from_pretrained(s.model_path, padding_side="left")  # nosec B615
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._model = model
        self._gen_model = HuggingFaceGenerationAdapter(model)

        # Text-only build (QWEN35_TEXT_ONLY=1): the compiled graph is a plain
        # text decoder with the stock NxDI forward signature. The VL llava_args
        # monkeypatch below is only needed for the 24-input VL artifact, so skip
        # it — the stock HuggingFaceGenerationAdapter + plain generate() works.
        if os.environ.get("QWEN35_TEXT_ONLY") == "1":
            self._tokenizer = tokenizer
            self._ready = True
            _log(
                logging.INFO,
                "Model ready (text-only)",
                served_model_name=s.served_model_name,
            )
            return

        # The compiled graph is a vision-language model with a 24-input forward
        # signature (last 3 = mrope_position_ids, vision_embeddings, vision_mask).
        # Passing llava_args to generate() does NOT survive HF 4.57's kwarg
        # handling into the per-step model call, so forward() fails with
        # "missing value for argument_9". Instead we wrap the adapter's
        # prepare_inputs_for_generation so EVERY step injects dummy text-only
        # VL tensors as llava_args, which _get_model_outputs unpacks into the
        # trailing positional args. Sized to the step's own input so the
        # text-only no-op vision branch is taken. No recompile needed.
        _hidden_size = int(self._model.config.hidden_size)
        _vdtype = self._model.config.neuron_config.torch_dtype
        # The CTE (prefill) graph runs at the compiled bucket sequence length,
        # NOT the raw prompt length. get_model_output rejects vision tensors whose
        # seq dim != the padded bucket (it logged "[4,17,1] does not match padded
        # sequence length 2048"), so the dummy VL tensors MUST be sized to the
        # compiled seq_len. Decode steps use seq dim 1.
        _cte_seq_len = int(self._model.config.neuron_config.seq_len)
        _orig_prepare = self._gen_model.prepare_inputs_for_generation

        def _prepare_with_llava(input_ids, *args, **kwargs):
            model_inputs = _orig_prepare(input_ids, *args, **kwargs)
            ids = model_inputs.get("input_ids")
            if ids is None:
                ids = input_ids
            bsz = int(ids.shape[0])
            step_len = int(ids.shape[1])
            # Prefill: size to the compiled bucket so get_model_output takes the
            # text-only no-op branch (seq dim == seq_length). Decode: size to 1.
            slen = _cte_seq_len if step_len > 1 else 1
            ve = torch.zeros((bsz, slen, _hidden_size), dtype=_vdtype)
            vm = torch.full((bsz, slen, 1), fill_value=slen - 1, dtype=torch.int32)
            mr = (
                torch.arange(0, slen, dtype=torch.int32)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(3, bsz, -1)
                .contiguous()
            )
            # NeuronQwen35ForCausalLM._get_model_outputs (our override) reads
            #   vision_embeddings = llava_args[0]
            #   vision_mask       = llava_args[1]
            #   mrope             = llava_args[2]
            # and builds the 14 empty placeholders itself, then calls the traced
            # model with 7 real + *empties(14) + mrope + vis_emb + vis_mask = 24.
            # So llava_args must be EXACTLY these 3 tensors in this order.
            model_inputs["llava_args"] = [ve, vm, mr]
            return model_inputs

        self._gen_model.prepare_inputs_for_generation = _prepare_with_llava

        self._tokenizer = tokenizer
        self._ready = True
        _log(logging.INFO, "Model ready", served_model_name=s.served_model_name)

    def start_worker(self) -> None:
        """Initialize the async request queue and start the batch worker loop."""
        self._queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._batch_loop())

    async def stop_worker(self) -> None:
        """Cancel the background batch worker loop cleanly."""
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    async def _batch_loop(self) -> None:
        """Worker loop that dynamically batches requests up to max_batch_size."""
        while True:
            try:
                assert self._queue is not None
                item = await self._queue.get()
            except asyncio.CancelledError:
                break

            items = [item]
            max_batch = self._settings.max_batch_size
            # 10ms micro-batch window: collect concurrent requests without adding latency
            batch_window_s = 0.010
            t_start = time.time()
            while len(items) < max_batch:
                time_left = max(0.001, batch_window_s - (time.time() - t_start))
                try:
                    next_item = await asyncio.wait_for(self._queue.get(), timeout=time_left)
                    items.append(next_item)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    break

            try:
                # Offload blocking inference to thread pool to keep event loop responsive
                results = await asyncio.to_thread(self._process_batch, items)
                for itm, res in zip(items, results):
                    if not itm.future.done():
                        itm.future.set_result(res)
            except Exception as exc:  # pragma: no cover
                _log(logging.ERROR, "batch processing failed", error=str(exc))
                for itm in items:
                    if not itm.future.done():
                        itm.future.set_exception(exc)

    def _process_batch(self, items: List[_BatchItem]) -> List[Dict[str, int | str]]:
        """Synchronously execute a batched generation on the compiled model."""
        if not self._ready:
            raise RuntimeError("Model not loaded")

        from transformers import GenerationConfig  # noqa: WPS433

        s = self._settings
        tokenizer = self._tokenizer
        prompts = [it.prompt for it in items]
        max_new_tokens = max(it.max_new_tokens for it in items)

        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        prompt_lengths = [int(mask.sum().item()) for mask in inputs.attention_mask]
        padded_prompt_len = int(inputs.input_ids.shape[-1])

        # ignore_eos (benchmark sends --ignore-eos): force the model to emit the
        # full max_new_tokens of real content instead of stopping at the first
        # EOS. This model is a thinking model whose greedy decode otherwise emits
        # end/special tokens almost immediately (decoded content was empty, so
        # vllm bench serve counted 0 output tokens). With ignore_eos we drop the
        # eos stop criterion and floor min_new_tokens so throughput is stable and
        # measurable. Any single item requesting ignore_eos applies to the batch.
        ignore_eos = any(getattr(it, "ignore_eos", False) for it in items)
        gen_kwargs = dict(
            do_sample=s.top_k_on_device > 1,
            top_k=s.top_k_on_device,
            pad_token_id=tokenizer.pad_token_id,
        )
        if ignore_eos:
            gen_kwargs["eos_token_id"] = None
            gen_kwargs["min_new_tokens"] = max_new_tokens
        else:
            gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
        gen_config = GenerationConfig(**gen_kwargs)

        # VL inputs (llava_args) are injected per-step by the wrapped
        # prepare_inputs_for_generation installed at load time, so the generate
        # call stays a plain text request.
        t0 = time.time()
        with self._lock:
            outputs = self._gen_model.generate(
                inputs.input_ids,
                generation_config=gen_config,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
            )
        latency_ms = round((time.time() - t0) * 1000.0, 1)

        results: List[Dict[str, int | str]] = []
        for i, itm in enumerate(items):
            full = outputs[i]
            completion_ids = full[padded_prompt_len:]
            text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            # Strip any leading thinking block so the returned content is the
            # answer, not the chain-of-thought (benchmark + quality want the
            # visible answer). No-op when enable_thinking=False already removed it.
            if "</think>" in text:
                text = text.split("</think>", 1)[1].lstrip("\n")
            completion_tokens = int(completion_ids.shape[-1])

            _log(logging.INFO, "generation complete", correlation_id=itm.correlation_id,
                 prompt_tokens=prompt_lengths[i], completion_tokens=completion_tokens,
                 batch_size=len(items), latency_ms=latency_ms)

            results.append({
                "text": text,
                "prompt_tokens": prompt_lengths[i],
                "completion_tokens": completion_tokens,
            })
        return results

    async def generate_async(
        self, prompt: str, max_new_tokens: int, correlation_id: str,
        ignore_eos: bool = False,
    ) -> Dict[str, int | str]:
        """Enqueue request into the dynamic batch queue and await result."""
        if not self._ready or self._queue is None:
            raise RuntimeError("Model not loaded or queue not ready")

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put(
            _BatchItem(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                correlation_id=correlation_id,
                future=future,
                ignore_eos=ignore_eos,
            )
        )
        return await future

    def generate(self, prompt: str, max_new_tokens: int, correlation_id: str) -> Dict[str, int | str]:
        """Synchronous fallback for direct generation."""
        item = _BatchItem(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            correlation_id=correlation_id,
            future=asyncio.Future(),
        )
        return self._process_batch([item])[0]

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Render chat messages to a prompt string via the model's chat template.

        Qwen3.5 is a thinking model: its default chat template appends an open
        <think> block, after which greedy on-device sampling emits reasoning /
        end tokens that decode to empty text (benchmark counted 0 output
        tokens). Render with enable_thinking=False so the assistant turn starts
        directly at the answer. Falls back to the default template on older
        tokenizers that don't accept the kwarg.
        """
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible request/response schemas (subset used by the harness)
# ──────────────────────────────────────────────────────────────────────────────


class CompletionRequest(BaseModel):
    # Tolerate any extra OpenAI fields vllm bench serve sends (logprobs,
    # stream_options, n, stop, presence_penalty, ...) so they don't 422.
    model_config = ConfigDict(extra="ignore")
    model: Optional[str] = None
    prompt: str
    max_tokens: Optional[int] = None
    # Accepted for API compatibility; sampling is governed by on-device config.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    ignore_eos: bool = False


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    # Tolerate any extra OpenAI fields vllm bench serve sends so they don't 422.
    model_config = ConfigDict(extra="ignore")
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    ignore_eos: bool = False


_SETTINGS = Settings()
_ENGINE = Qwen35Engine(_SETTINGS)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401
    """Load the model at startup; surface load failures as a crash (fail fast)."""
    _log(logging.INFO, "Starting Qwen3.5 NxDI shim", served_model_name=_SETTINGS.served_model_name,
         port=_SETTINGS.port)
    _ENGINE.load()
    _ENGINE.start_worker()
    yield
    await _ENGINE.stop_worker()


app = FastAPI(title="Qwen3.5 NxDI OpenAI-compatible shim", lifespan=lifespan)


@app.middleware("http")
async def _correlation_id_mw(request: Request, call_next):
    corr = request.headers.get("X-Correlation-ID") or uuid.uuid4().hex
    request.state.correlation_id = corr
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = corr
    return response


@app.get("/health")
async def health() -> JSONResponse:
    """Readiness: 200 only once the compiled model is loaded (matches vLLM /health)."""
    if not _ENGINE.ready:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return JSONResponse(status_code=200, content={"status": "ok"})


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": _SETTINGS.served_model_name,
                "object": "model",
                "owned_by": "observeai-inference-optimization",
            }
        ],
    }


def _max_tokens(requested: Optional[int]) -> int:
    return requested if requested and requested > 0 else _SETTINGS.default_max_tokens


def _sse_completion_stream(corr: str, model: str, result: Dict[str, Any]) -> Any:
    """Emit a text_completion result as OpenAI-style SSE chunks.

    The engine is non-streaming (full text computed at once), so we send the
    text as one content chunk then a final chunk with finish_reason=stop and
    the [DONE] sentinel. This satisfies `vllm bench serve`, which streams by
    default and tallies output tokens from the streamed deltas.
    """
    created = int(time.time())
    first = {
        "id": f"cmpl-{corr}", "object": "text_completion", "created": created,
        "model": model,
        "choices": [{"index": 0, "text": result["text"], "finish_reason": None, "logprobs": None}],
    }
    yield f"data: {json.dumps(first)}\n\n"
    final = {
        "id": f"cmpl-{corr}", "object": "text_completion", "created": created,
        "model": model,
        "choices": [{"index": 0, "text": "", "finish_reason": "stop", "logprobs": None}],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


def _sse_chat_stream(corr: str, model: str, result: Dict[str, Any]) -> Any:
    """Emit a chat.completion result as OpenAI-style SSE chunks (role, content, stop)."""
    created = int(time.time())
    role_chunk = {
        "id": f"chatcmpl-{corr}", "object": "chat.completion.chunk", "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(role_chunk)}\n\n"
    content_chunk = {
        "id": f"chatcmpl-{corr}", "object": "chat.completion.chunk", "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": result["text"]}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(content_chunk)}\n\n"
    final = {
        "id": f"chatcmpl-{corr}", "object": "chat.completion.chunk", "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(req: CompletionRequest, request: Request) -> Any:
    corr = getattr(request.state, "correlation_id", uuid.uuid4().hex)
    try:
        result = await _ENGINE.generate_async(
            req.prompt, _max_tokens(req.max_tokens), corr, ignore_eos=req.ignore_eos
        )
    except RuntimeError as exc:
        _log(logging.ERROR, "generation failed", correlation_id=corr, error=str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    model = req.model or _SETTINGS.served_model_name
    if req.stream:
        return StreamingResponse(
            _sse_completion_stream(corr, model, result), media_type="text/event-stream"
        )
    return {
        "id": f"cmpl-{corr}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "text": result["text"], "finish_reason": "stop", "logprobs": None}
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request) -> Any:
    corr = getattr(request.state, "correlation_id", uuid.uuid4().hex)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = _ENGINE.apply_chat_template(messages)
    try:
        result = await _ENGINE.generate_async(
            prompt, _max_tokens(req.max_tokens), corr, ignore_eos=req.ignore_eos
        )
    except RuntimeError as exc:
        _log(logging.ERROR, "chat generation failed", correlation_id=corr, error=str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    model = req.model or _SETTINGS.served_model_name
    if req.stream:
        return StreamingResponse(
            _sse_chat_stream(corr, model, result), media_type="text/event-stream"
        )
    return {
        "id": f"chatcmpl-{corr}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["text"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host=_SETTINGS.host, port=_SETTINGS.port, log_config=None)
