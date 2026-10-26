"""
File: server.py
Purpose: OpenAI-compatible HTTP shim serving the Qwen3.5 NxDI custom model on Trainium.
Owner: genai-platform@shellkode
Created: 2026-10-18 | Deps: fastapi, uvicorn, torch, transformers, neuronx_distributed_inference
Jira: client-observeai-inference-optimization (Qwen3.5 nxdi bring-up)

Why this exists
---------------
Qwen3.5 is a hybrid DeltaNet + GQA decoder that vLLM-Neuron cannot serve. The
reference NxDI contrib model (src/) exposes only a Python API, so this thin
FastAPI server implements the OpenAI subset the harness uses (/health,
/v1/completions, /v1/chat/completions, /v1/models, /metrics) on :8000.

One process = one TP replica. For data parallelism across the 16 Trainium2
chips of a trn2.48xlarge, router.py launches N of these on distinct NeuronCore
slices and load-balances across them.

Serving behaviour
-----------------
* Dynamic micro-batching up to the compiled decode batch (read from the
  artifact, not env), RIGHT padding (NxDI's KV/mask bookkeeping assumes it;
  left padding silently corrupted every row but the longest in a batch).
* True per-token SSE streaming: a StoppingCriteria hook sees every decode step
  and pushes tokens to the request, so TTFT = queue + prefill (previously the
  whole completion was sent as one chunk, so "TTFT" == end-to-end latency).
* Per-request max_tokens / EOS handling, prompt-length validation (HTTP 400
  instead of a device error), max_tokens capped to the compiled context.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, AsyncIterator, Dict, List, Optional, Set

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
            "worker": os.environ.get("QWEN35_WORKER_ID", "0"),
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
# Settings — env only. Compile-time shape params are only used for a first-boot
# compile; at serve time the limits come from the loaded artifact.
# ──────────────────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QWEN35_", extra="ignore")

    model_path: str = Field(default="/model")
    compiled_path: str = Field(default="/compiled")
    host: str = Field(default="0.0.0.0", description="Bind host")  # nosec B104
    port: int = Field(default=8000)
    served_model_name: str = Field(default="Qwen3.5-9B")
    default_max_tokens: int = Field(default=256)
    batch_window_ms: float = Field(default=10.0, description="Micro-batch collection window")


class PromptError(ValueError):
    """Client error (HTTP 400): prompt empty or longer than the compiled context."""


@dataclass
class _BatchItem:
    prompt_ids: List[int]
    max_new_tokens: int
    correlation_id: str
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop
    ignore_eos: bool = False
    stream: Optional[asyncio.Queue] = None


@dataclass
class _Metrics:
    running: int = 0
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0
    requests_total: int = 0
    batches_total: int = 0


@dataclass
class _RowState:
    tokens: List[int] = field(default_factory=list)
    done: bool = False
    finish_reason: str = "length"


class _StepTracker:
    """HF StoppingCriteria hook: called once per generated token for the whole batch.

    Pushes each row's new token to its stream, applies per-row max_tokens / EOS,
    and stops generation as soon as every row is finished.
    """

    def __init__(self, items: List[_BatchItem], eos_ids: Set[int],
                 vocab_size: int = 0, safe_token_id: int = 0) -> None:
        self.items = items
        self.eos_ids = eos_ids
        self.rows = [_RowState() for _ in items]
        self.vocab_size = int(vocab_size or 0)
        self.safe_token_id = int(safe_token_id)
        self.step = 0

    def __call__(self, input_ids: torch.Tensor, scores: Any = None, **kwargs: Any) -> torch.Tensor:
        last = input_ids[:, -1].tolist()
        self.step += 1
        if self.step == 1:
            _log(logging.DEBUG, "first sampled tokens (prefill output)", token_ids=last)
        for i, (item, row) in enumerate(zip(self.items, self.rows)):
            tok = int(last[i])
            # Device guard: the next decode step feeds this id straight into the
            # embedding gather. An id outside [0, vocab) (e.g. argmax over NaN
            # logits) triggers a scatter/gather OOB (status=1006) that kills the
            # whole batch. Swap in a valid id in place (HF reads input_ids[:, -1:]
            # next step) and end the row with an error instead.
            if self.vocab_size and not 0 <= tok < self.vocab_size:
                input_ids[i, -1] = self.safe_token_id
                if not row.done:
                    _log(logging.ERROR, "invalid sampled token id; ending row",
                         correlation_id=item.correlation_id, token_id=tok,
                         step=self.step, vocab_size=self.vocab_size)
                    row.done, row.finish_reason = True, "error"
                continue
            if row.done:
                continue
            if not item.ignore_eos and tok in self.eos_ids:
                row.done, row.finish_reason = True, "stop"
                continue
            row.tokens.append(tok)
            if item.stream is not None:
                item.loop.call_soon_threadsafe(item.stream.put_nowait, ("tok", tok))
            if len(row.tokens) >= item.max_new_tokens:
                row.done, row.finish_reason = True, "length"
        all_done = all(r.done for r in self.rows)
        return torch.full((input_ids.shape[0],), all_done, dtype=torch.bool, device=input_ids.device)


def _make_stopping_criteria(tracker: _StepTracker):
    from transformers import StoppingCriteria  # noqa: WPS433

    class _Hook(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):  # type: ignore[override]
            return tracker(input_ids, scores, **kwargs)

    return _Hook()


# ──────────────────────────────────────────────────────────────────────────────
# Model engine
# ──────────────────────────────────────────────────────────────────────────────


class Qwen35Engine:
    """Owns the compiled NxDI model + tokenizer; micro-batches requests onto it."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = Lock()
        self._model: Any = None
        self._gen_model: Any = None
        self._tokenizer: Any = None
        self._ready = False
        self._queue: Optional[asyncio.Queue[_BatchItem]] = None
        self._worker_task: Optional[asyncio.Task] = None
        self.seq_len = 0
        self.max_context_length = 0
        self.max_batch_size = 1
        self.pad_id = 0
        self.vocab_size = 0
        self.eos_ids: Set[int] = set()
        self.metrics = _Metrics()

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def waiting(self) -> int:
        return self._queue.qsize() if self._queue is not None else 0

    # ---- load ---------------------------------------------------------------

    def load(self) -> None:
        s = self._settings
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)

        try:
            from transformers import AutoTokenizer  # noqa: WPS433
            from neuronx_distributed_inference.utils.hf_adapter import (  # noqa: WPS433
                HuggingFaceGenerationAdapter,
            )
            from src.modeling_qwen35 import NeuronQwen35ForCausalLM  # noqa: WPS433
            import compile as qwen_compile  # noqa: WPS433  (sibling compile.py)
        except ImportError as exc:  # pragma: no cover - env-specific
            raise RuntimeError(f"NxDI runtime not available: {exc}") from exc

        if not os.path.isfile(os.path.join(s.model_path, "config.json")):
            raise RuntimeError(f"Weights not found: missing {s.model_path}/config.json")

        compiled_ready = os.path.isdir(s.compiled_path) and any(os.scandir(s.compiled_path))
        if not compiled_ready:
            params = qwen_compile.resolve_build_params()
            _log(logging.INFO, "No compiled artifact — compiling (first boot)", **params.__dict__)
            qwen_compile.compile_model(s.model_path, s.compiled_path, params)
        else:
            _log(logging.INFO, "Reusing cached compiled artifact", compiled_path=s.compiled_path)
            self._check_build_info(qwen_compile.BUILD_INFO_FILE)

        model = NeuronQwen35ForCausalLM(s.compiled_path)
        model.load(s.compiled_path)

        # Serve limits come from the artifact that was actually compiled.
        nc = model.config.neuron_config
        self.seq_len = int(nc.seq_len)
        self.max_context_length = int(getattr(nc, "max_context_length", nc.seq_len) or nc.seq_len)
        self.max_batch_size = int(getattr(nc, "tkg_batch_size", None) or nc.max_batch_size)
        if getattr(nc, "padding_side", "right") != "right":
            raise RuntimeError(f"artifact padding_side={nc.padding_side}; server pads right")
        self.pad_id = int(model.config.pad_token_id)
        self.vocab_size = int(getattr(model.config, "vocab_size", 0) or 0)

        tokenizer = AutoTokenizer.from_pretrained(s.model_path, padding_side="right")  # nosec B615
        self.eos_ids = {i for i in (tokenizer.eos_token_id, self.pad_id) if i is not None}
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end >= 0:
            self.eos_ids.add(im_end)

        self._model = model
        self._gen_model = HuggingFaceGenerationAdapter(model)
        # EOS is handled per row by _StepTracker. Clear the model-level default so
        # HF generate() doesn't re-inject it (on-device sampling bypasses
        # min_new_tokens, so an injected EOS would stop ignore_eos rows early).
        if getattr(self._gen_model, "generation_config", None) is not None:
            self._gen_model.generation_config.eos_token_id = None

        if os.environ.get("QWEN35_TEXT_ONLY") != "1":
            self._install_vl_shim()

        self._tokenizer = tokenizer
        self._ready = True
        _log(logging.INFO, "Model ready", served_model_name=s.served_model_name,
             seq_len=self.seq_len, max_context_length=self.max_context_length,
             max_batch_size=self.max_batch_size,
             continuous_batching=bool(getattr(nc, "is_continuous_batching", False)),
             ctx_buckets=getattr(nc, "context_encoding_buckets", None),
             tkg_buckets=getattr(nc, "token_generation_buckets", None))

    def _check_build_info(self, filename: str) -> None:
        path = os.path.join(self._settings.compiled_path, filename)
        if not os.path.isfile(path):
            return  # artifacts from before build-info existed
        with open(path, encoding="utf-8") as handle:
            info = json.load(handle)
        built = info.get("env", {}).get("QWEN35_TEXT_ONLY", "")
        now = os.environ.get("QWEN35_TEXT_ONLY", "")
        if built != now:
            raise RuntimeError(f"QWEN35_TEXT_ONLY mismatch: artifact={built!r} serve={now!r}")

    def _install_vl_shim(self) -> None:
        """Legacy VL (24-input) artifact support: inject dummy llava_args each step."""
        hidden = int(self._model.config.hidden_size)
        vdtype = self._model.config.neuron_config.torch_dtype
        cte_len = int(self._model.config.neuron_config.seq_len)
        orig = self._gen_model.prepare_inputs_for_generation

        def _prepare(input_ids, *args, **kwargs):
            inputs = orig(input_ids, *args, **kwargs)
            ids = inputs.get("input_ids", input_ids)
            bsz, slen = int(ids.shape[0]), (cte_len if int(ids.shape[1]) > 1 else 1)
            ve = torch.zeros((bsz, slen, hidden), dtype=vdtype)
            vm = torch.full((bsz, slen, 1), fill_value=slen - 1, dtype=torch.int32)
            mr = torch.arange(0, slen, dtype=torch.int32).unsqueeze(0).unsqueeze(0).expand(3, bsz, -1).contiguous()
            inputs["llava_args"] = [ve, vm, mr]
            return inputs

        self._gen_model.prepare_inputs_for_generation = _prepare

    # ---- request admission --------------------------------------------------

    def tokenize(self, prompt: str) -> List[int]:
        return list(self._tokenizer(prompt, add_special_tokens=False)["input_ids"])

    def admit(self, prompt_ids: List[int], requested: int) -> int:
        """Validate the prompt; return the max_new_tokens this request can have."""
        n = len(prompt_ids)
        if n == 0:
            raise PromptError("prompt is empty")
        if n > self.max_context_length:
            raise PromptError(
                f"prompt has {n} tokens; this model was compiled for prompts up to "
                f"{self.max_context_length} tokens (QWEN35_MAX_CONTEXT_LEN)"
            )
        allowance = self.seq_len - n - 1
        if allowance < 1:
            raise PromptError(f"prompt has {n} tokens; no room to generate within {self.seq_len}")
        return max(1, min(requested, allowance))

    async def submit(self, prompt_ids: List[int], max_new: int, corr: str,
                     ignore_eos: bool, stream: bool) -> _BatchItem:
        if not self._ready or self._queue is None:
            raise RuntimeError("Model not loaded")
        loop = asyncio.get_running_loop()
        item = _BatchItem(
            prompt_ids=prompt_ids, max_new_tokens=max_new, correlation_id=corr,
            future=loop.create_future(), loop=loop, ignore_eos=ignore_eos,
            stream=asyncio.Queue() if stream else None,
        )
        await self._queue.put(item)
        return item

    # ---- batch worker -------------------------------------------------------

    def start_worker(self) -> None:
        self._queue = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._batch_loop())

    async def stop_worker(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    async def _batch_loop(self) -> None:
        window = self._settings.batch_window_ms / 1000.0
        while True:
            try:
                assert self._queue is not None
                items = [await self._queue.get()]
            except asyncio.CancelledError:
                break
            t_start = time.monotonic()
            while len(items) < self.max_batch_size:
                left = window - (time.monotonic() - t_start)
                if left <= 0:
                    # Window elapsed: still drain anything already queued (no wait).
                    while len(items) < self.max_batch_size and not self._queue.empty():
                        items.append(self._queue.get_nowait())
                    break
                try:
                    items.append(await asyncio.wait_for(self._queue.get(), timeout=left))
                except asyncio.TimeoutError:
                    break

            self.metrics.running = len(items)
            try:
                results = await asyncio.to_thread(self._process_batch, items)
                for itm, res in zip(items, results):
                    if itm.stream is not None:
                        itm.stream.put_nowait(("done", res))
                    if not itm.future.done():
                        itm.future.set_result(res)
            except Exception as exc:  # pragma: no cover - device errors
                _log(logging.ERROR, "batch processing failed", error=str(exc))
                for itm in items:
                    if itm.stream is not None:
                        itm.stream.put_nowait(("error", str(exc)))
                    if not itm.future.done():
                        itm.future.set_exception(exc)
            finally:
                self.metrics.running = 0

    def _process_batch(self, items: List[_BatchItem]) -> List[Dict[str, Any]]:
        from transformers import GenerationConfig  # noqa: WPS433

        n = len(items)
        padded_len = max(len(it.prompt_ids) for it in items)
        input_ids = torch.full((n, padded_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n, padded_len), dtype=torch.long)
        for i, it in enumerate(items):
            L = len(it.prompt_ids)
            input_ids[i, :L] = torch.tensor(it.prompt_ids, dtype=torch.long)
            attention_mask[i, :L] = 1

        # All rows decode in lock-step, so the batch budget is bounded by the
        # longest prompt; each row is still cut at its own max_tokens.
        max_new = max(1, min(max(it.max_new_tokens for it in items), self.seq_len - padded_len - 1))
        tracker = _StepTracker(items, self.eos_ids, vocab_size=self.vocab_size,
                               safe_token_id=self.pad_id)
        top_k = int(getattr(self._model.config.neuron_config.on_device_sampling_config, "top_k", 1) or 1)
        gen_config = GenerationConfig(
            do_sample=top_k > 1, top_k=top_k, max_new_tokens=max_new,
            pad_token_id=self.pad_id, eos_token_id=None,
        )

        t0 = time.time()
        with self._lock:
            self._gen_model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
                stopping_criteria=[_make_stopping_criteria(tracker)],
            )
        latency_ms = round((time.time() - t0) * 1000.0, 1)

        results: List[Dict[str, Any]] = []
        for it, row in zip(items, tracker.rows):
            completion_tokens = len(row.tokens) + (1 if row.finish_reason == "stop" else 0)
            text = self._tokenizer.decode(row.tokens, skip_special_tokens=True)
            if "</think>" in text:
                text = text.split("</think>", 1)[1].lstrip("\n")
            self.metrics.prompt_tokens_total += len(it.prompt_ids)
            self.metrics.generation_tokens_total += completion_tokens
            self.metrics.requests_total += 1
            _log(logging.INFO, "generation complete", correlation_id=it.correlation_id,
                 prompt_tokens=len(it.prompt_ids), completion_tokens=completion_tokens,
                 finish_reason=row.finish_reason, batch_size=n, latency_ms=latency_ms)
            results.append({
                "text": text,
                "prompt_tokens": len(it.prompt_ids),
                "completion_tokens": completion_tokens,
                "finish_reason": row.finish_reason,
            })
        self.metrics.batches_total += 1
        return results

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Render chat messages; enable_thinking=False so the answer starts immediately."""
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def decode(self, ids: List[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=True)


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI-compatible schemas (subset used by the harness)
# ──────────────────────────────────────────────────────────────────────────────


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: Optional[str] = None
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    ignore_eos: bool = False


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    ignore_eos: bool = False


_SETTINGS = Settings()
_ENGINE = Qwen35Engine(_SETTINGS)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: D401
    _log(logging.INFO, "Starting Qwen3.5 NxDI shim", port=_SETTINGS.port)
    await asyncio.to_thread(_ENGINE.load)
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
    if not _ENGINE.ready:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return JSONResponse(status_code=200, content={"status": "ok"})


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    return {"object": "list", "data": [{
        "id": _SETTINGS.served_model_name, "object": "model",
        "owned_by": "observeai-inference-optimization", "max_model_len": _ENGINE.seq_len,
    }]}


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """vLLM-compatible gauge/counter names so load-test.py / probe samplers work."""
    m = _ENGINE.metrics
    slots = max(_ENGINE.max_batch_size, 1)
    lines = [
        f"vllm:num_requests_running {m.running}",
        f"vllm:num_requests_waiting {_ENGINE.waiting}",
        f"vllm:kv_cache_usage_perc {m.running / slots:.4f}",
        f"vllm:prompt_tokens_total {m.prompt_tokens_total}",
        f"vllm:generation_tokens_total {m.generation_tokens_total}",
        f"vllm:request_success_total {m.requests_total}",
        f"qwen35:batches_total {m.batches_total}",
        f"qwen35:decode_slots {slots}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n")


def _max_tokens(*candidates: Optional[int]) -> int:
    for c in candidates:
        if c and c > 0:
            return c
    return _SETTINGS.default_max_tokens


async def _admit(prompt: str, requested: int) -> tuple[List[int], int]:
    ids = await asyncio.to_thread(_ENGINE.tokenize, prompt)
    try:
        return ids, _ENGINE.admit(ids, requested)
    except PromptError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _usage(res: Dict[str, Any]) -> Dict[str, int]:
    return {"prompt_tokens": res["prompt_tokens"], "completion_tokens": res["completion_tokens"],
            "total_tokens": res["prompt_tokens"] + res["completion_tokens"]}


async def _sse(item: _BatchItem, corr: str, model: str, chat: bool) -> AsyncIterator[str]:
    """Stream one SSE chunk per generated token, then finish + usage-only chunk."""
    created = int(time.time())
    obj = "chat.completion.chunk" if chat else "text_completion"
    cid = f"{'chatcmpl' if chat else 'cmpl'}-{corr}"

    def chunk(text: Optional[str], finish: Optional[str], first: bool = False) -> str:
        if chat:
            delta: Dict[str, Any] = {} if text is None else {"content": text}
            if first:
                delta["role"] = "assistant"
            choice = {"index": 0, "delta": delta, "finish_reason": finish}
        else:
            choice = {"index": 0, "text": text or "", "finish_reason": finish, "logprobs": None}
        return f"data: {json.dumps({'id': cid, 'object': obj, 'created': created, 'model': model, 'choices': [choice]})}\n\n"

    assert item.stream is not None
    ids: List[int] = []
    emitted = ""
    first = True
    while True:
        kind, val = await item.stream.get()
        if kind == "tok":
            ids.append(val)
            text = _ENGINE.decode(ids)
            if text.endswith("\ufffd"):
                continue  # incomplete multi-byte char; wait for the next token
            delta, emitted = text[len(emitted):], text
            yield chunk(delta, None, first)
            first = False
        elif kind == "done":
            yield chunk(None if chat else "", val["finish_reason"], first)
            usage = {"id": cid, "object": obj, "created": created, "model": model,
                     "choices": [], "usage": _usage(val)}
            yield f"data: {json.dumps(usage)}\n\n"
            yield "data: [DONE]\n\n"
            return
        else:  # error after headers were sent
            yield f"data: {json.dumps({'error': {'message': val, 'type': 'server_error'}})}\n\n"
            yield "data: [DONE]\n\n"
            return


async def _run(prompt: str, requested: int, ignore_eos: bool, stream: bool,
               corr: str, model: str, chat: bool) -> Any:
    if not _ENGINE.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    ids, max_new = await _admit(prompt, requested)
    try:
        item = await _ENGINE.submit(ids, max_new, corr, ignore_eos, stream)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if stream:
        return StreamingResponse(_sse(item, corr, model, chat), media_type="text/event-stream")
    try:
        res = await item.future
    except Exception as exc:  # noqa: BLE001
        _log(logging.ERROR, "generation failed", correlation_id=corr, error=str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    base = {"id": f"{'chatcmpl' if chat else 'cmpl'}-{corr}", "created": int(time.time()),
            "model": model, "usage": _usage(res)}
    if chat:
        return {**base, "object": "chat.completion", "choices": [{
            "index": 0, "message": {"role": "assistant", "content": res["text"]},
            "finish_reason": res["finish_reason"]}]}
    return {**base, "object": "text_completion", "choices": [{
        "index": 0, "text": res["text"], "finish_reason": res["finish_reason"], "logprobs": None}]}


@app.post("/v1/completions")
async def completions(req: CompletionRequest, request: Request) -> Any:
    corr = getattr(request.state, "correlation_id", uuid.uuid4().hex)
    return await _run(req.prompt, _max_tokens(req.max_tokens), req.ignore_eos, req.stream,
                      corr, req.model or _SETTINGS.served_model_name, chat=False)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request) -> Any:
    corr = getattr(request.state, "correlation_id", uuid.uuid4().hex)
    if not _ENGINE.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    prompt = _ENGINE.apply_chat_template([{"role": m.role, "content": m.content} for m in req.messages])
    return await _run(prompt, _max_tokens(req.max_completion_tokens, req.max_tokens), req.ignore_eos,
                      req.stream, corr, req.model or _SETTINGS.served_model_name, chat=True)


if __name__ == "__main__":
    uvicorn.run(app, host=_SETTINGS.host, port=_SETTINGS.port, log_config=None,
                backlog=4096, timeout_keep_alive=75)
