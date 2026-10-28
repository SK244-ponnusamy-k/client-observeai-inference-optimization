"""
File: router.py
Purpose: Data-parallel front-end for the Qwen3.5 NxDI shim on one trn2.48xlarge.
Owner: genai-platform@shellkode
Deps: fastapi, uvicorn, httpx

A TP=4 replica at LNC=2 occupies one Trainium2 chip (4 logical NeuronCores).
trn2.48xlarge has 16 chips = 64 logical cores, so this launches QWEN35_DP_REPLICAS
copies of server.py, each pinned to its own core slice via NEURON_RT_VISIBLE_CORES
(the same pattern AWS uses for 16 x TP4 vLLM-Neuron servers on one instance),
and proxies the OpenAI API on :8000 with least-outstanding-requests dispatch.

All replicas load the SAME compiled artifact (it is core-slice agnostic).

Env:
  QWEN35_DP_REPLICAS        number of workers (default 16)
  QWEN35_TP_DEGREE          cores per worker (default 4)
  QWEN35_DP_CORE_OFFSET     first logical core (default 0)
  QWEN35_DP_BASE_PORT       first worker port (default 8100, bound to 127.0.0.1)
  QWEN35_DP_START_PARALLEL  workers loading concurrently after worker 0 (default 4)
  QWEN35_WORKER_THREADS     OMP/MKL threads per worker (default 8)
  QWEN35_PORT               router port (default 8000)

Startup: worker 0 loads alone first (populates any shared cache / fails fast on
a bad artifact), then the rest start QWEN35_DP_START_PARALLEL at a time to bound
host RAM. /health returns 200 only after every worker has become ready, so the
benchmark never runs against a partially loaded instance. If any worker process
dies the router exits non-zero and Kubernetes restarts the container (a
silently degraded instance would skew benchmark results).
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='{"ts":"%(asctime)s","level":"%(levelname)s","service":"qwen35-router","msg":"%(message)s"}')
log = logging.getLogger("qwen35_router")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DP = int(os.environ.get("QWEN35_DP_REPLICAS", "16"))
CORES = int(os.environ.get("QWEN35_TP_DEGREE", "4"))
CORE_OFFSET = int(os.environ.get("QWEN35_DP_CORE_OFFSET", "0"))
BASE_PORT = int(os.environ.get("QWEN35_DP_BASE_PORT", "8100"))
# Per-worker TP collective bootstrap port (distinct from worker HTTP ports).
COMM_BASE_PORT = int(os.environ.get("QWEN35_DP_COMM_BASE_PORT", "62000"))
START_PARALLEL = max(1, int(os.environ.get("QWEN35_DP_START_PARALLEL", "4")))
THREADS = os.environ.get("QWEN35_WORKER_THREADS", "8")
PORT = int(os.environ.get("QWEN35_PORT", "8000"))
SERVED = os.environ.get("QWEN35_SERVED_MODEL_NAME", "Qwen3.5-9B")


@dataclass
class Worker:
    idx: int
    port: int
    cores: str
    proc: Optional[subprocess.Popen] = None
    ready: bool = False
    inflight: int = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


WORKERS: List[Worker] = [
    Worker(i, BASE_PORT + i, f"{CORE_OFFSET + i * CORES}-{CORE_OFFSET + (i + 1) * CORES - 1}")
    for i in range(DP)
]
_rr = itertools.count()
_all_started = False
_client: Optional[httpx.AsyncClient] = None


def _spawn(w: Worker) -> None:
    env = os.environ.copy()
    env.pop("NEURON_RT_NUM_CORES", None)  # VISIBLE_CORES pins the exact slice
    env.update({
        "NEURON_RT_VISIBLE_CORES": w.cores,
        # Each worker is its own TP=4 collective group. On one host the groups
        # MUST use distinct bootstrap comm ids or their rank-0 "root" ports
        # collide and ranks hang ("awaiting root parameters", "root received
        # from 2 out of 4 ranks", RX timeout). Give every worker a unique id/port.
        "NEURON_RT_ROOT_COMM_ID": f"127.0.0.1:{COMM_BASE_PORT + w.idx}",
        "QWEN35_HOST": "127.0.0.1",
        "QWEN35_PORT": str(w.port),
        "QWEN35_WORKER_ID": str(w.idx),
        "OMP_NUM_THREADS": THREADS,
        "MKL_NUM_THREADS": THREADS,
        "TOKENIZERS_PARALLELISM": "false",
    })
    log.info(f"starting worker {w.idx} cores={w.cores} port={w.port}")
    w.proc = subprocess.Popen([sys.executable, "server.py"], cwd=APP_DIR, env=env)  # nosec B603


def _die(reason: str) -> None:
    log.error(f"FATAL: {reason} - stopping all workers")
    for w in WORKERS:
        if w.proc is not None and w.proc.poll() is None:
            w.proc.terminate()
    time.sleep(5)
    for w in WORKERS:
        if w.proc is not None and w.proc.poll() is None:
            w.proc.kill()
    os._exit(1)


async def _probe(w: Worker) -> bool:
    assert _client is not None
    try:
        r = await _client.get(f"{w.url}/health", timeout=5.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


async def _start_and_wait(w: Worker) -> None:
    _spawn(w)
    t0 = time.monotonic()
    while True:
        if w.proc is not None and w.proc.poll() is not None:
            _die(f"worker {w.idx} exited during load (rc={w.proc.returncode})")
        if await _probe(w):
            w.ready = True
            log.info(f"worker {w.idx} ready after {time.monotonic() - t0:.0f}s")
            return
        await asyncio.sleep(5)


async def _startup() -> None:
    global _all_started
    await _start_and_wait(WORKERS[0])
    sem = asyncio.Semaphore(START_PARALLEL)

    async def one(w: Worker) -> None:
        async with sem:
            await _start_and_wait(w)

    await asyncio.gather(*(one(w) for w in WORKERS[1:]))
    _all_started = True
    log.info(f"all {DP} workers ready")


async def _monitor() -> None:
    while True:
        await asyncio.sleep(5)
        for w in WORKERS:
            if w.proc is not None and w.proc.poll() is not None:
                _die(f"worker {w.idx} exited (rc={w.proc.returncode})")
        if _all_started:
            for w in WORKERS:
                ok = await _probe(w)
                if ok != w.ready:
                    log.warning(f"worker {w.idx} ready={ok}")
                w.ready = ok


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    limits = httpx.Limits(max_connections=4096, max_keepalive_connections=1024)
    _client = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(None, connect=10.0))
    tasks = [asyncio.create_task(_startup()), asyncio.create_task(_monitor())]
    yield
    for t in tasks:
        t.cancel()
    for w in WORKERS:
        if w.proc is not None and w.proc.poll() is None:
            w.proc.send_signal(signal.SIGTERM)
    await _client.aclose()


app = FastAPI(title="Qwen3.5 NxDI DP router", lifespan=lifespan)


def _pick() -> Optional[Worker]:
    ready = [w for w in WORKERS if w.ready]
    if not ready:
        return None
    low = min(w.inflight for w in ready)
    cands = [w for w in ready if w.inflight == low]
    return cands[next(_rr) % len(cands)]


async def _proxy(request: Request, path: str) -> Response:
    w = _pick()
    if w is None:
        return JSONResponse(status_code=503, content={"detail": "no ready workers"})
    assert _client is not None
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() in ("content-type", "x-correlation-id")}
    w.inflight += 1
    released = False

    def release() -> None:
        nonlocal released
        if not released:
            released = True
            w.inflight -= 1

    try:
        upstream = await _client.send(
            _client.build_request("POST", f"{w.url}{path}", content=body, headers=headers), stream=True
        )
    except httpx.HTTPError as exc:
        release()
        return JSONResponse(status_code=503, content={"detail": f"worker {w.idx}: {exc}"})

    resp_headers = {k: v for k, v in upstream.headers.items() if k.lower() in ("x-correlation-id",)}
    if upstream.headers.get("content-type", "").startswith("text/event-stream"):
        async def gen():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                release()

        return StreamingResponse(gen(), status_code=upstream.status_code,
                                 media_type="text/event-stream", headers=resp_headers)
    try:
        content = await upstream.aread()
    finally:
        await upstream.aclose()
        release()
    return Response(content=content, status_code=upstream.status_code,
                    media_type=upstream.headers.get("content-type", "application/json"), headers=resp_headers)


@app.post("/v1/completions")
async def completions(request: Request) -> Response:
    return await _proxy(request, "/v1/completions")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    return await _proxy(request, "/v1/chat/completions")


@app.get("/health")
async def health() -> JSONResponse:
    ready = sum(w.ready for w in WORKERS)
    ok = _all_started and ready > 0
    return JSONResponse(status_code=200 if ok else 503,
                        content={"status": "ok" if ok else "loading", "ready": ready, "total": DP})


@app.get("/v1/models")
async def models() -> Dict:
    return {"object": "list", "data": [{"id": SERVED, "object": "model",
                                        "owned_by": "observeai-inference-optimization"}]}


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Sum the label-free worker metrics; kv_cache_usage_perc is averaged."""
    assert _client is not None
    ready = [w for w in WORKERS if w.ready]

    async def fetch(w: Worker) -> str:
        try:
            return (await _client.get(f"{w.url}/metrics", timeout=3.0)).text
        except httpx.HTTPError:
            return ""

    totals: Dict[str, float] = {}
    for text in await asyncio.gather(*(fetch(w) for w in ready)):
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2 and not line.startswith("#"):
                try:
                    totals[parts[0]] = totals.get(parts[0], 0.0) + float(parts[1])
                except ValueError:
                    pass
    if ready and "vllm:kv_cache_usage_perc" in totals:
        totals["vllm:kv_cache_usage_perc"] /= len(ready)
    totals["qwen35:workers_ready"] = len(ready)
    totals["qwen35:workers_total"] = DP
    totals["qwen35:router_inflight"] = sum(w.inflight for w in WORKERS)
    return PlainTextResponse("".join(f"{k} {v:g}\n" for k, v in totals.items()))


if __name__ == "__main__":
    log.info(json.dumps({"dp": DP, "cores_per_replica": CORES, "base_port": BASE_PORT}))
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_config=None,  # nosec B104
                backlog=4096, timeout_keep_alive=75)
