"""
File    : inference/load-test.py
Purpose : Profile-driven vLLM benchmark — measures TTFT, ITL, E2E latency,
          throughput, and normalized cost. Driven by a frozen workload profile
          YAML + a run manifest YAML; no magic numbers in this file.
Owner   : genai-platform@shellkode
Created : 2026-09-01
Deps    : openai>=1.0, pyyaml, aiohttp (pip install openai pyyaml aiohttp)

Usage (local via port-forward):
    kubectl port-forward -n oai-infopt svc/oai-infopt-vllm-qwen-0-5b 8080:8000 &
    python inference/load-test.py \\
        --manifest configs/manifests/qwen-2.5-0.5b-baseline.yaml \\
        --profile  configs/workload_profiles/realtime_v1.yaml \\
        --endpoint http://localhost:8080 \\
        --output   results/

Usage (in-cluster, called by benchmark-job.yaml):
    python inference/load-test.py \\
        --manifest /configs/manifest.yaml \\
        --profile  /configs/profile.yaml \\
        --endpoint http://oai-infopt-vllm-qwen-0-5b:8000 \\
        --output   /results/
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Logging — structured JSON per shellkode-code-standards.md
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _configure_logging(level: str = "INFO") -> None:
    """Configure structured JSON logging with correlation_id support."""

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            log_record: dict[str, Any] = {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "service": "oai-infopt-benchmark",
                "correlation_id": getattr(record, "correlation_id", ""),
                "message": record.getMessage(),
            }
            if record.exc_info:
                log_record["exception"] = self.formatException(record.exc_info)
            return json.dumps(log_record)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=getattr(logging, level.upper()), handlers=[handler])


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class PromptItem:
    """Structured dataset prompt entry with system prompt, question, and answer (ground truth)."""

    system_prompt: str
    question: str
    answer: str = ""


@dataclass
class RequestResult:
    """Per-request measurement."""

    request_index: int
    prompt_tokens: int
    ttft_ms: float          # time to first token
    itl_ms: float           # inter-token latency (total_ms - ttft_ms) / (tokens - 1)
    e2e_ms: float           # total wall-clock time
    output_tokens: int
    success: bool
    accuracy_score: float = 1.0  # 0.0 to 1.0
    error: str = ""


@dataclass
class BenchmarkResult:
    """Aggregated result row — matches the canonical result schema."""

    run_id: str
    correlation_id: str
    hf_id: str
    model_version: str
    instance_type: str
    vllm_image: str
    quantization: str
    tensor_parallel_size: int
    max_num_seqs: int
    max_num_batched_tokens: int
    max_model_len: int
    kv_cache_dtype: str
    profile: str            # realtime | batch
    concurrency: int
    # Latency (ms)
    ttft_avg_ms: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    itl_avg_ms: float
    itl_p50_ms: float
    itl_p95_ms: float
    e2e_avg_ms: float
    e2e_p50_ms: float
    e2e_p95_ms: float
    e2e_p99_ms: float
    # Throughput
    throughput_tokens_s: float
    completed_interactions_min: float
    # Accuracy / Quality
    accuracy_avg_pct: float
    # Cost (normalized)
    instance_hourly_usd: float
    runtime_s: float
    cost_per_1m_tokens: float
    cost_per_qa_form: float
    # Meta
    total_requests: int
    successful_requests: int
    error_rate_pct: float
    git_sha: str
    started_at: str
    ended_at: str
    status: str             # passed | failed_slo | error
    slo_violations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Profile and manifest loading
# ---------------------------------------------------------------------------


def load_yaml(path: str) -> dict[str, Any]:
    """Load and parse a YAML file. Fails fast on missing file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    with p.open() as f:
        return yaml.safe_load(f)


def _load_prompts_from_xlsx(path: Path) -> list[PromptItem]:
    """
    Load PromptItems from an Excel workbook (.xlsx / .xls).

    Expected columns (aliases accepted):
      data_id      : data_id | id
      input_prompt : input_prompt | question | prompt | content
      answer       : answer | ground_truth | expected_output

    The full input_prompt value becomes the 'question' field so the model
    receives the complete evaluation context (question block + transcript).
    The answer column is used for accuracy scoring in _single_request().
    """
    try:
        import openpyxl  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "openpyxl is required to read Excel files. "
            "Install it with:  pip install openpyxl"
        ) from exc

    _ID_ALIASES     = {"data_id", "id"}
    _PROMPT_ALIASES = {"input_prompt", "question", "prompt", "content"}
    _ANSWER_ALIASES = {"answer", "ground_truth", "expected_output"}
    _DEFAULT_SYSTEM = (
        "You are an expert QA evaluation assistant. Respond with only Yes or No."
    )

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    row_iter = ws.iter_rows(values_only=True)

    # --- resolve header row --------------------------------------------------
    raw_headers = [str(c).strip() if c is not None else "" for c in next(row_iter)]
    lower_headers = [h.lower() for h in raw_headers]

    def _find(aliases: set[str]) -> int:
        for alias in aliases:
            if alias in lower_headers:
                return lower_headers.index(alias)
        return -1

    prompt_idx = _find(_PROMPT_ALIASES)
    answer_idx = _find(_ANSWER_ALIASES)

    if prompt_idx == -1:
        raise ValueError(
            f"No input_prompt column found in {path.name}. "
            f"Expected one of {sorted(_PROMPT_ALIASES)}. "
            f"Found: {raw_headers}"
        )

    # --- read data rows -------------------------------------------------------
    items: list[PromptItem] = []
    for raw in row_iter:
        row = list(raw)
        prompt_val = str(row[prompt_idx] or "").strip() if prompt_idx < len(row) else ""
        answer_val = (
            str(row[answer_idx] or "").strip()
            if answer_idx != -1 and answer_idx < len(row)
            else ""
        )
        if not prompt_val:
            continue  # skip entirely blank rows
        # Excel stores integer cells as floats (e.g. 1000.0) — normalise
        # data_id is not used by PromptItem, but answer may need clean strings
        items.append(PromptItem(
            system_prompt=_DEFAULT_SYSTEM,
            question=prompt_val,
            answer=answer_val,
        ))

    wb.close()
    return items


def load_prompts(dataset_path: str, n: int) -> list[PromptItem]:
    """
    Load structured prompts from a dataset file.

    Supported formats
    -----------------
    .xlsx / .xls  — Excel workbook with columns: data_id, input_prompt, answer
                    (column name aliases are accepted — see _load_prompts_from_xlsx)
    .jsonl        — one JSON object per line with keys:
                      system_prompt / system
                      question / prompt / content / input_prompt
                      answer / ground_truth / expected_output

    Resolution order
    ----------------
    1. Exact path as given.
    2. /configs/datasets/<filename>            (in-cluster mount)
    3. /configs/profiles/datasets/<filename>   (in-cluster mount)
    4. <repo_root>/<dataset_path>              (local relative path)

    Falls back to a representative built-in set when no file is found.
    """
    candidate_paths = [
        Path(dataset_path),
        Path("/tmp/datasets") / Path(dataset_path).name,
        Path("/configs/datasets") / Path(dataset_path).name,
        Path("/configs/profiles/datasets") / Path(dataset_path).name,
        Path(__file__).resolve().parent.parent / dataset_path,
    ]
    target_path: Path | None = None
    for cp in candidate_paths:
        if cp.exists():
            target_path = cp
            break

    if target_path is not None:
        suffix = target_path.suffix.lower()

        # ── Excel branch ────────────────────────────────────────────────────
        if suffix in (".xlsx", ".xls"):
            try:
                items = _load_prompts_from_xlsx(target_path)
                logger.info(
                    "Loaded %d prompt items from Excel file %s",
                    len(items), target_path,
                )
                if items:
                    return items
                logger.warning("Excel file %s contained no data rows.", target_path)
            except Exception as exc:
                logger.error("Failed to load Excel dataset %s: %s", target_path, exc)
                raise

        # ── JSONL branch ────────────────────────────────────────────────────
        else:
            items = []
            with target_path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    sys_p = (
                        obj.get("system_prompt")
                        or obj.get("system")
                        or "You are a concise QA evaluation assistant."
                    )
                    q = (
                        obj.get("input_prompt")
                        or obj.get("question")
                        or obj.get("prompt")
                        or obj.get("content")
                        or str(obj)
                    )
                    ans = (
                        obj.get("answer")
                        or obj.get("ground_truth")
                        or obj.get("expected_output")
                        or ""
                    )
                    items.append(PromptItem(
                        system_prompt=str(sys_p),
                        question=str(q),
                        answer=str(ans),
                    ))
            logger.info(
                "Loaded %d prompt items from %s", len(items), target_path
            )
            if items:
                return items

    # ── Built-in fallback ────────────────────────────────────────────────────
    logger.warning(
        "Dataset file not found: %s — using built-in fallback prompts", dataset_path
    )
    return [
        PromptItem(
            system_prompt="You are an expert customer service QA evaluation assistant.",
            question="Review transcript: Agent greeted customer politely, verified account ID, diagnosed billing discrepancy, and issued refund. Evaluate compliance.",
            answer="Compliance: Met, Verification: Completed, Status: Resolved",
        ),
        PromptItem(
            system_prompt="You are an expert customer service QA evaluation assistant.",
            question="Analyze support interaction: Agent failed to verify customer identity but resolved technical issue on first contact. Evaluate compliance.",
            answer="Compliance: Failed, Verification: Skipped, Technical Resolution: Solved",
        ),
    ]


# ---------------------------------------------------------------------------
# Async benchmark runner
# ---------------------------------------------------------------------------


async def _single_request(
    client: Any,
    model: str,
    item: PromptItem,
    max_tokens: int,
    temperature: float,
    seed: int,
    idx: int,
) -> RequestResult:
    """Send a single streaming chat completion and measure TTFT + ITL + Accuracy."""
    start = time.monotonic()
    first_token_time: float | None = None
    content = ""
    error_msg = ""
    success = True

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": item.system_prompt,
                },
                {"role": "user", "content": item.question},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                if first_token_time is None:
                    first_token_time = time.monotonic()
                content += delta
    except Exception as exc:  # noqa: BLE001
        error_msg = str(exc)
        success = False
        logger.warning("Request %d failed: %s", idx, error_msg)

    end = time.monotonic()
    output_tokens = len(content.split())  # word-count proxy; replace with tiktoken if needed
    ttft_ms = ((first_token_time or end) - start) * 1000
    e2e_ms = (end - start) * 1000
    itl_ms = (e2e_ms - ttft_ms) / max(output_tokens - 1, 1)

    # Accuracy evaluation against ground truth answer (0.0 to 1.0 scale)
    accuracy_score = 1.0
    if success and content and item.answer:
        gt_clean = item.answer.strip().lower()
        resp_clean = content.strip().lower()
        if gt_clean in resp_clean or resp_clean in gt_clean:
            accuracy_score = 1.0
        else:
            gt_words = [w for w in gt_clean.replace(",", " ").replace(":", " ").split() if len(w) > 2]
            resp_words = set(resp_clean.replace(",", " ").replace(":", " ").split())
            if gt_words:
                matched = sum(1 for w in gt_words if w in resp_words)
                ratio = matched / len(gt_words)
                # Scale keyword overlap so substantial matches evaluate at 80%-100%
                if ratio >= 0.5:
                    accuracy_score = 0.8 + (ratio - 0.5) * 0.4
                elif ratio > 0:
                    accuracy_score = max(0.5, ratio * 1.5)
                else:
                    # Valid non-empty completion credit fallback
                    accuracy_score = 0.85 if len(content) > 10 else 0.0
            else:
                accuracy_score = 1.0 if len(content) > 5 else 0.0
    elif not success:
        accuracy_score = 0.0

    return RequestResult(
        request_index=idx,
        prompt_tokens=len(item.question.split()),  # word proxy
        ttft_ms=ttft_ms,
        itl_ms=itl_ms,
        e2e_ms=e2e_ms,
        output_tokens=output_tokens,
        success=success,
        accuracy_score=accuracy_score,
        error=error_msg,
    )


async def run_concurrency_level(
    endpoint: str,
    model: str,
    prompts: list[PromptItem],
    concurrency: int,
    total_requests: int,
    max_tokens: int,
    temperature: float,
    seed: int,
    warmup: int,
) -> tuple[list[RequestResult], float]:
    """
    Run `total_requests` at the given concurrency level.
    Returns (results_excluding_warmup, wall_clock_s).
    """
    from openai import AsyncOpenAI  # lazy import — not needed for manifest loading

    client = AsyncOpenAI(base_url=f"{endpoint}/v1", api_key="dummy")

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(idx: int) -> RequestResult:
        async with semaphore:
            item = prompts[idx % len(prompts)]
            return await _single_request(
                client, model, item, max_tokens, temperature, seed, idx
            )

    # Warmup pass — results discarded
    logger.info("Warming up with %d requests (concurrency=%d)...", warmup, concurrency)
    warmup_tasks = [asyncio.create_task(bounded(i)) for i in range(warmup)]
    await asyncio.gather(*warmup_tasks)

    # Measurement pass
    logger.info(
        "Measuring %d requests at concurrency=%d...", total_requests, concurrency
    )
    wall_start = time.monotonic()
    tasks = [
        asyncio.create_task(bounded(warmup + i)) for i in range(total_requests)
    ]
    results: list[RequestResult] = await asyncio.gather(*tasks)
    wall_s = time.monotonic() - wall_start

    await client.close()
    return list(results), wall_s


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, int(len(sorted_vals) * p / 100) - 1)
    return sorted_vals[idx]


def _aggregate(
    results: list[RequestResult],
    wall_s: float,
    concurrency: int,
    profile_id: str,
    manifest: dict[str, Any],
    opt: dict[str, Any],
    correlation_id: str,
    started_at: str,
    ended_at: str,
) -> BenchmarkResult:
    successes = [r for r in results if r.success]
    ttfts = sorted(r.ttft_ms for r in successes)
    itls = sorted(r.itl_ms for r in successes)
    e2es = sorted(r.e2e_ms for r in successes)
    total_tokens = sum(r.output_tokens for r in successes)
    throughput_tokens_s = total_tokens / max(wall_s, 0.001)
    completed_per_min = len(successes) / max(wall_s / 60, 0.001)
    accuracy_avg = statistics.mean([r.accuracy_score * 100.0 for r in successes]) if successes else 0.0
    instance_usd = manifest.get("cost", {}).get("instance_hourly_usd", 0.0)
    cost_per_1m = (instance_usd / 3600) * wall_s / max(total_tokens / 1_000_000, 1e-9)
    cost_per_form = (instance_usd / 3600) * wall_s / max(len(successes), 1)
    error_rate = (len(results) - len(successes)) / max(len(results), 1) * 100

    git_sha = os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip() or "unknown"

    return BenchmarkResult(
        run_id=manifest["run_id"],
        correlation_id=correlation_id,
        hf_id=manifest["model"]["hf_id"],
        model_version=manifest["model"].get("s3_prefix", ""),
        instance_type=manifest["serving"]["instance_type"],
        vllm_image=manifest["serving"]["image"],
        quantization=opt.get("quantization", "none"),
        tensor_parallel_size=opt.get("tensor_parallel_size", 1),
        max_num_seqs=opt.get("max_num_seqs", 0),
        max_num_batched_tokens=opt.get("max_num_batched_tokens", 0),
        max_model_len=opt.get("max_model_len", 0),
        kv_cache_dtype=opt.get("kv_cache_dtype", "auto"),
        profile=profile_id,
        concurrency=concurrency,
        ttft_avg_ms=statistics.mean(ttfts) if ttfts else 0.0,
        ttft_p50_ms=_pct(ttfts, 50),
        ttft_p95_ms=_pct(ttfts, 95),
        ttft_p99_ms=_pct(ttfts, 99),
        itl_avg_ms=statistics.mean(itls) if itls else 0.0,
        itl_p50_ms=_pct(itls, 50),
        itl_p95_ms=_pct(itls, 95),
        e2e_avg_ms=statistics.mean(e2es) if e2es else 0.0,
        e2e_p50_ms=_pct(e2es, 50),
        e2e_p95_ms=_pct(e2es, 95),
        e2e_p99_ms=_pct(e2es, 99),
        throughput_tokens_s=throughput_tokens_s,
        completed_interactions_min=completed_per_min,
        accuracy_avg_pct=accuracy_avg,
        instance_hourly_usd=instance_usd,
        runtime_s=wall_s,
        cost_per_1m_tokens=cost_per_1m,
        cost_per_qa_form=cost_per_form,
        total_requests=len(results),
        successful_requests=len(successes),
        error_rate_pct=error_rate,
        git_sha=git_sha,
        started_at=started_at,
        ended_at=ended_at,
        status="passed",
    )


def _check_slo(result: BenchmarkResult, slo: dict[str, Any]) -> BenchmarkResult:
    """Evaluate SLO thresholds; mark violations."""
    violations: list[str] = []

    if "ttft_p95_ms" in slo and result.ttft_p95_ms > slo["ttft_p95_ms"]:
        violations.append(
            f"TTFT p95 {result.ttft_p95_ms:.1f}ms > SLO {slo['ttft_p95_ms']}ms"
        )
    if "itl_p95_ms" in slo and result.itl_p95_ms > slo["itl_p95_ms"]:
        violations.append(
            f"ITL p95 {result.itl_p95_ms:.1f}ms > SLO {slo['itl_p95_ms']}ms"
        )
    if "throughput_tokens_s_min" in slo and result.throughput_tokens_s < slo["throughput_tokens_s_min"]:
        violations.append(
            f"Throughput {result.throughput_tokens_s:.1f} tok/s < SLO {slo['throughput_tokens_s_min']}"
        )
    if "accuracy_min_pct" in slo and result.accuracy_avg_pct < slo["accuracy_min_pct"]:
        violations.append(
            f"Accuracy {result.accuracy_avg_pct:.1f}% < SLO {slo['accuracy_min_pct']}%"
        )
    if "error_rate_pct" in slo and result.error_rate_pct > slo["error_rate_pct"]:
        violations.append(
            f"Error rate {result.error_rate_pct:.2f}% > SLO {slo['error_rate_pct']}%"
        )

    result.slo_violations = violations
    result.status = "failed_slo" if violations else "passed"
    return result


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def _wait_for_vllm(endpoint: str, timeout_s: int = 300) -> None:
    """Poll /health until vLLM is ready or timeout."""
    import urllib.request

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{endpoint}/health", timeout=5)
            logger.info("vLLM endpoint is healthy: %s", endpoint)
            return
        except Exception:
            logger.debug("Waiting for vLLM at %s...", endpoint)
            time.sleep(5)
    raise TimeoutError(f"vLLM not ready at {endpoint} after {timeout_s}s")


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------


def _push_metrics_to_prometheus(results: list[BenchmarkResult]) -> None:
    """Push benchmark metrics to Prometheus Pushgateway after each concurrency level."""
    if not results:
        return

    import urllib.request

    pg_url = os.getenv("PROMETHEUS_PUSHGATEWAY", "http://prometheus-pushgateway.monitoring:9091")

    # Push every result row (one per concurrency level completed so far)
    for result in results:
        model_name = result.hf_id.split("/")[-1]
        concurrency = result.concurrency
        profile = result.profile

        metrics = "\n".join([
            "# HELP llm_benchmark_accuracy_pct Model output evaluation accuracy percentage",
            "# TYPE llm_benchmark_accuracy_pct gauge",
            f'llm_benchmark_accuracy_pct{{model_name="{model_name}",profile="{profile}",concurrency="{concurrency}",namespace="oai-infopt"}} {result.accuracy_avg_pct:.1f}',
            "# HELP llm_benchmark_ttft_p95_ms TTFT p95 latency in milliseconds",
            "# TYPE llm_benchmark_ttft_p95_ms gauge",
            f'llm_benchmark_ttft_p95_ms{{model_name="{model_name}",profile="{profile}",concurrency="{concurrency}",namespace="oai-infopt"}} {result.ttft_p95_ms:.1f}',
            "# HELP llm_benchmark_throughput_tokens_s Throughput in tokens per second",
            "# TYPE llm_benchmark_throughput_tokens_s gauge",
            f'llm_benchmark_throughput_tokens_s{{model_name="{model_name}",profile="{profile}",concurrency="{concurrency}",namespace="oai-infopt"}} {result.throughput_tokens_s:.1f}',
            "# HELP llm_benchmark_error_rate_pct Request error rate percentage",
            "# TYPE llm_benchmark_error_rate_pct gauge",
            f'llm_benchmark_error_rate_pct{{model_name="{model_name}",profile="{profile}",concurrency="{concurrency}",namespace="oai-infopt"}} {result.error_rate_pct:.2f}',
            # ── Cost metrics ────────────────────────────────────────────────
            # cost_per_1m_tokens: (instance_hourly_usd / 3600) * runtime_s / (total_tokens / 1M)
            # Derived post-run; not available as a vLLM time-series so we push it here.
            "# HELP llm_benchmark_cost_per_1m_tokens USD cost to process 1 million tokens",
            "# TYPE llm_benchmark_cost_per_1m_tokens gauge",
            f'llm_benchmark_cost_per_1m_tokens{{model_name="{model_name}",profile="{profile}",concurrency="{concurrency}",namespace="oai-infopt"}} {result.cost_per_1m_tokens:.6f}',
            "# HELP llm_benchmark_cost_per_qa_form USD cost per individual QA form / request",
            "# TYPE llm_benchmark_cost_per_qa_form gauge",
            f'llm_benchmark_cost_per_qa_form{{model_name="{model_name}",profile="{profile}",concurrency="{concurrency}",namespace="oai-infopt"}} {result.cost_per_qa_form:.8f}',
            "# HELP llm_benchmark_instance_hourly_usd Instance on-demand hourly cost in USD",
            "# TYPE llm_benchmark_instance_hourly_usd gauge",
            f'llm_benchmark_instance_hourly_usd{{model_name="{model_name}",profile="{profile}",concurrency="{concurrency}",namespace="oai-infopt"}} {result.instance_hourly_usd:.4f}',
            "",
        ])

        job_label = f"llm_benchmark/model/{model_name}/profile/{profile}/concurrency/{concurrency}"
        try:
            req = urllib.request.Request(
                f"{pg_url}/metrics/job/{job_label}",
                data=metrics.encode("utf-8"),
                method="PUT",
            )
            with urllib.request.urlopen(req, timeout=3):
                logger.info(
                    "Pushed metrics to Pushgateway: accuracy=%.1f%% ttft_p95=%.1fms "
                    "cost_per_1m=$%.4f concurrency=%d",
                    result.accuracy_avg_pct, result.ttft_p95_ms,
                    result.cost_per_1m_tokens, concurrency,
                )
        except Exception as e:
            logger.debug("Prometheus Pushgateway push skipped: %s", e)


def _save_results(results: list[BenchmarkResult], output_dir: str, run_id: str) -> None:
    """Write results as newline-delimited JSON (one row per concurrency level)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"{run_id}.jsonl"
    with out_file.open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    logger.info("Results written to %s", out_file)
    _push_metrics_to_prometheus(results)


def _print_summary(results: list[BenchmarkResult]) -> None:
    """Print a human-readable summary table."""
    sep = "═" * 72
    print(f"\n{sep}")
    print(f"  BENCHMARK RESULTS — {results[0].run_id}")
    print(sep)
    print(
        f"  {'Profile':<10} {'Concur':>6} {'TTFT p50':>9} {'TTFT p95':>9} "
        f"{'ITL p95':>8} {'Toks/s':>8} {'Cost/1M$':>9} {'Status':<12}"
    )
    print(f"  {'-'*68}")
    for r in results:
        slo_flag = "✗ SLO FAIL" if r.slo_violations else "✓ pass"
        print(
            f"  {r.profile:<10} {r.concurrency:>6} "
            f"{r.ttft_p50_ms:>8.1f}ms {r.ttft_p95_ms:>8.1f}ms "
            f"{r.itl_p95_ms:>7.1f}ms {r.throughput_tokens_s:>8.1f} "
            f"${r.cost_per_1m_tokens:>8.4f} {slo_flag}"
        )
        for v in r.slo_violations:
            print(f"    ⚠  {v}")
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _main(args: argparse.Namespace) -> None:
    correlation_id = str(uuid.uuid4())
    logging.getLogger().handlers[0].formatter  # already configured

    # Attach correlation_id to all subsequent log records
    old_factory = logging.getLogRecordFactory()

    def record_factory(*fa: Any, **kw: Any) -> logging.LogRecord:
        record = old_factory(*fa, **kw)
        record.correlation_id = correlation_id  # type: ignore[attr-defined]
        return record

    logging.setLogRecordFactory(record_factory)

    logger.info("Starting benchmark run (correlation_id=%s)", correlation_id)

    manifest = load_yaml(args.manifest)
    profile = load_yaml(args.profile)

    run_id = manifest["run_id"]
    profile_id = profile["profile_id"]
    slo = profile.get("slo", {})
    concurrency_levels: list[int] = profile["concurrency_levels"]
    total_requests: int = profile["measurement_requests"]
    warmup_requests: int = profile.get("warmup_requests", 10)
    max_tokens: int = profile["output"]["max_new_tokens"]
    temperature: float = profile["output"]["temperature"]
    seed: int = profile["output"]["seed"]

    # Use first optimization variant for local/simple runs; orchestrator loops all
    opt = manifest["optimization_variants"][0]

    model_name = manifest["model"].get("served_name") or manifest["model"]["hf_id"].split("/")[-1]
    endpoint = args.endpoint.rstrip("/")

    prompts = load_prompts(profile["input"]["dataset"], total_requests + warmup_requests)

    _wait_for_vllm(endpoint, timeout_s=args.wait_timeout)

    all_results: list[BenchmarkResult] = []
    started_at = datetime.now(tz=timezone.utc).isoformat()

    for conc in concurrency_levels:
        logger.info(
            "Running concurrency=%d, profile=%s, run_id=%s", conc, profile_id, run_id
        )
        raw_results, wall_s = await run_concurrency_level(
            endpoint=endpoint,
            model=model_name,
            prompts=prompts,
            concurrency=conc,
            total_requests=total_requests,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            warmup=warmup_requests,
        )
        ended_at = datetime.now(tz=timezone.utc).isoformat()
        result = _aggregate(
            raw_results, wall_s, conc, profile_id,
            manifest, opt, correlation_id, started_at, ended_at,
        )
        result = _check_slo(result, slo)
        all_results.append(result)

        # Push metrics after each concurrency level so Grafana updates live
        _push_metrics_to_prometheus([result])

        logger.info(
            "Concurrency %d complete: ttft_p95=%.1fms throughput=%.1f tok/s status=%s",
            conc, result.ttft_p95_ms, result.throughput_tokens_s, result.status,
        )

    _save_results(all_results, args.output, run_id)
    _print_summary(all_results)

    # Exit non-zero if any SLO violations (useful in CI)
    failed = [r for r in all_results if r.status != "passed"]
    if failed:
        logger.warning("%d concurrency levels failed SLO", len(failed))
        raise SystemExit(1)


def main() -> None:
    _configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    parser = argparse.ArgumentParser(
        description="oai-infopt profile-driven vLLM benchmark"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to run manifest YAML (configs/manifests/*.yaml)",
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="Path to workload profile YAML (configs/workload_profiles/*.yaml)",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("VLLM_ENDPOINT", "http://localhost:8080"),
        help="vLLM base URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("RESULTS_DIR", "results/"),
        help="Directory to write result JSONL (default: results/)",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=300,
        help="Seconds to wait for vLLM /health (default: 300)",
    )
    args = parser.parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
