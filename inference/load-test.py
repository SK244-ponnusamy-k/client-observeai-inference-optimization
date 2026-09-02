"""
File    : inference/load-test.py
Purpose : Profile-driven vLLM benchmark — measures TTFT, ITL, E2E latency,
          throughput, and normalized cost. Driven by a frozen workload profile
          YAML + a run manifest YAML; no magic numbers in this file.
Owner   : genai-platform@shellkode
Created : 2026-09-01
Deps    : openai>=1.0, pyyaml, aiohttp (pip install openai pyyaml aiohttp)

Usage (local via port-forward):
    kubectl port-forward -n oai-infopt svc/oai-infopt-vllm-qwen-0.5b 8080:8000 &
    python inference/load-test.py \\
        --manifest configs/manifests/qwen-2.5-0.5b-baseline.yaml \\
        --profile  configs/workload_profiles/realtime_v1.yaml \\
        --endpoint http://localhost:8080 \\
        --output   results/

Usage (in-cluster, called by benchmark-job.yaml):
    python inference/load-test.py \\
        --manifest /configs/manifest.yaml \\
        --profile  /configs/profile.yaml \\
        --endpoint http://oai-infopt-vllm-qwen-0.5b:8000 \\
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


def load_prompts(dataset_path: str, n: int) -> list[str]:
    """
    Load prompts from a JSONL dataset file (field: 'prompt' or 'content').
    Falls back to a small built-in set if the dataset file is absent.
    """
    p = Path(dataset_path)
    if p.exists():
        prompts: list[str] = []
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                prompts.append(obj.get("prompt") or obj.get("content") or str(obj))
        logger.info("Loaded %d prompts from %s", len(prompts), dataset_path)
        return prompts
    # Built-in fallback (representative QA-form prompts)
    logger.warning(
        "Dataset file not found: %s — using built-in fallback prompts", dataset_path
    )
    return [
        (
            "Review the following customer service call transcript and complete the QA form. "
            "Transcript: Agent greeted the customer, identified the issue as a billing discrepancy, "
            "escalated to billing team, confirmed resolution timeline of 3-5 business days, "
            "and closed the call politely. "
            "Complete: {greeting_score, issue_identification, resolution_offered, "
            "call_close_quality, overall_score}"
        ),
        (
            "Analyze this support interaction and fill in the evaluation form. "
            "The agent failed to verify the customer's identity before discussing account details, "
            "but resolved the technical issue within the first contact. "
            "Complete: {compliance_check, first_call_resolution, customer_satisfaction_score, "
            "coaching_notes}"
        ),
        (
            "QA evaluation required. The agent demonstrated strong product knowledge "
            "and empathy throughout the 8-minute interaction. Issue: refund request. "
            "Outcome: approved. Complete the standard QA scorecard fields."
        ),
        (
            "Score the following call. Agent did not follow the required script for "
            "data verification, skipped hold protocol, but achieved a positive outcome. "
            "Provide scores for: script_adherence, hold_protocol, outcome_quality, "
            "supervisor_review_needed"
        ),
    ]


# ---------------------------------------------------------------------------
# Async benchmark runner
# ---------------------------------------------------------------------------


async def _single_request(
    client: Any,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int,
    idx: int,
) -> RequestResult:
    """Send a single streaming chat completion and measure TTFT + ITL."""
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
                    "content": "You are a concise QA evaluation assistant.",
                },
                {"role": "user", "content": prompt},
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

    return RequestResult(
        request_index=idx,
        prompt_tokens=len(prompt.split()),  # word proxy
        ttft_ms=ttft_ms,
        itl_ms=itl_ms,
        e2e_ms=e2e_ms,
        output_tokens=output_tokens,
        success=success,
        error=error_msg,
    )


async def run_concurrency_level(
    endpoint: str,
    model: str,
    prompts: list[str],
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
            prompt = prompts[idx % len(prompts)]
            return await _single_request(
                client, model, prompt, max_tokens, temperature, seed, idx
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


def _save_results(results: list[BenchmarkResult], output_dir: str, run_id: str) -> None:
    """Write results as newline-delimited JSON (one row per concurrency level)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"{run_id}.jsonl"
    with out_file.open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    logger.info("Results written to %s", out_file)


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

    model_name = manifest["model"]["hf_id"].split("/")[-1]
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
