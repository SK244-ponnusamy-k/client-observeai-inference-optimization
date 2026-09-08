"""
File    : inference/load-test.py
Purpose : Model-agnostic vLLM benchmark ORCHESTRATOR.

          This does NOT hand-compute latency/throughput from response chunks.
          It shells out to `vllm bench serve` (the official, tokenizer-accurate
          load generator) for TTFT / TPOT / ITL / E2E percentiles + throughput,
          scrapes vLLM `/metrics` for engine state (KV-cache, queue depth) and
          an optional DCGM/Prometheus endpoint for GPU utilisation, looks up the
          instance $/hr, then derives cost and writes one canonical result row
          per (profile x concurrency) to JSONL + S3.

          Why bench serve instead of a custom async loop:
            - counts tokens with the model tokenizer (includes reasoning tokens
              for models like gpt-oss; the old word-split on delta.content
              silently dropped reasoning_content and collapsed throughput to
              ~0.6 tok/s)
            - correct TTFT (first token of ANY field), TPOT, ITL, percentiles
            - same tool OpenAI's gpt-oss recipe uses -> comparable numbers

Owner   : genai-platform@shellkode
Created : 2026-09-01 | Reworked: 2026-09-10 (bench-serve orchestrator)
Deps    : vllm (provides `vllm bench serve`), pyyaml, boto3
          Run from the vLLM DLC image so the CLI + tokenizers are present.

Usage (in-cluster, called by benchmark-job.yaml):
    python inference/load-test.py \\
        --manifest /configs/manifests/manifest.yaml \\
        --profile  /configs/profiles/profile.yaml \\
        --endpoint http://oai-infopt-vllm-gpt-oss-20b:8000 \\
        --output   /results/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess  # noqa: S404 - invoking the trusted `vllm` CLI, args are controlled
import tempfile
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging — structured JSON per shellkode-code-standards.md
# ---------------------------------------------------------------------------
def _configure_logging(level: str = "INFO") -> None:
    """Configure structured JSON logging with correlation_id support."""

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload: dict[str, Any] = {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "service": "oai-infopt-benchmark",
                "correlation_id": getattr(record, "correlation_id", ""),
                "message": record.getMessage(),
            }
            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)
            return json.dumps(payload)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=getattr(logging, level.upper()), handlers=[handler])


# ---------------------------------------------------------------------------
# Canonical result row (matches observeai-project-standards.md result schema)
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    """Aggregated result row — one per (profile, concurrency)."""

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

    # Latency (ms) — sourced from `vllm bench serve`
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    itl_p50_ms: float
    itl_p95_ms: float
    e2e_p50_ms: float
    e2e_p95_ms: float
    e2e_p99_ms: float

    # Throughput / tokens — sourced from `vllm bench serve` (tokenizer-counted)
    request_throughput_rps: float
    output_throughput_tokens_s: float
    total_token_throughput_s: float
    total_input_tokens: int
    total_output_tokens: int
    completed_interactions_min: float

    # Engine state — sourced from vLLM /metrics
    kv_cache_utilization_pct: float | None
    num_requests_waiting: float | None
    prefix_cache_hit_rate: float | None

    # GPU — sourced from DCGM/Prometheus (optional; None if not wired)
    gpu_utilization_pct: float | None
    gpu_mem_used_mib: float | None
    gpu_power_watts: float | None

    # Cost (derived; None when tokens are ~0 so we never emit $127k garbage)
    instance_hourly_usd: float
    runtime_s: float
    cost_per_1m_tokens: float | None
    cost_per_qa_form: float | None
    tokens_per_usd: float | None
    energy_per_1m_tokens_wh: float | None

    # Quality — NOT measured here. Perf harness != quality benchmark.
    # Accuracy comes from the separate lm-eval-harness stage (MMLU-Pro/GPQA/DROP/IFEval).
    accuracy_avg_pct: float | None

    # Meta
    total_requests: int
    successful_requests: int
    error_rate_pct: float
    reasoning_effort: str
    git_sha: str
    started_at: str
    ended_at: str
    status: str             # passed | failed_slo | error
    slo_violations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------
def load_yaml(path: str) -> dict[str, Any]:
    """Load and parse a YAML file. Fails fast on missing file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    with p.open() as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
def _wait_for_vllm(endpoint: str, timeout_s: int = 300) -> None:
    """Poll /health until vLLM is ready or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{endpoint}/health", timeout=5)  # noqa: S310
            logger.info("vLLM endpoint is healthy: %s", endpoint)
            return
        except Exception:  # noqa: BLE001
            time.sleep(5)
    raise TimeoutError(f"vLLM not ready at {endpoint} after {timeout_s}s")


# ---------------------------------------------------------------------------
# Prometheus text scraping (vLLM /metrics and DCGM)
# ---------------------------------------------------------------------------
def _fetch_text(url: str, timeout: int = 5) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return resp.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Scrape failed for %s: %s", url, exc)
        return None


def _sum_metric(text: str, metric_name: str) -> float | None:
    """Sum all samples of a Prometheus metric family (ignores labels)."""
    total: float | None = None
    pattern = re.compile(rf"^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+([0-9eE.+-]+)\s*$")
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = pattern.match(line)
        if m:
            try:
                total = (total or 0.0) + float(m.group(1))
            except ValueError:
                continue
    return total


def _scrape_vllm_metrics(endpoint: str) -> dict[str, float | None]:
    """Read vLLM /metrics: KV-cache usage, waiting queue, prefix-cache hit rate."""
    text = _fetch_text(f"{endpoint}/metrics")
    if not text:
        return {"kv_cache_utilization_pct": None, "num_requests_waiting": None,
                "prefix_cache_hit_rate": None, "generation_tokens_total": None}

    kv = _sum_metric(text, "vllm:gpu_cache_usage_perc")
    waiting = _sum_metric(text, "vllm:num_requests_waiting")
    gen_tokens = _sum_metric(text, "vllm:generation_tokens_total")
    hits = _sum_metric(text, "vllm:prefix_cache_hits_total")
    queries = _sum_metric(text, "vllm:prefix_cache_queries_total")
    hit_rate = (hits / queries) if (hits is not None and queries) else None

    return {
        # vLLM reports 0-1; normalise to percent
        "kv_cache_utilization_pct": (kv * 100.0) if kv is not None else None,
        "num_requests_waiting": waiting,
        "prefix_cache_hit_rate": hit_rate,
        "generation_tokens_total": gen_tokens,
    }


def _scrape_gpu_metrics(dcgm_url: str | None) -> dict[str, float | None]:
    """
    Best-effort GPU telemetry from a DCGM-exporter (or Prometheus federate) URL.
    Set DCGM_METRICS_URL to a scrape endpoint that exposes DCGM_FI_DEV_* for the
    GPU node under test. If unavailable, fields are None and GPU util should be
    joined from Amazon Managed Prometheus / Grafana post-hoc.
    """
    if not dcgm_url:
        return {"gpu_utilization_pct": None, "gpu_mem_used_mib": None,
                "gpu_power_watts": None}
    text = _fetch_text(dcgm_url)
    if not text:
        return {"gpu_utilization_pct": None, "gpu_mem_used_mib": None,
                "gpu_power_watts": None}
    return {
        "gpu_utilization_pct": _sum_metric(text, "DCGM_FI_DEV_GPU_UTIL"),
        "gpu_mem_used_mib": _sum_metric(text, "DCGM_FI_DEV_FB_USED"),
        "gpu_power_watts": _sum_metric(text, "DCGM_FI_DEV_POWER_USAGE"),
    }


# ---------------------------------------------------------------------------
# vLLM bench serve invocation + parsing
# ---------------------------------------------------------------------------
def _run_vllm_bench(
    *,
    base_url: str,
    served_model: str,
    tokenizer: str,
    concurrency: int,
    num_prompts: int,
    dataset_name: str,
    random_input_len: int,
    random_output_len: int,
    seed: int,
    ignore_eos: bool,
    extra_args: list[str],
) -> dict[str, Any]:
    """
    Run `vllm bench serve` and return the parsed JSON result dict.

    We rely on vLLM to count tokens with the tokenizer (reasoning-aware) and to
    compute TTFT/TPOT/ITL/E2E percentiles. No client-side token math here.
    """
    with tempfile.TemporaryDirectory() as tmp:
        result_file = Path(tmp) / "bench.json"
        cmd = [
            "vllm", "bench", "serve",
            "--backend", "openai-chat",
            "--base-url", base_url,
            "--endpoint", "/v1/chat/completions",
            "--model", served_model,
            "--tokenizer", tokenizer,
            "--dataset-name", dataset_name,
            "--max-concurrency", str(concurrency),
            "--num-prompts", str(num_prompts),
            "--seed", str(seed),
            "--percentile-metrics", "ttft,tpot,itl,e2el",
            "--metric-percentiles", "95,99",
            "--save-result",
            "--result-filename", str(result_file),
        ]
        if dataset_name == "random":
            cmd += ["--random-input-len", str(random_input_len),
                    "--random-output-len", str(random_output_len)]
        if ignore_eos:
            cmd += ["--ignore-eos"]
        cmd += extra_args

        logger.info("Running: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
        if proc.returncode != 0:
            logger.error("vllm bench serve failed (rc=%s): %s", proc.returncode, proc.stderr[-2000:])
            raise RuntimeError(f"vllm bench serve failed: rc={proc.returncode}")

        if not result_file.exists():
            raise RuntimeError("vllm bench serve produced no result file")
        with result_file.open() as f:
            return json.load(f)


def _g(d: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    """First present key from a bench-serve JSON dict (schema varies by version)."""
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                continue
    return default


# ---------------------------------------------------------------------------
# Cost derivation (guarded — never divide by ~zero)
# ---------------------------------------------------------------------------
def _derive_cost(
    instance_usd: float,
    runtime_s: float,
    total_output_tokens: int,
    successful: int,
    gpu_power_watts: float | None,
) -> dict[str, float | None]:
    """Derive cost/efficiency. Returns None (not garbage) when tokens are ~0."""
    if total_output_tokens < 1 or runtime_s <= 0 or instance_usd <= 0:
        logger.warning(
            "Cost not derivable: tokens=%s runtime=%.2fs usd/hr=%.4f -> emitting null",
            total_output_tokens, runtime_s, instance_usd,
        )
        return {"cost_per_1m_tokens": None, "cost_per_qa_form": None,
                "tokens_per_usd": None, "energy_per_1m_tokens_wh": None}

    run_cost = (instance_usd / 3600.0) * runtime_s
    cost_per_1m = run_cost / (total_output_tokens / 1_000_000)
    cost_per_form = run_cost / max(successful, 1)
    tokens_per_usd = total_output_tokens / run_cost if run_cost > 0 else None
    energy_per_1m = None
    if gpu_power_watts and gpu_power_watts > 0:
        wh = gpu_power_watts * (runtime_s / 3600.0)
        energy_per_1m = wh / (total_output_tokens / 1_000_000)
    return {
        "cost_per_1m_tokens": round(cost_per_1m, 6),
        "cost_per_qa_form": round(cost_per_form, 8),
        "tokens_per_usd": round(tokens_per_usd, 2) if tokens_per_usd else None,
        "energy_per_1m_tokens_wh": round(energy_per_1m, 3) if energy_per_1m else None,
    }


# ---------------------------------------------------------------------------
# Build one result row
# ---------------------------------------------------------------------------
def _build_result(
    *,
    bench: dict[str, Any],
    vmetrics_before: dict[str, float | None],
    vmetrics_after: dict[str, float | None],
    gpu: dict[str, float | None],
    concurrency: int,
    profile_id: str,
    manifest: dict[str, Any],
    opt: dict[str, Any],
    reasoning_effort: str,
    correlation_id: str,
    started_at: str,
    ended_at: str,
) -> BenchmarkResult:
    runtime_s = _g(bench, "duration")
    total_out = int(_g(bench, "total_output_tokens"))
    total_in = int(_g(bench, "total_input_tokens"))
    completed = int(_g(bench, "completed", "num_prompts"))
    completed_per_min = completed / max(runtime_s / 60.0, 1e-9)
    instance_usd = float(manifest.get("cost", {}).get("instance_hourly_usd", 0.0))

    cost = _derive_cost(instance_usd, runtime_s, total_out, completed,
                        gpu.get("gpu_power_watts"))

    git_sha = os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip() or "unknown"  # noqa: S605

    # KV / queue: take the peak observed (after-run snapshot is representative of load)
    kv = vmetrics_after.get("kv_cache_utilization_pct")
    waiting = vmetrics_after.get("num_requests_waiting")

    return BenchmarkResult(
        run_id=manifest["run_id"],
        correlation_id=correlation_id,
        hf_id=manifest["model"]["hf_id"],
        model_version=manifest["model"].get("s3_prefix", manifest["model"].get("s3_uri", "")),
        instance_type=manifest["serving"]["instance_type"],
        vllm_image=manifest["serving"]["image"],
        quantization=opt.get("quantization", "none"),
        tensor_parallel_size=int(opt.get("tensor_parallel_size", 1)),
        max_num_seqs=int(opt.get("max_num_seqs", 0)),
        max_num_batched_tokens=int(opt.get("max_num_batched_tokens", 0)),
        max_model_len=int(opt.get("max_model_len", 0)),
        kv_cache_dtype=opt.get("kv_cache_dtype", "auto"),
        profile=profile_id,
        concurrency=concurrency,
        ttft_p50_ms=_g(bench, "median_ttft_ms"),
        ttft_p95_ms=_g(bench, "p95_ttft_ms", "p99_ttft_ms"),
        ttft_p99_ms=_g(bench, "p99_ttft_ms"),
        tpot_p50_ms=_g(bench, "median_tpot_ms"),
        tpot_p95_ms=_g(bench, "p95_tpot_ms", "p99_tpot_ms"),
        itl_p50_ms=_g(bench, "median_itl_ms"),
        itl_p95_ms=_g(bench, "p95_itl_ms", "p99_itl_ms"),
        e2e_p50_ms=_g(bench, "median_e2el_ms"),
        e2e_p95_ms=_g(bench, "p95_e2el_ms", "p99_e2el_ms"),
        e2e_p99_ms=_g(bench, "p99_e2el_ms"),
        request_throughput_rps=_g(bench, "request_throughput"),
        output_throughput_tokens_s=_g(bench, "output_throughput"),
        total_token_throughput_s=_g(bench, "total_token_throughput"),
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        completed_interactions_min=round(completed_per_min, 2),
        kv_cache_utilization_pct=round(kv, 2) if kv is not None else None,
        num_requests_waiting=waiting,
        prefix_cache_hit_rate=vmetrics_after.get("prefix_cache_hit_rate"),
        gpu_utilization_pct=gpu.get("gpu_utilization_pct"),
        gpu_mem_used_mib=gpu.get("gpu_mem_used_mib"),
        gpu_power_watts=gpu.get("gpu_power_watts"),
        instance_hourly_usd=instance_usd,
        runtime_s=round(runtime_s, 2),
        cost_per_1m_tokens=cost["cost_per_1m_tokens"],
        cost_per_qa_form=cost["cost_per_qa_form"],
        tokens_per_usd=cost["tokens_per_usd"],
        energy_per_1m_tokens_wh=cost["energy_per_1m_tokens_wh"],
        accuracy_avg_pct=None,  # measured by the separate lm-eval stage, not here
        total_requests=completed,
        successful_requests=completed,
        error_rate_pct=0.0,
        reasoning_effort=reasoning_effort,
        git_sha=git_sha,
        started_at=started_at,
        ended_at=ended_at,
        status="passed",
    )


def _check_slo(result: BenchmarkResult, slo: dict[str, Any]) -> BenchmarkResult:
    """Evaluate SLO thresholds; mark violations. Zero-token runs fail as 'error'."""
    if result.total_output_tokens < 1:
        result.status = "error"
        result.slo_violations = ["zero output tokens — measurement invalid"]
        return result
    violations: list[str] = []
    if "ttft_p95_ms" in slo and result.ttft_p95_ms > slo["ttft_p95_ms"]:
        violations.append(f"TTFT p95 {result.ttft_p95_ms:.1f}ms > SLO {slo['ttft_p95_ms']}ms")
    if "itl_p95_ms" in slo and result.itl_p95_ms > slo["itl_p95_ms"]:
        violations.append(f"ITL p95 {result.itl_p95_ms:.1f}ms > SLO {slo['itl_p95_ms']}ms")
    if "throughput_tokens_s_min" in slo and result.output_throughput_tokens_s < slo["throughput_tokens_s_min"]:
        violations.append(
            f"Throughput {result.output_throughput_tokens_s:.1f} tok/s < SLO {slo['throughput_tokens_s_min']}"
        )
    result.slo_violations = violations
    result.status = "failed_slo" if violations else "passed"
    return result


# ---------------------------------------------------------------------------
# Persistence + Prometheus push
# ---------------------------------------------------------------------------
def _save_results(results: list[BenchmarkResult], output_dir: str, run_id: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"{run_id}.jsonl"
    with out_file.open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    logger.info("Results written to %s", out_file)
    _push_metrics_to_prometheus(results)


def _push_metrics_to_prometheus(results: list[BenchmarkResult]) -> None:
    """Push derived metrics to Prometheus Pushgateway (best effort)."""
    pg_url = os.getenv("PROMETHEUS_PUSHGATEWAY", "http://prometheus-pushgateway.monitoring:9091")
    for r in results:
        model_name = r.hf_id.split("/")[-1]
        labels = f'model_name="{model_name}",profile="{r.profile}",concurrency="{r.concurrency}",namespace="oai-infopt"'
        lines = [
            "# TYPE llm_benchmark_ttft_p95_ms gauge",
            f"llm_benchmark_ttft_p95_ms{{{labels}}} {r.ttft_p95_ms:.1f}",
            "# TYPE llm_benchmark_output_throughput_tokens_s gauge",
            f"llm_benchmark_output_throughput_tokens_s{{{labels}}} {r.output_throughput_tokens_s:.1f}",
        ]
        if r.cost_per_1m_tokens is not None:
            lines += ["# TYPE llm_benchmark_cost_per_1m_tokens gauge",
                      f"llm_benchmark_cost_per_1m_tokens{{{labels}}} {r.cost_per_1m_tokens:.6f}"]
        payload = ("\n".join(lines) + "\n").encode("utf-8")
        job = f"llm_benchmark/model/{model_name}/profile/{r.profile}/concurrency/{r.concurrency}"
        try:
            req = urllib.request.Request(f"{pg_url}/metrics/job/{job}", data=payload, method="PUT")  # noqa: S310
            with urllib.request.urlopen(req, timeout=3):  # noqa: S310
                logger.debug("Pushed metrics for concurrency=%d", r.concurrency)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Pushgateway push skipped: %s", exc)


def _print_summary(results: list[BenchmarkResult]) -> None:
    sep = "=" * 78
    print(f"\n{sep}\n  BENCHMARK RESULTS - {results[0].run_id}\n{sep}")
    print(f"  {'Profile':<10}{'Conc':>5}{'TTFT p95':>10}{'ITL p95':>9}{'Tok/s':>9}{'Cost/1M$':>11}  Status")
    for r in results:
        cost = f"{r.cost_per_1m_tokens:.4f}" if r.cost_per_1m_tokens is not None else "n/a"
        flag = "PASS" if r.status == "passed" else f"FAIL[{r.status}]"
        print(f"  {r.profile:<10}{r.concurrency:>5}{r.ttft_p95_ms:>9.1f}m{r.itl_p95_ms:>8.1f}m"
              f"{r.output_throughput_tokens_s:>9.1f}{cost:>11}  {flag}")
        for v in r.slo_violations:
            print(f"      - {v}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _run(args: argparse.Namespace) -> None:
    correlation_id = str(uuid.uuid4())
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
    out_cfg = profile.get("output", {})
    in_cfg = profile.get("input", {})
    max_new = int(out_cfg.get("max_new_tokens", 512))
    seed = int(out_cfg.get("seed", 42))
    reasoning_effort = str(out_cfg.get("reasoning_effort", "medium"))
    dataset_name = str(in_cfg.get("dataset_name", "random"))
    random_input_len = int(in_cfg.get("random_input_len", 1024))
    ignore_eos = bool(in_cfg.get("ignore_eos", dataset_name == "random"))
    extra_args: list[str] = list(profile.get("bench_extra_args", []))

    opt = manifest["optimization_variants"][0]
    hf_id = manifest["model"]["hf_id"]
    served_model = manifest["model"].get("served_name") or hf_id.split("/")[-1]
    tokenizer = manifest["model"].get("tokenizer", hf_id)
    endpoint = args.endpoint.rstrip("/")
    dcgm_url = os.getenv("DCGM_METRICS_URL")

    _wait_for_vllm(endpoint, timeout_s=args.wait_timeout)

    all_results: list[BenchmarkResult] = []
    for conc in concurrency_levels:
        num_prompts = max(conc * int(profile.get("num_prompts_factor", 5)),
                          int(profile.get("min_prompts", 100)))
        logger.info("Concurrency=%d, profile=%s, num_prompts=%d", conc, profile_id, num_prompts)
        started_at = datetime.now(tz=timezone.utc).isoformat()

        vmetrics_before = _scrape_vllm_metrics(endpoint)
        try:
            bench = _run_vllm_bench(
                base_url=endpoint,
                served_model=served_model,
                tokenizer=tokenizer,
                concurrency=conc,
                num_prompts=num_prompts,
                dataset_name=dataset_name,
                random_input_len=random_input_len,
                random_output_len=max_new,
                seed=seed,
                ignore_eos=ignore_eos,
                extra_args=extra_args,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Benchmark at concurrency=%d failed: %s", conc, exc)
            continue
        vmetrics_after = _scrape_vllm_metrics(endpoint)
        gpu = _scrape_gpu_metrics(dcgm_url)
        ended_at = datetime.now(tz=timezone.utc).isoformat()

        result = _build_result(
            bench=bench, vmetrics_before=vmetrics_before, vmetrics_after=vmetrics_after,
            gpu=gpu, concurrency=conc, profile_id=profile_id, manifest=manifest, opt=opt,
            reasoning_effort=reasoning_effort, correlation_id=correlation_id,
            started_at=started_at, ended_at=ended_at,
        )
        result = _check_slo(result, slo)
        all_results.append(result)
        _push_metrics_to_prometheus([result])
        logger.info("Concurrency %d: ttft_p95=%.1fms out=%.1f tok/s status=%s",
                    conc, result.ttft_p95_ms, result.output_throughput_tokens_s, result.status)

    if not all_results:
        raise SystemExit("No successful benchmark runs — check vLLM endpoint and logs.")

    _save_results(all_results, args.output, run_id)
    _print_summary(all_results)

    if any(r.status != "passed" for r in all_results):
        raise SystemExit(1)


def main() -> None:
    _configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    parser = argparse.ArgumentParser(description="oai-infopt vLLM benchmark orchestrator (bench serve)")
    parser.add_argument("--manifest", required=True, help="Run manifest YAML")
    parser.add_argument("--profile", required=True, help="Workload profile YAML")
    parser.add_argument("--endpoint", default=os.getenv("VLLM_ENDPOINT", "http://localhost:8080"),
                        help="vLLM base URL")
    parser.add_argument("--output", default=os.getenv("RESULTS_DIR", "results/"),
                        help="Directory to write result JSONL")
    parser.add_argument("--wait-timeout", type=int, default=300,
                        help="Seconds to wait for vLLM /health")
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
