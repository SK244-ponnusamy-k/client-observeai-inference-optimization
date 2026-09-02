#!/usr/bin/env python3
"""
monitoring/metrics-collector/collector.py

Pulls metrics from Amazon Managed Prometheus (AMP) and writes a structured
JSON report to S3 under s3://<RESULTS_BUCKET>/monitoring/reports/<timestamp>.json

Metrics collected:
  GPU:
    - GPU utilization (per GPU, per node)
    - GPU memory used / free
    - GPU temperature
    - GPU power draw
    - Tensor pipe active ratio
    - SM active / occupancy

  vLLM (per model):
    - Requests running / waiting / total
    - Token throughput (prompt + generation tokens/s)
    - TTFT histogram (p50, p95, p99)
    - ITL histogram (p50, p95, p99)
    - GPU cache usage fraction
    - Sequence length distribution

  Cluster:
    - Node count (total, GPU, ready)
    - Pod count (per namespace)
    - CPU / memory utilization per node

Report structure:
  {
    "report_time": "2026-09-02T10:00:00Z",
    "cluster": "ai-eks-cluster",
    "window_minutes": 30,
    "gpu": { ... },
    "vllm": { ... },
    "cluster": { ... },
    "slo_evaluation": { ... }
  }

Auth: EKS Pod Identity injects AWS credentials — no static keys needed.

Usage (standalone):
  python collector.py

Environment variables (injected by CronJob):
  AMP_ENDPOINT        — AMP workspace prometheus endpoint
  AMP_WORKSPACE_ID    — AMP workspace ID
  AWS_REGION          — AWS region
  RESULTS_BUCKET      — S3 bucket for reports
  CLUSTER_NAME        — EKS cluster name
  WINDOW_MINUTES      — lookback window in minutes (default: 30)
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

import boto3
import requests
from aws_requests_auth.boto_utils import BotoAWSRequestsAuth

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
AMP_ENDPOINT     = os.environ["AMP_ENDPOINT"].rstrip("/")
AWS_REGION       = os.environ["AWS_REGION"]
RESULTS_BUCKET   = os.environ["RESULTS_BUCKET"]
CLUSTER_NAME     = os.environ.get("CLUSTER_NAME", "unknown")
WINDOW_MINUTES   = int(os.environ.get("WINDOW_MINUTES", "30"))

# ---------------------------------------------------------------------------
# SLO targets (from workload profiles)
# ---------------------------------------------------------------------------
SLO = {
    "realtime": {
        "ttft_p95_ms": 500,
        "itl_p95_ms": 50,
    },
    "batch": {
        "throughput_tokens_per_s": 500,
    },
}


# ---------------------------------------------------------------------------
# AMP query helpers
# ---------------------------------------------------------------------------

def _amp_auth() -> BotoAWSRequestsAuth:
    """SigV4 auth for AMP API using Pod Identity credentials."""
    return BotoAWSRequestsAuth(
        aws_host=f"aps-workspaces.{AWS_REGION}.amazonaws.com",
        aws_region=AWS_REGION,
        aws_service="aps",
    )


def query_instant(promql: str) -> list[dict]:
    """Run an instant PromQL query against AMP. Returns result list."""
    url = f"{AMP_ENDPOINT}/api/v1/query"
    resp = requests.get(
        url,
        params={"query": promql},
        auth=_amp_auth(),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"AMP query failed: {data}")
    return data["data"]["result"]


def query_range(promql: str, duration: str = None) -> list[dict]:
    """
    Run a range query over the last WINDOW_MINUTES.
    Returns result list with values arrays.
    """
    if duration is None:
        duration = f"{WINDOW_MINUTES}m"
    url = f"{AMP_ENDPOINT}/api/v1/query"
    resp = requests.get(
        url,
        params={"query": f"({promql})[{duration}:]"},
        auth=_amp_auth(),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"AMP range query failed: {data}")
    return data["data"]["result"]


def scalar(results: list[dict], default: float = 0.0) -> float:
    """Extract first scalar value from instant query results."""
    if not results:
        return default
    try:
        return float(results[0]["value"][1])
    except (KeyError, IndexError, ValueError):
        return default


def scalar_by_label(
    results: list[dict], label: str
) -> dict[str, float]:
    """Build {label_value: float} dict from instant query results."""
    out: dict[str, float] = {}
    for r in results:
        key = r.get("metric", {}).get(label, "unknown")
        try:
            out[key] = float(r["value"][1])
        except (KeyError, ValueError):
            out[key] = 0.0
    return out


# ---------------------------------------------------------------------------
# GPU metrics
# ---------------------------------------------------------------------------

def collect_gpu_metrics() -> dict[str, Any]:
    log.info("Collecting GPU metrics...")

    util_by_gpu = scalar_by_label(
        query_instant('DCGM_FI_DEV_GPU_UTIL{job=~".*dcgm.*"}'),
        "gpu",
    )
    mem_used_by_gpu = scalar_by_label(
        query_instant('DCGM_FI_DEV_FB_USED{job=~".*dcgm.*"}'),
        "gpu",
    )
    mem_free_by_gpu = scalar_by_label(
        query_instant('DCGM_FI_DEV_FB_FREE{job=~".*dcgm.*"}'),
        "gpu",
    )
    temp_by_gpu = scalar_by_label(
        query_instant('DCGM_FI_DEV_GPU_TEMP{job=~".*dcgm.*"}'),
        "gpu",
    )
    power_by_gpu = scalar_by_label(
        query_instant('DCGM_FI_DEV_POWER_USAGE{job=~".*dcgm.*"}'),
        "gpu",
    )
    tensor_by_gpu = scalar_by_label(
        query_instant('DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{job=~".*dcgm.*"}'),
        "gpu",
    )
    sm_active_by_gpu = scalar_by_label(
        query_instant('DCGM_FI_PROF_SM_ACTIVE{job=~".*dcgm.*"}'),
        "gpu",
    )
    sm_occ_by_gpu = scalar_by_label(
        query_instant('DCGM_FI_PROF_SM_OCCUPANCY{job=~".*dcgm.*"}'),
        "gpu",
    )

    # Aggregate across GPUs
    gpu_ids = list(util_by_gpu.keys())
    avg = lambda d: round(sum(d.values()) / len(d), 2) if d else 0.0

    per_gpu = []
    for g in gpu_ids:
        mem_used = mem_used_by_gpu.get(g, 0.0)
        mem_free = mem_free_by_gpu.get(g, 0.0)
        mem_total = mem_used + mem_free
        per_gpu.append({
            "gpu_id": g,
            "utilization_pct": round(util_by_gpu.get(g, 0.0), 1),
            "memory_used_mib": round(mem_used, 0),
            "memory_free_mib": round(mem_free, 0),
            "memory_total_mib": round(mem_total, 0),
            "memory_used_pct": round(mem_used / mem_total * 100, 1) if mem_total else 0.0,
            "temperature_c": round(temp_by_gpu.get(g, 0.0), 1),
            "power_w": round(power_by_gpu.get(g, 0.0), 1),
            "tensor_active_ratio": round(tensor_by_gpu.get(g, 0.0), 4),
            "sm_active_ratio": round(sm_active_by_gpu.get(g, 0.0), 4),
            "sm_occupancy_ratio": round(sm_occ_by_gpu.get(g, 0.0), 4),
        })

    return {
        "gpu_count": len(gpu_ids),
        "avg_utilization_pct": avg(util_by_gpu),
        "avg_power_w": avg(power_by_gpu),
        "avg_temperature_c": avg(temp_by_gpu),
        "avg_tensor_active_ratio": avg(tensor_by_gpu),
        "avg_sm_active_ratio": avg(sm_active_by_gpu),
        "per_gpu": per_gpu,
    }


# ---------------------------------------------------------------------------
# vLLM metrics
# ---------------------------------------------------------------------------

def collect_vllm_metrics() -> dict[str, Any]:
    log.info("Collecting vLLM metrics...")

    window = f"{WINDOW_MINUTES}m"

    # Running / waiting requests per model
    running_by_model = scalar_by_label(
        query_instant('vllm:num_requests_running{namespace="oai-infopt"}'),
        "model_name",
    )
    waiting_by_model = scalar_by_label(
        query_instant('vllm:num_requests_waiting{namespace="oai-infopt"}'),
        "model_name",
    )

    # Throughput: rate of tokens generated over the window
    prompt_tps_by_model = scalar_by_label(
        query_instant(
            f'rate(vllm:prompt_tokens_total{{namespace="oai-infopt"}}[{window}])'
        ),
        "model_name",
    )
    gen_tps_by_model = scalar_by_label(
        query_instant(
            f'rate(vllm:generation_tokens_total{{namespace="oai-infopt"}}[{window}])'
        ),
        "model_name",
    )

    # TTFT histogram percentiles (milliseconds)
    ttft_p50 = scalar_by_label(
        query_instant(
            f'histogram_quantile(0.50, rate(vllm:time_to_first_token_seconds_bucket{{namespace="oai-infopt"}}[{window}])) * 1000'
        ),
        "model_name",
    )
    ttft_p95 = scalar_by_label(
        query_instant(
            f'histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket{{namespace="oai-infopt"}}[{window}])) * 1000'
        ),
        "model_name",
    )
    ttft_p99 = scalar_by_label(
        query_instant(
            f'histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket{{namespace="oai-infopt"}}[{window}])) * 1000'
        ),
        "model_name",
    )

    # ITL histogram percentiles (milliseconds)
    itl_p50 = scalar_by_label(
        query_instant(
            f'histogram_quantile(0.50, rate(vllm:inter_token_latency_seconds_bucket{{namespace="oai-infopt"}}[{window}])) * 1000'
        ),
        "model_name",
    )
    itl_p95 = scalar_by_label(
        query_instant(
            f'histogram_quantile(0.95, rate(vllm:inter_token_latency_seconds_bucket{{namespace="oai-infopt"}}[{window}])) * 1000'
        ),
        "model_name",
    )
    itl_p99 = scalar_by_label(
        query_instant(
            f'histogram_quantile(0.99, rate(vllm:inter_token_latency_seconds_bucket{{namespace="oai-infopt"}}[{window}])) * 1000'
        ),
        "model_name",
    )

    # GPU KV cache usage
    cache_usage = scalar_by_label(
        query_instant('vllm:gpu_cache_usage_perc{namespace="oai-infopt"}'),
        "model_name",
    )

    # Build per-model report
    all_models = set(running_by_model.keys()) | set(prompt_tps_by_model.keys())
    per_model = []
    for model in all_models:
        prompt_tps = round(prompt_tps_by_model.get(model, 0.0), 2)
        gen_tps = round(gen_tps_by_model.get(model, 0.0), 2)
        ttft_p95_ms = round(ttft_p95.get(model, 0.0), 2)
        itl_p95_ms = round(itl_p95.get(model, 0.0), 2)

        per_model.append({
            "model": model,
            "requests_running": int(running_by_model.get(model, 0)),
            "requests_waiting": int(waiting_by_model.get(model, 0)),
            "prompt_tokens_per_s": prompt_tps,
            "generation_tokens_per_s": gen_tps,
            "total_tokens_per_s": round(prompt_tps + gen_tps, 2),
            "ttft_ms": {
                "p50": round(ttft_p50.get(model, 0.0), 2),
                "p95": ttft_p95_ms,
                "p99": round(ttft_p99.get(model, 0.0), 2),
            },
            "itl_ms": {
                "p50": round(itl_p50.get(model, 0.0), 2),
                "p95": itl_p95_ms,
                "p99": round(itl_p99.get(model, 0.0), 2),
            },
            "gpu_cache_usage_pct": round(cache_usage.get(model, 0.0) * 100, 1),
        })

    return {"models": per_model}


# ---------------------------------------------------------------------------
# Cluster metrics
# ---------------------------------------------------------------------------

def collect_cluster_metrics() -> dict[str, Any]:
    log.info("Collecting cluster metrics...")

    # Node counts
    total_nodes = scalar(query_instant('count(kube_node_info)'))
    ready_nodes = scalar(
        query_instant('count(kube_node_status_condition{condition="Ready",status="true"})')
    )
    gpu_nodes = scalar(
        query_instant('count(kube_node_labels{label_workload="gpu"}) or vector(0)')
    )

    # Pod counts per namespace
    pod_counts = scalar_by_label(
        query_instant('count by (namespace) (kube_pod_info{phase="Running"})'),
        "namespace",
    )

    # Node CPU utilization
    cpu_by_node = scalar_by_label(
        query_instant(
            f'100 - (avg by (node) (rate(node_cpu_seconds_total{{mode="idle"}}[{WINDOW_MINUTES}m])) * 100)'
        ),
        "node",
    )

    # Node memory utilization
    mem_by_node = scalar_by_label(
        query_instant(
            '100 * (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes))'
        ),
        "node",
    )

    avg = lambda d: round(sum(d.values()) / len(d), 1) if d else 0.0

    return {
        "nodes_total": int(total_nodes),
        "nodes_ready": int(ready_nodes),
        "nodes_gpu": int(gpu_nodes),
        "avg_cpu_utilization_pct": avg(cpu_by_node),
        "avg_memory_utilization_pct": avg(mem_by_node),
        "pods_per_namespace": {k: int(v) for k, v in pod_counts.items()},
    }


# ---------------------------------------------------------------------------
# SLO evaluation
# ---------------------------------------------------------------------------

def evaluate_slos(vllm: dict[str, Any]) -> dict[str, Any]:
    """Check each model's metrics against SLO targets."""
    results = []
    for m in vllm.get("models", []):
        model = m["model"]
        ttft_p95 = m["ttft_ms"]["p95"]
        itl_p95  = m["itl_ms"]["p95"]
        tps      = m["total_tokens_per_s"]

        realtime_pass = (
            ttft_p95 <= SLO["realtime"]["ttft_p95_ms"] and
            itl_p95  <= SLO["realtime"]["itl_p95_ms"]
        ) if ttft_p95 > 0 else None  # None = no data

        batch_pass = (
            tps >= SLO["batch"]["throughput_tokens_per_s"]
        ) if tps > 0 else None

        results.append({
            "model": model,
            "realtime_slo": {
                "pass": realtime_pass,
                "ttft_p95_ms": ttft_p95,
                "ttft_target_ms": SLO["realtime"]["ttft_p95_ms"],
                "itl_p95_ms": itl_p95,
                "itl_target_ms": SLO["realtime"]["itl_p95_ms"],
            },
            "batch_slo": {
                "pass": batch_pass,
                "throughput_tokens_per_s": tps,
                "throughput_target": SLO["batch"]["throughput_tokens_per_s"],
            },
        })

    overall_pass = all(
        r["realtime_slo"]["pass"] is not False and
        r["batch_slo"]["pass"] is not False
        for r in results
    )

    return {
        "overall_pass": overall_pass,
        "per_model": results,
    }


# ---------------------------------------------------------------------------
# Write report to S3
# ---------------------------------------------------------------------------

def write_report(report: dict[str, Any]) -> str:
    """Upload report JSON to S3. Returns the S3 key."""
    s3 = boto3.client("s3", region_name=AWS_REGION)
    ts = report["report_time"].replace(":", "-").replace(".", "-")
    key = f"monitoring/reports/{ts}.json"

    s3.put_object(
        Bucket=RESULTS_BUCKET,
        Key=key,
        Body=json.dumps(report, indent=2, default=str),
        ContentType="application/json",
        Tagging="project=observeai-inference-optimization&component=metrics-collector",
    )
    log.info("Report uploaded: s3://%s/%s", RESULTS_BUCKET, key)

    # Also write a "latest" pointer for easy dashboard consumption
    s3.put_object(
        Bucket=RESULTS_BUCKET,
        Key="monitoring/reports/latest.json",
        Body=json.dumps(report, indent=2, default=str),
        ContentType="application/json",
    )
    return key


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    report_time = datetime.now(timezone.utc).isoformat()
    log.info("=== Metrics Collector starting ===")
    log.info("Cluster      : %s", CLUSTER_NAME)
    log.info("AMP Endpoint : %s", AMP_ENDPOINT)
    log.info("Results S3   : s3://%s/monitoring/reports/", RESULTS_BUCKET)
    log.info("Window       : %d minutes", WINDOW_MINUTES)

    errors: list[str] = []

    # Collect each section — continue on partial failure
    gpu_metrics: dict[str, Any] = {}
    try:
        gpu_metrics = collect_gpu_metrics()
        log.info("GPU metrics: %d GPU(s) found", gpu_metrics.get("gpu_count", 0))
    except Exception as e:
        log.warning("GPU metrics collection failed (no GPU nodes yet?): %s", e)
        errors.append(f"gpu: {e}")

    vllm_metrics: dict[str, Any] = {"models": []}
    try:
        vllm_metrics = collect_vllm_metrics()
        log.info("vLLM metrics: %d model(s) found", len(vllm_metrics.get("models", [])))
    except Exception as e:
        log.warning("vLLM metrics collection failed: %s", e)
        errors.append(f"vllm: {e}")

    cluster_metrics: dict[str, Any] = {}
    try:
        cluster_metrics = collect_cluster_metrics()
        log.info(
            "Cluster metrics: %d nodes (%d GPU)",
            cluster_metrics.get("nodes_total", 0),
            cluster_metrics.get("nodes_gpu", 0),
        )
    except Exception as e:
        log.warning("Cluster metrics collection failed: %s", e)
        errors.append(f"cluster: {e}")

    slo = evaluate_slos(vllm_metrics)
    log.info("SLO overall pass: %s", slo.get("overall_pass"))

    report: dict[str, Any] = {
        "report_time": report_time,
        "cluster": CLUSTER_NAME,
        "window_minutes": WINDOW_MINUTES,
        "gpu": gpu_metrics,
        "vllm": vllm_metrics,
        "cluster_info": cluster_metrics,
        "slo_evaluation": slo,
        "collection_errors": errors,
    }

    try:
        s3_key = write_report(report)
        log.info("=== Collection complete → s3://%s/%s ===", RESULTS_BUCKET, s3_key)
    except Exception as e:
        log.error("Failed to write report to S3: %s", e)
        # Print to stdout so CronJob logs capture it
        print(json.dumps(report, indent=2, default=str))
        return 1

    # Exit non-zero if SLO violated — makes CronJob logs easy to grep
    if slo.get("overall_pass") is False:
        log.warning("SLO VIOLATION detected — check s3://%s/monitoring/reports/latest.json", RESULTS_BUCKET)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
