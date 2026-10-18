"""
File    : tests/test_telemetry.py
Purpose : Regression tests for the benchmark harness telemetry (inference/load-test.py).

          Guards the two bugs behind the 24-Sep result sheet:
            1. KV-cache usage read 0 % on every row (scraped AFTER the run).
            2. GPU util / mem mixed GPUs from random nodes (400 %, 164 GB on an L4).

Run     : python -m unittest discover -s tests      (stdlib only + pyyaml)
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "load_test", Path(__file__).resolve().parents[1] / "inference" / "load-test.py"
)
lt = importlib.util.module_from_spec(_SPEC)
sys.modules["load_test"] = lt  # dataclasses resolve annotations via sys.modules
_SPEC.loader.exec_module(lt)  # type: ignore[union-attr]

POD = "oai-infopt-vllm-gpt-oss-20b-7c9f8d6b5d-x2k9q"
TAGGED_POD = "oai-infopt-vllm-gpt-oss-20b-g7e-5b8c7d9f4-p7m2n"

# One DCGM exporter page: our pod's GPU, a tagged parallel deploy's GPU, and an
# idle GPU on the same node with no pod assigned.
DCGM_PAGE = f"""# HELP DCGM_FI_DEV_GPU_UTIL GPU utilization (in %).
# TYPE DCGM_FI_DEV_GPU_UTIL gauge
DCGM_FI_DEV_GPU_UTIL{{gpu="0",Hostname="ip-10-0-1-5",namespace="oai-infopt",pod="{POD}",container="vllm"}} 97
DCGM_FI_DEV_GPU_UTIL{{gpu="1",Hostname="ip-10-0-1-5",namespace="oai-infopt",pod="{TAGGED_POD}",container="vllm"}} 100
DCGM_FI_DEV_GPU_UTIL{{gpu="2",Hostname="ip-10-0-1-5"}} 0
DCGM_FI_DEV_FB_USED{{gpu="0",namespace="oai-infopt",pod="{POD}",container="vllm"}} 21000
DCGM_FI_DEV_FB_USED{{gpu="1",namespace="oai-infopt",pod="{TAGGED_POD}",container="vllm"}} 40000
DCGM_FI_DEV_FB_USED{{gpu="2"}} 0
DCGM_FI_DEV_POWER_USAGE{{gpu="0",namespace="oai-infopt",pod="{POD}",container="vllm"}} 70.5
DCGM_FI_DEV_POWER_USAGE{{gpu="1",namespace="oai-infopt",pod="{TAGGED_POD}",container="vllm"}} 300
"""


class _FakeVllm:
    """Serves /metrics with a KV gauge that is high while 'busy' and 0 after."""

    def __init__(self) -> None:
        self.busy = True
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/metrics":
                    kv = 0.83 if outer.busy else 0.0
                    waiting = 12 if outer.busy else 0
                    body = (f'vllm:kv_cache_usage_perc{{engine="0",model_name="m"}} {kv}\n'
                            f'vllm:num_requests_waiting{{engine="0",model_name="m"}} {waiting}\n'
                            f'vllm:gpu_memory_utilization 0.9\n')
                else:
                    body = DCGM_PAGE
                data = body.encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a: object) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class GpuScopingTest(unittest.TestCase):
    def test_only_target_pod_counted(self) -> None:
        srv = _FakeVllm()
        try:
            g = lt._scrape_gpu_metrics(f"{srv.url}/dcgm", "oai-infopt-vllm-gpt-oss-20b", "oai-infopt")
        finally:
            srv.close()
        self.assertEqual(g["gpu_utilization_pct"], 97)
        self.assertEqual(g["gpu_mem_used_mib"], 21000)
        self.assertEqual(g["gpu_power_watts"], 70.5)

    def test_other_node_scrape_is_none(self) -> None:
        srv = _FakeVllm()
        try:
            g = lt._scrape_gpu_metrics(f"{srv.url}/dcgm", "oai-infopt-vllm-qwen3-5-4b", "oai-infopt")
        finally:
            srv.close()
        self.assertIsNone(g["gpu_utilization_pct"])
        self.assertIsNone(g["gpu_mem_used_mib"])

    def test_namespace_fallback_averages_util(self) -> None:
        match = lt._pod_matcher(None, "oai-infopt")
        samples = [v for labels, v in lt._metric_samples(DCGM_PAGE, "DCGM_FI_DEV_GPU_UTIL") if match(labels)]
        self.assertEqual(samples, [97, 100])  # idle, unassigned GPU excluded


class KvSamplingTest(unittest.TestCase):
    def test_kv_is_the_under_load_peak_not_the_drained_value(self) -> None:
        srv = _FakeVllm()
        try:
            with lt._TelemetrySampler(srv.url, f"{srv.url}/dcgm", "oai-infopt-vllm-gpt-oss-20b",
                                      "oai-infopt", interval_s=0.05) as s:
                time.sleep(0.3)
            srv.busy = False  # run finished, queue drained
            after = lt._scrape_vllm_metrics(srv.url)
        finally:
            srv.close()
        summary = s.summary()
        self.assertEqual(after["kv_cache_utilization_pct"], 0.0)  # the old, wrong reading
        self.assertAlmostEqual(summary["kv_cache_utilization_pct"], 83.0)
        self.assertEqual(summary["num_requests_waiting"], 12)
        self.assertGreater(summary["telemetry_samples"], 1)
        self.assertGreater(summary["gpu_telemetry_samples"], 1)
        self.assertEqual(summary["gpu_utilization_pct"], 97)

    def test_gpu_memory_utilization_is_not_read_as_kv(self) -> None:
        text = "vllm:gpu_memory_utilization 0.9\n"
        self.assertIsNone(lt._sum_metric(text, "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"))


if __name__ == "__main__":
    unittest.main()
