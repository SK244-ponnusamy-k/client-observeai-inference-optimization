#!/usr/bin/env python3
"""
scripts/compare_models.py

Parses all JSONL benchmark results from results/ directory (recursively)
or syncs from S3, then generates a multi-model comparison table and Excel/CSV report.

Usage:
  python scripts/compare_models.py
  python scripts/compare_models.py --s3-bucket shellkode-ai-results
  python scripts/compare_models.py --export-excel results/model_comparison.xlsx
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


def load_results_local(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    if not results_dir.exists():
        return rows

    for f in results_dir.rglob("*.jsonl"):
        if "quality" in f.parts or f.name.startswith("comparison") or f.name.startswith("summary"):
            continue
        with f.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "f1" not in data and "f1_score" not in data and "accuracy" not in data:
                            rows.append(data)
                    except json.JSONDecodeError:
                        pass
    return rows


def sync_results_from_s3(bucket: str, local_dir: Path) -> None:
    """Sync both load benchmark results and quality eval results from S3."""
    import subprocess

    cmd_perf = f"aws s3 sync s3://{bucket}/results/ {local_dir}/perf/"
    cmd_qual = f"aws s3 sync s3://{bucket}/quality/ {local_dir}/quality/"
    print(f"Syncing benchmark & quality results from S3 (s3://{bucket}) ...")
    try:
        subprocess.run(cmd_perf, shell=True, check=True)
        subprocess.run(cmd_qual, shell=True, check=True)
        print("S3 sync complete.")
    except Exception as e:
        print(f"Warning: S3 sync failed or AWS CLI not found: {e}")


def load_quality_results(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    if not results_dir.exists():
        return rows

    for f in results_dir.rglob("*.jsonl"):
        with f.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "f1" in data or "f1_score" in data or "accuracy" in data:
                            model = data.get("served_model") or data.get("model") or data.get("hf_id") or "gpt-oss-20b"
                            data["_clean_model"] = model
                            data["_clean_hw"] = data.get("hardware") or data.get("instance_type") or "-"
                            data["_clean_quant"] = data.get("quantization") or "-"
                            data["_clean_rows"] = int(_v(data, "n_total", "total_rows", "n_scored"))
                            data["_clean_acc"] = _v(data, "accuracy") * (100.0 if _v(data, "accuracy") <= 1.0 else 1.0)
                            data["_clean_f1"] = _v(data, "f1", "f1_score", "macro_f1")
                            rows.append(data)
                    except json.JSONDecodeError:
                        pass
    return rows


def get_all_headers(results: list[dict[str, Any]]) -> list[str]:
    """Extract union of all keys across all result dicts to prevent missing field errors."""
    headers = []
    seen = set()
    for r in results:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                headers.append(k)
    return headers


def export_csv(results: list[dict[str, Any]], csv_path: Path) -> None:
    if not results:
        return
    headers = get_all_headers(results)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[SUCCESS] Benchmark comparison CSV exported to: {csv_path.resolve()}")


def _v(r: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    """Safely extract float metric from dict, handling None values and key fallbacks."""
    for key in keys:
        val = r.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return default


def export_excel(perf_results: list[dict[str, Any]], qual_results: list[dict[str, Any]], xlsx_path: Path) -> None:
    """Generates a formatted Excel (.xlsx) report with Performance, Quality, and Detail sheets."""
    if not perf_results and not qual_results:
        print("No benchmark or quality results to export.")
        return

    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError:
        import subprocess
        print("Installing openpyxl in Python environment...")
        subprocess.run([sys.executable, "-m", "pip", "install", "openpyxl"], check=False)
        try:
            import openpyxl
            from openpyxl.styles import Alignment, Font, PatternFill
        except ImportError:
            print("Notice: openpyxl unavailable — exporting CSV report.")
            export_csv(perf_results, xlsx_path.with_suffix(".csv"))
            return

    wb = openpyxl.Workbook()

    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_fill_perf = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    header_fill_qual = PatternFill(start_color="276A3C", end_color="276A3C", fill_type="solid")

    # ── Sheet 1: Performance Comparison ────────────────────────────────────
    ws_perf = wb.active
    ws_perf.title = "Performance Benchmarks"

    headers_perf = [
        "Model (HF ID)",
        "Profile",
        "Concurrency",
        "TTFT p50 (ms)",
        "TTFT p95 (ms)",
        "ITL p95 (ms)",
        "Throughput (tok/s)",
        "Cost / 1M Tokens ($)",
        "Cost / Form ($)",
        "GPU Util (%)",
        "GPU Mem Used (MiB)",
        "KV Cache Usage (%)",
        "Instance Type",
        "Status",
    ]

    ws_perf.append(headers_perf)
    for col_num in range(1, len(headers_perf) + 1):
        cell = ws_perf.cell(row=1, column=col_num)
        cell.font = header_font
        cell.fill = header_fill_perf
        cell.alignment = Alignment(horizontal="center", vertical="center")

    seen_perf = set()
    unique_perf = []
    for r in perf_results:
        key = (r.get("hf_id", ""), r.get("profile", ""), r.get("concurrency", 0), r.get("started_at", ""))
        if key not in seen_perf:
            seen_perf.add(key)
            unique_perf.append(r)

    for r in sorted(unique_perf, key=lambda x: (x.get("hf_id", ""), x.get("profile", ""), x.get("concurrency", 0))):
        ws_perf.append([
            r.get("hf_id", "unknown").split("/")[-1],
            r.get("profile", "unknown"),
            r.get("concurrency", 1),
            round(_v(r, "ttft_p50_ms", "median_ttft_ms"), 1),
            round(_v(r, "ttft_p95_ms", "p95_ttft_ms"), 1),
            round(_v(r, "itl_p95_ms", "p95_itl_ms"), 1),
            round(_v(r, "throughput_tokens_s", "output_throughput_tokens_s"), 1),
            round(_v(r, "cost_per_1m_tokens"), 4),
            round(_v(r, "cost_per_qa_form"), 6),
            round(_v(r, "gpu_utilization_pct"), 1),
            round(_v(r, "gpu_memory_used_mib", "gpu_mem_used_mib"), 0),
            round(_v(r, "gpu_cache_usage_pct", "kv_cache_utilization_pct"), 1),
            r.get("instance_type", "unknown"),
            "PASSED" if r.get("status") == "passed" else "SLO FAIL",
        ])

    # ── Sheet 2: Quality & Accuracy Evaluation ──────────────────────────────
    ws_qual = wb.create_sheet(title="Quality Evaluation")
    headers_qual = [
        "Model",
        "Instance Type / HW",
        "Quantization",
        "Total Transcripts",
        "Accuracy (%)",
        "Precision",
        "Recall",
        "AutoQA F1 Score",
        "Timestamp",
    ]

    ws_qual.append(headers_qual)
    for col_num in range(1, len(headers_qual) + 1):
        cell = ws_qual.cell(row=1, column=col_num)
        cell.font = header_font
        cell.fill = header_fill_qual
        cell.alignment = Alignment(horizontal="center", vertical="center")

    seen_qual = set()
    unique_qual = []
    for q in qual_results:
        key = (q.get("_clean_model", ""), q.get("_clean_quant", ""), q.get("_clean_hw", ""), q.get("started_at", ""))
        if key not in seen_qual:
            seen_qual.add(key)
            unique_qual.append(q)

    for q in sorted(unique_qual, key=lambda x: (x.get("_clean_model", ""), x.get("_clean_quant", ""))):
        ws_qual.append([
            q.get("_clean_model", "gpt-oss-20b"),
            q.get("_clean_hw", "-"),
            q.get("_clean_quant", "-"),
            q.get("_clean_rows", 1200),
            round(q.get("_clean_acc", 0.0), 2),
            round(_v(q, "precision"), 4),
            round(_v(q, "recall"), 4),
            round(q.get("_clean_f1", 0.0), 4),
            str(q.get("started_at", q.get("timestamp", "-")))[:19].replace("T", " "),
        ])

    # ── Sheet 3: All Detailed Runs ──────────────────────────────────────────
    ws_detail = wb.create_sheet(title="All Detailed Perf Runs")
    if unique_perf:
        all_keys = get_all_headers(unique_perf)
        ws_detail.append(all_keys)

        for col_num in range(1, len(all_keys) + 1):
            cell = ws_detail.cell(row=1, column=col_num)
            cell.font = header_font
            cell.fill = PatternFill(start_color="2F5597", end_color="2F5597", fill_type="solid")

        for r in unique_perf:
            row_vals = []
            for k in all_keys:
                v = r.get(k, "")
                if isinstance(v, (list, dict)):
                    row_vals.append(json.dumps(v))
                else:
                    row_vals.append(v)
            ws_detail.append(row_vals)

    # Auto-adjust column widths
    for ws in [ws_perf, ws_qual, ws_detail]:
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = openpyxl.utils.get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx_path)
    print(f"\n[SUCCESS] Model comparison Excel report generated successfully!")
    print(f"[REPORT]  File Location: {xlsx_path.resolve()}\n")


def print_comparison_table(perf_results: list[dict[str, Any]], qual_results: list[dict[str, Any]]) -> None:
    sep = "=" * 105
    print(f"\n{sep}")
    print(f"  LLM BENCHMARK MODEL COMPARISON SUMMARY ({len(perf_results)} perf runs, {len(qual_results)} quality runs)")
    print(sep)

    # ── Section 1: Load Test Performance Results ─────────────────────────────
    if perf_results:
        print("\n  [ PERFORMANCE & EFFICIENCY BENCHMARKS ]")
        print(
            f"  {'Model':<22} {'HW':<8} {'Profile':<9} {'Concur':>6} "
            f"{'TTFT p95':>10} {'ITL p95':>9} {'Tok/s':>9} "
            f"{'Cost/1M':>9} {'GPU Util':>9} {'Status':<9}"
        )
        print(f"  {'-'*101}")

        seen = set()
        unique_perf = []
        for r in perf_results:
            key = (r.get("hf_id", ""), r.get("profile", ""), r.get("concurrency", 0), r.get("started_at", ""))
            if key not in seen:
                seen.add(key)
                unique_perf.append(r)

        for r in sorted(unique_perf, key=lambda x: (x.get("hf_id", ""), x.get("profile", ""), x.get("concurrency", 0))):
            model = r.get("hf_id", "unknown").split("/")[-1]
            hw = r.get("instance_type", r.get("hardware", "-"))
            profile = r.get("profile", "unknown")
            concurrency = r.get("concurrency", 1)
            ttft_p95 = f"{_v(r, 'ttft_p95_ms', 'p95_ttft_ms'):.1f}ms"
            itl_p95 = f"{_v(r, 'itl_p95_ms', 'p95_itl_ms'):.1f}ms"
            throughput = f"{_v(r, 'throughput_tokens_s', 'output_throughput_tokens_s'):.1f}"
            cost = f"${_v(r, 'cost_per_1m_tokens'):.4f}"
            gpu_util = f"{_v(r, 'gpu_utilization_pct'):.1f}%"
            status = "PASSED" if r.get("status") == "passed" else "SLO FAIL"

            print(
                f"  {model:<22} {hw:<8} {profile:<9} {concurrency:>6} "
                f"{ttft_p95:>10} {itl_p95:>9} {throughput:>9} "
                f"{cost:>9} {gpu_util:>9} {status:<9}"
            )

    # ── Section 2: Quality Evaluation Results ────────────────────────────────
    if qual_results:
        print("\n  [ QUALITY & ACCURACY BENCHMARKS (AutoQA F1 Score) ]")
        print(
            f"  {'Model':<22} {'HW':<8} {'Quant':<8} {'Rows':>6} "
            f"{'Accuracy':>10} {'Precision':>10} {'Recall':>9} {'F1 Score':>10}"
        )
        print(f"  {'-'*90}")

        seen_q = set()
        unique_qual = []
        for q in qual_results:
            key = (q.get("_clean_model", ""), q.get("_clean_quant", ""), q.get("_clean_hw", ""), q.get("started_at", ""))
            if key not in seen_q:
                seen_q.add(key)
                unique_qual.append(q)

        for q in sorted(unique_qual, key=lambda x: (x.get("_clean_model", ""), x.get("_clean_quant", ""))):
            model = str(q.get("_clean_model", "gpt-oss-20b"))
            hw = str(q.get("_clean_hw", "-"))
            quant = str(q.get("_clean_quant", "-"))
            rows = q.get("_clean_rows", 1200)
            acc = f"{q.get('_clean_acc', 0.0):.2f}%"
            prec = f"{_v(q, 'precision'):.4f}"
            rec = f"{_v(q, 'recall'):.4f}"
            f1 = f"{q.get('_clean_f1', 0.0):.4f}"

            print(
                f"  {model:<22} {hw:<8} {quant:<8} {rows:>6} "
                f"{acc:>10} {prec:>10} {rec:>9} {f1:>10}"
            )

    print(f"\n{sep}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LLM benchmark result runs")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Path to directory containing benchmark result files (default: results/)",
    )
    parser.add_argument(
        "--export-excel",
        default="results/model_comparison.xlsx",
        help="Path to save output Excel report (default: results/model_comparison.xlsx)",
    )
    parser.add_argument(
        "--export-csv",
        default="",
        help="Optional path to export comparison CSV (e.g. results/model_comparison.csv)",
    )
    parser.add_argument(
        "--s3-bucket",
        default="",
        help="Optional S3 bucket to sync historical results from (e.g. shellkode-ai-results)",
    )
    args = parser.parse_args()

    results_path = Path(args.results_dir)

    if args.s3_bucket:
        sync_results_from_s3(args.s3_bucket, results_path)

    perf_results = load_results_local(results_path)
    qual_results = load_quality_results(results_path)

    print_comparison_table(perf_results, qual_results)

    if args.export_csv and perf_results:
        export_csv(perf_results, Path(args.export_csv))

    if args.export_excel:
        export_excel(perf_results, qual_results, Path(args.export_excel))


if __name__ == "__main__":
    main()

