#!/usr/bin/env python3
"""
scripts/compare_models.py

Parses all JSONL benchmark results from results/ directory (recursively)
or syncs from S3, then generates a multi-model comparison table and Excel/CSV report.

Usage:
  python scripts/compare_models.py
  python scripts/compare_models.py --date today
  python scripts/compare_models.py --date 2026-09-15
  python scripts/compare_models.py --s3-bucket shellkode-ai-results
  python scripts/compare_models.py --export-excel results/model_comparison.xlsx
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
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
                            # Tag the row with its source path so date filtering can
                            # fall back to the S3 run-folder timestamp (results/<ts>/...)
                            # when the row's own started_at is on a different day.
                            data["_source_path"] = str(f)
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
                            data["_source_path"] = str(f)
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


def resolve_date_filter(date_arg: str | None) -> str | None:
    """
    Turns the --date argument into a YYYY-MM-DD string to filter on.
    Accepts: None (no filtering), 'today' (current UTC date), or an
    explicit 'YYYY-MM-DD' string (validated).
    """
    if not date_arg:
        return None

    if date_arg.lower() == "today":
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    try:
        # Validate format, but keep the original string for prefix matching.
        datetime.strptime(date_arg, "%Y-%m-%d")
    except ValueError:
        raise SystemExit(
            f"Invalid --date value: '{date_arg}'. Use 'today' or 'YYYY-MM-DD' (e.g. 2026-09-15)."
        )

    return date_arg


def _folder_date_from_path(path: str) -> str | None:
    """Extract the run-folder date from an S3-synced path as YYYY-MM-DD.

    Results are stored under results/<YYYYMMDD-HHMMSS>/... (and quality/<YYYYMMDD-...>).
    We pull the first 8-digit YYYYMMDD run-folder segment and format it as a date so
    it can be compared against the --date filter. Returns None if not found.
    """
    import re

    for seg in str(path).replace("\\", "/").split("/"):
        m = re.match(r"^(\d{4})(\d{2})(\d{2})-\d{6}", seg)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def filter_by_date(results: list[dict[str, Any]], date_str: str | None) -> list[dict[str, Any]]:
    """Keep rows for date_str (YYYY-MM-DD).

    A row matches if EITHER its own started_at/timestamp OR its S3 run-folder
    timestamp (results/<YYYYMMDD-HHMMSS>/...) falls on date_str. The folder date
    is the fallback because a run's started_at can land on a different UTC day
    than the folder it was uploaded under (e.g. a long batch that crossed midnight,
    or a manifest run_id embedding an older date).
    """
    if not date_str:
        return results

    filtered = []
    for r in results:
        ts = str(r.get("started_at") or r.get("timestamp") or "")
        folder_date = _folder_date_from_path(r.get("_source_path", ""))
        if ts.startswith(date_str) or folder_date == date_str:
            filtered.append(r)
    return filtered


def get_all_headers(results: list[dict[str, Any]]) -> list[str]:
    """Extract union of all keys across all result dicts to prevent missing field errors.

    Internal bookkeeping keys (prefixed with '_', e.g. _source_path, _clean_*) are
    excluded so they don't leak into the exported CSV/Excel detail columns.
    """
    headers = []
    seen = set()
    for r in results:
        for k in r.keys():
            if k.startswith("_"):
                continue
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


# EC2 on-demand $/hr price book (mirrors inference/load-test.py EC2_HOURLY_USD).
# Used ONLY by --fix-instance-labels to recompute cost when a row's recorded
# instance_type disagrees with the S3 run-folder hardware family.
_EC2_HOURLY_USD = {
    "g5.xlarge": 1.0060, "g5.2xlarge": 1.2120, "g5.4xlarge": 1.6240, "g5.8xlarge": 2.4480,
    "g5.12xlarge": 5.6720, "g5.16xlarge": 4.0960, "g5.24xlarge": 8.1440, "g5.48xlarge": 16.2880,
    "g6.xlarge": 0.8048, "g6.2xlarge": 0.9776, "g6.4xlarge": 1.3232, "g6.8xlarge": 2.0144,
    "g6.12xlarge": 4.6016, "g6.16xlarge": 3.3968, "g6.24xlarge": 6.6752, "g6.48xlarge": 13.3504,
    "g6e.xlarge": 1.8610, "g6e.2xlarge": 2.2420, "g6e.4xlarge": 3.0042, "g6e.8xlarge": 4.5286,
    "g6e.12xlarge": 10.6032, "g6e.16xlarge": 7.5772, "g6e.24xlarge": 15.0656, "g6e.48xlarge": 30.1312,
    "g7e.2xlarge": 3.3631, "g7e.4xlarge": 3.9982, "g7e.8xlarge": 5.2682,
    "g7e.12xlarge": 8.2861, "g7e.24xlarge": 16.5722, "g7e.48xlarge": 33.1443,
    "g7.2xlarge": 2.52, "trn1.32xlarge": 21.50, "trn2.48xlarge": 12.0,
}

# Known hardware families that can appear as an S3 run-folder segment.
_HW_FAMILIES = ("g5", "g6e", "g6", "g7e", "g7", "trn2", "trn1")


def _family_of_instance(instance_type: str) -> str | None:
    """'g6.12xlarge' -> 'g6', 'g7e.2xlarge' -> 'g7e'. None if empty."""
    it = str(instance_type or "").strip().lower()
    return it.split(".")[0] if it else None


def _folder_family_from_path(path: str) -> str | None:
    """The hardware family segment in an S3 path, e.g. results/<ts>/g6e/batch/... -> 'g6e'.

    Matches the exact folder segment against known families. Longer names first
    (g6e before g6) so 'g6e' is never mistaken for 'g6'.
    """
    segs = str(path).replace("\\", "/").split("/")
    for seg in segs:
        s = seg.strip().lower()
        for fam in _HW_FAMILIES:  # ordered longest-distinct first
            if s == fam:
                return fam
    return None


def fix_instance_labels(results: list[dict[str, Any]]) -> int:
    """One-time correction for rows whose recorded instance_type disagrees with the
    S3 run-folder hardware family (a pre-dynamic-instance mislabel).

    Rule:
      - Derive the TRUE family from the row's _source_path (results/<ts>/<family>/...).
      - Derive the LABELED family from row['instance_type'].
      - If they MATCH  -> leave the row exactly as-is (data is correct).
      - If they DIFFER -> the run really ran on <true family>; rewrite instance_type
        to the same SIZE under the true family (g6.12xlarge -> g6e.12xlarge) and
        RESCALE every price-derived cost field by (new_price / old_price), because
        all costs scale linearly with instance_hourly_usd. Performance metrics
        (throughput/latency/tokens) and energy_per_1m (power-based) are untouched.

    Returns the number of rows corrected.
    """
    _COST_FIELDS = (
        "cost_per_1m_tokens",
        "cost_per_1m_output_tokens",
        "cost_per_qa_form",
    )
    fixed = 0
    for r in results:
        true_fam = _folder_family_from_path(r.get("_source_path", ""))
        labeled = str(r.get("instance_type", "")).strip().lower()
        labeled_fam = _family_of_instance(labeled)
        if not true_fam or not labeled_fam or true_fam == labeled_fam:
            continue  # unknown folder, or already correct -> keep exact data

        # Same size under the true family: g6.12xlarge -> g6e.12xlarge
        size = labeled.split(".", 1)[1] if "." in labeled else ""
        corrected_instance = f"{true_fam}.{size}" if size else true_fam

        old_price = _EC2_HOURLY_USD.get(labeled)
        new_price = _EC2_HOURLY_USD.get(corrected_instance)
        if not new_price:
            # Can't price the corrected instance; still fix the label but warn.
            print(f"[FIX] {labeled} -> {corrected_instance} (label only; no price for {corrected_instance})")
            r["instance_type"] = corrected_instance
            fixed += 1
            continue

        r["instance_type"] = corrected_instance
        if old_price and old_price > 0:
            scale = new_price / old_price
            # Rescale price-derived cost fields WITHOUT re-rounding — keep full float
            # precision so tiny values (e.g. cost_per_qa_form ~0.001) are never lost.
            # Any rounding is done later at display time, not in the data.
            for fld in _COST_FIELDS:
                v = r.get(fld)
                if isinstance(v, (int, float)):
                    r[fld] = v * scale
            tpu = r.get("tokens_per_usd")
            if isinstance(tpu, (int, float)):
                r["tokens_per_usd"] = tpu / scale  # inverse: fewer tokens per $ at higher price
        r["instance_hourly_usd"] = new_price
        # Quality rows cache a display field derived from instance_type — refresh it.
        if "_clean_hw" in r:
            r["_clean_hw"] = corrected_instance
        fixed += 1
        print(f"[FIX] {labeled} -> {corrected_instance}  (cost x {new_price/old_price:.3f})" if old_price else
              f"[FIX] {labeled} -> {corrected_instance}")
    return fixed


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
    parser.add_argument(
        "--date",
        default=None,
        help=(
            "Only include results whose 'started_at' OR S3 run-folder timestamp "
            "falls on this date. Pass 'today' for the current UTC date, or an explicit "
            "'YYYY-MM-DD' value (e.g. 2026-09-15). Omit to include all dates."
        ),
    )
    parser.add_argument(
        "--fix-instance-labels",
        action="store_true",
        help=(
            "One-time correction: for any row whose recorded instance_type family "
            "disagrees with its S3 run-folder family (results/<ts>/<family>/...), "
            "rewrite instance_type to the folder's family (same size) and recompute "
            "cost from the correct instance price. Rows that already match are left "
            "exactly as-is. Performance metrics are never changed."
        ),
    )
    args = parser.parse_args()

    results_path = Path(args.results_dir)

    if args.s3_bucket:
        sync_results_from_s3(args.s3_bucket, results_path)

    perf_results = load_results_local(results_path)
    qual_results = load_quality_results(results_path)

    if args.fix_instance_labels:
        n = fix_instance_labels(perf_results) + fix_instance_labels(qual_results)
        print(f"[FIX] Corrected {n} mislabeled row(s) (instance_type + cost) from S3 folder family.")

    date_filter = resolve_date_filter(args.date)
    if date_filter:
        before_perf, before_qual = len(perf_results), len(qual_results)
        perf_results = filter_by_date(perf_results, date_filter)
        qual_results = filter_by_date(qual_results, date_filter)
        print(
            f"[FILTER] Date = {date_filter}: "
            f"perf runs {before_perf} -> {len(perf_results)}, "
            f"quality runs {before_qual} -> {len(qual_results)}"
        )

    print_comparison_table(perf_results, qual_results)

    if args.export_csv and perf_results:
        export_csv(perf_results, Path(args.export_csv))

    if args.export_excel:
        export_excel(perf_results, qual_results, Path(args.export_excel))


if __name__ == "__main__":
    main()