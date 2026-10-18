"""
File    : inference/validate_dataset.py
Purpose : Load and validate a benchmark dataset in the ObserveAI evaluation
          format, then optionally export a clean JSONL for load-test.py.

Supported input formats
-----------------------
  .xlsx / .xls  — Excel workbook; first sheet used.
  .csv           — comma-separated (or custom --delimiter).
  .tsv / .txt    — tab-separated.
  .jsonl         — one JSON object per line.

Required columns / keys
-----------------------
  data_id        unique row identifier
  input_prompt   full LLM prompt (QUESTION block + TRANSCRIPT block)
  answer         ground-truth label, e.g. "Yes" or "No"

Validation checks
-----------------
  1. Schema       – required columns are present.
  2. Completeness – no blank input_prompt or answer values.
  3. Label set    – answer values are in the allowed set (default: Yes | No).
  4. Duplicates   – duplicate data_id values are flagged.
  5. Structure    – warns if QUESTION / TRANSCRIPT markers are absent from
                    input_prompt (warn-only, does not fail validation).

Exit codes
----------
  0  dataset is valid (warnings may be present)
  1  dataset has hard errors

Usage examples
--------------
  # validate an Excel file
  python inference/validate_dataset.py \\
      configs/workload_profiles/datasets/qa_eval_v1.xlsx

  # validate and export a JSONL ready for load-test.py
  python inference/validate_dataset.py \\
      configs/workload_profiles/datasets/qa_eval_v1.xlsx --export-jsonl

  # preview first 3 rows
  python inference/validate_dataset.py \\
      configs/workload_profiles/datasets/qa_eval_v1.xlsx --show-rows 3

  # CSV with a custom label set
  python inference/validate_dataset.py my_data.csv \\
      --delimiter , --labels Yes No Maybe

Owner   : genai-platform@shellkode
Created : 2026-09-07
Deps    : openpyxl (pip install openpyxl)  — only for .xlsx/.xls input
          standard library for all other formats
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    format="%(levelname)-8s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DatasetRow:
    """One normalised benchmark row."""
    data_id: str
    input_prompt: str   # full prompt text (question block + transcript block)
    answer: str         # ground-truth label, e.g. "Yes" / "No"
    source_line: int    # 1-indexed row/line number in the source file


@dataclass
class ValidationReport:
    total_rows: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def error(self, msg: str) -> None:
        self.errors.append(msg)
        logger.error(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(msg)

    def print_summary(self) -> None:
        sep = "─" * 62
        print(f"\n{sep}")
        print("  DATASET VALIDATION SUMMARY")
        print(sep)
        print(f"  Total rows   : {self.total_rows}")
        print(f"  Errors       : {len(self.errors)}")
        print(f"  Warnings     : {len(self.warnings)}")
        print(sep)
        if self.errors:
            print("  ERRORS:")
            for e in self.errors:
                print(f"    ✗  {e}")
        if self.warnings:
            print("  WARNINGS:")
            for w in self.warnings:
                print(f"    ⚠  {w}")
        verdict = "✓  VALID" if not self.has_errors else "✗  INVALID — fix errors above"
        print(f"\n  Result: {verdict}")
        print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

# Column name aliases accepted in any format
_ID_ALIASES      = {"data_id", "id"}
_PROMPT_ALIASES  = {"input_prompt", "question", "prompt", "content"}
_ANSWER_ALIASES  = {"answer", "ground_truth", "expected_output"}

REQUIRED_ALIASES = (_ID_ALIASES, _PROMPT_ALIASES, _ANSWER_ALIASES)


def _resolve_columns(headers: list[str]) -> dict[str, str]:
    """
    Map logical role → actual header name.
    Returns dict with keys: 'data_id', 'input_prompt', 'answer'.
    Raises ValueError if any required column is missing.
    """
    lower_map = {h.strip().lower(): h.strip() for h in headers}
    resolved: dict[str, str] = {}

    role_names = ["data_id", "input_prompt", "answer"]
    for role, aliases in zip(role_names, REQUIRED_ALIASES):
        for alias in aliases:
            if alias in lower_map:
                resolved[role] = lower_map[alias]
                break
        if role not in resolved:
            raise ValueError(
                f"Required column not found for role '{role}'. "
                f"Expected one of {sorted(aliases)}. "
                f"Available columns: {list(lower_map.values())}"
            )
    return resolved


def _load_xlsx(path: Path) -> list[DatasetRow]:
    """Load first sheet of an Excel workbook (.xlsx / .xls)."""
    try:
        import openpyxl  # type: ignore[import]
    except ImportError:
        logger.error(
            "openpyxl is required to read Excel files.\n"
            "  Install it with:  pip install openpyxl"
        )
        sys.exit(1)

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)

    # First row is the header
    raw_headers = [str(c).strip() if c is not None else "" for c in next(rows_iter)]
    col = _resolve_columns(raw_headers)

    id_idx      = raw_headers.index(col["data_id"])
    prompt_idx  = raw_headers.index(col["input_prompt"])
    answer_idx  = raw_headers.index(col["answer"])

    rows: list[DatasetRow] = []
    for line_no, raw in enumerate(rows_iter, start=2):
        # Pad short rows
        row_vals = list(raw) + [""] * max(0, max(id_idx, prompt_idx, answer_idx) + 1 - len(raw))
        # Excel stores integer cells as floats (e.g. 1000.0) — normalise to int string
        raw_id = row_vals[id_idx]
        if isinstance(raw_id, float) and raw_id.is_integer():
            raw_id = int(raw_id)
        rows.append(DatasetRow(
            data_id=str(raw_id or "").strip(),
            input_prompt=str(row_vals[prompt_idx] or "").strip(),
            answer=str(row_vals[answer_idx] or "").strip(),
            source_line=line_no,
        ))
    wb.close()
    return rows


def _load_tsv_csv(path: Path, delimiter: str) -> list[DatasetRow]:
    """Load a delimiter-separated file; supports quoted multi-line cells."""
    rows: list[DatasetRow] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError("File appears empty — no header row found.")
        col = _resolve_columns(list(reader.fieldnames))
        for line_no, raw in enumerate(reader, start=2):
            rows.append(DatasetRow(
                data_id=(raw.get(col["data_id"]) or "").strip(),
                input_prompt=(raw.get(col["input_prompt"]) or "").strip(),
                answer=(raw.get(col["answer"]) or "").strip(),
                source_line=line_no,
            ))
    return rows


def _load_jsonl(path: Path) -> list[DatasetRow]:
    """Load a JSONL file; accepts flexible key names."""
    rows: list[DatasetRow] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            obj: dict[str, Any] = json.loads(line)
            lower_obj = {k.lower(): v for k, v in obj.items()}

            data_id = str(
                next((lower_obj[a] for a in _ID_ALIASES if a in lower_obj), line_no)
            )
            input_prompt = str(
                next((lower_obj[a] for a in _PROMPT_ALIASES if a in lower_obj), "")
            )
            answer = str(
                next((lower_obj[a] for a in _ANSWER_ALIASES if a in lower_obj), "")
            )
            rows.append(DatasetRow(
                data_id=data_id,
                input_prompt=input_prompt,
                answer=answer,
                source_line=line_no,
            ))
    return rows


def load_dataset(path: Path, delimiter: str) -> list[DatasetRow]:
    """Dispatch to the right loader based on file extension."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        logger.info("Detected format: Excel (%s)", suffix)
        return _load_xlsx(path)
    elif suffix == ".jsonl":
        logger.info("Detected format: JSONL")
        return _load_jsonl(path)
    elif suffix in (".csv", ".tsv", ".txt"):
        fmt = "TSV" if (suffix in (".tsv", ".txt") or delimiter == "\t") else "CSV"
        logger.info("Detected format: %s (delimiter=%r)", fmt, delimiter)
        return _load_tsv_csv(path, delimiter)
    else:
        logger.info("Unknown extension %r — attempting TSV load", suffix)
        return _load_tsv_csv(path, delimiter)


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------

_STRUCTURAL_MARKERS = ["=== QUESTION ===", "=== TRANSCRIPT ==="]


def validate(rows: list[DatasetRow], valid_labels: set[str], args_strict_structure: bool = False) -> ValidationReport:
    report = ValidationReport(total_rows=len(rows))

    if not rows:
        report.error("Dataset is empty — no data rows were loaded.")
        return report

    seen_ids: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    _marker_missing_counts: dict[str, int] = {}

    for row in rows:
        # 1. Blank data_id
        if not row.data_id:
            report.error(f"Row {row.source_line}: data_id is blank.")

        # 2. Blank input_prompt
        if not row.input_prompt:
            report.error(
                f"Row {row.source_line} (data_id={row.data_id!r}): input_prompt is blank."
            )

        # 3. Blank answer
        if not row.answer:
            report.error(
                f"Row {row.source_line} (data_id={row.data_id!r}): answer is blank."
            )

        # 4. Answer label in allowed set
        label_counts[row.answer] = label_counts.get(row.answer, 0) + 1
        if valid_labels and row.answer and row.answer not in valid_labels:
            report.error(
                f"Row {row.source_line} (data_id={row.data_id!r}): "
                f"answer {row.answer!r} not in allowed labels {sorted(valid_labels)}."
            )

        # 5. Duplicate data_id
        if row.data_id:
            if row.data_id in seen_ids:
                report.error(
                    f"Row {row.source_line}: duplicate data_id={row.data_id!r} "
                    f"(first seen at row {seen_ids[row.data_id]})."
                )
            else:
                seen_ids[row.data_id] = row.source_line

        # 6. Structural markers — summarise instead of spamming per-row warnings
        if row.input_prompt:
            for marker in _STRUCTURAL_MARKERS:
                if marker not in row.input_prompt:
                    # Collect counts; emit final summary warning after the loop
                    _marker_missing_counts[marker] = _marker_missing_counts.get(marker, 0) + 1

    # Emit structural-marker summary (one warning per marker, not per row)
    for marker, count in _marker_missing_counts.items():
        if args_strict_structure:
            report.warn(
                f"Structural marker {marker!r} missing from {count}/{report.total_rows} rows."
            )
        else:
            logger.info(
                "Structural marker %r absent in %d/%d rows "
                "(use --strict-structure to treat this as a warning).",
                marker, count, report.total_rows,
            )

    logger.info("Label distribution: %s", label_counts)
    return report


# ---------------------------------------------------------------------------
# JSONL export  (consumed by load_prompts() in load-test.py)
# ---------------------------------------------------------------------------

def export_as_jsonl(rows: list[DatasetRow], out_path: Path) -> None:
    """
    Write rows as JSONL using keys that load_prompts() in load-test.py
    already understands: system_prompt, question, answer.

    The full input_prompt becomes the 'question' field so the model receives
    the complete evaluation context (question block + transcript).
    """
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            obj = {
                "data_id": row.data_id,
                "system_prompt": (
                    "You are an expert QA evaluation assistant. "
                    "Respond with only Yes or No."
                ),
                "question": row.input_prompt,
                "answer": row.answer,
            }
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    logger.info("Exported %d rows → %s", len(rows), out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Load and validate a benchmark dataset for the ObserveAI QA eval pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "dataset",
        metavar="DATASET_PATH",
        help="Path to the dataset file (.xlsx, .csv, .tsv, or .jsonl).",
    )
    p.add_argument(
        "--delimiter",
        default="\t",
        help="Column delimiter for CSV/TSV files. Use ',' for CSV. Default: tab.",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=["Yes", "No"],
        metavar="LABEL",
        help="Allowed answer label values (case-sensitive). Default: Yes No.",
    )
    p.add_argument(
        "--no-label-check",
        action="store_true",
        help="Skip label validation entirely (useful for free-form answer datasets).",
    )
    p.add_argument(
        "--strict-structure",
        action="store_true",
        help=(
            "Treat missing === QUESTION === / === TRANSCRIPT === markers as "
            "warnings (shown in summary). By default only an INFO count is logged."
        ),
    )
    p.add_argument(
        "--export-jsonl",
        action="store_true",
        help=(
            "After validation, write a JSONL file (same name, .jsonl extension) "
            "that is directly consumable by inference/load-test.py."
        ),
    )
    p.add_argument(
        "--show-rows",
        type=int,
        default=0,
        metavar="N",
        help="Print the first N loaded rows for a quick sanity-check.",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error("Dataset file not found: %s", dataset_path)
        sys.exit(1)

    delimiter = "\t" if args.delimiter in ("\t", "\\t") else args.delimiter
    valid_labels: set[str] = set() if args.no_label_check else set(args.labels)

    logger.info("Loading: %s", dataset_path.resolve())

    try:
        rows = load_dataset(dataset_path, delimiter)
    except Exception as exc:
        logger.error("Failed to load dataset: %s", exc)
        sys.exit(1)

    logger.info("Loaded %d row(s)", len(rows))

    if args.show_rows > 0:
        print(f"\n{'─'*62}")
        print(f"  First {min(args.show_rows, len(rows))} row(s):")
        print(f"{'─'*62}")
        for row in rows[: args.show_rows]:
            preview = row.input_prompt[:100].replace("\n", " ↵ ")
            print(
                f"  [{row.source_line:>4}] id={row.data_id:<8} "
                f"answer={row.answer!r:<6}  prompt={preview!r}..."
            )
        print()

    report = validate(rows, valid_labels, args_strict_structure=args.strict_structure)
    report.print_summary()

    if args.export_jsonl:
        if report.has_errors:
            logger.warning("--export-jsonl skipped because the dataset has errors.")
        else:
            out_path = dataset_path.with_suffix(".jsonl")
            export_as_jsonl(rows, out_path)
            logger.info(
                "Point your workload profile's input.dataset to: %s", out_path
            )

    sys.exit(1 if report.has_errors else 0)


if __name__ == "__main__":
    main()
