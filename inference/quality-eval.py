"""
File    : inference/quality-eval.py
Purpose : Quality (accuracy / precision / recall / F1) evaluator for the AutoQA
          rubric on Observe.AI's labelled dataset. SEPARATE from the performance
          harness (load-test.py): this measures whether the model is *correct*,
          not how fast/cheap it serves.

          Task is binary classification (Yes/No): given a transcript + a QA
          question with sub-criteria, did the agent meet the criterion. Ground
          truth is the dataset's `answer` column.

          It calls the SAME vLLM OpenAI endpoint (/v1/chat/completions) at low
          concurrency, deterministically (temperature 0 + fixed seed), and by
          default uses vLLM GUIDED DECODING (`guided_choice`) so every response
          is exactly a label — clean parsing, fair across models. A `free_parse`
          mode is available to let reasoning models think first (then parse the
          final Yes/No).

          Accuracy is hardware-independent, so this runs once per (model,
          quantization) — NOT per GPU family. Results are written separately from
          the performance results.

Owner   : genai-platform@shellkode
Created : 2026-08-20 | Deps: pyyaml (stdlib for the rest)
Security: transcripts are customer data — NEVER logged. Only data_id + pred/gold
          + metrics are emitted. Dataset stays in S3, never committed to git.

Usage (in-cluster, called by run-quality.sh):
    python3 inference/quality-eval.py \\
        --config   /configs/quality/autoqa_v1.yaml \\
        --dataset  /tmp/autoqa_v1.csv \\
        --endpoint http://oai-infopt-vllm-gpt-oss-20b:8000 \\
        --output   /results
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)
csv.field_size_limit(10_000_000)  # transcripts are large


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='{"level":"%(levelname)s","service":"oai-infopt-quality","msg":"%(message)s"}',
    )


# ---------------------------------------------------------------------------
# Result schema (persisted; separate from the performance result row)
# ---------------------------------------------------------------------------
@dataclass
class QualityResult:
    eval_id: str
    served_model: str
    quantization: str
    hardware: str
    dataset_version: str
    decoding_mode: str
    reasoning_effort: str
    n_total: int

    n_scored: int
    n_unparseable: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    positive_label: str
    confusion: dict[str, int]
    per_question_f1: dict[str, float]
    started_at: str
    ended_at: str
    per_question_n: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset loading (CSV or JSONL) — returns list of (data_id, prompt, gold, question)
# ---------------------------------------------------------------------------
def load_dataset(path: str, cfg: dict[str, Any]) -> list[dict[str, str]]:
    text_f = cfg["dataset"]["text_field"]
    label_f = cfg["dataset"]["label_field"]
    q_f = cfg["dataset"].get("question_field", "")
    rows: list[dict[str, str]] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    def _row(d: dict[str, Any], i: int) -> dict[str, str]:
        lower_d = {str(k).strip().lower(): v for k, v in d.items() if k is not None}
        prompt_val = (
            d.get(text_f) or lower_d.get(text_f.lower()) or 
            lower_d.get("question") or lower_d.get("input_prompt") or lower_d.get("prompt") or ""
        )
        gold_val = (
            d.get(label_f) or lower_d.get(label_f.lower()) or 
            lower_d.get("answer") or lower_d.get("gold") or lower_d.get("ground_truth") or lower_d.get("label") or ""
        )
        q_val = d.get(q_f) or lower_d.get(q_f.lower()) or ""

        gold_str = str(gold_val).strip()
        if gold_str.lower() == "yes":
            gold_str = "Yes"
        elif gold_str.lower() == "no":
            gold_str = "No"

        prompt_str = str(prompt_val).strip()
        q_str = str(q_val).strip()
        if (not q_str or q_str == prompt_str) and "Question:" in prompt_str:
            q_match = re.search(r"Question:\s*(.*?)(?:\n|Sub-criteria:|$)", prompt_str, re.IGNORECASE | re.DOTALL)
            if q_match:
                q_str = q_match.group(1).strip()

        return {
            "data_id": str(d.get("data_id") or lower_d.get("data_id") or i),
            "prompt": prompt_str,
            "gold": gold_str,
            "question": (q_str[:60] if q_str else "all"),
        }

    # Auto-detect JSONL content by checking first non-empty characters
    first_chars = ""
    with p.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                first_chars = s[:10]
                break

    is_json = p.suffix.lower() == ".jsonl" or first_chars.startswith("{") or first_chars.startswith("[")

    if is_json:
        with p.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        rows.append(_row(json.loads(line), i))
                    except Exception as exc:
                        logger.warning("Failed to parse JSONL line %d: %s", i+1, exc)
    else:  # csv / tsv
        with p.open(newline="", encoding="utf-8") as f:
            sample = f.read(4096)
            f.seek(0)
            first_line = sample.splitlines()[0] if sample else ""
            delim = "\t" if "\t" in first_line else ","
            rows = [_row(d, i) for i, d in enumerate(csv.DictReader(f, delimiter=delim))]

    n = int(cfg.get("run", {}).get("max_samples", 0) or 0)
    if n > 0:
        rows = rows[:n]
    return rows



# ---------------------------------------------------------------------------
# One model call (vLLM OpenAI chat completions, stdlib HTTP)
# ---------------------------------------------------------------------------
def _post_chat(endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        return json.loads(resp.read())


def _guided_supported(endpoint: str, served_model: str, labels: list[str]) -> bool:
    """Probe ONCE whether the server accepts `guided_choice`. If the field name
    differs on this vLLM build, degrade the whole run to free_parse rather than
    turning every row into UNKNOWN."""
    body = {
        "model": served_model,
        "messages": [{"role": "user", "content": "Reply with one word."}],
        "temperature": 0.0, "max_tokens": 2, "guided_choice": labels,
    }
    try:
        _post_chat(endpoint, body)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("guided_choice not accepted by server (%s) — using free_parse.",
                       type(exc).__name__)
        return False


def _classify_one(
    *, endpoint: str, served_model: str, prompt: str, labels: list[str],
    dcfg: dict[str, Any], mcfg: dict[str, Any],
) -> str:
    messages: list[dict[str, str]] = []
    sys_prompt = (mcfg.get("system_prompt") or "").strip()
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": prompt})

    mode = dcfg.get("mode", "guided_choice")
    body: dict[str, Any] = {
        "model": served_model,
        "messages": messages,
        "temperature": float(dcfg.get("temperature", 0.0)),
        "seed": int(dcfg.get("seed", 42)),
    }
    # gpt-oss reasoning effort (ignored by models that don't support it)
    effort = (mcfg.get("reasoning_effort") or "").strip()
    if effort:
        body["reasoning_effort"] = effort

    if mode == "guided_choice":
        body["guided_choice"] = labels               # vLLM structured output
        body["max_tokens"] = int(dcfg.get("max_tokens", 8))
    else:  # free_parse — let reasoning models think, then parse
        body["max_tokens"] = int(dcfg.get("reasoning_max_tokens", 1024))

    out = _post_chat(endpoint, body)
    msg = out["choices"][0]["message"]
    # Combine content and reasoning fields in case vLLM places output in content, reasoning_content, or reasoning
    full_text = " ".join([
        str(msg.get("content") or ""),
        str(msg.get("reasoning_content") or ""),
        str(msg.get("reasoning") or ""),
        str(msg.get("text") or ""),
    ]).strip()
    return _parse_label(full_text, labels)


def _parse_label(content: str, labels: list[str]) -> str:
    """Exact match first; else last label mention; else fallback search; else UNKNOWN."""
    c = content.strip()
    if not c:
        return "UNKNOWN"
    for lab in labels:
        if c.lower() == lab.lower():
            return lab
    matches = re.findall(r"\b(" + "|".join(re.escape(x) for x in labels) + r")\b", c, re.IGNORECASE)
    if matches:
        last = matches[-1].lower()
        for lab in labels:
            if lab.lower() == last:
                return lab
    # No exact match and no word-boundary label → genuinely no clear verdict.
    # Deliberately NO substring fallback: "no" is a substring of not/know/now/cannot
    # and "yes" of yesterday, so `lab in text` mis-scores. Return UNKNOWN honestly
    # (it counts against accuracy and shows up in n_unparseable) rather than guessing.
    return "UNKNOWN"



# ---------------------------------------------------------------------------
# Metrics (binary; no sklearn dependency)
# ---------------------------------------------------------------------------
def _binary_metrics(preds: list[str], golds: list[str], pos: str) -> dict[str, float]:
    tp = fp = fn = tn = 0
    for p, g in zip(preds, golds):
        if g == pos and p == pos:
            tp += 1
        elif g != pos and p == pos:
            fp += 1
        elif g == pos and p != pos:
            fn += 1                    # actual-positive missed (includes UNKNOWN)
        elif g != pos and p == g:
            tn += 1                    # correct negative (pred == actual negative label)
        # else: unparseable/other on a negative → NOT a correct TN.
        # It's left out of TP/TN so accuracy = (TP+TN)/N counts it as wrong,
        # while positive-class precision/recall are unaffected (no Yes predicted).
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(len(preds), 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


def _f1_for(preds: list[str], golds: list[str], pos: str) -> float:
    return _binary_metrics(preds, golds, pos)["f1"]


# ---------------------------------------------------------------------------
# Health wait
# ---------------------------------------------------------------------------
def _wait_for_vllm(endpoint: str, timeout_s: int = 300) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{endpoint}/health", timeout=5)  # noqa: S310
            return
        except Exception:  # noqa: BLE001
            time.sleep(5)
    raise TimeoutError(f"vLLM not ready at {endpoint} after {timeout_s}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _run(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    labels: list[str] = [str(x) for x in cfg["labels"]]
    pos: str = str(cfg["positive_label"])
    dcfg = cfg.get("decoding", {})
    mcfg = cfg.get("model", {})
    served_model = args.model or mcfg.get("served_name") or "model"
    concurrency = int(cfg.get("run", {}).get("concurrency", 8))
    endpoint = args.endpoint.rstrip("/")

    rows = load_dataset(args.dataset, cfg)
    logger.info("Loaded %d rows; labels=%s positive=%s mode=%s",
                len(rows), labels, pos, dcfg.get("mode"))

    _wait_for_vllm(endpoint, timeout_s=args.wait_timeout)

    # Capability probe: if guided_choice isn't accepted on this vLLM build,
    # degrade to free_parse for the whole run (parser still extracts Yes/No).
    if str(dcfg.get("mode", "guided_choice")) == "guided_choice" and not _guided_supported(
        endpoint, served_model, labels
    ):
        dcfg = dict(dcfg)
        dcfg["mode"] = "free_parse"
        logger.warning("Decoding mode fell back to free_parse for this run.")

    started_at = datetime.now(tz=timezone.utc).isoformat()

    def _worker(row: dict[str, str]) -> dict[str, str]:
        try:
            pred = _classify_one(endpoint=endpoint, served_model=served_model,
                                 prompt=row["prompt"], labels=labels, dcfg=dcfg, mcfg=mcfg)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Request failed for data_id=%s: %s", row["data_id"], type(exc).__name__)
            pred = "UNKNOWN"
        return {"data_id": row["data_id"], "gold": row["gold"],
                "pred": pred, "question": row["question"]}

    results: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for i, r in enumerate(ex.map(_worker, rows), 1):
            results.append(r)
            if i % 100 == 0:
                logger.info("Scored %d/%d", i, len(rows))

    ended_at = datetime.now(tz=timezone.utc).isoformat()

    preds = [r["pred"] for r in results]
    golds = [r["gold"] for r in results]
    n_unparseable = sum(1 for p in preds if p == "UNKNOWN")

    m = _binary_metrics(preds, golds, pos)
    # macro-F1 over both classes
    other = next((x for x in labels if x != pos), pos)
    macro_f1 = (_f1_for(preds, golds, pos) + _f1_for(preds, golds, other)) / 2.0

    # per-question F1
    per_q_f1: dict[str, float] = {}
    per_q_n: dict[str, int] = {}
    for q in sorted({r["question"] for r in results}):
        qp = [r["pred"] for r in results if r["question"] == q]
        qg = [r["gold"] for r in results if r["question"] == q]
        per_q_f1[q] = round(_f1_for(qp, qg, pos), 4)
        per_q_n[q] = len(qp)

    result = QualityResult(
        eval_id=cfg.get("eval_id", "autoqa"),
        served_model=served_model,
        quantization=args.quantization,
        hardware=args.hardware,
        dataset_version=cfg.get("dataset", {}).get("version", "autoqa_v1"),
        decoding_mode=str(dcfg.get("mode", "guided_choice")),
        reasoning_effort=str(mcfg.get("reasoning_effort", "") or "n/a"),
        n_total=len(results),
        n_scored=len(results) - n_unparseable,
        n_unparseable=n_unparseable,
        accuracy=round(m["accuracy"], 4),
        precision=round(m["precision"], 4),
        recall=round(m["recall"], 4),
        f1=round(m["f1"], 4),
        macro_f1=round(macro_f1, 4),
        positive_label=pos,
        confusion={"tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"]},
        per_question_f1=per_q_f1,
        per_question_n=per_q_n,
        started_at=started_at,
        ended_at=ended_at,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"quality-{result.eval_id}-{served_model.replace('/', '_')}.jsonl"
    with out_file.open("w") as f:
        f.write(json.dumps(asdict(result)) + "\n")
    logger.info("Wrote %s", out_file)

    _print_summary(result)


def _print_summary(r: QualityResult) -> None:
    sep = "=" * 70
    print(f"\n{sep}\n  QUALITY (AutoQA) — {r.served_model}  [{r.quantization}]  ({r.hardware})\n{sep}")
    print(f"  dataset={r.dataset_version}  decoding={r.decoding_mode}  reasoning={r.reasoning_effort}")
    print(f"  n={r.n_total}  unparseable={r.n_unparseable}")
    print(f"  Accuracy : {r.accuracy:.4f}")
    print(f"  Precision: {r.precision:.4f}   Recall: {r.recall:.4f}")
    print(f"  F1 (pos='{r.positive_label}'): {r.f1:.4f}   Macro-F1: {r.macro_f1:.4f}")
    print(f"  Confusion: {r.confusion}")
    print("  Per-question F1:")
    for q, f1 in r.per_question_f1.items():
        print(f"    - [{r.per_question_n.get(q, 0):>4}] {f1:.4f}  {q}")
    print(sep + "\n")


def main() -> None:
    _configure_logging()
    ap = argparse.ArgumentParser(description="oai-infopt AutoQA quality evaluator (F1/accuracy)")
    ap.add_argument("--config", required=True, help="Eval config YAML")
    ap.add_argument("--dataset", required=True, help="Local dataset path (csv or jsonl)")
    ap.add_argument("--endpoint", required=True, help="vLLM base URL")
    ap.add_argument("--output", default="results/", help="Output dir for the quality JSONL")
    ap.add_argument("--model", default="", help="Served model name override")
    ap.add_argument("--quantization", default="unknown", help="Quant label for the result row")
    ap.add_argument("--hardware", default="unknown", help="Hardware/instance type label (e.g. g5, g6, g6e)")
    ap.add_argument("--wait-timeout", type=int, default=300)
    _run(ap.parse_args())



if __name__ == "__main__":
    main()
