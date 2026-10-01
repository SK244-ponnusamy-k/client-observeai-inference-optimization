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

# Distinct exit code for "the model endpoint never became ready" (health check
# timed out). Mirrors load-test.py so a quality-alone run reports model-not-ready
# the same way a benchmark run does, instead of a generic error (exit 1).
EXIT_MODEL_NOT_READY = 3


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
    # The per-request reasoning DEFAULT this run used, from the config's
    # model.chat_template_kwargs (e.g. {"enable_thinking": false}). Empty dict
    # means the run sent no chat-template override, so the SERVER default
    # (--default-chat-template-kwargs) governed thinking.
    chat_template_kwargs: dict[str, Any]
    # OBSERVED at inference time by probing the live server (see _probe_reasoning):
    # "ON"/"OFF"/"-". Since the quality config sends no thinking override, this is
    # what the SERVER default actually produced on this run.
    reasoning_observed: str
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
    # Full per-QID confusion + rates over the ENTIRE dataset, keyed by QID:
    #   {QID: {tp, fp, fn, tn, precision, recall, f1, accuracy, n, ...}}
    # Computed with the same _binary_metrics used for the overall matrix, so the
    # report can show accuracy/precision/recall/TP.. per QID without depending on
    # the (capped) sample dump.
    per_question_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    # Replies that hit the token budget (finish_reason == "length"). A high count
    # means the reasoning budget is too small for the model to reach a verdict —
    # an INTEGRATION signal, not a quality signal. Raise
    # decoding.reasoning_max_tokens before interpreting the accuracy numbers.
    n_truncated: int = 0
    # tp + fp + fn + tn + unparseable_neg + unscored_other. Must equal n_total;
    # published so the confusion matrix can be reconciled against the row count.
    confusion_total: int = 0
    # Rows whose verdict came from the constrained retry after the first attempt
    # ran out of tokens. These ARE scored; the count is published because those
    # verdicts were produced without the model's full reasoning.
    n_fallback: int = 0
    # Count of rows by how the verdict was obtained (answer / reasoning /
    # conclusion / loose / retry / none). Makes it visible when most verdicts did
    # NOT come from the model's own completed reply.
    verdict_sources: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Dataset loading (CSV or JSONL) — returns list of (data_id, prompt, gold, question)
# ---------------------------------------------------------------------------
def load_dataset(path: str, cfg: dict[str, Any], sheet: str = "") -> list[dict[str, str]]:
    dcol = cfg["dataset"]
    text_f = dcol["text_field"]
    label_f = dcol["label_field"]
    q_f = dcol.get("question_field", "")
    # Column holding the stable question id (QID). When present it is the
    # PREFERRED grouping key for per-question metrics, because it is stable across
    # the two dataset sheets (Test-ORG / Test-XL) whereas the question text can be
    # reworded. Falls back to the question text when absent (older datasets).
    qid_f = dcol.get("question_id_field", "question_id")
    # Column holding the raw call transcript (context). When set, the prompt sent
    # to the model is ASSEMBLED from this transcript plus the QID question text
    # (the agent instruction), instead of using a prebuilt input_prompt column.
    transcript_f = dcol.get("transcript_field", "")
    # Sheet-name overrides (xlsx). ``sheet`` (arg) selects the transcript sheet to
    # score; ``qid_sheet`` is the lookup table of question_id -> question text.
    transcript_sheet = sheet or dcol.get("sheet", "")
    qid_sheet = dcol.get("qid_sheet", "QIDs")
    qid_id_col = dcol.get("qid_id_column", "Question Id")
    qid_text_col = dcol.get("qid_text_column", "Question Text")
    # Template used to build the prompt from transcript + question. {question} is
    # the QID rubric text (instruction); {transcript} is the call context.
    prompt_template = dcol.get("prompt_template") or (
        "Evaluate the human agent against the given question and transcript.\n"
        "=== QUESTION ===\n{question}\n=== TRANSCRIPT ===\n{transcript}"
    )
    rows: list[dict[str, str]] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    is_xlsx = p.suffix.lower() in (".xlsx", ".xlsm")

    # QID -> question-text lookup, populated from the QIDs sheet for xlsx sources.
    # Empty for csv/jsonl, where the question travels with each row.
    qid_map: dict[str, str] = _load_qid_map(p, qid_sheet, qid_id_col, qid_text_col) if is_xlsx else {}

    def _row(d: dict[str, Any], i: int) -> dict[str, str]:
        lower_d = {str(k).strip().lower(): v for k, v in d.items() if k is not None}
        gold_val = (
            d.get(label_f) or lower_d.get(label_f.lower()) or 
            lower_d.get("answer") or lower_d.get("gold") or lower_d.get("ground_truth") or lower_d.get("label") or ""
        )
        q_val = d.get(q_f) or lower_d.get(q_f.lower()) or ""
        qid_val = d.get(qid_f) or lower_d.get(qid_f.lower()) or ""
        transcript_val = (d.get(transcript_f) or lower_d.get(transcript_f.lower()) or "") if transcript_f else ""

        gold_str = str(gold_val).strip()
        if gold_str.lower() == "yes":
            gold_str = "Yes"
        elif gold_str.lower() == "no":
            gold_str = "No"

        q_str = str(q_val).strip()
        qid_str = str(qid_val).strip()
        transcript_str = str(transcript_val).strip()

        # Resolve the rubric question. Prefer the QID lookup (authoritative full
        # text incl. sub-criteria); fall back to the row's own question column.
        if qid_str and qid_map.get(qid_str):
            q_str = qid_map[qid_str].strip()

        # Assemble the model prompt.
        #   1. transcript + QID question  -> build from the template (new xlsx)
        #   2. otherwise                   -> the prebuilt input_prompt column
        if transcript_str and q_str:
            prompt_str = prompt_template.format(question=q_str, transcript=transcript_str).strip()
        else:
            prompt_val = (
                d.get(text_f) or lower_d.get(text_f.lower()) or
                lower_d.get("input_prompt") or lower_d.get("question") or lower_d.get("prompt") or ""
            )
            prompt_str = str(prompt_val).strip()

        # If we still have no separate question, recover it from the prompt text
        # so the per-question grouping and report label are populated.
        if (not q_str or q_str == prompt_str) and "Question:" in prompt_str:
            q_match = re.search(r"Question:\s*(.*?)(?:\n|Sub-criteria:|$)", prompt_str, re.IGNORECASE | re.DOTALL)
            if q_match:
                q_str = q_match.group(1).strip()

        # Grouping key for per-question metrics. Prefer the stable QID (e.g.
        # "QID_1"); it is identical across Test-ORG and Test-XL, so the two
        # sheets' per-question breakdowns line up. Fall back to a compact slice
        # of the question text when no QID column is present.
        group_key = qid_str or (q_str[:60] if q_str else "all")

        return {
            "data_id": str(d.get("data_id") or lower_d.get("data_id") or i),
            "prompt": prompt_str,
            "gold": gold_str,
            # Key used to GROUP per-question F1. It is a label, not the model's
            # input: the QID when available, else a 60-char question stub.
            "question": group_key,
            # Stable question id on its own, carried through to the sample dump.
            "question_id": qid_str or "",
            # Full rubric question, carried through to the sample dump so the
            # report shows the criterion in full rather than a 60-char stub.
            # This is the QA rubric, not customer data.
            "question_full": q_str or "all",
            # Size of the real prompt sent to the model (transcript + question +
            # sub-criteria). Recorded so a reader can see how large the actual
            # input was WITHOUT persisting the transcript itself.
            "prompt_chars": str(len(prompt_str)),
        }

    if is_xlsx:
        rows = [_row(d, i) for i, d in enumerate(_read_xlsx_sheet(p, transcript_sheet))]
    else:
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
# xlsx readers (openpyxl). Auto-installed the same way report_generator does, so
# the in-cluster pod needs no extra baked-in dependency.
# ---------------------------------------------------------------------------
def _import_openpyxl():  # type: ignore[no-untyped-def]
    try:
        import openpyxl  # noqa: PLC0415
    except ImportError:
        import subprocess  # noqa: PLC0415
        logger.info("openpyxl not present — installing it to read the .xlsx dataset.")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "openpyxl"], check=False)
        import openpyxl  # noqa: PLC0415
    return openpyxl


def _read_xlsx_sheet(path: Path, sheet: str) -> list[dict[str, Any]]:
    """Read one worksheet into a list of {header: value} dicts.

    The first row is the header. ``sheet`` selects the worksheet by name; when
    empty the workbook's first sheet is used. read_only + values_only keeps memory
    flat for the large XL transcripts.

    data_only is FALSE on purpose. These transcripts frequently start with
    "===== CALL n =====", which openpyxl classifies as a formula cell; with
    data_only=True such a cell returns None (no cached result) and the transcript
    would be dropped. Reading raw values returns the literal text instead. The
    dataset carries no genuine spreadsheet formulas (Word/Token counts are
    literals), so nothing computed is lost.
    """
    openpyxl = _import_openpyxl()
    wb = openpyxl.load_workbook(path, read_only=True, data_only=False)
    try:
        if sheet:
            if sheet not in wb.sheetnames:
                raise KeyError(
                    f"Sheet '{sheet}' not found in {path.name}. "
                    f"Available: {', '.join(wb.sheetnames)}"
                )
            ws = wb[sheet]
        else:
            ws = wb[wb.sheetnames[0]]
        rows_iter = ws.iter_rows(values_only=True)
        try:
            header = [str(h).strip() if h is not None else "" for h in next(rows_iter)]
        except StopIteration:
            return []
        out: list[dict[str, Any]] = []
        for values in rows_iter:
            if values is None or all(v is None for v in values):
                continue
            record = {header[i]: values[i] for i in range(min(len(header), len(values)))}
            out.append(record)
        return out
    finally:
        wb.close()


def _load_qid_map(path: Path, sheet: str, id_col: str, text_col: str) -> dict[str, str]:
    """Build a ``question_id -> question_text`` map from the QIDs sheet.

    Returns an empty map (and logs a warning) when the sheet is absent, so a
    workbook without a QIDs sheet degrades to using each row's own question
    column rather than failing the whole run.
    """
    openpyxl = _import_openpyxl()
    wb = openpyxl.load_workbook(path, read_only=True, data_only=False)
    try:
        if sheet not in wb.sheetnames:
            logger.warning("QID sheet '%s' not found in %s; using per-row questions.", sheet, path.name)
            return {}
    finally:
        wb.close()

    mapping: dict[str, str] = {}
    for record in _read_xlsx_sheet(path, sheet):
        lower = {str(k).strip().lower(): v for k, v in record.items() if k}
        qid = record.get(id_col) or lower.get(id_col.lower()) or lower.get("question_id") or ""
        text = record.get(text_col) or lower.get(text_col.lower()) or lower.get("question_text") or ""
        qid_s = str(qid).strip()
        text_s = str(text).strip()
        if qid_s and text_s:
            mapping[qid_s] = text_s
    logger.info("Loaded %d QID -> question mappings from sheet '%s'.", len(mapping), sheet)
    return mapping



# ---------------------------------------------------------------------------
# One model call (vLLM OpenAI chat completions, stdlib HTTP)
# ---------------------------------------------------------------------------
# Per-request HTTP timeout (seconds) for a single model call. Set once from the
# --request-timeout CLI arg in _run(). Large/slow models with reasoning ON (e.g.
# gemma-4-31b on L4 TP=4) can take well over 2 minutes per request, so the old
# hardcoded 120s cut genuinely-working requests off as false TimeoutErrors.
_REQUEST_TIMEOUT_S = 300


def _post_chat(endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{endpoint}/v1/chat/completions", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:  # noqa: S310
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
    dcfg: dict[str, Any], mcfg: dict[str, Any], guided_ok: bool = False,
) -> tuple[str, str, str, str, str]:
    """Classify one row. Returns (label, answer, reasoning, finish_reason, source).

    A reasoning model can spend its whole token budget thinking and never state a
    verdict. When that happens the row is retried ONCE with generation constrained
    to a bare label, so a verbose model still produces a scoreable answer instead
    of being dropped as unparseable.
    """
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

    # Optional chat-template passthrough. Qwen3 enables thinking by default and it
    # is turned off per request with {"enable_thinking": false}; leaving this unset
    # keeps the request identical to before.
    ctk = mcfg.get("chat_template_kwargs")
    if isinstance(ctk, dict) and ctk:
        body["chat_template_kwargs"] = dict(ctk)

    if mode == "guided_choice":
        body["guided_choice"] = labels               # vLLM structured output
        body["max_tokens"] = int(dcfg.get("max_tokens", 8))
    else:  # free_parse — let reasoning models think, then parse
        body["max_tokens"] = int(dcfg.get("reasoning_max_tokens", 1024))

    out = _post_chat(endpoint, body)
    choice = out["choices"][0]
    finish_reason = str(choice.get("finish_reason") or "")
    msg = choice.get("message") or {}
    label, answer_text, reasoning_text, source = _extract_verdict(msg, labels, finish_reason)

    # ---- Constrained retry ---------------------------------------------------
    # Only when the first attempt produced no verdict AND it ran out of tokens.
    # A model that answered something unparseable for another reason is left
    # alone, so this never masks a genuine problem.
    wants_fallback = bool(dcfg.get("truncation_fallback", True))
    if label == "UNKNOWN" and finish_reason == "length" and mode != "guided_choice" and wants_fallback:
        retry_label, retry_answer = _forced_label_retry(
            endpoint=endpoint, served_model=served_model, messages=messages,
            labels=labels, dcfg=dcfg, mcfg=mcfg, guided_ok=guided_ok,
        )
        if retry_label != "UNKNOWN":
            # Preserve the first attempt's text as the reasoning evidence — it
            # shows what the model was doing when it ran out of budget. Models
            # that emit thinking as plain prose (no <think> tags) have it in
            # answer_text, so fall back to that when reasoning_text is empty.
            preserved = reasoning_text or answer_text
            return retry_label, retry_answer, preserved, finish_reason, "retry"

    return label, answer_text, reasoning_text, finish_reason, source


def _forced_label_retry(
    *, endpoint: str, served_model: str, messages: list[dict[str, str]],
    labels: list[str], dcfg: dict[str, Any], mcfg: dict[str, Any], guided_ok: bool,
) -> tuple[str, str]:
    """Ask again with generation constrained to a bare label. Returns (label, text).

    Two mechanisms, strongest first:
      1. ``guided_choice`` — vLLM structured output makes a label the only legal
         generation, so the model cannot emit chain-of-thought at all.
      2. an explicit one-word instruction plus ``enable_thinking: false``, for
         builds where guided decoding is unavailable.
    """
    nudge = f"Answer now with exactly one word and nothing else: {' or '.join(labels)}."
    base: dict[str, Any] = {
        "model": served_model,
        "messages": [*messages, {"role": "user", "content": nudge}],
        "temperature": float(dcfg.get("temperature", 0.0)),
        "seed": int(dcfg.get("seed", 42)),
        "max_tokens": int(dcfg.get("fallback_max_tokens", 16)),
    }

    attempts: list[dict[str, Any]] = []
    if guided_ok:
        attempts.append({**base, "guided_choice": labels})
    # Thinking-off attempt. Unknown chat-template kwargs are ignored by templates
    # that do not declare them, and the call is guarded, so this is safe to try.
    ctk = dict(mcfg.get("chat_template_kwargs") or {})
    ctk.setdefault("enable_thinking", False)
    attempts.append({**base, "chat_template_kwargs": ctk})
    attempts.append(base)  # plain nudge, last resort

    for attempt in attempts:
        try:
            out = _post_chat(endpoint, attempt)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Constrained retry attempt failed: %s", type(exc).__name__)
            continue
        choice = out["choices"][0]
        fr = str(choice.get("finish_reason") or "")
        label, answer, _, _ = _extract_verdict(choice.get("message") or {}, labels, fr)
        if label != "UNKNOWN":
            return label, answer
    return "UNKNOWN", ""


# Chain-of-thought wrappers emitted by reasoning models (Qwen3 <think>, etc.).
# _THINK_BLOCK_RE matches a COMPLETE block; _THINK_OPEN_RE catches a block that
# was cut off mid-thought because the token budget ran out.
_THINK_BLOCK_RE = re.compile(r"<\s*(think|thinking|reasoning)\s*>.*?<\s*/\s*\1\s*>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<\s*(think|thinking|reasoning)\s*>", re.IGNORECASE)

# Size of the closing region scanned by the "loose" parse tier. Wide enough for a
# verdict plus a short sign-off sentence, narrow enough to exclude an instruction
# echo sitting at the top of a long chain of thought.
_LOOSE_TAIL_CHARS = 300


def _conclusion_patterns(labels: list[str]) -> list[re.Pattern[str]]:
    """Patterns that capture a verdict the model EXPLICITLY declared.

    Reasoning models routinely reach a decision and then burn the rest of their
    token budget deliberating about output formatting, so the reply is truncated
    even though the verdict was already stated ("Conclusion: ... the word is
    Yes", "Therefore, the answer should be Yes", "Verdict: Yes"). Recovering that
    statement keeps the model's own judgement instead of discarding it.

    Each pattern requires a decision keyword next to the label, which is what
    keeps an echoed instruction ("end with exactly one word: Yes or No") from
    matching.
    """
    alt = "|".join(re.escape(x) for x in labels)
    raw = (
        rf"final\s+(?:decision|word|answer|verdict)\s*(?:is)?\s*[:\-]?\s*\**\s*[\"']?({alt})\b",
        rf"(?:the\s+)?(?:word|label)\s+is\s*\**\s*[\"']?({alt})\b",
        rf"answer\s+(?:to\s+[^.\n]{{0,40}}?\s+)?(?:is|should\s+be|would\s+be)\s*\**\s*[\"']?({alt})\b",
        rf"\b(?:verdict|decision|conclusion|result)\s*[:\-]\s*\**\s*[\"']?({alt})\b",
    )
    return [re.compile(p, re.IGNORECASE) for p in raw]


def _conclusion_label(text: str, labels: list[str]) -> str:
    """The LAST explicitly declared verdict in ``text``, or UNKNOWN."""
    if not text:
        return "UNKNOWN"
    best_pos = -1
    best_hit = ""
    for pattern in _conclusion_patterns(labels):
        for match in pattern.finditer(text):
            if match.start() > best_pos:
                best_pos = match.start()
                best_hit = match.group(1)
    if best_pos < 0:
        return "UNKNOWN"
    for lab in labels:
        if lab.lower() == best_hit.lower():
            return lab
    return "UNKNOWN"


def _split_think(text: str) -> tuple[str, str]:
    """Split a reply into (visible_answer, chain_of_thought).

    Complete ``<think>...</think>`` blocks move to the thought side. For a
    TRUNCATED block (an opening tag with no closing tag — what happens when the
    model runs out of reasoning budget) everything from the opening tag onward is
    thought, leaving NO verdict on the answer side. The row is then counted as
    unparseable instead of silently inheriting a stray label from the reasoning.

    Keeping the two sides apart is also what makes a sample dump auditable: a
    reviewer can see what the model ANSWERED separately from what it was thinking.
    """
    if not text:
        return "", ""
    thoughts = [m.group(0) for m in _THINK_BLOCK_RE.finditer(text)]
    visible = _THINK_BLOCK_RE.sub(" ", text)
    open_tag = _THINK_OPEN_RE.search(visible)
    if open_tag:
        thoughts.append(visible[open_tag.start():])
        visible = visible[: open_tag.start()]
    return visible.strip(), " ".join(t.strip() for t in thoughts if t.strip()).strip()


def _strip_think(text: str) -> str:
    """The visible answer side of a reply, with chain-of-thought removed."""
    return _split_think(text)[0]


def _extract_verdict(
    msg: dict[str, Any], labels: list[str], finish_reason: str
) -> tuple[str, str, str, str]:
    """Resolve the verdict, returning (label, answer_text, reasoning_text, source).

    ``source`` records HOW the verdict was obtained, so a run can be audited:
    ``answer`` (the reply's answer channel), ``reasoning`` (a reasoning channel),
    ``conclusion`` (an explicitly declared verdict inside truncated reasoning),
    ``loose`` (last mention in the closing region), or ``none``.

    The two text values are what the sample dump records: ``answer_text`` is the
    reply with chain-of-thought removed (what the model actually answered), and
    ``reasoning_text`` is the thinking. Reported separately because a single glued
    string makes it impossible to tell a verdict from a thought.

    Precedence matters. vLLM can place the answer in ``content`` and the chain of
    thought in ``reasoning_content``; concatenating them and taking the last label
    mentioned lets unfinished reasoning override an explicit answer, which shows
    up as an implausibly low recall on every hardware type at once.

      1. the ANSWER channel (``content``), CoT stripped, strict match
      2. the reasoning channels, strict match (final line only)
      3. loose last-mention scan, ONLY when the reply was not truncated

    A truncated reply with no explicit verdict returns UNKNOWN. That is the
    honest outcome: it is counted as unparseable and reported, rather than being
    scored as a confident wrong answer.
    """
    raw_content = str(msg.get("content") or "")
    reasoning_parts = [str(msg.get(k) or "") for k in ("reasoning_content", "reasoning", "text")]
    truncated = finish_reason == "length"

    answer_text, content_thought = _split_think(raw_content)
    reasoning_text = " ".join(
        p.strip() for p in [content_thought, *reasoning_parts] if p and p.strip()
    ).strip()

    def _result(label: str, source: str) -> tuple[str, str, str, str]:
        return label, answer_text, reasoning_text, source

    # A truncated reply has an incomplete final line, so a label sitting inside
    # that line is unfinished thought, not a verdict. Only a standalone label is
    # accepted in that case.
    strict_mode = "exact" if truncated else "tail"

    label = _parse_label(answer_text, labels, mode=strict_mode)
    if label != "UNKNOWN":
        return _result(label, "answer")

    for part in reasoning_parts:
        label = _parse_label(_strip_think(part), labels, mode=strict_mode)
        if label != "UNKNOWN":
            return _result(label, "reasoning")

    # An explicitly declared verdict counts even when the reply was cut off: the
    # model stated its decision and then ran out of budget. Checked before any
    # constrained retry so the model's own judgement is not thrown away.
    for text in (raw_content, *reasoning_parts):
        label = _conclusion_label(text, labels)
        if label != "UNKNOWN":
            return _result(label, "conclusion")

    if not truncated:
        label = _parse_label(answer_text, labels, mode="loose")
        if label != "UNKNOWN":
            return _result(label, "loose")
        for part in reasoning_parts:
            label = _parse_label(_strip_think(part), labels, mode="loose")
            if label != "UNKNOWN":
                return _result(label, "loose")

    return _result("UNKNOWN", "none")


def _parse_label(content: str, labels: list[str], *, mode: str = "loose") -> str:
    """Extract a label from model text. ``mode`` controls how much is trusted.

    "exact" : the whole reply IS a label, or the final non-empty line is a
              standalone label ("Yes", "**Yes**", "Yes."). Used for TRUNCATED
              replies, where a label buried in an unfinished sentence is thought
              in progress rather than a verdict.
    "tail"  : also accepts the last label mentioned WITHIN the final non-empty
              line ("Answer: Yes"). Used for complete replies.
    "loose" : also accepts the last label mentioned in the CLOSING region of the
              reply. Last resort for complete replies only. The window matters:
              reasoning models often restate the instruction ("end with exactly
              one word: Yes or No") near the top, and scanning the whole reply
              would return that echo — the last label in it is "No" — for every
              row that never reached a real verdict.

    Matching is always on WORD BOUNDARIES. A bare substring search would read
    "No" out of "cannot"/"not"/"none" and label a perfectly reasonable reply as a
    negative verdict, so it is deliberately not attempted.
    """
    c = (content or "").strip()
    if not c:
        return "UNKNOWN"

    def _match(value: str) -> str | None:
        for lab in labels:
            if value == lab.lower():
                return lab
        return None

    # Tier 1 — the whole reply is the label (guided_choice, or a one-word answer).
    hit = _match(c.lower())
    if hit:
        return hit

    pattern = r"\b(" + "|".join(re.escape(x) for x in labels) + r")\b"
    lines = [ln.strip() for ln in c.splitlines() if ln.strip()]
    tail = lines[-1] if lines else ""

    # Tier 2 — the final line is a standalone label, ignoring markdown/punctuation.
    if tail:
        hit = _match(re.sub(r"[^A-Za-z]+", "", tail).lower())
        if hit:
            return hit

    if mode == "exact":
        return "UNKNOWN"

    # Tier 3 — a label mentioned inside the final line ("Answer: Yes").
    if tail:
        tail_matches = re.findall(pattern, tail, re.IGNORECASE)
        if tail_matches:
            hit = _match(tail_matches[-1].lower())
            if hit:
                return hit

    if mode == "tail":
        return "UNKNOWN"

    # Tier 4 — last label in the closing region only (word boundaries only).
    matches = re.findall(pattern, c[-_LOOSE_TAIL_CHARS:], re.IGNORECASE)
    if matches:
        hit = _match(matches[-1].lower())
        if hit:
            return hit
    return "UNKNOWN"



# ---------------------------------------------------------------------------
# Metrics (binary; no sklearn dependency)
# ---------------------------------------------------------------------------
def _binary_metrics(preds: list[str], golds: list[str], pos: str) -> dict[str, float]:
    """Binary metrics plus the leftover buckets needed to reconcile the matrix.

    tp/fp/fn/tn keep their exact original meaning, so precision, recall, F1 and
    accuracy are unchanged. What is added is visibility: a row with a negative
    gold and an unparseable prediction lands in NEITHER tp/fp/fn/tn, so a bare
    2x2 silently sums to less than the row count. Those rows are now counted in
    ``unparseable_neg`` (and anything else odd in ``unscored_other``) so that

        tp + fp + fn + tn + unparseable_neg + unscored_other == len(preds)

    ``unparseable_pos`` is a SUBSET of fn, reported separately to show how much
    of the missed-positive count is a parsing failure rather than a wrong answer.
    """
    tp = fp = fn = tn = 0
    unparseable_pos = unparseable_neg = unscored_other = 0
    for p, g in zip(preds, golds):
        if g == pos and p == pos:
            tp += 1
        elif g != pos and p == pos:
            fp += 1
        elif g == pos and p != pos:
            fn += 1                    # actual-positive missed (includes UNKNOWN)
            if p == "UNKNOWN":
                unparseable_pos += 1   # subset of fn: missed because unparseable
        elif g != pos and p == g:
            tn += 1                    # correct negative (pred == actual negative label)
        elif p == "UNKNOWN":
            unparseable_neg += 1       # negative gold, no parseable verdict
        else:
            unscored_other += 1        # e.g. gold outside the label set
        # Unparseable rows stay out of TP/TN, so accuracy = (TP+TN)/N still counts
        # them as wrong, and positive-class precision/recall are unaffected
        # (no positive was predicted).
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(len(preds), 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "unparseable_pos": unparseable_pos, "unparseable_neg": unparseable_neg,
            "unscored_other": unscored_other,
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


def _f1_for(preds: list[str], golds: list[str], pos: str) -> float:
    return _binary_metrics(preds, golds, pos)["f1"]


# ---------------------------------------------------------------------------
# Inference-time reasoning probe
# ---------------------------------------------------------------------------
def _probe_reasoning(endpoint: str, served_model: str, timeout: int = 30) -> str:
    """Observe whether the SERVER actually reasons at inference time.

    The quality config sends no chat_template_kwargs, so thinking is decided by
    the vLLM server default (--default-chat-template-kwargs). Rather than infer
    that from config, this asks the live server one throwaway question and looks
    at the reply for chain-of-thought: a non-empty ``reasoning`` /
    ``reasoning_content`` field, or a ``<think>`` block in the content.

    Returns "ON", "OFF", or "-" (probe failed / inconclusive). Recorded on the
    result row so the report shows what the run actually did.
    """
    body = {
        "model": served_model,
        "messages": [{"role": "user", "content": "In one short sentence, is 17 a prime number?"}],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    try:
        out = _post_chat(endpoint, body)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reasoning probe failed (%s); recording reasoning_observed as '-'.",
                       type(exc).__name__)
        return "-"

    try:
        msg = (out.get("choices") or [{}])[0].get("message") or {}
    except (AttributeError, IndexError, TypeError):
        return "-"

    reasoning_field = str(msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
    content = str(msg.get("content") or "")
    has_think_block = bool(_THINK_OPEN_RE.search(content))
    observed = "ON" if (reasoning_field or has_think_block) else "OFF"
    logger.info("Reasoning probe: observed=%s (reasoning_field=%s, think_block=%s)",
                observed, bool(reasoning_field), has_think_block)
    return observed


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
    # Apply the per-request timeout for every model call this run (set once).
    global _REQUEST_TIMEOUT_S
    _REQUEST_TIMEOUT_S = int(getattr(args, "request_timeout", _REQUEST_TIMEOUT_S))
    logger.info("Per-request model-call timeout: %ds", _REQUEST_TIMEOUT_S)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    labels: list[str] = [str(x) for x in cfg["labels"]]
    pos: str = str(cfg["positive_label"])
    dcfg = cfg.get("decoding", {})
    mcfg = cfg.get("model", {})
    served_model = args.model or mcfg.get("served_name") or "model"
    concurrency = int(cfg.get("run", {}).get("concurrency", 8))
    endpoint = args.endpoint.rstrip("/")

    rows = load_dataset(args.dataset, cfg, sheet=getattr(args, "sheet", "") or "")
    logger.info("Loaded %d rows; labels=%s positive=%s mode=%s sheet=%s",
                len(rows), labels, pos, dcfg.get("mode"), getattr(args, "sheet", "") or "(default)")

    try:
        _wait_for_vllm(endpoint, timeout_s=args.wait_timeout)
    except TimeoutError as exc:
        # Endpoint never came up in the wait budget. Not a quality result — the
        # model never served. Exit with the dedicated code so the reason is
        # legible (infra/capacity), not a generic error.
        logger.error(
            "vLLM endpoint never became ready: %s. Skipping quality eval (model-not-ready). "
            "This is an infrastructure/capacity issue, not a quality failure.",
            exc,
        )
        raise SystemExit(EXIT_MODEL_NOT_READY) from exc

    # Capability probe, run once regardless of mode: guided decoding is both the
    # primary mechanism for guided_choice mode AND the strongest constrained-retry
    # mechanism for free_parse mode, so its availability is always worth knowing.
    guided_ok = _guided_supported(endpoint, served_model, labels)
    if str(dcfg.get("mode", "guided_choice")) == "guided_choice" and not guided_ok:
        dcfg = dict(dcfg)
        dcfg["mode"] = "free_parse"
        logger.warning("Decoding mode fell back to free_parse for this run.")
    logger.info(
        "guided_choice supported=%s; constrained retry on truncation=%s",
        guided_ok, bool(dcfg.get("truncation_fallback", True)),
    )

    # Observe reasoning at inference time. The quality config sends no thinking
    # override, so this captures what the SERVER default actually did on this run.
    reasoning_observed = _probe_reasoning(endpoint, served_model)

    started_at = datetime.now(tz=timezone.utc).isoformat()

    # Optional per-row sample capture (opt-in via --dump-samples). Captures the
    # MODEL OUTPUT only — never the input transcript — capped at
    # --max-dump-samples rows to keep the file tiny (customer-data policy).
    dump_samples = bool(getattr(args, "dump_samples", False))
    max_dump = int(getattr(args, "max_dump_samples", 200))
    # Including the input prompt means persisting call transcripts, so it is a
    # separate explicit opt-in and only takes effect alongside --dump-samples.
    dump_inputs = bool(getattr(args, "dump_inputs", False)) and dump_samples
    if getattr(args, "dump_inputs", False) and not dump_samples:
        logger.warning("--dump-inputs ignored: it requires --dump-samples.")
    if dump_inputs:
        logger.warning(
            "--dump-inputs is set: sample rows will CONTAIN the input prompts "
            "(call transcripts). Handle the samples file accordingly."
        )

    def _worker(row: dict[str, str]) -> dict[str, str]:
        answer_text = reasoning_text = ""
        finish_reason = ""
        source = "none"
        try:
            pred, answer_text, reasoning_text, finish_reason, source = _classify_one(
                endpoint=endpoint, served_model=served_model,
                prompt=row["prompt"], labels=labels, dcfg=dcfg, mcfg=mcfg,
                guided_ok=guided_ok,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Request failed for data_id=%s: %s", row["data_id"], type(exc).__name__)
            pred = "UNKNOWN"
            finish_reason = "error"
            source = "error"
            # Capture the REAL error reason so the report's response column shows
            # WHY this row has no verdict (e.g. the vLLM endpoint dropped/timed
            # out mid-run) instead of a blank '-'. This is an integration signal,
            # not model output. Message is truncated to keep sample rows compact.
            _msg = str(exc).strip().replace("\n", " ")
            if len(_msg) > 300:
                _msg = _msg[:300] + "..."
            answer_text = f"ERROR: request to model endpoint failed ({type(exc).__name__}): {_msg}"
            reasoning_text = answer_text
        # finish_reason is always kept (a short string): it is what distinguishes
        # "the model answered wrongly" from "the model never got to answer".
        out = {"data_id": row["data_id"], "gold": row["gold"],
               "pred": pred, "question": row["question"],
               "question_id": row.get("question_id", ""),
               "finish_reason": finish_reason,
               # How the verdict was obtained; "retry" means the model's own
               # reply never produced one.
               "verdict_source": source,
               "fallback": "true" if source == "retry" else ""}
        if dump_samples:
            out["answer"] = answer_text
            out["reasoning"] = reasoning_text
            out["question_full"] = row.get("question_full", row["question"])
            out["prompt_chars"] = row.get("prompt_chars", "")
        return out

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
    n_truncated = sum(1 for r in results if r.get("finish_reason") == "length")
    n_fallback = sum(1 for r in results if r.get("fallback"))
    # Verdict provenance. A run dominated by "retry" is measuring the model with
    # its reasoning discarded, which is not comparable to a run where the model
    # answered from its own completed reasoning.
    verdict_sources: dict[str, int] = {}
    for r in results:
        key = str(r.get("verdict_source") or "unknown")
        verdict_sources[key] = verdict_sources.get(key, 0) + 1

    m = _binary_metrics(preds, golds, pos)
    # macro-F1 over both classes
    other = next((x for x in labels if x != pos), pos)
    macro_f1 = (_f1_for(preds, golds, pos) + _f1_for(preds, golds, other)) / 2.0

    # per-question metrics over the FULL dataset (F1 + N kept for back-compat,
    # plus the complete confusion/rate set so the report needs no sample dump).
    per_q_f1: dict[str, float] = {}
    per_q_n: dict[str, int] = {}
    per_q_metrics: dict[str, dict[str, float]] = {}
    for q in sorted({r["question"] for r in results}):
        qp = [r["pred"] for r in results if r["question"] == q]
        qg = [r["gold"] for r in results if r["question"] == q]
        qm = _binary_metrics(qp, qg, pos)
        per_q_f1[q] = round(qm["f1"], 4)
        per_q_n[q] = len(qp)
        per_q_metrics[q] = {
            "n": len(qp),
            "tp": int(qm["tp"]), "fp": int(qm["fp"]),
            "fn": int(qm["fn"]), "tn": int(qm["tn"]),
            "unparseable_pos": int(qm["unparseable_pos"]),
            "unparseable_neg": int(qm["unparseable_neg"]),
            "unscored_other": int(qm["unscored_other"]),
            "precision": round(qm["precision"], 4),
            "recall": round(qm["recall"], 4),
            "f1": round(qm["f1"], 4),
            "accuracy": round(qm["accuracy"], 4),
        }

    result = QualityResult(
        eval_id=cfg.get("eval_id", "autoqa"),
        served_model=served_model,
        quantization=args.quantization,
        hardware=args.hardware,
        # Include the sheet name when one was selected, so Test-ORG and Test-XL
        # results are self-describing and never conflated in the report.
        dataset_version=(
            f"{cfg.get('dataset', {}).get('version', 'autoqa_v1')}:{args.sheet}"
            if getattr(args, "sheet", "")
            else cfg.get("dataset", {}).get("version", "autoqa_v1")
        ),
        decoding_mode=str(dcfg.get("mode", "guided_choice")),
        reasoning_effort=str(mcfg.get("reasoning_effort", "") or "n/a"),
        # Configured chat-template default ({} = none set -> server default) and
        # the inference-time observed reasoning behaviour.
        chat_template_kwargs=dict(mcfg.get("chat_template_kwargs") or {}),
        reasoning_observed=reasoning_observed,
        n_total=len(results),
        n_scored=len(results) - n_unparseable,
        n_unparseable=n_unparseable,
        accuracy=round(m["accuracy"], 4),
        precision=round(m["precision"], 4),
        recall=round(m["recall"], 4),
        f1=round(m["f1"], 4),
        macro_f1=round(macro_f1, 4),
        positive_label=pos,
        # unparseable_pos is a subset of fn; unparseable_neg + unscored_other are
        # the rows a bare 2x2 would drop. Together they reconcile to n_total.
        confusion={
            "tp": int(m["tp"]), "fp": int(m["fp"]), "fn": int(m["fn"]), "tn": int(m["tn"]),
            "unparseable_pos": int(m["unparseable_pos"]),
            "unparseable_neg": int(m["unparseable_neg"]),
            "unscored_other": int(m["unscored_other"]),
        },
        per_question_f1=per_q_f1,
        per_question_n=per_q_n,
        per_question_metrics=per_q_metrics,
        started_at=started_at,
        ended_at=ended_at,
        n_truncated=n_truncated,
        confusion_total=int(
            m["tp"] + m["fp"] + m["fn"] + m["tn"] + m["unparseable_neg"] + m["unscored_other"]
        ),
        n_fallback=n_fallback,
        verdict_sources=verdict_sources,
    )

    if result.confusion_total != result.n_total:
        # Should be impossible; surfaced loudly rather than shipping a matrix
        # that does not add up to the row count.
        logger.warning(
            "Confusion matrix does not reconcile: total=%d n_total=%d",
            result.confusion_total, result.n_total,
        )
    if n_truncated:
        logger.warning(
            "%d/%d replies hit the token budget (finish_reason=length); %d were "
            "recovered by the constrained retry. Raise decoding.reasoning_max_tokens "
            "(or set model.chat_template_kwargs.enable_thinking=false) so the model "
            "can reach a verdict on its own.",
            n_truncated, result.n_total, n_fallback,
        )
    if result.n_unparseable == result.n_total and result.n_total:
        logger.error(
            "NO row produced a usable verdict. This is an integration failure, not "
            "a model-quality result — do not report these metrics."
        )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"quality-{result.eval_id}-{served_model.replace('/', '_')}.jsonl"
    with out_file.open("w") as f:
        f.write(json.dumps(asdict(result)) + "\n")
    logger.info("Wrote %s", out_file)

    # Optional per-row sample dump (opt-in). Writes at most --max-dump-samples
    # rows of {data_id, question, gold, pred, output} — MODEL OUTPUT only, never
    # the input transcript. Tiny by design (~1-2 MB for the full 1200-row set at
    # the free_parse 256-token cap), so no storage/eviction risk.
    if dump_samples:
        samples_file = out_dir / f"samples-{result.eval_id}-{served_model.replace('/', '_')}.jsonl"
        written = 0
        with samples_file.open("w") as f:
            # ThreadPoolExecutor.map preserves input order, so results[i] is the
            # verdict for rows[i]. Zipping gives access to the source prompt
            # without holding a second copy of every transcript in memory.
            for r, src in zip(results, rows):
                if written >= max_dump:
                    break
                gold = r.get("gold", "")
                pred = r.get("pred", "")
                fr = r.get("finish_reason", "")
                # Each row carries its own model/hardware context so the report
                # can render samples without re-deriving anything from filenames.
                row_out: dict[str, Any] = {
                    "eval_id": result.eval_id,
                    "served_model": served_model,
                    "hardware": args.hardware,
                    "quantization": args.quantization,
                    "data_id": r.get("data_id", ""),
                    # Full rubric question, not the grouping key.
                    "question": r.get("question_full") or r.get("question", ""),
                    # Stable question id (QID), the grouping key for per-question
                    # metrics; emitted so a reader can join samples to the
                    # per_question_f1 breakdown.
                    "question_id": r.get("question_id", ""),
                    # Size of the prompt the model received, so a reviewer can see
                    # the real input was large even when it is not included.
                    "prompt_chars": r.get("prompt_chars", ""),
                    "gold": gold,
                    "pred": pred,
                    "match": bool(gold) and gold == pred,
                    "finish_reason": fr,
                    "truncated": fr == "length",
                    # Answer and reasoning kept apart: a reviewer can see what the
                    # model ANSWERED versus what it was thinking, which is what
                    # makes a verdict auditable.
                    "answer": r.get("answer", ""),
                    "reasoning": r.get("reasoning", ""),
                    # True when the verdict came from the constrained retry rather
                    # than the model's own free-form reply.
                    "fallback": bool(r.get("fallback")),
                    "verdict_source": r.get("verdict_source", ""),
                }
                if dump_inputs:
                    # Opt-in only. The prompt contains the call transcript, so this
                    # is off by default and must be requested explicitly.
                    row_out["input_prompt"] = src.get("prompt", "")
                f.write(json.dumps(row_out) + "\n")
                written += 1
        logger.info(
            "Wrote %d sample rows to %s (inputs %s)",
            written, samples_file, "INCLUDED" if dump_inputs else "excluded",
        )

    _print_summary(result)


def _print_summary(r: QualityResult) -> None:
    sep = "=" * 70
    print(f"\n{sep}\n  QUALITY (AutoQA) — {r.served_model}  [{r.quantization}]  ({r.hardware})\n{sep}")
    ctk_str = (
        " ".join(f"{k}={v}" for k, v in sorted(r.chat_template_kwargs.items()))
        if r.chat_template_kwargs else "server-default"
    )
    print(f"  dataset={r.dataset_version}  decoding={r.decoding_mode}  reasoning={r.reasoning_effort}")
    print(f"  chat_template_kwargs={ctk_str}  reasoning_observed={r.reasoning_observed}")
    print(f"  n={r.n_total}  unparseable={r.n_unparseable}  truncated={r.n_truncated}"
          f"  recovered_by_retry={r.n_fallback}")
    if r.verdict_sources:
        order = ["answer", "reasoning", "conclusion", "loose", "retry", "none", "error"]
        parts = [f"{k}={r.verdict_sources[k]}" for k in order if k in r.verdict_sources]
        parts += [f"{k}={v}" for k, v in sorted(r.verdict_sources.items()) if k not in order]
        print(f"  Verdict source: {'  '.join(parts)}")
    print(f"  Accuracy : {r.accuracy:.4f}")
    print(f"  Precision: {r.precision:.4f}   Recall: {r.recall:.4f}")
    print(f"  F1 (pos='{r.positive_label}'): {r.f1:.4f}   Macro-F1: {r.macro_f1:.4f}")
    c = r.confusion
    print(f"  Confusion: TP={c.get('tp', 0)}  FP={c.get('fp', 0)}  "
          f"FN={c.get('fn', 0)}  TN={c.get('tn', 0)}")
    print(f"     unparseable: gold={r.positive_label} -> {c.get('unparseable_pos', 0)} (inside FN), "
          f"gold=other -> {c.get('unparseable_neg', 0)}, other unscored -> {c.get('unscored_other', 0)}")
    reconciled = "OK" if r.confusion_total == r.n_total else "MISMATCH"
    print(f"     matrix total: {r.confusion_total} / n={r.n_total}  [{reconciled}]")
    if r.n_truncated:
        print(f"  NOTE: {r.n_truncated} repl{'y' if r.n_truncated == 1 else 'ies'} hit the token "
              f"budget and could not state a verdict — raise decoding.reasoning_max_tokens "
              f"before reading these numbers as model quality.")
    print("  Per-question F1:")
    for q, f1 in r.per_question_f1.items():
        print(f"    - [{r.per_question_n.get(q, 0):>4}] {f1:.4f}  {q}")
    print(sep + "\n")


def main() -> None:
    _configure_logging()
    ap = argparse.ArgumentParser(description="oai-infopt AutoQA quality evaluator (F1/accuracy)")
    ap.add_argument("--config", required=True, help="Eval config YAML")
    ap.add_argument("--dataset", required=True, help="Local dataset path (xlsx, csv or jsonl)")
    ap.add_argument("--sheet", default="",
                    help="For an .xlsx dataset: the worksheet (transcript sheet) to score, e.g. "
                         "'synthetic_autoqa_transcripts' (Test-ORG) or "
                         "'synthetic_autoqa_transcripts-v2' (Test-XL). Ignored for csv/jsonl.")
    ap.add_argument("--endpoint", required=True, help="vLLM base URL")
    ap.add_argument("--output", default="results/", help="Output dir for the quality JSONL")
    ap.add_argument("--model", default="", help="Served model name override")
    ap.add_argument("--quantization", default="unknown", help="Quant label for the result row")
    ap.add_argument("--hardware", default="unknown", help="Hardware/instance type label (e.g. g5, g6, g6e)")
    ap.add_argument("--wait-timeout", type=int, default=300,
                    help="Seconds to wait for the vLLM /health endpoint before giving up (model-not-ready).")
    ap.add_argument("--request-timeout", type=int, default=300,
                    help="Per-request HTTP timeout (seconds) for a single model call. Raise for large/slow "
                         "models with reasoning ON (e.g. gemma-4-31b on L4 TP=4) that can take >2 min/request; "
                         "the old fixed 120s cut working requests off as false TimeoutErrors.")
    # Opt-in per-row sample capture. OFF by default (customer-data policy).
    # Captures MODEL OUTPUT only (never the input transcript), capped by
    # --max-dump-samples to keep the file small.
    ap.add_argument("--dump-samples", action="store_true",
                    help="Also write samples-*.jsonl with the per-row rubric question, answer and "
                         "reasoning. Excludes the input prompt unless --dump-inputs is given.")
    ap.add_argument("--max-dump-samples", type=int, default=200,
                    help="Cap on rows written when --dump-samples is set (default 200).")
    ap.add_argument("--dump-inputs", action="store_true",
                    help="Include the full input prompt (CALL TRANSCRIPT) in each sample row so a "
                         "reviewer can audit the verdict. Requires --dump-samples. Off by default.")
    _run(ap.parse_args())



if __name__ == "__main__":
    main()
