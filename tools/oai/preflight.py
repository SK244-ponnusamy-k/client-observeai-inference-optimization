"""
Preflight checks shared across deploy / benchmark / quality.

The goal is to refuse starting a DUPLICATE run of the same model+tag while a
copy is already active: deploying or benchmarking/quality-testing the same
model+tag again collides on resource names and endpoint, contaminates results,
and confuses auto-stop. These checks are read-only (kubectl) and fail safe: if
kubectl cannot reach the cluster they return "nothing active" so a deploy is
never blocked spuriously.
"""

from __future__ import annotations

import json

from . import shell, ui

_NS = "oai-infopt"


def active_runs_for(
    model_id: str,
    tag: str | None,
    *,
    include_deployment: bool = True,
) -> list[str]:
    """Return a list of ACTIVE resources for this model+tag.

    "Active" means:
      - (optional) a vLLM Deployment named oai-infopt-vllm-<id>[-<tag>] exists, and
      - any benchmark/quality Job for this model+tag that is NOT terminal (a Job
        is terminal only when it has a Complete or Failed condition set True).

    Job names are oai-infopt-(bench|quality)-<id-dashed>[-<tag>]-<timestamp>-...,
    so we match on the name prefix (benchmark Jobs carry no run-tag label; the tag
    is reliably embedded only in the name).

    Returns an empty list when nothing is active OR kubectl can't reach the
    cluster (fail-safe: never block spuriously).
    """
    tag_seg = f"-{tag}" if tag else ""
    id_dashed = model_id.replace(".", "-")
    found: list[str] = []

    if include_deployment:
        dep_name = f"oai-infopt-vllm-{model_id}{tag_seg}"
        dep_out = shell.kubectl_json(["get", "deployment", dep_name, "-n", _NS, "-o", "name"])
        if dep_out and dep_out.strip():
            found.append(f"deployment/{dep_name}")

    jobs_out = shell.kubectl_json(["get", "jobs", "-n", _NS, "-o", "json"])
    if jobs_out:
        try:
            items = json.loads(jobs_out).get("items", [])
        except (ValueError, TypeError):
            items = []
        bench_prefix = f"oai-infopt-bench-{id_dashed}{tag_seg}-"
        qual_prefix = f"oai-infopt-quality-{id_dashed}{tag_seg}-"
        for j in items:
            name = (j.get("metadata") or {}).get("name", "")
            if not (name.startswith(bench_prefix) or name.startswith(qual_prefix)):
                continue
            conds = (j.get("status") or {}).get("conditions") or []
            terminal = any(
                c.get("type") in ("Complete", "Failed") and c.get("status") == "True"
                for c in conds
            )
            if not terminal:
                found.append(f"job/{name}")

    return found


def fail_if_active(
    model_id: str,
    tag: str | None,
    *,
    include_deployment: bool = True,
) -> None:
    """Abort (ui.fail) if a duplicate of this model+tag is already active.

    include_deployment=True for `oai deploy` (a leftover Deployment also blocks);
    False for standalone `oai benchmark` / `oai quality` (they only care that no
    other benchmark/quality run for this model+tag is in flight).
    """
    active = active_runs_for(model_id, tag, include_deployment=include_deployment)
    if not active:
        return

    tag_hint = f" (tag={tag})" if tag else " (no tag)"
    shown = "\n   -> ".join(active[:8])
    more = f"\n   -> ... and {len(active) - 8} more" if len(active) > 8 else ""
    new_tag = f"{tag or 'run'}2"
    ui.fail(
        f"'{model_id}'{tag_hint} already has an active run; refusing to start a duplicate "
        "(it would hit the same endpoint, contaminate results, and confuse auto-stop).",
        "Already active:\n   -> " + shown + more + "\n\n"
        "Use a DIFFERENT --tag for this run, or stop/clear the existing one first:\n"
        f"   -> oai stop {model_id}{f' --tag {tag}' if tag else ''} -y   (if a model is running)\n"
        f"   -> or wait for the active jobs to finish, or re-run with --tag {new_tag}.",
    )
