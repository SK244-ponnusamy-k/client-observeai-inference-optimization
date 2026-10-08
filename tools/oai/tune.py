"""
Auto-tuning via Bayesian optimization (`oai tune <id>`).

The problem this solves
-----------------------
A full `batch_v1` benchmark is ~5 h. Hand-searching parameter combinations pays
that cost on EVERY combination, so finding good params for a new model takes
days. Following the llm-tuna method, this module instead:

  1. Searches ONLY the two or three high-impact parameters
     (`max_num_batched_tokens`, `cuda_graph_sizes`, optionally
     `long_prefill_token_threshold`) — not every flag.
  2. Ranks candidates with a SHORT proxy load (`tune_v1.yaml`, ~1-3 min) against
     ONE target concurrency, just long enough for stable metrics.
  3. Uses Optuna (TPE) so each trial learns from the last — near-optimal in
     ~20-100 trials instead of a 23,000-combo grid.
  4. Prunes OOM / invalid-startup trials immediately so no time is wasted.
  5. Writes the single winning config back into `catalog/models/<id>.yaml`.

Quality is NEVER part of this loop. The tuned params are performance knobs; they
do not change model outputs, so accuracy is evaluated once, separately, on the
winning config.

Optuna is an OPTIONAL dependency. Install with:  pip install 'oai-infopt[tune]'
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from . import catalog, paths, shell, ui
from .catalog import ModelSpec
from .ui import OaiError


# ---------------------------------------------------------------------------
# Result of one trial — what the objective measured for a candidate config.
# ---------------------------------------------------------------------------
@dataclass
class TrialOutcome:
    params: dict[str, int]
    objective_value: float          # the number Optuna optimizes
    throughput_tokens_s: float
    e2e_p95_ms: float
    status: str                     # "ok" | "pruned" | "error"
    detail: str = ""


# ---------------------------------------------------------------------------
# Optuna import guard — keep the rest of the CLI working without optuna.
# ---------------------------------------------------------------------------
def _import_optuna():  # type: ignore[no-untyped-def]
    try:
        import optuna  # noqa: PLC0415
    except ImportError as exc:
        raise OaiError(
            "The 'optuna' package is required for `oai tune` but is not installed.",
            "Install the tuning extra:  pip install 'oai-infopt[tune]'  (or: pip install optuna).",
        ) from exc
    return optuna


# ---------------------------------------------------------------------------
# Manifest helpers — a trial serves vLLM from a RENDERED copy of the model's GPU
# manifest with the candidate parameters substituted into optimization_variants.
# ---------------------------------------------------------------------------
def _base_manifest_path(spec: ModelSpec) -> Path:
    """The manifest a trial starts from (the standard dynamic GPU manifest)."""
    gpu = paths.MANIFESTS_DIR / spec.gpu_manifest_filename
    if gpu.exists():
        return gpu
    default = paths.MANIFESTS_DIR / spec.manifest_filename
    if default.exists():
        return default
    raise OaiError(
        f"No benchmark manifest found for '{spec.id}'.",
        f"Generate it first:  oai model generate {spec.id}",
    )


def _render_trial_manifest(
    spec: ModelSpec, params: dict[str, int], trial_dir: Path, trial_number: int
) -> Path:
    """Write a trial-specific manifest with `params` applied to variant 0.

    Only the two or three high-impact knobs change; everything structural
    (image, TP, quantization, max_model_len) is copied verbatim so the trial
    isolates the effect of the tuned parameters, exactly as the study requires.
    """
    base = yaml.safe_load(_base_manifest_path(spec).read_text(encoding="utf-8"))

    variants = base.get("optimization_variants") or [{}]
    v = dict(variants[0])

    if "max_num_batched_tokens" in params:
        v["max_num_batched_tokens"] = int(params["max_num_batched_tokens"])

    # cuda_graph_sizes / long_prefill_token_threshold are plumbed as explicit
    # vLLM flags via the variant's extra_args, since the manifest's named fields
    # only cover batching. The deploy template appends variant.extra_args verbatim.
    extra = list(v.get("extra_args") or [])
    if "cuda_graph_sizes" in params:
        extra = _set_flag(extra, "--cuda-graph-sizes", str(int(params["cuda_graph_sizes"])))
    if "long_prefill_token_threshold" in params:
        extra = _set_flag(
            extra, "--long-prefill-token-threshold", str(int(params["long_prefill_token_threshold"]))
        )
    # Clean signal: prefix caching OFF during tuning so a cache hit can't mask the
    # effect of the parameter under test (matches the study's methodology).
    extra = _set_flag(extra, "--no-enable-prefix-caching", None)
    v["extra_args"] = extra

    variants[0] = v
    base["optimization_variants"] = variants
    base["run_id"] = f"{spec.id}-tune-t{trial_number:03d}"
    base["description"] = f"[TUNING trial {trial_number}] {spec.id} :: {params}"

    trial_dir.mkdir(parents=True, exist_ok=True)
    out = trial_dir / f"manifest-t{trial_number:03d}.yaml"
    out.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return out


def _set_flag(args: list[str], flag: str, value: str | None) -> list[str]:
    """Return args with `flag` set to `value` (or present as a bare switch).

    Replaces an existing occurrence so a trial never stacks duplicate flags.
    """
    out: list[str] = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a == flag or a.startswith(flag + "="):
            # Drop the old flag (and its separate value token, if any).
            if a == flag and value is not None:
                skip_next = True
            continue
        out.append(a)
    if value is None:
        out.append(flag)
    else:
        out.append(f"{flag}={value}")
    return out


# ---------------------------------------------------------------------------
# Objective — run ONE trial end to end and return the metric Optuna optimizes.
# ---------------------------------------------------------------------------
def _read_best_metric(result_dir: Path, concurrency: int) -> dict[str, Any] | None:
    """Pick the result row for the tuned concurrency from a trial's JSONL output.

    load-test.py writes one JSONL (named <run_id>.jsonl) with a row per
    concurrency level. tune_v1 pins a single level, so we take the row matching
    `concurrency` (falling back to the last valid row).
    """
    rows: list[dict[str, Any]] = []
    for jf in sorted(result_dir.glob("*.jsonl")):
        for line in jf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return None
    for r in rows:
        if int(r.get("concurrency", -1)) == concurrency:
            return r
    return rows[-1]


class _Objective:
    """Callable Optuna objective. Carries everything a trial needs.

    Each call: sample params -> render manifest -> run the short proxy load via
    the trial runner -> read throughput -> return it (or prune on failure).
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        params: list[str],
        concurrency: int,
        objective: str,
        work_dir: Path,
        runner: "TrialRunner",
    ) -> None:
        self.spec = spec
        self.params = params
        self.concurrency = concurrency
        self.objective = objective
        self.work_dir = work_dir
        self.runner = runner
        self.outcomes: list[TrialOutcome] = []

    def _suggest(self, trial: Any) -> dict[str, int]:
        chosen: dict[str, int] = {}
        for p in self.params:
            low, high = self.spec.tune.bounds_for(p)
            # long_prefill default bound is model-length aware.
            if p == "long_prefill_token_threshold" and p not in self.spec.tune.bounds:
                high = max(256, self.spec.serving.max_model_len)
            # max_num_batched_tokens must not exceed the context window either.
            if p == "max_num_batched_tokens":
                high = min(high, max(low + 1, self.spec.serving.max_model_len * self.spec.serving.max_num_seqs))
            step = 256 if p != "cuda_graph_sizes" else 1
            chosen[p] = int(trial.suggest_int(p, low, high, step=step))
        return chosen

    def __call__(self, trial: Any) -> float:
        optuna = _import_optuna()
        params = self._suggest(trial)
        trial_dir = self.work_dir / f"trial-{trial.number:03d}"
        manifest = _render_trial_manifest(self.spec, params, trial_dir, trial.number)

        outcome = self.runner.run_trial(
            self.spec, manifest, trial_dir, self.concurrency, params, self.objective
        )
        self.outcomes.append(outcome)

        if outcome.status == "pruned":
            ui.warn(f"trial {trial.number} pruned ({outcome.detail}) params={params}")
            raise optuna.TrialPruned(outcome.detail)
        if outcome.status == "error":
            ui.warn(f"trial {trial.number} error ({outcome.detail}) params={params}")
            raise optuna.TrialPruned(outcome.detail)

        ui.info(
            f"trial {trial.number}: {outcome.throughput_tokens_s:.1f} tok/s "
            f"p95={outcome.e2e_p95_ms:.0f}ms  params={params}"
        )
        return outcome.objective_value


# ---------------------------------------------------------------------------
# Trial runner — isolates HOW a trial is executed so it can be swapped/dry-run.
# ---------------------------------------------------------------------------
class TrialRunner:
    """Deploys a short-lived vLLM instance per trial and runs the proxy load.

    Default implementation drives the existing in-cluster scripts. It is a thin
    seam: `oai tune --dry-run` substitutes DryRunner to validate the whole search
    loop (sampling, pruning, write-back) without touching the cluster.
    """

    def __init__(self, hw: str | None, wait_timeout: int) -> None:
        self.hw = hw
        self.wait_timeout = wait_timeout

    def run_trial(
        self,
        spec: ModelSpec,
        manifest: Path,
        trial_dir: Path,
        concurrency: int,
        params: dict[str, int],
        objective: str,
    ) -> TrialOutcome:
        deploy = paths.ROOT / "vllm" / "models" / spec.id / "deploy.sh"
        if not deploy.exists():
            return TrialOutcome(params, 0.0, 0.0, 0.0, "error",
                                f"missing deploy.sh — run 'oai model generate {spec.id}'")

        trial_no = manifest.stem.split("-t")[-1]
        tag = f"tune-t{trial_no}"
        deployment = f"{spec.deployment_name}-{tag}"
        endpoint = f"http://{spec.service_name}-{tag}:8000"
        ns = self._namespace()

        ui.banner(f"TRIAL {trial_no}  —  {spec.id}  params={params}")
        try:
            # 1. Deploy a trial-tagged, isolated copy (baseline args from the
            #    generated template). deploy.sh does NOT accept per-param flags, so
            #    the candidate params are applied in step 2 via kubectl patch.
            ui.step(f"[trial {trial_no}] deploying {deployment} ...")
            rc = shell.run_bash(deploy, self._deploy_args(tag))
            if rc != 0:
                return self._classify_failure(spec, deployment, ns, params, trial_dir, "deploy failed")

            # 2. Patch the container args with THIS trial's candidate parameters,
            #    then wait for the fresh rollout. This is what actually makes each
            #    trial different — without it every trial would serve baseline args.
            ui.step(f"[trial {trial_no}] applying params via kubectl patch: {params}")
            if not self._patch_args(spec, deployment, ns, params, trial_dir):
                return self._classify_failure(spec, deployment, ns, params, trial_dir, "patch/rollout failed")

            # 3. Run the SHORT proxy load against this trial's endpoint. Output is
            #    tee'd to a per-trial log so you can follow each run.
            ui.step(f"[trial {trial_no}] running proxy load @ concurrency {concurrency} ...")
            self._run_proxy_load(spec, manifest, endpoint, trial_dir)

            row = _read_best_metric(trial_dir, concurrency)
            if row is None:
                return self._classify_failure(spec, deployment, ns, params, trial_dir, "no metrics produced")

            tput = float(row.get("output_throughput_tokens_s") or 0.0)
            p95 = float(row.get("e2e_p95_ms") or 0.0)
            status = str(row.get("status") or "")
            if status == "error" or tput <= 0.0:
                return TrialOutcome(params, 0.0, tput, p95, "pruned", "zero output tokens")

            value = tput if objective == "output_tokens_s" else -p95
            return TrialOutcome(params, value, tput, p95, "ok")
        finally:
            ui.step(f"[trial {trial_no}] tearing down {deployment} ...")
            self._stop_trial(spec, tag)

    def _namespace(self) -> str:
        """Resolve the serving namespace from config.env (default oai-infopt)."""
        env = paths.CONFIG_ENV
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.startswith("BENCHMARK_NAMESPACE") and "=" in s:
                    return s.split("=", 1)[1].strip().strip('"').strip("'")
        return "oai-infopt"

    def _deploy_args(self, tag: str) -> list[str]:
        args = ["--tag", tag]
        if self.hw:
            args += ["--hw", self.hw]
        return args

    def _vllm_arg_overrides(self, spec: ModelSpec, params: dict[str, int]) -> list[str]:
        """The vLLM CLI flags a trial overrides on the deployment container."""
        flags: list[str] = []
        if "max_num_batched_tokens" in params:
            flags.append(f"--max-num-batched-tokens={int(params['max_num_batched_tokens'])}")
        if "cuda_graph_sizes" in params:
            flags.append(f"--cuda-graph-sizes={int(params['cuda_graph_sizes'])}")
        if "long_prefill_token_threshold" in params:
            flags.append(f"--long-prefill-token-threshold={int(params['long_prefill_token_threshold'])}")
        return flags

    def _patch_args(
        self, spec: ModelSpec, deployment: str, ns: str, params: dict[str, int], trial_dir: Path
    ) -> bool:
        """Replace the tuned flags in the live deployment and wait for rollout.

        Uses a JSON Patch (RFC 6902) to surgically replace ONLY the container args
        array, without disturbing other fields like `image`. A merge patch on the
        containers list requires a merge key that Kubernetes applies inconsistently,
        which previously caused the patch to wipe the `image` field.
        """
        cur = shell.kubectl_json([
            "get", "deployment", deployment, "-n", ns,
            "-o", "jsonpath={.spec.template.spec.containers[0].args}",
        ])
        if not cur:
            return False
        try:
            args: list[str] = json.loads(cur)
        except json.JSONDecodeError:
            return False

        tuned_prefixes = ("--max-num-batched-tokens", "--cuda-graph-sizes", "--long-prefill-token-threshold")
        kept = [a for a in args if not any(a.startswith(p) for p in tuned_prefixes)]
        new_args = kept + self._vllm_arg_overrides(spec, params)

        # JSON Patch: replace the args of the first container at a precise path.
        # This does NOT touch image, env, resources, or any other container field.
        patch = json.dumps([
            {"op": "replace", "path": "/spec/template/spec/containers/0/args", "value": new_args}
        ])
        patch_file = trial_dir / "patch.json"
        patch_file.write_text(patch, encoding="utf-8")

        rc = shell.run_kubectl(["patch", "deployment", deployment, "-n", ns,
                                "--type", "json", "--patch-file", str(patch_file)])
        if rc != 0:
            return False
        # Wait for the patched pod to roll out and pass readiness (vLLM /health).
        rc = shell.run_kubectl(["rollout", "status", f"deployment/{deployment}", "-n", ns,
                                f"--timeout={self.wait_timeout}s"])
        return rc == 0

    def _run_proxy_load(self, spec: ModelSpec, manifest: Path, endpoint: str, trial_dir: Path) -> int:
        load_test = paths.ROOT / "inference" / "load-test.py"
        profile = paths.ROOT / spec.tune.profile
        return shell.run(
            [
                sys.executable, str(load_test),
                "--manifest", str(manifest),
                "--profile", str(profile),
                "--endpoint", endpoint,
                "--output", str(trial_dir),
                "--wait-timeout", str(self.wait_timeout),
            ]
        )

    def _stop_trial(self, spec: ModelSpec, tag: str) -> None:
        stop = paths.ROOT / "vllm" / "models" / spec.id / "stop.sh"
        if stop.exists():
            try:
                # stop.sh gates on the OAI_ASSUME_YES env var, not a -y flag, so a
                # tuning teardown must set it or the script blocks on an interactive
                # "Proceed? [y/N]" prompt and stalls the whole study.
                shell.run_bash(stop, ["--tag", tag], env={"OAI_ASSUME_YES": "true"})
            except Exception:  # noqa: BLE001  (cleanup is best-effort)
                ui.warn(f"trial cleanup: could not stop tagged copy '{tag}' — stop it manually if it lingers.")

    def _classify_failure(self, spec: ModelSpec, deployment: str, ns: str, params: dict[str, int], trial_dir: Path, detail: str) -> TrialOutcome:
        """Map a trial failure to prune (recoverable, learn from it) vs error.

        OOM and invalid-parameter startups are PRUNED — the study records them so
        the sampler avoids that region — rather than aborting the whole run.
        Capture pod logs into the trial directory for post-mortem.
        """
        # Capture pod logs to disk for debugging.
        log_blob = ""
        try:
            log_out = shell.kubectl_json(["logs", f"deployment/{deployment}", "-n", ns, "--tail=200"])
            if log_out:
                log_blob = log_out.lower()
                trial_dir.mkdir(parents=True, exist_ok=True)
                (trial_dir / "pod-logs.txt").write_text(log_out, encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

        # Also scan any local log files in the trial dir.
        for lg in trial_dir.glob("*.log"):
            log_blob += lg.read_text(encoding="utf-8", errors="ignore").lower()
        oom_markers = ("out of memory", "oom", "cuda out of memory", "exit code 137", "exitcode 137")
        startup_markers = ("valueerror", "max_num_batched_tokens", "no available memory for the cache")
        if any(m in log_blob for m in oom_markers):
            return TrialOutcome(params, 0.0, 0.0, 0.0, "pruned", "OOM")
        if any(m in log_blob for m in startup_markers):
            return TrialOutcome(params, 0.0, 0.0, 0.0, "pruned", "invalid startup config")
        return TrialOutcome(params, 0.0, 0.0, 0.0, "pruned", detail)


class DryRunner(TrialRunner):
    """Validates the search loop with a deterministic synthetic objective.

    No cluster, no deploy. Throughput is a smooth concave function of the params
    with a single interior optimum, so a correct TPE search should climb toward
    it — enough to exercise sampling, pruning bounds, and write-back offline.
    """

    def run_trial(self, spec, manifest, trial_dir, concurrency, params, objective):  # type: ignore[override]
        mnbt = params.get("max_num_batched_tokens", 8192)
        cg = params.get("cuda_graph_sizes", 128)
        # Concave bowl peaking near 32768 / 160 — arbitrary but stable.
        tput = 700.0 - ((mnbt - 32768) / 2048) ** 2 * 0.4 - ((cg - 160) / 16) ** 2 * 0.6
        tput = max(50.0, tput)
        p95 = 20000.0 / tput
        value = tput if objective == "output_tokens_s" else -p95
        return TrialOutcome(params, value, tput, p95, "ok")


# ---------------------------------------------------------------------------
# Catalog write-back — persist the winning params into the model's YAML.
# ---------------------------------------------------------------------------
def _write_back(spec: ModelSpec, best: dict[str, int], objective: str, value: float, concurrency: int) -> Path:
    """Update serving.* in catalog/models/<id>.yaml with the tuned values.

    A targeted text edit (not a full YAML re-dump) preserves the hand-written
    comments that document each model's quirks. A provenance comment records when
    and how the values were chosen so a reader knows they came from a tuning run.
    """
    path = paths.catalog_path(spec.id)
    text = path.read_text(encoding="utf-8")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    metric = "output tok/s" if objective == "output_tokens_s" else "p95 latency ms"
    shown = value if objective == "output_tokens_s" else -value
    provenance = (
        f"  # --- auto-tuned {stamp} (oai tune) ---\n"
        f"  # objective={metric} best={shown:.1f} @ concurrency={concurrency}\n"
    )

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_serving = False
    applied: set[str] = set()
    field_map = {
        "max_num_batched_tokens": "max_num_batched_tokens",
        "cuda_graph_sizes": "cuda_graph_sizes",
        "long_prefill_token_threshold": "long_prefill_token_threshold",
    }
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("serving:"):
            in_serving = True
            out.append(line)
            out.append(provenance)
            continue
        # Leaving the serving block: a new top-level key at column 0.
        if in_serving and line and not line[0].isspace() and ":" in line:
            # Emit any tuned fields that weren't already present in the block.
            for p, key in field_map.items():
                if p in best and key not in applied:
                    out.append(f"  {key}: {int(best[p])}\n")
                    applied.add(key)
            in_serving = False
        if in_serving:
            replaced = False
            for p, key in field_map.items():
                if p in best and stripped.startswith(f"{key}:"):
                    indent = line[: len(line) - len(line.lstrip())]
                    out.append(f"{indent}{key}: {int(best[p])}\n")
                    applied.add(key)
                    replaced = True
                    break
            if replaced:
                continue
        out.append(line)

    # If serving was the LAST block, flush remaining tuned fields at the end.
    if in_serving:
        for p, key in field_map.items():
            if p in best and key not in applied:
                out.append(f"  {key}: {int(best[p])}\n")

    path.write_text("".join(out), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def run(
    model_id: str,
    *,
    concurrency: int | None = None,
    trials: int | None = None,
    objective: str | None = None,
    params: list[str] | None = None,
    hw: str | None = None,
    wait_timeout: int = 1800,
    write_back: bool = True,
    dry_run: bool = False,
) -> int:
    optuna = _import_optuna()
    spec = catalog.load(model_id)

    # Validate the tune-relevant parts up front with a clear message.
    problems = [p for p in catalog.validate(spec) if p.startswith("tune.")]
    if problems:
        ui.error("Cannot tune - the catalog tune block has problems:")
        for p in problems:
            ui.hint(p)
        return 1

    obj = objective or spec.tune.objective
    conc = concurrency or spec.tune.target_concurrency
    search_params = params or spec.tune.params
    n_trials = trials or spec.tune.trials
    warmup = min(spec.tune.random_warmup, max(1, n_trials - 1))

    if obj not in ("output_tokens_s", "p95_latency"):
        raise OaiError(f"Unknown objective '{obj}'.", "Use output_tokens_s or p95_latency.")

    direction = "maximize"  # objective_value already encodes sign (p95 is negated)
    work_dir = paths.RESULTS_DIR / "tuning" / f"{spec.id}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    ui.banner(f"Auto-tune: {spec.id}{'  (DRY RUN)' if dry_run else ''}")
    ui.kv("Objective", "maximize output tok/s" if obj == "output_tokens_s" else "minimize p95 latency")
    ui.kv("Concurrency", str(conc))
    ui.kv("Params", ", ".join(search_params))
    ui.kv("Trials", f"{n_trials}  ({warmup} random warmup -> TPE)")
    ui.kv("Proxy profile", spec.tune.profile)
    ui.kv("Work dir", str(work_dir))
    ui.warn("Tuned configs are workload-specific — they may not transfer to a very different concurrency.")

    runner: TrialRunner = DryRunner(hw, wait_timeout) if dry_run else TrialRunner(hw, wait_timeout)
    objective_fn = _Objective(
        spec, params=search_params, concurrency=conc, objective=obj, work_dir=work_dir, runner=runner
    )

    sampler = optuna.samplers.TPESampler(n_startup_trials=warmup, seed=42)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective_fn, n_trials=n_trials, catch=())

    completed = [o for o in objective_fn.outcomes if o.status == "ok"]
    pruned = [o for o in objective_fn.outcomes if o.status != "ok"]
    if not completed:
        ui.error("No trial produced a valid measurement — nothing to adopt.")
        ui.hint(f"All {len(pruned)} trial(s) were pruned (OOM / invalid startup / no metrics).")
        ui.hint("Widen the bounds, lower max_num_batched_tokens, or check deploy logs under the work dir.")
        return 1

    best = study.best_trial
    best_params = {k: int(v) for k, v in best.params.items()}

    ui.banner("Best configuration")
    for k, v in best_params.items():
        ui.kv(k, str(v))
    ui.kv("Best value", f"{best.value:.1f}" + (" tok/s" if obj == "output_tokens_s" else " (neg p95 ms)"))
    ui.kv("Completed", f"{len(completed)}/{len(objective_fn.outcomes)} trials ({len(pruned)} pruned)")

    _write_study_log(work_dir, spec, objective_fn.outcomes, best_params, obj)

    if dry_run:
        ui.info("Dry run complete — search loop validated. No catalog changes written.")
        return 0

    if write_back:
        path = _write_back(spec, best_params, obj, best.value, conc)
        ui.info(f"Wrote tuned params into {path.relative_to(paths.ROOT).as_posix()}")
        ui.hint(f"Regenerate + validate ONCE on the full suite:  oai model generate {spec.id} && "
                f"oai deploy {spec.id} --benchmark --quality")
    else:
        ui.info("Skipped catalog write-back (--no-write-back). Apply the values above by hand.")
    return 0


def _write_study_log(
    work_dir: Path, spec: ModelSpec, outcomes: list[TrialOutcome], best: dict[str, int], objective: str
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    log = {
        "model_id": spec.id,
        "objective": objective,
        "best_params": best,
        "trials": [
            {
                "params": o.params,
                "throughput_tokens_s": o.throughput_tokens_s,
                "e2e_p95_ms": o.e2e_p95_ms,
                "status": o.status,
                "detail": o.detail,
            }
            for o in outcomes
        ],
    }
    (work_dir / "study.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    ui.kv("Study log", str((work_dir / "study.json")))
