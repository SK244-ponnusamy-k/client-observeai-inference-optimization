"""
Benchmark orchestration.

Wraps the existing in-cluster benchmark harness (inference/run-benchmark.sh),
driven by the model's catalog entry. Benchmarking is fully customizable:

  - Which profiles run comes from catalog `benchmark.profiles` (or --profile).
  - A custom dataset comes from catalog `benchmark.dataset` (or --dataset).
  - Cost math uses catalog `benchmark.instance_hourly_usd`, baked into the
    generated run manifest.

`oai deploy --benchmark` (or catalog benchmark.auto) calls run() right after a
successful deploy, so a user can onboard, deploy, and benchmark in one step.
"""

from __future__ import annotations

import re

from . import catalog, paths, shell, ui
from .catalog import ModelSpec

# Map a workload-profile path/name to the short name run-benchmark.sh expects.
_PROFILE_RE = re.compile(r"(realtime|batch)")


def _profile_short(profile_ref: str) -> str | None:
    m = _PROFILE_RE.search(profile_ref)
    return m.group(1) if m else None


def _selected_profiles(spec: ModelSpec, profile: str | None, skip_batch: bool) -> list[str]:
    if profile:
        short = _profile_short(profile)
        if not short:
            ui.fail(
                f"Unrecognised profile '{profile}'.",
                "Use 'realtime' or 'batch' (or a path like configs/workload_profiles/realtime_v1.yaml).",
            )
        return [short]
    shorts: list[str] = []
    for p in spec.benchmark.profiles:
        s = _profile_short(p)
        if s and s not in shorts:
            shorts.append(s)
    if skip_batch:
        shorts = [s for s in shorts if s != "batch"]
    return shorts or ["realtime"]


def _manifest_hw(spec: ModelSpec) -> str:
    """The <hw> token used in the generated manifest filename."""
    if spec.preferred_instance:
        return spec.preferred_instance.split(".")[0]
    return spec.hardware


def run(model_id: str, *, profile: str | None = None, dataset: str | None = None, skip_batch: bool = False) -> int:
    spec = catalog.load(model_id)

    manifest_rel = f"configs/manifests/{spec.manifest_filename}"
    if not (paths.ROOT / manifest_rel).exists():
        ui.fail(
            f"Benchmark manifest not found: {manifest_rel}",
            f"Generate it first: oai model generate {model_id}",
        )

    if spec.benchmark.instance_hourly_usd <= 0:
        ui.warn("benchmark.instance_hourly_usd is 0 in the catalog - cost figures will be null.")
        ui.hint(f"Set it in catalog/models/{model_id}.yaml and re-run 'oai model generate {model_id}' for cost math.")

    profiles = _selected_profiles(spec, profile, skip_batch)
    ds = dataset if dataset is not None else spec.benchmark.dataset
    hw = _manifest_hw(spec)

    script = paths.ROOT / "inference" / "run-benchmark.sh"
    if not script.exists():
        ui.fail("inference/run-benchmark.sh is missing.", "This is part of the base framework - check your checkout.")

    ui.banner(f"Benchmark: {spec.id}")
    ui.kv("Manifest", manifest_rel)
    ui.kv("Profiles", ", ".join(profiles))
    ui.kv("Dataset", ds or "built-in random (no PII)")

    overall = 0
    for prof in profiles:
        ui.step(f"Running the '{prof}' profile ...")
        args = ["--model", spec.id, "--profile", prof, "--hw", hw, "--manifest", manifest_rel]
        if ds:
            args += ["--dataset", ds]
        rc = shell.run_bash(script, args)
        if rc != 0:
            # run-benchmark.sh exits 1 on SLO violation too - results may still be valid.
            ui.warn(f"The '{prof}' profile finished non-zero (exit {rc}).")
            ui.hint("This can mean an SLO threshold was not met (results still uploaded) or a real error.")
            ui.hint("Check the log above and the S3 results path it printed.")
            overall = rc
        else:
            ui.info(f"'{prof}' profile complete.")

    if overall == 0:
        ui.info("Benchmark complete. Results are in the results bucket (path shown above).")
    return overall
