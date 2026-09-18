"""
Dynamic instance selection with availability + quota checks and graceful fallback.

Given a model's ordered list of candidate instance types, pick the first one that
is actually usable in the current region/account:

  1. The instance type must be offered in the region.
  2. The account must have enough service quota (vCPU) to launch it.
  3. (best effort) There must be no obvious capacity signal against it.

If the preferred instance is not usable, we fall back to the next candidate and
tell the user exactly why we moved on. If NOTHING is usable, we raise a single,
actionable error explaining every option (request quota, try another region,
use the managed node-group fallback, or wait for capacity).

All AWS calls are best-effort: if the CLI is missing or a call fails, we DEGRADE
to "assume available" rather than blocking a deploy, and we say so. This keeps
the tool usable in restricted environments while still helping in the common case.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass

from . import ui

# vCPU count per instance type - used for quota math against the region vCPU limit.
# Only the families this framework targets need to be listed.
_VCPUS = {
    "g5.xlarge": 4, "g5.2xlarge": 8, "g5.4xlarge": 16, "g5.12xlarge": 48, "g5.48xlarge": 192,
    "g6.xlarge": 4, "g6.2xlarge": 8, "g6.4xlarge": 16, "g6.12xlarge": 48, "g6.48xlarge": 192,
    "g6e.xlarge": 4, "g6e.2xlarge": 8, "g6e.4xlarge": 16, "g6e.12xlarge": 48, "g6e.48xlarge": 192,
    "trn1.2xlarge": 8, "trn1.32xlarge": 128,
    "trn2.48xlarge": 192,
    "inf2.xlarge": 4, "inf2.8xlarge": 32, "inf2.24xlarge": 96, "inf2.48xlarge": 192,
}

# GPU (accelerator) count per instance type — used for the TP-vs-GPU preflight so
# a model with tensor_parallel_size=N is never deployed on an instance with fewer
# than N accelerators (which would OOM or fail to schedule). Sizes not listed here
# skip the check (returns None). For Neuron, the "GPU" count is NeuronCores.
_GPU_COUNT = {
    # G5 (A10G), G6 (L4), G6e (L40S): 1 GPU on x/2x/4x/8x/16x; 4 on 12x/24x; 8 on 48x.
    "g5.xlarge": 1, "g5.2xlarge": 1, "g5.4xlarge": 1, "g5.8xlarge": 1, "g5.16xlarge": 1,
    "g5.12xlarge": 4, "g5.24xlarge": 4, "g5.48xlarge": 8,
    "g6.xlarge": 1, "g6.2xlarge": 1, "g6.4xlarge": 1, "g6.8xlarge": 1, "g6.16xlarge": 1,
    "g6.12xlarge": 4, "g6.24xlarge": 4, "g6.48xlarge": 8,
    "g6e.xlarge": 1, "g6e.2xlarge": 1, "g6e.4xlarge": 1, "g6e.8xlarge": 1, "g6e.16xlarge": 1,
    "g6e.12xlarge": 4, "g6e.24xlarge": 4, "g6e.48xlarge": 8,
    # G7 / G7e (RTX PRO 6000 Blackwell): same size→GPU mapping as G6e.
    "g7.2xlarge": 1, "g7.4xlarge": 1, "g7.8xlarge": 1, "g7.16xlarge": 1,
    "g7.12xlarge": 4, "g7.24xlarge": 4, "g7.48xlarge": 8,
    "g7e.2xlarge": 1, "g7e.4xlarge": 1, "g7e.8xlarge": 1, "g7e.16xlarge": 1,
    "g7e.12xlarge": 4, "g7e.24xlarge": 4, "g7e.48xlarge": 8,
    # Trainium NeuronCores (used as the "accelerator" count for TP checks).
    "trn1.2xlarge": 2, "trn1.32xlarge": 16,
    "trn2.48xlarge": 16,
}


def gpu_count(instance_type: str) -> int | None:
    """Accelerators (GPUs / NeuronCores) on an instance type, or None if unknown."""
    return _GPU_COUNT.get((instance_type or "").strip().lower())


def check_tp_fits(instance_type: str, tensor_parallel_size: int) -> str | None:
    """Return an error message if TP exceeds the instance's accelerator count.

    Returns None when the pairing is valid OR the instance's GPU count is unknown
    (unknown → skip the check rather than block a legitimate new instance type).
    """
    gpus = gpu_count(instance_type)
    if gpus is None:
        return None
    if tensor_parallel_size > gpus:
        return (
            f"tensor_parallel_size={tensor_parallel_size} but {instance_type} has only "
            f"{gpus} accelerator(s). Use an instance with >= {tensor_parallel_size} "
            f"(e.g. a .12xlarge/.24xlarge for 4, .48xlarge for 8), or lower TP."
        )
    return None


# Service Quota codes for the on-demand vCPU limit of each family (EC2 = "ec2").
# G/VT family shares one quota; Trn/Inf have their own.
_QUOTA_CODE = {
    "g": "L-DB2E81BA",     # Running On-Demand G and VT instances (vCPUs)
    "trn": "L-2C3B7624",   # Running On-Demand Trn instances (vCPUs)
    "inf": "L-1945791B",   # Running On-Demand Inf instances (vCPUs)
}


@dataclass
class Selection:
    instance_type: str
    reason: str
    checked: list[str]          # candidates we tried before this one
    degraded: bool = False      # True if we could not fully verify and assumed available


def _aws_available() -> bool:
    return shutil.which("aws") is not None


def _run_aws(args: list[str], region: str) -> tuple[int, str, str]:
    cmd = ["aws", *args, "--region", region, "--output", "json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)  # noqa: S603
        return proc.returncode, proc.stdout, proc.stderr
    except (subprocess.TimeoutExpired, OSError) as exc:
        return 1, "", str(exc)


def _quota_family(instance_type: str) -> str:
    fam = instance_type.split(".")[0]
    if fam.startswith("trn"):
        return "trn"
    if fam.startswith("inf"):
        return "inf"
    return "g"  # g5/g6/g6e/g7 share the G quota


def _is_offered(instance_type: str, region: str) -> bool | None:
    """True/False if we can determine it, None if the check could not run."""
    rc, out, _ = _run_aws(
        [
            "ec2", "describe-instance-type-offerings",
            "--location-type", "region",
            "--filters", f"Name=instance-type,Values={instance_type}",
        ],
        region,
    )
    if rc != 0:
        return None
    try:
        data = json.loads(out)
        return len(data.get("InstanceTypeOfferings", [])) > 0
    except (json.JSONDecodeError, KeyError):
        return None


def _quota_ok(instance_type: str, region: str) -> tuple[bool | None, str]:
    """Check the region vCPU quota covers one instance of this type."""
    code = _QUOTA_CODE[_quota_family(instance_type)]
    rc, out, _ = _run_aws(
        ["service-quotas", "get-service-quota", "--service-code", "ec2", "--quota-code", code],
        region,
    )
    if rc != 0:
        # Fall back to the applied default quota if the live value is unavailable.
        return None, "quota value unavailable"
    try:
        limit = json.loads(out)["Quota"]["Value"]
    except (json.JSONDecodeError, KeyError):
        return None, "quota value unparseable"
    needed = _VCPUS.get(instance_type, 0)
    if needed == 0:
        return None, "unknown vCPU size"
    if limit >= needed:
        return True, f"quota {int(limit)} vCPU >= {needed} needed"
    return False, f"quota is {int(limit)} vCPU but this instance needs {needed} vCPU"


def select(candidates: list[str], region: str) -> Selection:
    """
    Pick the first usable instance from the ordered candidate list.

    Raises ui.OaiError with an actionable message if none are usable.
    """
    if not candidates:
        ui.fail(
            "No candidate instances to choose from.",
            "Add an 'instances.candidates' list to the catalog entry.",
        )

    if not _aws_available():
        ui.warn("AWS CLI not found - cannot verify live instance availability.")
        ui.warn(f"Assuming the preferred instance '{candidates[0]}' is available.")
        return Selection(candidates[0], "AWS CLI unavailable; assumed preferred", [], degraded=True)

    checked: list[str] = []
    reasons: list[str] = []

    for inst in candidates:
        ui.step(f"Checking availability of {inst} in {region} ...")

        offered = _is_offered(inst, region)
        if offered is False:
            msg = f"{inst}: not offered in {region}"
            ui.warn(msg + " - trying next candidate.")
            reasons.append(msg)
            checked.append(inst)
            continue

        quota, quota_reason = _quota_ok(inst, region)
        if quota is False:
            msg = f"{inst}: {quota_reason}"
            ui.warn(msg + " - trying next candidate.")
            reasons.append(msg)
            checked.append(inst)
            continue

        # offered is True or None (unknown); quota is True or None (unknown).
        degraded = offered is None or quota is None
        reason = quota_reason if quota is not None else "availability assumed (could not fully verify)"
        if degraded:
            ui.warn(f"{inst}: could not fully verify ({reason}); proceeding with it.")
        else:
            ui.info(f"{inst}: available ({reason}).")
        return Selection(inst, reason, checked, degraded=degraded)

    # Nothing usable - build one actionable error covering all remedies.
    detail = "\n     ".join(reasons) if reasons else "no usable candidate found"
    fam = _quota_family(candidates[0])
    quota_hint = {
        "g": "Request more 'Running On-Demand G and VT instances' vCPUs in Service Quotas.",
        "trn": "Request more 'Running On-Demand Trn instances' vCPUs in Service Quotas.",
        "inf": "Request more 'Running On-Demand Inf instances' vCPUs in Service Quotas.",
    }[fam]
    ui.fail(
        f"None of the candidate instances are usable in {region}:\n     {detail}",
        (
            "Options:\n"
            f"   -> {quota_hint}\n"
            "   -> Add more/other instance types to 'instances.candidates' in the catalog.\n"
            "   -> Try a different AWS region (set AWS_REGION).\n"
            + (
                "   -> Use the managed node-group fallback (see 'node_group' in the catalog "
                "and deploy with --managed-ng).\n"
                if fam == "trn"
                else ""
            )
            + "   -> Capacity for accelerators fluctuates; retry in a few minutes."
        ),
    )
    raise AssertionError("unreachable")  # pragma: no cover
