"""
Cluster-level setup helpers exposed via `oai cluster ...`.

Right now this wraps the one-time Trainium (trn2) managed node-group setup so a
user does not have to remember the raw eksctl/CloudFormation commands.

Design principle: NEVER create what already exists. Before running the setup
script we check whether the target node group is already present. If it is, we
reuse it (no CloudFormation stack, no eksctl create) and just report its state.
The underlying script is itself idempotent, but checking first lets us give the
user a clear, fast answer when nothing needs to be done.
"""

from __future__ import annotations

import json

from . import config, paths, shell, ui


def _nodegroup_status(cluster: str, nodegroup: str, region: str) -> str | None:
    """Return the node group status string, or None if it does not exist."""
    import subprocess

    if not shell.have("aws"):
        return None
    try:
        proc = subprocess.run(  # noqa: S603
            [
                "aws", "eks", "describe-nodegroup",
                "--cluster-name", cluster,
                "--nodegroup-name", nodegroup,
                "--region", region,
                "--query", "nodegroup.status",
                "--output", "text",
            ],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    status = proc.stdout.strip()
    return status or None


def _aws_text(args: list[str], region: str) -> str | None:
    """Run an aws CLI command returning trimmed stdout text, or None on failure."""
    import subprocess

    if not shell.have("aws"):
        return None
    try:
        proc = subprocess.run(  # noqa: S603
            ["aws", *args, "--region", region, "--output", "text"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _cluster_exists(cluster: str, region: str) -> bool:
    return _aws_text(["eks", "describe-cluster", "--name", cluster, "--query", "cluster.name"], region) not in (
        None,
        "",
    )


def _amp_workspace_id(cluster: str, region: str) -> str | None:
    alias = f"amp-ws-{cluster}"
    val = _aws_text(["amp", "list-workspaces", "--alias", alias, "--query", "workspaces[0].workspaceId"], region)
    return val if val and val != "None" else None


def _monitoring_installed() -> bool:
    """True if the kube-prometheus-stack helm release is present in the monitoring ns."""
    import subprocess

    if not shell.have("helm"):
        # Fall back to checking the namespace + a known deployment.
        out = shell.kubectl_json(["get", "deploy", "-n", "monitoring", "-o", "name"])
        return bool(out and "kube-prometheus-stack" in out)
    try:
        proc = subprocess.run(  # noqa: S603
            ["helm", "list", "-n", "monitoring", "-o", "json"],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    if proc.returncode != 0:
        return False
    return "kube-prometheus-stack" in proc.stdout


def bootstrap(*, compute: str | None = None, force: bool = False) -> int:
    """
    Provision the full cluster (EKS + S3 + IAM + node pools + monitoring).

    Reuse-if-exists: if the cluster already exists, we do NOT recreate it. We
    report that and stop, unless --force is passed (bootstrap.sh is itself
    idempotent, so --force re-runs it to reconcile/add missing pieces).
    """
    cfg = config.load()
    region = cfg.get("AWS_REGION", "us-east-2")
    cluster = cfg.get("CLUSTER_NAME", "")

    ui.banner("Cluster bootstrap")
    ui.kv("Cluster", cluster or "(not set in config.env)")
    ui.kv("Region", region)

    shell.require_tool("aws", "provision the cluster")
    if _cluster_exists(cluster, region):
        if not force:
            ui.info(f"Cluster '{cluster}' already exists. Reusing it - nothing to create.")
            ui.hint("To reconcile / add missing pieces anyway, re-run with --force.")
            ui.hint("Set up monitoring with: oai cluster setup-monitoring")
            return 0
        ui.warn(f"Cluster '{cluster}' exists - re-running bootstrap to reconcile (--force).")

    shell.require_tool("eksctl", "create the EKS cluster")
    shell.require_tool("kubectl", "configure the cluster")

    script = paths.ROOT / "cluster" / "bootstrap.sh"
    if not script.exists():
        ui.fail("cluster/bootstrap.sh is missing.", "This is part of the base framework - check your checkout.")

    args: list[str] = []
    if compute:
        args += ["--compute", compute]

    rc = shell.run_bash(script, args)
    if rc != 0:
        ui.fail(
            f"Cluster bootstrap did not complete (exit {rc}).",
            "See the log above. Common causes: missing eksctl, insufficient IAM permissions, "
            "or a partially-created CloudFormation stack (safe to re-run - bootstrap is idempotent).",
        )
    ui.info("Cluster is ready.")
    ui.hint("Next: oai cluster setup-monitoring   then   oai model new")
    return 0


def setup_monitoring(*, force: bool = False) -> int:
    """
    Install the monitoring stack (AMP + Prometheus + Grafana + DCGM exporter).

    Reuse-if-exists: if the kube-prometheus-stack release is already installed,
    we report it and stop unless --force. The underlying script is idempotent.
    """
    cfg = config.load()
    region = cfg.get("AWS_REGION", "us-east-2")
    cluster = cfg.get("CLUSTER_NAME", "")

    ui.banner("Monitoring stack setup")
    ui.kv("Cluster", cluster or "(not set)")
    ui.kv("Region", region)

    if not _cluster_exists(cluster, region):
        ui.fail(
            f"Cluster '{cluster}' does not exist, so monitoring cannot be installed.",
            "Create the cluster first: oai cluster bootstrap",
        )

    if _monitoring_installed() and not force:
        ui.info("Monitoring stack (kube-prometheus-stack) is already installed. Reusing it - nothing to do.")
        amp = _amp_workspace_id(cluster, region)
        if amp:
            ui.kv("AMP workspace", amp)
        ui.hint("Open Grafana: kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n monitoring")
        ui.hint("Re-run with --force to reconcile / upgrade the stack.")
        return 0

    shell.require_tool("aws", "set up monitoring")
    shell.require_tool("kubectl", "set up monitoring")
    shell.require_tool("helm", "install the monitoring stack")

    script = paths.ROOT / "monitoring" / "setup-monitoring.sh"
    if not script.exists():
        ui.fail("monitoring/setup-monitoring.sh is missing.", "Check your checkout of the base framework.")

    rc = shell.run_bash(script)
    if rc != 0:
        ui.fail(
            f"Monitoring setup did not complete (exit {rc}).",
            "See the log above. Common causes: helm not installed, AMP permissions, "
            "or the AWS Load Balancer Controller not present (needed for the Grafana ingress).",
        )
    ui.info("Monitoring stack is ready.")
    ui.hint("Grafana admin password + URL were printed above.")
    return 0


def teardown(*, assume_yes: bool = False) -> int:
    """
    Delete the EKS cluster and associated AWS resources (keeps S3 weights + IAM).

    This is destructive and hard to reverse, so we require an explicit
    confirmation here in addition to the prompt inside teardown.sh.
    """
    cfg = config.load()
    region = cfg.get("AWS_REGION", "us-east-2")
    cluster = cfg.get("CLUSTER_NAME", "")

    ui.banner("Cluster teardown (DESTRUCTIVE)")
    ui.kv("Cluster", cluster or "(not set)")
    ui.kv("Region", region)

    if not _cluster_exists(cluster, region):
        ui.info(f"Cluster '{cluster}' does not exist - nothing to tear down.")
        return 0

    ui.warn("This deletes the EKS cluster and its VPC resources. S3 weights and IAM roles are kept.")
    if not assume_yes:
        try:
            answer = input("  Type the cluster name to confirm deletion: ").strip()
        except EOFError:
            answer = ""
        if answer != cluster:
            ui.info("Cancelled - the name did not match.")
            return 0

    script = paths.ROOT / "cluster" / "teardown.sh"
    if not script.exists():
        ui.fail("cluster/teardown.sh is missing.", "Check your checkout of the base framework.")

    env = {"OAI_ASSUME_YES": "true"} if assume_yes else {}
    rc = shell.run_bash(script, env=env)
    if rc != 0:
        ui.fail(
            f"Teardown reported an error (exit {rc}).",
            "Some resources may remain. Check the AWS console for leftover CloudFormation stacks / ENIs.",
        )
    ui.info("Cluster torn down. S3 weights and IAM roles were preserved.")
    return 0


def setup_neuron(*, dry_run: bool = False, az: str | None = None, capacity_type: str | None = None) -> int:
    """
    Ensure a trn2 managed node group exists. Reuse if present; create if not.

    Delegates the actual creation to cluster/setup-trn2-nodegroup.sh, which is
    idempotent and cost-safe (scales to 0 when done). We only pre-check so the
    common "already there" case is instant and obvious.
    """
    cfg = config.load()
    region = cfg.get("AWS_REGION", "us-east-2")
    cluster = cfg.get("CLUSTER_NAME", "")
    nodegroup = cfg.get("TRN2_NODEGROUP_NAME", "trn2-48xl-ngc")

    ui.banner("Trainium (trn2) node-group setup")
    ui.kv("Cluster", cluster or "(not set in config.env)")
    ui.kv("Node group", nodegroup)
    ui.kv("Region", region)

    if not shell.have("aws"):
        ui.warn("AWS CLI not found - cannot check or create the node group.")
        ui.hint("Install AWS CLI v2 and configure credentials, then re-run.")
        return 1

    # ---- Reuse-if-exists check ---------------------------------------------
    status = _nodegroup_status(cluster, nodegroup, region)
    if status is not None:
        ui.info(f"Node group '{nodegroup}' already exists (status: {status}). Reusing it - nothing to create.")
        if status != "ACTIVE":
            ui.warn(f"It is not ACTIVE yet (status: {status}). Wait for it to become ACTIVE before deploying.")
        ui.hint("Deploy a Neuron model onto it with: oai deploy <id> --compile --managed-ng")
        ui.hint(f"It stays idle at desiredSize=0 (no cost) until 'oai deploy ... --managed-ng' scales it up.")
        return 0

    ui.step(f"No node group named '{nodegroup}' found - running the one-time setup.")
    if dry_run:
        ui.info("Dry-run requested - will validate AZ/capacity only, create nothing.")

    script = paths.ROOT / "cluster" / "setup-trn2-nodegroup.sh"
    if not script.exists():
        ui.fail(
            "cluster/setup-trn2-nodegroup.sh is missing.",
            "This is part of the base framework - check your checkout.",
        )

    args: list[str] = ["--nodegroup", nodegroup]
    if az:
        args += ["--az", az]
    if capacity_type:
        args += ["--capacity-type", capacity_type]
    if dry_run:
        args += ["--dry-run"]

    shell.require_tool("bash", "run the Trainium node-group setup")
    shell.require_tool("eksctl", "create the managed node group")

    rc = shell.run_bash(script, args)
    if rc != 0:
        ui.fail(
            f"Trainium node-group setup did not complete (exit {rc}).",
            "Common causes: no trn2 capacity in any AZ (retry later or use --az), "
            "missing eksctl, or insufficient IAM permissions. See the log above.",
        )
    ui.info("Trainium node group is ready (idle at 0 - no cost until you deploy).")
    ui.hint("Next: oai deploy <neuron-model-id> --compile --managed-ng")
    return 0


def status() -> int:
    """Unified one-view health: cluster, monitoring, and Trainium node group."""
    cfg = config.load()
    region = cfg.get("AWS_REGION", "us-east-2")
    cluster = cfg.get("CLUSTER_NAME", "")
    nodegroup = cfg.get("TRN2_NODEGROUP_NAME", "trn2-48xl-ngc")

    ui.banner("Cluster + monitoring status")

    if not shell.have("aws"):
        ui.warn("AWS CLI not found - cannot query cluster resources.")
        ui.hint("Install AWS CLI v2 and configure credentials, then re-run.")
        return 1

    # ---- Cluster ------------------------------------------------------------
    if _cluster_exists(cluster, region):
        cluster_status = _aws_text(["eks", "describe-cluster", "--name", cluster, "--query", "cluster.status"], region)
        ui.info(f"Cluster '{cluster}': {cluster_status or 'present'}")
    else:
        ui.warn(f"Cluster '{cluster}': not found.")
        ui.hint("Create it with: oai cluster bootstrap")
        return 0

    # ---- Monitoring ---------------------------------------------------------
    if _monitoring_installed():
        ui.info("Monitoring: kube-prometheus-stack installed (Grafana + Prometheus).")
        amp = _amp_workspace_id(cluster, region)
        if amp:
            ui.kv("AMP workspace", amp)
        ui.hint("Grafana: kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n monitoring")
    else:
        ui.warn("Monitoring: not installed.")
        ui.hint("Install it with: oai cluster setup-monitoring")

    # ---- Trainium node group ------------------------------------------------
    _report_neuron_nodegroup(cluster, nodegroup, region)
    return 0


def _report_neuron_nodegroup(cluster: str, nodegroup: str, region: str) -> None:
    import subprocess

    try:
        proc = subprocess.run(  # noqa: S603
            [
                "aws", "eks", "describe-nodegroup",
                "--cluster-name", cluster, "--nodegroup-name", nodegroup,
                "--region", region, "--output", "json",
            ],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        ui.warn("Trainium node group: could not query AWS.")
        return

    if proc.returncode != 0:
        ui.info(f"Trainium node group '{nodegroup}': not created yet.")
        ui.hint("Create it once with: oai cluster setup-neuron")
        return

    try:
        ng = json.loads(proc.stdout)["nodegroup"]
        scaling = ng.get("scalingConfig", {})
        desired = scaling.get("desiredSize", 0)
        ui.info(f"Trainium node group '{ng.get('nodegroupName', nodegroup)}': {ng.get('status', '?')}")
        ui.kv("Desired size", str(desired))
        ui.kv("Capacity type", ng.get("capacityType", "?"))
        if desired == 0:
            ui.info("  Idle (desiredSize=0) - no trn2 instances running, no cost.")
        else:
            ui.warn(f"  Running {desired} node(s) - this is billing.")
    except (json.JSONDecodeError, KeyError):
        ui.warn("Trainium node group: could not parse the description.")
