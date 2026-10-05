"""
Lifecycle orchestration: download, deploy, stop, status.

These wrap the generated bash scripts, adding preflight checks and dynamic
instance selection so the user gets one friendly command instead of a sequence
of kubectl/aws/bash invocations. Every failure path explains the next step.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import catalog, config, generate, instances, paths, preflight, shell, ui
from .catalog import ModelSpec


def _ensure_generated(spec: ModelSpec) -> None:
    """Generate files if the deploy script is missing, so users can't forget."""
    deploy_sh = paths.model_dir(spec.id) / "deploy.sh"
    if not deploy_sh.exists():
        ui.step(f"Deployment files for '{spec.id}' not found - generating them now.")
        result = generate.generate(spec)
        for f in result.created:
            ui.info(f"created  {f}")
        if result.skipped_protected:
            ui.warn("Some files could not be generated (protected). Run 'oai model generate --force' if intended.")


def _preflight_neuron_nodegroup(spec: ModelSpec, cfg: config.Config, region: str) -> None:
    """
    Before using the managed node-group path, make sure the group exists.

    If it does not, stop with a clear instruction to run the one-time setup -
    we never auto-create expensive infrastructure implicitly during a deploy.
    """
    from . import cluster as cluster_mod

    cluster_name = cfg.get("CLUSTER_NAME", "")
    node_group = spec.instances.node_group
    status = cluster_mod._nodegroup_status(cluster_name, node_group, region)
    if status is None:
        ui.fail(
            f"The managed node group '{node_group}' does not exist yet, so this Neuron model "
            "cannot be deployed onto it.",
            "Create it once (idempotent - reused if it already exists):\n"
            "   -> oai cluster setup-neuron\n"
            "   -> then re-run this deploy.",
        )
    ui.info(f"Managed node group '{node_group}' found (status: {status}).")


def _preflight_cluster() -> None:
    """Warn early if kubectl cannot reach a cluster."""
    if not shell.have("kubectl"):
        ui.fail(
            "kubectl not found - cannot talk to the cluster.",
            "Install kubectl and run: aws eks update-kubeconfig --name <cluster> --region <region>.",
        )
    out = shell.kubectl_json(["version", "-o", "json"])
    if out is None:
        ui.warn("kubectl is installed but could not reach a cluster.")
        ui.hint("Run: aws eks update-kubeconfig --name <cluster> --region <region>")


# ---------------------------------------------------------------------------
def download(model_id: str) -> int:
    spec = catalog.load(model_id)
    _ensure_generated(spec)
    _preflight_cluster()

    if spec.source.gated:
        ui.step("This is a gated model - the download job needs the HuggingFace token secret 'hf-token'.")
        ui.hint(f"Accept the license first: https://huggingface.co/{spec.source.hf_id}")

    script = paths.download_dir(spec.id) / "download.sh"
    if not script.exists():
        ui.fail(f"download.sh missing for '{model_id}'.", f"Run: oai model generate {model_id}")
    rc = shell.run_bash(script)
    if rc != 0:
        ui.fail(
            f"Download of '{model_id}' did not complete (exit {rc}).",
            "Check the job logs shown above. Common causes: missing HF token secret, "
            "license not accepted, or the download ServiceAccount not created (run cluster/bootstrap.sh).",
        )
    ui.info(f"Weights for '{model_id}' are in S3. Next: oai deploy {model_id}")
    return 0


def deploy(
    model_id: str,
    *,
    hw_override: str | None = None,
    compile_first: bool = False,
    managed_ng: bool = False,
    validate: bool = False,
    do_benchmark: bool = False,
    do_quality: bool = False,
    profile: str | None = None,
    skip_instance_check: bool = False,
    tag: str | None = None,
    manifest: str | None = None,
    quality_config: str | None = None,
    quality_dataset_key: str | None = None,
    quality_dataset_file: str | None = None,
    quality_force_stage: bool = False,
    quality_sheets: list[str] | None = None,
    dump_samples: bool = False,
    max_dump_samples: int = 200,
    dump_inputs: bool = False,
    detach: bool = False,
    auto_stop: bool = False,
) -> int:
    spec = catalog.load(model_id)

    problems = [p for p in catalog.validate(spec) if "instance_hourly_usd is 0" not in p]
    if problems:
        ui.error("The catalog entry has problems that must be fixed before deploying:")
        for p in problems:
            ui.hint(p)
        return 1

    _ensure_generated(spec)
    _preflight_cluster()
    # Refuse a duplicate of the same model+tag (collides + contaminates results).
    # include_deployment=True: a leftover Deployment for this tag also blocks.
    preflight.fail_if_active(spec.id, tag, include_deployment=True)

    cfg = config.load()
    region = cfg.get("AWS_REGION", "us-east-2")

    # ---- Dynamic instance selection -----------------------------------------
    chosen = hw_override
    if hw_override:
        ui.step(f"Using the instance you specified: {hw_override} (skipping availability check).")
    elif skip_instance_check:
        chosen = spec.preferred_instance
        ui.step(f"Skipping availability check - using preferred instance {chosen}.")
    else:
        selection = instances.select(spec.instances.candidates, region)
        chosen = selection.instance_type
        if selection.checked:
            ui.info(f"Selected {chosen} after skipping: {', '.join(selection.checked)}")
        else:
            ui.info(f"Selected {chosen}.")

    # ---- Preflight: tensor-parallel size must fit the instance's accelerators -
    # A TP=4 model on a 1-GPU instance would OOM / never schedule. Fail fast with
    # a clear message instead of a confusing Pending/CrashLoop. Neuron TP = cores.
    tp = spec.instances.neuron_cores if spec.is_neuron else spec.instances.tensor_parallel_size
    if chosen and tp:
        tp_err = instances.check_tp_fits(chosen, tp)
        if tp_err:
            ui.fail(
                f"'{spec.id}' cannot run on {chosen}: {tp_err}",
                "Pick a larger instance with --hw, or adjust tensor_parallel_size in the catalog.",
            )

    # ---- Build args for the generated deploy.sh -----------------------------
    script = paths.model_dir(spec.id) / "deploy.sh"
    if not script.exists():
        ui.fail(f"deploy.sh missing for '{model_id}'.", f"Run: oai model generate {model_id}")

    args: list[str] = []
    if spec.is_neuron:
        if tag:
            ui.warn("--tag is not supported for Neuron models (they use the managed node group); ignoring it.")
            tag = None
        if compile_first:
            args += ["--compile"]
        want_managed = managed_ng or (
            chosen and not chosen.startswith(spec.neuron.instance_family)  # type: ignore[union-attr]
        )
        if want_managed and spec.instances.node_group:
            _preflight_neuron_nodegroup(spec, cfg, region)
            args += ["--managed-ng"]
            ui.step("Using the managed node-group path for this deploy.")
        args += ["--cores", str(spec.instances.neuron_cores)]
    else:
        args += ["--hw", chosen]
        if tag:
            args += ["--tag", tag]
            ui.step(f"Deploying as an independent tagged copy: tag={tag} (resource names suffixed -{tag}).")

    if validate:
        if detach:
            ui.warn("--validate is ignored with --detach (nothing waits locally to run the smoke test).")
        else:
            args += ["--validate"]

    # --detach: apply the Deployment and return immediately WITHOUT blocking on
    # readiness, so the whole pipeline can be submitted server-side and the user
    # can close their terminal. deploy.sh's --no-wait skips the kubectl wait and
    # exits 0 even though the model may still be provisioning; the benchmark and
    # quality Jobs each wait for the endpoint in-cluster.
    if detach and not spec.is_neuron:
        args += ["--no-wait"]

    ui.banner(f"Deploy: {spec.id}  ({chosen}){f'  tag={tag}' if tag else ''}")
    rc = shell.run_bash(script, args)
    if rc != 0:
        _explain_deploy_failure(spec, chosen)
        return rc

    if detach:
        ui.info(
            f"'{spec.id}' submitted on {chosen}{f' (tag={tag})' if tag else ''}. "
            "The model may still be provisioning; Jobs will wait for it in-cluster."
        )
    else:
        ui.info(f"'{spec.id}' deployed on {chosen}{f' (tag={tag})' if tag else ''}.")

    # ---- Optional performance + quality pipeline ----------------------------
    want_benchmark = do_benchmark or spec.benchmark.auto
    if not want_benchmark and not do_quality:
        if auto_stop:
            ui.warn(
                "--auto-stop has no pipeline to wait for (no --benchmark/--quality). "
                "Skipping auto-stop; the model stays up. Stop it with "
                f"'oai stop {spec.id}{f' --tag {tag}' if tag else ''}'."
            )
        ui.hint(
            f"To evaluate: oai benchmark {spec.id} and/or oai quality {spec.id} "
            "(or add --benchmark / --quality to deploy)"
        )
        return 0

    if auto_stop and spec.is_neuron:
        ui.warn("--auto-stop is not supported for Neuron models (managed node group lifecycle); ignoring it.")
        auto_stop = False

    # One timestamp groups realtime, batch, and quality. It also makes the exact
    # marker key deterministic before any Job is submitted.
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    ui.kv("Run group", run_timestamp)

    bench_mod = None
    if want_benchmark:
        from . import benchmark as bench_mod

    if want_benchmark and do_quality and bench_mod is not None:
        stages = [
            *bench_mod.selected_profiles(spec, profile, spec.benchmark.skip_batch),
            "quality",
        ]
        ui.info(
            f"Submitting an asynchronous pipeline: {' -> '.join(stages)}. "
            "The command returns after all Jobs are created; use the printed run group to monitor it."
        )

    if want_benchmark and bench_mod is not None:

        ui.banner(f"Auto-benchmark: {spec.id}")
        rc = bench_mod.run(
            model_id,
            profile=profile,
            skip_batch=spec.benchmark.skip_batch,
            tag=tag,
            hw=chosen,
            manifest=manifest,
            run_timestamp=run_timestamp,
            # When quality follows, return immediately after submitting both
            # performance Jobs so the quality Job can also be submitted up front.
            detach=do_quality,
        )
        if rc != 0:
            return rc

    if do_quality:
        from . import benchmark as bench_mod
        from . import quality as quality_mod

        wait_for_marker = None
        if want_benchmark:
            last_profile = bench_mod.completion_profile(spec, profile, spec.benchmark.skip_batch)
            tag_segment = f"{tag}/" if tag else ""
            wait_for_marker = (
                f"results/{run_timestamp}/{tag_segment}_markers/{spec.id}/{last_profile}.done"
            )

        ui.banner(f"Auto-quality: {spec.id}")
        rc = quality_mod.run(
            model_id,
            tag=tag,
            hw=chosen,
            config=quality_config,
            dataset_key=quality_dataset_key,
            dataset_file=quality_dataset_file,
            force_stage=quality_force_stage,
            sheets=quality_sheets,
            dump_samples=dump_samples,
            max_dump_samples=max_dump_samples,
            dump_inputs=dump_inputs,
            run_timestamp=run_timestamp,
            wait_for_marker=wait_for_marker,
            # Combined pipeline must survive terminal closure; all three Jobs
            # have already been submitted and ordering is enforced in-cluster.
            detach=want_benchmark or auto_stop,
        )
        if rc != 0 and not auto_stop:
            return rc

    # ---- Optional auto-stop cleanup Job -------------------------------------
    # Submits an in-cluster Job that waits for the whole pipeline to finish, then
    # stops this model+tag (frees its GPU node). Runs server-side so it works
    # even with --detach and a closed terminal.
    if auto_stop:
        _submit_auto_stop(spec, chosen, tag, run_timestamp, want_benchmark, do_quality,
                          profile, quality_sheets)

    return 0


def _submit_auto_stop(
    spec: ModelSpec,
    instance: str | None,
    tag: str | None,
    run_timestamp: str,
    want_benchmark: bool,
    do_quality: bool,
    profile: str | None,
    quality_sheets: list[str] | None,
) -> None:
    """Submit an in-cluster cleanup Job that stops the model after the pipeline.

    The Job waits for the final stage to finish (quality Jobs if quality was
    requested, otherwise the benchmark completion marker), then stops this
    model+tag to free its GPU node. It runs server-side so it survives the
    local terminal closing (works with --detach).
    """
    script = paths.ROOT / "inference" / "run-autostop.sh"
    if not script.exists():
        ui.warn(
            "inference/run-autostop.sh is missing; skipping auto-stop. "
            f"Stop manually later: oai stop {spec.id}{f' --tag {tag}' if tag else ''}."
        )
        return

    # Determine what the cleanup Job should wait on.
    #   - quality requested  -> wait for the quality Job(s) to complete (no marker exists)
    #   - benchmark only     -> wait for the final benchmark completion marker
    args = ["--model", spec.id, "--run-timestamp", run_timestamp]
    if tag:
        args += ["--tag", tag]
    if do_quality:
        args += ["--wait-mode", "quality-jobs"]
        # Which sheets run decides how many quality Jobs the cleanup must await.
        for sheet in (quality_sheets or ["org", "xl"]):
            args += ["--quality-sheet", sheet]
    elif want_benchmark:
        from . import benchmark as bench_mod

        last_profile = bench_mod.completion_profile(spec, profile, spec.benchmark.skip_batch)
        # Benchmark-only: wait on the final benchmark Job via KUBECTL (not the S3
        # marker). The auto-stop ServiceAccount has Kubernetes API access but no
        # AWS/S3 credentials, so the old marker (boto3) path failed with
        # NoCredentialsError. bench-jobs mode watches the Job's terminal state and
        # needs no AWS. last_profile is 'batch' normally, or 'realtime' when batch
        # is skipped, matching the benchmark Job that writes the final result.
        args += ["--wait-mode", "bench-jobs", "--wait-for-profile", last_profile]

    ui.banner(f"Auto-stop: {spec.id}{f'  (tag={tag})' if tag else ''}")
    rc = shell.run_bash(script, args)
    if rc != 0:
        ui.warn(
            f"Auto-stop Job submission returned non-zero (exit {rc}). "
            f"If the model is not stopped when the pipeline ends, run "
            f"'oai stop {spec.id}{f' --tag {tag}' if tag else ''}' manually."
        )
    else:
        ui.info(
            "Auto-stop Job submitted. It will stop this model automatically once the "
            "pipeline finishes, freeing the GPU node - even if you close this terminal."
        )


def _explain_deploy_failure(spec: ModelSpec, instance: str) -> None:
    ui.error(f"Deploy of '{spec.id}' did not become ready.")
    ui.hint("Most common causes and fixes:")
    ui.hint("- Node still provisioning: accelerator capacity can take a few minutes; re-run deploy.")
    ui.hint(f"- No capacity for {instance}: add more types to instances.candidates, or try another region.")
    ui.hint(f"- Weights missing in S3: run 'oai download {spec.id}' first.")
    ui.hint(f"- Inspect: kubectl describe pod -l app={spec.deployment_name} -n oai-infopt")
    ui.hint(f"- Logs:    kubectl logs deployment/{spec.deployment_name} -n oai-infopt --tail=50")


def stop(model_id: str, *, managed_ng: bool = False, assume_yes: bool = False, tag: str | None = None) -> int:
    spec = catalog.load(model_id)
    script = paths.model_dir(spec.id) / "stop.sh"
    if not script.exists():
        ui.fail(f"stop.sh missing for '{model_id}'.", f"Run: oai model generate {model_id}")
    args = ["--managed-ng"] if managed_ng else []
    if tag:
        if spec.is_neuron:
            ui.warn("--tag is not supported for Neuron models; ignoring it.")
        else:
            args += ["--tag", tag]
    env = {"OAI_ASSUME_YES": "true"} if assume_yes else {}
    rc = shell.run_bash(script, args, env=env)
    if rc != 0:
        ui.fail(
            f"Stop of '{model_id}' reported an error (exit {rc}).",
            "The deployment may already be gone. Verify with 'oai status'. "
            "If a node is still running, delete it manually to stop billing.",
        )
    return 0


def status() -> int:
    cfg = config.load()
    ns = cfg.get("BENCHMARK_NAMESPACE", "oai-infopt")
    ui.banner("Running models")

    if not shell.have("kubectl"):
        ui.warn("kubectl not found - cannot query the cluster.")
        ui.hint("Install kubectl and configure it, then re-run 'oai status'.")
        return 0

    import json

    out = shell.kubectl_json(
        ["get", "deployments", "-n", ns, "-l", "app.kubernetes.io/component=inference-server", "-o", "json"]
    )
    if out is None:
        ui.warn("Could not reach the cluster (or no inference deployments found).")
        ui.hint("Check: aws eks update-kubeconfig --name <cluster> --region <region>")
        return 0

    try:
        items = json.loads(out).get("items", [])
    except json.JSONDecodeError:
        items = []

    if not items:
        ui.info("No models are currently running. (Nothing is costing accelerator time.)")
        return 0

    print(f"  {'MODEL':<28}{'READY':<8}{'AVAILABLE':<10}")
    for d in items:
        name = d["metadata"]["name"]
        model = d["metadata"].get("labels", {}).get("model", name)
        status_obj = d.get("status", {})
        ready = f"{status_obj.get('readyReplicas', 0)}/{status_obj.get('replicas', 0)}"
        avail = "yes" if status_obj.get("availableReplicas", 0) > 0 else "no"
        print(f"  {model:<28}{ready:<8}{avail:<10}")

    ui.hint("Stop a model to free its node: oai stop <id>")

    # Show GPU/Neuron nodes so cost is visible.
    nodes_out = shell.kubectl_json(["get", "nodes", "-o", "json"])
    if nodes_out:
        try:
            nodes = json.loads(nodes_out).get("items", [])
        except json.JSONDecodeError:
            nodes = []
        accel = []
        for n in nodes:
            labels = n["metadata"].get("labels", {})
            itype = labels.get("node.kubernetes.io/instance-type", "")
            if itype.startswith(("g", "p", "trn", "inf")):
                accel.append(itype)
        if accel:
            ui.warn(f"Accelerator nodes running (billed): {', '.join(sorted(accel))}")
    return 0
