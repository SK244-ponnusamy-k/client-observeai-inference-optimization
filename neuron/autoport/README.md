# AutoPort boxes

Porting a model to Trainium with AutoPort needs a real trn2.48xlarge: the agent
compiles, runs and validates the port on hardware. These scripts stand that box
up inside a Capacity Block and prepare it, so the only manual step is signing in
to Claude Code and pasting a prompt.

| File | Runs on | Does |
|---|---|---|
| `launch-box.sh` | your laptop | trn2.48xlarge from the Neuron vLLM DLAMI into a Capacity Block; private subnet, SSM only |
| `setup-box.sh` | the box | vllm-neuron source workspace (editable), weights from S3, Claude Code + Neuron skills |
| `<model>/PROMPT.md` | the box | the AutoPort prompt with the model's architecture facts and acceptance bar |
| `backup-workspace.sh` | the box (cron) | snapshots the workspace diff + new files to S3; the box's disk dies with the block |

```bash
bash neuron/autoport/launch-box.sh --reservation cr-0123... --name autoport-gemma-4-31b --dry-run
bash neuron/autoport/launch-box.sh --reservation cr-0123... --name autoport-gemma-4-31b
# on the box (Session Manager), as ubuntu:
bash setup-box.sh --model-name gemma4 --weights-folder gemma-4-31B-it
cd ~/autoport/gemma4/vllm-neuron && claude
# ... done porting:
bash neuron/autoport/launch-box.sh --terminate i-0123...
```

**Back up the port.** The box's root volume is deleted when the Capacity Block
ends (EC2 reclaims the node up to ~40 min before the end time). Install
`backup-workspace.sh` as a cron job on the box (hourly, denser near the end):

```
7 * * * * ubuntu flock -n /tmp/oai-autoport-backup.lock /usr/local/bin/oai-autoport-backup \
    --workspace /home/ubuntu/autoport/<model> --dest s3://<bucket>/autoport-workspaces/<model>
```

Restore anywhere: check out the base commit in `latest/MANIFEST.json` and
`git apply latest/workspace.patch`.

**Capacity:** a Capacity Block with one instance serves either this box or the
EKS `neuron-inf` NodePool, not both. Terminate the box to hand the slot back.

**Which AutoPort?** These boxes target the vLLM Neuron backend
(`neuron-framework-autoport-vllm-neuron`), which the Neuron SDK now recommends:
NxD Inference is in maintenance mode, and SDK 2.33 notes that its Trn2/Trn3
support ends in 2.34. Existing NxD Inference ports keep working on images up
to SDK 2.33.
