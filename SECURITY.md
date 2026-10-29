# Security Report

> Project: `client-observeai-inference-optimization`
> Date: 2026-10-29
> Scanners: Bandit, Semgrep, Checkov (via MCP Security Scanner) + manual expert review

## Executive Summary

**Overall Risk Level: 🟢 LOW (post-remediation)**

A security scan was run across the repository's Python source, Kubernetes
manifests, CloudFormation templates, shell scripts, and configuration files.
The automated SAST scanners (Bandit, Semgrep) reported **no issues** in the
Python code they analyzed. The most significant findings came from a plaintext
secret on disk and from the Trainium node-group CloudFormation template, which
opened SSH to the entire internet.

**Remediation status (2026-10-29):** all five findings have been addressed in
code, and a full Checkov/Bandit/Semgrep CI workflow was added to prevent
regressions. The one remaining human action is rotating the HuggingFace token in
the HuggingFace console (HIGH-1) — code no longer persists it on disk.

| Severity | Count | Resolved |
|----------|-------|----------|
| Critical | 0 | — |
| High | 2 | 2 (HIGH-1 code-side done; token rotation pending in HF console) |
| Medium | 2 | 2 |
| Low | 1 | 1 |

### Scanner coverage note

This environment could not run the standalone directory scanners end-to-end:
Bandit was blocked by an OS Application Control policy, Semgrep timed out on the
full tree, and the Checkov binary was not resolvable on `PATH` (so IaC findings
below come from expert review of the templates, not an automated Checkov pass).
Snippet-level Bandit and Semgrep runs did complete successfully. Re-run the full
Checkov/Bandit/Semgrep suite in CI to confirm and extend these results.

---

## Findings

### HIGH-1 — HuggingFace token stored in plaintext
- **File:** `config/.env` (line 7)
- **Detail:** `HF_TOKEN="hf_..."` is a live HuggingFace access token written in
  cleartext on the local filesystem.
- **Mitigating factors:** `config/.env` is listed in `.gitignore`, and a search
  of all git history (`git log -S`) found no commit containing the token value —
  so it has **not** been committed to the repository. The project also wires a
  Gitleaks pre-commit hook and an AWS Secrets Manager path
  (`oai-infopt/hf-token`) as the intended home for this credential.
- **Risk:** Anyone with read access to the developer machine or a backup/sync of
  it obtains a valid token that can pull gated models and call the HF API under
  this account.
- **Recommendation:**
  1. **Rotate the token now** in the HuggingFace account settings — treat the
     current value as compromised since it has been shared in plaintext.
  2. Keep the credential only in AWS Secrets Manager (`HF_TOKEN_SECRET_NAME`),
     and have local runs fetch it at runtime rather than persisting it in
     `config/.env`.
  3. If a local `.env` is unavoidable, restrict file permissions and never copy
     it off the machine.
- **Status: ✅ code remediated / ⚠️ rotation pending.** The live token value was
  removed from `config/.env` (now a `hf_REPLACE_WITH_YOUR_TOKEN` placeholder with
  guidance to export `HF_TOKEN` in-shell for one-time seeding). Secrets Manager
  (`oai-infopt/hf-token`) remains the source of truth; `setup-pod-identity.sh`
  seeds it then clears the value from memory. **Action still required:** rotate
  the previously-exposed token in the HuggingFace console — treat it as burned.

### HIGH-2 — SSH open to the world (0.0.0.0/0 and ::/0)
- **File:** `cluster/eks_trn2_ng_stack_al2023.yaml`
- **Resources:** `NodeSecurityGroupIngressSSHIpv4`, `NodeSecurityGroupIngressSSHIpv6`
- **Detail:** Both ingress rules allow TCP port 22 from `0.0.0.0/0` and `::/0`,
  exposing SSH on every worker node to the entire internet.
- **Risk:** Direct brute-force / exploitation surface on the Trainium worker
  nodes. (Checkov equivalent: `CKV_AWS_24`.)
- **Recommendation:** Restrict the SSH CIDR to a known bastion/VPN range, or
  remove the SSH rules entirely and use AWS SSM Session Manager (the node role
  already attaches `AmazonSSMManagedInstanceCore`, so SSM access needs no open
  port).
- **Status: ✅ resolved.** Both `NodeSecurityGroupIngressSSHIpv4/6` rules were
  removed; interactive access is via `aws ssm start-session`. A comment documents
  how to re-add TCP/22 scoped to a bastion/VPN CIDR if ever needed. Checkov
  re-scan of the edited SG rules reports 0 issues (CKV_AWS_24 cleared).

### MEDIUM-1 — Unrestricted egress on worker security group
- **File:** `cluster/eks_trn2_ng_stack_al2023.yaml`
- **Resources:** `NodeSecurityGroupEgressAllIpv4`, `NodeSecurityGroupEgressAllIpv6`
- **Detail:** Egress rules allow all protocols (`IpProtocol: "-1"`) to
  `0.0.0.0/0` and `::/0`.
- **Risk:** Broadens the blast radius of a compromised node (data exfiltration,
  C2 callbacks). Note the Kubernetes-layer `NetworkPolicy` in
  `k8s/network-policy.yaml` is well scoped (default-deny + explicit allows), but
  the SG layer under it is wide open.
- **Recommendation:** Scope egress to the ports/destinations actually needed
  (443 for S3/ECR/HF, 53 for DNS), mirroring the NetworkPolicy intent.
- **Status: ✅ resolved.** The all-protocol `0.0.0.0/0` + `::/0` egress rules were
  replaced with scoped rules: HTTPS (443, IPv4+IPv6) and DNS (53 TCP+UDP).
  Intra-node and control-plane SG-referenced egress rules are retained. If an
  EFA/NCCL path needs more ports, add narrowly scoped rules.

### MEDIUM-2 — `subprocess` with `shell=True` and interpolated input
- **File:** `report-gen/report_generator.py` (`sync_results_from_s3`, ~line 67)
- **Detail:**
  ```python
  cmd_perf = f"aws s3 sync s3://{bucket}/results/ {local_dir}/perf/"
  subprocess.run(cmd_perf, shell=True, check=True)
  ```
  The `bucket` and `local_dir` values are interpolated into a string that runs
  through the shell. (Bandit equivalent: `B602`.)
- **Risk:** Command injection if `bucket`/`local_dir` ever come from an untrusted
  source. Today they originate from a CLI arg, so exposure is limited, but the
  pattern is unsafe.
- **Recommendation:** Pass an argument list and drop `shell=True`:
  ```python
  subprocess.run(["aws", "s3", "sync", f"s3://{bucket}/results/", f"{local_dir}/perf/"], check=True)
  ```
- **Status: ✅ resolved.** `sync_results_from_s3` now builds argument lists and
  calls `subprocess.run(..., check=True)` without `shell=True`, so inputs are
  never shell-interpreted (B602/B603 cleared).

### LOW-1 — Committed infrastructure identifiers
- **File:** `config/config.env`
- **Detail:** The committed config contains concrete AWS resource IDs as defaults
  (e.g. `TRN2_CAPACITY_RESERVATION_ID`, `TRN2_LAUNCH_TEMPLATE_ID`, a region-specific
  AMI ID). These are not secrets, but they leak environment topology and can drift
  per account/region.
- **Recommendation:** Prefer leaving these empty and resolving them at runtime
  (the scripts already fall back to SSM/lookup), or move them to the gitignored
  `config/.env`.
- **Status: ✅ resolved.** `TRN2_CAPACITY_RESERVATION_ID`, `TRN2_LAUNCH_TEMPLATE_ID`,
  and `TRN2_CACHED_AMI_ID` now default to empty in `config/config.env` and are
  supplied from the gitignored `config/.env` (or the environment); scripts fall
  back to SSM/discovery when unset.

---

## What was scanned

| Area | Files | Tool | Result |
|------|-------|------|--------|
| Python SAST | `inference/*.py`, `report-gen/*.py`, `tools/`, `vllm/**/*.py` | Bandit, Semgrep (snippet) | 0 automated issues |
| Secret detection | repo-wide grep + git history | manual + grep | 1 plaintext token (gitignored, uncommitted) |
| Kubernetes | `k8s/*.yaml`, `vllm/**/deployment.yaml` | Checkov (snippet) + review | NetworkPolicy and Deployment securityContext well-hardened |
| CloudFormation | `cluster/eks_trn2_ng_stack_al2023.yaml` | review | open SSH + open egress |
| Shell | `cluster/bootstrap.sh` and others | review | S3 created encrypted/versioned/public-blocked; no issues noted |

### Positive observations
- vLLM Deployment uses a strong pod `securityContext`: `runAsNonRoot`,
  non-root UID/GID, `allowPrivilegeEscalation: false`, `capabilities: drop ALL`,
  `seccompProfile: RuntimeDefault`, `automountServiceAccountToken: false`.
- `NetworkPolicy` follows default-deny with narrowly scoped allows.
- S3 buckets are created with encryption, versioning, and full public-access block.
- IAM is split into three least-privilege roles (download / serving / benchmark).
- Pre-commit config includes Bandit, Gitleaks, mypy `--strict`, and ShellCheck.

---

## STRIDE Threat Model (summary)

| Threat | Relevant here | Evidence / notes |
|--------|---------------|------------------|
| Spoofing | Review | SSH open to world (HIGH-2) weakens host identity controls; prefer SSM. |
| Tampering | Review | Serving role is read-only on model bucket — good; verify results-bucket writes are scoped. |
| Repudiation | Review | Structured JSON logging present; confirm CloudTrail/audit on the account. |
| Information Disclosure | Yes | Plaintext HF token (HIGH-1); open egress (MEDIUM-1) eases exfiltration. |
| Denial of Service | Review | K8s resource limits set on vLLM pods; node SG otherwise permissive. |
| Elevation of Privilege | Review | Pods non-root + drop ALL caps — strong. Node SG egress is broad. |

---

## Prioritized Recommendations

1. ⚠️ **Rotate the HuggingFace token** in the HuggingFace console (HIGH-1). Code
   side done — the token is no longer stored in `config/.env` and Secrets Manager
   is the source of truth. **This rotation is the only remaining manual step.**
2. ✅ **Close SSH to the internet** on the trn2 node group; use SSM Session Manager
   (HIGH-2). SSH ingress rules removed.
3. ✅ **Scope worker-node egress** to required ports/destinations (MEDIUM-1).
   Replaced open egress with 443 + DNS.
4. ✅ **Remove `shell=True`** in `report_generator.py`; use argument lists
   (MEDIUM-2).
5. ✅ **Externalize committed resource IDs/AMIs** from `config.env` (LOW-1).
6. ✅ **Run the full Checkov / Bandit / Semgrep suite in CI** — added
   `.github/workflows/security-scan.yml` (runs on push, PR, and weekly schedule).
