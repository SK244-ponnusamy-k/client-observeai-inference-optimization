"""
oai - the ObserveAI Inference framework onboarding + lifecycle toolkit.

This package turns a single catalog entry (catalog/models/<id>.yaml) into a full,
security-hardened deployment: Kubernetes manifests, download jobs, deploy/stop
scripts, benchmark run manifests, and (for Trainium) the Neuron compile job.

Public surface is the `oai` CLI (see cli.py). Everything else is internal.
"""

__version__ = "1.0.0"
