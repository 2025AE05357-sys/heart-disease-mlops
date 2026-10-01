"""Kubernetes manifest rules that a server-side dry run cannot catch."""

from pathlib import Path

import yaml

DEPLOYMENT = Path(__file__).resolve().parents[1] / "k8s" / "base" / "deployment.yaml"


def test_api_pods_drain_before_shutdown():
    """A terminating pod keeps serving until the Service has stopped routing to it.

    On SIGTERM uvicorn closes its socket at once, while kube-proxy and the LoadBalancer still send
    it new connections for a moment: without a preStop delay, 15 of 602 requests were reset
    across three rollouts on Docker Desktop.
    """
    pod = yaml.safe_load(DEPLOYMENT.read_text(encoding="utf-8"))["spec"]["template"]["spec"]
    api = next(c for c in pod["containers"] if c["name"] == "api")
    delay = api.get("lifecycle", {}).get("preStop", {}).get("sleep", {}).get("seconds", 0)
    assert delay >= 5, "the api container needs a preStop sleep of at least 5 s"
    # The grace period covers the delay plus uvicorn finishing its in-flight requests.
    assert pod["terminationGracePeriodSeconds"] >= delay + 10
