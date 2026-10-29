"""Pytest fixtures and CLI flags for llm-d conformance tests.

CLI flags here must stay in sync with ``cli.py`` (``flag_map`` + boolean
flags). ``cli.py:main()`` translates user flags into pytest ``-o`` options
and runs ``tests/test_conformance.py``.

Test case selection (``pytest_generate_tests``):
  - ``--profile`` — load a profile YAML and resolve its test case names
  - ``--testcase`` — comma-separated names from ``--testcase-dir``
  - neither — run every YAML under ``--testcase-dir``
  Parametrizes the class-scoped ``tc`` fixture (one ``TestCase`` per case).

Session-scoped fixtures:
  - ``deployer`` — kubectl wrapper; stops port-forwards at session end
  - ``report`` — JSON report; finalized into ``--report-dir``
  - ``no_cleanup`` / ``test_mode`` / ``mock_mode`` / ``guidellm_image`` —
    thin accessors for ``--nocleanup``, ``--mode``, ``--mock``,
    ``--guidellm-image``

Class-scoped fixtures (per ``tc``):
  - ``endpoint`` / ``client`` — gateway port-forward (inference)
  - ``pod_endpoint`` / ``pod_client`` — direct pod port-forward
    (health + ``/v1/models``; EPP only routes inference)
  - ``scraper`` — Prometheus metrics via kubectl exec / port-forward

Notable options: ``--mode`` (deploy|discover|cache), ``--mock``,
``--model-source``, ``--render-image``, auth/pull-secret, PVC storage,
P/D node selectors, GuideLLM image override.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conformance.config import (
    TestCase,
    apply_manifest_model_config,
    filter_by_names,
    load_profile,
    load_testcases_from_dir,
    resolve_profile,
)
from conformance.client import LLMClient
from conformance.deployer import Deployer
from conformance.metrics import Scraper
from conformance.report import Report

log = logging.getLogger("conformance")


def pytest_addoption(parser):
    parser.addoption("--profile", default="", help="Profile YAML path")
    parser.addoption("--testcase", default="", help="Test case name(s), comma-separated")
    parser.addoption("--testcase-dir", default="configs/testcases", help="Test case directory")
    parser.addoption("--platform", default="any", help="Platform: any, ocp, aks, gks")
    parser.addoption("--namespace", default="llm-conformance-test", help="Kubernetes namespace")
    parser.addoption("--kubeconfig", default="", help="Path to kubeconfig")
    parser.addoption("--report-dir", default="reports", help="Report output directory")
    parser.addoption("--mode", default="deploy", help="Mode: deploy, discover, cache")
    parser.addoption("--model-source", default="hf", help="Model source: hf, pvc")
    parser.addoption("--model", default="", help="Override model name")
    parser.addoption("--endpoint", default="", help="Service URL for discover mode")
    parser.addoption("--mock", default="", help="Simulator image (no GPU needed)")
    parser.addoption("--render-image", default="", help="vLLM CPU image for tokenizer render sidecar")
    parser.addoption("--pull-secret", default="", help="Pull secret name")
    parser.addoption("--bearer-token", default="", help="Bearer token for auth")
    parser.addoption("--disable-auth", action="store_true", help="Disable WASM auth")
    parser.addoption("--nocleanup", action="store_true", help="Keep resources after test")
    parser.addoption("--storage-class", default="", help="StorageClass for PVC")
    parser.addoption("--storage-size", default="", help="Override PVC size")
    parser.addoption("--guidellm-image", default="", help="GuideLLM benchmark image override")
    parser.addoption("--decode-node-selector", default="", help="Node selector for decode pods (key=value)")
    parser.addoption("--prefill-node-selector", default="", help="Node selector for prefill pods (key=value)")


def pytest_configure(config):
    """Record the active --profile so 3.6-only Praxis phases can gate at collection time."""
    from maas.praxis import set_profile_path

    set_profile_path(config.getoption("--profile"))


def _resolve_test_cases(config) -> list[TestCase]:
    testcase_dir = config.getoption("--testcase-dir")
    profile_path = config.getoption("--profile")
    testcase_names = config.getoption("--testcase")

    if profile_path:
        profile = load_profile(profile_path)
        cases = resolve_profile(profile, testcase_dir)
    else:
        cases = load_testcases_from_dir(testcase_dir)
        if testcase_names:
            names = [n.strip() for n in testcase_names.split(",")]
            cases = filter_by_names(cases, names)

    manifest_dir = Path(config.rootpath) / "deploy" / "manifests"
    model_override = config.getoption("--model")
    if config.getoption("--mode") == "discover":
        if model_override:
            for tc in cases:
                tc.model.name = model_override
        return cases

    for tc in cases:
        apply_manifest_model_config(
            tc,
            manifest_dir / tc.deployment.manifest_path,
            model_name_override=model_override,
        )
    return cases


def pytest_generate_tests(metafunc):
    if "tc" in metafunc.fixturenames:
        cases = _resolve_test_cases(metafunc.config)
        metafunc.parametrize("tc", cases, ids=[tc.name for tc in cases], scope="class")


def pytest_collection_modifyitems(config, items):
    """Deselect phases whose ``applies_when(tc)`` predicate is false for their test case.

    Phases a test case's config turns off are not applicable rather than skipped,
    so they do not appear in the run; runtime skips (deploy failed, missing
    manifest, not enough GPUs) still report as SKIPPED.
    """
    selected, deselected = [], []
    for item in items:
        marker = item.get_closest_marker("applies_when")
        callspec = getattr(item, "callspec", None)
        tc = callspec.params.get("tc") if callspec else None
        if marker and tc is not None and not marker.args[0](tc):
            deselected.append(item)
        else:
            selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.fixture(scope="session")
def deployer(request) -> Deployer:
    d = Deployer(
        kubeconfig=request.config.getoption("--kubeconfig"),
        platform=request.config.getoption("--platform"),
        namespace=request.config.getoption("--namespace"),
        model_source=request.config.getoption("--model-source"),
        mock_image=request.config.getoption("--mock"),
        model_override=request.config.getoption("--model"),
        render_image=request.config.getoption("--render-image"),
        pull_secret=request.config.getoption("--pull-secret"),
        disable_auth=request.config.getoption("--disable-auth"),
        decode_node_selector=request.config.getoption("--decode-node-selector"),
        prefill_node_selector=request.config.getoption("--prefill-node-selector"),
    )
    # The test namespace belongs to the suite: services left by a killed run hold
    # GPUs that this run needs. --nocleanup keeps them for inspection.
    no_cleanup = request.config.getoption("--nocleanup")
    if request.config.getoption("--mode") == "deploy" and not no_cleanup:
        for name in d.remove_stale_services():
            print(f"Deleted stale LLMInferenceService '{name}' from a previous run")
    yield d
    _teardown_deployer(d, no_cleanup)


def _teardown_deployer(deployer: Deployer, no_cleanup: bool) -> None:
    """Stop port-forwards, then delete pending test cases even if stopping them fails."""
    try:
        deployer.stop_port_forward()
    finally:
        if not no_cleanup:
            deployer.cleanup_pending()


@pytest.fixture(scope="session")
def report(request) -> Report:
    r = Report(
        profile=request.config.getoption("--profile"),
        platform=request.config.getoption("--platform"),
    )
    yield r
    r.finalize(request.config.getoption("--report-dir"))


@pytest.fixture(scope="class")
def endpoint(deployer: Deployer, tc: TestCase, request) -> str:
    explicit = request.config.getoption("--endpoint")
    if explicit:
        return explicit
    if not deployer.is_deployed(tc.name) and request.config.getoption("--mode") != "discover":
        pytest.skip(f"skipped — deploy failed or was skipped for '{tc.name}'")
    return deployer.get_endpoint(tc.name)


@pytest.fixture(scope="class")
def bearer_token(deployer: Deployer, tc: TestCase, request):
    supplied_token = request.config.getoption("--bearer-token")
    if supplied_token or request.config.getoption("--disable-auth"):
        yield supplied_token
    else:
        cleanup = tc.cleanup and not request.config.getoption("--nocleanup")
        with deployer.inference_token(deployer.manifest_service_names(tc), cleanup=cleanup) as token:
            yield token


@pytest.fixture(scope="class")
def client(endpoint: str, bearer_token: str) -> LLMClient:
    c = LLMClient(base_url=endpoint, bearer_token=bearer_token)
    yield c
    c.close()


@pytest.fixture(scope="class")
def pod_endpoint(deployer: Deployer, tc: TestCase, request) -> str:
    if not deployer.is_deployed(tc.name) and request.config.getoption("--mode") != "discover":
        pytest.skip(f"skipped — deploy failed or was skipped for '{tc.name}'")
    return deployer.get_pod_endpoint(tc.name)


@pytest.fixture(scope="class")
def pod_client(pod_endpoint: str, bearer_token: str) -> LLMClient:
    c = LLMClient(base_url=pod_endpoint, bearer_token=bearer_token)
    yield c
    c.close()


@pytest.fixture(scope="class")
def scraper(deployer: Deployer) -> Scraper:
    return Scraper(kubectl_fn=deployer.kubectl, namespace=deployer.namespace, kubeconfig=deployer.kubeconfig)


@pytest.fixture(scope="session")
def no_cleanup(request) -> bool:
    return request.config.getoption("--nocleanup")


@pytest.fixture(scope="session")
def test_mode(request) -> str:
    return request.config.getoption("--mode")


@pytest.fixture(scope="session")
def mock_mode(request) -> bool:
    return bool(request.config.getoption("--mock"))


@pytest.fixture(scope="session")
def guidellm_image(request) -> str:
    return request.config.getoption("--guidellm-image")
