"""Unit tests for the Praxis 3.6 fixtures (tests/maas/praxis.py).

All cluster interaction goes through ``FakeKubectl``; no cluster is required.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from conformance.deployer import Deployer

from maas import praxis
from maas.praxis import (
    PraxisTenant,
    check_praxis_prerequisites,
    create_praxis_tenant,
    praxis_profile_enabled,
    praxis_tenant_name,
    wait_aitenant_ready,
)

RUN_ID = "deadbeef"
NAME = "e2e-praxis-deadbeef"
TENANT_NS = f"ai-tenant-{NAME}"


def _conditions(*pairs: tuple[str, str]) -> list[dict]:
    return [{"type": kind, "status": status} for kind, status in pairs]


class FakeKubectl:
    """Records calls; serves canned ``get`` JSON, applies temp-file manifests, tracks deletes."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.applied: list[dict] = []
        self.deleted: list[tuple] = []
        self.patches: list[tuple] = []
        self.responses: dict[tuple[str, str], object] = {}
        self.deleted_keys: set[tuple[str, str]] = set()

    def serve(self, kind: str, name: str, *payloads: object) -> None:
        """Serve ``get kind name``: pops payloads in order, the last one repeats.

        A single callable payload is invoked on every call (dynamic state).
        """
        if len(payloads) == 1 and callable(payloads[0]):
            self.responses[(kind, name)] = payloads[0]
        else:
            self.responses[(kind, name)] = list(payloads)

    def __call__(self, *args: str, check: bool = True) -> str:
        self.calls.append(args)
        verb = args[0]
        if verb == "apply":
            path = args[args.index("-f") + 1]
            with open(path) as handle:
                self.applied.append(yaml.safe_load(handle))
            return "applied"
        if verb == "delete":
            self.deleted.append(args)
            self.deleted_keys.add((args[1], args[2]))
            return ""
        if verb == "patch":
            self.patches.append(args)
            return ""
        if verb == "get":
            kind, name = args[1], args[2]
            if (kind, name) in self.deleted_keys:
                raise RuntimeError(f"kubectl get {kind} {name} failed: NotFound")
            payload = self.responses.get((kind, name))
            if payload is None:
                raise RuntimeError(f"kubectl get {kind} {name} failed: NotFound")
            if callable(payload):
                return payload()
            if isinstance(payload, list):
                return payload.pop(0) if len(payload) > 1 else payload[0]
            return payload
        raise AssertionError(f"unexpected kubectl call: {args}")


def _namespace_that_disappears_with(fake: FakeKubectl, trigger: tuple[str, str]) -> object:
    """Namespace JSON that vanishes once the trigger resource is deleted (controller cleanup)."""

    def state() -> str:
        if trigger in fake.deleted_keys:
            raise RuntimeError("kubectl get namespace failed: NotFound")
        return json.dumps({"metadata": {"name": "tenant"}})

    return state


def _serve_cluster(
    fake: FakeKubectl,
    *,
    available: bool = True,
    extproc_entry: list | None = None,
    extproc_configmap: bool = True,
) -> None:
    """Serve the read-only 3.6 stack prerequisites."""
    fake.serve("namespace", "ai-tenants", json.dumps({"metadata": {"name": "ai-tenants"}}))
    fake.serve("namespace", "redhat-ods-applications", json.dumps({"metadata": {"name": "redhat-ods-applications"}}))
    fake.serve("crd", "aitenants.maas.opendatahub.io", json.dumps({"metadata": {"name": "aitenants"}}))
    fake.serve("crd", "maastenantconfigs.maas.opendatahub.io", json.dumps({"metadata": {"name": "maastenantconfigs"}}))
    fake.serve("gatewayclass", "istio", json.dumps({"metadata": {"name": "istio"}}))
    if extproc_entry is None:
        extproc_entry = [
            {
                "name": "RELATED_IMAGE_ODH_PRAXIS_EXTPROC_IMAGE",
                "valueFrom": {
                    "configMapKeyRef": {"name": "ai-gateway-controller-parameters", "key": "praxis-extproc-image"}
                },
            }
        ]
    fake.serve(
        "deployment",
        "ai-gateway-controller",
        json.dumps(
            {
                "status": {"conditions": _conditions(("Available", "True" if available else "False"))},
                "spec": {"template": {"spec": {"containers": [{"name": "manager", "env": extproc_entry}]}}},
            }
        ),
    )
    if extproc_configmap:
        fake.serve(
            "configmap",
            "ai-gateway-controller-parameters",
            json.dumps({"data": {"praxis-extproc-image": "quay.io/praxis-extproc@sha256:abc"}}),
        )


def _aitenant_json(*, phase: str, tenant_namespace: str = "", ready: str = "True") -> str:
    status: dict = {"phase": phase, "conditions": _conditions(("Ready", ready))}
    if tenant_namespace:
        status["tenantNamespace"] = tenant_namespace
        status["gatewayRef"] = {"name": NAME, "namespace": "redhat-ods-applications"}
    return json.dumps({"status": status})


def _serve_bootstrap(fake: FakeKubectl) -> None:
    """Serve a bootstrap that programs the Gateway, then reports a Ready AITenant."""
    fake.serve(
        "gateway",
        NAME,
        json.dumps({"status": {"conditions": _conditions(("Programmed", "False"))}}),
        json.dumps({"status": {"conditions": _conditions(("Programmed", "True"))}}),
    )
    fake.serve(
        "aitenants",
        NAME,
        _aitenant_json(phase="Pending"),
        _aitenant_json(phase="Pending", tenant_namespace=TENANT_NS, ready="False"),
        _aitenant_json(phase="Active", tenant_namespace=TENANT_NS),
    )
    fake.serve("maastenantconfigs", "default-tenant", _aitenant_json(phase="Active"))


def _deployer(fake: FakeKubectl) -> Deployer:
    deployer = Deployer()
    deployer.kubectl = fake  # type: ignore[method-assign]
    return deployer


# --- profile gate -----------------------------------------------------------


def test_profile_gate_enables_only_the_maas_36_profile():
    assert praxis_profile_enabled("configs/profiles/maas-3.6.yaml")
    assert praxis_profile_enabled("/anywhere/maas-3.6.yaml")
    assert not praxis_profile_enabled("configs/profiles/maas-3.5.yaml")
    assert not praxis_profile_enabled("configs/profiles/3.6.yaml")
    assert not praxis_profile_enabled("configs/profiles/3.5.yaml")
    assert not praxis_profile_enabled("")


# --- generated names ---------------------------------------------------------


def test_tenant_names_are_unique_and_dns_label_safe():
    first = praxis_tenant_name("0f1e2d3c")
    second = praxis_tenant_name(RUN_ID)
    assert first == "e2e-praxis-0f1e2d3c"
    assert first != second
    for name in (first, second, praxis_tenant_name("a1b2c3d4")):
        assert len(name) <= 63
        assert re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", name), name


# --- bootstrap manifests -----------------------------------------------------


def test_bootstrap_manifests_are_test_owned_and_clusterip():
    configmap = praxis.bootstrap_configmap_manifest(NAME, praxis.GATEWAY_NAMESPACE)
    gateway = praxis.bootstrap_gateway_manifest(NAME, praxis.GATEWAY_NAMESPACE)
    aitenant = praxis.aitenant_manifest(NAME, praxis.AITENANT_NAMESPACE, gateway_name=NAME)

    assert configmap["metadata"]["name"] == f"{NAME}-config"
    assert configmap["data"]["service"] == "spec:\n  type: ClusterIP\n"
    assert gateway["spec"]["gatewayClassName"] == praxis.GATEWAY_CLASS
    assert gateway["spec"]["infrastructure"]["parametersRef"]["name"] == f"{NAME}-config"
    assert gateway["spec"]["listeners"][0]["allowedRoutes"] == {"namespaces": {"from": "All"}}
    assert aitenant["metadata"]["namespace"] == praxis.AITENANT_NAMESPACE
    assert aitenant["spec"]["gateway"]["name"] == NAME


# --- prerequisites -----------------------------------------------------------


def test_prerequisites_pass_when_the_3_6_stack_is_installed():
    fake = FakeKubectl()
    _serve_cluster(fake)

    check_praxis_prerequisites(fake)


@pytest.mark.parametrize(
    ("kind", "name", "expected"),
    [
        ("namespace", "ai-tenants", r"namespace/ai-tenants"),
        ("namespace", "redhat-ods-applications", r"namespace/redhat-ods-applications"),
        ("crd", "aitenants.maas.opendatahub.io", r"crd/aitenants\.maas\.opendatahub\.io"),
        ("crd", "maastenantconfigs.maas.opendatahub.io", r"crd/maastenantconfigs\.maas\.opendatahub\.io"),
        ("gatewayclass", "istio", r"gatewayclass/istio"),
    ],
)
def test_prerequisites_name_the_exact_missing_resource(kind, name, expected):
    fake = FakeKubectl()
    _serve_cluster(fake)
    fake.responses.pop((kind, name))

    with pytest.raises(RuntimeError, match=expected):
        check_praxis_prerequisites(fake)


def test_prerequisites_require_an_available_controller():
    fake = FakeKubectl()
    _serve_cluster(fake, available=False)

    with pytest.raises(RuntimeError, match="ai-gateway-controller.*not Available"):
        check_praxis_prerequisites(fake)


def test_prerequisites_require_a_configured_extproc_image():
    fake = FakeKubectl()
    _serve_cluster(fake, extproc_configmap=False)

    with pytest.raises(RuntimeError, match="Praxis extproc image"):
        check_praxis_prerequisites(fake)


def test_extproc_image_resolves_literal_env_values_without_the_configmap():
    fake = FakeKubectl()
    _serve_cluster(
        fake, extproc_entry=[{"name": "RELATED_IMAGE_ODH_PRAXIS_EXTPROC_IMAGE", "value": "quay.io/x@sha256:1"}]
    )

    check_praxis_prerequisites(fake)

    assert not any(call[:2] == ("get", "configmap") for call in fake.calls)


# --- tenant lifecycle --------------------------------------------------------


def test_create_praxis_tenant_bootstraps_a_ready_tenant():
    fake = FakeKubectl()
    _serve_cluster(fake)
    _serve_bootstrap(fake)
    fake.serve("namespace", TENANT_NS, _namespace_that_disappears_with(fake, ("aitenants", NAME)))

    with create_praxis_tenant(
        _deployer(fake), timeout=5, poll_interval=0, run_id=RUN_ID, teardown_timeout=0.5
    ) as tenant:
        assert tenant == PraxisTenant(
            aitenant_name=NAME,
            aitenant_namespace="ai-tenants",
            tenant_namespace=TENANT_NS,
            gateway_name=NAME,
            gateway_namespace="redhat-ods-applications",
        )
        assert [doc["kind"] for doc in fake.applied] == ["ConfigMap", "Gateway", "AITenant"]

    deleted = [(args[1], args[2]) for args in fake.deleted]
    # the AITenant goes first; the Gateway only after the tenant namespace is gone
    assert deleted.index(("aitenants", NAME)) < deleted.index(("gateway", NAME))
    assert deleted.index(("gateway", NAME)) < deleted.index(("configmap", f"{NAME}-config"))


def test_teardown_never_touches_shared_resources():
    fake = FakeKubectl()
    _serve_cluster(fake)
    _serve_bootstrap(fake)
    fake.serve("namespace", TENANT_NS, _namespace_that_disappears_with(fake, ("aitenants", NAME)))

    with create_praxis_tenant(_deployer(fake), timeout=5, poll_interval=0, run_id=RUN_ID, teardown_timeout=0.5):
        pass

    for args in fake.deleted + fake.patches:
        joined = " ".join(args)
        assert "maas-default-gateway" not in joined
        assert "models-as-a-service" not in joined
        if args[0] == "delete" and args[1] == "namespace":
            assert args[2].startswith("ai-tenant-e2e-praxis-")


def test_teardown_forces_stuck_test_owned_finalizers():
    fake = FakeKubectl()
    _serve_cluster(fake)
    _serve_bootstrap(fake)
    # controller cleanup is wedged: the tenant namespace never disappears on its own
    fake.serve("namespace", TENANT_NS, json.dumps({"metadata": {"name": TENANT_NS}}))

    with create_praxis_tenant(_deployer(fake), timeout=5, poll_interval=0, run_id=RUN_ID, teardown_timeout=0.2):
        pass

    patched = [(args[1], args[2]) for args in fake.patches]
    assert ("maastenantconfigs", "default-tenant") in patched
    assert ("aitenants", NAME) in patched
    deleted = [(args[1], args[2]) for args in fake.deleted]
    assert ("maastenantconfigs", "default-tenant") in deleted
    assert ("namespace", TENANT_NS) in deleted
    # the bootstrap Gateway and its ConfigMap are still cleaned up afterwards
    assert ("gateway", NAME) in deleted
    assert ("configmap", f"{NAME}-config") in deleted


def test_wait_aitenant_ready_reports_the_tenant_config_error():
    fake = FakeKubectl()
    fake.serve("aitenants", NAME, _aitenant_json(phase="Pending", tenant_namespace=TENANT_NS, ready="False"))
    fake.serve(
        "maastenantconfigs",
        "default-tenant",
        json.dumps(
            {
                "status": {
                    "phase": "Failed",
                    "conditions": [
                        {"type": "Ready", "status": "False", "message": "apply ClusterRole /maas-api is forbidden"}
                    ],
                }
            }
        ),
    )

    with pytest.raises(RuntimeError, match="apply ClusterRole /maas-api is forbidden"):
        wait_aitenant_ready(fake, NAME, timeout=30, poll_interval=0)


def test_wait_aitenant_ready_tolerates_a_missing_tenant_config():
    fake = FakeKubectl()
    fake.serve("aitenants", NAME, _aitenant_json(phase="Pending", tenant_namespace=TENANT_NS, ready="False"))

    with pytest.raises(TimeoutError, match="MaasTenantConfig/default-tenant"):
        wait_aitenant_ready(fake, NAME, timeout=0.2, poll_interval=0)


def test_wait_aitenant_ready_fast_fails_only_terminal_aitenant_phase():
    fake = FakeKubectl()
    fake.serve("aitenants", NAME, _aitenant_json(phase="Failed", tenant_namespace="", ready="False"))

    with pytest.raises(RuntimeError, match="phase=Failed"):
        wait_aitenant_ready(fake, NAME, timeout=30, poll_interval=0)


# --- payload-processing annotation contract -----------------------------------


def test_payload_processing_type_distinguishes_absent_from_empty():
    absent = {"metadata": {"annotations": {}}}
    empty = {"metadata": {"annotations": {"maas.opendatahub.io/payload-processing-type": ""}}}
    praxis_value = {"metadata": {"annotations": {"maas.opendatahub.io/payload-processing-type": "praxis"}}}

    assert praxis.payload_processing_type(absent) is None
    assert praxis.payload_processing_type(empty) == ""
    assert praxis.payload_processing_type(praxis_value) == "praxis"


def test_praxis_cleanup_finalizer_detection():
    legacy = {"metadata": {"finalizers": ["maas.opendatahub.io/tenant-cleanup"]}}
    praxis_owned = {"metadata": {"finalizers": ["ai-gateway-controller.opendatahub.io/praxis-cleanup"]}}

    assert not praxis.has_praxis_finalizer(legacy)
    assert praxis.has_praxis_finalizer(praxis_owned)


def test_set_payload_processing_type_patches_the_exact_annotation():
    fake = FakeKubectl()
    tenant = PraxisTenant(
        aitenant_name=NAME,
        aitenant_namespace="ai-tenants",
        tenant_namespace=TENANT_NS,
        gateway_name=NAME,
        gateway_namespace="redhat-ods-applications",
    )

    praxis.set_payload_processing_type(fake, tenant, "praxis")
    praxis.set_payload_processing_type(fake, tenant, None)

    assert [args[1:4] for args in fake.patches] == [
        ("maastenantconfigs", "default-tenant", "-n"),
        ("maastenantconfigs", "default-tenant", "-n"),
    ]
    patches = [json.loads(args[args.index("-p") + 1]) for args in fake.patches]
    assert patches[0]["metadata"]["annotations"] == {"maas.opendatahub.io/payload-processing-type": "praxis"}
    # a null value removes the annotation (kubectl merge-patch semantics)
    assert patches[1]["metadata"]["annotations"] == {"maas.opendatahub.io/payload-processing-type": None}


def test_wait_for_praxis_finalizer_polls_until_it_appears():
    fake = FakeKubectl()
    tenant = _tenant()
    fake.serve(
        "maastenantconfigs",
        "default-tenant",
        json.dumps({"metadata": {"finalizers": []}}),
        json.dumps({"metadata": {"finalizers": ["maas.opendatahub.io/tenant-cleanup"]}}),
        json.dumps({"metadata": {"finalizers": ["ai-gateway-controller.opendatahub.io/praxis-cleanup"]}}),
    )

    praxis.wait_for_praxis_finalizer(fake, tenant, present=True, timeout=5, poll_interval=0)


def test_wait_for_praxis_finalizer_reports_observed_state_on_timeout():
    fake = FakeKubectl()
    tenant = _tenant()
    fake.serve(
        "maastenantconfigs",
        "default-tenant",
        json.dumps(
            {
                "metadata": {
                    "finalizers": ["maas.opendatahub.io/tenant-cleanup"],
                    "annotations": {"maas.opendatahub.io/payload-processing-type": "praxis"},
                }
            }
        ),
    )

    with pytest.raises(TimeoutError, match="praxis-cleanup"):
        praxis.wait_for_praxis_finalizer(fake, tenant, present=True, timeout=0.2, poll_interval=0)


def _tenant() -> PraxisTenant:
    return PraxisTenant(
        aitenant_name=NAME,
        aitenant_namespace="ai-tenants",
        tenant_namespace=TENANT_NS,
        gateway_name=NAME,
        gateway_namespace="redhat-ods-applications",
    )


# --- legacy IPP / Praxis bundle resource model ---------------------------------


def test_per_tenant_legacy_ipp_and_maas_api_names():
    tenant = _tenant()
    assert praxis.legacy_ipp_post_processing_name(tenant) == f"payload-processing-{NAME}"
    assert praxis.legacy_ipp_pre_processing_name(tenant) == f"payload-pre-processing-{NAME}"
    assert praxis.legacy_ipp_plugins_configmap_name(tenant) == f"payload-processing-plugins-{NAME}"
    assert praxis.maas_api_deployment_name(tenant) == f"maas-api-{NAME}"
    assert praxis.maas_api_route_name(tenant) == f"maas-api-route-{NAME}"


def test_legacy_ipp_config_keys_and_container_args_are_pinned():
    assert praxis.LEGACY_IPP_CUSTOM_CONFIG_DATA_KEY == "custom-ipp-config.yaml"
    assert praxis.LEGACY_IPP_CUSTOM_PRE_CONFIG_DATA_KEY == "custom-pre-processing-ipp-config.yaml"
    assert praxis.LEGACY_POST_PROCESSING_CONTAINER_CONFIG_ARG == "/config/custom-ipp-config.yaml"
    assert praxis.LEGACY_PRE_PROCESSING_CONTAINER_CONFIG_ARG == "/config/custom-pre-processing-ipp-config.yaml"


def test_praxis_bundle_config_keys_and_container_args_are_pinned():
    assert praxis.PRAXIS_EXTPROC_CONFIG_DATA_KEY == "extproc.yaml"
    assert praxis.PRAXIS_PRE_EXTPROC_CONFIG_DATA_KEY == "pre-extproc.yaml"
    assert praxis.PRAXIS_POST_PROCESSING_CONTAINER_CONFIG_ARG == "/etc/praxis/extproc.yaml"
    assert praxis.PRAXIS_PRE_PROCESSING_CONTAINER_CONFIG_ARG == "/etc/praxis/pre-extproc.yaml"


def _deployment_json(args: list[str], available: str = "True") -> str:
    return json.dumps(
        {
            "spec": {"template": {"spec": {"containers": [{"name": "extproc", "args": args}]}}},
            "status": {"conditions": _conditions(("Available", available))},
        }
    )


def _serve_legacy_ipp(fake: FakeKubectl) -> None:
    """Serve the full per-tenant legacy IPP stack in the gateway namespace."""
    fake.serve(
        "configmap",
        f"payload-processing-plugins-{NAME}",
        json.dumps(
            {
                "data": {
                    "custom-ipp-config.yaml": "# maas",
                    "custom-pre-processing-ipp-config.yaml": "# maas",
                }
            }
        ),
    )
    fake.serve("deployment", f"payload-processing-{NAME}", _deployment_json(["/config/custom-ipp-config.yaml"]))
    fake.serve(
        "deployment",
        f"payload-pre-processing-{NAME}",
        _deployment_json(["/config/custom-pre-processing-ipp-config.yaml"]),
    )
    fake.serve("envoyfilter", f"payload-processing-{NAME}", json.dumps({"spec": {}}))


def _serve_praxis_bundle(fake: FakeKubectl) -> None:
    """Serve the per-tenant Praxis extproc bundle in the gateway namespace."""
    fake.serve(
        "configmap",
        f"payload-processing-plugins-{NAME}",
        json.dumps({"data": {"extproc.yaml": "# praxis", "pre-extproc.yaml": "# praxis"}}),
    )
    fake.serve("deployment", f"payload-processing-{NAME}", _deployment_json(["/etc/praxis/extproc.yaml"]))
    fake.serve("deployment", f"payload-pre-processing-{NAME}", _deployment_json(["/etc/praxis/pre-extproc.yaml"]))
    fake.serve("envoyfilter", f"payload-processing-{NAME}", json.dumps({"spec": {}}))


def test_legacy_ipp_stack_ready_requires_the_full_stack():
    tenant = _tenant()
    fake = FakeKubectl()
    _serve_legacy_ipp(fake)
    assert praxis.legacy_ipp_stack_ready(fake, tenant)

    incomplete = FakeKubectl()
    _serve_legacy_ipp(incomplete)
    incomplete.responses.pop(("envoyfilter", f"payload-processing-{NAME}"))
    assert not praxis.legacy_ipp_stack_ready(incomplete, tenant)

    mixed = FakeKubectl()
    _serve_legacy_ipp(mixed)
    mixed.serve(
        "configmap",
        f"payload-processing-plugins-{NAME}",
        json.dumps({"data": {"custom-ipp-config.yaml": "", "extproc.yaml": ""}}),
    )
    assert not praxis.legacy_ipp_stack_ready(mixed, tenant)


def test_praxis_bundle_ready_requires_keys_args_and_envoyfilter():
    tenant = _tenant()
    fake = FakeKubectl()
    _serve_praxis_bundle(fake)
    assert praxis.praxis_bundle_ready(fake, tenant)

    incomplete = FakeKubectl()
    _serve_praxis_bundle(incomplete)
    incomplete.responses.pop(("envoyfilter", f"payload-processing-{NAME}"))
    assert not praxis.praxis_bundle_ready(incomplete, tenant)

    mixed = FakeKubectl()
    _serve_praxis_bundle(mixed)
    mixed.serve(
        "configmap",
        f"payload-processing-plugins-{NAME}",
        json.dumps({"data": {"extproc.yaml": "", "pre-extproc.yaml": "", "custom-ipp-config.yaml": ""}}),
    )
    assert not praxis.praxis_bundle_ready(mixed, tenant)


def test_uses_praxis_selects_everything_but_explicit_ipp():
    def config(annotations: dict) -> dict:
        return {"metadata": {"annotations": annotations}}

    key = praxis.PAYLOAD_PROCESSING_TYPE_ANNOTATION
    assert praxis.uses_praxis(config({}))
    assert praxis.uses_praxis(config({key: "praxis"}))
    assert praxis.uses_praxis(config({key: "foo"}))
    assert praxis.uses_praxis(config({key: ""}))
    assert not praxis.uses_praxis(config({key: "ipp"}))


def test_maas_ipp_handoff_completes_via_status_or_released_markers():
    tenant = _tenant()

    def handoff(annotations: dict, plugins_data: dict | None = None) -> bool:
        fake = FakeKubectl()
        fake.serve("maastenantconfigs", "default-tenant", json.dumps({"metadata": {"annotations": annotations}}))
        if plugins_data is not None:
            fake.serve("configmap", f"payload-processing-plugins-{NAME}", json.dumps({"data": plugins_data}))
        return praxis.maas_ipp_handoff_complete(fake, tenant)

    status, type_key = praxis.PAYLOAD_PROCESSING_STATUS_ANNOTATION, praxis.PAYLOAD_PROCESSING_TYPE_ANNOTATION
    # MaaS's cleanup-complete, or ai-gateway-controller's steady claim, completes it on its own
    assert handoff({status: "cleanup-complete"})
    assert handoff({status: "steady"})
    # Praxis selected (absent annotation is the default) with the legacy markers released
    assert handoff({}, {"extproc.yaml": ""})
    assert handoff({type_key: "praxis"}, {"extproc.yaml": ""})
    # legacy markers still in the gateway namespace keep it incomplete
    assert not handoff({type_key: "praxis"}, {"custom-ipp-config.yaml": ""})
    # an explicit ipp opt-out never hands off, even with no markers
    assert not handoff({type_key: "ipp"}, {})


def test_wait_legacy_ipp_installed_reports_observed_state_on_timeout():
    fake = FakeKubectl()
    _serve_legacy_ipp(fake)
    fake.responses.pop(("envoyfilter", f"payload-processing-{NAME}"))

    with pytest.raises(TimeoutError, match=f"EnvoyFilter/payload-processing-{NAME} present=False"):
        praxis.wait_legacy_ipp_installed(fake, _tenant(), timeout=0.2, poll_interval=0)


def test_wait_praxis_signals_absent_reports_observed_state_on_timeout():
    fake = FakeKubectl()
    _serve_praxis_bundle(fake)

    with pytest.raises(TimeoutError, match="/etc/praxis/extproc.yaml"):
        praxis.wait_praxis_signals_absent(fake, _tenant(), timeout=0.2, poll_interval=0)


def test_wait_maastenantconfig_ready_rejects_envoyfilter_degraded():
    tenant = _tenant()
    healthy = FakeKubectl()
    healthy.serve(
        "maastenantconfigs",
        "default-tenant",
        json.dumps(
            {"status": {"conditions": _conditions(("Ready", "True")) + [{"type": "Degraded", "status": "False"}]}}
        ),
    )
    praxis.wait_maastenantconfig_ready(healthy, tenant, timeout=0.2, poll_interval=0)

    degraded = FakeKubectl()
    degraded.serve(
        "maastenantconfigs",
        "default-tenant",
        json.dumps(
            {
                "status": {
                    "conditions": _conditions(("Ready", "True"))
                    + [{"type": "Degraded", "status": "True", "message": "EnvoyFilter workload entry failed"}]
                }
            }
        ),
    )
    with pytest.raises(AssertionError, match="EnvoyFilter"):
        praxis.wait_maastenantconfig_ready(degraded, tenant, timeout=0.2, poll_interval=0)


def test_wait_maas_api_available_waits_for_the_per_tenant_deployment():
    fake = FakeKubectl()
    fake.serve(
        "maastenantconfigs",
        "default-tenant",
        json.dumps({"status": {"conditions": _conditions(("DeploymentsAvailable", "True"))}}),
    )
    fake.serve("deployment", f"maas-api-{NAME}", _deployment_json(["serve"]))

    praxis.wait_maas_api_available(fake, _tenant(), timeout=0.2, poll_interval=0)


def _serve_roundtrip_cluster(fake: FakeKubectl) -> None:
    """Serve a tenant whose IPP resources follow the payload-processing-type annotation."""

    def current_type() -> str | None:
        value = None
        for args in fake.patches:
            patch = json.loads(args[args.index("-p") + 1])
            annotations = (patch.get("metadata") or {}).get("annotations") or {}
            for key, patched in annotations.items():
                if key == praxis.PAYLOAD_PROCESSING_TYPE_ANNOTATION:
                    value = patched
        return value

    def on_praxis() -> bool:
        return current_type() != praxis.IPP_VALUE

    def mtc() -> str:
        value = current_type()
        annotations = {praxis.PAYLOAD_PROCESSING_TYPE_ANNOTATION: value} if value is not None else {}
        finalizers = [praxis.PRAXIS_CLEANUP_FINALIZER] if on_praxis() else []
        return json.dumps({"metadata": {"annotations": annotations, "finalizers": finalizers}})

    def plugins_configmap() -> str:
        if on_praxis():
            data = {"extproc.yaml": "# praxis", "pre-extproc.yaml": "# praxis"}
        else:
            data = {"custom-ipp-config.yaml": "# maas", "custom-pre-processing-ipp-config.yaml": "# maas"}
        return json.dumps({"data": data})

    def post_deployment() -> str:
        config_arg = (
            praxis.PRAXIS_POST_PROCESSING_CONTAINER_CONFIG_ARG
            if on_praxis()
            else praxis.LEGACY_POST_PROCESSING_CONTAINER_CONFIG_ARG
        )
        return _deployment_json([config_arg])

    def pre_deployment() -> str:
        config_arg = (
            praxis.PRAXIS_PRE_PROCESSING_CONTAINER_CONFIG_ARG
            if on_praxis()
            else praxis.LEGACY_PRE_PROCESSING_CONTAINER_CONFIG_ARG
        )
        return _deployment_json([config_arg])

    fake.serve("maastenantconfigs", "default-tenant", mtc)
    fake.serve("configmap", f"payload-processing-plugins-{NAME}", plugins_configmap)
    fake.serve("deployment", f"payload-processing-{NAME}", post_deployment)
    fake.serve("deployment", f"payload-pre-processing-{NAME}", pre_deployment)
    fake.serve("envoyfilter", f"payload-processing-{NAME}", json.dumps({"spec": {}}))


def test_praxis_roundtrip_orchestration():
    fake = FakeKubectl()
    tenant = _tenant()
    _serve_roundtrip_cluster(fake)

    # a fresh tenant is already on Praxis
    praxis.wait_praxis_selected(fake, tenant, timeout=0.5, poll_interval=0)
    assert praxis.praxis_bundle_ready(fake, tenant)

    praxis.opt_tenant_into_ipp(fake, tenant, timeout=0.5, poll_interval=0)
    assert praxis.legacy_ipp_stack_ready(fake, tenant)

    praxis.restore_tenant_default(fake, tenant, timeout=0.5, poll_interval=0)
    assert praxis.praxis_bundle_ready(fake, tenant)


def test_sweep_list_covers_only_test_owned_shared_namespace_resources():
    resources = praxis.tenant_owned_shared_namespace_resources(_tenant())
    by_kind_name = {(kind, name) for kind, name, _ in resources}
    assert ("deployment", f"payload-processing-{NAME}") in by_kind_name
    assert ("deployment", f"payload-pre-processing-{NAME}") in by_kind_name
    assert ("envoyfilter", f"payload-processing-{NAME}") in by_kind_name
    assert ("configmap", f"payload-processing-plugins-{NAME}") in by_kind_name
    assert ("deployment", f"maas-api-{NAME}") in by_kind_name
    assert ("service", f"maas-api-{NAME}") in by_kind_name
    assert ("httproute", f"maas-api-route-{NAME}") in by_kind_name
    assert ("cronjob", f"maas-api-key-cleanup-{NAME}") in by_kind_name
    assert ("job", f"maas-api-revoke-keys-{NAME}") in by_kind_name
    for _kind, name, namespace in resources:
        assert NAME in name, name
        assert namespace in (praxis.GATEWAY_NAMESPACE, praxis.MAAS_API_INFRA_NAMESPACE)


def test_forced_teardown_sweeps_shared_namespace_ipp_resources():
    fake = FakeKubectl()
    _serve_cluster(fake)
    _serve_bootstrap(fake)
    # the tenant namespace never disappears: controller cleanup is stuck
    fake.serve("namespace", TENANT_NS, json.dumps({"metadata": {"name": TENANT_NS}}))

    with create_praxis_tenant(
        _deployer(fake), timeout=5, poll_interval=0, run_id=RUN_ID, teardown_timeout=0.2
    ) as tenant:
        swept = praxis.tenant_owned_shared_namespace_resources(tenant)

    deleted = {(args[1], args[2], args[args.index("-n") + 1]) for args in fake.deleted if "-n" in args}
    for kind, name, namespace in swept:
        assert (kind, name, namespace) in deleted
    # every delete is issued without waiting: a finalizer-stuck object must never
    # block the subprocess past the kubectl timeout and abort the remaining teardown
    for args in fake.deleted:
        assert "--wait=false" in args


def test_shared_state_guard_captures_and_compares_shared_resources():
    from maas.praxis_phases import _assert_shared_state_unchanged, _capture_shared_state

    fake = FakeKubectl()
    fake.serve(
        "gateway",
        "maas-default-gateway",
        json.dumps({"metadata": {"uid": "gw-uid"}, "spec": {"gatewayClassName": "istio"}}),
    )
    fake.serve("service", "maas-default-gateway-istio", json.dumps({"spec": {"type": "LoadBalancer"}}))
    fake.serve(
        "maastenantconfigs",
        "default-tenant",
        json.dumps({"metadata": {"uid": "mtc-uid", "annotations": {"a": "b"}}}),
    )

    state = _capture_shared_state(fake)
    assert state["gateway_uid"] == "gw-uid"
    assert state["service_type"] == "LoadBalancer"
    assert state["default_tenant_uid"] == "mtc-uid"

    _assert_shared_state_unchanged(fake, state)

    fake.serve("service", "maas-default-gateway-istio", json.dumps({"spec": {"type": "ClusterIP"}}))
    with pytest.raises(AssertionError, match="shared"):
        _assert_shared_state_unchanged(fake, state)


def test_failed_bootstrap_still_cleans_up_every_test_owned_resource():
    """A tenant whose reconcile never reaches Ready (e.g. the RBAC defect) must not leak."""
    fake = FakeKubectl()
    _serve_cluster(fake)
    fake.serve("gateway", NAME, json.dumps({"status": {"conditions": _conditions(("Programmed", "True"))}}))
    # the status reports the generated refs, but Ready never comes and the detail is persistent
    fake.serve("aitenants", NAME, _aitenant_json(phase="Pending", tenant_namespace=TENANT_NS, ready="False"))
    fake.serve(
        "maastenantconfigs",
        "default-tenant",
        json.dumps(
            {
                "status": {
                    "phase": "Failed",
                    "conditions": [
                        {"type": "Ready", "status": "False", "message": "apply ClusterRole /maas-api: forbidden"}
                    ],
                }
            }
        ),
    )
    fake.serve("namespace", TENANT_NS, _namespace_that_disappears_with(fake, ("aitenants", NAME)))

    with pytest.raises(RuntimeError, match="not Ready"):
        with create_praxis_tenant(
            _deployer(fake), timeout=5, poll_interval=0, run_id=RUN_ID, teardown_timeout=0.5
        ) as tenant:
            raise AssertionError(f"the tenant must not be yielded, got {tenant}")

    deleted = {(args[1], args[2]) for args in fake.deleted}
    assert ("aitenants", NAME) in deleted
    assert ("gateway", NAME) in deleted
    assert ("configmap", f"{NAME}-config") in deleted


def test_failed_bootstrap_before_status_cleans_up_with_derived_refs():
    """A bootstrap that fails before the status reports refs still tears everything down."""
    fake = FakeKubectl()
    _serve_cluster(fake)
    fake.serve("gateway", NAME, json.dumps({"status": {"conditions": _conditions(("Programmed", "True"))}}))
    fake.serve("aitenants", NAME, _aitenant_json(phase="Pending"))
    fake.serve("namespace", TENANT_NS, _namespace_that_disappears_with(fake, ("aitenants", NAME)))

    with pytest.raises(TimeoutError, match="tenantNamespace"):
        with create_praxis_tenant(_deployer(fake), timeout=0.2, poll_interval=0, run_id=RUN_ID, teardown_timeout=0.5):
            raise AssertionError("the tenant must not be yielded")

    deleted = {(args[1], args[2]) for args in fake.deleted}
    assert ("aitenants", NAME) in deleted
    assert ("gateway", NAME) in deleted
    assert ("configmap", f"{NAME}-config") in deleted
    # the derived tenant namespace (the controller's deterministic ai-tenant-{name})
    # is what teardown waits on; the controller removes it with the AITenant
    assert any(call[:3] == ("get", "namespace", TENANT_NS) for call in fake.calls)
