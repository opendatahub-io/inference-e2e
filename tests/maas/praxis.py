"""Praxis (payload processing) 3.6-only test fixtures.

Ported from upstream ``opendatahub-tests`` ``tests/ai_gateway/models_as_a_service/praxis``,
adapted for xKS: the AITenant admission webhook enforces a dedicated Gateway per AITenant
(maas-controller ``pkg/webhook/aitenant_webhook.go``), so every run provisions its own
test-owned bootstrap Gateway. The Gateway's infrastructure ConfigMap pins the generated
Service to ``ClusterIP`` — no LoadBalancer, no public IP — mirroring upstream's
``bootstrap_gateway_context``. The shared ``maas-default-gateway``, its Service, and the
``models-as-a-service`` default tenant are never created, patched, or deleted.

Resource identifiers are pinned from upstream ``praxis/constants.py`` and maas-controller
``pkg/platform/tenantreconcile``: the payload-processing-type annotation and its ``praxis``
/``ipp`` values, the praxis-cleanup finalizer, and the per-tenant legacy-IPP resource names.

Selection contract (models-as-a-service#1579, ai-gateway-controller#91): an absent,
``praxis``, or unrecognized payload-processing-type selects Praxis; only ``ipp`` opts the
tenant back into maas-controller legacy IPP. A fresh AITenant therefore starts on Praxis.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import yaml

from conformance.deployer import Deployer

# The only profile that selects the Praxis phases.
PRAXIS_PROFILE = "maas-3.6.yaml"

# Shared, chart-managed locations the fixtures consume but never modify.
AITENANT_NAMESPACE = "ai-tenants"
GATEWAY_NAMESPACE = "redhat-ods-applications"
GATEWAY_CLASS = "istio"
AITENANT_CRD = "aitenants.maas.opendatahub.io"
MAASTENANTCONFIG_CRD = "maastenantconfigs.maas.opendatahub.io"
CONTROLLER_DEPLOYMENT = "ai-gateway-controller"
PRAXIS_EXTPROC_IMAGE_ENV = "RELATED_IMAGE_ODH_PRAXIS_EXTPROC_IMAGE"

# The chart-managed shared gateway (and istio's <gateway>-<class> Service) plus the
# default tenant, guarded read-only by the shared-state check.
SHARED_GATEWAY_NAME = "maas-default-gateway"
SHARED_GATEWAY_SERVICE_NAME = "maas-default-gateway-istio"
DEFAULT_TENANT_NAMESPACE = "models-as-a-service"

# Where the tenant reconciler deploys per-tenant maas-api (chart INFRA_NAMESPACE).
MAAS_API_INFRA_NAMESPACE = "redhat-ai-gateway-infra"

# The namespace-local MaasTenantConfig generated for every AITenant.
TENANT_CONFIG_NAME = "default-tenant"

# The controller's deterministic tenant namespace prefix (platform_context.go).
TENANT_NAMESPACE_PREFIX = "ai-tenant-"

# Payload-processing contract (upstream praxis/constants.py).
PAYLOAD_PROCESSING_TYPE_ANNOTATION = "maas.opendatahub.io/payload-processing-type"
PAYLOAD_PROCESSING_STATUS_ANNOTATION = "maas.opendatahub.io/payload-processing-status"
CLEANUP_COMPLETE_VALUE = "cleanup-complete"
STEADY_VALUE = "steady"
PRAXIS_VALUE = "praxis"
IPP_VALUE = "ipp"
PRAXIS_CLEANUP_FINALIZER = "ai-gateway-controller.opendatahub.io/praxis-cleanup"

# Per-tenant legacy IPP names in the gateway namespace (maas-controller tenantreconcile
# constants.go, naming_test.go) and the maas-api infra names.
LEGACY_IPP_POST_PROCESSING_NAME_BASE = "payload-processing"
LEGACY_IPP_PRE_PROCESSING_NAME_BASE = "payload-pre-processing"
LEGACY_IPP_PLUGINS_CONFIGMAP_NAME_BASE = "payload-processing-plugins"
MAAS_API_NAME_BASE = "maas-api"
MAAS_API_ROUTE_NAME_BASE = "maas-api-route"
MAAS_API_KEY_CLEANUP_NAME_BASE = "maas-api-key-cleanup"
MAAS_API_REVOKE_KEYS_NAME_BASE = "maas-api-revoke-keys"

# Legacy IPP ConfigMap keys and container args (upstream praxis/constants.py).
LEGACY_IPP_CUSTOM_CONFIG_DATA_KEY = "custom-ipp-config.yaml"
LEGACY_IPP_CUSTOM_PRE_CONFIG_DATA_KEY = "custom-pre-processing-ipp-config.yaml"
LEGACY_POST_PROCESSING_CONTAINER_CONFIG_ARG = "/config/custom-ipp-config.yaml"
LEGACY_PRE_PROCESSING_CONTAINER_CONFIG_ARG = "/config/custom-pre-processing-ipp-config.yaml"

# Praxis extproc ConfigMap keys and container args (ai-gateway-controller overlay).
PRAXIS_EXTPROC_CONFIG_DATA_KEY = "extproc.yaml"
PRAXIS_PRE_EXTPROC_CONFIG_DATA_KEY = "pre-extproc.yaml"
PRAXIS_POST_PROCESSING_CONTAINER_CONFIG_ARG = "/etc/praxis/extproc.yaml"
PRAXIS_PRE_PROCESSING_CONTAINER_CONFIG_ARG = "/etc/praxis/pre-extproc.yaml"

# Upstream waits: legacy/bundle 300s, switch-back 600s, poll 5s.
LEGACY_IPP_WAIT_TIMEOUT = 300
LEGACY_IPP_SWITCH_BACK_WAIT_TIMEOUT = 600
LEGACY_IPP_POLL_INTERVAL = 5

# A persistent not-Ready detail across this many polls means the controller is stuck.
_FAILED_POLLS = 3

# Cleanup must finish even when the controller cannot (see the forced-finalizer path).
_FORCE_WAIT_SECONDS = 60


def praxis_profile_enabled(profile_path: str) -> bool:
    """True only when the pytest ``--profile`` resolves to the 3.6 MaaS profile."""
    return bool(profile_path) and Path(profile_path).name == PRAXIS_PROFILE


_PROFILE_PATH = ""


def set_profile_path(profile_path: str) -> None:
    """Record the active pytest ``--profile``; called from conftest at configure time."""
    global _PROFILE_PATH
    _PROFILE_PATH = profile_path or ""


def praxis_profile_active() -> bool:
    """True when the current run selected the 3.6 MaaS profile."""
    return praxis_profile_enabled(_PROFILE_PATH)


def praxis_tenant_name(run_id: str) -> str:
    """Unique, DNS-label-safe AITenant name for one Praxis run."""
    return f"e2e-praxis-{run_id}"


@dataclass(frozen=True)
class PraxisTenant:
    """The test-owned AITenant and its generated resources."""

    aitenant_name: str
    aitenant_namespace: str
    tenant_namespace: str
    gateway_name: str
    gateway_namespace: str


# --- generated manifests -------------------------------------------------------


def bootstrap_configmap_manifest(gateway_name: str, namespace: str) -> dict:
    """Infrastructure ConfigMap pinning the bootstrap Gateway's Service to ClusterIP."""
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": f"{gateway_name}-config", "namespace": namespace},
        "data": {"service": "spec:\n  type: ClusterIP\n"},
    }


def bootstrap_gateway_manifest(gateway_name: str, namespace: str) -> dict:
    """Test-owned Gateway the AITenant webhook and MaaS reconcile require."""
    return {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "Gateway",
        "metadata": {"name": gateway_name, "namespace": namespace},
        "spec": {
            "gatewayClassName": GATEWAY_CLASS,
            "infrastructure": {"parametersRef": {"group": "", "kind": "ConfigMap", "name": f"{gateway_name}-config"}},
            "listeners": [
                {
                    "name": "http",
                    "port": 80,
                    "protocol": "HTTP",
                    "allowedRoutes": {"namespaces": {"from": "All"}},
                }
            ],
        },
    }


def aitenant_manifest(name: str, namespace: str, gateway_name: str) -> dict:
    """The test-owned AITenant, bound to its bootstrap Gateway."""
    return {
        "apiVersion": "maas.opendatahub.io/v1alpha1",
        "kind": "AITenant",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {"gateway": {"name": gateway_name}},
    }


# --- state probes --------------------------------------------------------------


def _get_json(kubectl: Callable[..., str], kind: str, name: str, namespace: str = "") -> dict:
    args = ("get", kind, name) + (("-n", namespace) if namespace else ()) + ("-o", "json")
    return json.loads(kubectl(*args))


def get_resource(kubectl: Callable[..., str], kind: str, name: str, namespace: str) -> dict | None:
    """A resource's JSON, or ``None`` when it does not exist (shared-state guards)."""
    try:
        return _get_json(kubectl, kind, name, namespace)
    except RuntimeError:
        return None


def _condition_true(conditions: list | None, condition_type: str) -> bool:
    return any(
        condition.get("type") == condition_type and condition.get("status") == "True" for condition in conditions or []
    )


def _false_condition_messages(conditions: list | None) -> list[str]:
    messages = []
    for condition in conditions or []:
        if condition.get("status") == "False" and condition.get("message"):
            message = str(condition["message"]).strip()
            if message and message not in messages:
                messages.append(message)
    return messages


def praxis_extproc_image(kubectl: Callable[..., str], deployment: dict) -> str:
    """The controller's Praxis extproc image, from a literal env value or its ConfigMap ref."""
    entry = {}
    for container in ((deployment.get("spec") or {}).get("template") or {}).get("spec", {}).get("containers") or []:
        for env in container.get("env") or []:
            if env.get("name") == PRAXIS_EXTPROC_IMAGE_ENV:
                entry = env
                break
    if entry.get("value"):
        return str(entry["value"])
    ref = (entry.get("valueFrom") or {}).get("configMapKeyRef") or {}
    if not ref.get("name"):
        return ""
    try:
        configmap = _get_json(kubectl, "configmap", ref["name"], GATEWAY_NAMESPACE)
    except RuntimeError:
        return ""
    return str((configmap.get("data") or {}).get(ref.get("key") or "", ""))


def check_praxis_prerequisites(kubectl: Callable[..., str]) -> None:
    """Read-only checks for the 3.6 stack; raises naming the exact missing prerequisite."""
    for namespace in (AITENANT_NAMESPACE, GATEWAY_NAMESPACE):
        try:
            _get_json(kubectl, "namespace", namespace)
        except RuntimeError:
            raise RuntimeError(f"missing Praxis prerequisite: namespace/{namespace} not found") from None
    for crd in (AITENANT_CRD, MAASTENANTCONFIG_CRD):
        try:
            _get_json(kubectl, "crd", crd)
        except RuntimeError:
            raise RuntimeError(f"missing Praxis prerequisite: crd/{crd} not found") from None
    try:
        _get_json(kubectl, "gatewayclass", GATEWAY_CLASS)
    except RuntimeError:
        raise RuntimeError(f"missing Praxis prerequisite: gatewayclass/{GATEWAY_CLASS} not found") from None

    deployment = _get_json(kubectl, "deployment", CONTROLLER_DEPLOYMENT, GATEWAY_NAMESPACE)
    if not _condition_true((deployment.get("status") or {}).get("conditions"), "Available"):
        raise RuntimeError(
            f"missing Praxis prerequisite: deployment/{CONTROLLER_DEPLOYMENT} in {GATEWAY_NAMESPACE} is not Available"
        )
    if not praxis_extproc_image(kubectl, deployment):
        raise RuntimeError(
            f"missing Praxis prerequisite: deployment/{CONTROLLER_DEPLOYMENT} has no Praxis extproc image "
            f"configured ({PRAXIS_EXTPROC_IMAGE_ENV})"
        )


def _aitenant_not_ready_detail(kubectl: Callable[..., str], aitenant: dict) -> tuple[str, bool]:
    """Why the AITenant is not Ready, and whether its state is terminally failed."""
    status = aitenant.get("status") or {}
    failed = status.get("phase") == "Failed"
    parts = [f"phase={status.get('phase', 'Pending')}"]
    parts += _false_condition_messages(status.get("conditions"))
    tenant_namespace = status.get("tenantNamespace", "")
    if tenant_namespace:
        try:
            config = _get_json(kubectl, "maastenantconfigs", TENANT_CONFIG_NAME, tenant_namespace)
        except RuntimeError:
            parts.append(f"MaasTenantConfig/{TENANT_CONFIG_NAME} not found in {tenant_namespace}")
        else:
            config_status = config.get("status") or {}
            failed = failed or config_status.get("phase") == "Failed"
            parts += _false_condition_messages(config_status.get("conditions"))
    return "; ".join(parts), failed


def wait_aitenant_ready(kubectl: Callable[..., str], name: str, timeout: float, poll_interval: float = 5) -> dict:
    """Wait for ``AITenant/{name}`` Ready=True; return its status.

    Fails fast when a terminal not-Ready detail is identical across consecutive polls
    (e.g. a stuck tenant-config reconcile error) and otherwise times out with the
    observed status, always including the MaasTenantConfig conditions as diagnostics.
    """
    deadline = time.monotonic() + timeout
    persistent_detail = ""
    strikes = 0
    while True:
        aitenant = _get_json(kubectl, "aitenants", name, AITENANT_NAMESPACE)
        status = aitenant.get("status") or {}
        if _condition_true(status.get("conditions"), "Ready"):
            return status
        detail, failed = _aitenant_not_ready_detail(kubectl, aitenant)
        strikes = strikes + 1 if failed and detail == persistent_detail else (1 if failed else 0)
        persistent_detail = detail
        if strikes >= _FAILED_POLLS:
            raise RuntimeError(f"AITenant/{name} is not Ready: {detail}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out after {timeout:g}s waiting for AITenant/{name} Ready: {detail}")
        time.sleep(poll_interval)


# --- payload-processing contract -----------------------------------------------


def payload_processing_type(config: dict) -> str | None:
    """The payload-processing-type annotation on any object; ``None`` when the key is absent.

    Distinguishes an absent annotation from an explicitly empty one (both select Praxis).
    """
    annotations = (config.get("metadata") or {}).get("annotations") or {}
    if PAYLOAD_PROCESSING_TYPE_ANNOTATION not in annotations:
        return None
    return str(annotations[PAYLOAD_PROCESSING_TYPE_ANNOTATION])


def uses_praxis(config: dict) -> bool:
    """Whether the tenant selects Praxis: everything except an explicit ``ipp`` does.

    Mirrors maas-controller ``resolveSkipIPP`` and ai-gateway-controller ``UsesPraxis``.
    """
    return payload_processing_type(config) != IPP_VALUE


def has_praxis_finalizer(config: dict) -> bool:
    """True when the object carries the ai-gateway-controller praxis-cleanup finalizer."""
    return PRAXIS_CLEANUP_FINALIZER in ((config.get("metadata") or {}).get("finalizers") or [])


def maastenantconfig(kubectl: Callable[..., str], tenant: PraxisTenant) -> dict:
    """The tenant's generated namespace-local MaasTenantConfig/default-tenant."""
    return _get_json(kubectl, "maastenantconfigs", TENANT_CONFIG_NAME, tenant.tenant_namespace)


def set_payload_processing_type(kubectl: Callable[..., str], tenant: PraxisTenant, value: str | None) -> None:
    """Set — or remove, when ``value`` is ``None`` — the payload-processing-type annotation."""
    patch = {"metadata": {"annotations": {PAYLOAD_PROCESSING_TYPE_ANNOTATION: value}}}
    kubectl(
        "patch",
        "maastenantconfigs",
        TENANT_CONFIG_NAME,
        "-n",
        tenant.tenant_namespace,
        "--type=merge",
        "-p",
        json.dumps(patch),
    )


def aitenant_payload_processing_type(kubectl: Callable[..., str], tenant: PraxisTenant) -> str | None:
    """The payload-processing-type annotation on the AITenant itself (must stay unset)."""
    aitenant = _get_json(kubectl, "aitenants", tenant.aitenant_name, tenant.aitenant_namespace)
    return payload_processing_type(aitenant)


def wait_for_praxis_finalizer(
    kubectl: Callable[..., str], tenant: PraxisTenant, present: bool, timeout: float = 120, poll_interval: float = 2
) -> None:
    """Poll until the praxis-cleanup finalizer matches ``present`` on the tenant's MaasTenantConfig."""
    deadline = time.monotonic() + timeout
    while True:
        config = maastenantconfig(kubectl, tenant)
        if has_praxis_finalizer(config) == present:
            return
        if time.monotonic() >= deadline:
            metadata = config.get("metadata") or {}
            raise TimeoutError(
                f"Timed out after {timeout:g}s waiting for finalizer '{PRAXIS_CLEANUP_FINALIZER}' to "
                f"{'appear on' if present else 'be removed from'} MaasTenantConfig/{TENANT_CONFIG_NAME} in "
                f"{tenant.tenant_namespace} (annotations={json.dumps(metadata.get('annotations') or {})}, "
                f"finalizers={json.dumps(metadata.get('finalizers') or [])})"
            )
        time.sleep(poll_interval)


# --- legacy IPP and Praxis bundle (gateway namespace) --------------------------


def legacy_ipp_post_processing_name(tenant: PraxisTenant) -> str:
    """Per-tenant legacy post-auth IPP Deployment — and EnvoyFilter — name."""
    return f"{LEGACY_IPP_POST_PROCESSING_NAME_BASE}-{tenant.aitenant_name}"


def legacy_ipp_pre_processing_name(tenant: PraxisTenant) -> str:
    """Per-tenant legacy pre-auth IPP Deployment name."""
    return f"{LEGACY_IPP_PRE_PROCESSING_NAME_BASE}-{tenant.aitenant_name}"


def legacy_ipp_plugins_configmap_name(tenant: PraxisTenant) -> str:
    """Per-tenant legacy IPP plugins ConfigMap name."""
    return f"{LEGACY_IPP_PLUGINS_CONFIGMAP_NAME_BASE}-{tenant.aitenant_name}"


def maas_api_deployment_name(tenant: PraxisTenant) -> str:
    """Per-tenant maas-api Deployment (and Service) name in the maas-api infra namespace."""
    return f"{MAAS_API_NAME_BASE}-{tenant.aitenant_name}"


def maas_api_route_name(tenant: PraxisTenant) -> str:
    """Per-tenant maas-api HTTPRoute name in the maas-api infra namespace."""
    return f"{MAAS_API_ROUTE_NAME_BASE}-{tenant.aitenant_name}"


def maas_api_key_cleanup_name(tenant: PraxisTenant) -> str:
    """Per-tenant API key cleanup CronJob name in the maas-api infra namespace."""
    return f"{MAAS_API_KEY_CLEANUP_NAME_BASE}-{tenant.aitenant_name}"


def maas_api_revoke_keys_name(tenant: PraxisTenant) -> str:
    """The CronJob's deterministic Job name (observed on xKS)."""
    return f"{MAAS_API_REVOKE_KEYS_NAME_BASE}-{tenant.aitenant_name}"


def tenant_owned_shared_namespace_resources(tenant: PraxisTenant) -> list[tuple[str, str, str]]:
    """Per-tenant resources in shared namespaces the forced teardown must sweep.

    The tenant reconcilers normally delete these together with the AITenant; when
    controller cleanup is stuck (forced teardown), they would be orphaned in the
    gateway and maas-api infra namespaces. Every name carries the unique tenant
    suffix, so nothing shared is ever swept.
    """
    return [
        ("deployment", legacy_ipp_post_processing_name(tenant), tenant.gateway_namespace),
        ("deployment", legacy_ipp_pre_processing_name(tenant), tenant.gateway_namespace),
        ("envoyfilter", legacy_ipp_post_processing_name(tenant), tenant.gateway_namespace),
        ("configmap", legacy_ipp_plugins_configmap_name(tenant), tenant.gateway_namespace),
        ("deployment", maas_api_deployment_name(tenant), MAAS_API_INFRA_NAMESPACE),
        ("service", maas_api_deployment_name(tenant), MAAS_API_INFRA_NAMESPACE),
        ("httproute", maas_api_route_name(tenant), MAAS_API_INFRA_NAMESPACE),
        ("cronjob", maas_api_key_cleanup_name(tenant), MAAS_API_INFRA_NAMESPACE),
        ("job", maas_api_revoke_keys_name(tenant), MAAS_API_INFRA_NAMESPACE),
    ]


def _resource_exists(kubectl: Callable[..., str], kind: str, name: str, namespace: str) -> bool:
    return get_resource(kubectl, kind, name, namespace) is not None


def _deployment_args(kubectl: Callable[..., str], deployment_name: str, namespace: str) -> list[str]:
    deployment = get_resource(kubectl, "deployment", deployment_name, namespace)
    if not deployment:
        return []
    args: list[str] = []
    for container in ((deployment.get("spec") or {}).get("template") or {}).get("spec", {}).get("containers") or []:
        args += [str(arg) for arg in container.get("args") or []]
    return args


def _deployment_available(kubectl: Callable[..., str], deployment_name: str, namespace: str) -> bool:
    deployment = get_resource(kubectl, "deployment", deployment_name, namespace)
    return bool(deployment) and _condition_true((deployment.get("status") or {}).get("conditions"), "Available")


def plugins_configmap_data_keys(kubectl: Callable[..., str], tenant: PraxisTenant) -> set[str]:
    """Data keys on the per-tenant plugins ConfigMap, or an empty set when absent."""
    configmap = get_resource(kubectl, "configmap", legacy_ipp_plugins_configmap_name(tenant), tenant.gateway_namespace)
    return set((configmap.get("data") or {}).keys()) if configmap else set()


def maas_legacy_ipp_markers_present(kubectl: Callable[..., str], tenant: PraxisTenant) -> bool:
    """True when the plugins ConfigMap still carries maas legacy IPP keys."""
    return bool(
        plugins_configmap_data_keys(kubectl, tenant)
        & {LEGACY_IPP_CUSTOM_CONFIG_DATA_KEY, LEGACY_IPP_CUSTOM_PRE_CONFIG_DATA_KEY}
    )


def legacy_ipp_stack_ready(kubectl: Callable[..., str], tenant: PraxisTenant) -> bool:
    """Full legacy IPP stack: ConfigMap keys, Deployments, EnvoyFilter, legacy container args."""
    keys = plugins_configmap_data_keys(kubectl, tenant)
    if not {LEGACY_IPP_CUSTOM_CONFIG_DATA_KEY, LEGACY_IPP_CUSTOM_PRE_CONFIG_DATA_KEY}.issubset(keys):
        return False
    if keys & {PRAXIS_EXTPROC_CONFIG_DATA_KEY, PRAXIS_PRE_EXTPROC_CONFIG_DATA_KEY}:
        return False
    post, pre = legacy_ipp_post_processing_name(tenant), legacy_ipp_pre_processing_name(tenant)
    for name in (post, pre):
        if not _resource_exists(kubectl, "deployment", name, tenant.gateway_namespace):
            return False
    if not _resource_exists(kubectl, "envoyfilter", post, tenant.gateway_namespace):
        return False
    post_args = _deployment_args(kubectl, post, tenant.gateway_namespace)
    pre_args = _deployment_args(kubectl, pre, tenant.gateway_namespace)
    return (
        LEGACY_POST_PROCESSING_CONTAINER_CONFIG_ARG in post_args
        and LEGACY_PRE_PROCESSING_CONTAINER_CONFIG_ARG in pre_args
        and PRAXIS_POST_PROCESSING_CONTAINER_CONFIG_ARG not in post_args
        and PRAXIS_PRE_PROCESSING_CONTAINER_CONFIG_ARG not in pre_args
    )


def praxis_bundle_ready(kubectl: Callable[..., str], tenant: PraxisTenant) -> bool:
    """Praxis extproc bundle: ConfigMap keys, EnvoyFilter, and praxis container args."""
    keys = plugins_configmap_data_keys(kubectl, tenant)
    if not {PRAXIS_EXTPROC_CONFIG_DATA_KEY, PRAXIS_PRE_EXTPROC_CONFIG_DATA_KEY}.issubset(keys):
        return False
    if keys & {LEGACY_IPP_CUSTOM_CONFIG_DATA_KEY, LEGACY_IPP_CUSTOM_PRE_CONFIG_DATA_KEY}:
        return False
    post, pre = legacy_ipp_post_processing_name(tenant), legacy_ipp_pre_processing_name(tenant)
    if not _resource_exists(kubectl, "envoyfilter", post, tenant.gateway_namespace):
        return False
    return PRAXIS_POST_PROCESSING_CONTAINER_CONFIG_ARG in _deployment_args(
        kubectl, post, tenant.gateway_namespace
    ) and PRAXIS_PRE_PROCESSING_CONTAINER_CONFIG_ARG in _deployment_args(kubectl, pre, tenant.gateway_namespace)


def praxis_signals_present(kubectl: Callable[..., str], tenant: PraxisTenant) -> bool:
    """Any praxis ConfigMap key or container arg remaining in the gateway namespace."""
    keys = plugins_configmap_data_keys(kubectl, tenant)
    post, pre = legacy_ipp_post_processing_name(tenant), legacy_ipp_pre_processing_name(tenant)
    return bool(
        keys & {PRAXIS_EXTPROC_CONFIG_DATA_KEY, PRAXIS_PRE_EXTPROC_CONFIG_DATA_KEY}
        or PRAXIS_POST_PROCESSING_CONTAINER_CONFIG_ARG in _deployment_args(kubectl, post, tenant.gateway_namespace)
        or PRAXIS_PRE_PROCESSING_CONTAINER_CONFIG_ARG in _deployment_args(kubectl, pre, tenant.gateway_namespace)
    )


def maas_ipp_handoff_complete(kubectl: Callable[..., str], tenant: PraxisTenant) -> bool:
    """MaaS released legacy IPP: a cleanup-complete/steady status, or Praxis selected with markers gone."""
    config = get_resource(kubectl, "maastenantconfigs", TENANT_CONFIG_NAME, tenant.tenant_namespace)
    if not config:
        return False
    annotations = (config.get("metadata") or {}).get("annotations") or {}
    if annotations.get(PAYLOAD_PROCESSING_STATUS_ANNOTATION) in (CLEANUP_COMPLETE_VALUE, STEADY_VALUE):
        return True
    if not uses_praxis(config):
        return False
    return not maas_legacy_ipp_markers_present(kubectl, tenant)


def _describe_ipp_state(kubectl: Callable[..., str], tenant: PraxisTenant) -> str:
    """Observed IPP state for timeout diagnostics."""
    post, pre = legacy_ipp_post_processing_name(tenant), legacy_ipp_pre_processing_name(tenant)
    parts = [
        f"ConfigMap/{legacy_ipp_plugins_configmap_name(tenant)} in {tenant.gateway_namespace} "
        f"keys={sorted(plugins_configmap_data_keys(kubectl, tenant))}"
    ]
    for name in (post, pre):
        available = _deployment_available(kubectl, name, tenant.gateway_namespace)
        args = _deployment_args(kubectl, name, tenant.gateway_namespace)
        parts.append(f"Deployment/{name} available={available} args={args}")
    parts.append(
        f"EnvoyFilter/{post} present={_resource_exists(kubectl, 'envoyfilter', post, tenant.gateway_namespace)}"
    )
    return "; ".join(parts)


def _wait_until(
    ready: Callable[[], bool], describe: Callable[[], str], what: str, timeout: float, poll_interval: float
) -> None:
    """Poll ``ready()`` until true; time out with ``describe()`` as the observed state."""
    deadline = time.monotonic() + timeout
    while True:
        if ready():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out after {timeout:g}s waiting for {what}: observed {describe()}")
        time.sleep(poll_interval)


def wait_legacy_ipp_installed(
    kubectl: Callable[..., str],
    tenant: PraxisTenant,
    timeout: float = LEGACY_IPP_WAIT_TIMEOUT,
    poll_interval: float = LEGACY_IPP_POLL_INTERVAL,
) -> None:
    """Wait for the full legacy IPP stack and its Deployments to be Available."""
    post, pre = legacy_ipp_post_processing_name(tenant), legacy_ipp_pre_processing_name(tenant)
    describe = lambda: _describe_ipp_state(kubectl, tenant)  # noqa: E731
    _wait_until(
        lambda: legacy_ipp_stack_ready(kubectl, tenant),
        describe,
        f"the legacy IPP stack for {tenant.aitenant_name} in {tenant.gateway_namespace}",
        timeout,
        poll_interval,
    )
    _wait_until(
        lambda: (
            _deployment_available(kubectl, post, tenant.gateway_namespace)
            and _deployment_available(kubectl, pre, tenant.gateway_namespace)
        ),
        describe,
        f"legacy IPP Deployments {post} and {pre} Available in {tenant.gateway_namespace}",
        timeout,
        poll_interval,
    )


def wait_legacy_ipp_markers_absent(
    kubectl: Callable[..., str],
    tenant: PraxisTenant,
    timeout: float = LEGACY_IPP_WAIT_TIMEOUT,
    poll_interval: float = LEGACY_IPP_POLL_INTERVAL,
) -> None:
    """Wait until maas legacy IPP markers are gone from the plugins ConfigMap."""
    _wait_until(
        lambda: not maas_legacy_ipp_markers_present(kubectl, tenant),
        lambda: _describe_ipp_state(kubectl, tenant),
        f"removal of maas legacy IPP markers for {tenant.aitenant_name} in {tenant.gateway_namespace}",
        timeout,
        poll_interval,
    )


def wait_maas_ipp_handoff(
    kubectl: Callable[..., str],
    tenant: PraxisTenant,
    timeout: float = LEGACY_IPP_WAIT_TIMEOUT,
    poll_interval: float = LEGACY_IPP_POLL_INTERVAL,
) -> None:
    """Wait for MaaS to release legacy IPP before ai-gateway applies the Praxis bundle."""
    _wait_until(
        lambda: maas_ipp_handoff_complete(kubectl, tenant),
        lambda: _describe_ipp_state(kubectl, tenant),
        f"the MaaS IPP handoff for AITenant/{tenant.aitenant_name} "
        f"({PAYLOAD_PROCESSING_STATUS_ANNOTATION}={CLEANUP_COMPLETE_VALUE}/{STEADY_VALUE} or Praxis "
        f"selected with legacy markers removed)",
        timeout,
        poll_interval,
    )


def wait_praxis_bundle_installed(
    kubectl: Callable[..., str],
    tenant: PraxisTenant,
    timeout: float = LEGACY_IPP_WAIT_TIMEOUT,
    poll_interval: float = LEGACY_IPP_POLL_INTERVAL,
) -> None:
    """Wait for the Praxis extproc bundle and its Deployments to be Available."""
    post, pre = legacy_ipp_post_processing_name(tenant), legacy_ipp_pre_processing_name(tenant)
    describe = lambda: _describe_ipp_state(kubectl, tenant)  # noqa: E731
    _wait_until(
        lambda: praxis_bundle_ready(kubectl, tenant),
        describe,
        f"the Praxis extproc bundle for {tenant.aitenant_name} in {tenant.gateway_namespace}",
        timeout,
        poll_interval,
    )
    _wait_until(
        lambda: (
            _deployment_available(kubectl, post, tenant.gateway_namespace)
            and _deployment_available(kubectl, pre, tenant.gateway_namespace)
        ),
        describe,
        f"Praxis extproc Deployments {post} and {pre} Available in {tenant.gateway_namespace}",
        timeout,
        poll_interval,
    )


def wait_praxis_signals_absent(
    kubectl: Callable[..., str],
    tenant: PraxisTenant,
    timeout: float = LEGACY_IPP_SWITCH_BACK_WAIT_TIMEOUT,
    poll_interval: float = LEGACY_IPP_POLL_INTERVAL,
) -> None:
    """Wait until no praxis ConfigMap keys or container args remain in the gateway namespace."""
    _wait_until(
        lambda: not praxis_signals_present(kubectl, tenant),
        lambda: _describe_ipp_state(kubectl, tenant),
        f"removal of the Praxis extproc bundle for {tenant.aitenant_name} in {tenant.gateway_namespace}",
        timeout,
        poll_interval,
    )


def wait_maastenantconfig_ready(
    kubectl: Callable[..., str],
    tenant: PraxisTenant,
    timeout: float = LEGACY_IPP_WAIT_TIMEOUT,
    poll_interval: float = LEGACY_IPP_POLL_INTERVAL,
) -> None:
    """Wait for MaasTenantConfig Ready, rejecting a Degraded legacy-IPP EnvoyFilter error."""

    def ready() -> bool:
        config = get_resource(kubectl, "maastenantconfigs", TENANT_CONFIG_NAME, tenant.tenant_namespace)
        return bool(config) and _condition_true((config.get("status") or {}).get("conditions"), "Ready")

    def describe() -> str:
        config = get_resource(kubectl, "maastenantconfigs", TENANT_CONFIG_NAME, tenant.tenant_namespace) or {}
        conditions = (config.get("status") or {}).get("conditions") or []
        return f"MaasTenantConfig/{TENANT_CONFIG_NAME} in {tenant.tenant_namespace} conditions={json.dumps(conditions)}"

    _wait_until(ready, describe, f"MaasTenantConfig/{TENANT_CONFIG_NAME} Ready", timeout, poll_interval)
    config = maastenantconfig(kubectl, tenant)
    for condition in (config.get("status") or {}).get("conditions") or []:
        if condition.get("type") == "Degraded" and condition.get("status") == "True":
            message = str(condition.get("message") or "")
            assert "EnvoyFilter" not in message, (
                f"MaasTenantConfig/{TENANT_CONFIG_NAME} in {tenant.tenant_namespace} is Degraded with a "
                f"legacy IPP EnvoyFilter error: {message}"
            )


def wait_maas_api_available(
    kubectl: Callable[..., str],
    tenant: PraxisTenant,
    timeout: float = LEGACY_IPP_WAIT_TIMEOUT,
    poll_interval: float = LEGACY_IPP_POLL_INTERVAL,
) -> None:
    """Wait for the per-tenant maas-api Deployment to be Available in the infra namespace."""

    def mtc_deployments_available() -> bool:
        config = get_resource(kubectl, "maastenantconfigs", TENANT_CONFIG_NAME, tenant.tenant_namespace)
        return bool(config) and _condition_true((config.get("status") or {}).get("conditions"), "DeploymentsAvailable")

    def describe_mtc() -> str:
        config = get_resource(kubectl, "maastenantconfigs", TENANT_CONFIG_NAME, tenant.tenant_namespace) or {}
        conditions = (config.get("status") or {}).get("conditions") or []
        return f"MaasTenantConfig/{TENANT_CONFIG_NAME} in {tenant.tenant_namespace} conditions={json.dumps(conditions)}"

    _wait_until(
        mtc_deployments_available,
        describe_mtc,
        f"MaasTenantConfig/{TENANT_CONFIG_NAME} DeploymentsAvailable",
        timeout,
        poll_interval,
    )
    deployment_name = maas_api_deployment_name(tenant)

    def api_available() -> bool:
        return _deployment_available(kubectl, deployment_name, MAAS_API_INFRA_NAMESPACE)

    def describe_api() -> str:
        deployment = get_resource(kubectl, "deployment", deployment_name, MAAS_API_INFRA_NAMESPACE) or {}
        conditions = (deployment.get("status") or {}).get("conditions") or []
        return f"Deployment/{deployment_name} in {MAAS_API_INFRA_NAMESPACE} conditions={json.dumps(conditions)}"

    _wait_until(
        api_available,
        describe_api,
        f"Deployment/{deployment_name} Available in {MAAS_API_INFRA_NAMESPACE}",
        timeout,
        poll_interval,
    )


def wait_praxis_selected(
    kubectl: Callable[..., str],
    tenant: PraxisTenant,
    timeout: float = LEGACY_IPP_WAIT_TIMEOUT,
    poll_interval: float = LEGACY_IPP_POLL_INTERVAL,
) -> None:
    """Wait for a Praxis-selected tenant to settle: finalizer, MaaS handoff, Praxis bundle ready."""
    wait_for_praxis_finalizer(kubectl, tenant, present=True, timeout=timeout, poll_interval=poll_interval)
    wait_maas_ipp_handoff(kubectl, tenant, timeout, poll_interval)
    wait_legacy_ipp_markers_absent(kubectl, tenant, timeout, poll_interval)
    wait_praxis_bundle_installed(kubectl, tenant, timeout, poll_interval)


def opt_tenant_into_ipp(
    kubectl: Callable[..., str],
    tenant: PraxisTenant,
    timeout: float = LEGACY_IPP_SWITCH_BACK_WAIT_TIMEOUT,
    poll_interval: float = LEGACY_IPP_POLL_INTERVAL,
) -> None:
    """Set the explicit ``ipp`` opt-out and wait for Praxis to leave and legacy IPP to return."""
    set_payload_processing_type(kubectl, tenant, IPP_VALUE)
    wait_for_praxis_finalizer(kubectl, tenant, present=False, timeout=timeout, poll_interval=poll_interval)
    wait_praxis_signals_absent(kubectl, tenant, timeout, poll_interval)
    wait_legacy_ipp_installed(kubectl, tenant, timeout, poll_interval)


def restore_tenant_default(
    kubectl: Callable[..., str],
    tenant: PraxisTenant,
    timeout: float = LEGACY_IPP_WAIT_TIMEOUT,
    poll_interval: float = LEGACY_IPP_POLL_INTERVAL,
) -> None:
    """Remove the annotation and wait for the default (Praxis) path to return."""
    set_payload_processing_type(kubectl, tenant, None)
    observed = payload_processing_type(maastenantconfig(kubectl, tenant))
    assert observed is None, (
        f"MaasTenantConfig/{TENANT_CONFIG_NAME} in {tenant.tenant_namespace} should have no "
        f"{PAYLOAD_PROCESSING_TYPE_ANNOTATION} annotation after restoring the default; got {observed!r}"
    )
    wait_praxis_selected(kubectl, tenant, timeout, poll_interval)


# --- tenant lifecycle ----------------------------------------------------------


def _apply(kubectl: Callable[..., str], manifest: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(manifest, handle)
        path = handle.name
    try:
        kubectl("apply", "-f", path)
    finally:
        os.unlink(path)


def _wait_gateway_programmed(
    kubectl: Callable[..., str], name: str, namespace: str, timeout: float, poll_interval: float
) -> None:
    deadline = time.monotonic() + timeout
    while True:
        gateway = _get_json(kubectl, "gateway", name, namespace)
        conditions = (gateway.get("status") or {}).get("conditions")
        if _condition_true(conditions, "Programmed"):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout:g}s waiting for Gateway/{name} Programmed (conditions={json.dumps(conditions)})"
            )
        time.sleep(poll_interval)


def _wait_aitenant_status(
    kubectl: Callable[..., str], name: str, timeout: float, poll_interval: float
) -> tuple[str, str, str]:
    """Wait for the generated ``tenantNamespace`` and ``gatewayRef``; return both."""
    deadline = time.monotonic() + timeout
    while True:
        status = _get_json(kubectl, "aitenants", name, AITENANT_NAMESPACE).get("status") or {}
        gateway_ref = status.get("gatewayRef") or {}
        if status.get("tenantNamespace") and gateway_ref.get("name") and gateway_ref.get("namespace"):
            return status["tenantNamespace"], gateway_ref["name"], gateway_ref["namespace"]
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout:g}s waiting for AITenant/{name} status.tenantNamespace/gatewayRef "
                f"(status={json.dumps(status)})"
            )
        time.sleep(poll_interval)


def _namespace_exists(kubectl: Callable[..., str], namespace: str) -> bool:
    try:
        kubectl("get", "namespace", namespace, "-o", "json")
        return True
    except RuntimeError:
        return False


def _wait_namespace_gone(kubectl: Callable[..., str], namespace: str, timeout: float, poll_interval: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if not _namespace_exists(kubectl, namespace):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_interval)


def _clear_finalizers(kubectl: Callable[..., str], kind: str, name: str, namespace: str) -> None:
    try:
        kubectl("patch", kind, name, "-n", namespace, "--type=merge", "-p", '{"metadata":{"finalizers":[]}}')
    except RuntimeError as error:
        print(f"  → could not clear finalizers on {kind}/{name}: {error}")


def _delete(kubectl: Callable[..., str], kind: str, name: str, namespace: str) -> None:
    """Delete without waiting: finalizer-stuck objects must never hang the subprocess.

    Completion is handled by the fixture's own bounded waits (namespace gone, forced
    cleanup); a blocking ``kubectl delete`` would instead hit the 300s subprocess
    timeout and abort the remaining teardown steps.
    """
    args = ("delete", kind, name) + (("-n", namespace) if namespace else ()) + ("--ignore-not-found", "--wait=false")
    kubectl(*args)


def _force_tenant_cleanup(kubectl: Callable[..., str], tenant: PraxisTenant) -> None:
    """Break-glass cleanup of test-owned tenant resources when controller cleanup is stuck.

    The generated MaasTenantConfig and AITenant keep their cleanup finalizers for as long
    as the controller cannot complete the platform reconcile; clearing them (and deleting
    the generated namespace) is limited to resources this run created. The per-tenant IPP
    and maas-api resources live in shared namespaces, so they are swept explicitly —
    nothing without the unique tenant suffix is ever deleted.
    """
    print(f"  → AITenant cleanup is stuck; forcing test-owned cleanup for {tenant.tenant_namespace}")
    _clear_finalizers(kubectl, "maastenantconfigs", TENANT_CONFIG_NAME, tenant.tenant_namespace)
    _delete(kubectl, "maastenantconfigs", TENANT_CONFIG_NAME, tenant.tenant_namespace)
    _clear_finalizers(kubectl, "aitenants", tenant.aitenant_name, tenant.aitenant_namespace)
    _delete(kubectl, "namespace", tenant.tenant_namespace, "")
    for kind, name, namespace in tenant_owned_shared_namespace_resources(tenant):
        _delete(kubectl, kind, name, namespace)


def _teardown(kubectl: Callable[..., str], tenant: PraxisTenant, timeout: float, poll_interval: float) -> None:
    """Delete only test-owned resources, in the order the controller requires.

    The MaasTenantConfig cleanup path needs the Gateway to exist, so the bootstrap Gateway
    and its ConfigMap are deleted only after the tenant namespace is gone.
    """
    _delete(kubectl, "aitenants", tenant.aitenant_name, tenant.aitenant_namespace)
    if not _wait_namespace_gone(kubectl, tenant.tenant_namespace, timeout, poll_interval):
        _force_tenant_cleanup(kubectl, tenant)
        if not _wait_namespace_gone(kubectl, tenant.tenant_namespace, _FORCE_WAIT_SECONDS, poll_interval):
            raise RuntimeError(f"tenant namespace {tenant.tenant_namespace} still exists after forced cleanup")
    _delete(kubectl, "gateway", tenant.gateway_name, tenant.gateway_namespace)
    _delete(kubectl, "configmap", f"{tenant.gateway_name}-config", tenant.gateway_namespace)


def _derived_tenant(name: str) -> PraxisTenant:
    """The tenant's generated refs, derived from the applied manifests.

    Used when bootstrap fails before the AITenant status reports them: the controller's
    namespace prefix is deterministic and the Gateway is the bootstrap one this run applied.
    """
    return PraxisTenant(
        aitenant_name=name,
        aitenant_namespace=AITENANT_NAMESPACE,
        tenant_namespace=f"{TENANT_NAMESPACE_PREFIX}{name}",
        gateway_name=name,
        gateway_namespace=GATEWAY_NAMESPACE,
    )


@contextmanager
def create_praxis_tenant(
    deployer: Deployer,
    timeout: float = 600,
    poll_interval: float = 5,
    run_id: str | None = None,
    teardown_timeout: float = 180,
) -> Iterator[PraxisTenant]:
    """Yield a Ready, test-owned AITenant with its bootstrap Gateway; clean everything up.

    The AITenant admission webhook requires a dedicated Gateway per tenant, so each run
    provisions a ClusterIP bootstrap Gateway in the shared gateway namespace. Only
    test-owned resources are created or deleted; teardown runs even when the body fails
    or the bootstrap itself never reaches Ready.
    """
    kubectl = deployer.kubectl
    check_praxis_prerequisites(kubectl)
    name = praxis_tenant_name(run_id or uuid.uuid4().hex[:8])
    tenant: PraxisTenant | None = None
    try:
        _apply(kubectl, bootstrap_configmap_manifest(name, GATEWAY_NAMESPACE))
        _apply(kubectl, bootstrap_gateway_manifest(name, GATEWAY_NAMESPACE))
        _wait_gateway_programmed(kubectl, name, GATEWAY_NAMESPACE, timeout, poll_interval)
        _apply(kubectl, aitenant_manifest(name, AITENANT_NAMESPACE, gateway_name=name))
        tenant_namespace, gateway_name, gateway_namespace = _wait_aitenant_status(kubectl, name, timeout, poll_interval)
        tenant = PraxisTenant(
            aitenant_name=name,
            aitenant_namespace=AITENANT_NAMESPACE,
            tenant_namespace=tenant_namespace,
            gateway_name=gateway_name,
            gateway_namespace=gateway_namespace,
        )
        wait_aitenant_ready(kubectl, name, timeout, poll_interval)
        yield tenant
    finally:
        # a bootstrap that never reaches Ready (e.g. the controller cannot reconcile the
        # tenant) must not leak the applied Gateway, ConfigMap, and AITenant either
        _teardown(kubectl, tenant or _derived_tenant(name), teardown_timeout, poll_interval)
