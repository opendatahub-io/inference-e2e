"""Praxis 3.6-only conformance phases.

``PraxisPhases`` is mixed into ``TestConformance`` (tests/test_conformance.py) after
``MaaSPhases``. Phases are deselected at collection unless the run resolves the
``maas-3.6`` profile (recorded at configure time by conftest) and the testcase
manifest declares a ``MaaSModelRef``. Each phase provisions its own test-owned
AITenant and its one Gateway via ``maas.praxis`` and never touches the shared
gateway, its Service, or the default tenant.

Praxis is the default: an absent, ``praxis``, or unrecognized payload-processing-type
selects it, and only ``ipp`` opts a tenant back into legacy IPP.

To add a Praxis check, add a ``test_30<letter>_maas_praxis_<name>`` method here.
"""

from __future__ import annotations

import pytest

from conformance.config import TestCase
from conformance.deployer import Deployer

from maas.phases import _declares_model_ref
from maas.praxis import (
    IPP_VALUE,
    PRAXIS_VALUE,
    GATEWAY_NAMESPACE,
    SHARED_GATEWAY_NAME,
    SHARED_GATEWAY_SERVICE_NAME,
    DEFAULT_TENANT_NAMESPACE,
    TENANT_CONFIG_NAME,
    aitenant_payload_processing_type,
    create_praxis_tenant,
    get_resource,
    maastenantconfig,
    opt_tenant_into_ipp,
    payload_processing_type,
    praxis_profile_active,
    restore_tenant_default,
    set_payload_processing_type,
    wait_for_praxis_finalizer,
    wait_maas_api_available,
    wait_maastenantconfig_ready,
    wait_praxis_selected,
)

# Explicit values that, like an absent annotation, select Praxis.
PRAXIS_SELECTING_VALUES = (PRAXIS_VALUE, "foo", "")


def _praxis_applies(tc: TestCase) -> bool:
    return praxis_profile_active() and _declares_model_ref(tc)


# Applied per method: a class-level mark would be inherited by every TestConformance phase.
_for_praxis = pytest.mark.applies_when.with_args(_praxis_applies)


def _capture_shared_state(kubectl) -> dict:
    """Snapshot the shared gateway, its Service, and the default tenant (read-only)."""
    gateway = get_resource(kubectl, "gateway", SHARED_GATEWAY_NAME, GATEWAY_NAMESPACE) or {}
    service = get_resource(kubectl, "service", SHARED_GATEWAY_SERVICE_NAME, GATEWAY_NAMESPACE) or {}
    default_tenant = get_resource(kubectl, "maastenantconfigs", TENANT_CONFIG_NAME, DEFAULT_TENANT_NAMESPACE) or {}
    return {
        "gateway_spec": gateway.get("spec"),
        "gateway_uid": (gateway.get("metadata") or {}).get("uid"),
        "service_type": (service.get("spec") or {}).get("type"),
        "default_tenant_annotations": dict((default_tenant.get("metadata") or {}).get("annotations") or {}),
        "default_tenant_uid": (default_tenant.get("metadata") or {}).get("uid"),
    }


def _assert_shared_state_unchanged(kubectl, expected: dict) -> None:
    """Fail when anything the run must never disturb changed on the shared stack."""
    observed = _capture_shared_state(kubectl)
    assert observed == expected, (
        f"shared resources changed during the Praxis phases: expected {expected!r}, observed {observed!r}"
    )


class PraxisPhases:
    """3.6-only Praxis payload-processing checks, mixed into TestConformance."""

    @_for_praxis
    def test_30p_maas_praxis_annotation_contract(self, deployer: Deployer, tc: TestCase):
        """Praxis is the default on MaasTenantConfig; only ``ipp`` opts out of the praxis-cleanup finalizer."""
        kubectl = deployer.kubectl
        with create_praxis_tenant(deployer) as tenant:
            # a fresh tenant carries no annotation and is already on Praxis
            assert payload_processing_type(maastenantconfig(kubectl, tenant)) is None, (
                "a fresh tenant must not carry the payload-processing-type annotation"
            )
            wait_for_praxis_finalizer(kubectl, tenant, present=True)
            assert aitenant_payload_processing_type(kubectl, tenant) is None

            # each explicit value is stored on MaasTenantConfig only; starting from the
            # ipp opt-out makes the Praxis selection observable as the finalizer returning
            for value in PRAXIS_SELECTING_VALUES:
                set_payload_processing_type(kubectl, tenant, IPP_VALUE)
                wait_for_praxis_finalizer(kubectl, tenant, present=False)
                assert payload_processing_type(maastenantconfig(kubectl, tenant)) == IPP_VALUE

                set_payload_processing_type(kubectl, tenant, value)
                wait_for_praxis_finalizer(kubectl, tenant, present=True)
                assert payload_processing_type(maastenantconfig(kubectl, tenant)) == value, (
                    f"annotation value {value!r} must be stored on MaasTenantConfig"
                )
                assert aitenant_payload_processing_type(kubectl, tenant) is None, (
                    "the AITenant must not mirror the MaasTenantConfig annotation"
                )

    @_for_praxis
    def test_30q_maas_praxis_legacy_ipp_roundtrip(self, deployer: Deployer, tc: TestCase):
        """Praxis ⇄ legacy IPP roundtrip on a test-owned tenant: default, ipp opt-out, default restored."""
        kubectl = deployer.kubectl
        shared_state = _capture_shared_state(kubectl)
        with create_praxis_tenant(deployer) as tenant:
            wait_praxis_selected(kubectl, tenant)
            wait_maastenantconfig_ready(kubectl, tenant)
            wait_maas_api_available(kubectl, tenant)

            opt_tenant_into_ipp(kubectl, tenant)
            wait_maastenantconfig_ready(kubectl, tenant)
            wait_maas_api_available(kubectl, tenant)

            restore_tenant_default(kubectl, tenant)
            wait_maastenantconfig_ready(kubectl, tenant)
            wait_maas_api_available(kubectl, tenant)
        _assert_shared_state_unchanged(kubectl, shared_state)
