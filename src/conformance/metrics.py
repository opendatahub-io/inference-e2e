"""Prometheus metrics scraping and per-topology validation.

Scraper discovers pods by label, scrapes ``/metrics`` via in-pod tooling
(python3/wget) with port-forward + httpx fallback for minimal images, and
uses a service-account bearer token for EPP when
``--metrics-endpoint-auth=true`` (RHOAI 3.5+).

Pod discovery labels:
  - Workload / decode — ``llminferenceservice-workload``
  - Prefill — ``llminferenceservice-workload-prefill``
  - EPP — ``EPP_LABELS`` (several component names across llm-d versions)

Validators (wired from ``test_conformance`` phases 10–15 / 21):
  - ``validate_vllm_basic`` — request_success on workload pods
  - ``validate_cache_aware`` — prefix cache queries/hits (+ EPP indexer)
  - ``validate_pd`` — decode/prefill token-by-source and NIXL transfer signals
  - ``validate_scheduler`` — EPP request / ready-pod metrics
  - ``validate_flow_control`` — EPP dispatch / queue / saturation
  - ``validate_lora`` — ``vllm:lora_requests_info`` on workload pods

Also: ``parse_prometheus`` (text exposition), ``dump_raw_metrics`` (report
artifacts), ``Metric`` / ``ScrapeResult`` / ``CheckResult`` dataclasses.
"""

from __future__ import annotations

import logging
import re
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

# vLLM metrics
VLLM_REQUEST_SUCCESS = "vllm:request_success_total"
VLLM_PROMPT_TOKENS = "vllm:prompt_tokens_total"
VLLM_GEN_TOKENS = "vllm:generation_tokens_total"
VLLM_GPU_CACHE = "vllm:gpu_cache_usage_perc"
VLLM_PREEMPTIONS = "vllm:num_preemptions_total"
VLLM_PREFIX_QUERIES = "vllm:prefix_cache_queries"
VLLM_PREFIX_QUERIES_ALT = "vllm:prefix_cache_queries_total"
VLLM_PREFIX_HITS = "vllm:prefix_cache_hits"
VLLM_PREFIX_HITS_ALT = "vllm:prefix_cache_hits_total"

# NIXL metrics
NIXL_XFER_COUNT = "vllm:nixl_xfer_time_seconds_count"
NIXL_XFER_SUM = "vllm:nixl_xfer_time_seconds_sum"
NIXL_BYTES_SUM = "vllm:nixl_bytes_transferred_sum"
NIXL_FAILED_TRANSFERS = "vllm:nixl_num_failed_transfers_total"
NIXL_FAILED_NOTIFICATIONS = "vllm:nixl_num_failed_notifications_total"

# P/D token source metrics
VLLM_PROMPT_BY_SOURCE = "vllm:prompt_tokens_by_source_total"
VLLM_DECODE_TIME = "vllm:request_decode_time_seconds_sum"

# EPP / Scheduler metrics
# Current llm-d-router metrics use the llm_d_epp namespace. Keep the legacy
# names for clusters that still expose the older Inference Extension series.
SCHED_E2E = "llm_d_epp_scheduler_e2e_duration_seconds_count"
SCHED_E2E_LEGACY = "inference_extension_scheduler_e2e_duration_seconds_count"
SCHED_REQUEST_TOTAL = "llm_d_epp_request_total"
SCHED_REQUEST_ERROR = "llm_d_epp_request_error_total"
SCHED_REQUEST_ERROR_LEGACY = "inference_objective_request_error_total"
READY_ENDPOINTS = "llm_d_epp_ready_endpoints"
READY_ENDPOINTS_LEGACY = "inference_pool_ready_pods"
PREFIX_INDEXER_SIZE = "llm_d_epp_prefix_indexer_size"

# LoRA metrics (vLLM reports adapter state via lora_requests_info)
VLLM_LORA_REQUESTS_INFO = "vllm:lora_requests_info"

# Flow Control metrics
FC_DISPATCH_CYCLE = "llm_d_epp_flow_control_dispatch_cycle_duration_seconds_count"
FC_DISPATCH_CYCLE_LEGACY = "inference_extension_flow_control_dispatch_cycle_duration_seconds_count"
FC_POOL_SATURATION = "llm_d_epp_flow_control_pool_saturation"
FC_POOL_SATURATION_LEGACY = "inference_extension_flow_control_pool_saturation"
FC_REQUEST_ENQUEUE = "llm_d_epp_flow_control_request_enqueue_duration_seconds_count"
FC_REQUEST_ENQUEUE_LEGACY = "inference_extension_flow_control_request_enqueue_duration_seconds_count"
FC_QUEUE_DURATION = "llm_d_epp_flow_control_request_queue_duration_seconds_count"
FC_QUEUE_DURATION_LEGACY = "inference_extension_flow_control_request_queue_duration_seconds_count"


# KV-cache offloading metrics (vLLM native offloading KV connector).
# Legacy (vLLM 0.23.x and the native CPU offloading backend): kv_offload_total_bytes_total deprecated in vLLM >= 0.26 but still emitted for CPUOffloadingSpec.
# KV_OFFLOAD_STORE_BYTES = "vllm:kv_offload_store_bytes_total"  # GPU -> offload storage
# KV_OFFLOAD_LOAD_BYTES = "vllm:kv_offload_load_bytes_total"  # offload storage -> GPU
KV_OFFLOAD_TOTAL_BYTES = "vllm:kv_offload_total_bytes_total"  # deprecated legacy counter

# Label patterns for pod discovery
WORKLOAD_LABEL = "app.kubernetes.io/name={name},app.kubernetes.io/component=llminferenceservice-workload"
PREFILL_LABEL = "app.kubernetes.io/name={name},app.kubernetes.io/component=llminferenceservice-workload-prefill"
EPP_LABELS = [
    "app.kubernetes.io/name={name}-epp",
    "app.kubernetes.io/component=llminferenceservice-router-scheduler,app.kubernetes.io/name={name}",
    "app.kubernetes.io/component=endpoint-picker,app.kubernetes.io/name={name}",
    "app.kubernetes.io/component=router-scheduler,app.kubernetes.io/name={name}",
    "kserve.io/component=scheduler,app.kubernetes.io/name={name}",
]


@dataclass
class Metric:
    name: str
    labels: dict[str, str] = field(default_factory=dict)
    value: float = 0.0


@dataclass
class ScrapeResult:
    source: str
    metrics: dict[str, list[Metric]] = field(default_factory=dict)
    raw_text: str = ""

    def get(self, name: str, fallback: str = "", **label_filter) -> float | None:
        for key in (name, fallback) if fallback else (name,):
            if key in self.metrics:
                values = self.metrics[key]
                if label_filter:
                    values = [m for m in values if all(m.labels.get(k) == v for k, v in label_filter.items())]
                if values:
                    return sum(m.value for m in values)
        return None

    def has(self, name: str, fallback: str = "") -> bool:
        return self.get(name, fallback) is not None


@dataclass
class CheckResult:
    name: str
    metric: str
    source: str
    value: float
    passed: bool
    message: str


def parse_prometheus(text: str) -> dict[str, list[Metric]]:
    """Parse Prometheus text exposition format into indexed metrics."""
    result: dict[str, list[Metric]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{?(.*?)\}?\s+([\d.eE+\-]+|NaN|Inf|-Inf)$", line)
        if not match:
            match = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([\d.eE+\-]+|NaN|Inf|-Inf)$", line)
            if match:
                name, val = match.group(1), match.group(2)
                try:
                    m = Metric(name=name, value=float(val))
                except ValueError:
                    continue
                result.setdefault(name, []).append(m)
            continue
        name, label_str, val = match.group(1), match.group(2), match.group(3)
        labels = {}
        if label_str:
            for lm in re.finditer(r'(\w+)="([^"]*)"', label_str):
                labels[lm.group(1)] = lm.group(2)
        try:
            m = Metric(name=name, labels=labels, value=float(val))
        except ValueError:
            continue
        result.setdefault(name, []).append(m)
    return result


class Scraper:
    """Scrapes Prometheus metrics from pods via kubectl exec or port-forward."""

    def __init__(self, kubectl_fn, namespace: str, kubeconfig: str = ""):
        self._kubectl = kubectl_fn
        self.namespace = namespace
        self.kubeconfig = kubeconfig

    def _scrape_via_exec(self, pod: str, port: int) -> str:
        # Try HTTPS first (workload pods with TLS), then HTTP (EPP metrics port).
        for scheme in ("https", "http"):
            script = (
                f"import urllib.request,ssl; "
                f"print(urllib.request.urlopen('{scheme}://localhost:{port}/metrics'"
                + (",context=ssl._create_unverified_context()" if scheme == "https" else "")
                + ").read().decode())"
            )
            try:
                return self._kubectl("exec", pod, "-n", self.namespace, "--", "python3", "-c", script)
            except RuntimeError:
                try:
                    wget_args = ["--no-check-certificate"] if scheme == "https" else []
                    return self._kubectl(
                        "exec",
                        pod,
                        "-n",
                        self.namespace,
                        "--",
                        "wget",
                        *wget_args,
                        "-qO-",
                        f"{scheme}://localhost:{port}/metrics",
                    )
                except RuntimeError:
                    continue
        raise RuntimeError(f"exec scrape failed for {pod}:{port} on both https and http")

    def _get_pod_sa_token(self, pod: str) -> str:
        """Get a bearer token for the pod's service account via kubectl create token."""
        sa = self._kubectl(
            "get",
            "pod",
            pod,
            "-n",
            self.namespace,
            "-o",
            "jsonpath={.spec.serviceAccountName}",
            check=False,
        )
        if not sa:
            return ""
        cmd = ["kubectl"]
        if self.kubeconfig:
            cmd += ["--kubeconfig", self.kubeconfig]
        cmd += ["create", "token", sa, "-n", self.namespace]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout.strip() if result.returncode == 0 else ""

    def _scrape_via_port_forward(self, pod: str, port: int) -> str:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            local_port = s.getsockname()[1]
        cmd = ["kubectl"]
        if self.kubeconfig:
            cmd += ["--kubeconfig", self.kubeconfig]
        cmd += ["port-forward", "-n", self.namespace, pod, f"{local_port}:{port}"]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            time.sleep(2)
            if proc.poll() is not None:
                raise RuntimeError(f"Port-forward to {pod}:{port} failed to start")
            token = self._get_pod_sa_token(pod)
            headers_with_auth = {"Authorization": f"Bearer {token}"} if token else {}
            for scheme in ("https", "http"):
                for headers in (headers_with_auth, {}) if headers_with_auth else ({},):
                    try:
                        r = httpx.get(
                            f"{scheme}://localhost:{local_port}/metrics",
                            headers=headers,
                            verify=False,
                            timeout=15,
                        )
                        r.raise_for_status()
                        return r.text
                    except (httpx.ConnectError, httpx.HTTPStatusError):
                        continue
            raise RuntimeError(f"Could not reach {pod}:{port} metrics via port-forward")
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def scrape_pod(self, pod: str, port: int = 8000) -> ScrapeResult:
        try:
            text = self._scrape_via_exec(pod, port)
        except RuntimeError:
            log.info("exec failed for %s, falling back to port-forward", pod)
            text = self._scrape_via_port_forward(pod, port)
        return ScrapeResult(source=pod, metrics=parse_prometheus(text), raw_text=text)

    def scrape_pods_by_label(self, label: str, port: int = 8000) -> list[ScrapeResult]:
        output = self._kubectl(
            "get",
            "pods",
            "-n",
            self.namespace,
            "-l",
            label,
            "-o",
            "jsonpath={.items[*].metadata.name}",
        )
        pods = output.split() if output else []
        results = []
        for pod in pods:
            try:
                results.append(self.scrape_pod(pod, port))
            except RuntimeError as e:
                log.warning("Failed to scrape %s: %s", pod, e)
        return results

    def scrape_vllm(self, name: str) -> list[ScrapeResult]:
        label = WORKLOAD_LABEL.format(name=name)
        return self.scrape_pods_by_label(label, port=8000)

    def scrape_prefill(self, name: str) -> list[ScrapeResult]:
        label = PREFILL_LABEL.format(name=name)
        return self.scrape_pods_by_label(label, port=8000)

    def scrape_epp(self, name: str) -> list[ScrapeResult]:
        for label_tmpl in EPP_LABELS:
            label = label_tmpl.format(name=name)
            results = self.scrape_pods_by_label(label, port=9090)
            if results:
                return results
        return []


def dump_raw_metrics(results: list[ScrapeResult], output_dir: str, label: str = "") -> list[str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for r in results:
        if not r.raw_text:
            continue
        prefix = f"{label}-" if label else ""
        safe_name = r.source.replace("/", "_")
        path = out / f"{prefix}{safe_name}.prom"
        path.write_text(r.raw_text)
        paths.append(str(path))
    return paths


def validate_vllm_basic(results: list[ScrapeResult]) -> list[CheckResult]:
    checks = []
    total_success = 0.0
    for r in results:
        val = r.get(VLLM_REQUEST_SUCCESS)
        if val is not None:
            total_success += val
        checks.append(
            CheckResult(
                name="request_success",
                metric=VLLM_REQUEST_SUCCESS,
                source=r.source,
                value=val or 0,
                passed=True,
                message=f"request_success={val}" if val else "no traffic yet",
            )
        )
    checks.append(
        CheckResult(
            name="request_success_aggregate",
            metric=VLLM_REQUEST_SUCCESS,
            source="all-pods",
            value=total_success,
            passed=total_success > 0,
            message=f"aggregate request_success={total_success}",
        )
    )
    return checks


def validate_cache_aware(vllm: list[ScrapeResult], epp: list[ScrapeResult]) -> list[CheckResult]:
    checks = validate_vllm_basic(vllm)
    total_queries = 0.0
    total_hits = 0.0
    for r in vllm:
        queries = r.get(VLLM_PREFIX_QUERIES, VLLM_PREFIX_QUERIES_ALT)
        hits = r.get(VLLM_PREFIX_HITS, VLLM_PREFIX_HITS_ALT)
        if queries is not None:
            total_queries += queries
        if hits is not None:
            total_hits += hits
        checks.append(
            CheckResult(
                name="prefix_queries",
                metric=VLLM_PREFIX_QUERIES,
                source=r.source,
                value=queries or 0,
                passed=True,
                message=f"prefix_queries={queries}" if queries is not None else "no traffic on this pod",
            )
        )
        checks.append(
            CheckResult(
                name="prefix_hits",
                metric=VLLM_PREFIX_HITS,
                source=r.source,
                value=hits or 0,
                passed=True,
                message=f"prefix_hits={hits}" if hits is not None else "no traffic on this pod",
            )
        )
    checks.append(
        CheckResult(
            name="prefix_queries_aggregate",
            metric=VLLM_PREFIX_QUERIES,
            source="all-pods",
            value=total_queries,
            passed=total_queries > 0,
            message=f"aggregate prefix_queries={total_queries}",
        )
    )
    checks.append(
        CheckResult(
            name="prefix_hits_aggregate",
            metric=VLLM_PREFIX_HITS,
            source="all-pods",
            value=total_hits,
            passed=total_hits > 0,
            message=f"aggregate prefix_hits={total_hits}",
        )
    )
    if total_queries > 0:
        rate = total_hits / total_queries * 100
        checks.append(
            CheckResult(
                name="prefix_hit_rate",
                metric="prefix_cache_hit_rate",
                source="all-pods",
                value=rate,
                passed=rate > 0,
                message=f"aggregate hit_rate={rate:.1f}%",
            )
        )
    return checks


def validate_pd(decode: list[ScrapeResult], prefill: list[ScrapeResult]) -> list[CheckResult]:
    checks = validate_vllm_basic(decode)
    total_gen = sum((r.get(VLLM_GEN_TOKENS) or 0) for r in decode)
    checks.append(
        CheckResult(
            name="decode_gen_tokens",
            metric=VLLM_GEN_TOKENS,
            source="decode-aggregate",
            value=total_gen,
            passed=total_gen > 0,
            message=f"decode generation_tokens={total_gen}",
        )
    )
    total_prompt = sum((r.get(VLLM_PROMPT_TOKENS) or 0) for r in prefill)
    checks.append(
        CheckResult(
            name="prefill_prompt_tokens",
            metric=VLLM_PROMPT_TOKENS,
            source="prefill-aggregate",
            value=total_prompt,
            passed=total_prompt > 0,
            message=f"prefill prompt_tokens={total_prompt}",
        )
    )
    nixl_xfers = sum((r.get(NIXL_XFER_COUNT) or 0) for r in decode)
    checks.append(
        CheckResult(
            name="nixl_transfers",
            metric=NIXL_XFER_COUNT,
            source="decode-aggregate",
            value=nixl_xfers,
            passed=nixl_xfers > 0,
            message=f"nixl transfers={nixl_xfers:.0f}",
        )
    )
    nixl_failed = sum((r.get(NIXL_FAILED_TRANSFERS) or 0) for r in decode + prefill)
    checks.append(
        CheckResult(
            name="nixl_failed_transfers",
            metric=NIXL_FAILED_TRANSFERS,
            source="all-pods",
            value=nixl_failed,
            passed=nixl_failed == 0,
            message=f"nixl failed_transfers={nixl_failed:.0f}",
        )
    )
    nixl_bytes = sum((r.get(NIXL_BYTES_SUM) or 0) for r in decode)
    checks.append(
        CheckResult(
            name="nixl_bytes_transferred",
            metric=NIXL_BYTES_SUM,
            source="decode-aggregate",
            value=nixl_bytes,
            passed=nixl_bytes > 0,
            message=f"nixl bytes_transferred={nixl_bytes:.0f}",
        )
    )
    decode_by_source = {}
    prefill_by_source = {}
    for role, pods, store in [("decode", decode, decode_by_source), ("prefill", prefill, prefill_by_source)]:
        for src in ("local_compute", "local_cache_hit", "external_kv_transfer"):
            total = sum((r.get(VLLM_PROMPT_BY_SOURCE, source=src) or 0) for r in pods)
            store[src] = total
            is_required = (role == "decode" and src == "external_kv_transfer") or (
                role == "prefill" and src == "local_compute"
            )
            checks.append(
                CheckResult(
                    name=f"{role}_{src}",
                    metric=VLLM_PROMPT_BY_SOURCE,
                    source=f"{role}-aggregate",
                    value=total,
                    passed=total > 0 if is_required else True,
                    message=f"{role} {src}={total:.0f}",
                )
            )
    kv_transfer = decode_by_source.get("external_kv_transfer", 0)
    local_compute = decode_by_source.get("local_compute", 0)
    checks.append(
        CheckResult(
            name="decode_kv_over_compute",
            metric=VLLM_PROMPT_BY_SOURCE,
            source="decode-aggregate",
            value=kv_transfer,
            passed=kv_transfer > local_compute,
            message=f"decode kv_transfer={kv_transfer:.0f} vs local_compute={local_compute:.0f}",
        )
    )
    return checks


def validate_scheduler(epp: list[ScrapeResult]) -> list[CheckResult]:
    checks = []
    for r in epp:
        e2e = r.get(SCHED_E2E, SCHED_E2E_LEGACY)
        checks.append(
            CheckResult(
                name="scheduler_e2e",
                metric=SCHED_E2E,
                source=r.source,
                value=e2e or 0,
                passed=e2e is not None and e2e > 0,
                message=f"scheduler_e2e_count={e2e}",
            )
        )
        errors = r.get(SCHED_REQUEST_ERROR, SCHED_REQUEST_ERROR_LEGACY)
        if errors is not None:
            checks.append(
                CheckResult(
                    name="request_errors",
                    metric=SCHED_REQUEST_ERROR,
                    source=r.source,
                    value=errors,
                    passed=errors == 0,
                    message=f"request_errors={errors}",
                )
            )
        endpoints = r.get(READY_ENDPOINTS, READY_ENDPOINTS_LEGACY)
        if endpoints is not None:
            checks.append(
                CheckResult(
                    name="ready_endpoints",
                    metric=READY_ENDPOINTS,
                    source=r.source,
                    value=endpoints,
                    passed=endpoints > 0,
                    message=f"ready_endpoints={endpoints}",
                )
            )
    return checks


def validate_flow_control(epp: list[ScrapeResult]) -> list[CheckResult]:
    """Validate flow control metrics from EPP pods."""
    checks = []
    for r in epp:
        dispatch = r.get(FC_DISPATCH_CYCLE, FC_DISPATCH_CYCLE_LEGACY)
        checks.append(
            CheckResult(
                name="fc_dispatch_cycle",
                metric=FC_DISPATCH_CYCLE,
                source=r.source,
                value=dispatch or 0,
                passed=dispatch is not None and dispatch > 0,
                message=f"dispatch_cycle_count={dispatch}",
            )
        )
        saturation = r.get(FC_POOL_SATURATION, FC_POOL_SATURATION_LEGACY)
        checks.append(
            CheckResult(
                name="fc_pool_saturation",
                metric=FC_POOL_SATURATION,
                source=r.source,
                value=saturation or 0,
                passed=saturation is not None,
                message=f"pool_saturation={saturation}",
            )
        )
        enqueue = r.get(FC_REQUEST_ENQUEUE, FC_REQUEST_ENQUEUE_LEGACY)
        checks.append(
            CheckResult(
                name="fc_request_enqueue",
                metric=FC_REQUEST_ENQUEUE,
                source=r.source,
                value=enqueue or 0,
                passed=enqueue is not None and enqueue > 0,
                message=f"request_enqueue_count={enqueue}",
            )
        )
        dispatched = r.get(FC_QUEUE_DURATION, FC_QUEUE_DURATION_LEGACY)
        checks.append(
            CheckResult(
                name="fc_request_dispatched",
                metric=FC_QUEUE_DURATION,
                source=r.source,
                value=dispatched or 0,
                passed=dispatched is not None and dispatched > 0,
                message=f"request_queue_dispatched_count={dispatched}",
            )
        )
    return checks


def validate_lora(vllm: list[ScrapeResult]) -> list[CheckResult]:
    """Validate LoRA adapter metrics from vLLM workload pods."""
    checks = []
    for r in vllm:
        lora_info = r.get(VLLM_LORA_REQUESTS_INFO)
        checks.append(
            CheckResult(
                name="lora_requests_info",
                metric=VLLM_LORA_REQUESTS_INFO,
                source=r.source,
                value=lora_info or 0,
                passed=lora_info is not None,
                message=f"lora_requests_info={'present' if lora_info is not None else 'missing'}",
            )
        )
    return checks


def validate_kvcache_offloading_cpu(vllm: list[ScrapeResult]) -> list[CheckResult]:
    checks = []
    total_offloaded = 0.0  # aggregated value when we have multiple vllm pods
    for r in vllm:
        offloaded = r.get(KV_OFFLOAD_TOTAL_BYTES)
        if offloaded is not None:
            total_offloaded += offloaded
        checks.append(
            CheckResult(
                name="kv_offload_bytes",
                metric=KV_OFFLOAD_TOTAL_BYTES,  # we need extend this in later version
                source=r.source,
                value=offloaded or 0,
                passed=True,  # dont care only need total_offloaded to be increased
                message=(
                    f"kv_offload_total_bytes={offloaded}"
                    if offloaded is not None
                    else "kv offload metric not exposed on this pod"
                ),
            )
        )
    checks.append(
        CheckResult(
            name="kv_offload_total",
            metric=KV_OFFLOAD_TOTAL_BYTES,
            source="aggregate",
            value=total_offloaded,
            passed=total_offloaded > 0,
            message=f"total_kv_offload_bytes={total_offloaded}",
        )
    )
    return checks


def validate_kvcache_offloading_fs(
    usage_by_pod: dict[str, int | None], path: str, min_bytes: int = 1048576
) -> list[CheckResult]:
    metric = f"disk_usage:{path}"
    checks = []
    total_used = 0
    for pod, used in usage_by_pod.items():
        if used is not None:
            total_used += used
        checks.append(
            CheckResult(
                name="kv_offload_fs_bytes",
                metric=metric,
                source=pod,
                value=used or 0,
                passed=True,
                message=(f"{path} uses {used} bytes" if used is not None else f"{path} not present on this pod"),
            )
        )
    checks.append(
        CheckResult(
            name="kv_offload_fs_total",
            metric=metric,
            source="aggregate",
            value=total_used,
            passed=total_used > min_bytes,
            message=f"total_fs_offload_bytes={total_used} under {path} (threshold {min_bytes})",
        )
    )
    return checks
