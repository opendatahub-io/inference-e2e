# Troubleshooting

Symptoms first; causes and fixes second. For setup see `docs/quick-start.md`.

## The run

| Symptom | Cause / fix |
|---|---|
| Every phase SKIPPED instantly, "needs N GPU(s) but cluster has 0" | kubectl could not reach the cluster — the suite reads that as zero GPUs. Check `kubectl cluster-info` with the same `KUBECONFIG`; expired token and VPN are the usual culprits. |
| Case SKIPPED, "needs N GPU(s) but cluster has M" | Not enough free GPUs. Check for leftovers: `kubectl get llminferenceservice -A`. The suite only sweeps its own leftovers (label `llm-d-e2e.opendatahub.io/managed=true`) — someone else's deployment may be holding GPUs. |
| Case SKIPPED in `--mock`, "requiresGpu set" | Expected: the case validates real GPU behavior (see `docs/test-cases.md`). |
| "manifest not found for this branch" | The manifest isn't in the branch you cloned — rerun `--setup` with a branch that has it (`--list-testcases` shows ✓/✗ per case). |
| Deploy fails fast with operator ImagePullBackOff hints | Operator pods can't pull images — cluster-side registry problem, not the test. |

## Pods

| Symptom | Cause / fix |
|---|---|
| `CreateContainerConfigError: secrets "rhai-pull-secret" not found` | The manifests reference a pull secret the cluster doesn't have. If the nodes pull images without pod-level auth, create a placeholder in `default` (see `docs/quick-start.md`) — the suite copies it into the test namespace. |
| Pods `ImagePullBackOff` | The workload image genuinely can't be pulled — check the registry mirror / pull secret on the cluster. |
| Pods stay `Pending` | Unschedulable — usually GPUs already allocated (see above) or node taints. |
| CrashLoopBackOff | The suite fails the deploy after ~45 s instead of waiting the full timeout. Common cause: a manifest using EPP plugins the installed RHAII version doesn't have — check the manifest branch against the cluster version. |
| Ready never becomes True | The suite fast-fails when the Ready condition's reason+message stops changing; the final message names the blocker (e.g. "no Gateway controller has accepted" → gateway/HTTPRoute problem, see below). |

## Routing and inference

| Symptom | Cause / fix |
|---|---|
| Phase 04 times out, "Gateway ... not programmed" | The Gateway the service's HTTPRoute attaches to has no address. `kubectl get gateway -A` and check its Programmed condition; the timeout error includes the last observed reason. |
| Inference returns 401/403 | The default per-testcase service-account token was rejected. Try `--bearer-token <token>`, or `--disable-auth` if the endpoint is anonymous. |
| Health/models return 503 through the gateway | Expected — the EPP only routes inference. The suite reaches those endpoints directly on the pod; a 503 here means the port-forward path in the suite found the wrong pod. |
| Model download hangs | The workload node lacks egress to huggingface.co. |

## Interrupting and leftovers

One Ctrl-C is always safe — pytest finishes deleting the deployed resources
before exiting (a second Ctrl-C force-quits). A run killed outright (SIGKILL,
closed terminal) leaves its LLMInferenceServices behind; the next run sweeps
them automatically at start unless `--nocleanup` is set. To inspect manually:

```bash
kubectl -n llm-conformance-test get llminferenceservice,pods,httproute
```
