"""CLI entry point (``llm-d-e2e``) — translates flags to pytest and runs phases.

``main()`` parses argparse options, handles utility commands, then builds a
pytest argv for ``tests/test_conformance.py``. Every value flag in
``flag_map`` must have a matching ``pytest_addoption`` in ``conftest.py``;
boolean flags (``--disable-auth``, ``--nocleanup``) are appended separately.

Utility commands (may exit before pytest):
  - ``--list-testcases`` / ``--list-profiles`` — print configs + manifest ref
  - ``--setup [REF]`` — clone ``llm-d-conformance-manifests``, prune stale
    YAML, write ``deploy/manifests/.manifest-ref``; interactive branch pick
    if REF omitted and stdin is a TTY

Run modes (``--mode``): deploy (default), discover (validate existing;
needs ``--endpoint`` or auto-detect), cache (PVC download only).

Other notable flags: ``--mock`` (simulator; default image
``DEFAULT_MOCK_IMAGE``), ``--render-image``, auth/pull-secret, PVC storage,
P/D node selectors, GuideLLM image, ``--html`` / ``--fail-fast`` / 6h timeout.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_MOCK_IMAGE = "ghcr.io/llm-d/llm-d-inference-sim:latest"
MANIFEST_REPO = "https://github.com/opendatahub-io/llm-d-conformance-manifests.git"


def main():
    parser = argparse.ArgumentParser(
        prog="llm-d-e2e",
        description="End-to-end conformance tests for llm-d / KServe LLMInferenceService",
    )

    # Test selection
    parser.add_argument("--testcase", "-t", default="", help="Test case name(s), comma-separated")
    parser.add_argument("--profile", "-p", default="", help="Profile YAML path")
    parser.add_argument("--testcase-dir", default="configs/testcases", help="Test case directory")

    # Cluster
    parser.add_argument("--platform", default="any", choices=["any", "ocp", "aks", "gks"], help="Platform")
    parser.add_argument("--namespace", "-n", default="llm-conformance-test", help="Kubernetes namespace")
    parser.add_argument("--kubeconfig", default="", help="Path to kubeconfig")

    # Mode
    parser.add_argument("--mode", default="deploy", choices=["deploy", "discover", "cache"], help="Run mode")
    parser.add_argument("--model-source", default="hf", choices=["hf", "pvc"], help="Model source")
    parser.add_argument("--model", default="", help="Override model name")
    parser.add_argument("--endpoint", default="", help="Service URL for discover mode")
    parser.add_argument(
        "--mock",
        default="",
        nargs="?",
        const=DEFAULT_MOCK_IMAGE,
        help=f"Use simulator image (no GPU needed). Default: {DEFAULT_MOCK_IMAGE}",
    )
    parser.add_argument(
        "--render-image", default="", help="vLLM CPU image for tokenizer render sidecar (used with --mock)"
    )

    # Auth
    parser.add_argument("--pull-secret", default="", help="Pull secret name")
    parser.add_argument("--bearer-token", default="", help="Bearer token for auth")
    parser.add_argument("--disable-auth", action="store_true", help="Disable WASM auth annotation")

    # Storage
    parser.add_argument("--storage-class", default="", help="StorageClass for PVC")
    parser.add_argument("--storage-size", default="", help="Override PVC size")

    # Benchmark
    parser.add_argument("--guidellm-image", default="", help="GuideLLM benchmark image override")

    # Node placement
    parser.add_argument("--decode-node-selector", default="", help="Node selector for decode pods (key=value)")
    parser.add_argument("--prefill-node-selector", default="", help="Node selector for prefill pods (key=value)")

    # Behavior
    parser.add_argument("--nocleanup", action="store_true", help="Keep resources after test")
    parser.add_argument("--report-dir", default="reports", help="Report output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fail-fast", "-x", action="store_true", help="Stop on first failure")
    parser.add_argument("--html", default="", help="Generate HTML report at path")

    # Utility subcommands
    parser.add_argument("--list-testcases", action="store_true", help="List available test cases")
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles")
    parser.add_argument(
        "--setup",
        default=None,
        metavar="REF",
        nargs="?",
        const="",
        help="Clone manifest repo (interactive if no branch given)",
    )
    parser.add_argument(
        "--manifest-repo",
        default=MANIFEST_REPO,
        metavar="URL",
        help=f"Git repo to clone manifests from with --setup (default: {MANIFEST_REPO})",
    )

    args = parser.parse_args()

    # Handle utility commands
    if args.list_testcases:
        _list_testcases(args.testcase_dir)
        return

    if args.list_profiles:
        _list_profiles()
        return

    if args.setup is None and args.manifest_repo != MANIFEST_REPO:
        print("Warning: --manifest-repo has no effect without --setup", file=sys.stderr)

    if args.setup is not None:
        if args.setup == "":
            ref = _interactive_setup(args.manifest_repo) if sys.stdin.isatty() else "main"
        else:
            ref = args.setup
        _setup_manifests(ref, args.manifest_repo)
        sys.stdout.flush()
        if not args.testcase and not args.profile:
            return

    # Build pytest args
    pytest_args = ["tests/test_conformance.py"]

    # Pass all flags through to pytest
    flag_map = {
        "testcase": "--testcase",
        "profile": "--profile",
        "testcase_dir": "--testcase-dir",
        "platform": "--platform",
        "namespace": "--namespace",
        "kubeconfig": "--kubeconfig",
        "mode": "--mode",
        "model_source": "--model-source",
        "model": "--model",
        "endpoint": "--endpoint",
        "mock": "--mock",
        "render_image": "--render-image",
        "pull_secret": "--pull-secret",
        "bearer_token": "--bearer-token",
        "storage_class": "--storage-class",
        "storage_size": "--storage-size",
        "report_dir": "--report-dir",
        "guidellm_image": "--guidellm-image",
        "decode_node_selector": "--decode-node-selector",
        "prefill_node_selector": "--prefill-node-selector",
    }

    for attr, flag in flag_map.items():
        val = getattr(args, attr)
        if val:
            pytest_args.extend([flag, val])

    if args.disable_auth:
        pytest_args.append("--disable-auth")
    if args.nocleanup:
        pytest_args.append("--nocleanup")
    if args.verbose:
        pytest_args.append("-v")
    if args.fail_fast:
        pytest_args.append("-x")
    if args.html:
        pytest_args.extend(["--html", args.html, "--self-contained-html"])

    pytest_args.extend(["--tb", "short", "--timeout", "21600"])

    sys.exit(_run_pytest(["python", "-m", "pytest"] + pytest_args))


def _run_pytest(cmd: list[str]) -> int:
    """Run pytest, letting it finish session teardown when interrupted.

    Ctrl-C reaches pytest directly (same process group) and its teardown deletes
    deployed services. ``subprocess.call`` would SIGKILL pytest 0.25s after Ctrl-C,
    leaving those services holding GPUs. A second Ctrl-C force-quits.
    """
    proc = subprocess.Popen(cmd)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        print(
            "\nInterrupted: waiting for pytest to delete deployed resources (Ctrl-C again to force quit)...",
            file=sys.stderr,
        )
        try:
            return proc.wait()
        except KeyboardInterrupt:
            proc.kill()
            proc.wait()
            raise


def _list_testcases(testcase_dir: str):
    from conformance.config import iter_config_yamls, load_testcase
    from conformance.deployer import Deployer

    ref_file = Path("deploy/manifests/.manifest-ref")
    if ref_file.exists():
        info = ref_file.read_text().strip()
        print(f"Manifests: {info}")
    else:
        print("Manifests: not set up (run --setup <branch>)")
    print()
    print("Test cases:")
    dep = Deployer()
    for f in iter_config_yamls(testcase_dir):
        tc = load_testcase(str(f))
        # GPU count is sourced from the manifest (single source of truth); "?" when
        # manifests aren't cloned yet (--list-testcases can run before --setup).
        try:
            gpus = dep.manifest_gpu_needed(tc)
        except FileNotFoundError:
            gpus = "?"
        print(f"  {tc.name:<28s} [{gpus} GPU]  {tc.description}")


def _list_profiles():
    import yaml

    from conformance.config import iter_config_yamls

    ref_file = Path("deploy/manifests/.manifest-ref")
    if ref_file.exists():
        info = ref_file.read_text().strip()
        print(f"Manifests: {info}")
    else:
        print("Manifests: not set up (run --setup <branch>)")
    print()
    print("Profiles:")
    for f in iter_config_yamls("configs/profiles"):
        with open(f) as fh:
            data = yaml.safe_load(fh)
        name = data.get("name", f.stem)
        desc = data.get("description", "")
        cases = ", ".join(data.get("testCases", []))
        print(f"  {name:<20s} {desc}")
        print(f"  {'':20s} tests: {cases}")


def _interactive_setup(repo: str = MANIFEST_REPO) -> str:
    print(f"Fetching branches from {repo}...")
    result = subprocess.run(
        ["git", "ls-remote", "--heads", repo],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Failed to fetch branches, defaulting to main")
        return "main"

    branches = sorted(line.split("refs/heads/")[-1] for line in result.stdout.strip().splitlines())
    print("\nAvailable branches:")
    for i, b in enumerate(branches, 1):
        print(f"  {i}) {b}")

    print()
    choice = input("Select branch number (or Enter for main): ").strip()
    if choice:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(branches):
                return branches[idx]
            print("Invalid selection, using main")
        except ValueError:
            print("Invalid input, using main")
    return "main"


def _setup_manifests(ref: str, repo: str = MANIFEST_REPO):
    import yaml
    from datetime import datetime, timezone

    manifest_dir = Path("deploy/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cloning manifests from {ref}...")
    with tempfile.TemporaryDirectory(prefix="llm-d-manifests-") as temp_dir:
        clone_dir = Path(temp_dir) / "repo"
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", ref, repo, str(clone_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Failed to clone manifests: {result.stderr.strip()}")
            sys.exit(1)

        commit = subprocess.run(
            ["git", "-C", str(clone_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()

        for stale in manifest_dir.glob("*.yaml"):
            stale.unlink()
        for f in clone_dir.glob("*.yaml"):
            (manifest_dir / f.name).write_text(f.read_text())

    ref_file = manifest_dir / ".manifest-ref"
    ref_file.write_text(
        f"branch: {ref}\n"
        f"repo: {repo}\n"
        f"commit: {commit}\n"
        f"date: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
    )

    print(f"Manifests ready in {manifest_dir}/ (branch: {ref}, commit: {commit[:8]})")

    from conformance.config import iter_config_yamls

    testcase_dir = Path("configs/testcases")
    if testcase_dir.exists():
        print("\nTest cases:")
        for tc_file in iter_config_yamls(testcase_dir):
            with open(tc_file) as fh:
                data = yaml.safe_load(fh)
            name = data.get("name", tc_file.stem)
            manifest = data.get("deployment", {}).get("manifestPath", "")
            if manifest and (manifest_dir / manifest).exists():
                print(f"  \033[32m✓\033[0m {name:<28s} → {manifest}")
            else:
                print(f"  \033[31m✗\033[0m {name:<28s} → {manifest} (missing)")


if __name__ == "__main__":
    main()
