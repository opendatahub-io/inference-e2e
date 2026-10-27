"""Unit tests for CLI helpers: manifest setup, require_manifest, scaffolding scripts.

Run with: ``uv run pytest tests/test_cli.py -v``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conformance.config import load_testcase


@pytest.mark.parametrize("auth_args", [[], ["--disable-auth"], ["--bearer-token", "provided-token"]])
def test_auth_options_are_forwarded_to_pytest(monkeypatch, auth_args):
    import conformance.cli as cli

    commands = []
    monkeypatch.setattr(sys, "argv", ["llm-d-e2e", "-t", "single-gpu-smoke", *auth_args])
    monkeypatch.setattr(cli, "_run_pytest", lambda cmd: commands.append(cmd) or 0)
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 0
    assert all(arg in commands[0] for arg in auth_args)
    assert "--service-account-auth" not in commands[0]


def test_setup_manifests_removes_stale_files(tmp_path, monkeypatch):
    """Switching manifest branches must remove stale files from the previous branch."""
    import shutil
    from unittest.mock import MagicMock, patch

    import conformance.cli as cli_mod

    monkeypatch.chdir(tmp_path)

    manifest_dir = tmp_path / "deploy" / "manifests"
    manifest_dir.mkdir(parents=True)
    for stale in ["flow-control-tokens.yaml", "flow-control.yaml", "pd-performance.yaml"]:
        (manifest_dir / stale).write_text("stale: true")

    clone_dir = Path("/tmp/llm-d-manifests")
    clone_dir.mkdir(exist_ok=True)
    for new in ["single-gpu.yaml", "cache-aware.yaml"]:
        (clone_dir / new).write_text("branch: 3.4-stable")

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "abc1234deadbeef\n"
        result.stderr = ""
        if cmd[0] == "rm":
            shutil.rmtree(str(clone_dir), ignore_errors=True)
        return result

    with patch.object(cli_mod, "subprocess") as mock_sub:
        mock_sub.run.side_effect = fake_run
        cli_mod._setup_manifests("3.4-stable")

    remaining = {f.name for f in manifest_dir.glob("*.yaml")}
    assert "flow-control-tokens.yaml" not in remaining
    assert "flow-control.yaml" not in remaining
    assert "pd-performance.yaml" not in remaining
    assert "single-gpu.yaml" in remaining
    assert "cache-aware.yaml" in remaining


def test_setup_manifests_uses_custom_repo(tmp_path, monkeypatch):
    """--manifest-repo <URL> must clone from the given repo, not the default."""
    import shutil
    from unittest.mock import MagicMock, patch

    import conformance.cli as cli_mod

    monkeypatch.chdir(tmp_path)

    custom_repo = "https://github.com/my-org/my-repo.git"

    clone_dir = Path("/tmp/llm-d-manifests")
    clone_dir.mkdir(exist_ok=True)
    (clone_dir / "single-gpu.yaml").write_text("branch: my-branch")

    clone_cmds = []

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "abc1234deadbeef\n"
        result.stderr = ""
        if cmd[:2] == ["git", "clone"]:
            clone_cmds.append(cmd)
        if cmd[0] == "rm":
            shutil.rmtree(str(clone_dir), ignore_errors=True)
        return result

    with patch.object(cli_mod, "subprocess") as mock_sub:
        mock_sub.run.side_effect = fake_run
        cli_mod._setup_manifests("my-branch", custom_repo)

    assert clone_cmds, "git clone was never invoked"
    assert custom_repo in clone_cmds[0]
    assert cli_mod.MANIFEST_REPO not in clone_cmds[0]

    ref_file = tmp_path / "deploy" / "manifests" / ".manifest-ref"
    assert f"repo: {custom_repo}" in ref_file.read_text()


def test_setup_manifests_defaults_to_upstream_repo(tmp_path, monkeypatch):
    """Without --manifest-repo, _setup_manifests clones the upstream default."""
    import shutil
    from unittest.mock import MagicMock, patch

    import conformance.cli as cli_mod

    monkeypatch.chdir(tmp_path)

    clone_dir = Path("/tmp/llm-d-manifests")
    clone_dir.mkdir(exist_ok=True)
    (clone_dir / "single-gpu.yaml").write_text("branch: main")

    clone_cmds = []

    def fake_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        result.stdout = "abc1234deadbeef\n"
        result.stderr = ""
        if cmd[:2] == ["git", "clone"]:
            clone_cmds.append(cmd)
        if cmd[0] == "rm":
            shutil.rmtree(str(clone_dir), ignore_errors=True)
        return result

    with patch.object(cli_mod, "subprocess") as mock_sub:
        mock_sub.run.side_effect = fake_run
        cli_mod._setup_manifests("main")

    assert clone_cmds, "git clone was never invoked"
    assert cli_mod.MANIFEST_REPO in clone_cmds[0]


def test_require_manifest_skips_when_missing(tmp_path):
    """test_01_prereq and test_02_deploy skip when the manifest file is absent."""
    from dataclasses import dataclass

    @dataclass
    class FakeDeployConfig:
        manifest_path: str = "nonexistent.yaml"

    @dataclass
    class FakeTestCase:
        deployment: FakeDeployConfig = None

        def __post_init__(self):
            self.deployment = FakeDeployConfig()

    sys.path.insert(0, str(Path(__file__).parent))
    import test_conformance as tc_mod

    original = tc_mod._MANIFEST_DIR
    try:
        tc_mod._MANIFEST_DIR = tmp_path
        with pytest.raises(pytest.skip.Exception, match="nonexistent.yaml"):
            tc_mod._require_manifest(FakeTestCase())
    finally:
        tc_mod._MANIFEST_DIR = original


def test_require_manifest_does_not_skip_when_present(tmp_path):
    """_require_manifest should not skip when the manifest exists."""
    from dataclasses import dataclass

    @dataclass
    class FakeDeployConfig:
        manifest_path: str = "exists.yaml"

    @dataclass
    class FakeTestCase:
        deployment: FakeDeployConfig = None

        def __post_init__(self):
            self.deployment = FakeDeployConfig()

    (tmp_path / "exists.yaml").write_text("kind: LLMInferenceService")

    sys.path.insert(0, str(Path(__file__).parent))
    import test_conformance as tc_mod

    original = tc_mod._MANIFEST_DIR
    try:
        tc_mod._MANIFEST_DIR = tmp_path
        tc_mod._require_manifest(FakeTestCase())
    finally:
        tc_mod._MANIFEST_DIR = original


def test_new_testcase_script_generates_loadable_config(tmp_path, monkeypatch):
    """new-testcase.sh must produce a config YAML that load_testcase() can parse."""
    import subprocess

    import yaml

    script = Path(__file__).parent.parent / "scripts" / "new-testcase.sh"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs" / "testcases").mkdir(parents=True)
    (tmp_path / "deploy" / "manifests").mkdir(parents=True)

    result = subprocess.run([str(script), "my-gen-test"], capture_output=True, text=True)
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    config_path = tmp_path / "configs" / "testcases" / "my-gen-test.yaml"
    assert config_path.exists()

    tc = load_testcase(str(config_path))
    assert tc.name == "my-gen-test"
    assert tc.deployment.manifest_path == "my-gen-test.yaml"
    assert tc.validation.health_port == 8000
    assert tc.validation.test_prompts == ["What is 2+2?"]
    assert tc.validation.metrics_check.check_vllm is True
    assert tc.validation.metrics_check.check_scheduler is True
    assert tc.model.name == "Qwen/Qwen3-0.6B"

    manifest_path = tmp_path / "deploy" / "manifests" / "my-gen-test.yaml"
    assert manifest_path.exists()
    manifest = yaml.safe_load(manifest_path.read_text())
    assert manifest["kind"] == "LLMInferenceService"
    assert manifest["metadata"]["name"] == "my-gen-test"
    assert manifest["spec"]["replicas"] == 1


def test_new_testcase_script_rejects_duplicate(tmp_path, monkeypatch):
    """new-testcase.sh must refuse to overwrite an existing config."""
    import subprocess

    script = Path(__file__).parent.parent / "scripts" / "new-testcase.sh"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs" / "testcases").mkdir(parents=True)
    (tmp_path / "deploy" / "manifests").mkdir(parents=True)

    subprocess.run([str(script), "dupe-test"], capture_output=True, text=True)
    result = subprocess.run([str(script), "dupe-test"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "already exists" in result.stdout or "already exists" in result.stderr
