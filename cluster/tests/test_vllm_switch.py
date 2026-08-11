# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SWITCH = ROOT / "cluster/vllm-switch"
RUNNER = ROOT / "cluster/vllm-profile-runner"


class VllmSwitchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.profiles = self.base / "profiles"
        self.profiles.mkdir()
        shutil.copy2(RUNNER, self.profiles / "vllm-profile-runner")
        (self.profiles / "vllm-profile-runner").chmod(0o755)
        self.systemctl = self.base / "systemctl"
        self.systemctl.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                case "$1" in
                  is-active) grep -qx active "$FAKE_SERVICE_STATE" ;;
                  start) echo active > "$FAKE_SERVICE_STATE" ;;
                  stop) echo inactive > "$FAKE_SERVICE_STATE" ;;
                  cat) cat "$FAKE_UNIT" ;;
                  *) exit 0 ;;
                esac
                """
            )
        )
        self.systemctl.chmod(0o755)
        self.unit = self.base / "unit"
        self.unit.write_text("ExecStart=/path/current-special-start\ncontainer-current\n")
        self.service_state = self.base / "service-state"
        self.service_state.write_text("active\n")
        self.state = self.base / "active"
        self.previous = self.base / "previous"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def env(self, **extra: str) -> dict[str, str]:
        result = os.environ.copy()
        result.update(
            {
                "VLLM_SWITCH_PROFILES_DIR": str(self.profiles),
                "VLLM_SWITCH_RUNNER": str(self.profiles / "vllm-profile-runner"),
                "VLLM_SWITCH_STATE_FILE": str(self.state),
                "VLLM_SWITCH_PREVIOUS_FILE": str(self.previous),
                "VLLM_SWITCH_SERVICE_FILE": str(self.base / "service"),
                "VLLM_SWITCH_SYSTEMCTL": str(self.systemctl),
                "VLLM_SWITCH_SUDO": "",
                "VLLM_SWITCH_OFFLINE": "1",
                "FAKE_UNIT": str(self.unit),
                "FAKE_SERVICE_STATE": str(self.service_state),
            }
        )
        result.update(extra)
        return result

    def run_switch(self, *args: str, check: bool = True, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(SWITCH), *args],
            env=self.env(**env),
            text=True,
            capture_output=True,
            check=check,
        )

    def write_scripted(self, name: str = "scripted") -> Path:
        profile = self.profiles / f"{name}.conf"
        profile.write_text(
            textwrap.dedent(
                """\
                # A specialized profile
                SERVICE_MODE=scripted
                MODEL_PATH=/model
                MODEL_NAME=model-canary
                DOCKER_IMAGE=image:tag
                START_COMMAND=true
                STOP_COMMAND=true
                TRACK_COMMAND='sleep 0.01'
                RESTART_POLICY=no
                STARTUP_TIMEOUT=30
                """
            )
        )
        return profile

    def test_list_does_not_source_profiles_or_depend_on_unit_grep(self) -> None:
        self.write_scripted()
        (self.profiles / "must-not-run.conf").write_text("# Listed safely\nexit 91\n")
        result = self.run_switch("list")
        self.assertIn("scripted", result.stdout)
        self.assertIn("must-not-run", result.stdout)

    def test_status_uses_recorded_profile(self) -> None:
        self.write_scripted("current")
        self.state.write_text("current\n")
        result = self.run_switch("status")
        self.assertIn("running", result.stdout)
        self.assertIn("profile=current", result.stdout)

    def test_rendered_unit_has_one_runner_start_and_no_flattened_json(self) -> None:
        self.write_scripted()
        result = self.run_switch("render", "scripted")
        self.assertIn(f"ExecStart={self.profiles}/vllm-profile-runner start scripted", result.stdout)
        self.assertIn("Restart=no", result.stdout)
        self.assertNotIn("START_COMMAND", result.stdout)
        self.assertEqual(result.stdout.count("ExecStart="), 1)

    def test_invalid_restart_policy_fails_validation(self) -> None:
        profile = self.write_scripted()
        profile.write_text(profile.read_text().replace("RESTART_POLICY=no", "RESTART_POLICY=maybe"))
        result = self.run_switch("validate", "scripted", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid RESTART_POLICY", result.stderr)

    def test_adopt_requires_active_unit_markers(self) -> None:
        profile = self.write_scripted("current")
        profile.write_text(profile.read_text() + "ADOPT_UNIT_CONTAINS=(current-special-start container-current)\n")
        self.run_switch("adopt", "current")
        self.assertEqual(self.state.read_text(), "current\n")

    def test_scripted_runner_records_and_clears_state(self) -> None:
        output = self.base / "actions"
        profile = self.write_scripted("runner")
        profile.write_text(
            profile.read_text()
            .replace("START_COMMAND=true", f"START_COMMAND=\"printf 'start\\n' >> '{output}'\"")
            .replace("STOP_COMMAND=true", f"STOP_COMMAND=\"printf 'stop\\n' >> '{output}'\"")
        )
        subprocess.run(
            [str(RUNNER), "start", "runner"],
            env=self.env(),
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertEqual(self.state.read_text(), "runner\n")
        subprocess.run(
            [str(RUNNER), "stop", "runner"],
            env=self.env(),
            check=True,
            text=True,
            capture_output=True,
        )
        self.assertFalse(self.state.exists())
        self.assertEqual(output.read_text().splitlines(), ["start", "stop"])

    def test_switch_installs_unit_and_final_prestart_reset(self) -> None:
        self.write_scripted("current")
        self.write_scripted("candidate")
        self.state.write_text("current\n")
        service_file = self.base / "vllm-cluster.service"
        dropin_file = self.base / "vllm-cluster.service.d/zz-vllm-switch.conf"
        self.run_switch(
            "switch",
            "candidate",
            VLLM_SWITCH_SERVICE_FILE=str(service_file),
            VLLM_SWITCH_SERVICE_DROPIN_FILE=str(dropin_file),
        )
        self.assertIn("vllm-profile-runner start candidate", service_file.read_text())
        self.assertEqual(dropin_file.read_text().count("ExecStartPre="), 1)
        self.assertEqual(self.state.read_text(), "candidate\n")
        self.assertEqual(self.previous.read_text(), "current\n")

    def test_failed_readiness_runs_stop_then_restores_rollback_unit(self) -> None:
        self.write_scripted("current")
        cleanup = self.base / "cleanup"
        candidate = self.write_scripted("candidate")
        candidate.write_text(
            candidate.read_text()
            .replace("STARTUP_TIMEOUT=30", "STARTUP_TIMEOUT=0")
            .replace("STOP_COMMAND=true", f"STOP_COMMAND=\"printf 'cleaned\\n' >> '{cleanup}'\"")
            + "READINESS_URL=http://127.0.0.1:9/v1/models\n"
            + "AUTO_ROLLBACK=1\n"
            + "ROLLBACK_PROFILE=current\n"
        )
        self.state.write_text("current\n")
        service_file = self.base / "vllm-cluster.service"
        dropin_file = self.base / "vllm-cluster.service.d/zz-vllm-switch.conf"
        result = self.run_switch(
            "switch",
            "candidate",
            check=False,
            VLLM_SWITCH_SERVICE_FILE=str(service_file),
            VLLM_SWITCH_SERVICE_DROPIN_FILE=str(dropin_file),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(cleanup.read_text(), "cleaned\n")
        self.assertIn("vllm-profile-runner start current", service_file.read_text())
        self.assertEqual(self.state.read_text(), "current\n")


if __name__ == "__main__":
    unittest.main()
