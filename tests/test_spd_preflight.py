import json

from spd_vr import preflight


def test_preflight_reports_structured_failure_without_mutating_adb(tmp_path):
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(list(command))
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    results = preflight.run_checks(
        repo_root=tmp_path,
        run_command=fake_run,
        sdk_loader=lambda path: object(),
        dependency_loader=lambda name: object(),
        display_env={"DISPLAY": ":99"},
        port_checker=lambda endpoint: (True, "free"),
        session_checker=lambda name: (True, "not running"),
        artifact_checker=lambda manifest, urdf: (False, "authoritative URDF hash mismatch"),
    )
    assert all(isinstance(item, preflight.CheckResult) for item in results)
    assert next(item for item in results if item.name == "artifacts").ok is False
    assert not any(command[1:2] == ["reverse"] and command[-1] != "--list" for command in calls)


def test_preflight_matches_dynamic_robotics_listener_and_reverse():
    def fake_run(command, **_kwargs):
        if command[:2] == ["adb", "devices"]:
            output = "List of devices attached\nPICO-1\tdevice\n"
        elif command[:2] == ["ss", "-H"]:
            output = 'LISTEN 0 128 0.0.0.0:15555 0.0.0.0:* users:(("RoboticsService",pid=42,fd=3))'
        else:
            output = "UsbFfs tcp:15555 tcp:15555\n"
        return type("Completed", (), {"returncode": 0, "stdout": output, "stderr": ""})()

    results = preflight._check_adb(fake_run, selected_serial="PICO-1")
    assert all(item.ok for item in results)


def test_preflight_contact_gate_is_explicit_and_fail_closed(tmp_path):
    results = preflight.run_checks(
        repo_root=tmp_path,
        fake_source_path=tmp_path / "input.jsonl",
        dependency_loader=lambda name: object(),
        port_checker=lambda endpoint: (True, "free"),
        session_checker=lambda name: (True, "not running"),
        artifact_checker=lambda manifest, urdf: (True, "verified"),
        contact_checker=lambda path, **kwargs: (_ for _ in ()).throw(ValueError("p95 exceeds gate")),
        require_contact=True,
    )
    gate = next(item for item in results if item.name == "contact_gate")
    assert gate.ok is False
    assert "p95" in gate.detail


def test_preflight_accepts_positional_contact_checker(tmp_path):
    results = preflight.run_checks(
        repo_root=tmp_path,
        fake_source_path=tmp_path / "input.jsonl",
        dependency_loader=lambda name: object(),
        port_checker=lambda endpoint: (True, "free"),
        session_checker=lambda name: (True, "not running"),
        artifact_checker=lambda manifest, urdf: (True, "verified"),
        contact_checker=lambda manifest, urdf: True,
        require_contact=True,
    )
    assert next(item for item in results if item.name == "contact_gate").ok


def test_preflight_rejects_contact_checker_without_positive_report(tmp_path):
    results = preflight.run_checks(
        repo_root=tmp_path,
        fake_source_path=tmp_path / "input.jsonl",
        dependency_loader=lambda name: object(),
        port_checker=lambda endpoint: (True, "free"),
        session_checker=lambda name: (True, "not running"),
        artifact_checker=lambda manifest, urdf: (True, "verified"),
        contact_checker=lambda manifest, **kwargs: None,
        require_contact=True,
    )
    gate = next(item for item in results if item.name == "contact_gate")
    assert gate.ok is False
    assert "positive" in gate.detail


def test_preflight_cli_emits_json_and_nonzero_on_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        preflight,
        "run_checks",
        lambda **kwargs: [preflight.CheckResult("artifacts", False, "missing model_manifest.yaml")],
    )
    assert preflight.main(["--repo-root", str(tmp_path)]) == 1
    record = json.loads(capsys.readouterr().out.strip())
    assert record == {"detail": "missing model_manifest.yaml", "name": "artifacts", "ok": False}
