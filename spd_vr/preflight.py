"""Fail-closed checks for the SPD-VR process graph.

The preflight command only inspects local state.  In particular, the ADB
probe reads ``adb reverse --list`` and never creates or tears down a device
connection.  It is useful before a live PICO session and as a machine-readable
acceptance record for CI or a lab notebook.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .model_compiler.artifacts import verify_artifacts
from .model_compiler.qualification import verify_contact_qualification_receipt


DEFAULT_ENDPOINT = "tcp/127.0.0.1:7447"
DEFAULT_SDK_LIBRARY = "/opt/apps/roboticsservice/SDK/x64/libPXREARobotSDK.so"
ARTIFACT_FILES = (
    "unified_plant.xml",
    "arm_ik.xml",
    "model_manifest.yaml",
    "collision_manifest.yaml",
    "actuator_calibration.yaml",
)
DEPENDENCIES = ("mujoco", "osqp", "coacd", "zenoh", "rtree")


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One preflight result suitable for JSON serialization."""

    name: str
    ok: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def _run_command(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False, **kwargs)


def _endpoint_host_port(endpoint: str) -> tuple[str, int]:
    value = str(endpoint)
    if value.startswith("tcp/"):
        value = value[4:]
    host, separator, port = value.rpartition(":")
    if not separator or not host or not port.isdigit():
        raise ValueError(f"unsupported endpoint: {endpoint}")
    return host.strip("[]"), int(port)


def _adb_command(run_command: RunCommand, command: Sequence[str]) -> tuple[bool, str]:
    try:
        result = run_command(command)
    except OSError as exc:
        return False, f"adb unavailable: {exc}"
    if result.returncode != 0:
        return False, (result.stderr or result.stdout or "adb command failed").strip()
    return True, (result.stdout or "").strip()


def _detect_robotics_service_port(run_command: RunCommand) -> tuple[int | None, str]:
    ok, output = _adb_command(run_command, [os.environ.get("SS_BIN", "ss"), "-H", "-ltnp"])
    if not ok:
        return None, output or "cannot inspect host TCP listeners"
    ports: set[int] = set()
    for line in output.splitlines():
        if 'users:(("RoboticsService"' not in line:
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        address = fields[3]
        if address.startswith(("127.", "[::1]", "[::ffff:127.")):
            continue
        port = address.rsplit(":", 1)[-1].rstrip("]")
        if port.isdigit():
            ports.add(int(port))
    if len(ports) != 1:
        return None, "expected exactly one non-loopback RoboticsService listener"
    return ports.pop(), ""


def _check_adb(
    run_command: RunCommand,
    *,
    selected_serial: str | None = None,
    expected_reverse: str | None = None,
) -> list[CheckResult]:
    ok, devices = _adb_command(run_command, ["adb", "devices"])
    if not ok:
        failure = CheckResult("pico_device", False, devices)
        return [failure, CheckResult("adb_reverse", False, devices), CheckResult("robotics_service", False, devices)]
    online = [
        line.split()[0]
        for line in devices.splitlines()[1:]
        if len(line.split()) >= 2 and line.split()[1] == "device"
    ]
    serial = selected_serial if selected_serial in online else (
        online[0] if len(online) == 1 and selected_serial is None else None
    )
    device_result = CheckResult(
        "pico_device",
        bool(serial),
        f"online PICO: {serial}" if serial else "selected PICO is not online or no uniquely selected online PICO",
    )
    service_port, service_detail = _detect_robotics_service_port(run_command)
    reverse_target = expected_reverse or (f"tcp:{service_port} tcp:{service_port}" if service_port else "")
    reverse_command = ["adb", "-s", serial, "reverse", "--list"] if serial else ["adb", "reverse", "--list"]
    ok, reverse = _adb_command(run_command, reverse_command)
    reverse_lines = reverse.splitlines()
    reverse_fields = reverse_target.split()
    reverse_ok = ok and bool(serial) and len(reverse_fields) == 2 and any(
        line.split()[1:3] == reverse_fields
        for line in reverse_lines
        if len(line.split()) >= 3
    )
    reverse_detail = reverse if reverse else f"expected reverse entry missing: {reverse_target or 'dynamic RoboticsService port'}"
    return [
        device_result,
        CheckResult("adb_reverse", reverse_ok, reverse_detail),
        CheckResult(
            "robotics_service",
            service_port is not None,
            f"non-loopback RoboticsService listener: {service_port}" if service_port else service_detail,
        ),
    ]


def _check_sdk(path: Path, loader: Callable[[str], Any]) -> CheckResult:
    try:
        client = loader(str(path))
        close = getattr(client, "close", None)
        if close is not None:
            close()
    except Exception as exc:  # SDK errors vary by ctypes/platform loader.
        return CheckResult("sdk", False, f"cannot load {path}: {exc}")
    return CheckResult("sdk", True, f"loaded {path}")


def _check_dependencies(loader: Callable[[str], Any]) -> CheckResult:
    missing: list[str] = []
    for name in DEPENDENCIES:
        try:
            loader(name)
        except Exception as exc:
            missing.append(f"{name}: {exc}")
    if missing:
        return CheckResult("python_dependencies", False, "; ".join(missing))
    return CheckResult("python_dependencies", True, ", ".join(DEPENDENCIES))


def _check_display(environment: Mapping[str, str]) -> CheckResult:
    value = environment.get("DISPLAY") or environment.get("WAYLAND_DISPLAY")
    return CheckResult("display", bool(value), str(value) if value else "DISPLAY or WAYLAND_DISPLAY is not set")


def _port_is_free(endpoint: str) -> tuple[bool, str]:
    host, port = _endpoint_host_port(endpoint)
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            return False, f"{endpoint} is unavailable: {exc}"
    return True, f"{endpoint} is free"


def _check_port(endpoint: str, checker: Callable[[str], tuple[bool, str]] | None = None) -> CheckResult:
    try:
        ok, detail = checker(endpoint) if checker is not None else _port_is_free(endpoint)
    except Exception as exc:
        return CheckResult("port_7447", False, str(exc))
    return CheckResult("port_7447", bool(ok), str(detail))


def default_supervisor_pid_file(
    environment: Mapping[str, str] | None = None,
) -> Path:
    values = os.environ if environment is None else environment
    runtime_root = Path(values.get("XDG_RUNTIME_DIR", "/tmp"))
    return runtime_root / f"spd-vr-{os.getuid()}" / "supervisor.pid"


def _check_supervisor(pid_file: Path) -> tuple[bool, str]:
    if not pid_file.exists():
        return True, f"foreground supervisor is absent: {pid_file}"
    try:
        value = pid_file.read_text(encoding="utf-8").strip()
        pid = int(value)
    except (OSError, ValueError) as exc:
        return False, f"invalid supervisor pid file {pid_file}: {exc}"
    if pid <= 1:
        return False, f"invalid supervisor pid in {pid_file}: {pid}"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True, f"stale supervisor pid file (pid {pid} is absent): {pid_file}"
    except PermissionError:
        return False, f"foreground supervisor pid {pid} exists but is not inspectable"
    return False, f"foreground supervisor is already running: pid={pid}"


def _check_artifacts(
    manifest_path: Path,
    urdf_path: Path,
    checker: Callable[[str | Path, str | Path], Any],
) -> CheckResult:
    missing = [name for name in ARTIFACT_FILES if not (manifest_path.parent / name).is_file()]
    if missing and checker is verify_artifacts:
        return CheckResult("artifacts", False, "missing generated artifacts: " + ", ".join(missing))
    if not urdf_path.is_file() and checker is verify_artifacts:
        return CheckResult("artifacts", False, f"missing authoritative URDF: {urdf_path}")
    try:
        result = checker(manifest_path, urdf_path)
        if isinstance(result, tuple) and len(result) == 2:
            return CheckResult("artifacts", bool(result[0]), str(result[1]))
        if result is False:
            return CheckResult("artifacts", False, "artifact checker rejected generated files")
        return CheckResult("artifacts", True, f"verified {manifest_path.parent}")
    except Exception as exc:
        return CheckResult("artifacts", False, str(exc))


def run_checks(
    *,
    repo_root: str | Path | None = None,
    sdk_library: str | Path | None = None,
    manifest_path: str | Path | None = None,
    urdf_path: str | Path | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    supervisor_pid_file: str | Path | None = None,
    selected_serial: str | None = None,
    expected_reverse: str | None = None,
    fake_source_path: str | Path | None = None,
    require_contact: bool = False,
    run_command: RunCommand | None = None,
    sdk_loader: Callable[[str], Any] | None = None,
    dependency_loader: Callable[[str], Any] | None = None,
    display_env: Mapping[str, str] | None = None,
    port_checker: Callable[[str], tuple[bool, str]] | None = None,
    supervisor_checker: Callable[[Path], tuple[bool, str]] | None = None,
    artifact_checker: Callable[[str | Path, str | Path], Any] | None = None,
    qualification_checker: Callable[[str | Path, str | Path], Any] | None = None,
) -> list[CheckResult]:
    """Run read-only checks and return every result, including failures.

    ``fake_source_path`` is intended for protocol/CI smoke only.  It skips
    ADB and the vendor shared object but still checks Python dependencies,
    artifacts, the endpoint and the foreground supervisor.
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    sdk_default = os.environ.get(
        "PXREA_SDK_LIBRARY",
        f"{os.environ.get('PXREA_SDK_ROOT', '/opt/apps/roboticsservice/SDK')}/x64/libPXREARobotSDK.so",
    )
    sdk = Path(sdk_library) if sdk_library is not None else Path(sdk_default)
    manifest = Path(manifest_path) if manifest_path is not None else root / "generated" / "spd_vr" / "model_manifest.yaml"
    urdf = Path(urdf_path) if urdf_path is not None else root / "assets" / "tianji_wuji2" / "tianji_wuji2.urdf"
    command = _run_command if run_command is None else run_command
    load_sdk = sdk_loader
    if load_sdk is None:
        from .pxrea_sdk import PXREAClient

        load_sdk = lambda path: PXREAClient.load_library(path)
    load_dependency = importlib.import_module if dependency_loader is None else dependency_loader
    environment = os.environ if display_env is None else display_env
    pid_file = (
        default_supervisor_pid_file(environment)
        if supervisor_pid_file is None
        else Path(supervisor_pid_file)
    )
    check_supervisor = _check_supervisor if supervisor_checker is None else supervisor_checker
    check_artifact = verify_artifacts if artifact_checker is None else artifact_checker
    if fake_source_path is None:
        results = _check_adb(command, selected_serial=selected_serial, expected_reverse=expected_reverse)
        results.extend((_check_sdk(sdk, load_sdk), _check_dependencies(load_dependency), _check_display(environment)))
    else:
        fake_source = Path(fake_source_path)
        results = [
            CheckResult(
                "fake_source",
                fake_source.is_file(),
                f"missing fake source: {fake_source}" if not fake_source.is_file() else f"readable {fake_source}",
            ),
            _check_dependencies(load_dependency),
        ]
    if require_contact:
        checker = (
            verify_contact_qualification_receipt
            if qualification_checker is None
            else qualification_checker
        )
        try:
            result = checker(manifest.parent, urdf)
            positive = result is not False and result is not None
            detail = (
                f"cached contact qualification ({result.get('collision_pieces', '?')} pieces)"
                if positive and isinstance(result, Mapping)
                else "cached contact qualification"
                if positive
                else "qualification checker returned no positive result"
            )
            results.append(CheckResult("artifacts", positive, detail))
            results.append(CheckResult("contact_gate", positive, detail))
        except Exception as exc:
            results.append(CheckResult("artifacts", False, str(exc)))
            results.append(CheckResult("contact_gate", False, str(exc)))
    else:
        results.append(_check_artifacts(manifest, urdf, check_artifact))
    results.append(_check_port(endpoint, port_checker))
    try:
        supervisor_ok, supervisor_detail = check_supervisor(pid_file)
    except Exception as exc:
        supervisor_ok, supervisor_detail = False, str(exc)
    results.append(CheckResult("supervisor", bool(supervisor_ok), str(supervisor_detail)))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--sdk-library", type=Path, default=None)
    parser.add_argument("--fake-source", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--expected-reverse", default=None)
    parser.add_argument("--supervisor-pid-file", type=Path, default=None)
    parser.add_argument("--require-contact", action="store_true", help="also require the contact surface-quality gate")
    args = parser.parse_args(argv)
    results = run_checks(
        repo_root=args.repo_root,
        sdk_library=args.sdk_library,
        manifest_path=args.manifest,
        urdf_path=args.urdf,
        endpoint=args.endpoint,
        supervisor_pid_file=args.supervisor_pid_file,
        selected_serial=args.serial,
        fake_source_path=args.fake_source,
        expected_reverse=args.expected_reverse,
        require_contact=args.require_contact,
    )
    for result in results:
        print(json.dumps(result.as_dict(), sort_keys=True))
    return 0 if all(result.ok for result in results) else 1


__all__ = [
    "ARTIFACT_FILES",
    "CheckResult",
    "DEFAULT_ENDPOINT",
    "default_supervisor_pid_file",
    "main",
    "run_checks",
]

if __name__ == "__main__":
    raise SystemExit(main())
