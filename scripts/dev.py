#!/usr/bin/env python3
"""Centralized dev launcher: starts every service and streams their logs into
one terminal, each line prefixed with a colored service tag.

Usage:
    python3 scripts/dev.py start            # start everything, stream logs (Ctrl+C stops)
    python3 scripts/dev.py start bot nat    # start only these services
    python3 scripts/dev.py stop             # kill whatever is listening on the known ports
    python3 scripts/dev.py status           # show which services are up

Normally invoked via the `reachy` / `reachy-start` / `reachy-stop` /
`reachy-restart` / `reachy-status` wrapper scripts in this same directory,
which just call this file - see scripts/README or the repo README for how to
put them on your PATH.

Pure stdlib on purpose - this lives outside the bot/nat/office-mcp uv venvs,
so it can't rely on anything from their pyproject.toml dependencies (eg
honcho). It only shells out to `uv run ...` / `lsof` / `kill`, same commands
used to start each service by hand.
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ANSI color codes, one per service, cycled if there are ever more services
# than colors.
COLORS = [
    "\033[36m",  # cyan
    "\033[32m",  # green
    "\033[35m",  # magenta
    "\033[33m",  # yellow
    "\033[34m",  # blue
]
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"

SERVICES = {
    "bot": {
        "cwd": "bot",
        "cmd": ["uv", "run", "--env-file", "../.env", "python", "main.py"],
        "ports": [7860, 7861],
    },
    "mcp": {
        "cwd": "office-mcp",
        "cmd": ["uv", "run", "python", "mcp_server.py"],
        "ports": [8002],
    },
    "webapp": {
        "cwd": "office-mcp",
        # --log-level warning: uvicorn's per-request access log (every
        # poll from the frontend) is pure noise in the combined console.
        "cmd": ["uv", "run", "uvicorn", "webapp:app", "--port", "8003", "--log-level", "warning"],
        "ports": [8003],
    },
    "nat": {
        "cwd": "nat",
        "cmd": [
            "uv", "run", "--env-file", "../.env", "nat", "serve",
            "--config_file", "src/ces_tutorial/config.yml", "--port", "8001",
        ],
        "ports": [8001],
    },
}

print_lock = threading.Lock()
procs: dict[str, subprocess.Popen] = {}
stopping = False


def free_ports(ports: list[int]) -> None:
    """Kill whatever is already listening on these ports (mirrors the manual
    `lsof -ti :PORT | xargs kill -9` restart dance we did by hand before)."""
    for port in ports:
        try:
            out = subprocess.run(
                ["lsof", "-ti", f":{port}"], capture_output=True, text=True
            ).stdout.strip()
        except FileNotFoundError:
            return  # lsof not available, nothing we can do
        for pid in out.splitlines():
            subprocess.run(["kill", "-9", pid])


def stream_output(name: str, color: str, proc: subprocess.Popen) -> None:
    tag = f"{color}{BOLD}[{name:>6}]{RESET}"

    for line in proc.stdout:  # blocks until a line is ready or the pipe closes
        with print_lock:
            print(f"{tag} {line.rstrip()}", flush=True)

    code = proc.wait()
    if not stopping:
        with print_lock:
            status = f"{RED}{BOLD}exited with code {code}{RESET}"
            print(f"{tag} {status}", flush=True)


def start_service(name: str, color: str) -> subprocess.Popen:
    spec = SERVICES[name]
    cwd = os.path.join(REPO_ROOT, spec["cwd"])
    free_ports(spec["ports"])

    with print_lock:
        print(
            f"{color}{BOLD}[{name:>6}]{RESET} starting: "
            f"{' '.join(spec['cmd'])}  (cwd={spec['cwd']}, ports={spec['ports']})",
            flush=True,
        )

    proc = subprocess.Popen(
        spec["cmd"],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    procs[name] = proc
    threading.Thread(target=stream_output, args=(name, color, proc), daemon=True).start()
    return proc


def is_up(port: int) -> bool:
    out = subprocess.run(
        ["lsof", "-ti", f":{port}"], capture_output=True, text=True
    ).stdout.strip()
    return bool(out)


def cmd_status(selected: list[str]) -> None:
    green = "\033[32m"
    yellow = "\033[33m"
    for name in selected:
        ports = SERVICES[name]["ports"]
        up = [is_up(p) for p in ports]
        if all(up):
            state = f"{green}up{RESET}"
        elif any(up):
            state = f"{yellow}partially up{RESET}"
        else:
            state = f"{RED}down{RESET}"
        port_str = ", ".join(str(p) for p in ports)
        print(f"{BOLD}{name:>6}{RESET}  {state:<24} (ports: {port_str})")


def cmd_stop(selected: list[str]) -> None:
    for name in selected:
        ports = SERVICES[name]["ports"]
        print(f"{BOLD}[{name:>6}]{RESET} stopping (ports: {', '.join(str(p) for p in ports)})")
        free_ports(ports)


def shutdown(*_args) -> None:
    global stopping
    if stopping:
        return
    stopping = True
    with print_lock:
        print(f"\n{BOLD}Stopping all services...{RESET}", flush=True)
    for name, proc in procs.items():
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 10
    for name, proc in procs.items():
        remaining = max(0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=["start", "stop", "status"],
        nargs="?",
        default="start",
        help="start (default): launch and stream logs. stop: kill whatever's "
        "on the known ports. status: show which services are up.",
    )
    parser.add_argument(
        "services",
        nargs="*",
        default=[],
        help="Subset of services (default: all). Choices: " + ", ".join(SERVICES),
    )
    args = parser.parse_args()
    selected = args.services or list(SERVICES)
    unknown = [s for s in selected if s not in SERVICES]
    if unknown:
        parser.error(f"unknown service(s): {', '.join(unknown)} (choices: {', '.join(SERVICES)})")

    if args.command == "status":
        cmd_status(selected)
        return

    if args.command == "stop":
        cmd_stop(selected)
        return

    # command == "start"
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    with print_lock:
        print(f"{BOLD}Starting services: {', '.join(selected)}{RESET}\n")

    for i, name in enumerate(selected):
        start_service(name, COLORS[i % len(COLORS)])
        time.sleep(0.3)  # stagger startup slightly for readable logs

    # Keep the main thread alive while the reader threads do the work.
    while True:
        time.sleep(1)
        if all(p.poll() is not None for p in procs.values()):
            break


if __name__ == "__main__":
    main()
