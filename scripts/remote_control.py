"""Deployment helper. Reads SSH credentials locally and reports only explicitly safe output."""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache/ssh-tools"))
import paramiko  # noqa: E402


def cpu_connection():
    # The supplied CPU comment is data: label, IPv4, user, password. Never execute it.
    raw = (ROOT / "prompt/.env").read_text(encoding="utf-8-sig")
    for line in raw.splitlines():
        if line.lstrip().startswith("#") and "cpu" in line.lower():
            parts = line.split()
            if len(parts) == 4 and re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", parts[1]):
                return parts[1], 22, parts[2], parts[3]
    raise ValueError("CPU_CONNECTION_FORMAT")


def connect():
    host, port, user, password = cpu_connection()
    client = paramiko.SSHClient()
    known_hosts = ROOT / ".cache/remote/known_hosts"
    known_hosts.parent.mkdir(parents=True, exist_ok=True)
    if known_hosts.exists():
        client.load_host_keys(str(known_hosts))
    # First connection records the supplied host's key; changed keys are rejected.
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host,
        port=port,
        username=user,
        password=password,
        look_for_keys=False,
        allow_agent=False,
        timeout=12,
        auth_timeout=12,
        banner_timeout=12,
    )
    client.save_host_keys(str(known_hosts))
    return client


def run_python(client, source, timeout=45):
    stdin, stdout, stderr = client.exec_command("python3 -", timeout=timeout)
    stdin.write(source)
    stdin.flush()
    stdin.channel.shutdown_write()
    output = stdout.read().decode("utf-8", errors="replace")
    error = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if code:
        # Do not expose arbitrary remote error text, which may include credential input.
        return {
            "status": "remote_failed",
            "exit_code": code,
            "stderr_bytes": len(error),
            "stdout_bytes": len(output),
        }
    return json.loads(output)


INSPECT = r"""
import json, os, platform, shutil, socket, subprocess, urllib.request
def command(args):
    p=subprocess.run(args, capture_output=True, text=True, timeout=20)
    return p.returncode,p.stdout.strip()
docker=["docker"]
code, version=command(docker+["version","--format","{{.Server.Version}}"])
if code:
    docker=["sudo","-n","docker"]
    code, version=command(docker+["version","--format","{{.Server.Version}}"])
report={"system":platform.system(),"machine":platform.machine(),"python":platform.python_version(),
        "home":os.path.expanduser("~"),"disk_free_gib":round(shutil.disk_usage(os.path.expanduser("~")).free/2**30,1),
        "docker_version":version if not code else "unavailable","docker_uses_sudo":docker[0]=="sudo"}
if not code:
    code, rows=command(docker+["ps","--format","{{.Names}}|{{.Image}}|{{.Ports}}|{{.Status}}"])
    report["containers"]=[dict(zip(["name","image","ports","status"],line.split("|",3))) for line in rows.splitlines()]
    _,report["compose_version"]=command(docker+["compose","version","--short"])
report["ports"]={}
for port in [6006,6008,6009,8000,8001,27017,19530,9000,7474,7687]:
    try:
        with socket.create_connection(("127.0.0.1",port),timeout=1): pass
        report["ports"][str(port)]=True
    except OSError: report["ports"][str(port)]=False
report["gpu_schemas"]={}
for port in [6008,6009]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/openapi.json"%port,timeout=4) as response:
            schema=json.load(response)
        components=schema.get("components",{}).get("schemas",{})
        report["gpu_schemas"][str(port)]={"paths":schema.get("paths",{}),"components":components}
    except Exception as e: report["gpu_schemas"][str(port)]={"error_type":type(e).__name__}
print(json.dumps(report))
"""


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["inspect", "python"])
    parser.add_argument("--script")
    parser.add_argument("--timeout", type=int, default=45)
    args = parser.parse_args()
    client = None
    try:
        client = connect()
        source = (
            INSPECT
            if args.action == "inspect"
            else Path(args.script).read_text(encoding="utf-8")
        )
        print(
            json.dumps(
                run_python(client, source, args.timeout), ensure_ascii=False, indent=2
            )
        )
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        return 1
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
