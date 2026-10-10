"""Deploy the project from this Windows PC to a DGX Spark in one command (over SSH).

    1  pack (tools/pack_for_spark.py) into a local folder
    2  copy it to the Spark (scp; Windows has the OpenSSH client)
    3  run tools/setup_spark.sh there (interactive: sudo password, Isaac Sim EULA)

Usage:
    C:\\isaacsim\\kit\\python\\python.exe tools/deploy_to_spark.py --host user@spark-ip
        [--port 22] [--tum-cameras] [--ollama] [--code-only] [--out D:/cctv_transfer]

--code-only sends only the git bundle and re-runs the setup (later updates: seconds
instead of copying the datasets again). SSH asks for the password at each step unless a
key is set up (ssh-keygen; then: type %USERPROFILE%\\.ssh\\id_ed25519.pub | ssh user@spark-ip
"cat >> ~/.ssh/authorized_keys").
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REMOTE_DIR = "cctv_transfer"          # in the remote home


def run(cmd, **kw):
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    parser = argparse.ArgumentParser(description="Pack, copy and set up the project on a DGX Spark over SSH")
    parser.add_argument("--host", required=True, help="user@host of the Spark")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--out", default="D:/cctv_transfer", help="local staging folder")
    parser.add_argument("--tum-cameras", action="store_true", help="also the TUM camera folders (~30 GB)")
    parser.add_argument("--ollama", action="store_true", help="install Ollama + qwen2.5vl:7b on the Spark")
    parser.add_argument("--code-only", action="store_true", help="only the code (git bundle), then setup")
    parser.add_argument("--no-isaac-build", action="store_true", help="Isaac Sim is installed already")
    args = parser.parse_args()
    for tool in ("ssh", "scp", "git"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found (Windows: Settings > Optional features > OpenSSH Client)")

    out = Path(args.out)
    ssh = ["ssh", "-p", args.port]
    scp = ["scp", "-P", args.port]
    if args.code_only:
        out.mkdir(parents=True, exist_ok=True)
        run(["git", "-C", ROOT, "bundle", "create", out / "CCTV_Project.bundle", "--all"])
        shutil.copy2(ROOT / "tools" / "setup_spark.sh", out / "setup_spark.sh")
        run(ssh + [args.host, f"mkdir -p ~/{REMOTE_DIR}"])
        run(scp + [out / "CCTV_Project.bundle", out / "setup_spark.sh", f"{args.host}:{REMOTE_DIR}/"])
    else:
        pack = [sys.executable, ROOT / "tools" / "pack_for_spark.py", "--out", out]
        if args.tum_cameras:
            pack.append("--tum-cameras")
        run(pack)
        run(ssh + [args.host, f"mkdir -p ~/{REMOTE_DIR}"])
        # the folder's contents into ~/cctv_transfer (scp -r of the folder itself would nest it)
        run(scp + ["-r"] + sorted(out.iterdir()) + [f"{args.host}:{REMOTE_DIR}/"])

    setup = f"bash ~/{REMOTE_DIR}/setup_spark.sh"
    if args.ollama:
        setup += " --ollama"
    if args.no_isaac_build:
        setup += " --no-isaac-build"
    run(ssh + ["-t", args.host, setup])       # -t: sudo and the Isaac Sim EULA ask interactively
    print(f"\nDone. On the Spark: cd ~/CCTV_Project && ./run_isaac.sh")


if __name__ == "__main__":
    main()
