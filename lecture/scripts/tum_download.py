"""Download parts of TUM Traffic Accid3nD (real highway accidents, CC BY-NC-SA 4.0) without its 47 GB of zips.

The dataset is five ~9 GB zips on a public Nextcloud share (accident-dataset.github.io). remote_zip.py
reads each zip's table of contents and fetches only the photos asked for, with HTTP Range requests.
A camera can be split over two zips, so every zip is searched.

    python3 tum_download.py --project DIR --datasets DIR --list
    python3 tum_download.py ... --scenes [NAME ...]        accident scenes, -20 s .. +15 s around the contact
    python3 tum_download.py ... --configs                  every camera a config/*.yaml replays from data/
    python3 tum_download.py ... --camera r09_s10 north50   one whole camera (~3000 photos, ~1 GB)
    --labels   also the 3D labels (OpenLABEL JSON) of those photos

Where the photos go, as the project's configs expect them (${DATASETS} = --datasets):
    scenes:  ${DATASETS}/TUMTraf-Accid3nD/accident_scenes/<seq>_<cam>/images/*.jpg   (config/tum_*.yaml)
    cameras: ${DATASETS}/TUMTraf-Accid3nD/data/<seq>/images/<camera>/*.jpg          (config/auto_*.yaml)
The scene window and folder names are the ones tools/export_accident_scenes.py uses.
"""
import argparse
import base64
import re
import struct
import sys
import threading
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_zip import open_zip  # noqa: E402

SHARE = "https://nx21496.your-storageshare.de/public.php/webdav/accid3nd_dataset"
SHARE_TOKEN = "Zd4rwsMwwXMaXiK"          # public share: the token is the user name, no password
ZIPS = [f"train{i}.zip" for i in range(1, 6)]
CAMERAS = {"north16": "s040_camera_basler_north_16mm", "north50": "s040_camera_basler_north_50mm",
           "south16": "s050_camera_basler_south_16mm", "south50": "s050_camera_basler_south_50mm"}
SHORT = {v: k for k, v in CAMERAS.items()}
CLIP_BEFORE, CLIP_AFTER = 20.0, 15.0     # s around the contact, as tools/export_accident_scenes.py
WORKERS = 32                             # the share answers each request slowly; parallel requests hide that


def accidents(project: Path) -> dict:
    """{(sequence, camera): contact unix time} from config/tum_accid3nd_accidents.yaml (no PyYAML needed)."""
    out, seq, time = {}, None, None
    for line in (project / "config" / "tum_accid3nd_accidents.yaml").read_text(encoding="utf-8").splitlines():
        if m := re.match(r"^(r09_s\d+):", line):
            seq = m.group(1)
        elif m := re.match(r"^\s+- time:\s*([\d.]+)", line):
            time = float(m.group(1))
        elif (m := re.match(r"^\s+(s0\d0_camera_basler_\w+):", line)) and seq and time:
            out[(seq, m.group(1))] = time
    return out


def scenes(project: Path) -> dict:
    """{scene name: (sequence, camera, contact time)} for the accident scenes a config replays."""
    contact = accidents(project)
    found = {}
    for cfg in sorted((project / "config").glob("*.yaml")):
        for name in re.findall(r"accident_scenes/(r09_s\d+)_(\w+?)/images", cfg.read_text(encoding="utf-8")):
            seq, short = name
            cam = CAMERAS.get(short)
            if cam and (seq, cam) in contact:
                found[f"{seq}_{short}"] = (seq, cam, contact[(seq, cam)])
    return found


def config_cameras(project: Path) -> set:
    """{(sequence, camera)} that some config replays from the full data/ folders."""
    cams = set()
    for cfg in (project / "config").glob("*.yaml"):
        cams |= set(re.findall(r"TUMTraf-Accid3nD/data/(r09_s\d+)/images/(s0\d0_camera_basler_\w+?)['/\s,}]",
                               cfg.read_text(encoding="utf-8")))
    return cams


def stamp(name: str) -> float:
    """Unix time of a photo from its file name: 1617874292_786000000_s050_... -> 1617874292.786"""
    sec, nsec = name.split("/")[-1].split("_")[:2]
    return int(sec) + int(nsec) / 1e9


class Remote:
    """The five zips' tables of contents; each photo is then fetched with one Range request."""

    def __init__(self):
        self.index = {}                        # entry name -> (zip url, ZipInfo)
        for z in ZIPS:
            url = f"{SHARE}/{z}"
            for info in open_zip(url, SHARE_TOKEN).infolist():
                if not info.filename.endswith("/"):
                    self.index[info.filename] = (url, info)
        self.auth = {"Authorization": "Basic " + base64.b64encode(f"{SHARE_TOKEN}:".encode()).decode()}

    def get(self, url, start, end) -> bytes:
        req = urllib.request.Request(url, headers={**self.auth, "Range": f"bytes={start}-{end}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return r.read()
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))

    def read(self, name) -> bytes:
        """One zip member: local header + data in one request (the header's extra field is short)."""
        url, info = self.index[name]
        start = info.header_offset
        blob = self.get(url, start, start + 30 + len(info.orig_filename.encode()) + 1024 + info.compress_size - 1)
        name_len, extra_len = struct.unpack("<HH", blob[26:30])
        begin = 30 + name_len + extra_len
        raw = blob[begin:begin + info.compress_size]
        if len(raw) < info.compress_size:      # unusually long extra field
            raw = self.get(url, start + begin, start + begin + info.compress_size - 1)
        if info.compress_type == zipfile.ZIP_DEFLATED:
            raw = zlib.decompressobj(-15).decompress(raw)
        elif info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{name}: unsupported compression {info.compress_type}")
        if zlib.crc32(raw) != info.CRC:
            raise ValueError(f"{name}: CRC mismatch")
        return raw

    def fetch(self, jobs):
        """jobs: [(entry name, target path)]. Skips photos already complete; prints progress."""
        todo = [(n, t) for n, t in jobs
                if not (t.exists() and t.stat().st_size == self.index[n][1].file_size)]
        print(f"  {len(jobs) - len(todo)} already here, {len(todo)} to download", flush=True)
        done = [0]
        lock = threading.Lock()

        def one(job):
            n, target = job
            data = self.read(n)
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_suffix(target.suffix + ".part")
            part.write_bytes(data)
            part.replace(target)
            with lock:
                done[0] += 1
                if done[0] % 100 == 0 or done[0] == len(todo):
                    print(f"  {done[0]}/{len(todo)}", flush=True)

        with ThreadPoolExecutor(WORKERS) as pool:
            list(pool.map(one, todo))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--project", required=True, type=Path)
    p.add_argument("--datasets", required=True, type=Path)
    p.add_argument("--list", action="store_true")
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--configs", action="store_true")
    p.add_argument("--camera", nargs=2, metavar=("SEQUENCE", "CAMERA"))
    p.add_argument("--labels", action="store_true")
    a = p.parse_args()
    root = a.datasets / "TUMTraf-Accid3nD"
    known = scenes(a.project)

    if a.list:
        print("Accident scenes (--scenes NAME):")
        for name, (seq, cam, t) in known.items():
            print(f"  {name:18} {cam}")
        print("Cameras used by config/auto_*.yaml (--configs):")
        for seq, cam in sorted(config_cameras(a.project)):
            print(f"  {seq} {SHORT.get(cam, cam)}")
        print("Any camera (--camera SEQUENCE CAMERA): r09_s01 .. r09_s12 (not r09_s05: a broken copy of s06), "
              "cameras " + ", ".join(CAMERAS))
        return

    print("Reading the tables of contents of the 5 zips ...", flush=True)
    remote = Remote()
    kinds = ("images", "labels") if a.labels else ("images",)

    def entries(seq, cam):
        prefixes = tuple(f"{seq}/{k}/{cam}/" for k in kinds)
        return sorted(n for n in remote.index if n.startswith(prefixes))

    if a.scenes is not None:
        names = a.scenes or list(known)
        for name in names:
            if name not in known:
                sys.exit(f"unknown scene {name}; choose from {', '.join(known)}")
            seq, cam, t = known[name]
            picked = [n for n in entries(seq, cam) if t - CLIP_BEFORE <= stamp(n) <= t + CLIP_AFTER]
            print(f"Scene {name}: {sum('/images/' in n for n in picked)} photos", flush=True)
            remote.fetch([(n, root / "accident_scenes" / name / n.split("/")[1] / n.split("/")[-1])
                          for n in picked])

    cams = set()
    if a.configs:
        cams |= config_cameras(a.project)
    if a.camera:
        seq, cam = a.camera
        cams.add((seq, CAMERAS.get(cam, cam)))
    for seq, cam in sorted(cams):
        picked = entries(seq, cam)
        if not picked:
            sys.exit(f"{seq} {cam}: not in the dataset")
        print(f"Camera {seq} {SHORT.get(cam, cam)}: {sum('/images/' in n for n in picked)} photos", flush=True)
        remote.fetch([(n, root / "data" / n) for n in picked])


if __name__ == "__main__":
    main()
