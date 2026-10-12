"""Read files out of a zip on the web without downloading the whole zip.

The web server must support HTTP Range requests (Hugging Face and Nextcloud do). zipfile only
needs a file object that can seek and read; this one fetches each read with a Range request.

    python3 remote_zip.py URL --list [--grep TEXT]
    python3 remote_zip.py URL --dirs                      top-level folders in the zip
    python3 remote_zip.py URL --extract PREFIX [PREFIX ...] --out DIR
    --user NAME[:PASSWORD]   HTTP basic auth (a Nextcloud public share: the share token, no password)
"""
import argparse
import base64
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path


class HttpFile(io.RawIOBase):
    def __init__(self, url, user=None):
        self.url, self.pos = url, 0
        self.headers = {}
        if user:
            token = user if ":" in user else user + ":"
            self.headers["Authorization"] = "Basic " + base64.b64encode(token.encode()).decode()
        try:
            req = urllib.request.Request(url, method="HEAD", headers=self.headers)
            with urllib.request.urlopen(req) as r:
                self.url = r.geturl()  # follow the redirect once (Hugging Face -> CDN)
                self.size = int(r.headers["Content-Length"])
        except OSError:
            # Some servers (Nextcloud WebDAV) refuse HEAD: ask for one byte, read the total size
            req = urllib.request.Request(url, headers={**self.headers, "Range": "bytes=0-0"})
            with urllib.request.urlopen(req) as r:
                self.url = r.geturl()
                self.size = int(re.search(r"/(\d+)$", r.headers["Content-Range"]).group(1))

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={**self.headers, "Range": f"bytes={self.pos}-{end}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == 4:
                    raise
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def open_zip(url, user=None) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpFile(url, user), buffer_size=1 << 20))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--list", action="store_true")
    p.add_argument("--grep", default="")
    p.add_argument("--dirs", action="store_true", help="print the top-level folders")
    p.add_argument("--extract", nargs="+", default=None, help="extract entries whose name starts with one of these")
    p.add_argument("--out", default=".")
    p.add_argument("--user", default=None, help="HTTP basic auth NAME[:PASSWORD]")
    a = p.parse_args()
    z = open_zip(a.url, a.user)
    names = z.namelist()
    if a.list:
        for n in names:
            if a.grep in n:
                print(n)
        return
    if a.dirs:
        for d in sorted({n.split("/")[0] for n in names if "/" in n}):
            print(d)
        return
    if not a.extract:
        p.error("one of --list, --dirs, --extract is required")
    picked = [n for n in names if n.startswith(tuple(a.extract)) and not n.endswith("/")]
    if not picked:
        sys.exit(f"nothing in the zip starts with {a.extract!r}")
    out = Path(a.out)
    for i, n in enumerate(picked, 1):
        target = out / n
        if target.exists() and target.stat().st_size == z.getinfo(n).file_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(z.read(n))
        if i % 50 == 0 or i == len(picked):
            print(f"  {i}/{len(picked)}  {n}", flush=True)


if __name__ == "__main__":
    main()
