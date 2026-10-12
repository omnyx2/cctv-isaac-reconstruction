"""Read files out of a zip on the web without downloading the whole zip.

The web server must support HTTP Range requests (Hugging Face does). zipfile only
needs a file object that can seek and read; this one fetches each read with a Range request.

    python3 remote_zip.py URL --list [--grep TEXT]
    python3 remote_zip.py URL --extract PREFIX [PREFIX ...] --out DIR
"""
import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path


class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos = url, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.url = r.geturl()  # follow the redirect once (Hugging Face -> CDN)
            self.size = int(r.headers["Content-Length"])

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
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--list", action="store_true")
    p.add_argument("--grep", default="")
    p.add_argument("--extract", nargs="+", default=None, help="extract entries whose name starts with one of these")
    p.add_argument("--out", default=".")
    a = p.parse_args()
    z = zipfile.ZipFile(io.BufferedReader(HttpFile(a.url), buffer_size=1 << 20))
    names = z.namelist()
    if a.list:
        for n in names:
            if a.grep in n:
                print(n)
        return
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
