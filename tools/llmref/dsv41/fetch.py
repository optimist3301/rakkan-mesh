#!/usr/bin/env python3
"""Download files of DeepSeek-V4.1-Flash at the pinned revision, verified.

files.json pins the revision and every file's size and hash: the git blob
SHA-1 for small files, the LFS SHA-256 for the weight shards. A file already
in the cache is verified again rather than trusted.

usage: fetch.py [--cache DIR] FILE...
  FILE  a path in the repository, e.g. tokenizer.json or
        model-00005-of-00048.safetensors (layer 2, 6.9 GiB)
Prints the local path of each file. The cache defaults to
$XDG_CACHE_HOME/rakkan-llmref/dsv41/<revision>.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def pins():
    with open(os.path.join(HERE, "files.json")) as f:
        return json.load(f)


def default_cache(revision):
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "rakkan-llmref", "dsv41", revision)


def digest(path, pin):
    """Whether the file at path matches its pin."""
    if os.path.getsize(path) != pin["size"]:
        return False
    if "sha256" in pin:
        h = hashlib.sha256()
    else:
        h = hashlib.sha1(b"blob %d\0" % pin["size"])
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest() == pin.get("sha256", pin.get("git_blob_sha1"))


def fetch(name, cache=None):
    p = pins()
    pin = p["files"].get(name)
    if pin is None:
        raise SystemExit(f"{name}: not in files.json")
    cache = cache or default_cache(p["revision"])
    path = os.path.join(cache, name)
    if os.path.exists(path):
        if digest(path, pin):
            return path
        os.remove(path)
    os.makedirs(cache, exist_ok=True)
    url = f"https://huggingface.co/{p['repo']}/resolve/{p['revision']}/{name}"
    tmp = path + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 22)
    if not digest(tmp, pin):
        os.remove(tmp)
        raise SystemExit(f"{name}: downloaded file does not match its pinned hash")
    os.replace(tmp, path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    for name in a.files:
        print(fetch(name, a.cache))


if __name__ == "__main__":
    sys.exit(main())
