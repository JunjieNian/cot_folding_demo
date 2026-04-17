#!/usr/bin/env python3
"""
Deploy COT Folding Map to Alibaba Cloud OSS + CDN.

Prerequisites:
  pip install oss2

Usage:
  # 1. First time: configure credentials
  python deploy_oss.py --configure

  # 2. Pre-gzip frontend + data text assets
  python deploy_oss.py --gzip

  # 3. Upload everything
  python deploy_oss.py --upload

  # 4. Or do it all in one go
  python deploy_oss.py --gzip --upload

  # 5. Upload only frontend (after code changes, no data re-upload)
  python deploy_oss.py --upload --frontend-only
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import mimetypes
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DIST_DIR = PROJECT_DIR / "dist"
DATA_DIR = PROJECT_DIR / "public" / "data"
GZIP_DIR = PROJECT_DIR / ".gzip_cache"
CONFIG_FILE = PROJECT_DIR / ".oss_config.json"

# ─── MIME types ───
MIME_MAP = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
    ".b64": "text/plain; charset=utf-8",
}

# ─── Cache policies ───
# Order matters: more specific prefixes must come first.
CACHE_RULES = {
    "index.html": "no-cache",
    "assets/": "public, max-age=31536000, immutable",  # 1 year — hashed filenames
}

# Data files: index/meta/config JSONs revalidate; bundles/text immutable.
DATA_MUTABLE_SUFFIXES = {
    "problems.index.json", "app.json", "checkpoints.json",
    "trajectory.json", "problems.meta.json", "overview.json",
    "semantic_validation.json", "semantic_validation_binned.png",
}


def _is_mutable_data(key: str) -> bool:
    """Check if a data file should use short cache (revalidate on each visit)."""
    basename = key.rsplit("/", 1)[-1] if "/" in key else key
    return basename in DATA_MUTABLE_SUFFIXES

GZIP_SUFFIXES = {".html", ".js", ".css", ".json", ".svg", ".txt", ".b64"}


def get_cache_control(key: str) -> str:
    for prefix, cc in CACHE_RULES.items():
        if key.startswith(prefix) or key == prefix:
            return cc
    # Data files: mutable index/meta files revalidate; bundles immutable
    if key.startswith("data/"):
        if _is_mutable_data(key):
            return "no-cache"  # always revalidate with server
        return "public, max-age=2592000, immutable"  # 30 days for bundles/text
    return "public, max-age=86400"  # 1 day default


def get_content_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return MIME_MAP.get(ext, "application/octet-stream")


def should_gzip(path: Path) -> bool:
    return path.suffix.lower() in GZIP_SUFFIXES


# ─── Configuration ───

def configure():
    print("=== OSS Configuration ===")
    print("You'll need: AccessKey ID, AccessKey Secret, Endpoint, Bucket Name")
    print("Get credentials from: https://ram.console.aliyun.com/manage/ak")
    print("Endpoint list: https://help.aliyun.com/document_detail/31837.html")
    print()

    config = {}
    if CONFIG_FILE.exists():
        config = json.loads(CONFIG_FILE.read_text())
        print(f"Existing config found. Press Enter to keep current values.")
        print()

    config["access_key_id"] = input(
        f"  AccessKey ID [{config.get('access_key_id', '')[:6]}...]: "
    ).strip() or config.get("access_key_id", "")

    config["access_key_secret"] = input(
        f"  AccessKey Secret [{config.get('access_key_secret', '')[:4]}...]: "
    ).strip() or config.get("access_key_secret", "")

    config["endpoint"] = input(
        f"  Endpoint [{config.get('endpoint', 'oss-cn-shanghai.aliyuncs.com')}]: "
    ).strip() or config.get("endpoint", "oss-cn-shanghai.aliyuncs.com")

    config["bucket"] = input(
        f"  Bucket name [{config.get('bucket', 'cot-folding-demo')}]: "
    ).strip() or config.get("bucket", "cot-folding-demo")

    config["cdn_domain"] = input(
        f"  CDN domain (optional) [{config.get('cdn_domain', '')}]: "
    ).strip() or config.get("cdn_domain", "")

    CONFIG_FILE.write_text(json.dumps(config, indent=2))
    os.chmod(CONFIG_FILE, 0o600)  # only owner can read
    print(f"\n  Config saved to {CONFIG_FILE}")
    print(f"  (This file contains secrets — do NOT commit it to git)")
    return config


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        print("No config found. Run: python deploy_oss.py --configure")
        sys.exit(1)
    return json.loads(CONFIG_FILE.read_text())


# ─── Pre-gzip ───

def gzip_file(src: Path, dst: Path) -> int:
    """Gzip a file. Returns compressed size."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f_in:
        with gzip.open(dst, "wb", compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)
    return dst.stat().st_size


def iter_gzip_targets():
    # Frontend text assets from dist/
    for src in DIST_DIR.rglob("*"):
        if src.is_dir() or src.name == ".DS_Store":
            continue
        rel = src.relative_to(DIST_DIR)
        if str(rel).startswith("data") or not should_gzip(src):
            continue
        yield "frontend", src, rel

    # Data files from public/data -> uploaded under data/
    for src in DATA_DIR.rglob("*"):
        if src.is_dir() or not should_gzip(src):
            continue
        rel = src.relative_to(DATA_DIR.parent)
        yield "data", src, rel


def pre_gzip():
    """Pre-compress frontend/data text assets."""
    print(f"\n[Pre-gzip] Compressing frontend + data text assets...")
    print(f"  Frontend: {DIST_DIR}")
    print(f"  Data:     {DATA_DIR}")
    print(f"  Cache:  {GZIP_DIR}")

    targets = list(iter_gzip_targets())
    total = len(targets)
    frontend_total = sum(1 for kind, _, _ in targets if kind == "frontend")
    data_total = total - frontend_total
    print(f"  Files:  {total} ({frontend_total} frontend, {data_total} data)")

    GZIP_DIR.mkdir(parents=True, exist_ok=True)
    orig_total = 0
    gz_total = 0
    skipped = 0
    done = 0

    for _, src, rel in targets:
        dst = GZIP_DIR / rel

        # Skip if gzip cache is newer than source
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            skipped += 1
            gz_total += dst.stat().st_size
            orig_total += src.stat().st_size
        else:
            gz_size = gzip_file(src, dst)
            gz_total += gz_size
            orig_total += src.stat().st_size

        done += 1
        if done % 200 == 0 or done == total:
            print(f"  [{done}/{total}] ({skipped} cached)")

    ratio = gz_total / orig_total * 100 if orig_total > 0 else 0
    print(f"\n  Original: {orig_total / 1024 / 1024:.0f} MB")
    print(f"  Gzipped:  {gz_total / 1024 / 1024:.0f} MB ({ratio:.0f}%)")
    print(f"  Saved:    {(orig_total - gz_total) / 1024 / 1024:.0f} MB")


# ─── Upload ───

def get_oss_bucket(config: dict):
    try:
        import oss2
    except ImportError:
        print("oss2 SDK not installed. Run: pip install oss2")
        sys.exit(1)

    auth = oss2.Auth(config["access_key_id"], config["access_key_secret"])
    endpoint = config["endpoint"]
    if not endpoint.startswith("http"):
        endpoint = f"https://{endpoint}"
    return oss2.Bucket(auth, endpoint, config["bucket"])


def upload_file(bucket, local_path: Path, oss_key: str, content_type: str,
                cache_control: str, content_encoding: str | None = None):
    """Upload a single file to OSS."""
    headers = {
        "Content-Type": content_type,
        "Cache-Control": cache_control,
        "x-oss-storage-class": "Standard",
    }
    if content_encoding:
        headers["Content-Encoding"] = content_encoding

    bucket.put_object_from_file(oss_key, str(local_path), headers=headers)
    return oss_key


def upload(config: dict, frontend_only: bool = False, workers: int = 8):
    """Upload files to OSS."""
    bucket = get_oss_bucket(config)
    bucket_name = config["bucket"]

    tasks = []  # (local_path, oss_key, content_type, cache_control, content_encoding)

    # ── Frontend assets (from dist/) ──
    print(f"\n[Upload] Scanning frontend assets...")
    for f in DIST_DIR.rglob("*"):
        if f.is_dir() or f.name == ".DS_Store":
            continue
        rel = f.relative_to(DIST_DIR)
        # Skip the data symlink
        if str(rel).startswith("data"):
            continue
        oss_key = str(rel)
        ct = get_content_type(str(f))
        cc = get_cache_control(oss_key)
        gz_path = GZIP_DIR / rel
        if should_gzip(f) and gz_path.exists() and gz_path.stat().st_mtime >= f.stat().st_mtime:
            tasks.append((gz_path, oss_key, ct, cc, "gzip"))
        else:
            tasks.append((f, oss_key, ct, cc, None))

    print(f"  Frontend files: {len(tasks)}")

    # ── Data files (pre-gzipped if available) ──
    if not frontend_only:
        print(f"[Upload] Scanning data files...")
        data_count = 0
        data_files = [f for f in DATA_DIR.rglob("*") if f.is_file() and f.name != ".DS_Store"]
        for f in data_files:
            rel = f.relative_to(DATA_DIR.parent)  # data/aime24/...
            oss_key = str(rel)
            ct = get_content_type(str(f))
            cc = get_cache_control(oss_key)

            # Use pre-gzipped version if available (text files only)
            gz_path = GZIP_DIR / rel
            if should_gzip(f) and gz_path.exists():
                tasks.append((gz_path, oss_key, ct, cc, "gzip"))
            else:
                tasks.append((f, oss_key, ct, cc, None))
            data_count += 1

        print(f"  Data files: {data_count}")
        gzipped = sum(1 for t in tasks if t[4] == "gzip")
        print(f"  Pre-gzipped: {gzipped}")

    total = len(tasks)
    print(f"\n[Upload] Uploading {total} files to oss://{bucket_name}/ ...")

    uploaded = 0
    failed = 0
    bytes_sent = 0
    t0 = time.time()

    def do_upload(task):
        local_path, oss_key, ct, cc, ce = task
        upload_file(bucket, local_path, oss_key, ct, cc, ce)
        return local_path.stat().st_size

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(do_upload, t): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                size = future.result()
                uploaded += 1
                bytes_sent += size
            except Exception as e:
                failed += 1
                print(f"  [FAIL] {task[1]}: {e}")

            if uploaded % 100 == 0 or uploaded + failed == total:
                elapsed = time.time() - t0
                speed = bytes_sent / 1024 / 1024 / elapsed if elapsed > 0 else 0
                print(f"  [{uploaded + failed}/{total}] {bytes_sent / 1024 / 1024:.0f} MB sent ({speed:.1f} MB/s)")

    elapsed = time.time() - t0
    print(f"\n{'=' * 50}")
    print(f"  Upload complete in {elapsed:.0f}s")
    print(f"  Uploaded: {uploaded}, Failed: {failed}")
    print(f"  Total sent: {bytes_sent / 1024 / 1024:.0f} MB")

    # ── Print access URL ──
    endpoint = config["endpoint"].replace("https://", "").replace("http://", "")
    cdn = config.get("cdn_domain", "")
    print(f"\n  OSS URL: https://{bucket_name}.{endpoint}/index.html")
    if cdn:
        print(f"  CDN URL: https://{cdn}/")
    print(f"{'=' * 50}")


# ─── Setup bucket for static website hosting ───

def setup_bucket(config: dict):
    """Configure OSS bucket for static website hosting."""
    try:
        import oss2
    except ImportError:
        print("oss2 SDK not installed. Run: pip install oss2")
        sys.exit(1)

    bucket = get_oss_bucket(config)

    # Enable static website hosting
    from oss2.models import BucketWebsite
    bucket.put_bucket_website(BucketWebsite("index.html", "index.html"))
    print("  Static website hosting enabled (index: index.html)")

    # Set CORS (needed for font files and API calls from custom domains)
    from oss2.models import BucketCors, CorsRule
    rule = CorsRule(
        allowed_origins=["*"],
        allowed_methods=["GET", "HEAD"],
        allowed_headers=["*"],
        max_age_seconds=86400,
    )
    bucket.put_bucket_cors(BucketCors([rule]))
    print("  CORS configured (GET/HEAD from all origins)")

    print("  Bucket setup complete.")


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="Deploy to Alibaba Cloud OSS + CDN")
    parser.add_argument("--configure", action="store_true", help="Configure OSS credentials")
    parser.add_argument("--gzip", action="store_true", help="Pre-gzip frontend + data text assets")
    parser.add_argument("--upload", action="store_true", help="Upload to OSS")
    parser.add_argument("--setup", action="store_true", help="Setup bucket (static hosting, CORS)")
    parser.add_argument("--frontend-only", action="store_true", help="Only upload frontend, skip data")
    parser.add_argument("--workers", type=int, default=8, help="Upload parallelism (default: 8)")
    args = parser.parse_args()

    if not any([args.configure, args.gzip, args.upload, args.setup]):
        parser.print_help()
        print("\nTypical workflow:")
        print("  1. python deploy_oss.py --configure")
        print("  2. python deploy_oss.py --gzip --upload --setup")
        print("  3. (after code change) python deploy_oss.py --upload --frontend-only")
        return

    if args.configure:
        configure()

    if args.gzip:
        pre_gzip()

    if args.setup or args.upload:
        config = load_config()

    if args.setup:
        print("\n[Setup] Configuring bucket...")
        setup_bucket(config)

    if args.upload:
        upload(config, frontend_only=args.frontend_only, workers=args.workers)


if __name__ == "__main__":
    main()
