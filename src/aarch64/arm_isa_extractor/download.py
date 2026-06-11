from __future__ import annotations
import shutil
import tarfile
import urllib.request
from pathlib import Path
from .models import ExtractionError

_RELEASES = {
    "A64-2025-09": (
        "https://developer.arm.com/-/cdn-downloads/permalink/Exploration-Tools-A64-ISA/"
        "ISA_A64/ISA_A64_xml_A_profile-2025-09_ASL1.tar.gz",
        "ISA_A64_xml_A_profile-2025-09_ASL1",
    ),
}


def download_xml(release: str = "latest", cache_dir=None, force: bool = False) -> Path:
    """Stage 1: fetch + cache the ARM A64 ISA XML; return the directory of extracted XML files."""
    if release == "latest":
        release = max(_RELEASES)
    if release not in _RELEASES:
        raise ExtractionError(f"unknown release {release!r}; known: {sorted(_RELEASES)}")
    url, stem = _RELEASES[release]
    cache = Path(cache_dir or Path.home() / ".cache" / "arm_isa_xml")
    xml_dir = cache / stem
    if xml_dir.exists() and not force:
        return xml_dir

    cache.mkdir(parents=True, exist_ok=True)
    tarball = cache / f"{stem}.tar.gz"
    if not tarball.exists() or force:
        tmp = tarball.with_suffix(".tmp")
        try:
            with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as fh:
                shutil.copyfileobj(resp, fh)
            tmp.replace(tarball)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    xml_dir.mkdir(parents=True)
    with tarfile.open(tarball, "r:gz") as tf:
        for member in tf.getmembers():
            p = Path(member.name)
            if member.isfile() and p.suffix == ".xml":   # flatten the tree; regular files only
                member.name = p.name
                tf.extract(member, xml_dir)
    return xml_dir
