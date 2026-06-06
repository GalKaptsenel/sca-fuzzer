"""
Downloader for ARM ISA XML specification archives.

Downloads, caches, and extracts ARM ISA XML files from ARM's CDN.
Re-downloads only when forced or the cache is absent.
"""
from __future__ import annotations

import logging
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry of known releases.
# Key  : human-readable release identifier (used as CLI/API argument)
# Value: (download_url, archive_stem)  where archive_stem is the top-level
#        directory name inside the tarball.
# ---------------------------------------------------------------------------
KNOWN_RELEASES: dict[str, tuple[str, str]] = {
    "A64-2025-09": (
        "https://developer.arm.com/-/cdn-downloads/permalink/"
        "Exploration-Tools-A64-ISA/ISA_A64/"
        "ISA_A64_xml_A_profile-2025-09_ASL1.tar.gz",
        "ISA_A64_xml_A_profile-2025-09_ASL1",
    ),
    # Add new releases here as ARM publishes them, e.g.:
    # "A32-2025-09": ("https://...", "ISA_AArch32_xml_A_profile-2025-09"),
}

# These files exist in the archive but are not instruction XMLs.
_EXCLUDED_FILES: frozenset[str] = frozenset({"encodingindex.xml"})


class ISADownloader:
    """
    Download, cache, and iterate over ARM ISA XML files.

    Parameters
    ----------
    release:
        A key from ``KNOWN_RELEASES``, or ``"latest"`` for the newest entry.
    cache_dir:
        Where tarballs and extracted XMLs are stored between runs.
        Defaults to ``~/.cache/arm_isa_parser``.
    force_download:
        Re-download and re-extract even when the cache already exists.
    """

    def __init__(
        self,
        release: str = "latest",
        cache_dir: Path | str | None = None,
        *,
        force_download: bool = False,
    ) -> None:
        if release == "latest":
            release = max(KNOWN_RELEASES)   # lexicographic → newest key
        if release not in KNOWN_RELEASES:
            raise ValueError(
                f"Unknown release {release!r}. Available: {sorted(KNOWN_RELEASES)}"
            )

        self.release = release
        self.url, self.archive_stem = KNOWN_RELEASES[release]
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "arm_isa_parser")
        self.force_download = force_download
        self._xml_dir = self.cache_dir / self.archive_stem

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def xml_files(self) -> list[Path]:
        """Return a sorted list of instruction XML paths."""
        self._ensure_ready()
        return sorted(
            p for p in self._xml_dir.iterdir()
            if p.suffix == ".xml" and p.name not in _EXCLUDED_FILES
        )

    def iter_xml_files(self) -> Iterator[Path]:
        """Yield instruction XML paths one at a time (lazy)."""
        yield from self.xml_files()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> None:
        if self._xml_dir.exists() and not self.force_download:
            logger.info("Cache hit — using %s", self._xml_dir)
            return
        tarball = self._download()
        self._extract(tarball)

    def _download(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        dest = self.cache_dir / f"{self.archive_stem}.tar.gz"

        if dest.exists() and not self.force_download:
            logger.info("Tarball already present at %s", dest)
            return dest

        logger.info("Downloading %s", self.url)
        tmp = dest.with_suffix(".tmp")
        try:
            with urllib.request.urlopen(self.url, timeout=60) as resp, tmp.open("wb") as fh:
                total = int(resp.headers.get("Content-Length", 0))
                done = 0
                while chunk := resp.read(1 << 16):   # 64 KiB
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done * 100 // total:3d}%  {done:,}/{total:,} B",
                              end="", flush=True)
            print()
            tmp.replace(dest)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        logger.info("Download complete: %s (%d bytes)", dest, dest.stat().st_size)
        return dest

    def _extract(self, tarball: Path) -> None:
        if self._xml_dir.exists():
            shutil.rmtree(self._xml_dir)
        self._xml_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Extracting %s → %s", tarball.name, self._xml_dir)
        with tarfile.open(tarball, "r:gz") as tf:
            for member in tf.getmembers():
                # Only extract regular files — skip dirs, symlinks, hardlinks and devices
                # (tar-slip hardening: a .xml-named symlink/link must never be materialized).
                if not member.isfile():
                    continue
                p = Path(member.name)
                if p.suffix != ".xml":
                    continue
                member.name = p.name   # flatten the archive's directory tree
                tf.extract(member, self._xml_dir)

        count = sum(1 for _ in self._xml_dir.glob("*.xml"))
        logger.info("Extracted %d XML files", count)
