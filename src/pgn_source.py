"""Shared PGN file discovery for the extraction scripts.

parse_pgn.py, extract_policy.py and extract_cheat_features.py all resolved
their input files with the same hardcoded block (local data/raw, falling back
to the Lichess Elite folder on the Google Drive, filtered to monthly
`lichess_elite_YYYY-MM` stems). That block lives here once so any of them can
also be pointed at another corpus — e.g. the Elo-banded ChessBase export on
the external drive:

    --pgn-dir "/Volumes/My Passport Pro/ChessBase/split_by_elo" --pattern '.*'

Defaults reproduce the original behaviour exactly.
"""

import re
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent

DRIVE_DIR = Path("/Volumes/Google Drive/Data Science/Chess Data/Lichess/Lichess Elite Database")
LOCAL_RAW_DIR = _BASE / "data" / "raw"

# Monthly files only. A merged "…DownloadConflict" file on the Drive duplicates
# them, which would double-count every game if it were picked up.
DEFAULT_PATTERN = r"^lichess_elite_\d{4}-\d{2}$"


def add_source_args(ap) -> None:
    """Add --pgn-dir / --pattern to an ArgumentParser."""
    ap.add_argument("--pgn-dir", type=Path, default=None,
                    help="Directory to read PGNs from (default: data/raw/ plus "
                         "the Lichess Elite folder on the Google Drive)")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN,
                    help=f"Regex a file stem must match (default: {DEFAULT_PATTERN})")


def find_pgn_files(pgn_dir: Path | None = None,
                   pattern: str = DEFAULT_PATTERN) -> list[Path]:
    """Resolve the PGN files to process.

    With --pgn-dir: every *.pgn in that directory whose stem matches `pattern`.
    Without: local copies win over the network drive (more reliable), and the
    drive contributes only stems the local dir does not already provide.
    """
    rx = re.compile(pattern)

    if pgn_dir is not None:
        if not pgn_dir.exists():
            raise SystemExit(f"PGN directory not found: {pgn_dir}")
        files = sorted(p for p in pgn_dir.glob("*.pgn") if rx.match(p.stem))
        if not files:
            raise SystemExit(f"No PGNs matching {pattern!r} in {pgn_dir}")
        return files

    local = {p.stem: p for p in LOCAL_RAW_DIR.glob("lichess_elite_*.pgn")
             if rx.match(p.stem)} if LOCAL_RAW_DIR.exists() else {}
    drive = {}
    if DRIVE_DIR.exists():
        drive = {p.stem: p for p in DRIVE_DIR.glob("lichess_elite_*.pgn")
                 if rx.match(p.stem) and p.stem not in local}
    files = sorted((local | drive).values())
    if not files:
        raise SystemExit("No PGN files found in data/raw/ or the Google Drive folder")
    return files
