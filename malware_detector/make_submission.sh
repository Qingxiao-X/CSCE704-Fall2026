#!/usr/bin/env bash
# Build the defense ZIP for the course platform. Usage:
#   ./make_submission.sh [version]      -> dist/defense-<version>.zip
# The ZIP holds only what the Docker build needs (Dockerfile at its root),
# taken from the working tree but filtered through .gitignore, so local data/,
# .venv/ and caches never leak in.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VERSION="${1:-1.0.0}"
OUT="dist/defense-${VERSION}.zip"
FILES=(Dockerfile requirements-runtime.txt src models/detector)

mkdir -p dist
rm -f "$OUT"
git ls-files --cached --others --exclude-standard -- "${FILES[@]}" | python3 -c '
import sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for path in sys.stdin.read().splitlines():
        z.write(path)
' "$OUT"

list_zip() { python3 -c 'import sys, zipfile; print("\n".join(zipfile.ZipFile(sys.argv[1]).namelist()))' "$OUT"; }

n_dockerfiles=$(list_zip | grep -cE '(^|/)Dockerfile$' || true)
if [ "$n_dockerfiles" -ne 1 ]; then
    echo "error: expected exactly one Dockerfile in $OUT, found $n_dockerfiles" >&2
    exit 1
fi
if list_zip | grep -qiE '\.(exe|dll|sys|scr)$'; then
    echo "error: $OUT contains PE binaries; defense ZIPs must not include samples" >&2
    exit 1
fi

echo "wrote $OUT ($(du -h "$OUT" | cut -f1)):"
list_zip | sed 's/^/  /'
