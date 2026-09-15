#!/usr/bin/env bash
#
# Resolve the `NEXT` placeholder in Sphinx version directives.
#
# ---------------------------------------------------------------------------------------
# THIS SCRIPT IS COPIED, NOT SHARED. Identical copies live in:
#
#     OasisLMF/scripts/resolve-version-markers.sh
#     OasisPlatform/scripts/resolve-version-markers.sh
#     ODS_Tools/scripts/resolve-version-markers.sh
#
# A fix here is NOT a fix there. Port every change to all three, or the repos will resolve
# markers differently and nobody will notice until a release stamps the wrong version.
#
# Exactly one thing is meant to differ between copies: the SCOPE exclusion below, which
# names that repo's file documenting the convention (it necessarily contains literal
# example markers). Everything else should stay byte-identical.
# ---------------------------------------------------------------------------------------
#
# A PR author cannot know which release their change will ship in — the version is only
# written at release time, by .github/workflows/version.yml and the update-version job in
# .github/workflows/release-pr.yml. So docs and docstrings are authored with NEXT in place
# of the version, and this script rewrites NEXT to the real version on the release branch,
# in the same commit as the __init__.py version bump.
#
# Both directive syntaxes are handled, since the docs are a mix of reStructuredText and
# MyST Markdown:
#
#     .. versionadded:: NEXT        (rst, and Python docstrings)
#     ```{versionadded} NEXT        (MyST; ::: fences too)
#
# Only the version argument is touched, and only when NEXT is the whole argument — a NEXT
# appearing anywhere else in the prose is left alone.
#
# Usage:
#   scripts/resolve-version-markers.sh 2.5.8   # rewrite NEXT -> 2.5.8 in place
#   scripts/resolve-version-markers.sh --check # exit 1 if any NEXT marker remains
#   scripts/resolve-version-markers.sh --lint  # exit 1 only on markers the rewrite
#                                              # cannot resolve (safe to run on any PR)

set -euo pipefail

DIRECTIVES='versionadded|versionchanged|versionremoved|deprecated'

# The directive up to and including the space before its version argument.
# Group 1 is the whole prefix INCLUDING indentation, so the rewrite preserves it.
PREFIX="^([[:space:]]*(\\.\\. (${DIRECTIVES}):: |(\`{3,}|:{3,})\\{(${DIRECTIVES})\\} ))"

# NEXT as the whole argument — what gets rewritten.
MARKER_RE="${PREFIX}NEXT[[:space:]]*$"

# A version argument that mentions NEXT but is not exactly NEXT, e.g.
# ".. versionadded:: NEXT (see below)". The rewrite deliberately leaves these alone, so
# --check reports them rather than letting them ship unresolved.
MALFORMED_RE="${PREFIX}(.+NEXT|NEXT[[:space:]]+[^[:space:]].*)$"

# CONTRIBUTING.md documents the convention, so it necessarily contains literal example
# markers; rewriting them would destroy the instructions. Anywhere else that explains the
# convention should link here rather than repeat the example on a line of its own.
SCOPE=(-- . ':(exclude)CONTRIBUTING.md')

if [[ "${1:-}" == "--check" || "${1:-}" == "--lint" ]]; then
    STATUS=0

    # An unresolvable marker is a mistake wherever it appears, including on main — so
    # --lint reports it on any branch. A plain NEXT marker is only wrong at release time.
    if MALFORMED=$(git grep -nE "${MALFORMED_RE}" "${SCOPE[@]}" 2>/dev/null); then
        echo "Version directives mentioning NEXT that the rewrite cannot resolve:"
        echo "${MALFORMED}"
        echo
        echo "NEXT must be the whole version argument, e.g. '.. versionadded:: NEXT'."
        STATUS=1
    fi

    if [[ "${1}" == "--check" ]] && MATCHES=$(git grep -nE "${MARKER_RE}" "${SCOPE[@]}" 2>/dev/null); then
        echo "Unresolved NEXT version markers found:"
        echo "${MATCHES}"
        echo
        echo "Run: scripts/resolve-version-markers.sh <version>"
        STATUS=1
    fi

    if [[ "${STATUS}" -eq 0 ]]; then
        echo "NEXT version markers OK."
    fi
    exit "${STATUS}"
fi

VERSION="${1:-}"
if [[ -z "${VERSION}" ]]; then
    echo "usage: $0 <version> | --check | --lint" >&2
    exit 2
fi

# Guard the sed replacement: an unescaped '/' or '&' in $VERSION would corrupt every file
# it touches, so only accept something that is actually a release version.
if ! [[ "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+(rc[0-9]+)?$ ]]; then
    echo "Not a valid semver release version: ${VERSION}" >&2
    exit 2
fi

# `git grep -l` scopes the rewrite to tracked files, so a build directory or an untracked
# scratch file in the working tree is never rewritten.
FILES=$(git grep -lE "${MARKER_RE}" "${SCOPE[@]}" || true)
if [[ -z "${FILES}" ]]; then
    echo "No NEXT version markers to resolve."
    exit 0
fi

# -z/-0 rather than a plain pipe: xargs splits on whitespace, which would break on a
# tracked path containing a space.
git grep -lzE "${MARKER_RE}" "${SCOPE[@]}" \
    | xargs -0 sed -i -E "s/${MARKER_RE}/\\1${VERSION}/"

echo "Resolved NEXT -> ${VERSION} in:"
echo "${FILES}" | sed 's/^/  /'
