#!/bin/bash
# Installs the post sorter skill for Claude Code on a Mac.
#
#   ./install.sh                      install the skill and its Python packages
#   ./install.sh --with-google-sheet  also install the optional Google Sheet packages
#   ./install.sh --no-packages        copy the skill only (no downloads)
#
# What it does:
#   1. Copies skill/ to ~/.claude/skills/briefpost-management
#   2. Creates a private Python environment in that folder (.venv) and installs the packages
#   3. Creates ~/.config/briefpost-management/.env for your API key (only if it does not exist)
#   4. Runs the offline self-test
#
# It never overwrites settings you changed: an edited config/defaults.yaml is kept
# (the new one is saved next to it as defaults.yaml.new), an existing .env is left alone,
# and project folders are never touched.

set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)/skill"
DEST="${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}/briefpost-management"
CONF="$HOME/.config/briefpost-management"
INSTALL_PACKAGES=1
WITH_GOOGLE=0

for arg in "$@"; do
  case "$arg" in
    --no-packages) INSTALL_PACKAGES=0 ;;
    --with-google-sheet) WITH_GOOGLE=1 ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)"; exit 1 ;;
  esac
done

say() { printf '%s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ -d "$SRC" ] || fail "skill folder not found next to install.sh ($SRC)"
[ -f "$SRC/SKILL.md" ] || fail "skill/SKILL.md is missing, the download looks incomplete"

# --- 1. Python ---------------------------------------------------------------
command -v python3 >/dev/null 2>&1 || fail "python3 not found. Install it with: xcode-select --install  (or from https://www.python.org/downloads/)"
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
  fail "Python 3.9 or newer is needed, found $(python3 --version 2>&1)"
fi
say "Python: $(python3 --version 2>&1)"

# --- 2. Copy the skill -------------------------------------------------------
mkdir -p "$DEST"
KEEP_DEFAULTS=0
if [ -f "$DEST/config/defaults.yaml" ] && ! cmp -s "$SRC/config/defaults.yaml" "$DEST/config/defaults.yaml"; then
  KEEP_DEFAULTS=1
  cp "$DEST/config/defaults.yaml" "$DEST/config/defaults.yaml.keep"
fi

# Copy everything except Python caches. Files that exist only in the target (.venv) stay.
(cd "$SRC" && find . -type d ! -name '__pycache__' ! -path '*/__pycache__/*' -print) | while IFS= read -r dir; do
  mkdir -p "$DEST/$dir"
done
(cd "$SRC" && find . -type f ! -name '*.pyc' ! -name '.DS_Store' ! -path '*/__pycache__/*' -print) | while IFS= read -r file; do
  cp "$SRC/$file" "$DEST/$file"
done

if [ "$KEEP_DEFAULTS" -eq 1 ]; then
  cp "$DEST/config/defaults.yaml" "$DEST/config/defaults.yaml.new"
  mv "$DEST/config/defaults.yaml.keep" "$DEST/config/defaults.yaml"
  say "Kept your edited config/defaults.yaml. The new version is saved as config/defaults.yaml.new"
fi
say "Skill copied to: $DEST"

# --- 3. Place for the API key ------------------------------------------------
mkdir -p "$CONF"
chmod 700 "$CONF"
if [ -e "$CONF/.env" ]; then
  say "Kept your existing key file: $CONF/.env"
else
  ( umask 077; printf '%s\n' \
      '# Gemini API key for the post sorter. Create one at https://aistudio.google.com/apikey' \
      '# Remove the "#" in front of the next line and paste your key after the "=".' \
      '# GOOGLE_API_KEY=' > "$CONF/.env" )
  say "Created key file (empty): $CONF/.env"
fi

# --- 4. Python packages in a private environment -----------------------------
PY="python3"
if [ "$INSTALL_PACKAGES" -eq 1 ]; then
  if [ ! -x "$DEST/.venv/bin/python" ]; then
    say "Creating a private Python environment..."
    python3 -m venv "$DEST/.venv" || fail "could not create the Python environment in $DEST/.venv"
  fi
  PY="$DEST/.venv/bin/python"
  say "Installing packages (needs internet, takes a minute)..."
  "$PY" -m pip install --quiet --disable-pip-version-check --upgrade pip >/dev/null 2>&1 || true
  "$PY" -m pip install --quiet --disable-pip-version-check -r "$DEST/requirements.txt" \
    || fail "package install failed. Check your internet connection and run ./install.sh again"
  if [ "$WITH_GOOGLE" -eq 1 ]; then
    "$PY" -m pip install --quiet --disable-pip-version-check -r "$DEST/requirements-google-sheet.txt" \
      || fail "Google Sheet package install failed. Run ./install.sh --with-google-sheet again"
  fi
  say "Packages installed."
else
  [ -x "$DEST/.venv/bin/python" ] && PY="$DEST/.venv/bin/python"
  say "Skipped package install (--no-packages)."
fi

# --- 5. Self-test ------------------------------------------------------------
say ""
say "Running the offline self-test..."
if "$PY" "$DEST/scripts/selftest.py"; then
  say ""
  say "Installed and working."
else
  say ""
  say "The skill is copied, but the self-test did not pass."
  if [ "$INSTALL_PACKAGES" -eq 0 ]; then
    say "You skipped the package install. Run ./install.sh without --no-packages."
  fi
  exit 1
fi

say ""
say "Next steps:"
say "  1. Put your Gemini API key in $CONF/.env"
say "  2. Create a project folder:  \"$PY\" \"$DEST/scripts/cli.py\" ~/post-project --init"
say "  3. Add your letter photos to ~/post-project/raw-images/unsorted/"
say "  4. In Claude Code, say: sort my post in ~/post-project"
say "     (or run:  \"$PY\" \"$DEST/scripts/cli.py\" ~/post-project )"
