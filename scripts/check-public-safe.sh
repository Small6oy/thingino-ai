#!/usr/bin/env bash
# Reject household/private details before they reach the public mirror.
# Usage: scripts/check-public-safe.sh   (run manually or from a pre-commit hook)
set -eu
pat='([0-9]{1,3}\.){3}[0-9]{1,3}|\.lan([^a-z]|$)|/Volumes/|/srv/|/home/|/Users/|forgejo'
fail=0
while IFS= read -r f; do
  case "$f" in
    models/*|data/*|scripts/check-public-safe.sh) continue ;;
  esac
  if grep -InE "$pat" -- "$f" >/dev/null 2>&1; then
    echo "public-safe: '$f' has a private pattern:" >&2
    grep -InE "$pat" -- "$f" >&2
    fail=1
  fi
done < <(git ls-files)
[ "$fail" -eq 0 ] || { echo "public-safe: FAILED" >&2; exit 1; }
echo "public-safe: ok"
