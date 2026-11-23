#!/usr/bin/env bash
# Reject household/private details before they reach the public mirror.
# Usage: scripts/check-public-safe.sh   (run manually or from a pre-commit hook)
set -eu
pat='(192\.168(\.[0-9]{1,3}){1,2}|10(\.[0-9]{1,3}){3}|172\.(1[6-9]|2[0-9]|3[01])(\.[0-9]{1,3}){1,2})|\.lan([^a-z]|$)|/Volumes/|/srv/|/home/|/Users/|forgejo'
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
# Ingenic vendor binaries must never reach the public mirror (licensing).
vb=$(git ls-files | grep -E "(^|/)(libjzdl|libvenus|libaip|libdrivers|libstdc\\+\\+|magik_model_)" || true)
if [ -n "$vb" ]; then echo "public-safe: Ingenic vendor binaries must not be committed:" >&2; echo "$vb" >&2; fail=1; fi
[ "$fail" -eq 0 ] || { echo "public-safe: FAILED" >&2; exit 1; }
echo "public-safe: ok"
