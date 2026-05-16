#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALLOWLIST="${ROOT_DIR}/tools/incomplete-marker-allowlist.txt"
MARKERS='TODO|FIXME|not implemented|not yet implemented|fatalError|try!|as!|precondition\('

cd "${ROOT_DIR}"

findings="$(rg -n "${MARKERS}" Source/MLX Source/MLXLinalg Source/MLXNN Source/MLXOptimizers -g '*.swift' || true)"
if [[ -z "${findings}" ]]; then
  exit 0
fi

unallowed="${findings}"
while IFS= read -r pattern; do
  [[ -z "${pattern}" || "${pattern}" =~ ^# ]] && continue
  unallowed="$(printf '%s\n' "${unallowed}" | grep -Ev "${pattern}" || true)"
done < "${ALLOWLIST}"

if [[ -n "${unallowed}" ]]; then
  printf '%s\n' "Unallowlisted incomplete/crash markers:"
  printf '%s\n' "${unallowed}"
  exit 1
fi
