#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$root/upstream.lock"

build_root="${BUILD_ROOT:-$root/.build}"
source_dir="${VLLM_SOURCE_DIR:-$build_root/vllm}"
series="$root/patches/vllm/series"

mkdir -p "$(dirname "$source_dir")"
if [[ ! -d "$source_dir/.git" ]]; then
  git clone --filter=blob:none "$VLLM_REPOSITORY" "$source_dir"
fi

if ! git -C "$source_dir" cat-file -e "$VLLM_COMMIT^{commit}" 2>/dev/null; then
  git -C "$source_dir" fetch --tags origin "$VLLM_COMMIT"
fi

# This directory is an isolated build checkout owned by this script.
git -C "$source_dir" checkout --detach "$VLLM_COMMIT"
git -C "$source_dir" reset --hard "$VLLM_COMMIT"
git -C "$source_dir" clean -fdx

"$root/scripts/check-upstream-contract.sh" "$source_dir" base

while IFS= read -r patch_name; do
  [[ -n "$patch_name" && "$patch_name" != \#* ]] || continue
  patch="$root/patches/vllm/$patch_name"
  if [[ ! -f "$patch" ]]; then
    echo "Patch listed in series is missing: $patch" >&2
    exit 1
  fi
  echo "Applying $patch_name"
  git -C "$source_dir" apply --check "$patch"
  git -C "$source_dir" apply "$patch"
done < "$series"

"$root/scripts/check-upstream-contract.sh" "$source_dir" patched
echo "Prepared vLLM source at $source_dir"
