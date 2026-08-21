#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

# Patterns that would identify a specific homelab. Keep this file free of
# operator usernames and private DNS — encode those as generic classes.
patterns=(
  'sk-[A-Za-z0-9_-]{16,}'
  '-----BEGIN [A-Z ]*PRIVATE KEY-----'
  '/Users/[^ /]+'
  '/home/anemll'
  '192\.168\.[0-9]+\.[0-9]+'
  '10\.0\.10\.[0-9]+'
  '[A-Za-z0-9.-]+\.whyland\.com'
  '(^|[^[:alnum:]-])spark-[12]([^[:alnum:]-]|$)'
  'ghp_[A-Za-z0-9]{20,}'
  'github_pat_[A-Za-z0-9_]{20,}'
  '(password|passwd|api[_-]?key)[[:space:]]*=[[:space:]]*[^$<{][^[:space:]]+'
  '\bHindsight\b'
  'semantic-memory'
)

failed=0
for pattern in "${patterns[@]}"; do
  if grep -RInE --exclude-dir=.git --exclude='check-private-data.sh' \
      --exclude-dir=.build --exclude-dir=__pycache__ --exclude='*.pyc' \
      -e "$pattern" .; then
    failed=1
  fi
done

# Operator home paths that are not placeholders.
if grep -RInE --exclude-dir=.git --exclude='check-private-data.sh' \
    -e '/home/[a-z][a-z0-9_-]+' . \
    | grep -vE '/home/(CHANGEME|user|operator|ubuntu|debian)([:/[:space:]]|$)' \
    | grep -vE 'runtime/patches/vllm/' ; then
  echo "Non-placeholder /home/<user> path found." >&2
  failed=1
fi

if find . -type d \( -name .git -o -name .build -o -name __pycache__ \) \
    -prune -o -type f \
    \( -name '.env' -o -name '*.pem' -o -name '*.key' -o \
       -name 'id_rsa*' -o -name 'id_ed25519*' \) -print | grep -q .; then
  echo "Private-looking files found." >&2
  failed=1
fi

(( failed == 0 )) || { echo "Private-data check failed." >&2; exit 1; }
echo "Private-data check passed."
