#!/usr/bin/env bash
# Step 1: get the project code from GitHub, at the version this lab was tested with.
#   PROJECT_COMMIT=<commit> ./01_get_code.sh   -> another version (untested with this lab)
set -euo pipefail
source "$(dirname "$0")/env.sh"
# The lab was tested on this commit (master with PR #1, the lab fixes:
# Linux extension start, VLM model setting, VLM memory). Newer commits may change things the lab relies on.
COMMIT="${PROJECT_COMMIT:-9a4c346}"

say "Get the code -> $PROJECT"
if [ -d "$PROJECT/.git" ]; then
  git -C "$PROJECT" fetch -q origin || fail "could not fetch (offline?), using the code we have"
else
  git clone "$REPO_URL" "$PROJECT"
fi

say "Version $COMMIT"
if [ "$(git -C "$PROJECT" rev-parse HEAD)" = "$(git -C "$PROJECT" rev-parse "$COMMIT^{commit}")" ] &&
   git -C "$PROJECT" diff --quiet; then
  ok "already on $COMMIT"
else
  # Older lab fixes (the patch earlier versions of this script applied) or own edits:
  # keep them in a git stash (git stash list), then start clean.
  if ! git -C "$PROJECT" diff --quiet; then
    git -C "$PROJECT" stash push -q -m "before $COMMIT $(date +%F_%T)" && ok "previous changes saved: git stash list"
  fi
  git -C "$PROJECT" checkout -q "$COMMIT"
fi
git -C "$PROJECT" log --oneline -1
ok "Code ready"
