#!/usr/bin/env bash
set -euo pipefail

# Command substitution normally strips trailing newlines, which would make an
# existing --body-file comment look different during the duplicate check.
body=$(cat comment.md; printf '\034')
body=${body%$'\034'}
endpoint="repos/$REPOSITORY/issues/$ISSUE_NUMBER/comments"
comments_file=$(mktemp)
trap 'rm -f "$comments_file"' EXIT

verdict_exists() {
  gh api "$endpoint?per_page=100" > "$comments_file" || return 2
  if jq -e --arg body "$body" \
    'any(.[]; .user.login == "github-actions[bot]" and .body == $body)' \
    "$comments_file" >/dev/null; then
    return 0
  else
    result=$?
    if (( result == 1 )); then return 1; fi
    return 2
  fi
}

# Re-running a job after the comment succeeded must not create a second copy.
if verdict_exists; then
  exit 0
else
  result=$?
fi
if (( result == 2 )); then
  echo '::warning::Could not check whether the verdict was already posted.'
  exit 1
fi

for attempt in 1 2 3; do
  if gh api "$endpoint" --method POST -f "body=$body" >/dev/null; then
    exit 0
  fi

  # A response can fail after GitHub has stored the comment. Check before
  # retrying so an ambiguous failure does not post the verdict twice.
  if verdict_exists; then
    exit 0
  else
    result=$?
  fi
  if (( result == 2 )); then
    echo '::warning::Could not check whether the verdict was posted.'
    exit 1
  fi

  if (( attempt < 3 )); then sleep $((attempt * 2)); fi
done

echo '::warning::Could not post the verdict; check the accepted issue manually.'
exit 1
