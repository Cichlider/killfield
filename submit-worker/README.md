# One-click leaderboard gateway

This Cloudflare Worker accepts a browser replay, verifies a single-use
Turnstile token, applies a per-client daily limit in KV, and opens the labelled
GitHub Issue that starts the existing authoritative replay workflow.

Required bindings and variables:

- KV binding `RATE_LIMITS`
- secrets `GITHUB_TOKEN`, `TURNSTILE_SECRET`, and `RATE_LIMIT_SALT`
- `ALLOWED_ORIGIN=https://cichlider.github.io`
- `TURNSTILE_HOSTNAME=cichlider.github.io`
- `GITHUB_REPOSITORY=Cichlider/killfield`
- `DAILY_LIMIT=10`

`GITHUB_TOKEN` should be a fine-grained token restricted to this repository
with only Issues read/write permission. Never place it in viewer code or git.
