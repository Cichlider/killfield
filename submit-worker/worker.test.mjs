import assert from "node:assert/strict";
import test from "node:test";
import worker, { cleanSubmission } from "./worker.js";

const record = {
  v: 1,
  name: "rekursion",
  github: "Cichlider",
  seed: 1,
  opponent: "killfield",
  delayFrames: 0,
  openingDelaySeconds: 0.5,
  engine: "0123456789abcdef",
  policy: "fedcba9876543210",
  claim: 3,
  rounds: 3,
  frames: 3,
  startedAt: 1,
  endedAt: 2,
  track: "AAAA",
};

test("cheap validation bounds attacker-controlled records", () => {
  assert.equal(cleanSubmission(record).name, "rekursion");
  assert.equal(cleanSubmission({ ...record, name: " admin\u202e\u0000 " }).name, "admin");
  assert.equal(cleanSubmission({ ...record, github: "" }).value.github, null);
  assert.throws(() => cleanSubmission({ ...record, track: "!" }), /payload/);
  assert.throws(() => cleanSubmission({ ...record, frames: 60_001 }), /dimensions/);
  assert.throws(() => cleanSubmission({ ...record, claim: 4, rounds: 3 }), /dimensions/);
  assert.throws(() => cleanSubmission({ ...record, claim: 0 }), /dimensions/);
  assert.throws(() => cleanSubmission({ ...record, delayFrames: 1 }), /ranked settings/);
  assert.throws(() => cleanSubmission({ ...record, openingDelaySeconds: 0.6 }), /ranked settings/);
  assert.throws(() => cleanSubmission({ ...record, github: "not valid!" }), /GitHub/);
  assert.throws(() => cleanSubmission({ ...record, github: "bad--handle" }), /GitHub/);
  assert.equal(cleanSubmission({ ...record, opponent: "laika" }).value.opponent, "laika");
  assert.throws(() => cleanSubmission({ ...record, opponent: "unknown" }), /not supported/);
  assert.equal(cleanSubmission({ ...record, injected: "not forwarded" }).value.injected, undefined);
});

test("a valid request verifies the challenge and creates a labelled issue", async () => {
  const calls = [];
  const events = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    calls.push({ url: String(url), options });
    if (String(url).includes("siteverify")) {
      return Response.json({ success: true, hostname: "cichlider.github.io", action: "leaderboard-submit" });
    }
    events.push("issue");
    return Response.json({ number: 42, html_url: "https://github.com/Cichlider/killfield/issues/42" });
  };
  const writes = [];
  const env = {
    ALLOWED_ORIGIN: "https://cichlider.github.io",
    TURNSTILE_SECRET: "secret",
    TURNSTILE_HOSTNAME: "cichlider.github.io",
    RATE_LIMIT_SALT: "salt",
    GITHUB_TOKEN: "token",
    GITHUB_REPOSITORY: "Cichlider/killfield",
    DAILY_LIMIT: "10",
    RATE_LIMITS: {
      get: async () => null,
      put: async (...args) => { events.push("reserve"); writes.push(args); },
    },
  };
  try {
    const request = new Request("https://worker.example/submit", {
      method: "POST",
      headers: {
        origin: "https://cichlider.github.io",
        "cf-connecting-ip": "203.0.113.5",
        "content-type": "application/json",
      },
      body: JSON.stringify({
        submission: { ...record, opponent: "laika" }, turnstileToken: "challenge",
      }),
    });
    const response = await worker.fetch(request, env);
    assert.equal(response.status, 202);
    assert.deepEqual(await response.json(), {
      ok: true, issue: 42, url: "https://github.com/Cichlider/killfield/issues/42",
    });
    assert.equal(calls.length, 2);
    const issue = JSON.parse(calls[1].options.body);
    assert.deepEqual(issue.labels, ["leaderboard"]);
    assert.equal(issue.title, "[wins] 3 vs Laika");
    assert.match(issue.body, /```json/);
    assert.match(issue.body, /killfield-gateway:v1:[a-f0-9]{16}/);
    assert.equal(writes.length, 1);
    assert.match(writes[0][0], /^daily:[a-f0-9]{16}$/);
    assert.equal(writes[0][2].expirationTtl, 86_400);
    assert.deepEqual(events, ["reserve", "issue"]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("foreign origins are rejected before consuming a challenge", async () => {
  const response = await worker.fetch(new Request("https://worker.example/submit", {
    method: "POST",
    headers: { origin: "https://attacker.example" },
    body: "{}",
  }), { ALLOWED_ORIGIN: "https://cichlider.github.io" });
  assert.equal(response.status, 403);
});
