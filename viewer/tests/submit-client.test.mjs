import assert from "node:assert/strict";
import test from "node:test";
import { submissionTitle, submitToGateway } from "../src/submit.js";

test("Laika submissions name the correct board", () => {
  assert.equal(submissionTitle({ opponent: "laika", name: "player", claim: 2 }),
    "[score] player — 2 vs Laika");
});

test("one-click submission posts the record and challenge token", async () => {
  const originalFetch = globalThis.fetch;
  let captured;
  globalThis.fetch = async (url, options) => {
    captured = { url, options };
    return Response.json({ ok: true, issue: 42, url: "https://example.test/42" }, { status: 202 });
  };
  try {
    const result = await submitToGateway("https://worker.test/submit", { v: 1 }, "token");
    assert.equal(result.issue, 42);
    assert.equal(captured.url, "https://worker.test/submit");
    assert.equal(captured.options.headers["content-type"], "text/plain;charset=UTF-8");
    assert.deepEqual(JSON.parse(captured.options.body), {
      submission: { v: 1 }, turnstileToken: "token",
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("gateway errors reach the player", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => Response.json({ ok: false, error: "Daily limit reached." }, {
    status: 429,
  });
  try {
    await assert.rejects(
      submitToGateway("https://worker.test/submit", { v: 1 }, "token"),
      /Daily limit reached/,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("network failures have a stable code for localized UI copy", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => { throw new TypeError("Failed to fetch"); };
  try {
    await assert.rejects(
      submitToGateway("https://worker.test/submit", { v: 1 }, "token"),
      (error) => error.code === "SUBMISSION_NETWORK_ERROR",
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});
