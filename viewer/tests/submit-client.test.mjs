import assert from "node:assert/strict";
import test from "node:test";
import { submitToGateway } from "../src/submit.js";

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
