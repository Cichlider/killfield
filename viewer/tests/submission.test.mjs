import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { deflateRawSync } from "node:zlib";
import { LIMITS, encodeSession, packSession, unpackSession } from "../src/replay.js";
import { buildStamps, submissionBody } from "../src/submit.js";

// A real qualifying run, recorded once by tools/make_test_fixture.mjs: finding
// one takes minutes, because the "human" that plays it is the policy driving
// itself through the keyboard path, which is weaker than the discrete path it
// was trained on — continuous input skips the ten-degree turn lattice.
const FIXTURE = new URL("fixtures/qualifying-run.json", import.meta.url);
const fixture = JSON.parse(fs.readFileSync(FIXTURE, "utf8"));

const WASM = fs.readFileSync(new URL("../kf_engine.wasm", import.meta.url));
const weightBytes = fs.readFileSync(new URL("../assets/hybrid.bin", import.meta.url));
const stamps = await buildStamps(WASM, new Float32Array(
  weightBytes.buffer, weightBytes.byteOffset, weightBytes.byteLength / 4,
));
assert.equal(fixture.engine, stamps.engine,
  "the fixture was recorded on a different engine build — rerun tools/make_test_fixture.mjs");
assert.equal(fixture.policy, stamps.policy,
  "the fixture was recorded on different policy weights — rerun tools/make_test_fixture.mjs");

// The fixture has to be a record the page could actually have produced.
const session = await unpackSession(fixture.track);
assert.ok(session.frames.length > 0 && session.frames.length <= LIMITS.maxFrames);
assert.equal(fixture.delayFrames, 0);

const workdir = fs.mkdtempSync(path.join(os.tmpdir(), "kf-submission-"));
const verifier = new URL("../../tools/verify_replay.mjs", import.meta.url).pathname;
// A scratch board, never the real viewer/leaderboard.json. Running the suite
// with --write must not depend on, or mutate, production state — the same
// entry accepted twice across two independent test runs has to be accepted
// twice, not rejected as "already on the board" because a previous run (or a
// developer's manual check) happened to leave it there.
const boardPath = path.join(workdir, "leaderboard.json");
fs.writeFileSync(boardPath, JSON.stringify({ updated: null, entries: [] }));

let isolatedBoardNumber = 0;
/** Run the verifier the way CI does, with body passed through the environment.
 *  Most cases get their own empty board so one assertion cannot contaminate
 *  another. The duplicate tests explicitly share `boardPath`, whose first
 *  record is written by the initial acceptance case. */
function runVerifier(body, {
  author = "tester", write = false, shared = false,
  gatewayAuthor = "",
  boardContents = JSON.stringify({ updated: null, entries: [] }),
} = {}) {
  const bodyPath = path.join(workdir, "body.md");
  fs.writeFileSync(bodyPath, body);
  const selectedBoard = shared
    ? boardPath
    : path.join(workdir, `isolated-board-${isolatedBoardNumber++}.json`);
  if (!shared) fs.writeFileSync(selectedBoard, boardContents);
  const args = [verifier, "--file", bodyPath, "--board", selectedBoard];
  if (write) args.push("--write");
  try {
    execFileSync(process.execPath, args, {
      cwd: workdir, env: {
        ...process.env,
        ISSUE_AUTHOR: author,
        ISSUE_NUMBER: "7",
        LEADERBOARD_GATEWAY_AUTHOR: gatewayAuthor,
      },
      stdio: "pipe",
      // A genuine replay of the ~12k-frame fixture takes several seconds; this
      // is a backstop against an actual hang (e.g. a malformed record that
      // stalls the engine), not a performance budget.
      timeout: 60_000,
    });
  } catch (error) {
    if (error.signal === "SIGTERM" || error.code === "ETIMEDOUT") {
      throw new Error(`verifier hung (>60s) on: ${body.slice(0, 200)}`);
    }
    // A rejection exits non-zero; the verdict file still holds the reason.
  }
  return {
    verdict: JSON.parse(fs.readFileSync(path.join(workdir, "verdict.json"), "utf8")),
    comment: fs.readFileSync(path.join(workdir, "comment.md"), "utf8"),
  };
}

// ------------------------------------------------------------------- accepts

const accepted = runVerifier(submissionBody(fixture), { write: true, shared: true });
assert.equal(accepted.verdict.ok, true,
  `verifier rejected an honest record: ${accepted.verdict.reason}`);
assert.equal(accepted.verdict.entry.score, fixture.claim);
assert.equal(accepted.verdict.entry.board, "hybrid");
assert.equal(accepted.verdict.entry.name, "Test Runner");
assert.match(accepted.comment, /\*\*Verified\.\*\*/);

// A record goes up under a name. Showing the account behind it is optional,
// and only kept when it is the account that actually opened the issue.
assert.equal(accepted.verdict.entry.github, null, "an undeclared handle was filled in");
assert.ok(accepted.verdict.entry.submitter, "the rate-limit key must survive an opt-out");
const owned = runVerifier(submissionBody({ ...fixture, github: "tester" }));
assert.equal(owned.verdict.entry.github, "tester");
const atPrefixed = runVerifier(submissionBody({ ...fixture, github: "@TESTER" }));
assert.equal(atPrefixed.verdict.entry.github, "tester", "handles compare case-insensitively");

// One-click submissions are opened by the service account, not the player.
// Their optional GitHub field is intentionally self-reported, while the
// opaque gateway key keeps different clients out of one shared rate bucket.
const gatewayBody = submissionBody({ ...fixture, github: "someone-else" })
  + "\n<!-- killfield-gateway:v1:0123456789abcdef -->\n";
const gateway = runVerifier(gatewayBody, {
  author: "Cichlider", gatewayAuthor: "Cichlider",
});
assert.equal(gateway.verdict.ok, true);
assert.equal(gateway.verdict.entry.github, "someone-else");
assert.equal(gateway.verdict.entry.submitter, "0123456789abcdef");
const forgedGateway = runVerifier(gatewayBody, {
  author: "attacker", gatewayAuthor: "Cichlider",
});
assert.equal(forgedGateway.verdict.ok, false);
assert.match(forgedGateway.verdict.reason, /configured gateway account/);

// The score is the replay's, never the submission's.
const inflated = runVerifier(submissionBody({ ...fixture, claim: fixture.claim + 50 }));
assert.equal(inflated.verdict.entry.score, fixture.claim);

// Resubmitting the identical track is refused even under a different name —
// records are public, so this is the cheapest forgery there is. Recompressing
// the same session (a different deflate encoding of identical frames) must be
// caught too, which is why the identity hash is computed from the decoded,
// canonically re-encoded session rather than from the submitted base64.
const resubmitted = runVerifier(
  submissionBody({ ...fixture, name: "Someone Else" }), { shared: true },
);
assert.equal(resubmitted.verdict.ok, false);
assert.match(resubmitted.verdict.reason, /already on the board/);
const recompressedTrack = deflateRawSync(encodeSession(session), { level: 9 }).toString("base64");
assert.notEqual(recompressedTrack, fixture.track,
  "the alternate compressor must actually produce a different wire payload");
const recompressed = runVerifier(submissionBody({
  ...fixture, name: "Recompressed Thief", track: recompressedTrack,
}), { shared: true });
assert.equal(recompressed.verdict.ok, false);
assert.match(recompressed.verdict.reason, /already on the board/);

// ------------------------------------------------------------------ rejects

const rejected = (body, expect) => {
  const { verdict, comment } = runVerifier(body);
  assert.equal(verdict.ok, false, `expected a rejection for ${expect}`);
  assert.match(verdict.reason, expect);
  assert.match(comment, /\*\*Not verified\.\*\*/);
};

const corruptBoard = runVerifier(submissionBody(fixture), { boardContents: "{" });
assert.equal(corruptBoard.verdict.ok, false);
assert.match(corruptBoard.verdict.reason, /leaderboard file is corrupt/);

rejected("no record here at all", /no ```json record block/);
rejected("```json\n{not json}\n```", /not valid JSON/);
rejected(submissionBody({ ...fixture, v: 99 }), /unsupported record version/);
rejected(submissionBody({ ...fixture, opponent: "laika" }), /is not a board/);
rejected(submissionBody({ ...fixture, opponent: { toString: 1 } }), /is not a board/);
rejected(submissionBody({ ...fixture, seed: -1 }), /seed is not a u32/);
rejected(submissionBody({ ...fixture, seed: 1.5 }), /seed is not a u32/);
rejected(submissionBody({ ...fixture, engine: "0".repeat(16) }), /this repo ships/);
rejected(submissionBody({ ...fixture, policy: "0".repeat(16) }), /this repo ships/);
rejected(submissionBody({ ...fixture, track: "!!!!" }), /not base64/);
rejected(submissionBody({ ...fixture, track: 42 }), /has no track/);

// Nothing goes on the board anonymously, and nothing goes on it wearing
// somebody else's account.
rejected(submissionBody({ ...fixture, name: "" }), /has no name on it/);
rejected(submissionBody({ ...fixture, name: "   " }), /has no name on it/);
rejected(submissionBody({ ...fixture, name: "​‮" }), /has no name on it/);
rejected(submissionBody({ ...fixture, github: "torvalds" }),
  /claims @torvalds but the issue was opened by @tester/);

// Ranked is the default match: every handicap that makes the opponent easier
// is refused, whichever of the two knobs it came from.
rejected(submissionBody({ ...fixture, delayFrames: 1 }), /only takes runs at zero/);
rejected(submissionBody({ ...fixture, delayFrames: 3 }), /only takes runs at zero/);
rejected(submissionBody({ ...fixture, openingDelaySeconds: 1.5 }), /allows at most/);
rejected(submissionBody({ ...fixture, openingDelaySeconds: 3 }), /allows at most/);

// A different seed replays a different maze, so the run evaporates.
rejected(submissionBody({ ...fixture, seed: (fixture.seed + 1) >>> 0 }),
  /the board starts at|do not match the policy/);

// Standing the opponent still is the cheapest forgery, and the one the action
// audit exists for. It is caught before the score is even computed.
const idleTrack = await packSession({
  frames: session.frames.map((frame) => ({ ...frame, action: 8 })),
  events: session.events,
});
rejected(submissionBody({ ...fixture, track: idleTrack }), /do not match the policy/);

// ------------------------------------------------------------ shape of a body

// The payload must survive being pasted alongside arbitrary prose, and nothing
// outside the fence may be read.
const noisy = `Hi! Here is my run.\n\n<!-- ${"x".repeat(500)} -->\n\n`
  + submissionBody(fixture) + "\nThanks!\n";
assert.equal(runVerifier(noisy).verdict.ok, true, "prose around the fence broke parsing");

// The name is sanitised on the way in, not merely on the way out.
const spoofed = runVerifier(submissionBody({
  ...fixture, name: "admin‮  x".repeat(9),
}));
assert.ok(!spoofed.verdict.entry.name.includes("‮"));
assert.ok(!spoofed.verdict.entry.name.includes(" "));
assert.ok(spoofed.verdict.entry.name.length <= LIMITS.maxNameLength);

// A rejection reason quotes values the submitter chose, and must not be able
// to break out of the fence it is shown in.
const backticked = runVerifier(submissionBody({ ...fixture, opponent: "```\nhi" }));
assert.equal(backticked.verdict.ok, false);
assert.equal(backticked.comment.split("```").length, 3,
  "a submitted backtick escaped the fence in the issue comment");

fs.rmSync(workdir, { recursive: true, force: true });
console.log(`submission: a ${fixture.claim}-round shutout over ${fixture.rounds} rounds `
  + `(${session.frames.length} frames) accepted; `
  + "forged, mistuned and malformed records all rejected");
