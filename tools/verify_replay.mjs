#!/usr/bin/env node
/**
 * Verify one leaderboard submission and, if it holds up, place it on the board.
 *
 * This is the only code in the project that runs on input from a stranger: a
 * public issue triggers it, and it holds a token that can write to the repo.
 * Two rules follow from that and are not negotiable.
 *
 *   1. Nothing from the issue is ever interpolated into a shell command. The
 *      body arrives through the environment and is parsed here, in JS. The
 *      workflow that calls this must pass it with `env:`, never `${{ }}` inside
 *      a `run:` block, which is a remote code execution hole.
 *   2. Every bound is checked before the expensive work starts, and the cheap
 *      rejections come first: size, then shape, then the replay.
 *
 * The score is never read from the submission. It is recomputed by replaying
 * the recorded inputs through the same engine binary the player used, and the
 * opponent's recorded actions are audited against the policy's own decisions
 * on the identical observations (see src/ranked.js).
 *
 * Usage:
 *   ISSUE_BODY=... ISSUE_AUTHOR=... ISSUE_NUMBER=... node tools/verify_replay.mjs
 *   node tools/verify_replay.mjs --file submission.md   (local dry run)
 *   node tools/verify_replay.mjs --file submission.md --board /tmp/board.json (test isolation)
 */

import fs from "node:fs";
import { createHash } from "node:crypto";
import {
  LIMITS, MIN_SUBMITTABLE_SHUTOUT, RejectedSubmission,
  encodeSession, longestShutout, sanitiseHandle, sanitiseName, unpackSession,
} from "../viewer/src/replay.js";
import {
  FPS, RANKED_DELAY_FRAMES, RANKED_OPENING_DELAY_SECONDS, replaySession,
} from "../viewer/src/ranked.js";
import { HybridPolicy } from "../viewer/src/hybrid.js";

const VIEWER = new URL("../viewer/", import.meta.url);
// Overridable so the test suite never reads or writes the real, committed
// board — a prior run of these same tests left "Test Runner" sitting in
// viewer/leaderboard.json and CI itself has no other way to test --write
// without mutating production data.
const boardFlagIndex = process.argv.indexOf("--board");
const BOARD_PATH = boardFlagIndex === -1
  ? new URL("leaderboard.json", VIEWER)
  : process.argv[boardFlagIndex + 1];
const SUBMISSION_VERSION = 1;
const BOARDS = { hybrid: "hybrid", killfield: "killfield" };
/** Ceiling on entries one account can land in a day, to bound Actions spend. */
const DAILY_SUBMISSION_LIMIT = 10;
const MAX_JSON_CHARS = 80_000;

const reject = (why) => { throw new RejectedSubmission(why); };

// --------------------------------------------------------------- extraction

/** Pull the fenced JSON block out of an issue body. Everything else is prose
 *  for humans and is never interpreted. */
function extractSubmission(body) {
  if (typeof body !== "string" || body.length === 0) reject("the issue body is empty");
  if (body.length > MAX_JSON_CHARS * 2) reject("the issue body is too large to parse");
  const fence = /```json\s*\n([\s\S]*?)\n```/.exec(body);
  if (fence === null) reject("no ```json record block in the issue body");
  const text = fence[1];
  if (text.length > MAX_JSON_CHARS) reject("the record block is too large");
  try {
    return JSON.parse(text);
  } catch {
    reject("the record block is not valid JSON");
  }
  return null;
}

const isInteger = (value, low, high) =>
  Number.isInteger(value) && value >= low && value <= high;

function validateShape(submission) {
  if (submission === null || typeof submission !== "object" || Array.isArray(submission)) {
    reject("the record is not an object");
  }
  if (submission.v !== SUBMISSION_VERSION) reject(`unsupported record version ${submission.v}`);
  if (!Object.hasOwn(BOARDS, submission.opponent)) {
    reject(`${JSON.stringify(submission.opponent)} is not a board`);
  }
  // Ranked is the default match. Both handicaps make the opponent easier, so
  // the delay must be absent outright and the opening pause may only be
  // shorter than the default, never longer.
  if (submission.delayFrames !== RANKED_DELAY_FRAMES) {
    reject(`the opponent was given a ${submission.delayFrames}-frame delay; `
      + "the board only takes runs at zero");
  }
  if (!Number.isFinite(submission.openingDelaySeconds)
      || submission.openingDelaySeconds < 0
      || submission.openingDelaySeconds > RANKED_OPENING_DELAY_SECONDS) {
    reject(`the opening pause was ${submission.openingDelaySeconds}s; `
      + `the board allows at most ${RANKED_OPENING_DELAY_SECONDS}s`);
  }
  if (!isInteger(submission.seed, 0, 0xffffffff)) reject("the seed is not a u32");
  if (sanitiseName(submission.name) === "") reject("the record has no name on it");
  if (typeof submission.track !== "string") reject("the record has no track");
  if (typeof submission.engine !== "string" || typeof submission.policy !== "string") {
    reject("the record does not name the binaries it was played on");
  }
  return {
    opponent: submission.opponent,
    delayFrames: submission.delayFrames,
    seed: submission.seed,
    openingDelaySeconds: submission.openingDelaySeconds,
  };
}

// ------------------------------------------------------------------ binaries

function stampOf(bytes) {
  return createHash("sha256").update(bytes).digest("hex").slice(0, 16);
}

function loadEngine() {
  const bytes = fs.readFileSync(new URL("kf_engine.wasm", VIEWER));
  return { bytes, stamp: stampOf(bytes) };
}

function loadPolicy() {
  const manifest = JSON.parse(fs.readFileSync(new URL("assets/hybrid.json", VIEWER)));
  const bytes = fs.readFileSync(new URL("assets/hybrid.bin", VIEWER));
  const weights = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  return { policy: new HybridPolicy(manifest, weights), stamp: stampOf(bytes) };
}

// --------------------------------------------------------------------- board

/**
 * A missing board starts empty — the normal state before the first record
 * lands. Anything else wrong with it (corrupt JSON, the wrong shape, a read
 * error that isn't "not found") aborts instead: silently treating a damaged
 * board as empty would mean the very next accepted record commits a file that
 * has quietly dropped every entry before it.
 */
function loadBoard() {
  let text;
  try {
    text = fs.readFileSync(BOARD_PATH, "utf8");
  } catch (error) {
    if (error.code === "ENOENT") return { updated: null, entries: [] };
    reject("the leaderboard file cannot be read; refusing to replace it");
  }
  let board;
  try {
    board = JSON.parse(text);
  } catch {
    reject("the leaderboard file is corrupt; refusing to replace it");
  }
  if (!Array.isArray(board.entries)) {
    reject("the leaderboard file has no entries array; refusing to replace it");
  }
  return board;
}

function checkNotADuplicate(board, trackHash) {
  // Records are public in the issues, so the cheapest forgery is submitting
  // someone else's verbatim.
  if (board.entries.some((entry) => entry.trackHash === trackHash)) {
    reject("this exact record is already on the board");
  }
}

/**
 * Showing a GitHub account on the board is optional, but never self-asserted:
 * a handle is only kept when it is the account that opened the issue, so a
 * record cannot arrive wearing somebody else's name.
 */
function declaredHandle(raw, author) {
  const declared = sanitiseHandle(raw);
  if (declared === null) return null;
  if (author === null) reject("a GitHub handle was declared with no issue to check it against");
  if (declared.toLowerCase() !== author.toLowerCase()) {
    reject(`the record claims @${declared} but the issue was opened by @${author}`);
  }
  return author;
}

/**
 * The rate limit has to count every submitter, including the ones who chose
 * not to show a handle, so it keys on a hash of the account rather than on the
 * displayed one. The hash is a counting key and not a secret — the issue it
 * came from is public — but it keeps plaintext accounts out of a file that
 * exists to be read.
 */
function submitterKey(author) {
  return author === null
    ? null
    : createHash("sha256").update(author.toLowerCase()).digest("hex").slice(0, 16);
}

function checkRateLimit(board, key) {
  if (key === null) return;
  const dayAgo = Date.now() - 24 * 60 * 60 * 1000;
  const today = board.entries.filter((entry) => (
    entry.submitter === key && Date.parse(entry.verifiedAt ?? 0) > dayAgo
  ));
  if (today.length >= DAILY_SUBMISSION_LIMIT) {
    reject(`this account has already landed ${today.length} records today`);
  }
}

// ----------------------------------------------------------------- verifying

async function verify({ body, author, issue }) {
  const submission = extractSubmission(body);
  const config = validateShape(submission);

  const engine = loadEngine();
  if (submission.engine !== engine.stamp) {
    reject(`recorded against engine ${submission.engine}, but this repo ships ${engine.stamp}`);
  }
  const { policy, stamp: policyStamp } = loadPolicy();
  if (submission.policy !== policyStamp) {
    reject(`recorded against policy ${submission.policy}, but this repo ships ${policyStamp}`);
  }

  const board = loadBoard();
  const submitter = submitterKey(author);
  checkRateLimit(board, submitter);

  // Bounded and shape-checked before a single frame is stepped.
  const session = await unpackSession(submission.track);
  if (session.frames.length === 0) reject("the track has no frames");

  // Fingerprinted from the DECODED, re-encoded session plus the settings it
  // was played under — never from the submitted base64 itself. Compression
  // has slack (padding and block-size choices) that lets many different
  // base64 strings decode to the identical replay;
  // hashing the raw string would let each of those bypass the duplicate check
  // under a different name.
  const canonical = encodeSession(session);
  const identity = Buffer.concat([
    Buffer.from(JSON.stringify([
      config.seed, config.opponent, config.delayFrames,
      Math.round(config.openingDelaySeconds * FPS),
    ])),
    Buffer.from(canonical.buffer, canonical.byteOffset, canonical.byteLength),
  ]);
  const trackHash = createHash("sha256").update(identity).digest("hex").slice(0, 32);
  checkNotADuplicate(board, trackHash);

  const { instance } = await WebAssembly.instantiate(engine.bytes, {});
  const { winners, suspect } = replaySession({
    wasm: instance.exports, policy, config, session,
  });
  if (suspect.length > 0) {
    const worst = suspect.reduce((a, b) => (a.gap > b.gap ? a : b));
    reject(`${suspect.length} opponent actions do not match the policy `
      + `(worst at frame ${worst.frame}: recorded ${worst.recorded}, `
      + `expected ${worst.expected}, ${worst.gap.toFixed(3)} below the best logit)`);
  }

  const scored = longestShutout(winners);
  if (scored.best < MIN_SUBMITTABLE_SHUTOUT) {
    reject(`the replay scores ${scored.best}; the board starts at ${MIN_SUBMITTABLE_SHUTOUT}`);
  }
  if (submission.claim !== scored.best) {
    // Not fatal. The claim is decoration; the replay is the record.
    process.stderr.write(`claimed ${submission.claim}, replayed ${scored.best}\n`);
  }

  return {
    board,
    entry: {
      name: sanitiseName(submission.name),
      github: declaredHandle(submission.github, author),
      submitter,
      board: BOARDS[config.opponent],
      score: scored.best,
      rounds: winners.length,
      firstRound: scored.firstRound,
      seed: config.seed,
      frames: session.frames.length,
      issue: issue ?? null,
      trackHash,
      verifiedAt: new Date().toISOString(),
    },
  };
}

// ---------------------------------------------------------------------- main

const fileArgument = process.argv.indexOf("--file");
const body = fileArgument === -1
  ? process.env.ISSUE_BODY
  : fs.readFileSync(process.argv[fileArgument + 1], "utf8");
const author = process.env.ISSUE_AUTHOR || null;
const issue = process.env.ISSUE_NUMBER ? Number(process.env.ISSUE_NUMBER) : null;
const write = process.argv.includes("--write");

let verdict;
try {
  const { board, entry } = await verify({ body, author, issue });
  if (write) {
    board.entries.push(entry);
    board.entries.sort((a, b) => b.score - a.score || Date.parse(a.verifiedAt) - Date.parse(b.verifiedAt));
    board.updated = new Date().toISOString();
    fs.writeFileSync(BOARD_PATH, `${JSON.stringify(board, null, 2)}\n`);
  }
  verdict = { ok: true, entry };
} catch (error) {
  if (!(error instanceof RejectedSubmission)) throw error;
  verdict = { ok: false, reason: error.message };
}

// The workflow reads these files rather than parsing stdout, so nothing the
// submission controls can shape a shell command downstream.
fs.writeFileSync("verdict.json", `${JSON.stringify(verdict, null, 2)}\n`);

/** Rejection reasons quote values the submitter chose. Backticks are the only
 *  character that could break out of the fence they are shown in. */
const fenced = (text) => `\`\`\`\n${text.replaceAll("`", "'")}\n\`\`\`\n`;
fs.writeFileSync("comment.md", verdict.ok
  ? `**Verified.** A ${verdict.entry.score}-round shutout against `
    + `${verdict.entry.board === "hybrid" ? "Hybrid" : "Killfield"}, replayed over `
    + `${verdict.entry.rounds} rounds and ${verdict.entry.frames} frames.\n\n`
    + "It is on the board now. The score above is the replay's, not the one "
    + "the record claimed.\n"
  : "**Not verified.** Nothing went on the board.\n\n"
    + fenced(verdict.reason)
    + "\nIf you think this is wrong, leave a comment — the record is still here "
    + "and can be replayed again.\n");

process.stdout.write(`${verdict.ok ? "accepted" : "rejected"}: `
  + `${verdict.ok ? `${verdict.entry.score} on ${verdict.entry.board}` : verdict.reason}\n`);
process.exitCode = verdict.ok ? 0 : 1;
