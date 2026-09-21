/**
 * Turning a finished ranked session into something a stranger can verify.
 *
 * The submission carries no score. It carries the seed, the settings and the
 * inputs, and CI derives the score by replaying them — so there is nothing in
 * here worth forging except the inputs themselves, which is the thing the
 * replay checks. The claimed figure travels only so the player can be told
 * when the two disagree.
 */

import { LIMITS, packSession, sanitiseHandle, sanitiseName } from "./replay.js";

export const SUBMISSION_VERSION = 1;
const REPO = "Cichlider/killfield";

/** Short content hashes of the two binaries a replay is only valid against.
 *  SubtleCrypto needs a secure context; over plain http the stamps come back
 *  empty and buildSubmission refuses rather than shipping an unverifiable
 *  record. */
export async function buildStamps(engineBytes, policyWeights) {
  if (!globalThis.crypto?.subtle) return { engine: "", policy: "" };
  const digest = async (bytes) => {
    const hash = await crypto.subtle.digest("SHA-256", bytes);
    return [...new Uint8Array(hash)].slice(0, 8)
      .map((b) => b.toString(16).padStart(2, "0")).join("");
  };
  return {
    engine: await digest(engineBytes),
    policy: await digest(new Uint8Array(
      policyWeights.buffer, policyWeights.byteOffset, policyWeights.byteLength,
    )),
  };
}

export async function buildSubmission({ result, name, github, stamps, enforceUploadLimit = true }) {
  if (!stamps.engine || !stamps.policy) {
    throw new Error("This page must be served over https to sign a record.");
  }
  const player = sanitiseName(name);
  if (player === "") throw new Error("A record needs a name to go on the board.");
  const handle = sanitiseHandle(github);
  const track = await packSession(result.recorder.session());
  if (enforceUploadLimit && track.length > LIMITS.maxBase64) {
    throw new Error("This session is too large for automatic submission. Download the replay and send the JSON file to the maintainer.");
  }
  return {
    v: SUBMISSION_VERSION,
    name: player,
    github: handle,
    seed: result.config.seed,
    opponent: result.config.opponent,
    delayFrames: result.config.delayFrames,
    openingDelaySeconds: result.config.openingDelaySeconds,
    engine: stamps.engine,
    policy: stamps.policy,
    claim: result.stats.wins,
    rounds: result.winners.length,
    frames: result.frames,
    startedAt: result.startedAt,
    endedAt: result.endedAt,
    track,
  };
}

/**
 * What the player pastes: the record and nothing else.
 *
 * The issue form wraps this in a ```json fence itself, so the clipboard must
 * not carry one — a fence inside a fence is what CI would try to parse.
 */
export function submissionPayload(submission) {
  return JSON.stringify(submission);
}

/** The issue body CI ends up parsing, as the form renders it. Prose around the
 *  fence is for humans and is never interpreted. */
export function submissionBody(submission) {
  return `### Record\n\n\`\`\`json\n${submissionPayload(submission)}\n\`\`\`\n`;
}

export function submissionTitle(submission) {
  const label = { hybrid: "Hybrid", laika: "Laika", killfield: "Killfield" }[submission.opponent]
    ?? submission.opponent;
  // Keep player-controlled text out of the title: a name containing @handle
  // must not generate a mention when the player submits the fallback Issue.
  return `[wins] ${submission.claim} vs ${label}`;
}

/** Submit without exposing a repository credential to the static page. */
export async function submitToGateway(endpoint, submission, turnstileToken) {
  if (!endpoint) throw new Error("Score submission is not configured yet.");
  let response;
  try {
    response = await fetch(endpoint, {
      method: "POST",
      // text/plain is CORS-safelisted, so browsers can send the record in one
      // request instead of relying on an OPTIONS preflight that some networks
      // and privacy filters block. The Worker still parses the JSON body and
      // enforces the exact Origin plus Turnstile verification.
      headers: { "content-type": "text/plain;charset=UTF-8" },
      body: JSON.stringify({ submission, turnstileToken }),
    });
  } catch (cause) {
    const error = new Error("Could not reach the score submission service.", { cause });
    error.code = "SUBMISSION_NETWORK_ERROR";
    throw error;
  }
  let result;
  try { result = await response.json(); } catch { result = null; }
  if (!response.ok || result?.ok !== true) {
    throw new Error(result?.error ?? "Score submission failed. Try again shortly.");
  }
  return result;
}

/**
 * Hand the record to GitHub. The payload goes to the clipboard rather than
 * into the URL: a long session's track runs to tens of thousands of characters
 * and would be truncated or refused as a query string.
 */
export async function openSubmissionIssue(submission) {
  if (submission.track?.length > LIMITS.maxIssueBase64) {
    const error = new Error("This replay needs the one-click submission service because it is too large for one GitHub issue.");
    error.code = "SUBMISSION_REQUIRES_GATEWAY";
    throw error;
  }
  const body = submissionPayload(submission);
  let copied = false;
  try {
    await navigator.clipboard.writeText(body);
    copied = true;
  } catch {
    // Clipboard access needs a permission this browser withheld; the caller
    // falls back to showing the text for a manual copy.
  }
  const url = new URL(`https://github.com/${REPO}/issues/new`);
  url.searchParams.set("template", "leaderboard.yml");
  url.searchParams.set("title", submissionTitle(submission));
  return { url: url.href, body, copied };
}
