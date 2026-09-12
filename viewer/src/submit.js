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

export async function buildSubmission({ result, name, github, stamps }) {
  if (!stamps.engine || !stamps.policy) {
    throw new Error("This page must be served over https to sign a record.");
  }
  const player = sanitiseName(name);
  if (player === "") throw new Error("A record needs a name to go on the board.");
  const handle = sanitiseHandle(github);
  const track = await packSession(result.recorder.session());
  if (track.length > LIMITS.maxBase64) {
    throw new Error("This session is too long to submit in one issue.");
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
    claim: result.best,
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
  const label = submission.opponent === "hybrid" ? "Hybrid" : "Killfield";
  return `[score] ${submission.name} — ${submission.claim} vs ${label}`;
}

/** Submit without exposing a repository credential to the static page. */
export async function submitToGateway(endpoint, submission, turnstileToken) {
  if (!endpoint) throw new Error("Score submission is not configured yet.");
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ submission, turnstileToken }),
  });
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
  url.searchParams.set("labels", "leaderboard");
  url.searchParams.set("title", submissionTitle(submission));
  return { url: url.href, body, copied };
}
