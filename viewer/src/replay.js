/**
 * Ranked-session replay codec, shared by the recorder in the browser and the
 * verifier in CI. Both sides import this one file on purpose: a format that
 * drifts between them silently invalidates every record on the board.
 *
 * What a session is: one continuous `kf_new(seed)` handle played in Play mode,
 * human on tank 1. The engine runs round after round off a single RNG chain,
 * so the whole session replays bit-exactly from the seed plus the inputs the
 * human actually produced — see the determinism probe in docs/LEADERBOARD.md.
 * Pausing produces no frames at all (the loop simply stops calling kf_step),
 * so it needs no representation here.
 *
 * The score is the longest run of consecutive rounds the human won anywhere in
 * the session. The verifier recomputes it from its own replay and ignores
 * whatever the submission claimed.
 *
 * Everything in LIMITS is a rejection boundary, not a hint. The submission
 * path is reachable by anyone who can open an issue, so each field is bounded
 * before it reaches the engine: a NaN strength propagates into tank
 * coordinates and can leave a round that never ends, and a few KB of base64
 * can inflate into gigabytes.
 */

export const HUMAN_SEAT = 1;
export const DRAW = 2;
export const ACTION_COUNT = 18;
/** No opponent action to replay: Killfield and Laika are driven inside kf_step. */
export const NO_ACTION = 255;

/**
 * How many rounds running a player has to take before a run is worth a place
 * on the board. The page enforces this to keep players from wasting a
 * submission, and the verifier enforces it again because the page's copy of
 * the rule is advisory — a submission is just text, and anyone can write it.
 */
export const MIN_SUBMITTABLE_SHUTOUT = 3;

export const LIMITS = {
  /** ~200 rounds at the ~165 frames/round this engine averages, with headroom. */
  maxFrames: 60_000,
  maxRounds: 200,
  maxEvents: 20_000,
  /** A GitHub issue body holds 65,536 characters; leave room for the prose. */
  maxBase64: 60_000,
  /** Hard ceiling on what the inflater may produce, checked as it streams. */
  maxInflatedBytes: 4 << 20,
  maxNameLength: 24,
};

const MAGIC = 0x3152464b; // "KFR1"
const HEADER_BYTES = 16;
const FRAME_BYTES = 18;
const EVENT_BYTES = 6;
export const EVENT_FIRE = 0;

// ------------------------------------------------------------------ binary

/**
 * @param {{frames: Array, events: Array}} session
 * @returns {Uint8Array}
 */
export function encodeSession({ frames, events }) {
  const bytes = new Uint8Array(
    HEADER_BYTES + frames.length * FRAME_BYTES + events.length * EVENT_BYTES,
  );
  const view = new DataView(bytes.buffer);
  view.setUint32(0, MAGIC, true);
  view.setUint32(4, frames.length, true);
  view.setUint32(8, events.length, true);
  view.setUint32(12, 0, true);

  let offset = HEADER_BYTES;
  for (const frame of frames) {
    view.setFloat32(offset, frame.forward, true);
    view.setFloat32(offset + 4, frame.backup, true);
    view.setFloat32(offset + 8, frame.turnLeft, true);
    view.setFloat32(offset + 12, frame.turnRight, true);
    view.setUint8(offset + 16, frame.fire ? 1 : 0);
    view.setUint8(offset + 17, frame.action);
    offset += FRAME_BYTES;
  }
  for (const event of events) {
    view.setUint32(offset, event.frame, true);
    view.setUint8(offset + 4, event.kind);
    view.setUint8(offset + 5, event.value);
    offset += EVENT_BYTES;
  }
  return bytes;
}

class RejectedSubmission extends Error {}

/** Throws RejectedSubmission on anything malformed; never returns partial data. */
export function decodeSession(bytes) {
  const reject = (why) => { throw new RejectedSubmission(why); };
  if (bytes.length < HEADER_BYTES) reject("track is shorter than its header");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(0, true) !== MAGIC) reject("not a Killfield ranked track");

  const frameCount = view.getUint32(4, true);
  const eventCount = view.getUint32(8, true);
  if (frameCount > LIMITS.maxFrames) reject(`too many frames (${frameCount})`);
  if (eventCount > LIMITS.maxEvents) reject(`too many events (${eventCount})`);
  const expected = HEADER_BYTES + frameCount * FRAME_BYTES + eventCount * EVENT_BYTES;
  if (bytes.length !== expected) {
    reject(`track length ${bytes.length} does not match its header's ${expected}`);
  }

  const frames = new Array(frameCount);
  let offset = HEADER_BYTES;
  for (let i = 0; i < frameCount; i += 1) {
    const frame = {
      forward: view.getFloat32(offset, true),
      backup: view.getFloat32(offset + 4, true),
      turnLeft: view.getFloat32(offset + 8, true),
      turnRight: view.getFloat32(offset + 12, true),
      fire: view.getUint8(offset + 16),
      action: view.getUint8(offset + 17),
    };
    for (const axis of ["forward", "backup", "turnLeft", "turnRight"]) {
      const value = frame[axis];
      // NaN fails every comparison, so this rejects it too.
      if (!(value >= 0 && value <= 1)) reject(`frame ${i}: ${axis} is ${value}`);
    }
    if (frame.fire > 1) reject(`frame ${i}: fire is ${frame.fire}`);
    if (frame.action >= ACTION_COUNT && frame.action !== NO_ACTION) {
      reject(`frame ${i}: action ${frame.action} is outside Discrete(18)`);
    }
    frames[i] = frame;
    offset += FRAME_BYTES;
  }

  const events = new Array(eventCount);
  let previousFrame = 0;
  for (let i = 0; i < eventCount; i += 1) {
    const event = {
      frame: view.getUint32(offset, true),
      kind: view.getUint8(offset + 4),
      value: view.getUint8(offset + 5),
    };
    if (event.frame > frameCount) reject(`event ${i} lands past the end of the track`);
    if (event.frame < previousFrame) reject(`event ${i} is out of order`);
    if (event.kind !== EVENT_FIRE) reject(`event ${i} has unknown kind ${event.kind}`);
    if (event.value > 1) reject(`event ${i} has value ${event.value}`);
    previousFrame = event.frame;
    events[i] = event;
    offset += EVENT_BYTES;
  }
  return { frames, events };
}

// ------------------------------------------------------------- compression

function toBase64(bytes) {
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

export async function packSession(session) {
  const raw = encodeSession(session);
  const stream = new Blob([raw]).stream()
    .pipeThrough(new CompressionStream("deflate-raw"));
  return toBase64(new Uint8Array(await new Response(stream).arrayBuffer()));
}

/**
 * Inflate a submitted payload, refusing to materialise more than
 * `LIMITS.maxInflatedBytes`. The cap is enforced chunk by chunk as the stream
 * produces them, because by the time a bomb has fully decompressed the damage
 * is already done.
 */
export async function unpackSession(base64) {
  const reject = (why) => { throw new RejectedSubmission(why); };
  const text = base64.trim();
  if (text.length > LIMITS.maxBase64) reject(`payload is ${text.length} characters`);
  if (!/^[A-Za-z0-9+/=]+$/.test(text)) reject("payload is not base64");

  let packed;
  try {
    const binary = atob(text);
    packed = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) packed[i] = binary.charCodeAt(i);
  } catch {
    reject("payload is not valid base64");
  }

  const reader = new Blob([packed]).stream()
    .pipeThrough(new DecompressionStream("deflate-raw"))
    .getReader();
  const chunks = [];
  let total = 0;
  for (;;) {
    let chunk;
    try {
      chunk = await reader.read();
    } catch {
      reject("payload is not valid deflate-raw data");
    }
    if (chunk.done) break;
    total += chunk.value.length;
    if (total > LIMITS.maxInflatedBytes) {
      await reader.cancel();
      reject("payload inflates past the size limit");
    }
    chunks.push(chunk.value);
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  return decodeSession(bytes);
}

// ------------------------------------------------------------------ scoring

/**
 * The longest run of consecutive rounds `seat` won. A draw or double kill
 * scores for nobody and breaks nobody's run.
 *
 * @param {number[]} winners one entry per finished round, from the engine
 */
export function longestShutout(winners, seat = HUMAN_SEAT) {
  let best = 0;
  let run = 0;
  // A draw inside a run keeps the run alive without scoring, so the span is
  // wider than the win count and the start has to be carried, not derived.
  let runStart = 0;
  let firstRound = 0;
  let lastRound = -1;
  winners.forEach((winner, round) => {
    if (winner === seat) {
      if (run === 0) runStart = round;
      run += 1;
      if (run > best) { best = run; firstRound = runStart; lastRound = round; }
    } else if (winner !== DRAW) {
      run = 0;
    }
  });
  return { best, firstRound, lastRound };
}

// --------------------------------------------------------------------- name

/**
 * A display name is attacker-controlled text that ends up in a committed JSON
 * file and on a public page. Strip the classes of character that exist to
 * impersonate rather than to name: control characters, zero-width joiners and
 * the bidirectional overrides that let one string render as another.
 */
export function sanitiseName(raw) {
  if (typeof raw !== "string") return "";
  const cleaned = raw
    .normalize("NFC")
    // eslint-disable-next-line no-control-regex
    .replace(/[ --]/g, "")
    .replace(/[​-‏‪-‮⁦-⁩﻿]/g, "")
    .trim()
    .slice(0, LIMITS.maxNameLength)
    .trim();
  return cleaned;
}

/**
 * A GitHub handle is optional on the board, but it is never taken on trust:
 * the verifier only keeps it when it matches the account that opened the
 * issue, so claiming someone else's is refused rather than displayed.
 */
export function sanitiseHandle(raw) {
  if (typeof raw !== "string") return null;
  const cleaned = raw.trim().replace(/^@/, "");
  if (cleaned === "") return null;
  // GitHub's own rule: alphanumerics and single hyphens, 39 characters.
  return /^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$/.test(cleaned) ? cleaned : null;
}

export { RejectedSubmission };
