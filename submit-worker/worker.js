const MAX_BODY_BYTES = 90_000;
const MAX_TRACK_CHARS = 60_000;
const MAX_FRAMES = 60_000;
const MAX_ROUNDS = 200;
const DEFAULT_DAILY_LIMIT = 10;

// Keep this cheap edge validation aligned with viewer/src/replay.js. The
// authoritative verifier still repeats every check before replaying anything.
function sanitiseName(raw) {
  if (typeof raw !== "string") return "";
  return raw.normalize("NFC")
    // eslint-disable-next-line no-control-regex
    .replace(/[\u0000-\u001f\u007f-\u009f]/g, "")
    .replace(/[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]/g, "")
    .trim().slice(0, 24).trim();
}

function sanitiseHandle(raw) {
  if (typeof raw !== "string") return null;
  const cleaned = raw.trim().replace(/^@/, "");
  if (cleaned === "") return null;
  return /^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$/.test(cleaned)
    ? cleaned : null;
}

const json = (body, status, origin = null) => new Response(JSON.stringify(body), {
  status,
  headers: {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
    ...(origin ? { "access-control-allow-origin": origin, vary: "Origin" } : {}),
  },
});

function corsOrigin(request, env) {
  const origin = request.headers.get("origin");
  return origin === env.ALLOWED_ORIGIN ? origin : null;
}

function fail(message, status = 400) {
  const error = new Error(message);
  error.status = status;
  throw error;
}

function cleanSubmission(value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    fail("The record is not an object.");
  }
  const name = sanitiseName(value.name);
  if (name === "") fail("Enter a name of at most 24 characters.");
  const github = sanitiseHandle(value.github);
  if (value.github != null && String(value.github).trim() !== "" && github === null) {
    fail("The optional GitHub handle is not valid.");
  }
  if (value.v !== 1 || !["hybrid", "laika", "killfield"].includes(value.opponent)) {
    fail("The record version or opponent is not supported.");
  }
  if (!Number.isInteger(value.seed) || value.seed < 0 || value.seed > 0xffffffff) {
    fail("The record seed is invalid.");
  }
  if (value.delayFrames !== 0 || !Number.isFinite(value.openingDelaySeconds)
      || value.openingDelaySeconds < 0 || value.openingDelaySeconds > 0.5) {
    fail("The record does not use ranked settings.");
  }
  if (!Number.isInteger(value.claim) || value.claim < 1 || value.claim > MAX_ROUNDS
      || !Number.isInteger(value.rounds) || value.rounds < 1 || value.rounds > MAX_ROUNDS
      || value.claim > value.rounds
      || !Number.isInteger(value.frames) || value.frames < 1 || value.frames > MAX_FRAMES) {
    fail("The record dimensions are invalid.");
  }
  if (typeof value.track !== "string" || value.track.length < 1
      || value.track.length > MAX_TRACK_CHARS || !/^[A-Za-z0-9+/=]+$/.test(value.track)) {
    fail("The replay payload is invalid.");
  }
  if (!/^[a-f0-9]{16}$/.test(value.engine) || !/^[a-f0-9]{16}$/.test(value.policy)) {
    fail("The record does not identify a supported build.");
  }
  // Forward only the versioned record schema. Besides keeping Issues small and
  // predictable, this prevents extra attacker-controlled fields from looking
  // like gateway metadata to downstream tooling.
  const record = {
    v: value.v,
    name,
    github,
    seed: value.seed,
    opponent: value.opponent,
    delayFrames: value.delayFrames,
    openingDelaySeconds: value.openingDelaySeconds,
    engine: value.engine,
    policy: value.policy,
    claim: value.claim,
    rounds: value.rounds,
    frames: value.frames,
    startedAt: value.startedAt,
    endedAt: value.endedAt,
    track: value.track,
  };
  const encoded = JSON.stringify(record);
  if (encoded.length > 80_000) fail("The record is too large.", 413);
  return { value: record, encoded, name };
}

async function turnstilePasses(token, ip, env) {
  if (typeof token !== "string" || token.length < 1 || token.length > 2_048) return false;
  const form = new FormData();
  form.set("secret", env.TURNSTILE_SECRET);
  form.set("response", token);
  form.set("remoteip", ip);
  const response = await fetch("https://challenges.cloudflare.com/turnstile/v0/siteverify", {
    method: "POST",
    body: form,
  });
  if (!response.ok) return false;
  const verdict = await response.json();
  return verdict.success === true
    && verdict.hostname === env.TURNSTILE_HOSTNAME
    && verdict.action === "leaderboard-submit";
}

async function opaqueClientKey(ip, env) {
  const bytes = new TextEncoder().encode(`${env.RATE_LIMIT_SALT}\0${ip}`);
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest.subarray(0, 8)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function createIssue(record, encoded, submitter, env) {
  const board = { hybrid: "Hybrid", laika: "Laika", killfield: "Killfield" }[record.opponent];
  // Do not place the player-controlled name in the title: an @handle there can
  // generate unwanted mention notifications. The name stays inside JSON code
  // fencing and is rendered as text on the board.
  const title = `[wins] ${record.claim} vs ${board}`;
  const body = `### Record\n\n\`\`\`json\n${encoded}\n\`\`\`\n\n`
    + `<!-- killfield-gateway:v1:${submitter} -->\n`;
  const response = await fetch(`https://api.github.com/repos/${env.GITHUB_REPOSITORY}/issues`, {
    method: "POST",
    headers: {
      accept: "application/vnd.github+json",
      authorization: `Bearer ${env.GITHUB_TOKEN}`,
      "content-type": "application/json",
      "user-agent": "killfield-leaderboard-gateway",
      "x-github-api-version": "2022-11-28",
    },
    body: JSON.stringify({ title, body, labels: ["leaderboard"] }),
  });
  if (!response.ok) {
    console.error("GitHub issue creation failed", response.status, await response.text());
    fail("The verifier could not be started. Try again shortly.", 502);
  }
  return response.json();
}

export default {
  async fetch(request, env) {
    const origin = corsOrigin(request, env);
    if (request.method === "OPTIONS") {
      if (!origin) return new Response(null, { status: 403 });
      return new Response(null, {
        status: 204,
        headers: {
          "access-control-allow-origin": origin,
          "access-control-allow-methods": "POST",
          "access-control-allow-headers": "content-type",
          "access-control-max-age": "86400",
          vary: "Origin",
        },
      });
    }
    if (request.method !== "POST" || new URL(request.url).pathname !== "/submit") {
      return json({ ok: false, error: "Not found." }, 404, origin);
    }
    if (!origin) return json({ ok: false, error: "This origin may not submit records." }, 403);

    try {
      const declaredLength = Number(request.headers.get("content-length") ?? 0);
      if (declaredLength > MAX_BODY_BYTES) fail("The request is too large.", 413);
      const text = await request.text();
      if (new TextEncoder().encode(text).length > MAX_BODY_BYTES) {
        fail("The request is too large.", 413);
      }
      let payload;
      try { payload = JSON.parse(text); } catch { fail("The request is not valid JSON."); }
      const { value: record, encoded } = cleanSubmission(payload?.submission);
      const ip = request.headers.get("cf-connecting-ip");
      if (!ip) fail("The client address is unavailable.", 400);
      if (!await turnstilePasses(payload?.turnstileToken, ip, env)) {
        fail("Human verification failed. Please try again.", 403);
      }

      const submitter = await opaqueClientKey(ip, env);
      const limit = Number(env.DAILY_LIMIT ?? DEFAULT_DAILY_LIMIT);
      // A fixed 24-hour window is closer to the verifier's rolling-day check
      // than a UTC calendar key (which allowed two full quotas around midnight).
      const key = `daily:${submitter}`;
      const used = Number(await env.RATE_LIMITS.get(key) ?? 0);
      if (!Number.isFinite(used) || used >= limit) {
        fail("This connection has reached today's submission limit.", 429);
      }

      // Reserve the attempt before the irreversible side effect. If GitHub is
      // temporarily unavailable this may consume one attempt, but it cannot
      // tell the browser "failed" after creating an Issue and cause a retry to
      // create a duplicate.
      await env.RATE_LIMITS.put(key, String(used + 1), { expirationTtl: 86_400 });
      const issue = await createIssue(record, encoded, submitter, env);
      return json({ ok: true, issue: issue.number, url: issue.html_url }, 202, origin);
    } catch (error) {
      const status = error.status ?? 500;
      if (status === 500) console.error("Unexpected submission failure", error);
      return json({
        ok: false,
        error: status === 500
          ? "Score submission failed. Try again shortly."
          : (error.message ?? "Submission failed."),
      }, status, origin);
    }
  },
};

export { cleanSubmission };
