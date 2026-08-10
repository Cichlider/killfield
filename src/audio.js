/**
 * Small Web Audio soundboard for simulation events.
 *
 * Browsers do not allow a page to start audio until the user interacts with
 * it, so unlock() is deliberately separate from playEvent(). The main module
 * calls it from pointer/keyboard handlers and game events remain synchronous.
 */

const FILES = {
  fire: [new URL("../assets/audio/fire.wav", import.meta.url)],
  bounce: [
    new URL("../assets/audio/bounce-1.wav", import.meta.url),
    new URL("../assets/audio/bounce-2.wav", import.meta.url),
  ],
  destroy: [
    new URL("../assets/audio/destroy.wav", import.meta.url),
    new URL("../assets/audio/destroy-2.wav", import.meta.url),
    new URL("../assets/audio/destroy-3.wav", import.meta.url),
  ],
  expire: [new URL("../assets/audio/expire.wav", import.meta.url)],
};

const GAINS = {
  fire: 0.48,
  bounce: 0.16,
  // The three destruction samples are deliberately layered. Keep each layer
  // below the single-effect gain so their sum retains headroom.
  destroy: 0.34,
  expire: 0.35,
};

// A ricochet can touch more than one wall probe in a single logic step. Treat
// that as one audible impact instead of stacking several sharp transients.
const BOUNCE_COOLDOWN_MS = 45;

export class SoundEffects {
  constructor({
    Context = globalThis.AudioContext || globalThis.webkitAudioContext,
    fetcher = (...args) => globalThis.fetch(...args),
  } = {}) {
    this.Context = Context;
    this.fetcher = fetcher;
    this.context = null;
    this.buffers = new Map();
    this.loadPromise = null;
    this.enabled = true;
    this.nextVariant = new Map();
    this.lastBounceAt = -Infinity;
  }

  /** Start the audio context from a user gesture and begin decoding assets. */
  unlock() {
    if (!this.Context) return;
    if (this.context === null) {
      this.context = new this.Context();
      this.loadPromise = this.load().catch(() => {
        // Audio is enhancement-only. A missing/unsupported asset must never
        // interrupt the fixed-step game loop.
      });
    }
    if (this.context.state === "suspended") this.context.resume().catch(() => {});
  }

  async load() {
    const entries = Object.entries(FILES).flatMap(([kind, urls]) =>
      urls.map(async (url, index) => {
        const response = await this.fetcher(url);
        if (!response.ok) throw new Error(`Could not load sound: ${url}`);
        const data = await response.arrayBuffer();
        const buffer = await this.context.decodeAudioData(data);
        this.buffers.set(`${kind}:${index}`, buffer);
      }));
    await Promise.all(entries);
  }

  setEnabled(enabled) {
    this.enabled = !!enabled;
  }

  playEvent(event, now = performance.now()) {
    const kind = event[0];
    if (kind !== "fire" && kind !== "bounce" && kind !== "destroy" && kind !== "expire") return;
    if (kind === "bounce") {
      if (now - this.lastBounceAt < BOUNCE_COOLDOWN_MS) return;
      this.lastBounceAt = now;
    }
    this.play(kind);
  }

  play(kind) {
    if (!this.enabled || this.context === null || this.context.state !== "running") return;
    const variants = FILES[kind];
    if (kind === "destroy") {
      // Tank Trouble's destruction is a composite effect, not a choice among
      // three alternatives. Starting them on the same audio clock keeps the
      // attack aligned even when several simulation events arrive together.
      const when = this.context.currentTime;
      for (let index = 0; index < variants.length; index++) {
        this.playBuffer(kind, index, when);
      }
      return;
    }
    const index = this.nextVariant.get(kind) || 0;
    this.nextVariant.set(kind, (index + 1) % variants.length);
    this.playBuffer(kind, index);
  }

  playBuffer(kind, index, when = 0) {
    const buffer = this.buffers.get(`${kind}:${index}`);
    if (!buffer) return;

    const source = this.context.createBufferSource();
    const gain = this.context.createGain();
    source.buffer = buffer;
    gain.gain.value = GAINS[kind];
    source.connect(gain);
    gain.connect(this.context.destination);
    source.start(when);
  }
}
