import {
  AudioBufferSource,
  BufferTarget,
  CanvasSource,
  Mp4OutputFormat,
  Output,
  StreamTarget,
  canEncodeAudio,
  canEncodeVideo,
} from "../vendor/mediabunny-1.61.0.min.mjs";

export const VIDEO_PRESETS = Object.freeze({
  recommended: Object.freeze({ width: 1280, height: 720, fps: 60, bitrate: 4_000_000 }),
  hd: Object.freeze({ width: 1920, height: 1080, fps: 60, bitrate: 8_000_000 }),
  ultra: Object.freeze({ width: 1920, height: 1080, fps: 120, bitrate: 12_000_000 }),
});

export const DIRECT_DOWNLOAD_MAX_BYTES = 512 * 1024 * 1024;
export const AUDIO_SAMPLE_RATE = 44_100;
export const AUDIO_BITRATE = 128_000;

const SOUND_URLS = Object.freeze({
  fire: [new URL("../assets/audio/fire.wav", import.meta.url)],
  destroy: [
    new URL("../assets/audio/destroy.wav", import.meta.url),
    new URL("../assets/audio/destroy-2.wav", import.meta.url),
    new URL("../assets/audio/destroy-3.wav", import.meta.url),
  ],
  expire: [new URL("../assets/audio/expire.wav", import.meta.url)],
});

const SOUND_GAINS = Object.freeze({ fire: 0.48, destroy: 0.34, expire: 0.35 });

export function estimatedBytes(seconds, bitrate) {
  return Math.ceil(Math.max(0, seconds) * Math.max(0, bitrate) / 8);
}

export function safeVideoFilename(name) {
  const stem = String(name || "replay")
    .normalize("NFKC")
    .replace(/[\\/:*?"<>|\u0000-\u001f]/g, "-")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^[.-]+|[.-]+$/g, "")
    .slice(0, 80) || "replay";
  return `killfield-${stem}.mp4`;
}

export async function supportsVideoPreset(preset, withAudio = false) {
  if (typeof globalThis.VideoEncoder === "undefined" || typeof globalThis.VideoFrame === "undefined") {
    return false;
  }
  const videoSupported = await canEncodeVideo("avc", {
    width: preset.width,
    height: preset.height,
    bitrate: preset.bitrate,
    frameRate: preset.fps,
  });
  if (!videoSupported || !withAudio) return videoSupported;
  if (typeof globalThis.AudioEncoder === "undefined" || typeof globalThis.AudioData === "undefined") {
    return false;
  }
  return canEncodeAudio("aac", {
    numberOfChannels: 1,
    sampleRate: AUDIO_SAMPLE_RATE,
    bitrate: AUDIO_BITRATE,
  });
}

/** Decode the PCM16 WAV assets without creating a live AudioContext. */
export function decodePcm16Wav(buffer) {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = (offset) => String.fromCharCode(...bytes.subarray(offset, offset + 4));
  if (bytes.byteLength < 44 || tag(0) !== "RIFF" || tag(8) !== "WAVE") {
    throw new Error("Unsupported sound asset");
  }

  let channels = 0;
  let sampleRate = 0;
  let bitsPerSample = 0;
  let format = 0;
  let dataOffset = -1;
  let dataLength = 0;
  for (let offset = 12; offset + 8 <= bytes.byteLength;) {
    const id = tag(offset);
    const length = view.getUint32(offset + 4, true);
    const payload = offset + 8;
    if (payload + length > bytes.byteLength) throw new Error("Truncated sound asset");
    if (id === "fmt ") {
      format = view.getUint16(payload, true);
      channels = view.getUint16(payload + 2, true);
      sampleRate = view.getUint32(payload + 4, true);
      bitsPerSample = view.getUint16(payload + 14, true);
    } else if (id === "data") {
      dataOffset = payload;
      dataLength = length;
    }
    offset = payload + length + (length & 1);
  }
  if (format !== 1 || channels < 1 || bitsPerSample !== 16 || dataOffset < 0 || !sampleRate) {
    throw new Error("Only PCM16 WAV sounds are supported");
  }

  const frames = Math.floor(dataLength / (channels * 2));
  const samples = new Float32Array(frames);
  for (let frame = 0; frame < frames; frame += 1) {
    let sum = 0;
    for (let channel = 0; channel < channels; channel += 1) {
      sum += view.getInt16(dataOffset + (frame * channels + channel) * 2, true) / 32768;
    }
    samples[frame] = sum / channels;
  }
  return { sampleRate, samples };
}

export async function loadReplaySoundBank(fetcher = (...args) => globalThis.fetch(...args)) {
  const entries = await Promise.all(Object.entries(SOUND_URLS).map(async ([kind, urls]) => {
    const clips = await Promise.all(urls.map(async (url) => {
      const response = await fetcher(url);
      if (!response.ok) throw new Error(`Could not load sound: ${url}`);
      return decodePcm16Wav(await response.arrayBuffer());
    }));
    return [kind, clips];
  }));
  return Object.fromEntries(entries);
}

/** Mix one exact interval of the replay soundtrack into mono float PCM. */
export function mixReplaySoundChunk(bank, events, startSample, endSample,
  outputSampleRate = AUDIO_SAMPLE_RATE) {
  const output = new Float32Array(Math.max(0, endSample - startSample));
  const chunkStart = startSample / outputSampleRate;
  const chunkEnd = endSample / outputSampleRate;
  for (const event of events) {
    const clips = bank[event.kind];
    if (!clips) continue;
    for (const clip of clips) {
      const clipEnd = event.time + clip.samples.length / clip.sampleRate;
      if (event.time >= chunkEnd || clipEnd <= chunkStart) continue;
      const first = Math.max(0, Math.floor((event.time - chunkStart) * outputSampleRate));
      const last = Math.min(output.length, Math.ceil((clipEnd - chunkStart) * outputSampleRate));
      const gain = SOUND_GAINS[event.kind] ?? 1;
      for (let index = first; index < last; index += 1) {
        const time = chunkStart + index / outputSampleRate - event.time;
        const source = time * clip.sampleRate;
        if (source < 0 || source >= clip.samples.length) continue;
        const lo = Math.floor(source);
        const hi = Math.min(lo + 1, clip.samples.length - 1);
        const alpha = source - lo;
        output[index] += (clip.samples[lo] + (clip.samples[hi] - clip.samples[lo]) * alpha) * gain;
      }
    }
  }
  for (let index = 0; index < output.length; index += 1) {
    output[index] = Math.max(-1, Math.min(1, output[index]));
  }
  return output;
}

/**
 * Open the destination while the click still carries a user gesture. Chromium
 * writes directly to disk; other browsers fall back to an in-memory Blob.
 */
export async function chooseVideoDestination(filename, expectedBytes = 0) {
  if (expectedBytes <= DIRECT_DOWNLOAD_MAX_BYTES
      || typeof globalThis.showSaveFilePicker !== "function") {
    return { target: new BufferTarget(), stream: null, filename };
  }
  const handle = await globalThis.showSaveFilePicker({
    suggestedName: filename,
    types: [{ description: "MP4 video", accept: { "video/mp4": [".mp4"] } }],
  });
  const stream = await handle.createWritable();
  return {
    target: new StreamTarget(stream, { chunked: true, chunkSize: 4 * 1024 * 1024 }),
    stream,
    filename,
  };
}

export async function createVideoWriter(canvas, preset, destination, { withAudio = false } = {}) {
  const output = new Output({
    format: new Mp4OutputFormat({ fastStart: false }),
    target: destination.target,
  });
  const videoSource = new CanvasSource(canvas, {
    codec: "avc",
    bitrate: preset.bitrate,
    keyFrameInterval: 2,
    latencyMode: "quality",
  });
  output.addVideoTrack(videoSource, { frameRate: preset.fps });
  const audioSource = withAudio ? new AudioBufferSource({
    codec: "aac",
    bitrate: AUDIO_BITRATE,
  }) : null;
  if (audioSource) output.addAudioTrack(audioSource, { name: "Game sound" });
  await output.start();

  let finished = false;
  return {
    async addFrame(index) {
      const duration = 1 / preset.fps;
      await videoSource.add(index * duration, duration);
    },
    async addAudioChunk(samples) {
      if (!audioSource || samples.length === 0) return;
      const buffer = new AudioBuffer({
        length: samples.length,
        numberOfChannels: 1,
        sampleRate: AUDIO_SAMPLE_RATE,
      });
      buffer.copyToChannel(samples, 0);
      await audioSource.add(buffer);
    },
    async finish() {
      if (finished) return;
      finished = true;
      videoSource.close();
      audioSource?.close();
      await output.finalize();
      if (destination.stream) {
        return { bytes: null, url: null, filename: destination.filename };
      }
      const buffer = destination.target.buffer;
      if (!buffer) throw new Error("The MP4 encoder produced no output.");
      const url = URL.createObjectURL(new Blob([buffer], { type: "video/mp4" }));
      return { bytes: buffer.byteLength, url, filename: destination.filename };
    },
    async cancel() {
      if (finished) return;
      finished = true;
      await output.cancel();
    },
  };
}
