import {
  BufferTarget,
  CanvasSource,
  Mp4OutputFormat,
  Output,
  StreamTarget,
  canEncodeVideo,
} from "../vendor/mediabunny-1.61.0.min.mjs";

export const VIDEO_PRESETS = Object.freeze({
  recommended: Object.freeze({ width: 1280, height: 720, fps: 60, bitrate: 4_000_000 }),
  hd: Object.freeze({ width: 1920, height: 1080, fps: 60, bitrate: 8_000_000 }),
  ultra: Object.freeze({ width: 1920, height: 1080, fps: 120, bitrate: 12_000_000 }),
});

export const DIRECT_DOWNLOAD_MAX_BYTES = 512 * 1024 * 1024;

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

export async function supportsVideoPreset(preset) {
  if (typeof globalThis.VideoEncoder === "undefined" || typeof globalThis.VideoFrame === "undefined") {
    return false;
  }
  return canEncodeVideo("avc", {
    width: preset.width,
    height: preset.height,
    bitrate: preset.bitrate,
    frameRate: preset.fps,
  });
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

export async function createVideoWriter(canvas, preset, destination) {
  const output = new Output({
    format: new Mp4OutputFormat({ fastStart: false }),
    target: destination.target,
  });
  const source = new CanvasSource(canvas, {
    codec: "avc",
    bitrate: preset.bitrate,
    keyFrameInterval: 2,
    latencyMode: "quality",
  });
  output.addVideoTrack(source, { frameRate: preset.fps });
  await output.start();

  let finished = false;
  return {
    async addFrame(index) {
      const duration = 1 / preset.fps;
      await source.add(index * duration, duration);
    },
    async finish() {
      if (finished) return;
      finished = true;
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
