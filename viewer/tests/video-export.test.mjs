import assert from "node:assert/strict";
import test from "node:test";

import {
  AUDIO_BITRATE,
  AUDIO_SAMPLE_RATE,
  DIRECT_DOWNLOAD_MAX_BYTES,
  VIDEO_PRESETS,
  decodePcm16Wav,
  estimatedBytes,
  mixReplaySoundChunk,
  safeVideoFilename,
} from "../src/video-export.js";

test("video presets match the three published export tiers", () => {
  assert.deepEqual(VIDEO_PRESETS.recommended,
    { width: 1280, height: 720, fps: 60, bitrate: 4_000_000 });
  assert.deepEqual(VIDEO_PRESETS.hd,
    { width: 1920, height: 1080, fps: 60, bitrate: 8_000_000 });
  assert.deepEqual(VIDEO_PRESETS.ultra,
    { width: 1920, height: 1080, fps: 120, bitrate: 12_000_000 });
});

test("bitrate estimate and filenames are safe", () => {
  assert.equal(DIRECT_DOWNLOAD_MAX_BYTES, 512 * 1024 * 1024);
  assert.equal(estimatedBytes(60, 4_000_000), 30_000_000);
  assert.equal(safeVideoFilename("top/player: 27 wins"),
    "killfield-top-player-27-wins.mp4");
  assert.equal(safeVideoFilename(".."), "killfield-replay.mp4");
});

test("PCM16 effects decode and mix at replay timestamps", () => {
  const wav = new ArrayBuffer(48);
  const bytes = new Uint8Array(wav);
  const view = new DataView(wav);
  const ascii = (offset, value) => bytes.set([...value].map((char) => char.charCodeAt(0)), offset);
  ascii(0, "RIFF");
  view.setUint32(4, 40, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, 4, true);
  view.setUint32(28, 8, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, 4, true);
  view.setInt16(44, 16_384, true);
  view.setInt16(46, -16_384, true);

  const clip = decodePcm16Wav(wav);
  assert.equal(clip.sampleRate, 4);
  assert.deepEqual([...clip.samples], [0.5, -0.5]);
  const mixed = mixReplaySoundChunk({ fire: [clip] }, [{ kind: "fire", time: 0.25 }], 0, 4, 4);
  assert.ok(Math.abs(mixed[1] - 0.24) < 1e-6);
  assert.ok(Math.abs(mixed[2] + 0.24) < 1e-6);
  assert.equal(mixed[0], 0);
  assert.equal(mixed[3], 0);
  assert.equal(AUDIO_SAMPLE_RATE, 44_100);
  assert.equal(AUDIO_BITRATE, 128_000);
});
