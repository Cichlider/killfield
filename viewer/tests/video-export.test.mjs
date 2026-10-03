import assert from "node:assert/strict";
import test from "node:test";

import {
  DIRECT_DOWNLOAD_MAX_BYTES,
  VIDEO_PRESETS,
  estimatedBytes,
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
