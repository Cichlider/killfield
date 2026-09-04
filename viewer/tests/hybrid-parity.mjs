import fs from "node:fs";
import assert from "node:assert/strict";
import { HybridPolicy } from "../src/hybrid.js";

const manifest = JSON.parse(fs.readFileSync(new URL("../assets/hybrid.json", import.meta.url)));
const bytes = fs.readFileSync(new URL("../assets/hybrid.bin", import.meta.url));
const weights = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
const fixture = JSON.parse(fs.readFileSync(new URL("../assets/hybrid-parity.json", import.meta.url)));
const policy = new HybridPolicy(manifest, weights);
const actual = policy.logits(Float32Array.from(fixture.obs), fixture.mask, Float32Array.from(fixture.dodge));
let maxError = 0;
for (let i = 0; i < actual.length; i += 1) maxError = Math.max(maxError, Math.abs(actual[i] - fixture.logits[i]));
assert.ok(maxError < 2e-5, `PyTorch/browser max logit error ${maxError}`);
assert.equal(policy.act(Float32Array.from(fixture.obs), fixture.mask, Float32Array.from(fixture.dodge)), fixture.action);
console.log(`Hybrid parity passed: action ${fixture.action}, max logit error ${maxError}`);
