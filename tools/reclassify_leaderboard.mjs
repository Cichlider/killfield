#!/usr/bin/env node
/**
 * One-time/backfill utility for the Human/Bot leaderboard split.
 *
 * It deliberately takes an already-downloaded GitHub Issues API response;
 * the migration is deterministic and never needs network access itself.
 *
 * Usage:
 *   node tools/reclassify_leaderboard.mjs --issues issues.json
 *   node tools/reclassify_leaderboard.mjs --issues issues.json --write
 */

import fs from "node:fs";
import { createHash } from "node:crypto";
import { HybridPolicy } from "../viewer/src/hybrid.js";
import { summariseResults, unpackSession } from "../viewer/src/replay.js";
import { replaySession } from "../viewer/src/ranked.js";

const VIEWER = new URL("../viewer/", import.meta.url);
const argument = (name) => {
  const index = process.argv.indexOf(name);
  return index === -1 ? null : process.argv[index + 1];
};
const issuesPath = argument("--issues");
const boardPath = argument("--board") ?? new URL("leaderboard.json", VIEWER);
if (!issuesPath) throw new Error("--issues FILE is required");

const issues = JSON.parse(fs.readFileSync(issuesPath, "utf8"));
const board = JSON.parse(fs.readFileSync(boardPath, "utf8"));
if (!Array.isArray(issues) || !Array.isArray(board.entries)) {
  throw new Error("issues and board must both contain arrays");
}
const byNumber = new Map(issues.map((issue) => [issue.number, issue]));

const engineBytes = fs.readFileSync(new URL("kf_engine.wasm", VIEWER));
const engineStamp = createHash("sha256").update(engineBytes).digest("hex").slice(0, 16);
const manifest = JSON.parse(fs.readFileSync(new URL("assets/hybrid.json", VIEWER)));
const weightBytes = fs.readFileSync(new URL("assets/hybrid.bin", VIEWER));
const policyStamp = createHash("sha256").update(weightBytes).digest("hex").slice(0, 16);
const policy = new HybridPolicy(manifest, new Float32Array(
  weightBytes.buffer, weightBytes.byteOffset, weightBytes.byteLength / 4,
));
const { instance } = await WebAssembly.instantiate(engineBytes, {});

for (const entry of board.entries) {
  if (!process.argv.includes("--all")
      && (entry.playerClass === "human" || entry.playerClass === "bot")
      && Number.isFinite(entry.hybridMovementMatch)
      && Number.isInteger(entry.hybridMovementMatches)
      && Number.isInteger(entry.hybridMovementFrames)) {
    continue;
  }
  const issue = byNumber.get(entry.issue);
  if (!issue) throw new Error(`Issue ${entry.issue} for ${entry.name} is missing`);
  const fence = /```json\s*\n([\s\S]*?)\n```/.exec(issue.body ?? "");
  if (!fence) throw new Error(`Issue ${entry.issue} has no record`);
  const submission = JSON.parse(fence[1]);
  if (submission.track === null) {
    throw new Error(`Issue ${entry.issue} is chunked; supply its restored track before backfill`);
  }
  if (submission.engine !== engineStamp || submission.policy !== policyStamp) {
    throw new Error(`Issue ${entry.issue} was recorded against different binaries`);
  }
  const session = await unpackSession(submission.track);
  const { winners, suspect, hybridMovement } = replaySession({
    wasm: instance.exports,
    policy,
    config: {
      seed: submission.seed,
      opponent: submission.opponent,
      delayFrames: submission.delayFrames,
      openingDelaySeconds: submission.openingDelaySeconds,
    },
    session,
  });
  if (suspect.length > 0) throw new Error(`Issue ${entry.issue} failed its old action audit`);
  const stats = summariseResults(winners);
  if (stats.wins !== entry.wins || stats.rounds !== entry.rounds) {
    throw new Error(`Issue ${entry.issue} no longer reproduces its committed score`);
  }
  entry.playerClass = hybridMovement.playerClass;
  entry.hybridMovementMatch = hybridMovement.rate;
  entry.hybridMovementMatches = hybridMovement.matches;
  entry.hybridMovementFrames = hybridMovement.frames;
  process.stdout.write(`${entry.issue}\t${entry.name}\t${entry.playerClass}\t`
    + `${(hybridMovement.rate * 100).toFixed(1)}%\n`);
}

if (process.argv.includes("--write")) {
  fs.writeFileSync(boardPath, `${JSON.stringify(board, null, 2)}\n`);
}
