/**
 * Behavioural benchmark for the scripted AI.
 *
 * The port assertions in suite.js check mechanics. This checks character: an
 * AI that is subtly mis-ported still passes every mechanical assertion while
 * winning too often, too rarely, or too slowly. Run it against the same
 * scenario on the reference implementation and compare the shape of the
 * numbers, not the exact values — the two use different random streams.
 *
 * Usage:  node test/benchmark.js [randomRounds] [idleRounds]
 */

import { Game } from "../src/game.js";
import { LaikaAI } from "../src/laika.js";
import { Rng } from "../src/rng.js";

const SETTLEMENT_FRAMES = 75;

/**
 * @param {"random"|"idle"} mode  how the human-side tank behaves
 * @param {number} rounds         rounds to play out
 */
export function benchmark(mode, rounds) {
  const rng = new Rng(20260803);
  const game = new Game({ seed: 5150, aiFactory: (g, t) => new LaikaAI(g, t) });
  let aiWins = 0;
  let playerWins = 0;
  let mutual = 0;
  let done = 0;
  let frames = 0;
  let roundStart = 0;
  const roundLengths = [];

  while (done < rounds && frames < 2_000_000) {
    const t = game.tanks[0];
    if (mode === "random") {
      // Re-roll every five frames so the flailing is coarse enough to move.
      if (frames % 5 === 0) {
        t.forward = rng.random() < 0.45;
        t.backup = rng.random() < 0.2;
        t.turnLeft = rng.random() < 0.3;
        t.turnRight = rng.random() < 0.3;
        t.fire = rng.random() < 0.15;
      }
    } else {
      t.forward = t.backup = t.turnLeft = t.turnRight = t.fire = false;
    }

    for (const e of game.step()) {
      if (e[0] === "round_end") {
        done++;
        roundLengths.push(game.frame - roundStart);
        if (e[1] === 1) aiWins++;
        else if (e[1] === 0) playerWins++;
        else mutual++;
      }
      if (e[0] === "new_round") roundStart = game.frame;
    }
    frames++;
  }

  const mean = roundLengths.reduce((a, b) => a + b, 0) / roundLengths.length;
  return {
    mode,
    rounds: done,
    aiWins,
    playerWins,
    mutual,
    // Rounds are scored 75 frames after the kill, so back that out.
    secondsToKill: Number(((mean - SETTLEMENT_FRAMES) / 25).toFixed(1)),
  };
}

const isMain = typeof process !== "undefined"
  && process.argv[1]
  && process.argv[1].endsWith("benchmark.js");

if (isMain) {
  const randomRounds = Number(process.argv[2] ?? 100);
  const idleRounds = Number(process.argv[3] ?? 60);
  console.log("JS vs random :", benchmark("random", randomRounds));
  console.log("JS vs idle   :", benchmark("idle", idleRounds));
}
