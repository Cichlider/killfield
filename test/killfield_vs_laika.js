/**
 * Reproducible full-match benchmark: KillField (tank 0) vs Laika (tank 1).
 *
 * Usage: node test/killfield_vs_laika.js [rounds] [timeoutSeconds] [seed] [rays]
 *
 * A round that has no death before the active-play timeout is recorded as a
 * draw. Mutual kills already arrive from the engine as round_end(null) and
 * are reported separately. The post-kill settlement window is always played
 * in full; only the cosmetic frozen pause before the next maze is skipped.
 */

import * as C from "../src/constants.js";
import { Game } from "../src/game.js";
import { LaikaAI } from "../src/laika.js";
import { KillFieldAgent } from "../src/killfield/teacher.js";

export function benchmarkKillFieldVsLaika({
  rounds = 1000,
  timeoutSeconds = 60,
  seed = 20260814,
  rays = 2048,
  progressEvery = 25,
} = {}) {
  const game = new Game({
    seed,
    aiFactory: (g, tank) => new LaikaAI(g, tank),
  });
  const agent = new KillFieldAgent({ seed: 0, rayCount: rays, oppModel: "L2" });
  const timeoutFrames = Math.round(timeoutSeconds * C.FPS);
  const started = performance.now();

  let completed = 0;
  let wins = 0;
  let losses = 0;
  let draws = 0;
  let mutualKills = 0;
  let timeouts = 0;
  let currentWinStreak = 0;
  let maxWinStreak = 0;
  let activeFrames = 0;
  let totalActiveFrames = 0;

  const record = (winner, timeout = false) => {
    completed += 1;
    totalActiveFrames += activeFrames;
    if (winner === 0) {
      wins += 1;
      currentWinStreak += 1;
      maxWinStreak = Math.max(maxWinStreak, currentWinStreak);
    } else {
      currentWinStreak = 0;
      if (winner === 1) losses += 1;
      else {
        draws += 1;
        if (timeout) timeouts += 1;
        else mutualKills += 1;
      }
    }
    if (progressEvery > 0 && completed % progressEvery === 0) {
      const elapsed = ((performance.now() - started) / 1000).toFixed(1);
      console.error(
        `[${completed}/${rounds}] ${wins}W ${losses}L ${draws}D, `
        + `max streak ${maxWinStreak}, ${elapsed}s`,
      );
    }
  };

  const nextRoundNow = () => {
    game.cleanUpBattle();
    game.frozen = false;
    game.endCount = -1;
    game.resetCount = -1;
    game.setupBattle();
    activeFrames = 0;
  };

  while (completed < rounds) {
    if (!game.frozen && game.tanks[0].alive) agent.drive(game);
    if (!game.frozen && game.tanks[0].alive && game.tanks[1].alive) {
      activeFrames += 1;
    }

    const events = game.step();
    const roundEnd = events.find((event) => event[0] === "round_end");
    if (roundEnd) {
      record(roundEnd[1]);
      if (completed < rounds) nextRoundNow();
      continue;
    }

    if (!game.frozen && game.tanks[0].alive && game.tanks[1].alive
        && activeFrames >= timeoutFrames) {
      record(null, true);
      if (completed < rounds) nextRoundNow();
    }
  }

  const elapsedSeconds = (performance.now() - started) / 1000;
  return {
    rounds: completed,
    seed,
    rays,
    timeoutSeconds,
    wins,
    losses,
    draws,
    mutualKills,
    timeouts,
    maxWinStreak,
    winRate: wins / completed,
    decisiveWinRate: wins / Math.max(1, wins + losses),
    meanActiveSeconds: totalActiveFrames / completed / C.FPS,
    elapsedSeconds,
  };
}

const isMain = typeof process !== "undefined"
  && process.argv[1]
  && process.argv[1].endsWith("killfield_vs_laika.js");

if (isMain) {
  const rounds = Number(process.argv[2] ?? 1000);
  const timeoutSeconds = Number(process.argv[3] ?? 60);
  const seed = Number(process.argv[4] ?? 20260814);
  const rays = Number(process.argv[5] ?? 2048);
  console.log(JSON.stringify(benchmarkKillFieldVsLaika({
    rounds, timeoutSeconds, seed, rays,
  }), null, 2));
}
