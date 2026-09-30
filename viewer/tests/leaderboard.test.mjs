import assert from "node:assert/strict";
import {
  compareLeaderboardEntries, leaderboardWinRate, leaderboardWins,
} from "../src/leaderboard-ranking.js";

const earlier = "2026-09-01T00:00:00.000Z";
const later = "2026-09-02T00:00:00.000Z";
const entries = [
  { name: "fewer wins", wins: 9, rounds: 9, verifiedAt: earlier },
  { name: "lower rate", wins: 10, rounds: 20, verifiedAt: earlier },
  { name: "newer exact tie", wins: 10, rounds: 10, verifiedAt: later },
  { name: "older exact tie", score: 10, rounds: 10, verifiedAt: earlier },
];

assert.equal(leaderboardWins(entries[3]), 10, "legacy score aliases remain supported");
assert.equal(leaderboardWinRate(entries[1]), 0.5);
assert.equal(leaderboardWinRate({ wins: 4, rounds: 0 }), 0);
assert.deepEqual(entries.toSorted(compareLeaderboardEntries).map(entry => entry.name), [
  "older exact tie",
  "newer exact tie",
  "lower rate",
  "fewer wins",
]);

console.log("leaderboard: wins, then win rate, then verification time");
