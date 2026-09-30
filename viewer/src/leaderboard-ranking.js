export function leaderboardWins(entry) {
  const wins = entry.wins ?? entry.score;
  return Number.isFinite(wins) ? wins : 0;
}

export function leaderboardWinRate(entry) {
  const rounds = Number(entry.rounds);
  return Number.isFinite(rounds) && rounds > 0
    ? leaderboardWins(entry) / rounds
    : 0;
}

export function compareLeaderboardEntries(a, b) {
  const aVerified = Date.parse(a.verifiedAt);
  const bVerified = Date.parse(b.verifiedAt);
  return leaderboardWins(b) - leaderboardWins(a)
    || leaderboardWinRate(b) - leaderboardWinRate(a)
    || (Number.isFinite(aVerified) ? aVerified : Number.POSITIVE_INFINITY)
      - (Number.isFinite(bVerified) ? bVerified : Number.POSITIVE_INFINITY);
}
