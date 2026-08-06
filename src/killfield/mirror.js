/**
 * A read-through view of a Game with tanks 0 and 1 swapped.
 *
 * The agent and its scoring code never hardcode a tank's identity — they
 * always treat `tanks[0]` as "me" and `tanks[1]` as "the enemy", resolved
 * fresh from whatever game object was passed in. That means a second
 * KillFieldAgent can drive tank 1 by planning against this view instead of
 * the real game: from its side, `tanks[0]` resolves to the real tank 1, so
 * every read (and `drive()`'s writes) land on the right tank without any
 * change to teacher.js, score.js, sandbox.js, or field.js.
 *
 * This is a standing view, not a sandbox snapshot — every field except
 * `tanks`/`scores` is a live getter back onto the real game, so it costs
 * nothing to build once per round and reuse every frame.
 */

const PASSTHROUGH = [
  "maze", "walls", "wallHalfT", "scale", "wallGrid", "distancesForMaze",
  "deadEnds", "reachable", "reachableIndex", "settingsMaxBullets",
  "tanksCount", "aiFactory", "rng", "bullets", "tankFields", "events",
  "frame", "aliveCount", "endCount", "resetCount", "frozen", "shake",
  "crateTimer", "bulletDepth", "roundNumber",
];

export function mirrorView(realGame) {
  const view = Object.create(Object.getPrototypeOf(realGame));
  for (const key of PASSTHROUGH) {
    Object.defineProperty(view, key, { get: () => realGame[key], enumerable: true });
  }
  Object.defineProperty(view, "tanks", {
    get: () => [realGame.tanks[1], realGame.tanks[0]],
    enumerable: true,
  });
  Object.defineProperty(view, "scores", {
    get: () => [realGame.scores[1], realGame.scores[0]],
    enumerable: true,
  });
  return view;
}
