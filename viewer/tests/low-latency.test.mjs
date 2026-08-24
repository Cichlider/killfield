import assert from "node:assert/strict";
import { predictLocalTank, simulationBudget } from "../src/low-latency.js";

const idle = predictLocalTank(
  { x: 10, y: 20, rotation: 0 },
  { forward: 0, backup: 0, turnLeft: 0, turnRight: 0 },
  50, 1,
);
assert.deepEqual(idle, { x: 10, y: 20, rotation: 0 });

const forward = predictLocalTank(
  { x: 10, y: 20, rotation: 0 },
  { forward: 1, backup: 0, turnLeft: 0, turnRight: 0 },
  50, 1,
);
assert.ok(Math.abs(forward.x - 10) < 1e-9);
assert.ok(Math.abs(forward.y - 16) < 1e-9);

const reverseHalf = predictLocalTank(
  { x: 10, y: 20, rotation: 0 },
  { forward: 0, backup: 1, turnLeft: 0, turnRight: 0 },
  50, 0.5,
);
assert.ok(Math.abs(reverseHalf.y - 21.25) < 1e-9);

const curved = predictLocalTank(
  { x: 0, y: 0, rotation: 0 },
  { forward: 1, backup: 0, turnLeft: 0, turnRight: 1 },
  50, 1,
);
assert.equal(curved.rotation, 10);
assert.ok(curved.x > 0);
assert.ok(curved.y < 0);

assert.deepEqual(simulationBudget(0, 100, 40, 250, true), {
  steps: 1, remainder: 20, dropped: 1,
});
assert.deepEqual(simulationBudget(0, 100, 40, 250, false), {
  steps: 2, remainder: 20, dropped: 0,
});
assert.deepEqual(simulationBudget(30, 5, 40, 250, true), {
  steps: 0, remainder: 35, dropped: 0,
});

console.log("low-latency input prediction and frame budgeting OK");
