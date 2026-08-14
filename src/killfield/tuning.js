/**
 * Live reward tuning shared by every KillFieldAgent on the page.
 *
 * Defaults remain the committed, benchmarked policy. The UI mutates only this
 * object, so a slider change affects the next MPC plan without rebuilding the
 * game or changing deterministic engine physics.
 */

export const TUNING_SCHEMA = Object.freeze([
  { key: "fieldAscentWeight", group: "navigation", default: 34, min: 0, max: 80, step: 1 },
  { key: "fieldPeakWeight", group: "navigation", default: 6, min: 0, max: 30, step: 1 },
  { key: "guidanceProgressWeight", group: "navigation", default: 120, min: 0, max: 300, step: 5 },
  { key: "huntChainGainWeight", group: "navigation", default: 12, min: 0, max: 40, step: 1 },
  { key: "huntTimeScaleSeconds", group: "navigation", default: 10, min: 2, max: 30, step: 1 },
  { key: "huntTimeMaxMultiplier", group: "navigation", default: 8, min: 1, max: 16, step: 0.5 },
  { key: "alignmentWeight", group: "navigation", default: 190, min: 0, max: 500, step: 10 },
  { key: "mobilityWeight", group: "navigation", default: 60, min: 0, max: 200, step: 5 },

  { key: "goodFireBonus", group: "fire", default: 1800, min: 0, max: 3500, step: 50 },
  { key: "shotFlightTimeWeight", group: "fire", default: 12, min: 0, max: 40, step: 1 },
  { key: "ammoReserveWeight", group: "fire", default: 450, min: 0, max: 1200, step: 25 },
  { key: "ammoFlightPressure", group: "fire", default: 1.5, min: 0, max: 4, step: 0.1 },
  { key: "failedFirePenalty", group: "fire", default: 260, min: 0, max: 1000, step: 20 },
  { key: "suicideFirePenalty", group: "fire", default: 2500, min: 0, max: 5000, step: 100 },

  { key: "activeKillTimeWeight", group: "safety", default: 8, min: 0, max: 30, step: 1 },
  { key: "riskWeight", group: "safety", default: 320, min: 0, max: 1000, step: 20 },
]);

export const TUNING_DEFAULTS = Object.freeze(Object.fromEntries(
  TUNING_SCHEMA.map((spec) => [spec.key, spec.default]),
));

export const tuning = { ...TUNING_DEFAULTS };

function specFor(key) {
  return TUNING_SCHEMA.find((spec) => spec.key === key) ?? null;
}

function decimalsFor(step) {
  const text = String(step);
  return text.includes(".") ? text.length - text.indexOf(".") - 1 : 0;
}

/** Clamp and snap one value to the range exposed by the UI. */
export function setTuning(key, rawValue) {
  const spec = specFor(key);
  if (spec === null) return null;
  if (rawValue === "") return tuning[key];
  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed)) return tuning[key];
  const clamped = Math.max(spec.min, Math.min(spec.max, parsed));
  const snapped = spec.min + Math.round((clamped - spec.min) / spec.step) * spec.step;
  const value = Number(snapped.toFixed(decimalsFor(spec.step)));
  tuning[key] = value;
  return value;
}

export function applyTuning(values) {
  if (values === null || typeof values !== "object") return tuningSnapshot();
  for (const [key, value] of Object.entries(values)) setTuning(key, value);
  return tuningSnapshot();
}

export function resetTuning() {
  Object.assign(tuning, TUNING_DEFAULTS);
  return tuningSnapshot();
}

export function tuningSnapshot() {
  return { ...tuning };
}
