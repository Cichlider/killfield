/**
 * Keyboard input.
 *
 * Tracks which keys are physically down and projects that onto a tank's five
 * input booleans each frame. A fresh keydown is also latched until one game
 * tick consumes it: otherwise a quick tap that begins and ends between two
 * 25 FPS samples disappears completely. Firing is edge triggered inside the
 * simulation, so holding the fire key is safe here.
 */

const BINDINGS = {
  forward: ["e", "w", "arrowup"],
  backup: ["d", "s", "arrowdown"],
  turnLeft: ["a", "arrowleft"],
  turnRight: ["f", "arrowright"],
  fire: ["q", " ", "m"],
};

// Keys we consume, so the page does not scroll out from under the game.
const SWALLOW = new Set([
  "arrowup", "arrowdown", "arrowleft", "arrowright", " ",
]);

export class Keyboard {
  constructor(target = window) {
    this.pressed = new Set();
    this.pending = new Set();
    this.onReroll = null;
    this.onPause = null;

    this._down = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const target = e.target;
      if (typeof HTMLElement !== "undefined" && target instanceof HTMLElement
          && (target.matches("input, select, textarea, button") || target.isContentEditable)) {
        return;
      }
      const k = e.key.toLowerCase();
      if (SWALLOW.has(k)) e.preventDefault();
      if (k === "r") {
        if (this.onReroll) this.onReroll();
        return;
      }
      if (k === "p") {
        if (this.onPause) this.onPause();
        return;
      }
      if (!this.pressed.has(k)) this.pending.add(k);
      this.pressed.add(k);
    };
    this._up = (e) => {
      this.pressed.delete(e.key.toLowerCase());
    };
    // A tab switch or alert can eat the keyup, leaving a key stuck down.
    this._blur = () => this.clear();

    target.addEventListener("keydown", this._down);
    target.addEventListener("keyup", this._up);
    target.addEventListener("blur", this._blur);
  }

  has(action) {
    for (const key of BINDINGS[action]) {
      if (this.pressed.has(key) || this.pending.has(key)) return true;
    }
    return false;
  }

  applyTo(tank) {
    tank.forward = this.has("forward");
    tank.backup = this.has("backup");
    tank.turnLeft = this.has("turnLeft");
    tank.turnRight = this.has("turnRight");
    tank.fire = this.has("fire");
    this.pending.clear();
  }

  clear() {
    this.pressed.clear();
    this.pending.clear();
  }
}
