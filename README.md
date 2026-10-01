# Think You Can Beat My Tank Trouble AI?

I built two AIs for Tank Trouble that you can play against in your browser:

- **Hybrid** — a PPO neural policy trained through self-play against a league of its past versions. It beats Killfield about 93% of the time.
- **Killfield** — a real-time planner that simulates moves ahead and loves ricochet shots. It beats Laika about 90% of the time.

Laika is in there too if you want a warm-up.

## Play

No download, no account:

**[Play now](https://cichlider.github.io/killfield/viewer/)**

Works on mobile too. The on-screen stick defaults to **Wheel (128-direction)** with **No turn-rate limit** enabled. Switch to **Arrow keys (8-way)** for keyboard-like steering: eight sectors on a wheel, where each sector is a direction-key combination — up is forward, left turns left, up-left drives them together — and the distance from the centre sets the speed, so a full push is exactly the keyboard. Every number behind that stick is adjustable under **Stick feel** — centre and transition zones, stick size, and one handle per boundary line, draggable directly on the pad while the panel is open. Switch the stick between a **circle** (stop / ramp / full-speed zones) and a **square** (a 3x3 grid with no ramp), or pick **Three-finger pads** — forward/reverse under the left thumb, turn under the right, fire on its own in the top-right — whose buttons can be dragged anywhere. The settings are saved in the browser and importable/exportable as JSON. The wheel uses world-heading steering and keeps its own **No turn-rate limit** setting. Your touch control choice is saved in the browser. You can also just watch the AIs fight each other.

## Leaderboard

**[View the leaderboard](https://cichlider.github.io/killfield/viewer/leaderboard.html)**

Pick an opponent, click **Ranked run**, win at least one round, then submit your name. Each opponent has its own board, ranked by total wins in a single run (max 200 rounds).

Ranked runs use fixed settings: no opponent delay and no turn-rate assist. Changing the opponent, rerolling the map, or changing a setting ends the recording. Every submission is replayed frame by frame on the server with the same engine, so scores can't be faked.

## How It Works

The physics is adapted from TankTrouble2 and simplified so the AI could train on hundreds of millions of steps, so it won't feel 100% identical to the real game.

Hybrid is trained with PPO across 256 parallel games. Its 1,028-dimensional observation covers the maze, tanks, bullets, ammo and per-action safety signals, and it picks one of 18 move-and-fire actions every frame. The trained network runs locally in your browser.

The full story — observation space, reward, architecture, training lineage and evaluation — is in the interactive paper:

**[Read the paper](https://cichlider.github.io/killfield/paper/)**

If you enjoy it, a ⭐ on this repo helps a lot!

## Contact

Game AI opportunities, collaboration, business: **cichlid1234@outlook.com**

## License

[MIT](LICENSE)
