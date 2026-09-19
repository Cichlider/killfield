# Three AIs, One Tank Trouble Arena

I built a browser-based Tank Trouble AI project featuring three distinct opponents:

- **Hybrid** — a powerful PPO-trained neural policy refined through an evolving self-play league.
- **Killfield** — a real-time planning agent that simulates possible actions and specializes in ricochet shots.
- **Laika** — a fast, rule-based classic bot and a good starting challenge.

## How Hybrid Is Trained

Hybrid uses PPO actor–critic reinforcement learning. It trains across 256 parallel game environments, collecting 128 steps from each environment per update. Its 1,028-dimensional observation describes the maze, tanks, bullets, ammunition, and action-specific safety signals. The network outputs one of 18 movement-and-fire actions every physics frame.

Training combines GAE and clipped PPO optimization with a league of frozen previous Hybrid checkpoints, plus Laika and Killfield. This exposes the policy to different play styles while preventing it from overfitting to a single opponent. The final policy is exported from PyTorch and runs locally in the browser.

Read the interactive technical paper for the complete observation space, reward function, network architecture, training lineage, loss design, and evaluation results:

**[Hybrid Technical Paper](https://cichlider.github.io/killfield/paper/)**

## Play Instantly

No download or account is required. Challenge any opponent or watch AI-vs-AI battles:

**[Play Tank Trouble AI](https://cichlider.github.io/killfield/viewer/)**

### Mobile Controls

The game also works on mobile. Landscape mode is recommended: use the on-screen joystick on the left to move and the fire control on the right to shoot.

For more responsive mobile handling, enabling **Instant turn** is recommended. Ranked runs disable this assistance automatically so that every submitted record uses the same competitive settings.

## Join the Leaderboard

**[View the Leaderboard](https://cichlider.github.io/killfield/viewer/leaderboard.html)**

To submit a record:

1. Open the game and select **Play**.
2. Choose Hybrid, Killfield, or Laika as your opponent.
3. Click **Ranked run**.
4. Play until you end the run or reach the 200-round limit.
5. Win at least one round, enter your player name, and click **Submit**.

Hybrid, Killfield, and Laika have separate leaderboards. Rankings are based on the total number of wins in a single run. The board also displays total rounds, losses, and double KOs.

Ranked runs use fixed competitive settings: zero-frame opponent delay, no instant-turn assistance, and no extended opening pause. Changing the opponent, rerolling the map, or changing a ranked setting ends the current recording.

Every submission includes a deterministic input replay. The server replays the entire run frame by frame using the same engine and independently calculates the result, so leaderboard scores are not accepted directly from the browser.

## About the Physics

This project is adapted from TankTrouble2. Because AI training requires hundreds of millions of simulation steps—and because my training budget is limited—I did not attempt to reproduce the official game's physics engine exactly. Doing so would have made large-scale training significantly slower.

## Contact

For business inquiries, Game AI opportunities, recruitment, or collaboration:

**cichlid1234@outlook.com**

## License

This project is released under the [MIT License](LICENSE).
