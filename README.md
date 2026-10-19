# bot1

A game-agnostic bot that learns to play by looking at the screen.

It captures a game window, decides what to press (keys, mouse look, clicks), watches what
changed, and learns from that. Nothing about any particular game is hard-coded: no memory
reading, no game-specific hooks, no score parsing. It only ever sees pixels and it only ever
acts through the keyboard and mouse.

```text
╭─ Synthetic corridor ────────────────────────────────────────────────────────────────── RUNNING ─╮
│ steps    1,619   198.3/s   2% in game   00:08   11 episode(s)   01:32 left   ╭─ vision ────────╮│
│ reward   +2.3846/step   ▁█                                                   │░░             ░░││
│ update   loss -0.034   pi -0.000   v 0.027   ent 4.75   kl +0.0168   2.02s   │░░             ░░││
│ rollout  32%  83/256  █████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │░░             ░░││
│ actions  no keys+turn 44%   1 key(s)+turn 31%   no keys 9%                   │░░░░░░░░░░░░░░░░░││
│ learning 2 update(s) applied, 1 in flight, 3 rollout(s) dropped              │▒░░░░░░░░░░░░░░░▒││
│                                                                              │░░         ▒   ░░││
│                                                                              │░░             ░░││
│                                                                              │░░             ░░││
│                                                                              ╰─────────────────╯│
╰─ F8 pause   F9 save   F10 quit   Ctrl+C stop ───────────────────────────────────────────────────╯
```

## Install

Prebuilt single-file binaries are on the [releases page](https://github.com/jreoka/bot1/releases)
(Windows `.exe`, Linux, macOS). Nothing else has to be installed - no Python, no CUDA, no
driver beyond the OS itself.

Building it yourself takes one command and about five minutes on a laptop:

```bash
cargo build --release      # target/release/bot1
```

On Linux the binary talks X11 directly and needs no `libX11`; on macOS it uses CoreGraphics;
on Windows it uses Win32. A headless machine can still build and run `--selftest`,
`--report` and every unit test - only driving a window needs a desktop.

## Use

1. Start the game, in a **window**. (Exclusive fullscreen cannot be captured or driven.)
2. Run it once to teach it your keys, and let it start learning:

   ```bash
   bot1 --setup
   ```

   It asks which window to drive, then watches you play for a moment and writes down the
   keys you actually use. Only those keys are ever pressed from then on. Then training
   starts: nothing is sent until you press **F8**.
3. Leave it running. Press **F8** to pause, **F9** to save, **F10** to stop (or Ctrl+C -
   every key is released first, always).
4. Tomorrow, open the same game and run `bot1` again. It picks up exactly where it left off.
5. Ask what it has learned:

   ```bash
   bot1 --report
   ```

After that, plain `bot1` is the only command you need.

## What it looks like

The run owns the whole terminal - the alternate screen, cleared once - with the log tail on top and
a panel that repaints in place underneath it, instead of a wall of `[Tag] text` scrolling past:

`bot1 --demo` on the synthetic corridor, at 100 columns:

```text
╭─ Synthetic corridor ────────────────────────────────────────────────────────────────── RUNNING ─╮
│ steps    1,619   198.3/s   2% in game   00:08   11 episode(s)   01:32 left   ╭─ vision ────────╮│
│ reward   +2.3846/step   ▁█                                                   │░░             ░░││
│ update   loss -0.034   pi -0.000   v 0.027   ent 4.75   kl +0.0168   2.02s   │░░             ░░││
│ rollout  32%  83/256  █████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │░░             ░░││
│ actions  no keys+turn 44%   1 key(s)+turn 31%   no keys 9%                   │░░░░░░░░░░░░░░░░░││
│ learning 2 update(s) applied, 1 in flight, 3 rollout(s) dropped              │▒░░░░░░░░░░░░░░░▒││
│                                                                              │░░         ▒   ░░││
│                                                                              │░░             ░░││
│                                                                              │░░             ░░││
│                                                                              ╰─────────────────╯│
╰─ F8 pause   F9 save   F10 quit   Ctrl+C stop ───────────────────────────────────────────────────╯
```

- **It shows you what it sees.** The `vision` box is the newest frame of the observation -
  the same grey square the model is handed - area-averaged down to one character per pixel.
  Beside the numbers, shrunk to whatever width they do not need - a value that had to be cut
  ends in `...` rather than stopping mid-word - and under them only on a terminal too narrow
  to put both side by side.
- **The header and the numbers move**: steps, decisions per second, episodes, elapsed time,
  and - with `--steps` - how long is left. Nothing is guessed; every number is the one the
  loop is working from.
- **The reward has a shape.** One glyph per update, newest on the right: the absolute value
  means little, the direction is the question.
- **Log lines are kept, not overwritten.** A checkpoint, a health warning, a pause: it prints in
  the log above the panel, newest nearest the panel, and the panel never moves.
- **It adapts to the terminal.** Too narrow for a border, and it becomes one dense line; no
  colour support, and it is ASCII; piped into a file or `tee`, and it is plain `[Tag] lines`
  with no escapes at all, exactly as before, so scripts keep working.
- **`--ui-preview`** draws the panel with invented numbers if you want to see it without a
  game, and **`--demo`** runs the real loop on the synthetic corridor if you want to watch it
  fill in.

The dashboard takes the terminal's alternate screen and turns off its echo, so pressing F8 with
the terminal in front of you does not type `[19~` into the middle of the panel; both are given
back when the run ends, on a normal exit, on Ctrl+C, and on a panic.

On Windows the console is switched to UTF-8 and to ANSI escape handling at startup - a
Windows 10 console does not understand either until it is asked - and on any OS the size comes
from the actual window, so the panel is the width of the terminal it is in. `--no-color`, or
`BOT_UI=plain`, forces the plain output anywhere.

## Does it actually learn?

Yes, and it is checked rather than claimed. `bot1 --selftest` trains the *identical* code -
same model, same reward, same PPO update - on a synthetic corridor it has never seen, and
compares three things:

- how far the policy gets before and after training,
- how far an agent pressing keys at random gets (the control that makes "it improved" mean
  something),
- whether the reward prefers walking forward to walking backward or standing still.

```bash
bot1 --demo                        # the loop and the panel, on the corridor, in seconds
bot1 --selftest                    # ~24k decisions, a few minutes
bot1 --selftest --steps 6000       # a quicker look
bot1 --selftest --frame-size 64    # faster still on a weak machine
```

A real game is much harder than that corridor. Expect hours of play, not minutes, before
something recognisable emerges - and expect it to be bad for a long time first. That is what
"learns over time" means here: the thing that makes it usable is that stopping does not lose
anything, and every session starts from what the last one learned.

## Where it keeps what it learned

```text
runs/<game>/
  checkpoints/     last.safetensors, numbered copies, index.json
  training.jsonl   one JSON object per update: step, episodes, mean_reward, loss, ...
  model.safetensors what the last session ended with
  session.json     which window this directory belongs to
```

Every game gets its own directory, named after its window title. A policy trained on one game
is never silently resumed into another, and `bot1 --report` reads those files, so it works with
no game running.

Calibrating *more* keys later does not throw that away: the part of the model that understands
the game carries over, and only the key-choice heads are re-initialised for the wider action
space.

## Controls

| Key | What it does |
| --- | --- |
| **F8** | start (first press) / pause / resume |
| **F9** | save a checkpoint now |
| **F10** | save and stop |
| **Ctrl+C** | save and stop, releasing every held key first |

Nothing is sent while the game window is not in front, and the bot refuses to drive its own
terminal or a window belonging to the shell that launched it.

## What the flags are for

Everyday:

| Flag | Meaning |
| --- | --- |
| `--setup` | calibrate keys, then train (first run) |
| `--report [DIR]` | what it has learned so far |
| `--window HWND` | drive a specific window (see `--list-windows`) |
| `--steps N` | stop after N decisions |
| `--fresh` | ignore existing checkpoints |
| `--demo` | the real loop and panel, on the synthetic corridor, no game needed |
| `--ui-preview` | what the dashboard looks like, with invented numbers |
| `--no-color` | plain output even in a terminal (`BOT_UI=plain` does the same) |
| `--dry-run` | run the whole loop, inject nothing |
| `--checkpoint-dir DIR` | put this game's runs somewhere else |
| `--runs-dir DIR` | where the per-game directories live (default `runs/`) |

Diagnostics, when something looks wrong:

| Flag | Meaning |
| --- | --- |
| `--list-windows` | every window it can see, and which it refuses |
| `--check-capture` | what the capture actually yields, for five seconds |
| `--actions` | the key whitelist and the action space it implies |
| `--benchmark` | what one decision costs here (capture vs model) |
| `--platform` | what this build can and cannot do on this OS |
| `--hotkeys` | register F8/F9/F10 and report what arrives |

Tuning, if you want to experiment: `--fps`, `--frame-size`, `--rollout`, `--lr`,
`--entropy`, `--update-budget`, `--checkpoint-every`, `--max-held-keys`, `--embed-dim`,
`--layers`, `--mouse-turn`. Every field also has a `BOT_<FIELD>` environment override.

## When it does not work

**It sits there and nothing happens.** Nothing is sent before the first F8, by design. If
that key cannot be registered the bot says so and starts immediately.

**The screen never changes / capture is black.** The game is probably paused while unfocused
or running exclusive fullscreen. Use borderless windowed and turn off "pause when unfocused".
`--check-capture` says exactly what it sees.

**It never presses anything.** The key whitelist is what it may press. Run `--setup` (or
`--calibrate`) and actually play for a bit - `--actions` shows what it learned.

**It is slow.** The model is a small transformer on the CPU; `--benchmark` reports the split
between capture and inference. Lower `--frame-size` or `--rollout` if the machine is weak.

**It was learning and then stopped getting better.** `--report` says whether the reward has
flattened. Usually the action set cannot reach new situations: calibrate again with the keys
that open the next part of the game.

## How it works

A few choices matter more than the rest:

- **The observation is grey frames plus a frame-difference channel.** Colour triples the
  input cost for very little control information; the difference channel is what lets motion
  be seen without inferring it from appearance.
- **The policy is one small transformer** (RoPE attention, GELU feed-forward at 4x the
  embedding, a rolling memory window) with factored heads: which keys to hold, which way to
  look, how far, and one momentary button. Factored because a flat action space over nine
  keys and four look directions is thousands of mostly-nonsense combinations.
- **The reward is intrinsic.** Progress is "this screen is new, and getting here took
  effort": episodic novelty, depth into the level, and persistence, with penalties for
  standing still and for dying. It is never told what the score is.
- **The critic and the curiosity head share the transformer.** The model predicts the next
  frame's representation as one of its own heads; its prediction error is the novelty signal.
- **Updates run on their own thread**, so the bot keeps playing while it learns instead of
  freezing for a second every rollout.
- **Stopping is not losing.** Checkpoints carry the weights, the step and episode counters,
  the reward's running estimates, and the bot's own mouse speed.

## Tests

```bash
cargo test              # unit and integration tests
bot1 --selftest         # the learning check end to end
```

The self-test is the one that matters: it fails if the algorithm stops learning, if the
reward can be farmed by dying, or if standing still pays.
