//! The checks that tell a user what the bot sees and what it may press.
//! Ported from `botcore/diagnostics.py`.
//!
//! These exist because the failure modes of a bot that drives someone else's
//! window are all silent: a window that refuses to render, a game that pauses
//! while unfocused, a calibration that recorded nothing, a policy that cannot
//! press the one key that matters. Each check below measures one of those and
//! says which one it is, rather than leaving the user to watch a log.

use crate::capture::FrameStack;
use crate::config::Config;
use crate::keys::{ActionSpace, Keymap};
use crate::platform::{self, Handle, WindowInfo};

/// Watch a window for a few seconds and report what the capture actually yields.
pub fn check_capture(cfg: &Config, handle: Handle, seconds: f32) -> i32 {
    use crate::ui::{Tone, ui};
    let ui = ui();
    ui.section("Capture check");
    let pid = platform::current::get_window_pid(handle);
    let process = platform::current::get_process_name(pid);
    ui.rows(&[
        ("window", format!("{handle}  process {}", if process.is_empty() { "?" } else { &process })),
        ("observing", format!("{seconds:.0}s at {}px", cfg.frame_size)),
    ]);

    let mut grabber = platform::FrameGrabber::new(handle);
    let mut stack = FrameStack::new(cfg.frame_size, cfg.frame_stack);
    let mut deltas: Vec<f32> = Vec::new();
    let mut failures = 0u32;
    let mut frames = 0u32;
    let mut black = 0u32;
    let mut previous: Option<Vec<f32>> = None;
    let pace = std::time::Duration::from_secs_f32(1.0 / cfg.target_fps.max(1.0));
    let started = std::time::Instant::now();

    while started.elapsed().as_secs_f32() < seconds {
        let Some(raw) = grabber.grab() else {
            failures += 1;
            std::thread::sleep(std::time::Duration::from_millis(50));
            continue;
        };
        let observation = stack.push(&raw);
        let motion = observation.channel(cfg.frame_stack);
        let mean_motion = mean(motion);
        if previous.is_some() {
            deltas.push(mean_motion);
        }
        // `observation[0]` is the oldest frame in the ring, normalized.
        if mean(observation.channel(0)) < 0.01 {
            black += 1;
        }
        previous = Some(observation.data.clone());
        frames += 1;
        // The panel is the progress bar here: watching a still number for five
        // seconds is exactly what makes people think it has hung.
        ui.live_now(
            &crate::ui::Frame::new("Capture check")
                .badge(format!("{frames} frame(s)"), Tone::Accent)
                .line(
                    "grab rate",
                    format!("{:.1} fps, {:.1} ms per grab", frames as f64 / started.elapsed().as_secs_f64().max(1e-6), grabber.mean_ms()),
                    Tone::Value,
                )
                .line("failures", failures.to_string(), Tone::Body),
        );
        std::thread::sleep(pace);
    }
    ui.finish();
    let elapsed = started.elapsed().as_secs_f64();
    ui.rows(&[(
        "captured",
        format!(
            "{frames} frame(s) in {elapsed:.1}s ({:.1} fps), {:.1} ms per grab",
            frames as f64 / elapsed.max(1e-6),
            grabber.mean_ms()
        ),
    )]);
    if failures > 0 {
        ui.event("Capture", format!("{failures} grab(s) returned nothing."));
        if let Some(error) = &grabber.last_error {
            ui.event("Capture", format!("last error: {error}"));
        }
    }

    if frames < 3 {
        ui.event_as(
            "Fail",
            Tone::Bad,
            "The window is not giving up any frames. Is it minimized, or is the game still \
             loading?",
        );
        return 1;
    }
    let mean_delta = mean(&deltas);
    let changing = 100.0 * deltas.iter().filter(|delta| **delta > 1e-3).count() as f64
        / deltas.len().max(1) as f64;
    if black as f64 > frames as f64 * 0.9 {
        ui.event_as(
            "Fail",
            Tone::Bad,
            "Almost every frame is black. A hardware-accelerated game sometimes refuses \
             PrintWindow; try borderless windowed mode, or run the game windowed.",
        );
        return 1;
    }
    if mean_delta < 1e-4 {
        ui.event_as(
            "Warn",
            Tone::Warn,
            "The image is not changing at all. The bot will learn nothing from a still \
             screen: the game is probably paused while unfocused, or the window is showing a \
             loading screen.",
        );
    } else {
        ui.event_as(
            "OK",
            Tone::Good,
            format!(
                "Image is live: mean change {mean_delta:.4}, changing on {changing:.0}% of \
                 frames."
            ),
        );
    }
    ui.blank();
    0
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

/// Every window the bot can see, and which of them it will not drive.
pub fn list_windows() -> i32 {
    use crate::ui::{Tone, ui};
    let ui = ui();
    for line in platform::capability_report() {
        ui.event("Platform", line);
    }
    let windows = platform::current::enumerate_windows(platform::MIN_AREA);
    if windows.is_empty() {
        ui.event_as(
            "Windows",
            Tone::Warn,
            "No windows found. `bot1 --platform` says what this build can see here.",
        );
        return 1;
    }
    ui.section("Windows");
    ui.raw(&platform::format_window_table(&windows));
    let games: Vec<&WindowInfo> = windows
        .iter()
        .filter(|window| platform::looks_like_game(window))
        .collect();
    match games.first() {
        Some(game) => {
            ui.event_as(
                "Pick",
                Tone::Good,
                format!(
                    "Largest likely game: {} ({}) hwnd={} - that is what the bot picks \
                     automatically.",
                    game.title, game.process, game.handle
                ),
            );
            ui.event(
                "Pick",
                "`bot1 --window HWND` drives a different one; `bot1 --pick` asks every time.",
            );
        }
        None => ui.event_as(
            "Pick",
            Tone::Warn,
            "Nothing here looks like a game, so the bot will ask you to pick one.",
        ),
    }
    ui.event(
        "Pick",
        "Marked windows are never auto-selected: the terminal this bot runs in, its own \
         console, and programs that are not games.",
    );
    ui.blank();
    0
}

/// Straighten out a key left down, and every mouse button with it.
///
/// A cursor left confined to a window is repaired here too: it is the other
/// piece of desktop-wide state a run that died can leave behind, and it is the
/// state a user notices first, because it takes their mouse away.
pub fn release_all_keys(cfg: &Config, handle: Handle) -> i32 {
    let keymap = Keymap::load(&cfg.keymap_path).unwrap_or_else(Keymap::default_layout);
    let mut injector = platform::Injector::new(handle, cfg.focus_policy, cfg.clip_cursor);
    let ui = crate::ui::ui();
    ui.section("Release");
    ui.line("Lifting every key in the whitelist and all mouse buttons.");
    injector.acquire_focus(true);
    injector.begin_action();
    for code in keymap.vks.values() {
        injector.release_vk(*code);
    }
    for code in [0x01u32, 0x02, 0x04] {
        injector.release_vk(code);
    }
    injector.end_action();
    // Unconditional, and last: not "a clip this call took", but "nothing is
    // holding the cursor" - the repair is for the run that died, not for this
    // one, which never confined anything.
    platform::free_cursor_clip();
    ui.event_as(
        "Release",
        crate::ui::Tone::Good,
        "Done. The cursor is free to move anywhere again.",
    );
    0
}

/// What the bot may do, in full: the whitelist and the action space it implies.
pub fn show_actions(cfg: &Config, mut keymap: Keymap) -> i32 {
    use crate::ui::{Tone, ui};
    let ui = ui();
    keymap.set_mouse_turn(if cfg.mouse_turn_pixels == 0 {
        None
    } else {
        Some(cfg.mouse_turn_pixels)
    });
    let space = ActionSpace::build(
        &keymap,
        cfg.max_held_keys,
        &cfg.speed_levels,
        true,
        true,
    );
    let base = keymap.default_mouse_turn.max(1);

    ui.section("Action space");
    ui.rows(&[
        (
            "keymap",
            keymap
                .path
                .clone()
                .unwrap_or_else(|| "(built-in defaults)".to_string()),
        ),
        ("keys", keymap.describe()),
        ("size", space.describe()),
    ]);

    ui.section(&format!(
        "Hold keys - each decided on its own, at most {} at a time",
        space.max_held
    ));
    for (index, name) in space.hold_names.iter().enumerate() {
        ui.line(&format!("[{index}] {name}"));
    }

    ui.section("Turning - a direction and a speed every step");
    ui.rows(&[
        (
            "base step",
            format!(
                "the bot's own: it starts at {base}px and moves it inside [{}, {}] as it \
                 learns this game",
                cfg.mouse_turn_min, cfg.mouse_turn_max
            ),
        ),
    ]);
    for (index, (name, delta)) in space.directions.iter().enumerate() {
        let rendered = if *delta == (0, 0) {
            "do not turn".to_string()
        } else {
            format!("move mouse by {:+}, {:+} per 1x", delta.0, delta.1)
        };
        ui.line(&format!("[{index}] {name:<9} {rendered}"));
    }
    let speeds: Vec<String> = space
        .speed_levels
        .iter()
        .enumerate()
        .map(|(index, level)| {
            let pixels = ((base as f32) * level).round().max(1.0) as i32;
            format!("[{index}] x{level} -> {pixels:+}px")
        })
        .collect();
    ui.line(&format!("speeds   {}", speeds.join("   ")));

    ui.section("Tap / click - one choice every step");
    for (index, tap) in space.taps.iter().enumerate() {
        ui.line(&format!("[{index}] {}", tap.label));
    }
    if space.taps.len() > 6 {
        ui.event_as(
            "Check",
            Tone::Warn,
            format!(
                "{} tap/click choices are enabled, so a large share of every batch is \
                 spent clicking. In a game with an inventory or an attack button this is \
                 mostly wasted input.",
                space.taps.len()
            ),
        );
    }
    ui.blank();
    0
}

/// Measure what one decision actually costs, on this machine, for this window.
///
/// The Python's benchmark, and the only honest way to answer "is it fast
/// enough?": the model and the capture are timed separately because they have
/// completely different fixes - a smaller window, or a smaller model.
pub fn benchmark<B: burn::tensor::backend::AutodiffBackend>(
    cfg: &Config,
    keymap: &Keymap,
    handle: Handle,
    decisions: usize,
    device: B::Device,
) -> i32 {
    let space = ActionSpace::build(keymap, cfg.max_held_keys, &cfg.speed_levels, true, true);
    // A dry run: this measures the loop, and injecting nothing is not only safer
    // but the only way to time the model rather than the game's reaction to it.
    //
    // Pacing is switched off for the measurement. The Python times `session.step`
    // with the pacer running, and the pacer's job is to *sleep* until the next
    // slot - so its "env + capture" number is mostly the sleep it asked for, and
    // its comparison against a per-step budget can essentially never pass. What a
    // user wants to know is what the work costs, and a deliberate wait is not
    // work.
    let mut bench_cfg = cfg.clone();
    bench_cfg.target_fps = 0.0;
    bench_cfg.capture_delay = 0.0;
    let env = match crate::game::RealGameEnv::new(&bench_cfg, handle, &space, true) {
        Ok(env) => env,
        Err(error) => {
            crate::ui::ui().event_as(
                "Bench",
                crate::ui::Tone::Bad,
                format!("Could not open the window: {error}"),
            );
            return 1;
        }
    };
    let mut session = match crate::session::GameSession::<B>::new(
        &bench_cfg,
        Box::new(env),
        space,
        Some(keymap),
        device,
    ) {
        Ok(session) => session,
        Err(error) => {
            crate::ui::ui().event_as(
                "Bench",
                crate::ui::Tone::Bad,
                format!("Could not build a session: {error}"),
            );
            return 1;
        }
    };

    let ui = crate::ui::ui();
    ui.section("Benchmark - dry run, nothing is injected");
    ui.rows(&[("config", cfg.describe())]);

    // Warm up: the first forward pass initialises buffers that later ones reuse.
    for _ in 0..15 {
        let action = session.decide();
        session.step(&action);
    }
    let mut model_times: Vec<f64> = Vec::with_capacity(decisions);
    let mut env_times: Vec<f64> = Vec::with_capacity(decisions);
    let mut action = session.decide();
    for decision in 0..decisions {
        let started = std::time::Instant::now();
        action = session.decide();
        model_times.push(started.elapsed().as_secs_f64() * 1000.0);
        let started = std::time::Instant::now();
        session.step(&action);
        env_times.push(started.elapsed().as_secs_f64() * 1000.0);
        // The progress bar for a measurement: without it the command looks
        // hung for as long as it runs.
        ui.live(
            &crate::ui::Frame::new("Benchmark")
                .badge("measuring", crate::ui::Tone::Accent)
                .meter(
                    "decisions",
                    format!("{}/{}", decision + 1, decisions),
                    (decision + 1) as f32 / decisions.max(1) as f32,
                    crate::ui::Tone::Accent,
                )
                .footer("this measures the loop, not the game's reaction to it"),
        );
    }
    ui.finish();
    let median = |values: &mut Vec<f64>| {
        values.sort_by(|a, b| a.total_cmp(b));
        values.get(values.len() / 2).copied().unwrap_or(0.0)
    };
    let model_ms = median(&mut model_times);
    let env_ms = median(&mut env_times);
    let total_ms = model_ms + env_ms;
    // One decision covers `action_repeat` game steps, so the budget it has to
    // fit is that many step budgets - not one. Comparing a per-decision cost
    // against a per-step budget is what makes the Python's version report
    // "too slow" on a machine that is keeping up.
    let step_budget_ms = 1000.0 / cfg.target_fps.max(1.0) as f64;
    let decision_budget_ms = step_budget_ms * cfg.action_repeat.max(1) as f64;
    ui.rows(&[
        ("model", format!("{model_ms:6.2} ms per decision")),
        (
            "env + capture",
            format!(
                "{env_ms:6.2} ms  ({} game step(s), no pacing)",
                cfg.action_repeat
            ),
        ),
        (
            "total",
            format!(
                "{total_ms:6.2} ms  ({:.1} decisions/s)",
                1000.0 / total_ms.max(1e-6)
            ),
        ),
        (
            "target",
            format!(
                "{step_budget_ms:6.2} ms per game step ({decision_budget_ms:.2} ms per \
                 decision at action_repeat {})",
                cfg.action_repeat.max(1)
            ),
        ),
    ]);
    if total_ms > decision_budget_ms {
        ui.event_as(
            "Warn",
            crate::ui::Tone::Warn,
            format!(
                "Too slow for the target rate ({total_ms:.0} ms of {decision_budget_ms:.0} \
                 ms). The bot will still run; it will react late."
            ),
        );
        if env_ms > model_ms {
            ui.event(
                "Bench",
                format!(
                    "The time is in capture ({env_ms:.0} ms), not the model ({model_ms:.0} \
                     ms). A big window costs more to grab than a small one; run the game \
                     windowed, or lower target_fps. Cross-check with: bot1 --check-capture"
                ),
            );
        } else {
            ui.event(
                "Bench",
                format!(
                    "The time is in the model ({model_ms:.0} ms). Lower embed_dim (now {}), \
                     mem_tokens (now {}), transformer_layers (now {}) or frame_size (now {}).",
                    cfg.embed_dim, cfg.mem_tokens, cfg.transformer_layers, cfg.frame_size
                ),
            );
        }
    } else {
        ui.event_as(
            "OK",
            crate::ui::Tone::Good,
            "Fast enough for the target rate.",
        );
    }
    ui.blank();
    let _ = action;
    session.close();
    0
}

/// The advice worth giving about an action space, given what it contains.
pub fn action_space_advice(cfg: &Config, keymap: &Keymap, space: &ActionSpace) -> Vec<String> {
    let mut advice = Vec::new();
    let holds: Vec<String> = space
        .hold_names
        .iter()
        .map(|name| name.to_uppercase())
        .collect();
    let movement = ["W", "A", "S", "D"];
    if !holds.iter().any(|hold| movement.contains(&hold.as_str())) {
        let listed = if holds.is_empty() {
            "-".to_string()
        } else {
            holds.join(", ")
        };
        advice.push(format!(
            "No movement key in the whitelist: hold=[{listed}]. A bot with no way to walk \
             cannot make progress on any game - it will press what it has and look like it \
             is shaking on the spot."
        ));
        advice.push(format!(
            "Fix: re-run --calibrate and spend the recording actually walking around, or \
             edit '{}' to add W/A/S/D (or whatever this game calls forward).",
            cfg.keymap_path
        ));
    }
    if keymap.default_mouse_turn < 4 {
        advice.push(format!(
            "The measured mouse turn step is {} px, which is the signature of a game that \
             locks the cursor: the recorder cannot see the real look speed, so the \
             measurement is a twitch. The bot now sets its own speed - it starts from this \
             number, scales it up when a turn does not move the view, and chooses the \
             multiplier per decision - so this is not fatal. If the view still never turns, \
             pass --mouse-turn 20 (or more) to start it higher.",
            keymap.default_mouse_turn
        ));
    }
    if space.taps.len() > 6 {
        advice.push(format!(
            "{} tap/click choices are enabled, so a large share of every batch is spent on \
             clicking things. In a game with an inventory or an attack button this is mostly \
             wasted input.",
            space.taps.len()
        ));
    }
    advice
}

/// Everything worth telling the user before a run starts.
pub fn preflight_warnings(
    cfg: &Config,
    keymap: Option<&Keymap>,
    target: Option<&WindowInfo>,
) -> Vec<String> {
    let mut warnings = Vec::new();
    if cfg.torch_threads > 2 {
        warnings.push(format!(
            "The tensor backend is using {} threads; on a small model this is usually slower \
             than 1. Set BOT_TORCH_THREADS=1 to compare.",
            cfg.torch_threads
        ));
    }
    match keymap {
        None => warnings.push(
            "No keymap file: using built-in defaults. Run --calibrate once so the bot may \
             only press the keys you actually use."
                .to_string(),
        ),
        Some(keymap) if keymap.holds.is_empty() => warnings.push(
            "The keymap has no hold keys, so the bot cannot walk or hold anything down."
                .to_string(),
        ),
        _ => {}
    }
    if let Some(target) = target {
        if platform::current::is_non_game_process(&target.process.to_lowercase()) {
            warnings.push(format!(
                "The selected window is {}, which is not a game. Every key the bot presses \
                 will go to that program instead of the game. Check --list-windows, then \
                 restart with --window HANDLE.",
                platform::describe_window(Some(target))
            ));
        }
        if let Some(keymap) = keymap
            && let Some(recorded) = keymap.game_hwnd
            && recorded as Handle != target.handle
        {
            warnings.push(format!(
                "--calibrate recorded the game as '{}' (handle {recorded}), but this run \
                 selected {}. If the recorded handle is still the game, restart with \
                 --window {recorded}.",
                keymap.game,
                platform::describe_window(Some(target))
            ));
        }
    }
    if let Some(keymap) = keymap
        && keymap.default_mouse_turn < 4
    {
        warnings.push(format!(
            "The mouse turn step is {} pixel(s). A game that locks the cursor hides your real \
             look speed from the recorder. The bot starts from this number and scales it up \
             on its own when turns do not move the view, but if you already know the right \
             step, --mouse-turn sets it.",
            keymap.default_mouse_turn
        ));
    }
    warnings
}

/// [`preflight_warnings`], plus the advice the action space itself provokes.
pub fn preflight(
    cfg: &Config,
    keymap: Option<&Keymap>,
    target: Option<&WindowInfo>,
) -> Vec<String> {
    let mut warnings = preflight_warnings(cfg, keymap, target);
    if let Some(keymap) = keymap {
        let space = ActionSpace::build(
            keymap,
            cfg.max_held_keys,
            &cfg.speed_levels,
            true,
            true,
        );
        warnings.extend(action_space_advice(cfg, keymap, &space));
    }
    warnings
}
