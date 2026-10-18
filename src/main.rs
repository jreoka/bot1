//! bot1's command line: the run, and the checks around it.
//!
//! `bot1` with no flags trains on a game window, and every game's training lives
//! in its own directory under `runs/` so that stopping is not losing and running
//! it again tomorrow continues from there. `--setup` is the first-run path
//! (calibrate the keys, then train), and `--report` says what it has learned so
//! far. Everything else is a diagnostic: what the bot can see, what it may
//! press, what a step costs, and whether the algorithm still learns at all.

use std::cell::RefCell;
use std::sync::mpsc::{Receiver, TryRecvError};

use burn::backend::{Autodiff, Flex};

type Backend = Autodiff<Flex>;

fn main() -> anyhow::Result<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    // The renderer is configured from the environment on first use, so
    // `--no-color` has to be in the environment before anything is printed.
    // `NO_COLOR` is the variable every other tool honours, so setting it is
    // also what a pipe or a wrapper script would do.
    if arguments.iter().any(|argument| argument == "--no-color") {
        unsafe { std::env::set_var("NO_COLOR", "1") };
    }
    if arguments.iter().any(|argument| argument == "--help" || argument == "-h") {
        print_help();
        return Ok(());
    }
    // What the panels look like, with plausible numbers and no game. Useful for
    // a screenshot, for checking a theme over SSH, and for CI that renders it.
    if arguments.iter().any(|argument| argument == "--ui-preview") {
        return ui_preview();
    }
    let value_of = |name: &str, fallback: u64| -> u64 {
        arguments
            .iter()
            .position(|argument| argument == name)
            .and_then(|index| arguments.get(index + 1))
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(fallback)
    };
    // The same loop, the same panel, no window: the run a first-time user sees
    // before they have a game open.
    if arguments.iter().any(|argument| argument == "--demo") {
        return demo(value_of("--steps", 3000));
    }
    if arguments.iter().any(|argument| argument == "--selftest") {
        let device = Default::default();
        let result = bot1::synth::run_learning_check::<Backend>(
            device,
            true,
            value_of("--seed", 0),
            value_of("--steps", 24_000),
            48,
            1024,
            60,
        )?;
        std::process::exit(if result.ok { 0 } else { 1 });
    }
    if arguments.iter().any(|argument| argument == "--list-windows") {
        return code(bot1::diagnostics::list_windows());
    }
    if arguments.iter().any(|argument| argument == "--report") {
        return report(&arguments);
    }
    if arguments.iter().any(|argument| argument == "--version") {
        println!("bot1 {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if arguments.iter().any(|argument| argument == "--platform") {
        return platform_report();
    }
    if arguments.iter().any(|argument| argument == "--actions") {
        let cfg = bot1::Config::default();
        let keymap = bot1::keys::Keymap::load(&cfg.keymap_path)
            .unwrap_or_else(bot1::keys::Keymap::default_layout);
        return code(bot1::diagnostics::show_actions(&cfg, keymap));
    }
    if arguments.iter().any(|argument| argument == "--check-capture") {
        return check_capture(&arguments, value_of("--window", 0));
    }
    if arguments.iter().any(|argument| argument == "--hotkeys") {
        return hotkeys(value_of("--seconds", 3));
    }
    if arguments.iter().any(|argument| argument == "--benchmark") {
        return benchmark(value_of("--window", 0), value_of("--decisions", 120));
    }
    if arguments.iter().any(|argument| argument == "--calibrate") {
        return calibrate(value_of("--window", 0), value_of("--seconds", 0));
    }
    // The one-command path: learn the keys, then start learning the game, in a
    // single process. This is what the README tells a new user to type.
    if arguments.iter().any(|argument| argument == "--setup") {
        calibrate(value_of("--window", 0), value_of("--seconds", 0))?;
        return train(&arguments);
    }
    // Anything that is not one of the modes above is a run, which is what the
    // Python does with no flags at all.
    return train(&arguments);
}

/// The default mode: drive a window, learn from it, checkpoint, and stop
/// cleanly on Ctrl+C.
fn train(arguments: &[String]) -> anyhow::Result<()> {
    use bot1::control::Control as _;
    use bot1::platform;
    let has = |name: &str| arguments.iter().any(|argument| argument == name);
    let value_of = |name: &str, fallback: u64| -> u64 {
        arguments
            .iter()
            .position(|argument| argument == name)
            .and_then(|index| arguments.get(index + 1))
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(fallback)
    };

    let dry_run = has("--dry-run");
    let mut cfg = bot1::Config {
        seed: value_of("--seed", 0),
        enable_hotkeys: !has("--no-hotkeys"),
        wait_for_start: !has("--start-now"),
        // On by default: a game that frees the cursor (an inventory, a pause
        // menu) lets the bot's own mouse steps walk the pointer out of the
        // window, and the next click lands on whatever is under it.
        clip_cursor: !has("--no-clip-cursor"),
        ..bot1::Config::default()
    }
    .env_overrides();
    if let Some(index) = arguments.iter().position(|a| a == "--focus") {
        cfg.focus_policy = match arguments.get(index + 1).map(String::as_str) {
            Some("always") => bot1::config::FocusPolicy::Always,
            Some("never") => bot1::config::FocusPolicy::Never,
            _ => bot1::config::FocusPolicy::Once,
        };
    }
    if let Some(index) = arguments.iter().position(|a| a == "--fps") {
        if let Some(fps) = arguments.get(index + 1).and_then(|v| v.parse::<f32>().ok()) {
            cfg.target_fps = fps;
        }
    }
    if let Some(index) = arguments.iter().position(|a| a == "--checkpoint-every") {
        if let Some(seconds) = arguments
            .get(index + 1)
            .and_then(|value| value.parse::<f32>().ok())
        {
            cfg.checkpoint_interval_sec = seconds;
        }
    }
    if let Some(index) = arguments.iter().position(|a| a == "--runs-dir") {
        if let Some(directory) = arguments.get(index + 1) {
            cfg.runs_dir = directory.clone();
        }
    }
    if let Some(index) = arguments.iter().position(|a| a == "--checkpoint-dir") {
        if let Some(directory) = arguments.get(index + 1) {
            cfg.checkpoint_dir = directory.clone();
        }
    }
    if let Some(index) = arguments.iter().position(|a| a == "--keep") {
        if let Some(keep) = arguments
            .get(index + 1)
            .and_then(|value| value.parse::<usize>().ok())
        {
            cfg.checkpoint_keep = keep;
        }
    }
    // The tuning flags: the same tunables the config carries, and the same ones
    // the `BOT_*` environment overrides reach. A macro rather than twenty copies
    // of the same six lines, so a flag cannot end up wired to the wrong field.
    macro_rules! numeric_flags {
        ($($flag:literal => $field:ident : $ty:ty),* $(,)?) => {
            $(
                if let Some(index) = arguments.iter().position(|a| a == $flag) {
                    match arguments.get(index + 1).and_then(|value| value.parse::<$ty>().ok()) {
                        Some(value) => cfg.$field = value,
                        None => bot1::ui::ui().event(
                            "Config",
                            format!("{} needs a {}; ignoring it.", $flag, stringify!($ty)),
                        ),
                    }
                }
            )*
        };
    }
    numeric_flags! {
        "--frame-size" => frame_size: usize,
        "--frame-stack" => frame_stack: usize,
        "--action-repeat" => action_repeat: usize,
        "--max-held-keys" => max_held_keys: usize,
        "--embed-dim" => embed_dim: usize,
        "--layers" => transformer_layers: usize,
        "--memory-window" => mem_tokens: usize,
        "--patch-size" => patch_size: usize,
        "--patch-width" => patch_width: usize,
        "--heads" => attention_heads: usize,
        "--ffn-hidden" => ffn_hidden: usize,
        "--seq-len" => seq_len: usize,
        "--rollout" => rollout_steps: usize,
        "--minibatch" => minibatch_size: usize,
        "--epochs" => epochs_per_update: usize,
        "--lr" => adam_lr: f32,
        "--entropy" => entropy_coef: f32,
        "--clip" => clip_eps: f32,
        "--gamma" => gamma: f32,
        "--update-budget" => update_seconds_budget: f32,
        "--status-every" => status_every: usize,
        "--mouse-turn" => mouse_turn_pixels: i32,
        "--mouse-adapt-rate" => mouse_adapt_rate: f32,
        "--tap-seconds" => tap_seconds: f32,
        "--torch-threads" => torch_threads: usize,
    }
    if let Some(index) = arguments.iter().position(|a| a == "--mouse-speed-levels") {
        match arguments.get(index + 1) {
            Some(raw) => {
                let levels: Vec<f32> = raw
                    .replace(',', " ")
                    .split_whitespace()
                    .filter_map(|part| part.parse::<f32>().ok())
                    .collect();
                if levels.is_empty() {
                    bot1::ui::ui().event(
                        "Config",
                        format!(
                            "--mouse-speed-levels '{raw}' is not a list of numbers; using the \
                             default."
                        ),
                    );
                } else {
                    cfg.speed_levels = levels;
                }
            }
            None => bot1::ui::ui().event(
                "Config",
                "--mouse-speed-levels needs a list, e.g. 0.5,1,2.",
            ),
        }
    }
    if let Some(index) = arguments.iter().position(|a| a == "--log") {
        cfg.log_json = arguments.get(index + 1).cloned();
    }
    if let Some(index) = arguments.iter().position(|a| a == "--priority") {
        cfg.priority = match arguments.get(index + 1).map(String::as_str) {
            Some("normal") => bot1::config::Priority::Normal,
            Some("high") => bot1::config::Priority::High,
            _ => bot1::config::Priority::BelowNormal,
        };
    }
    cfg.validate()?;

    let keymap = bot1::keys::Keymap::load(&cfg.keymap_path);

    let request = platform::TargetRequest {
        configured: if value_of("--window", 0) == 0 {
            None
        } else {
            Some(value_of("--window", 0) as platform::Handle)
        },
        prefer_game_window: !has("--pick"),
        allow_prompt: true,
        remembered_handle: keymap.as_ref().and_then(|keymap| keymap.game_hwnd).map(|handle| handle as platform::Handle),
        remembered_title: keymap.as_ref().map(|keymap| keymap.game.clone()).unwrap_or_default(),
        ..Default::default()
    };
    let ui = bot1::ui::ui();
    ui.banner(env!("CARGO_PKG_VERSION"), &ui.platform_line());
    let Some(window) = platform::resolve_target_window(&request) else {
        ui.event("Run", "No window chosen, so there is nothing to drive.");
        ui.event(
            "Run",
            "`bot1 --list-windows` lists every window this build can see; \
             `bot1 --window HWND` picks one explicitly.",
        );
        if !platform::is_supported() {
            for line in platform::capability_report() {
                ui.event("Platform", line);
            }
        }
        return code(1);
    };
    let risk = platform::window_risk(Some(&window));
    if !risk.is_empty() && !dry_run {
        ui.event(
            "Window",
            format!(
                "Refusing to drive {}: {risk}.",
                platform::describe_window(Some(&window))
            ),
        );
        return code(1);
    }

    // ---- what it is about to drive, and with what keys -------------------
    ui.section("Window");
    ui.rows(&[
        ("game", platform::describe_window(Some(&window))),
        (
            "size",
            format!(
                "{}x{} captured{}",
                window.client.0,
                window.client.1,
                if window.minimized { ", currently minimized" } else { "" }
            ),
        ),
        (
            "choice",
            "this terminal, open browsers and editors, and this process's own windows \
             are never auto-selected. --pick or --window HWND drives a different one."
                .to_string(),
        ),
    ]);

    ui.section("Keys");
    match &keymap {
        Some(keymap) => ui.rows(&[
            ("file", keymap.path.clone().unwrap_or_else(|| "(built-in defaults)".to_string())),
            ("layout", keymap.describe()),
        ]),
        None => ui.event(
            "Check",
            format!(
                "No '{}': using built-in defaults. Run `bot1 --setup` (or --calibrate) \
                 once so the bot may only press the keys you actually use.",
                cfg.keymap_path
            ),
        ),
    }
    let keymap = keymap.unwrap_or_else(bot1::keys::Keymap::default_layout);

    let action_space = bot1::keys::ActionSpace::build(
        &keymap,
        cfg.max_held_keys,
        &cfg.speed_levels,
        true,
        true,
    );

    // ---- where this game's training lives ----------------------------------
    //
    // Every game gets its own directory under `runs/`, named after its window,
    // so the policy that learned one game is never resumed into another and
    // "keep training it" is the same command as "start training it". An
    // explicit --checkpoint-dir or BOT_CHECKPOINT_DIR opts out and is used
    // verbatim, which is what scripts already rely on.
    let explicit_checkpoints =
        has("--checkpoint-dir") || std::env::var("BOT_CHECKPOINT_DIR").is_ok();
    let explicit_log = has("--log") || std::env::var("BOT_LOG_JSON").is_ok();
    let game_title = if window.title.is_empty() {
        window.process.clone()
    } else {
        window.title.clone()
    };
    let run = if explicit_checkpoints {
        bot1::progress::RunDir::at(&cfg.checkpoint_dir)
    } else {
        bot1::progress::RunDir::new(&cfg.runs_dir, &game_title)
    };
    if !explicit_checkpoints {
        cfg.checkpoint_dir = run.checkpoints().to_string_lossy().to_string();
    }
    if !explicit_log {
        cfg.log_json = Some(run.log().to_string_lossy().to_string());
    }
    let _ = run.note_window(&game_title, &window.process);

    let existing = bot1::progress::read(&run);
    ui.section("Run");
    let mut rows = vec![(
        "folder",
        run.root.display().to_string(),
    )];
    if existing.steps > 0 {
        rows.push((
            "history",
            format!(
                "{} decision(s) over {} episode(s) {} {}",
                existing.steps,
                existing.episodes,
                ui.bullet(),
                existing.verdict()
            ),
        ));
    } else {
        rows.push((
            "history",
            "first session on this game: it starts knowing nothing. --report shows \
             what it has learned."
                .to_string(),
        ));
    }
    ui.rows(&rows);

    // ---- the model, in one screen ----------------------------------------
    ui.section("Model");
    ui.rows(&[
        (
            "observes",
            format!(
                "{}x{}x{} grey (+diff), repeat {}",
                cfg.frame_stack, cfg.frame_size, cfg.frame_size, cfg.action_repeat
            ),
        ),
        (
            "thinks",
            format!(
                "rope transformer {}d x {}L x {}H, GELU ffn {}, memory {} frames, \
                 {} patch token(s)/frame",
                cfg.embed_dim,
                cfg.transformer_layers,
                cfg.attention_heads,
                cfg.ffn_width(),
                cfg.mem_tokens,
                cfg.frame_tokens()
            ),
        ),
        (
            "learns",
            format!(
                "rollout {} x {} epoch(s) in {}-step minibatches, lr {:.0e}, entropy {}",
                cfg.rollout_steps, cfg.epochs_per_update, cfg.minibatch_size, cfg.adam_lr, cfg.entropy_coef
            ),
        ),
        ("acts", action_space.describe()),
    ]);

    let warnings = bot1::diagnostics::preflight(&cfg, Some(&keymap), Some(&window));
    if !warnings.is_empty() {
        ui.section("Checks");
        for warning in warnings {
            ui.event("Check", warning);
        }
    }
    if dry_run {
        ui.event("Run", "DRY RUN: nothing will be injected into any window.");
    }

    let env = bot1::game::RealGameEnv::new(&cfg, window.handle, &action_space, dry_run)?;
    let device = Default::default();
    let mut session = bot1::session::GameSession::<Backend>::new(
        &cfg,
        Box::new(env),
        action_space,
        Some(&keymap),
        device,
    )?;

    // The panel's heading: the game, not the program.
    session.set_title(game_title.clone());

    let mut control = bot1::control::SignalController::new(&cfg);
    control.start();

    // Checkpointing: one manager owns the directory, and the save closure is
    // what the run calls when the timer or the F9 key says it is due.
    let fresh = has("--fresh");
    let manager = bot1::checkpoint::CheckpointManager::new(&cfg.checkpoint_dir, cfg.checkpoint_keep);
    if let (true, Some(path)) = (cfg.resume && !fresh, manager.newest()) {
        match session.load_checkpoint(&path) {
            Ok(meta) => ui.event(
                "Resume",
                format!(
                    "Continuing {} at step {} ({} episode(s)) from {}.",
                    platform::describe_window(Some(&window)),
                    meta.step,
                    meta.episodes,
                    path.display()
                ),
            ),
            Err(error) => {
                // A checkpoint that no longer fits - a keymap that gained or
                // lost keys - still holds the part that understands the game.
                // Keep what transfers instead of starting from nothing.
                match session.warm_start(&path) {
                    Ok(matched) => ui.event(
                        "Resume",
                        format!(
                            "{} does not fit this action space ({error}), so its decision \
                             heads were re-initialised; the other {matched} weight(s) carry \
                             on from where they were.",
                            path.display()
                        ),
                    ),
                    Err(_) => ui.event(
                        "Resume",
                        format!("Checkpoint not usable ({error}); starting fresh."),
                    ),
                }
            }
        }
    } else if !manager.list_kept().is_empty() {
        ui.event(
            "Resume",
            format!(
                "{} checkpoint(s) in '{}' left alone (--fresh).",
                manager.list_kept().len(),
                cfg.checkpoint_dir
            ),
        );
    }

    ui.event(
        "Run",
        format!(
            "Playing {}. F8 pauses, F9 checkpoints, F10 quits, Ctrl+C stops and releases \
             every key.",
            platform::describe_window(Some(&window))
        ),
    );

    let steps = value_of("--steps", 0);
    let max_decisions = if steps == 0 { None } else { Some(steps) };
    {
        // Checkpoints are serialised on their own thread and only renamed here.
        //
        // Writing one means copying every weight out of the model and writing
        // them to disk - twice, once for the numbered file and once for the
        // rolling last.safetensors. That is hundreds of milliseconds to seconds
        // taken out of the control loop, which is a pause in the game for no
        // reason at all: the weights do not change while the bytes are being
        // written. So the bytes go to a thread, and this loop does only the
        // renames, which are file-system metadata.
        //
        // One write is in flight at a time: a slow disk costs one checkpoint
        // rather than a queue of them, which is the rule the interval itself
        // follows too.
        let saver = RefCell::new((manager, None::<Receiver<StagedCheckpoint>>, 0u64));

        let mut save = |session: &bot1::session::GameSession<Backend>, reason: &str| {
            let mut saver = saver.borrow_mut();
            let (manager, in_flight, last_step) = &mut *saver;

            // Collect a finished write before deciding whether another may start.
            if let Some(channel) = in_flight {
                match channel.try_recv() {
                    Ok((plan, Ok(()))) => {
                        let path = manager.finish_write(plan);
                        ui.event(
                            "Checkpoint",
                            format!(
                                "step {} ({reason}): {}",
                                session.total_steps,
                                path.file_name().unwrap_or_default().to_string_lossy()
                            ),
                        );
                        *in_flight = None;
                    }
                    Ok((_, Err(error))) => {
                        ui.event_as("Checkpoint", bot1::ui::Tone::Bad, format!("FAILED to save: {error}"));
                        *in_flight = None;
                    }
                    // Still writing: nothing to collect yet.
                    Err(TryRecvError::Empty) => {}
                    // The writing thread died without sending. Its temporary
                    // files are not trustworthy, but the run must not stop
                    // taking checkpoints because of it: forget the write and let
                    // the next interval try again.
                    Err(TryRecvError::Disconnected) => {
                        ui.event_as(
                            "Checkpoint",
                            bot1::ui::Tone::Bad,
                            "The writing thread stopped without finishing; that checkpoint is \
                             lost and the next interval will try again.",
                        );
                        *in_flight = None;
                    }
                }
            }

            let meta = session.meta();
            if meta.step == *last_step {
                // Nothing has been learned since the last request. Claiming a
                // sequence number for an identical checkpoint would push the
                // older, more interesting ones out of the kept set.
                return;
            }
            if in_flight.is_some() {
                // The previous write has not landed. Skipping this request rather
                // than queueing it is deliberate: a checkpoint is the newest
                // state, and the request comes round again in one interval.
                ui.event(
                    "Checkpoint",
                    format!(
                        "step {}: the previous write is still running; skipping this one.",
                        meta.step
                    ),
                );
                return;
            }

            let started = std::time::Instant::now();
            // In an `Arc` so the writing thread can have it without a second copy:
            // these are every weight in the model.
            let tensors = std::sync::Arc::new(session.policy.to_weights());
            let plan = manager.plan_write(&meta, reason);
            let (finished_tx, finished_rx) = std::sync::mpsc::channel::<StagedCheckpoint>();
            let spawned = std::thread::Builder::new()
                .name("bot1-checkpoint".to_string())
                .spawn({
                    let plan = plan.clone();
                    let meta = meta.clone();
                    let tensors = tensors.clone();
                    move || {
                        let outcome = bot1::checkpoint::CheckpointManager::stage_write(
                            &plan, &tensors, &meta,
                        );
                        let _ = finished_tx.send((plan, outcome));
                    }
                });
            match spawned {
                Ok(_) => *in_flight = Some(finished_rx),
                Err(error) => {
                    // No thread: stage it here rather than lose the checkpoint.
                    ui.event(
                        "Checkpoint",
                        format!("Could not start the writing thread ({error}); saving inline."),
                    );
                    match bot1::checkpoint::CheckpointManager::stage_write(&plan, &tensors, &meta)
                    {
                        Ok(()) => {
                            let path = manager.finish_write(plan);
                            ui.event(
                                "Checkpoint",
                                format!(
                                    "step {} ({reason}): {}",
                                    meta.step,
                                    path.file_name().unwrap_or_default().to_string_lossy()
                                ),
                            );
                        }
                        Err(error) => {
                            ui.event_as("Checkpoint", bot1::ui::Tone::Bad, format!("FAILED to save: {error}"))
                        }
                    }
                }
            }
            *last_step = meta.step;
            let elapsed = started.elapsed().as_secs_f64();
            if elapsed > 0.05 {
                ui.event(
                    "Checkpoint",
                    format!(
                        "Preparing a checkpoint cost {:.0}ms of the control loop; that is the \
                         weight copy, and it is the part that cannot leave this thread.",
                        elapsed * 1000.0
                    ),
                );
            }
        };

        session.run(Some(&mut control), max_decisions, Some(&mut save));

        // Whatever is still being written lands before the final line is printed,
        // so the list below is the list on disk.
        let (manager, in_flight, _) = saver.into_inner();
        let mut manager = manager;
        if let Some(channel) = in_flight {
            match channel.recv() {
                Ok((plan, Ok(()))) => {
                    let step = plan.step;
                    let path = manager.finish_write(plan);
                    ui.event(
                        "Checkpoint",
                        format!(
                            "step {step} (on the way out): {}",
                            path.file_name().unwrap_or_default().to_string_lossy()
                        ),
                    );
                }
                Ok((_, Err(error))) => {
                    ui.event_as("Checkpoint", bot1::ui::Tone::Bad, format!("FAILED to save: {error}"))
                }
                Err(_) => {}
            }
        }

        // A run that stopped for any reason still writes what it learned: a run
        // that ends without a checkpoint is a run that lost everything since the
        // last one.
        if session.total_steps > 0 {
            let final_path = run.model();
            match session.save_checkpoint(&final_path) {
                Ok(()) => ui.event(
                    "Checkpoint",
                    format!(
                        "step {} (final): {}",
                        session.total_steps,
                        final_path.display()
                    ),
                ),
                Err(error) => ui.event_as(
                    "Checkpoint",
                    bot1::ui::Tone::Bad,
                    format!("FAILED to save the final model: {error}"),
                ),
            }
        }
        control.stop();
        session.close();
        ui.finish();
        ui.event("Run", format!("Stopped ({}).", control.stop_reason()));
        if !manager.list_kept().is_empty() {
            ui.section("Checkpoints");
            for entry in manager.list_kept() {
                ui.event(
                    "Checkpoint",
                    format!("step {:>9}  ({})  {}", entry.step, entry.reason, entry.path),
                );
            }
            ui.blank();
        }
        ui.panel(
            "What next",
            &format!(
                "`bot1 --report` sums up what this game has learned so far, from the run \
                 directory '{}'. Run `bot1` again with the same game open and it continues \
                 from here.",
                run.root.display()
            ),
            bot1::ui::Tone::Accent,
        );
        return Ok(());
    }
}

/// A checkpoint's staged bytes on their way back from the writing thread, with
/// the outcome of writing them.
type StagedCheckpoint = (bot1::checkpoint::CheckpointPlan, anyhow::Result<()>);

/// `--report [DIR]`: what the bot has learned, without a game running.
///
/// With no argument it reports every game it has trained, newest first. With a
/// directory it reports that one - `runs/<game>` as printed at startup.
fn report(arguments: &[String]) -> anyhow::Result<()> {
    use bot1::ui::{Frame, Tone};
    use bot1::progress::{self, RunDir};
    let ui = bot1::ui::ui();
    ui.banner(env!("CARGO_PKG_VERSION"), &ui.platform_line());
    let target = arguments
        .iter()
        .position(|argument| argument == "--report")
        .and_then(|index| arguments.get(index + 1))
        .filter(|value| !value.starts_with("--"));

    let directories: Vec<RunDir> = match target {
        Some(path) => vec![RunDir::at(path)],
        None => {
            let base = arguments
                .iter()
                .position(|argument| argument == "--runs-dir")
                .and_then(|index| arguments.get(index + 1))
                .cloned()
                .unwrap_or_else(|| bot1::Config::default().env_overrides().runs_dir);
            RunDir::list(&base)
        }
    };

    if directories.is_empty() {
        ui.panel(
            "Nothing has been trained yet",
            &format!(
                "Start a game, then run `bot1` - it picks the window and begins learning, \
                 and writes everything it learns to '{}'. This command then reports on it \
                 with no game running.",
                bot1::Config::default().env_overrides().runs_dir
            ),
            Tone::Warn,
        );
        return code(0);
    }

    for directory in directories {
        let progress = progress::read(&directory);
        let (game, process) = progress
            .window
            .clone()
            .unwrap_or_else(|| ("(unknown game)".to_string(), "?".to_string()));
        let (badge, tone) = verdict_badge(&progress.verdict());
        let mut frame = Frame::new(format!("{game} ({process})"))
            .badge(badge, tone)
            .line(
                "trained",
                format!(
                    "{} decision(s) over {} episode(s), {} update(s)",
                    bot1::ui::thousands(progress.steps),
                    bot1::ui::thousands(progress.episodes),
                    progress.updates
                ),
                Tone::Value,
            )
            .line("folder", progress.directory.display().to_string(), Tone::Body);
        if progress.seconds > 0.0 {
            frame = frame.line(
                "ran for",
                format!("{:.0} minute(s) of wall clock", progress.seconds / 60.0),
                Tone::Body,
            );
        }
        if progress.checkpoints > 0 {
            frame = frame.line(
                "saved",
                format!(
                    "{} checkpoint(s), last model {} kB",
                    progress.checkpoints,
                    progress.model_bytes / 1024
                ),
                Tone::Body,
            );
        }
        let points: Vec<f32> = progress.rewards.iter().map(|value| *value as f32).collect();
        if let Some((start, end)) = progress.trend() {
            frame = frame.spark(
                "reward",
                format!("{start:+.4} -> {end:+.4}"),
                points,
                tone,
            );
        } else if let Some(last) = progress.rewards.last() {
            frame = frame.spark("reward", format!("{last:+.4}"), points, Tone::Dim);
        }
        if let Some(best) = progress
            .rewards
            .iter()
            .copied()
            .fold(None::<f64>, |acc, value| Some(acc.map_or(value, |a| a.max(value))))
        {
            frame = frame.line("best", format!("{best:+.4} reward/decision"), Tone::Body);
        }
        frame = frame.footer("one glyph per update   ·   right is more recent");
        ui.raw(&ui.render(&frame));
        ui.line(&progress.verdict());
        ui.blank();
    }
    ui.panel(
        "Training continues where it left off",
        "Run `bot1` again with the same game open: it reloads this directory, continues \
         from the newest checkpoint, and keeps reporting here.",
        Tone::Accent,
    );
    code(0)
}

/// `--demo [--steps N]`: the real loop and the real panel, on the synthetic
/// corridor, with nothing captured and nothing injected.
///
/// `--selftest` answers "does it learn"; this answers "what does a run look
/// like", which is a different question and the one a new user asks first.
fn demo(steps: u64) -> anyhow::Result<()> {
    use bot1::ui::ui;
    let ui = ui();
    ui.banner(env!("CARGO_PKG_VERSION"), &ui.platform_line());
    // Small, so updates land in seconds: the point is the interface, and a
    // panel that says "collecting the first rollout" for four minutes shows
    // none of it.
    let cfg = bot1::Config {
        frame_size: 64,
        frame_stack: 3,
        embed_dim: 64,
        transformer_layers: 2,
        attention_heads: 4,
        mem_tokens: 8,
        rollout_steps: 256,
        minibatch_size: 64,
        epochs_per_update: 2,
        target_fps: 0.0,
        // Nothing to inject, so nothing to wait for - and no reason to steal
        // F8/F9/F10 from whatever else is on this desktop. Ctrl+C still stops
        // it, and still through the same controller a real run uses, so the
        // panel comes down and the cursor comes back.
        wait_for_start: false,
        enable_hotkeys: false,
        ..bot1::Config::default()
    };
    ui.section("Demo");
    ui.rows(&[
        (
            "what",
            "the synthetic corridor `--selftest` uses: no window is captured and nothing is \
             injected anywhere."
                .to_string(),
        ),
        (
            "why",
            "it is the same session loop a game uses, so this is the panel a real run draws \
             - just small enough to update in seconds."
                .to_string(),
        ),
    ]);
    let space = bot1::synth::synthetic_action_space(&cfg);
    let env = bot1::synth::SyntheticEnv::new(&cfg, 32, 300);
    let device = Default::default();
    let mut session =
        bot1::session::GameSession::<Backend>::new(&cfg, Box::new(env), space, None, device)?;
    session.set_title("Synthetic corridor");
    let mut control = bot1::control::SignalController::new(&cfg);
    control.start();
    session.run(Some(&mut control), Some(steps.max(256)), None);
    control.stop();
    session.close();
    ui.finish();
    ui.event(
        "Run",
        format!("Demo finished after {} decision(s).", session.total_steps),
    );
    ui.blank();
    ui.panel(
        "That was the interface",
        "A real run looks the same, with the game's title in the heading and the numbers \
         coming from your machine. `bot1 --setup` is the first thing to run against a game.",
        bot1::ui::Tone::Accent,
    );
    Ok(())
}

/// `--ui-preview`: what a run looks like, with plausible numbers and no game.
///
/// It exists because the panel is the interface, and "does it look right" is a
/// question for the person using it - without a game, a window to capture, or a
/// model trained for an hour. The numbers are invented; the layout is the one a
/// real run draws.
fn ui_preview() -> anyhow::Result<()> {
    use bot1::ui::{Tone, ui};
    let ui = ui();
    ui.banner(env!("CARGO_PKG_VERSION"), &ui.platform_line());
    ui.section("Window");
    ui.rows(&[
        (
            "game",
            "'Minecraft 26.3 - Singleplayer' (javaw.exe) hwnd=158664994".to_string(),
        ),
        ("size", "1921x1080 captured".to_string()),
        (
            "choice",
            "this terminal, open browsers and editors, and this process's own windows are \
             never auto-selected. --pick or --window HWND drives a different one."
                .to_string(),
        ),
    ]);
    ui.section("Keys");
    ui.rows(&[
        ("file", "keymap.json".to_string()),
        ("layout", "hold=[W, A, S, D, SPACE]  tap=[E, LEFT]  mouse=[left]".to_string()),
    ]);
    ui.section("Checks");
    ui.event(
        "Check",
        "8 tap/click choices are enabled, so a large share of every rollout is spent on \
         clicking things. In a game with an inventory or an attack button this is mostly \
         wasted input.",
    );
    ui.event(
        "Run",
        "Playing 'Minecraft 26.3 - Singleplayer' (javaw.exe). F8 pauses, F9 checkpoints, \
         F10 quits, Ctrl+C stops and releases every key.",
    );

    let frames = bot1::ui::preview_frames();
    if !ui.live_enabled() {
        // Not a terminal: show the panel once, as text.
        if let Some(last) = frames.last() {
            ui.raw(&ui.render(last));
        }
        return Ok(());
    }
    for frame in &frames {
        ui.live_now(frame);
        std::thread::sleep(std::time::Duration::from_millis(60));
    }
    // A log line arriving mid-run: the panel steps aside, the line is kept.
    ui.event(
        "Checkpoint",
        "step 31,204 (every 10 minutes): 0000031204.safetensors",
    );
    ui.event_as(
        "Health",
        Tone::Warn,
        "The last update took 3.4s, so the model is competing with the game for the CPU; \
         raise --update-budget or lower --rollout if the frame rate matters.",
    );
    std::thread::sleep(std::time::Duration::from_millis(400));
    if let Some(last) = frames.last() {
        ui.live_now(last);
    }
    std::thread::sleep(std::time::Duration::from_millis(600));
    ui.finish();
    // Leave the panel on screen rather than erasing it: after the preview there
    // is nothing to redraw it, so this is the picture the person came for.
    if let Some(last) = frames.last() {
        ui.raw(&ui.render(last));
    }
    ui.blank();
    ui.line(
        "That is the whole interface: the header and the sections above, a live panel that \
         redraws in place while it learns, and log lines that are kept rather than \
         overwritten.",
    );
    ui.blank();
    Ok(())
}

/// The short word on a report panel, and how it should be coloured.
fn verdict_badge(verdict: &str) -> (&'static str, bot1::ui::Tone) {
    use bot1::ui::Tone;
    if verdict.starts_with("improving") {
        ("IMPROVING", Tone::Good)
    } else if verdict.starts_with("getting worse") {
        ("GETTING WORSE", Tone::Bad)
    } else if verdict.starts_with("flat") {
        ("FLAT", Tone::Warn)
    } else {
        ("TOO EARLY", Tone::Dim)
    }
}

/// `--benchmark [--window HANDLE] [--decisions N]`: what a step costs here.
fn benchmark(configured: u64, decisions: u64) -> anyhow::Result<()> {
    use bot1::platform;
    let cfg = bot1::Config::default();
    let keymap = bot1::keys::Keymap::load(&cfg.keymap_path)
        .unwrap_or_else(bot1::keys::Keymap::default_layout);
    let request = platform::TargetRequest {
        configured: if configured == 0 {
            None
        } else {
            Some(configured as platform::Handle)
        },
        prefer_game_window: true,
        allow_prompt: true,
        remembered_handle: keymap.game_hwnd.map(|handle| handle as platform::Handle),
        remembered_title: keymap.game.clone(),
        ..Default::default()
    };
    let Some(window) = platform::resolve_target_window(&request) else {
        bot1::ui::ui().event(
            "Bench",
            "Needs a window: pass --window HANDLE (see --list-windows).",
        );
        return code(1);
    };
    let device = Default::default();
    let status = bot1::diagnostics::benchmark::<Backend>(
        &cfg,
        &keymap,
        window.handle,
        decisions.max(1) as usize,
        device,
    );
    code(status)
}

/// `--hotkeys [--seconds N]`: register the control keys and report what arrived.
///
/// The keys are held by this process for those few seconds, which is exactly
/// what a real run does with them; nothing else is touched.
fn hotkeys(seconds: u64) -> anyhow::Result<()> {
    use bot1::platform;
    let cfg = bot1::Config::default();
    let mut hotkeys = platform::Hotkeys::new(
        &cfg.hotkey_pause,
        &cfg.hotkey_save,
        &cfg.hotkey_quit,
    );
    let ui = bot1::ui::ui();
    ui.section("Hotkeys");
    let registered = hotkeys.start();
    for message in hotkeys.drain_messages() {
        ui.event("Hotkeys", message);
    }
    if registered {
        ui.rows(&[(
            "registered",
            format!("{} (work from any window)", hotkeys.describe()),
        )]);
    } else {
        ui.event_as(
            "Hotkeys",
            bot1::ui::Tone::Warn,
            "No global hotkeys here; Ctrl+C still saves and quits.",
        );
    }
    ui.event(
        "Hotkeys",
        format!("Listening for {seconds}s - press them now."),
    );
    let started = std::time::Instant::now();
    while started.elapsed().as_secs() < seconds {
        for message in hotkeys.drain_messages() {
            ui.event("Hotkeys", message);
        }
        for action in hotkeys.poll() {
            ui.event_as(
                "Hotkeys",
                bot1::ui::Tone::Good,
                format!("{} pressed", action.name().to_uppercase()),
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(40));
    }
    hotkeys.stop();
    ui.event("Hotkeys", "Released.");
    Ok(())
}

/// `--calibrate [--window HANDLE] [--seconds N]`: watch the user play, and write
/// down what they actually press.
///
/// Nothing is injected: the hooks only listen, and every keystroke reaches the
/// game unchanged.
fn calibrate(configured: u64, seconds: u64) -> anyhow::Result<()> {
    use bot1::keys::parse_hotkey;
    use bot1::platform;
    let cfg = bot1::Config::default();
    let request = platform::TargetRequest {
        configured: if configured == 0 {
            None
        } else {
            Some(configured as platform::Handle)
        },
        // Not "the largest likely game": the user is about to point at the game
        // themselves, so guessing is worse than asking.
        prefer_game_window: false,
        allow_prompt: true,
        ..Default::default()
    };
    let Some(window) = platform::resolve_target_window(&request) else {
        bot1::ui::ui().event("Calibrate", "No window chosen.");
        return code(1);
    };
    let toggle_vk = parse_hotkey(&cfg.calibrate_key).map(|(_, vk)| vk).unwrap_or(0x77);
    let ui = bot1::ui::ui();
    ui.section("Key calibration");
    ui.rows(&[("driving", platform::describe_window(Some(&window)))]);
    ui.panel(
        "Play the game - the recorder is listening",
        &format!(
            "Press {} to start recording and {} again to stop and write the keymap. Nothing \
             is sent to the game: the hooks only listen, so play normally.",
            cfg.calibrate_key.to_uppercase(),
            cfg.calibrate_key.to_uppercase()
        ),
        bot1::ui::Tone::Accent,
    );

    let mut recorder = platform::KeyRecorder::start(toggle_vk, window.handle, true);
    if !recorder.installed() {
        ui.event_as(
            "Calibrate",
            bot1::ui::Tone::Bad,
            "The keyboard hooks could not be installed, so nothing can be recorded here. On \
             Linux this needs an X11 session; on macOS, the Accessibility permission.",
        );
        return code(1);
    }
    let started = std::time::Instant::now();
    let mut last_report = 0.0f64;
    loop {
        recorder.pump();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = started.elapsed().as_secs_f64();
        if elapsed - last_report >= 0.2 {
            last_report = elapsed;
            // The recorder is the whole screen while it runs: a five-second
            // silence with no counter is what makes people think it is broken.
            let recording = recorder.recording();
            ui.live_now(
                &bot1::ui::Frame::new("Recording your keys")
                    .badge(
                        if recording { "RECORDING" } else { "WAITING" },
                        if recording {
                            bot1::ui::Tone::Bad
                        } else {
                            bot1::ui::Tone::Dim
                        },
                    )
                    .line(
                        "elapsed",
                        format!("{:.0}s", elapsed),
                        bot1::ui::Tone::Value,
                    )
                    .line(
                        "events",
                        format!("{} key/mouse event(s) so far", recorder.events()),
                        bot1::ui::Tone::Body,
                    )
                    .footer(&format!(
                        "press {} to {} recording",
                        cfg.calibrate_key.to_uppercase(),
                        if recording { "stop" } else { "start" }
                    )),
            );
        }
        // Stop once recording has been switched off at least once.
        if !recorder.recording() && recorder.toggles() >= 1 {
            break;
        }
        if seconds > 0 && elapsed >= seconds as f64 {
            ui.event("Calibrate", "Time limit reached.");
            break;
        }
    }
    recorder.stop();

    let recording = recorder.snapshot(Some(&window));
    ui.finish();
    ui.section("What was recorded");
    ui.raw(&recording.report(cfg.calibrate_min_hold as f64));
    let keymap = recording.build_keymap(None, cfg.calibrate_min_hold as f64);
    if keymap.holds.is_empty() && keymap.vks.is_empty() {
        ui.event_as(
            "Calibrate",
            bot1::ui::Tone::Bad,
            format!(
                "Nothing was recorded, so nothing was saved. Press {} while the recorder is \
                 running, then play.",
                cfg.calibrate_key.to_uppercase()
            ),
        );
        return code(1);
    }
    let path = keymap.save(&cfg.keymap_path)?;
    ui.event_as(
        "Calibrate",
        bot1::ui::Tone::Good,
        format!("Saved {path}  ({})", keymap.describe()),
    );
    bot1::diagnostics::show_actions(&cfg, keymap);
    Ok(())
}

/// `--check-capture [--window HANDLE]`: what the capture actually yields.
fn check_capture(arguments: &[String], configured: u64) -> anyhow::Result<()> {
    use bot1::platform;
    let cfg = bot1::Config::default();
    let request = platform::TargetRequest {
        configured: if configured == 0 {
            None
        } else {
            Some(configured as platform::Handle)
        },
        prefer_game_window: true,
        allow_prompt: true,
        ..Default::default()
    };
    let Some(window) = platform::resolve_target_window(&request) else {
        bot1::ui::ui().event("Check", "No window chosen.");
        return code(1);
    };
    if !platform::is_supported() {
        for line in platform::capability_report() {
            bot1::ui::ui().event("Platform", line);
        }
        return code(1);
    }
    let _ = arguments;
    code(bot1::diagnostics::check_capture(&cfg, window.handle, 5.0))
}

/// Exit with a status, so shell users can chain on it.
fn code(status: i32) -> anyhow::Result<()> {
    std::process::exit(status);
}

/// `--platform`: what this build can and cannot do here, and why.
fn platform_report() -> anyhow::Result<()> {
    use bot1::platform;
    let ui = bot1::ui::ui();
    ui.banner(env!("CARGO_PKG_VERSION"), &ui.platform_line());
    ui.section("Platform");
    ui.rows(&[
        (
            "running",
            format!(
                "{} {} {} this build drives windows: {}",
                std::env::consts::OS,
                std::env::consts::ARCH,
                ui.bullet(),
                if platform::is_supported() { "yes" } else { "no" }
            ),
        ),
        (
            "terminal",
            format!(
                "{} columns x {} rows, {}{}",
                ui.width(),
                ui.height(),
                if ui.rich() { "colour and panels on" } else { "plain output" },
                if ui.live_enabled() { "" } else { ", no live panel (not a terminal)" }
            ),
        ),
    ]);
    let report = platform::capability_report();
    if report.is_empty() {
        ui.line(
            "Nothing to report: this platform's window layer is available here, and \
             every window can be listed, captured and driven.",
        );
    }
    for line in report {
        ui.event("Platform", line);
    }
    ui.blank();
    ui.line(
        "The model, the reward, the PPO update and --selftest need no window layer and \
         run on every platform this builds for.",
    );
    ui.blank();
    Ok(())
}

fn print_help() {
    use bot1::ui::ui;
    let ui = ui();
    ui.banner(env!("CARGO_PKG_VERSION"), &ui.platform_line());

    ui.section("The short version");
    ui.line("1. open the game, in a window (not exclusive fullscreen)");
    ui.line("2. bot1 --setup         learn your keys, then start learning");
    ui.line("3. let it run: F8 pauses, F9 saves, F10 quits");
    ui.line("4. bot1 --report        what it has learned so far");
    ui.line(
        "Running it again tomorrow continues from what it learned - same game, same folder.",
    );

    ui.section("Running with no flags");
    ui.rows(&[
        ("bot1", "pick a window, wait for the start key, learn".to_string()),
        ("--window HWND", "drive a specific window (see --list-windows)".to_string()),
        ("--steps N", "stop after N decisions".to_string()),
        ("--dry-run", "run the whole loop and inject nothing".to_string()),
    ]);

    ui.section("Modes");
    ui.rows(&[
        ("--setup", "calibrate the keys (watching you play), then train - the first run in one command".to_string()),
        ("--report [DIR]", "how much this bot has learned: steps, episodes, and whether the reward is improving. Works with no game running".to_string()),
        ("--selftest", "prove the identical algorithm learns a game it has never seen, on a synthetic corridor whose progress is known, with a random-action baseline and controls".to_string()),
        ("--demo", "the same loop and the same panel on that corridor, at a size that updates in seconds - no game needed".to_string()),
        ("--calibrate", "watch you play and write down the keys you use".to_string()),
        ("--actions", "print the whitelist and the action space it implies".to_string()),
        ("--list-windows", "every window the bot can see, and which it will not drive".to_string()),
        ("--check-capture", "watch a window for five seconds and report what the capture actually yields".to_string()),
        ("--benchmark", "what one decision costs here, model against capture".to_string()),
        ("--hotkeys", "register F8/F9/F10 and report what arrives".to_string()),
        ("--platform", "what this build can and cannot do on this OS".to_string()),
        ("--ui-preview", "what a training run's dashboard looks like, with invented numbers and no game".to_string()),
        ("--version", "print the version".to_string()),
    ]);

    ui.section("Options");
    ui.line(
        "--seed N, --steps N, --window N, --fps N, --focus once|always|never, --no-hotkeys, \
         --no-clip-cursor, --start-now, --pick, --dry-run, --fresh, --checkpoint-dir DIR, \
         --keep N, --log FILE, --priority normal|high|below-normal, --calibrate --seconds N, \
         --mouse-turn PX, --frame-size N, --rollout N and every other BOT_* tunable.",
    );
    ui.line(
        "--no-color strips the colour and the panels (BOT_UI=plain does the same); when the \
         output is a pipe or a file, that is what happens automatically, so `bot1 | tee` \
         still writes plain lines.",
    );

    ui.section("Hotkeys");
    ui.rows(&[
        ("F8", "pause / resume - nothing is sent while paused".to_string()),
        ("F9", "write a checkpoint now".to_string()),
        ("F10", "stop, releasing every key first".to_string()),
        ("Ctrl+C", "the same as F10, from any window".to_string()),
    ]);

    ui.section("Where things live");
    ui.line(
        "Each game's training lives in runs/<game>/: checkpoints, a log line per update, and \
         the model it ended with. --checkpoint-dir overrides that.",
    );
    ui.line(
        "The cursor is confined to the game window while the bot moves it, so a game that \
         frees the cursor (an inventory) cannot have clicks land outside it; \
         --no-clip-cursor turns that off, and BOT_CLIP_CURSOR=0 does the same.",
    );
    ui.line(
        "Every key is released on exit, including on Ctrl+C. Nothing is sent while the game \
         window is not in front, and the bot refuses to drive its own terminal.",
    );
    ui.blank();
}
