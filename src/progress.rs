//! Where one game's training lives, and what it has learned so far.
//!
//! A run only "learns over time" if the thing it learned last time is still
//! there next time - and it has to be there *for that game*, not mixed in with
//! whatever else the bot was playing. So every game gets its own directory,
//! named after the window it drives:
//!
//! ```text
//! runs/<game>/
//!   checkpoints/            last.safetensors + numbered copies + index.json
//!   training.jsonl          one JSON object per PPO update, appended forever
//!   model.safetensors       what the last run ended with
//!   session.json            the window this directory belongs to
//! ```
//!
//! The directory is the whole point: `--resume` looks only inside it, so a
//! policy trained on one game is never silently continued on another, and
//! `--report` can answer "is it getting better" without the user reading a log.
//!
//! Nothing here needs a window: `--report` works on a machine with no game
//! running, which is where most of the reading happens.

use std::path::{Path, PathBuf};

/// A directory name a human can read and any filesystem accepts.
///
/// `"Minecraft 26.3 - Singleplayer"` becomes `minecraft-26-3-singleplayer`.
/// Empty or entirely unprintable titles fall back to `game`, because a run that
/// cannot be named is still a run that must be saved.
pub fn slug(text: &str) -> String {
    let mut out = String::new();
    let mut last_dash = true;
    for character in text.chars() {
        if character.is_ascii_alphanumeric() {
            out.push(character.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
        if out.len() >= 48 {
            break;
        }
    }
    let trimmed = out.trim_matches('-').to_string();
    if trimmed.is_empty() {
        "game".to_string()
    } else {
        trimmed
    }
}

/// One game's training directory.
#[derive(Debug, Clone)]
pub struct RunDir {
    pub root: PathBuf,
}

impl RunDir {
    /// `<runs_dir>/<slug>`, created on demand.
    pub fn new(runs_dir: impl AsRef<Path>, title: &str) -> Self {
        let root = runs_dir.as_ref().join(slug(title));
        let _ = std::fs::create_dir_all(&root);
        Self { root }
    }

    /// The directory itself, for a caller that already has one.
    pub fn at(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    pub fn checkpoints(&self) -> PathBuf {
        self.root.join("checkpoints")
    }

    pub fn log(&self) -> PathBuf {
        self.root.join("training.jsonl")
    }

    /// Where a finished run leaves its model, for a reader that just wants the
    /// latest weights.
    pub fn model(&self) -> PathBuf {
        self.root.join("model.safetensors")
    }

    /// Remember which window this directory was created for, so a later
    /// `--report` can say what the bot is learning.
    pub fn note_window(&self, title: &str, process: &str) -> anyhow::Result<()> {
        let record = serde_json::json!({
            "title": title,
            "process": process,
            "name": slug(title),
        });
        std::fs::write(
            self.root.join("session.json"),
            serde_json::to_string_pretty(&record)?,
        )?;
        Ok(())
    }

    pub fn window(&self) -> Option<(String, String)> {
        let text = std::fs::read_to_string(self.root.join("session.json")).ok()?;
        let value: serde_json::Value = serde_json::from_str(&text).ok()?;
        Some((
            value.get("title")?.as_str()?.to_string(),
            value
                .get("process")
                .and_then(|p| p.as_str())
                .unwrap_or("?")
                .to_string(),
        ))
    }

    /// Every game directory under a runs directory, newest first.
    pub fn list(runs_dir: impl AsRef<Path>) -> Vec<RunDir> {
        let Ok(entries) = std::fs::read_dir(runs_dir.as_ref()) else {
            return Vec::new();
        };
        let mut found: Vec<(f64, RunDir)> = entries
            .flatten()
            .filter(|entry| entry.path().is_dir())
            .map(|entry| {
                let path = entry.path();
                let stamp = std::fs::metadata(&path)
                    .and_then(|meta| meta.modified())
                    .ok()
                    .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|age| age.as_secs_f64())
                    .unwrap_or(0.0);
                (stamp, RunDir::at(path))
            })
            .collect();
        found.sort_by(|a, b| b.0.total_cmp(&a.0));
        found.into_iter().map(|(_, dir)| dir).collect()
    }
}

/// One line of `training.jsonl`, as far as the report cares.
#[derive(Debug, Clone, Default)]
pub struct UpdateRecord {
    pub step: u64,
    pub mean_reward: Option<f64>,
    pub episodes: Option<u64>,
}

/// What one directory has accumulated.
#[derive(Debug, Clone, Default)]
pub struct Progress {
    pub directory: PathBuf,
    pub window: Option<(String, String)>,
    /// The furthest step any session reached. Step counters accumulate across
    /// resumes, so this is "how much training", not "how long this run was".
    pub steps: u64,
    pub episodes: u64,
    pub updates: u64,
    pub seconds: f64,
    /// Mean reward per decision, per update, in order.
    pub rewards: Vec<f64>,
    pub checkpoints: usize,
    pub model_bytes: u64,
}

impl Progress {
    /// Mean reward over the first and last thirds of the recorded updates.
    ///
    /// Neither number means anything alone - the reward is intrinsic and the
    /// scale drifts - but their *ratio* is the question the user is actually
    /// asking: is this getting better?
    pub fn trend(&self) -> Option<(f64, f64)> {
        if self.rewards.len() < 6 {
            return None;
        }
        let third = (self.rewards.len() / 3).max(1);
        let mean = |slice: &[f64]| slice.iter().sum::<f64>() / slice.len() as f64;
        Some((
            mean(&self.rewards[..third]),
            mean(&self.rewards[self.rewards.len() - third..]),
        ))
    }

    /// A plain sentence about whether it is learning.
    pub fn verdict(&self) -> String {
        let Some((start, end)) = self.trend() else {
            return if self.steps == 0 {
                "no training yet - run `bot1` with a game window and it starts here".to_string()
            } else {
                format!(
                    "{} step(s) recorded, not enough updates yet to say whether it is \
                     improving (needs about six)",
                    self.steps
                )
            };
        };
        // The reward has a negative scale floor, so compare against the
        // magnitude rather than dividing blindly through zero.
        let scale = start.abs().max(1e-6);
        let change = (end - start) / scale * 100.0;
        if change > 15.0 {
            format!("improving: reward per decision is up {change:.0}% since the early updates")
        } else if change < -15.0 {
            format!(
                "getting worse: reward per decision is down {:.0}% since the early updates - \
                 the game may have changed, or the action set no longer reaches new states",
                change.abs()
            )
        } else {
            format!(
                "flat: reward per decision has moved {change:+.0}%, which is noise. Longer \
                 training, or more keys with --calibrate, is what changes this"
            )
        }
    }
}

/// Read one directory: its log, its checkpoints, and the newest checkpoint's
/// counters.
pub fn read(dir: &RunDir) -> Progress {
    let mut progress = Progress {
        directory: dir.root.clone(),
        window: dir.window(),
        ..Default::default()
    };

    if let Ok(text) = std::fs::read_to_string(dir.log()) {
        for line in text.lines() {
            let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let record = UpdateRecord {
                step: value.get("step").and_then(|v| v.as_u64()).unwrap_or(0),
                mean_reward: value.get("mean_reward").and_then(|v| v.as_f64()),
                episodes: value.get("episodes").and_then(|v| v.as_u64()),
            };
            progress.updates += 1;
            progress.steps = progress.steps.max(record.step);
            progress.episodes = progress.episodes.max(record.episodes.unwrap_or(0));
            if let Some(reward) = record.mean_reward
                && reward.is_finite()
            {
                progress.rewards.push(reward);
            }
        }
    }

    // The checkpoints know the step and episode count even when the log has
    // been deleted, and the newest one is what a resume would actually load.
    if let Ok(entries) = std::fs::read_dir(dir.checkpoints()) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".safetensors") {
                progress.checkpoints += 1;
            }
        }
    }
    let newest = dir.checkpoints().join("last.safetensors");
    if newest.exists()
        && let Ok(weights) = crate::weights::Weights::open(&newest)
        && let Ok(meta) = crate::checkpoint::SessionMeta::from_metadata(&weights.metadata())
    {
        progress.steps = progress.steps.max(meta.step);
        progress.episodes = progress.episodes.max(meta.episodes);
        progress.seconds = progress.seconds.max(meta.seconds);
    }
    if let Ok(meta) = std::fs::metadata(dir.model()) {
        progress.model_bytes = meta.len();
    }
    progress
}

/// A block a human can read, which is the whole reason this module exists.
pub fn render(progress: &Progress) -> String {
    let mut out = String::new();
    let game = progress
        .window
        .as_ref()
        .map(|(title, process)| format!("'{title}' ({process})"))
        .unwrap_or_else(|| "(unknown game)".to_string());
    out.push_str(&format!("  game            {game}\n"));
    out.push_str(&format!("  directory       {}\n", progress.directory.display()));
    out.push_str(&format!(
        "  trained         {} decision(s) over {} episode(s), {} update(s)\n",
        progress.steps, progress.episodes, progress.updates
    ));
    if progress.seconds > 0.0 {
        out.push_str(&format!(
            "  last session    {:.0} minute(s)\n",
            progress.seconds / 60.0
        ));
    }
    if progress.checkpoints > 0 {
        out.push_str(&format!(
            "  saved           {} checkpoint(s), best model {} kB\n",
            progress.checkpoints,
            progress.model_bytes / 1024
        ));
    }
    if let Some((start, end)) = progress.trend() {
        out.push_str(&format!(
            "  reward/decision {start:+.4} early -> {end:+.4} recent (higher is better)\n"
        ));
    }
    if let Some(best) = progress
        .rewards
        .iter()
        .copied()
        .fold(None::<f64>, |acc, value| Some(acc.map_or(value, |a| a.max(value))))
    {
        out.push_str(&format!("  best update     {best:+.4} reward/decision\n"));
    }
    out.push_str(&format!("  verdict         {}\n", progress.verdict()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_slug_is_a_directory_name_a_human_can_read() {
        assert_eq!(slug("Minecraft 26.3 - Singleplayer"), "minecraft-26-3-singleplayer");
        assert_eq!(slug("  ...  "), "game");
        assert_eq!(slug("DOOM (1993)"), "doom-1993");
        // Long titles are cut, and the cut never leaves a trailing dash.
        let long = slug(&"a".repeat(200));
        assert!(long.len() <= 48);
        assert!(!long.ends_with('-'));
    }

    #[test]
    fn the_report_reads_what_the_log_says() {
        let dir = std::env::temp_dir().join("bot1-progress-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("runs").join("demo")).unwrap();
        let run = RunDir::at(dir.join("runs").join("demo"));
        std::fs::write(
            run.log(),
            "{\"step\":1024,\"mean_reward\":0.10,\"episodes\":3}\n\
             {\"step\":2048,\"mean_reward\":0.20,\"episodes\":5}\n\
             {\"step\":3072,\"mean_reward\":0.30,\"episodes\":7}\n",
        )
        .unwrap();
        let progress = read(&run);
        assert_eq!(progress.steps, 3072);
        assert_eq!(progress.episodes, 7);
        assert_eq!(progress.updates, 3);
        assert_eq!(progress.rewards, vec![0.10, 0.20, 0.30]);
        // Three updates is below the six the trend needs, so it says so rather
        // than inventing a direction.
        assert!(progress.verdict().contains("not enough updates"));
    }

    #[test]
    fn a_rising_trend_reads_as_improving() {
        let progress = Progress {
            steps: 8192,
            rewards: (0..9).map(|index| 0.10 + index as f64 * 0.05).collect(),
            ..Default::default()
        };
        assert!(progress.verdict().starts_with("improving"), "{}", progress.verdict());
    }
}
