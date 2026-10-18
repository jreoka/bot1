//! The terminal, as a user sees it.
//!
//! Every line the run prints goes through this module, so there is exactly one
//! place that decides what a line looks like. That is what stops the output from
//! being a wall of `[Tag] text`: the tags become coloured labels, the long
//! warnings wrap, the run summary becomes a panel, and the run itself becomes a
//! live dashboard that redraws in place instead of scrolling.
//!
//! Two modes, chosen by what stdout actually is:
//!
//! * **rich** - stdout is a terminal. Colour, box drawing, word wrap, and a live
//!   panel that is erased and redrawn each time it changes.
//! * **plain** - stdout is a pipe, a file, `NO_COLOR=1`, `TERM=dumb`, or
//!   `BOT_UI=plain`. Exactly the old `[Tag] message` lines, one per line, no
//!   escapes: `bot1 | tee run.log`, CI and a scrollback search all keep working.
//!
//! The cross-platform part is that "is this a VT terminal" is not assumed:
//!
//! * Windows consoles do not understand ANSI until the process asks, so the
//!   console mode is switched on here (`ENABLE_VIRTUAL_TERMINAL_PROCESSING`), and
//!   the output code page is set to UTF-8 for the box characters. If either
//!   fails - an old console, a redirected handle - the glyph set falls back to
//!   ASCII rather than printing mojibake.
//! * The size comes from the console on Windows, `TIOCGWINSZ` on unix, and
//!   `COLUMNS`/`LINES` if neither is available, so a panel is the width of the
//!   actual window instead of a guess.
//! * Nothing here can panic on a non-tty, and no line is ever wider than the
//!   window: a wrapped line breaks the "move up N lines and redraw" bookkeeping,
//!   so the composer pads to `width - 1` and truncates.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

// =====================================================================
// Colour
// =====================================================================

/// What a piece of text is for, rather than what colour it is.
///
/// The renderer maps it to a colour, and the plain renderer ignores it - which
/// is why a call site says `Tone::Warn` and never writes an escape code.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tone {
    /// Names, headings, the panel border.
    Accent,
    /// Something that worked.
    Good,
    /// Something the user should read.
    Warn,
    /// Something that is wrong.
    Bad,
    /// Secondary text: labels, units, hints.
    Dim,
    /// A number that matters.
    Value,
    /// Body text.
    Body,
}

impl Tone {
    fn sgr(self) -> &'static str {
        match self {
            // Bright variants: on the 16-colour Windows console the plain ones
            // are nearly invisible against a dark background.
            Tone::Accent => "\x1b[96m",
            Tone::Good => "\x1b[92m",
            Tone::Warn => "\x1b[93m",
            Tone::Bad => "\x1b[91m",
            Tone::Dim => "\x1b[90m",
            Tone::Value => "\x1b[97;1m",
            Tone::Body => "\x1b[0m",
        }
    }
}

/// The tag a message carries, and how it is drawn.
///
/// The tags are the ones the code already used (`[Run]`, `[Window]`, ...), so a
/// log line stays recognisable and the plain mode is unchanged.
fn tone_for_tag(tag: &str) -> Tone {
    match tag.to_ascii_lowercase().as_str() {
        "run" | "resume" | "control" => Tone::Accent,
        "check" | "config" | "warn" | "platform" => Tone::Warn,
        "health" | "error" | "fail" | "log" => Tone::Bad,
        "actions" | "calibrate" | "keymap" | "checkpoint" => Tone::Good,
        _ => Tone::Dim,
    }
}

// =====================================================================
// Characters
// =====================================================================

/// The box-drawing and bar characters, in a Unicode set and an ASCII one.
///
/// A terminal that cannot render UTF-8 gets the ASCII set: `+---+` and `###...`
/// read perfectly well, and half a box with a `?` in it does not.
#[derive(Clone, Copy)]
struct Glyphs {
    top_left: &'static str,
    top_right: &'static str,
    bottom_left: &'static str,
    bottom_right: &'static str,
    horizontal: &'static str,
    vertical: &'static str,
    dot: &'static str,
    accent: &'static str,
    bar_full: &'static str,
    bar_empty: &'static str,
    spark: [&'static str; 8],
    bullet: &'static str,
}

const UNICODE: Glyphs = Glyphs {
    top_left: "╭",
    top_right: "╮",
    bottom_left: "╰",
    bottom_right: "╯",
    horizontal: "─",
    vertical: "│",
    dot: "●",
    accent: "▌",
    bar_full: "█",
    bar_empty: "░",
    spark: ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"],
    bullet: "·",
};

const ASCII: Glyphs = Glyphs {
    top_left: "+",
    top_right: "+",
    bottom_left: "+",
    bottom_right: "+",
    horizontal: "-",
    vertical: "|",
    dot: "*",
    accent: "|",
    bar_full: "#",
    bar_empty: ".",
    spark: [".", ":", "-", "=", "+", "*", "#", "@"],
    bullet: "-",
};

// =====================================================================
// Text measurement
// =====================================================================

/// Display width of one character.
///
/// Only three cases matter: ANSI escapes (zero, handled by the caller), wide
/// East Asian characters and emoji (two columns), and combining marks (zero).
/// Everything else - including every box-drawing character - is one column.
fn char_width(character: char) -> usize {
    let code = character as u32;
    match code {
        0x0300..=0x036F | 0x200B..=0x200F | 0xFE00..=0xFE0F | 0xFEFF => 0,
        0x1100..=0x115F
        | 0x2E80..=0x303E
        | 0x3041..=0x33FF
        | 0x3400..=0x4DBF
        | 0x4E00..=0x9FFF
        | 0xA000..=0xA4CF
        | 0xAC00..=0xD7A3
        | 0xF900..=0xFAFF
        | 0xFE30..=0xFE6F
        | 0xFF00..=0xFF60
        | 0xFFE0..=0xFFE6
        | 0x1F300..=0x1FAFF => 2,
        _ => 1,
    }
}

/// `26624` -> `26,624`: a count a human can read at a glance.
pub fn thousands(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// Columns a string occupies on screen, ignoring ANSI escape sequences.
pub fn visible_width(text: &str) -> usize {
    let mut width = 0;
    let mut characters = text.chars();
    while let Some(character) = characters.next() {
        if character == '\x1b' {
            skip_escape(&mut characters);
        } else {
            width += char_width(character);
        }
    }
    width
}

/// Consume an escape sequence, leaving the iterator just past it.
///
/// Only CSI is emitted here (`ESC [ params letter`), and the introducer is
/// consumed before the parameters are scanned - reading `[` as the final byte
/// is what once made `\x1b[96m` measure four columns wide.
fn skip_escape(characters: &mut std::str::Chars<'_>) {
    let Some(next) = characters.next() else {
        return;
    };
    if next != '[' {
        // A two-character escape (ESC 7, ESC c, ...), already consumed.
        return;
    }
    for parameter in characters.by_ref() {
        if ('@'..='~').contains(&parameter) {
            return;
        }
    }
}

/// `text`, cut to at most `width` columns.
///
/// Escapes are copied through, so a truncated coloured string still resets the
/// colour it set instead of leaking it into the rest of the line.
fn truncate(text: &str, width: usize) -> String {
    let mut out = String::new();
    let mut used = 0;
    let mut characters = text.chars();
    let mut escapes: Vec<String> = Vec::new();
    while let Some(character) = characters.next() {
        if character == '\x1b' {
            let mut sequence = String::from(character);
            let mut rest = characters.as_str().chars();
            if let Some(next) = rest.next() {
                sequence.push(next);
                if next == '[' {
                    let mut consumed = 1;
                    for parameter in rest {
                        sequence.push(parameter);
                        consumed += 1;
                        if ('@'..='~').contains(&parameter) {
                            break;
                        }
                    }
                    for _ in 0..consumed {
                        characters.next();
                    }
                } else {
                    characters.next();
                }
            }
            out.push_str(&sequence);
            escapes.push(sequence);
            continue;
        }
        let cells = char_width(character);
        if used + cells > width {
            break;
        }
        used += cells;
        out.push(character);
    }
    // Truncation can cut a coloured string before its reset; put it back.
    if escapes.iter().any(|escape| escape.ends_with('m')) && !out.ends_with("\x1b[0m") {
        out.push_str("\x1b[0m");
    }
    out
}

/// Exactly `width` columns: truncated if longer, space-padded if shorter.
fn fit(text: &str, width: usize) -> String {
    let mut out = truncate(text, width);
    let used = visible_width(&out);
    out.push_str(&" ".repeat(width.saturating_sub(used)));
    out
}

/// Word-wrap to `width` columns, keeping explicit newlines.
///
/// A word longer than the line is cut rather than allowed to overflow: an
/// overlong path in a message must not push the panel out of shape.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(8);
    let mut lines = Vec::new();
    for paragraph in text.split('\n') {
        let mut line = String::new();
        for word in paragraph.split_whitespace() {
            let mut word = word.to_string();
            if !line.is_empty() && visible_width(&line) + 1 + visible_width(&word) > width {
                lines.push(std::mem::take(&mut line));
            }
            // A word longer than the line is cut rather than allowed to
            // overflow.
            while visible_width(&word) > width {
                let cut = truncate(&word, width);
                let taken = cut.chars().count();
                lines.push(cut);
                word = word.chars().skip(taken).collect();
            }
            if word.is_empty() {
                continue;
            }
            if line.is_empty() {
                line.push_str(&word);
            } else {
                line.push(' ');
                line.push_str(&word);
            }
        }
        lines.push(line);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

// =====================================================================
// The live panel
// =====================================================================

/// One row of the live panel.
#[derive(Clone)]
pub enum Row {
    /// A label and a value.
    Line {
        label: String,
        value: String,
        tone: Tone,
    },
    /// A label, a value, and a bar showing how far through something is.
    Meter {
        label: String,
        value: String,
        fraction: f32,
        tone: Tone,
    },
    /// A label, a value, and a sparkline of the recent history.
    Spark {
        label: String,
        value: String,
        points: Vec<f32>,
        tone: Tone,
    },
    /// An empty line, for grouping.
    Blank,
}

/// The panel a run redraws in place: a border, a heading, a badge, rows,
/// and a footer with the keys.
#[derive(Clone)]
pub struct Frame {
    pub heading: String,
    pub badge: String,
    pub badge_tone: Tone,
    pub rows: Vec<Row>,
    pub footer: String,
}

impl Frame {
    pub fn new(heading: impl Into<String>) -> Self {
        Self {
            heading: heading.into(),
            badge: String::new(),
            badge_tone: Tone::Accent,
            rows: Vec::new(),
            footer: String::new(),
        }
    }

    pub fn badge(mut self, badge: impl Into<String>, tone: Tone) -> Self {
        self.badge = badge.into();
        self.badge_tone = tone;
        self
    }

    pub fn line(mut self, label: impl Into<String>, value: impl Into<String>, tone: Tone) -> Self {
        self.rows.push(Row::Line {
            label: label.into(),
            value: value.into(),
            tone,
        });
        self
    }

    pub fn meter(
        mut self,
        label: impl Into<String>,
        value: impl Into<String>,
        fraction: f32,
        tone: Tone,
    ) -> Self {
        self.rows.push(Row::Meter {
            label: label.into(),
            value: value.into(),
            fraction,
            tone,
        });
        self
    }

    pub fn spark(
        mut self,
        label: impl Into<String>,
        value: impl Into<String>,
        points: Vec<f32>,
        tone: Tone,
    ) -> Self {
        self.rows.push(Row::Spark {
            label: label.into(),
            value: value.into(),
            points,
            tone,
        });
        self
    }

    pub fn blank(mut self) -> Self {
        self.rows.push(Row::Blank);
        self
    }

    pub fn footer(mut self, footer: impl Into<String>) -> Self {
        self.footer = footer.into();
        self
    }
}

// =====================================================================
// The renderer
// =====================================================================

struct LiveState {
    /// Lines currently on screen at the bottom of the terminal.
    drawn: usize,
    last_draw: Instant,
    frame: Option<Frame>,
    cursor_hidden: bool,
    width_at_draw: usize,
}

/// The one renderer. Reached through [`ui`].
pub struct Ui {
    color: bool,
    glyphs: Glyphs,
    /// stdout is a terminal, so a redrawn panel is meaningful at all.
    interactive: bool,
    width: AtomicUsize,
    height: AtomicUsize,
    live: Mutex<LiveState>,
}

/// How long to wait between redraws.
///
/// The collection loop calls [`Ui::live`] on every decision; 100 ms keeps the
/// numbers readable (a counter flickering at 60 Hz is unreadable) and costs
/// nothing next to a forward pass.
const REDRAW: Duration = Duration::from_millis(100);

/// No panel on a terminal this small: the border alone would leave no room.
const MIN_PANEL_WIDTH: usize = 46;
const MIN_PANEL_HEIGHT: usize = 12;

static INSTANCE: OnceLock<Ui> = OnceLock::new();

/// The process-wide renderer, configured once from the environment.
pub fn ui() -> &'static Ui {
    INSTANCE.get_or_init(Ui::from_env)
}

impl Ui {
    fn from_env() -> Self {
        let stdout_is_terminal = std::io::stdout().is_terminal();
        let forced = std::env::var("BOT_UI").ok().unwrap_or_default();
        let forced = forced.trim().to_ascii_lowercase();
        let plain = forced == "plain" || forced == "off" || forced == "0";
        let force_rich = forced == "rich" || forced == "on" || forced == "1";

        let no_color = std::env::var_os("NO_COLOR").is_some();
        let term_dumb = std::env::var("TERM")
            .map(|term| term.eq_ignore_ascii_case("dumb"))
            .unwrap_or(false);
        let color_pipe = std::env::var("CLICOLOR_FORCE")
            .map(|value| value != "0")
            .unwrap_or(false)
            || std::env::var("FORCE_COLOR")
                .map(|value| value != "0")
                .unwrap_or(false);

        // Windows needs the console switched over before any of this is true.
        #[cfg(windows)]
        let (virtual_terminal, utf8_console) = prepare_windows_console();
        #[cfg(not(windows))]
        let (virtual_terminal, utf8_console) = (true, true);

        let color = !plain
            && !no_color
            && !term_dumb
            && virtual_terminal
            && ((stdout_is_terminal && std::env::var("CLICOLOR").map(|v| v != "0").unwrap_or(true))
                || force_rich
                || color_pipe);

        let unicode = !plain && utf8_console && locale_is_utf8();

        // A panic with the cursor hidden leaves a terminal with no cursor and a
        // panel frozen mid-frame. The hook is cheap and runs before the panic
        // message, so the message is readable too.
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic| {
            let mut out = std::io::stdout().lock();
            let _ = out.write_all(b"\x1b[?25h\x1b[0J");
            let _ = out.flush();
            drop(out);
            previous_hook(panic);
        }));

        Self {
            color,
            glyphs: if unicode { UNICODE } else { ASCII },
            interactive: stdout_is_terminal && !plain && virtual_terminal,
            width: AtomicUsize::new(0),
            height: AtomicUsize::new(0),
            live: Mutex::new(LiveState {
                drawn: 0,
                last_draw: Instant::now() - REDRAW,
                frame: None,
                cursor_hidden: false,
                width_at_draw: 0,
            }),
        }
    }

    /// A renderer with a fixed mode, for tests and for `--ui-preview`.
    #[doc(hidden)]
    pub fn fixed(color: bool, unicode: bool, interactive: bool) -> Self {
        Self {
            color,
            glyphs: if unicode { UNICODE } else { ASCII },
            interactive,
            width: AtomicUsize::new(0),
            height: AtomicUsize::new(0),
            live: Mutex::new(LiveState {
                drawn: 0,
                last_draw: Instant::now() - REDRAW,
                frame: None,
                cursor_hidden: false,
                width_at_draw: 0,
            }),
        }
    }

    /// The separator used between items in a one-line summary.
    pub fn bullet(&self) -> &'static str {
        self.glyphs.bullet
    }

    /// `linux x86_64 · 132x40` - what this build is running on, and in what.
    pub fn platform_line(&self) -> String {
        let base = format!("{} {}", std::env::consts::OS, std::env::consts::ARCH);
        if self.interactive {
            format!("{base} {} {}x{}", self.glyphs.bullet, self.width(), self.height())
        } else {
            base
        }
    }

    // ---- capabilities ----

    /// Colour and box drawing are on.
    pub fn rich(&self) -> bool {
        self.color
    }

    /// A redrawn panel is possible here.
    pub fn live_enabled(&self) -> bool {
        self.interactive
    }

    pub fn width(&self) -> usize {
        let current = terminal_size().map(|(width, _)| width).unwrap_or(0);
        let remembered = self.width.load(Ordering::Relaxed);
        if current > 0 {
            if current != remembered {
                self.width.store(current, Ordering::Relaxed);
            }
            return current.clamp(24, 240);
        }
        if remembered > 0 {
            return remembered;
        }
        let fallback = std::env::var("COLUMNS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(100);
        self.width.store(fallback, Ordering::Relaxed);
        fallback.clamp(24, 240)
    }

    /// The width every rendered line must fit inside.
    ///
    /// One column short of the terminal, always: a terminal wraps a line that
    /// exactly fills the last column, and a wrapped line breaks the bookkeeping
    /// that erases the panel before redrawing it - the panel would smear down
    /// the screen instead of redrawing in place.
    fn usable(&self) -> usize {
        self.width().saturating_sub(1).max(20)
    }

    pub fn height(&self) -> usize {
        let current = terminal_size().map(|(_, height)| height).unwrap_or(0);
        let remembered = self.height.load(Ordering::Relaxed);
        if current > 0 {
            self.height.store(current, Ordering::Relaxed);
            return current.clamp(10, 200);
        }
        if remembered > 0 {
            return remembered;
        }
        let fallback = std::env::var("LINES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(30);
        self.height.store(fallback, Ordering::Relaxed);
        fallback.clamp(10, 200)
    }

    // ---- paint ----

    fn paint(&self, tone: Tone, text: &str) -> String {
        if !self.color || text.is_empty() {
            return text.to_string();
        }
        format!("{}{}\x1b[0m", tone.sgr(), text)
    }

    fn write(&self, text: &str) {
        let mut out = std::io::stdout().lock();
        let _ = out.write_all(text.as_bytes());
        let _ = out.flush();
    }

    // ---- static blocks ----

    /// The opening panel: the wordmark, and what this build is.
    pub fn banner(&self, version: &str, platform: &str) {
        self.finish();
        if !self.rich() {
            self.write(&format!("bot1 {version} - {platform}\n"));
            return;
        }
        let wordmark = [
            "██████╗  ██████╗ ████████╗ ██╗",
            "██╔══██╗██╔═══██╗╚══██╔══╝███║",
            "██████╔╝██║   ██║   ██║   ╚██║",
            "██╔══██╗██║   ██║   ██║    ██║",
            "██████╔╝╚██████╔╝   ██║    ██║",
            "╚═════╝  ╚═════╝    ╚═╝    ╚═╝",
        ];
        let shades = [
            Tone::Accent,
            Tone::Accent,
            Tone::Accent,
            Tone::Accent,
            Tone::Dim,
            Tone::Dim,
        ];
        let mut out = String::from("\n");
        if self.glyphs.top_left == "+" {
            // No UTF-8 console: the wordmark would be garbage, so say the name.
            out.push_str(&format!(
                "   {}\n",
                self.paint(Tone::Accent, "  b o t 1  ")
            ));
        } else {
            for (index, line) in wordmark.iter().enumerate() {
                out.push_str(&format!("   {}\n", self.paint(shades[index], line)));
            }
        }
        out.push_str(&format!(
            "   {}   {} {}\n\n",
            self.paint(Tone::Dim, "learns to play from pixels"),
            self.paint(Tone::Value, version),
            self.paint(Tone::Dim, &format!("{} {platform}", self.glyphs.bullet)),
        ));
        self.write(&out);
    }

    /// A section heading: `▌ WINDOW ────────────────`.
    pub fn section(&self, title: &str) {
        if !self.rich() {
            self.write(&format!("\n{title}\n"));
            return;
        }
        let width = self.usable();
        let label = title.to_uppercase();
        let head = format!("{} {}", self.paint(Tone::Accent, self.glyphs.accent), self.paint(Tone::Value, &label));
        let used = 2 + visible_width(&label);
        let fill = width.saturating_sub(used + 1);
        self.write(&format!(
            "\n{head} {}\n",
            self.paint(Tone::Dim, &self.glyphs.horizontal.repeat(fill))
        ));
    }

    /// Label/value rows under a section heading.
    ///
    /// The label column sizes itself to the longest label in the block, so a
    /// label is never cut in half by a width guessed at the call site.
    pub fn rows(&self, rows: &[(&str, String)]) {
        let label_width = rows
            .iter()
            .map(|(label, _)| visible_width(label))
            .max()
            .unwrap_or(4)
            .clamp(4, 16);
        for (label, value) in rows {
            if !self.rich() {
                self.write(&format!("  {label}: {value}\n"));
                continue;
            }
            let width = self.usable();
            let head = format!("  {} ", self.paint(Tone::Dim, &fit(label, label_width)));
            let indent = 2 + label_width + 1;
            let mut first = true;
            for line in wrap(value, width.saturating_sub(indent)) {
                if first {
                    self.write(&format!("{head}{}\n", self.paint(Tone::Body, &line)));
                    first = false;
                } else {
                    self.write(&format!("{}{}\n", " ".repeat(indent), self.paint(Tone::Body, &line)));
                }
            }
        }
    }

    /// A framed paragraph, for "what to do next" moments.
    pub fn panel(&self, title: &str, body: &str, tone: Tone) {
        if !self.rich() {
            self.write(&format!("{title}\n"));
            for line in body.lines() {
                self.write(&format!("  {line}\n"));
            }
            return;
        }
        let inner = self.usable().min(96) - 2;
        let mut out = String::from("\n");
        out.push_str(&self.border(
            self.glyphs.top_left,
            self.glyphs.top_right,
            &self.paint(tone, title),
            "",
            inner,
        ));
        for line in wrap(body, inner.saturating_sub(2)) {
            out.push_str(&format!(
                "{} {} {}\n",
                self.paint(Tone::Dim, self.glyphs.vertical),
                fit(&self.paint(Tone::Body, &line), inner.saturating_sub(2)),
                self.paint(Tone::Dim, self.glyphs.vertical),
            ));
        }
        out.push_str(&self.border(
            self.glyphs.bottom_left,
            self.glyphs.bottom_right,
            "",
            "",
            inner,
        ));
        self.announce(&out);
    }

    // ---- messages ----

    /// One tagged line, wrapped and coloured.
    ///
    /// This is the replacement for `println!("[Tag] ...")`: in plain mode it
    /// prints exactly that, and in rich mode it prints a coloured label and a
    /// wrapped message.
    pub fn event(&self, tag: &str, message: impl AsRef<str>) {
        self.event_as(tag, tone_for_tag(tag), message);
    }

    /// [`Ui::event`], with the colour chosen by the caller.
    ///
    /// For the messages whose tag does not imply their seriousness: a failed
    /// checkpoint is still a `[Checkpoint]` line, and it is still red.
    pub fn event_as(&self, tag: &str, tone: Tone, message: impl AsRef<str>) {
        let message = message.as_ref();
        if !self.rich() {
            self.announce(&format!("[{tag}] {message}\n"));
            return;
        }
        let width = self.usable();
        let label = format!(
            "{} {} ",
            self.paint(tone, self.glyphs.dot),
            self.paint(tone, &fit(&tag.to_uppercase(), 10)),
        );
        // Two spaces of indent, then "●", a space, a ten-column tag and a
        // space: wrapped lines line up under the first word of the message.
        let indent = 2 + 1 + 1 + 10 + 1;
        let mut out = String::new();
        for (index, line) in wrap(message, width.saturating_sub(indent)).iter().enumerate() {
            if index == 0 {
                out.push_str(&format!("  {label}{}\n", self.paint(Tone::Body, line)));
            } else {
                out.push_str(&format!("{}{}\n", " ".repeat(indent), self.paint(Tone::Body, line)));
            }
        }
        self.announce(&out);
    }

    /// A message with no tag: body text, already formatted.
    pub fn line(&self, text: &str) {
        if !self.rich() {
            self.announce(&format!("{text}\n"));
            return;
        }
        let mut out = String::new();
        for line in wrap(text, self.usable().saturating_sub(2)) {
            out.push_str(&format!("  {}\n", self.paint(Tone::Body, &line)));
        }
        self.announce(&out);
    }

    /// An unformatted block, printed as-is (already rendered by the caller).
    pub fn raw(&self, text: &str) {
        self.announce(text);
    }

    pub fn blank(&self) {
        self.announce("\n");
    }

    // ---- the live panel ----

    /// A frame as text, without touching the live region.
    ///
    /// For the places a panel is a report rather than a dashboard: `--report`
    /// and `--ui-preview` print one, then print under it.
    pub fn render(&self, frame: &Frame) -> String {
        self.compose(frame)
    }

    /// Draw the panel, replacing the previous one.
    ///
    /// Throttled: the caller may call this on every decision.
    pub fn live(&self, frame: &Frame) {
        self.draw(frame, false, false);
    }

    /// Draw now, throttle or not.
    pub fn live_now(&self, frame: &Frame) {
        self.draw(frame, true, false);
    }

    /// Redraw the current panel at once - after a pause, a resume, a save.
    pub fn refresh(&self) {
        let mut live = self.live.lock().unwrap();
        if let Some(frame) = live.frame.clone() {
            live.last_draw = Instant::now() - REDRAW;
            drop(live);
            self.draw(&frame, true, false);
        }
    }

    /// Remove the panel and give the cursor back. Safe to call more than once.
    pub fn finish(&self) {
        let mut live = self.live.lock().unwrap();
        let mut out = String::new();
        if live.drawn > 0 {
            out.push_str(&erase(live.drawn));
            live.drawn = 0;
        }
        if live.cursor_hidden {
            out.push_str("\x1b[?25h");
            live.cursor_hidden = false;
        }
        live.frame = None;
        drop(live);
        if !out.is_empty() {
            self.write(&out);
        }
    }

    fn draw(&self, frame: &Frame, force: bool, _final_frame: bool) {
        if !self.interactive {
            return;
        }
        let mut live = self.live.lock().unwrap();
        if !force && live.last_draw.elapsed() < REDRAW {
            // Keep the newest frame so the next log line redraws the right one.
            live.frame = Some(frame.clone());
            return;
        }
        let width = self.usable();
        let mut out = String::new();
        if live.drawn > 0 {
            // After a resize the old lines may have reflowed into a different
            // number of rows, and erasing by the old count would take log lines
            // with it. Start a fresh panel instead and leave the old one where
            // it is: a duplicated panel is recoverable, eaten log lines are not.
            if live.width_at_draw == width {
                out.push_str(&erase(live.drawn));
            }
            live.drawn = 0;
        }
        if !live.cursor_hidden {
            out.push_str("\x1b[?25l");
            live.cursor_hidden = true;
        }
        let rendered = self.compose(frame);
        let count = rendered.matches('\n').count();
        out.push_str(&rendered);
        live.drawn = count;
        live.width_at_draw = width;
        live.last_draw = Instant::now();
        live.frame = Some(frame.clone());
        drop(live);
        self.write(&out);
    }

    /// Print a block above the panel: erase the panel, print, redraw.
    fn announce(&self, block: &str) {
        let mut live = self.live.lock().unwrap();
        let mut out = String::new();
        if live.drawn > 0 {
            if live.width_at_draw == self.usable() {
                out.push_str(&erase(live.drawn));
            }
            live.drawn = 0;
        }
        out.push_str(block);
        if let Some(frame) = live.frame.clone() {
            let rendered = self.compose(&frame);
            out.push_str(&rendered);
            live.drawn = rendered.matches('\n').count();
            live.last_draw = Instant::now();
        }
        drop(live);
        self.write(&out);
    }

    // ---- composition ----

    fn compose(&self, frame: &Frame) -> String {
        let width = self.usable();
        let height = self.height();
        if width < MIN_PANEL_WIDTH || height < MIN_PANEL_HEIGHT {
            return self.compose_compact(frame, width);
        }
        let inner = width - 2;
        let mut out = String::new();

        // Top border: heading on the left, badge on the right.
        let heading = self.paint(Tone::Value, &frame.heading);
        let badge = if frame.badge.is_empty() {
            String::new()
        } else {
            self.paint(frame.badge_tone, &frame.badge)
        };
        out.push_str(&self.border(
            self.glyphs.top_left,
            self.glyphs.top_right,
            &heading,
            &badge,
            inner,
        ));

        // The label column fits the longest label in the frame, so a caller
        // never has to guess a width and have a label cut in half by it.
        let label_width = frame
            .rows
            .iter()
            .map(|row| match row {
                Row::Line { label, .. } | Row::Meter { label, .. } | Row::Spark { label, .. } => {
                    visible_width(label)
                }
                Row::Blank => 0,
            })
            .max()
            .unwrap_or(6)
            .clamp(6, 12);
        // One column of air on the right, so nothing is ever drawn against the
        // border: a bar that touches the frame reads as a clipping bug.
        let head_width = 1 + label_width + 1;
        for row in &frame.rows {
            let content = match row {
                Row::Blank => " ".repeat(inner),
                Row::Line { label, value, tone } => {
                    let head = format!(
                        " {} ",
                        self.paint(Tone::Dim, &fit(&format!("{label} "), label_width)),
                    );
                    let space = inner.saturating_sub(head_width + 1);
                    format!("{head}{}", fit(&self.paint(*tone, value), space))
                }
                Row::Meter {
                    label,
                    value,
                    fraction,
                    tone,
                } => {
                    let head = format!(
                        " {} ",
                        self.paint(Tone::Dim, &fit(&format!("{label} "), label_width)),
                    );
                    let space = inner.saturating_sub(head_width + 1);
                    let (bar_width, value_text) = if space < visible_width(value) + 12 {
                        // Too narrow for a bar: the number is the whole row.
                        (0, fit(&self.paint(*tone, value), space))
                    } else {
                        (
                            space - visible_width(value) - 2,
                            format!("{} ", self.paint(*tone, value)),
                        )
                    };
                    let used = visible_width(&value_text);
                    let bar = self.bar(*fraction, bar_width, *tone);
                    let gap = space.saturating_sub(used + bar_width);
                    format!("{head}{}{}{bar}", value_text, " ".repeat(gap))
                }
                Row::Spark {
                    label,
                    value,
                    points,
                    tone,
                } => {
                    let head = format!(
                        " {} ",
                        self.paint(Tone::Dim, &fit(&format!("{label} "), label_width)),
                    );
                    let space = inner.saturating_sub(head_width + 1);
                    let value_text = format!("{}   ", self.paint(*tone, value));
                    let used = visible_width(&value_text);
                    // The trend sits right after the number it belongs to: a
                    // sparkline pushed to the far edge reads as unrelated.
                    let spark_width = space.saturating_sub(used).min(32);
                    let spark = self.sparkline(points, spark_width, *tone);
                    let gap = space.saturating_sub(used + spark_width);
                    format!("{head}{value_text}{spark}{}", " ".repeat(gap))
                }
            };
            out.push_str(&format!(
                "{}{}{}\n",
                self.paint(Tone::Dim, self.glyphs.vertical),
                fit(&content, inner),
                self.paint(Tone::Dim, self.glyphs.vertical),
            ));
        }

        out.push_str(&self.border(
            self.glyphs.bottom_left,
            self.glyphs.bottom_right,
            &self.paint(Tone::Dim, &frame.footer),
            "",
            inner,
        ));
        out
    }

    /// `╭─ left ─────── right ─╮`, exactly `inner + 2` columns wide.
    fn border(
        &self,
        left_corner: &str,
        right_corner: &str,
        left: &str,
        right: &str,
        inner: usize,
    ) -> String {
        let mut left_text = if left.is_empty() {
            String::new()
        } else {
            format!("{} {} ", self.paint(Tone::Dim, self.glyphs.horizontal), left)
        };
        let mut right_text = if right.is_empty() {
            String::new()
        } else {
            format!(" {} {}", right, self.paint(Tone::Dim, self.glyphs.horizontal))
        };
        // The heading is the point of the border; the badge is the decoration,
        // so it is the badge that goes when there is not room for both.
        if visible_width(&left_text) + visible_width(&right_text) > inner {
            right_text.clear();
        }
        if visible_width(&left_text) > inner {
            left_text = format!(
                "{} {} ",
                self.paint(Tone::Dim, self.glyphs.horizontal),
                truncate(left, inner.saturating_sub(3))
            );
        }
        let fill = inner.saturating_sub(visible_width(&left_text) + visible_width(&right_text));
        format!(
            "{}{}{}{}{}\n",
            self.paint(Tone::Dim, left_corner),
            left_text,
            self.paint(Tone::Dim, &self.glyphs.horizontal.repeat(fill)),
            right_text,
            self.paint(Tone::Dim, right_corner),
        )
    }

    /// A filled bar: `████████░░░░░░ 58%`.
    fn bar(&self, fraction: f32, width: usize, tone: Tone) -> String {
        if width == 0 {
            return String::new();
        }
        let filled = ((fraction.clamp(0.0, 1.0)) * width as f32).round() as usize;
        let filled = filled.min(width);
        format!(
            "{}{}",
            self.paint(tone, &self.glyphs.bar_full.repeat(filled)),
            self.paint(Tone::Dim, &self.glyphs.bar_empty.repeat(width - filled)),
        )
    }

    /// A sparkline of `points`, scaled to its own range.
    ///
    /// Its own range is deliberate: the reward's absolute scale means nothing
    /// ("is 0.041 good?"), while its shape is the question actually being asked.
    fn sparkline(&self, points: &[f32], width: usize, tone: Tone) -> String {
        if points.is_empty() || width == 0 {
            return String::new();
        }
        let visible: Vec<f32> = points
            .iter()
            .rev()
            .take(width)
            .rev()
            .copied()
            .filter(|value| value.is_finite())
            .collect();
        if visible.is_empty() {
            return String::new();
        }
        let low = visible.iter().copied().fold(f32::INFINITY, f32::min);
        let high = visible.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let span = (high - low).max(1e-9);
        let glyphs = self.glyphs.spark;
        let rendered: String = visible
            .iter()
            .map(|value| {
                let position = (((value - low) / span) * (glyphs.len() - 1) as f32).round() as usize;
                glyphs[position.min(glyphs.len() - 1)]
            })
            .collect();
        self.paint(tone, &rendered)
    }

    /// Too narrow or too short for a border: one dense line.
    fn compose_compact(&self, frame: &Frame, width: usize) -> String {
        let mut parts: Vec<String> = Vec::new();
        if !frame.badge.is_empty() {
            parts.push(self.paint(frame.badge_tone, &frame.badge));
        }
        for row in &frame.rows {
            match row {
                Row::Blank => {}
                Row::Line { label, value, tone } | Row::Meter { label, value, tone, .. } => {
                    parts.push(format!(
                        "{} {}",
                        self.paint(Tone::Dim, label),
                        self.paint(*tone, value)
                    ))
                }
                Row::Spark { label, value, tone, .. } => parts.push(format!(
                    "{} {}",
                    self.paint(Tone::Dim, label),
                    self.paint(*tone, value)
                )),
            }
        }
        let separator = format!(" {} ", self.glyphs.bullet);
        let mut line = String::new();
        for part in parts {
            let candidate = if line.is_empty() {
                part.clone()
            } else {
                format!("{line}{separator}{part}")
            };
            if visible_width(&candidate) > width.saturating_sub(1) {
                break;
            }
            line = candidate;
        }
        let mut out = format!("{}\n", fit(&line, width));
        if !frame.footer.is_empty() && self.height() > 6 {
            let footer = self.paint(Tone::Dim, &frame.footer);
            out.push_str(&format!("{}\n", fit(&footer, width)));
        }
        out
    }
}

/// Move up `lines`, back to column 0, and erase everything from there down.
///
/// The `\x1b[G` is not cosmetic: cursor-up keeps the column it started at, so
/// after a panel whose last line ended at column 99 the next draw would begin
/// at column 99 and smear to the right instead of redrawing in place.
fn erase(lines: usize) -> String {
    format!("\x1b[{lines}A\x1b[G\x1b[0J")
}

// =====================================================================
// The terminal itself
// =====================================================================

fn locale_is_utf8() -> bool {
    for name in ["LC_ALL", "LC_CTYPE", "LANG"] {
        if let Ok(value) = std::env::var(name)
            && !value.is_empty()
        {
            let lower = value.to_ascii_lowercase();
            return lower.contains("utf-8") || lower.contains("utf8");
        }
    }
    // Nothing says otherwise: every terminal in 2026 is UTF-8, and a wrong
    // answer only costs box-drawing characters.
    true
}

#[cfg(windows)]
fn prepare_windows_console() -> (bool, bool) {
    use windows::Win32::System::Console::{
        GetConsoleMode, GetStdHandle, SetConsoleMode, SetConsoleOutputCP,
        ENABLE_PROCESSED_OUTPUT, ENABLE_VIRTUAL_TERMINAL_PROCESSING, STD_OUTPUT_HANDLE,
    };
    let mut virtual_terminal = false;
    unsafe {
        if let Ok(handle) = GetStdHandle(STD_OUTPUT_HANDLE) {
            let mut mode = windows::Win32::System::Console::CONSOLE_MODE(0);
            if GetConsoleMode(handle, &mut mode).is_ok() {
                let wanted = windows::Win32::System::Console::CONSOLE_MODE(
                    mode.0 | ENABLE_PROCESSED_OUTPUT.0 | ENABLE_VIRTUAL_TERMINAL_PROCESSING.0,
                );
                virtual_terminal = SetConsoleMode(handle, wanted).is_ok();
            }
        }
    }
    // The box characters are multi-byte; a console left on code page 850 prints
    // them as garbage. Setting 65001 is what makes them appear.
    let utf8 = unsafe { SetConsoleOutputCP(65001).is_ok() };
    (virtual_terminal, utf8)
}

#[cfg(windows)]
fn terminal_size() -> Option<(usize, usize)> {
    use windows::Win32::System::Console::{
        GetConsoleScreenBufferInfo, GetStdHandle, CONSOLE_SCREEN_BUFFER_INFO, STD_OUTPUT_HANDLE,
    };
    unsafe {
        let handle = GetStdHandle(STD_OUTPUT_HANDLE).ok()?;
        let mut info = std::mem::zeroed::<CONSOLE_SCREEN_BUFFER_INFO>();
        if GetConsoleScreenBufferInfo(handle, &mut info).is_err() {
            return None;
        }
        let width = (info.srWindow.Right - info.srWindow.Left + 1).max(0) as usize;
        let height = (info.srWindow.Bottom - info.srWindow.Top + 1).max(0) as usize;
        if width == 0 || height == 0 {
            None
        } else {
            Some((width, height))
        }
    }
}

#[cfg(unix)]
fn terminal_size() -> Option<(usize, usize)> {
    unsafe {
        let mut size: libc::winsize = std::mem::zeroed();
        // 1 is stdout; asking for the size of the terminal the output goes to.
        if libc::ioctl(1, libc::TIOCGWINSZ, &mut size) != 0 {
            return None;
        }
        if size.ws_col == 0 || size.ws_row == 0 {
            return None;
        }
        Some((size.ws_col as usize, size.ws_row as usize))
    }
}

#[cfg(not(any(unix, windows)))]
fn terminal_size() -> Option<(usize, usize)> {
    None
}

// =====================================================================
// A cheap, fake-data preview so the panel can be seen without a game
// =====================================================================

/// Frames of a plausible run, for `--ui-preview`.
pub fn preview_frames() -> Vec<Frame> {
    let mut frames = Vec::new();
    for tick in 0..40 {
        let progress = tick as f32 / 39.0;
        let steps = 24_800 + (progress * 12_000.0) as u64;
        let reward = 0.012 + progress * 0.031;
        let points: Vec<f32> = (0..24)
            .map(|index| {
                let phase = (index as f32 * 0.7 + tick as f32 * 0.35).sin();
                reward + phase * 0.004 + index as f32 * 0.0005
            })
            .collect();
        frames.push(
            Frame::new("Minecraft 26.3 - Singleplayer")
                .badge("RUNNING", Tone::Good)
                .line(
                    "steps",
                    format!(
                        "{steps:>9}   {:.1}/s   {:.0}% in game   {:02}:{:02}:{:02}   {} episode(s)",
                        41.2 + progress * 2.0,
                        97.0,
                        12, 34, 0, 14
                    ),
                    Tone::Value,
                )
                .spark(
                    "reward",
                    format!("{reward:+.4}/step"),
                    points,
                    Tone::Good,
                )
                .line(
                    "update",
                    format!(
                        "loss {:+.3}   pi {:+.3}   v {:.3}   ent {:.2}   kl {:+.4}   0.42s/8mb",
                        -0.021 + progress * 0.01,
                        -0.008,
                        0.004,
                        1.24,
                        0.0012
                    ),
                    Tone::Body,
                )
                .meter(
                    "rollout",
                    format!("{:.0}%  {}/1024", progress * 100.0, (progress * 1024.0) as u32),
                    progress,
                    Tone::Accent,
                )
                .line(
                    "actions",
                    "W 42%   S 11%   A 9%   D 8%      W/A/S/D held 78% of steps",
                    Tone::Body,
                )
                .footer("F8 pause   F9 save   F10 quit   Ctrl+C stop"),
        );
    }
    frames
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_escape_sequence_has_no_width() {
        assert_eq!(visible_width("\x1b[96mabc\x1b[0m"), 3);
        assert_eq!(visible_width("plain"), 5);
        assert_eq!(visible_width(""), 0);
    }

    #[test]
    fn wide_characters_take_two_columns() {
        assert_eq!(visible_width("日本"), 4);
        assert_eq!(visible_width("a日b"), 4);
        // Box drawing is one column, not the two a byte count would say.
        assert_eq!(visible_width("╭─╮"), 3);
    }

    #[test]
    fn fit_is_always_exactly_the_width_it_was_asked_for() {
        for width in [0, 1, 5, 40, 80] {
            for text in ["", "a", "hello world", "╭─╮ ╰─╯"] {
                assert_eq!(visible_width(&fit(text, width)), width, "fit({text:?}, {width})");
            }
        }
    }

    #[test]
    fn truncate_keeps_the_colour_reset_it_cut_through() {
        let cut = truncate("\x1b[96mabcdef\x1b[0m", 3);
        assert!(cut.ends_with("\x1b[0m"), "{cut:?}");
        assert_eq!(visible_width(&cut), 3);
    }

    #[test]
    fn wrapping_keeps_every_line_inside_the_width() {
        let text = "a long message that will certainly need to wrap more than once because \
                    it keeps going and going and going";
        for line in wrap(text, 20) {
            assert!(visible_width(&line) <= 20, "{line:?}");
        }
        // A single unbreakable word is cut rather than allowed to overflow.
        for line in wrap(&"x".repeat(100), 10) {
            assert!(visible_width(&line) <= 10);
        }
    }

    #[test]
    fn a_sparkline_has_one_glyph_per_point_and_never_exceeds_its_width() {
        let ui = Ui::fixed(false, true, false);
        let line = ui.sparkline(&[0.0, 0.5, 1.0, 0.25], 4, Tone::Good);
        assert_eq!(line.chars().count(), 4);
        let clipped = ui.sparkline(&[1.0; 100], 8, Tone::Good);
        assert_eq!(clipped.chars().count(), 8);
        assert!(ui.sparkline(&[], 8, Tone::Good).is_empty());
    }

    #[test]
    fn a_bar_shows_the_ends_and_the_middle() {
        let ui = Ui::fixed(false, true, false);
        assert_eq!(ui.bar(0.0, 4, Tone::Accent), "░░░░");
        assert_eq!(ui.bar(1.0, 4, Tone::Accent), "████");
        assert_eq!(ui.bar(0.5, 4, Tone::Accent), "██░░");
        // Out-of-range input is clamped rather than panicking.
        assert_eq!(ui.bar(4.0, 2, Tone::Accent), "██");
        assert_eq!(ui.bar(-1.0, 2, Tone::Accent), "░░");
    }

    #[test]
    fn every_line_of_a_panel_is_the_same_width() {
        // A fixed width without a terminal: the composer asks `terminal_size`,
        // which returns nothing here, so COLUMNS is the fallback.
        unsafe { std::env::set_var("COLUMNS", "80") };
        let ui = Ui::fixed(true, true, true);
        ui.width.store(80, Ordering::Relaxed);
        ui.height.store(30, Ordering::Relaxed);
        let frame = Frame::new("Minecraft 26.3 - Singleplayer")
            .badge("RUNNING", Tone::Good)
            .line("steps", "24812   41.2/s   12 episode(s)", Tone::Value)
            .spark("reward", "+0.0412/step", vec![0.01, 0.02, 0.03, 0.02], Tone::Good)
            .meter("rollout", "58%  604/1024", 0.58, Tone::Accent)
            .footer("F8 pause   F9 save   F10 quit");
        let composed = ui.compose(&frame);
        let widths: Vec<usize> = composed.lines().map(visible_width).collect();
        // One column short of the terminal, every line, or the terminal wraps
        // the panel and the redraw smears down the screen.
        assert!(
            widths.iter().all(|width| *width == 79),
            "widths {widths:?}\n{composed}"
        );
    }

    #[test]
    fn a_narrow_terminal_falls_back_to_one_line() {
        let ui = Ui::fixed(true, true, true);
        ui.width.store(30, Ordering::Relaxed);
        ui.height.store(8, Ordering::Relaxed);
        let frame = Frame::new("game")
            .badge("RUNNING", Tone::Good)
            .line("steps", "24812", Tone::Value);
        let composed = ui.compose(&frame);
        assert_eq!(composed.lines().count(), 1, "{composed}");
        assert!(visible_width(composed.trim_end()) <= 29);
    }

    #[test]
    fn plain_mode_is_the_old_output_exactly() {
        let ui = Ui::fixed(false, false, false);
        assert!(!ui.rich());
        assert!(!ui.live_enabled());
        assert_eq!(tone_for_tag("Run"), Tone::Accent);
        assert_eq!(tone_for_tag("Health"), Tone::Bad);
        assert_eq!(tone_for_tag("Check"), Tone::Warn);
    }
}
