//! The terminal, as a user sees it.
//!
//! Every line the run prints goes through this module, so there is exactly one
//! place that decides what a line looks like. That is what stops the output from
//! being a wall of `[Tag] text`: the tags become coloured labels, the long
//! warnings wrap, the run summary becomes a panel, and the run itself becomes a
//! live dashboard on the whole terminal.
//!
//! Two modes, chosen by what stdout actually is:
//!
//! * **rich** - stdout is a terminal. Colour, box drawing, word wrap, and a
//!   dashboard that owns the screen: the alternate screen buffer, the log tail
//!   on top, the panel at the bottom, every row painted in place.
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
//!   window: a wrapped line corrupts the row below it, so the composer pads to
//!   `width - 1` and truncates.

use std::collections::VecDeque;
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::vision::Picture;

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

/// How a picture's pixels can be coloured.
///
/// A capability, not a preference: it is decided once from the environment, and
/// the picture is drawn through the shade ramp either way - the colour is what
/// makes a corridor readable at 32 columns, not what makes it visible.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PixelColour {
    /// 24-bit foreground (`38;2;r;g;b`).
    True,
    /// The xterm 256-colour cube (`38;5;n`).
    Palette,
    /// No colour: the shade ramp is the whole picture.
    Off,
}

/// The nearest xterm-256 colour to an RGB triple.
///
/// The cube is 6x6x6 levels of 0, 95, 135, 175, 215, 255, which is not a linear
/// ramp - so the nearest level is found by looking at the actual values rather
/// than dividing by 51.
fn ansi256(red: u8, green: u8, blue: u8) -> u8 {
    const LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];
    let nearest = |value: u8| {
        let mut best = 0usize;
        for (index, level) in LEVELS.iter().enumerate() {
            if value.abs_diff(*level) < value.abs_diff(LEVELS[best]) {
                best = index;
            }
        }
        best as usize
    };
    let (r, g, b) = (nearest(red), nearest(green), nearest(blue));
    let cube = 16 + 36 * r as u8 + 6 * g as u8 + b as u8;
    let distance = |colour: (i32, i32, i32)| {
        let (cr, cg, cb) = colour;
        3 * (cr - red as i32).pow(2) + 5 * (cg - green as i32).pow(2) + 2 * (cb - blue as i32).pow(2)
    };
    let cube_distance = distance((
        LEVELS[r] as i32,
        LEVELS[g] as i32,
        LEVELS[b] as i32,
    ));
    // The 24-step grey ramp (232..255) is finer than the cube for greys, and most
    // of a game's picture is grey-ish, so it is worth looking at both.
    let average = (red as i32 + green as i32 + blue as i32) / 3;
    let grey_index = ((average - 8 + 5) / 10).clamp(0, 23);
    let grey = (8 + 10 * grey_index) as u8;
    if distance((grey as i32, grey as i32, grey as i32)) < cube_distance {
        232 + grey_index as u8
    } else {
        cube
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
    /// The mark a value gets when it had to be cut: `…`, or `...` in ASCII.
    ellipsis: &'static str,
    /// Dark to bright, one glyph per pixel of the vision view.
    shade: [&'static str; 5],
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
    ellipsis: "…",
    shade: [" ", "░", "▒", "▓", "█"],
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
    ellipsis: "...",
    shade: [" ", ".", ":", "*", "#"],
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
    /// A label and a pixelated picture of what the bot is looking at.
    ///
    /// `picture` is one small RGB frame - `1 / 0` or `[0, 1]`, the renderer
    /// detects the range the same way the fingerprint does. The renderer decides
    /// how many rows and columns the panel has room for, so the caller passes the
    /// frame it has, not a size.
    View {
        label: String,
        picture: Picture,
    },
    /// An empty line, for grouping.
    Blank,
}

impl Row {
    /// The label column this row draws with, whatever kind of row it is.
    fn label(&self) -> &str {
        match self {
            Row::Line { label, .. }
            | Row::Meter { label, .. }
            | Row::Spark { label, .. }
            | Row::View { label, .. } => label,
            Row::Blank => "",
        }
    }

    /// The value this row draws, without its bar or its sparkline.
    ///
    /// The value is what the row is for, so the layout measures it to decide how
    /// much room the picture beside the numbers may have. A bar or a sparkline
    /// shrinks instead: they say less than the number they sit next to.
    fn value(&self) -> &str {
        match self {
            Row::Line { value, .. } | Row::Meter { value, .. } | Row::Spark { value, .. } => value,
            Row::View { .. } | Row::Blank => "",
        }
    }
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

    /// One grey frame, pixelated, as a block inside the panel.
    ///
    /// The panel draws it in colour when the terminal and the renderer can: three
    /// equal channels are grey, so this is the same picture either way.
    pub fn view(
        mut self,
        label: impl Into<String>,
        plane: Vec<f32>,
        height: usize,
        width: usize,
    ) -> Self {
        self.rows.push(Row::View {
            label: label.into(),
            picture: Picture::from_grey(&plane, height, width),
        });
        self
    }

    /// The bot's own frame, in colour, as a block inside the panel.
    pub fn picture(mut self, label: impl Into<String>, picture: Picture) -> Self {
        self.rows.push(Row::View {
            label: label.into(),
            picture,
        });
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
    /// The alternate screen is ours and has to be given back.
    fullscreen: bool,
    /// What `event` would have printed, newest last: the log tail above the
    /// panel. Kept as rendered lines, escapes and all.
    log: VecDeque<String>,
    frame: Option<Frame>,
    last_draw: Instant,
}

/// The one renderer. Reached through [`ui`].
pub struct Ui {
    color: bool,
    glyphs: Glyphs,
    /// stdout is a terminal, so a redrawn panel is meaningful at all.
    interactive: bool,
    /// How the panel's picture may be coloured.
    pixels: PixelColour,
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

/// Take the whole screen: the alternate buffer, cleared, with no cursor.
///
/// The dashboard is the terminal rather than a block that moves down it. That is
/// what makes a repaint honest: nothing is ever printed into the scrollback
/// while a run is live, so no frame can be left half-erased above the next one,
/// nothing scrolls under the cursor, and the row a line lands on never depends on
/// how many rows the previous frame happened to draw.
const ENTER_DASHBOARD: &str = "\x1b[?1049h\x1b[H\x1b[2J\x1b[?25l";

/// Give the terminal back: the cursor first, then the screen the user was on.
const LEAVE_DASHBOARD: &str = "\x1b[?25h\x1b[?1049l";

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

        // Colour for the picture only: the text already uses the 16-colour
        // palette, which every terminal has. A picture does not survive 16
        // colours, and 24-bit is not something to guess at - so it is asked for
        // by name (`COLORTERM`), with the 256-colour cube as the fallback.
        let colourterm = std::env::var("COLORTERM")
            .unwrap_or_default()
            .to_ascii_lowercase();
        let term = std::env::var("TERM").unwrap_or_default().to_ascii_lowercase();
        let pixels = if !color {
            PixelColour::Off
        } else if colourterm.contains("truecolor") || colourterm.contains("24bit") {
            PixelColour::True
        } else if colourterm.contains("256") || term.contains("256") {
            PixelColour::Palette
        } else if cfg!(windows) {
            // The Windows console has understood 24-bit SGR since long before it
            // understood to set `COLORTERM`; it is the one case worth assuming.
            PixelColour::True
        } else {
            PixelColour::Off
        };

        // A panic on the dashboard would otherwise print its message into the
        // alternate screen - which the shell throws away on exit - with the
        // cursor still hidden. Give the terminal back first, then let the default
        // hook report: the message lands in the scrollback where it can be read.
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic| {
            echo::on();
            let mut out = std::io::stdout().lock();
            let _ = out.write_all(LEAVE_DASHBOARD.as_bytes());
            let _ = out.flush();
            drop(out);
            previous_hook(panic);
        }));

        Self {
            color,
            glyphs: if unicode { UNICODE } else { ASCII },
            interactive: stdout_is_terminal && !plain && virtual_terminal,
            pixels,
            width: AtomicUsize::new(0),
            height: AtomicUsize::new(0),
            live: Mutex::new(LiveState {
                fullscreen: false,
                log: VecDeque::new(),
                last_draw: Instant::now() - REDRAW,
                frame: None,
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
            // Tests and `--ui-preview` get the full 24-bit path: it is the one
            // with escapes in it, so it is the one worth exercising.
            pixels: if color { PixelColour::True } else { PixelColour::Off },
            width: AtomicUsize::new(0),
            height: AtomicUsize::new(0),
            live: Mutex::new(LiveState {
                fullscreen: false,
                log: VecDeque::new(),
                last_draw: Instant::now() - REDRAW,
                frame: None,
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
    /// exactly fills the last column, and the wrapped half lands on the row
    /// below - which on the dashboard is the row after it, so the bottom of a
    /// panel would be drawn on top of itself.
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
        self.announce(&format!(
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
        let mut out = String::new();
        for (label, value) in rows {
            if !self.rich() {
                out.push_str(&format!("  {label}: {value}\n"));
                continue;
            }
            let width = self.usable();
            let head = format!("  {} ", self.paint(Tone::Dim, &fit(label, label_width)));
            let indent = 2 + label_width + 1;
            let mut first = true;
            for line in wrap(value, width.saturating_sub(indent)) {
                if first {
                    out.push_str(&format!("{head}{}\n", self.paint(Tone::Body, &line)));
                    first = false;
                } else {
                    out.push_str(&format!("{}{}\n", " ".repeat(indent), self.paint(Tone::Body, &line)));
                }
            }
        }
        self.announce(&out);
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
        self.draw(frame, false);
    }

    /// Draw now, throttle or not.
    pub fn live_now(&self, frame: &Frame) {
        self.draw(frame, true);
    }

    /// Redraw the current panel at once - after a pause, a resume, a save.
    pub fn refresh(&self) {
        let frame = {
            let mut live = self.live.lock().unwrap();
            let Some(frame) = live.frame.clone() else {
                return;
            };
            live.last_draw = Instant::now() - REDRAW;
            frame
        };
        self.draw(&frame, true);
    }

    /// Give the terminal back. Safe to call more than once.
    ///
    /// The dashboard lives on the alternate screen, so from the caller's point of
    /// view this is a no-op on a scrollback: the lines printed before the run are
    /// still there, and the next `event` (the run summary) prints under them.
    pub fn finish(&self) {
        let mut live = self.live.lock().unwrap();
        live.frame = None;
        if !live.fullscreen {
            return;
        }
        live.fullscreen = false;
        live.log.clear();
        echo::on();
        self.write(LEAVE_DASHBOARD);
    }

    fn draw(&self, frame: &Frame, force: bool) {
        if !self.interactive {
            return;
        }
        let mut live = self.live.lock().unwrap();
        if !force && live.last_draw.elapsed() < REDRAW {
            // Keep the newest frame so the next log line repaints the right one.
            live.frame = Some(frame.clone());
            return;
        }
        let mut out = String::new();
        if !live.fullscreen {
            echo::off();
            out.push_str(ENTER_DASHBOARD);
            live.fullscreen = true;
            live.log.clear();
        }
        out.push_str(&self.screen(&live.log, Some(frame)));
        live.frame = Some(frame.clone());
        live.last_draw = Instant::now();
        // Written with the lock still held: two threads that both decide to
        // repaint must not interleave halves of two different screens.
        self.write(&out);
    }

    /// Print a block above the panel.
    ///
    /// On the dashboard it becomes the log tail and the screen is repainted; on a
    /// terminal with no dashboard yet it is just printed.
    fn announce(&self, block: &str) {
        let mut live = self.live.lock().unwrap();
        if !live.fullscreen {
            self.write(block);
            return;
        }
        self.push_log(&mut live, block);
        let frame = live.frame.clone();
        let out = self.screen(&live.log, frame.as_ref());
        live.last_draw = Instant::now();
        self.write(&out);
    }

    /// Keep what was printed as lines of the log tail, newest last.
    fn push_log(&self, live: &mut LiveState, block: &str) {
        let width = self.usable();
        let mut lines: Vec<&str> = block.split('\n').collect();
        // The block ends with a newline; that is a line terminator, not a row.
        if lines.last() == Some(&"") {
            lines.pop();
        }
        for line in lines {
            // Padded to the width so a line can never wrap onto the row below it,
            // which on a dashboard would land in the middle of the panel.
            live.log.push_back(fit(line, width));
        }
        // Bounded: the visible window is the terminal's height, and the rest is
        // only there for the case where the window gets taller mid-run.
        let keep = self.height().saturating_mul(4).max(48);
        while live.log.len() > keep {
            live.log.pop_front();
        }
    }

    /// The whole terminal as one write: the log tail on top, the panel below it.
    ///
    /// Every row is addressed absolutely (`ESC[row;1H`) and cleared to the end of
    /// the line, so a repaint never depends on what the last frame drew, on where
    /// the cursor was left, or on how the window has been resized since. The rows
    /// below the panel are the panel's own last row, so nothing is left over.
    fn screen(&self, log: &VecDeque<String>, frame: Option<&Frame>) -> String {
        let width = self.usable();
        let height = self.height();
        let mut panel: Vec<String> = match frame {
            Some(frame) => self
                .compose(frame)
                .lines()
                .map(|line| fit(line, width))
                .collect(),
            None => Vec::new(),
        };
        // `compose` already sizes the panel to the terminal. This is the backstop
        // that makes that a guarantee: a panel one row too tall would scroll the
        // screen and take the top of the dashboard with it.
        panel.truncate(height);
        let log_rows = height.saturating_sub(panel.len());
        let shown: Vec<&String> = log.iter().rev().take(log_rows).rev().collect();
        let blanks = log_rows.saturating_sub(shown.len());
        let mut out = String::with_capacity((width + 10) * height);
        for row in 0..blanks {
            out.push_str(&format!("\x1b[{};1H\x1b[K", row + 1));
        }
        for (index, line) in shown.iter().enumerate() {
            out.push_str(&format!("\x1b[{};1H{}\x1b[K", blanks + index + 1, line));
        }
        for (index, line) in panel.iter().enumerate() {
            out.push_str(&format!("\x1b[{};1H{}\x1b[K", log_rows + index + 1, line));
        }
        out
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
                Row::Blank => 0,
                other => visible_width(other.label()),
            })
            .max()
            .unwrap_or(6)
            .clamp(6, 12);
        // One column of air on the right, so nothing is ever drawn against the
        // border: a bar that touches the frame reads as a clipping bug.
        let head_width = 1 + label_width + 1;

        // The picture is one row of the frame that draws as a block: beside the
        // numbers when there is room for both, under them when there is not. A
        // panel that is mostly picture is a worse dashboard than one that is
        // mostly numbers, which is what decides the two.
        let mut stats: Vec<&Row> = Vec::new();
        let mut picture: Option<(&str, &Picture)> = None;
        for row in &frame.rows {
            match row {
                Row::View { label, picture: drawn } => picture = Some((label, drawn)),
                other => stats.push(other),
            }
        }

        // The numbers decide how much of the panel the picture may have: a value
        // cut in half ("00:15 l") reads as a rendering bug, so the box shrinks to
        // what the longest one leaves. It never shrinks below 18 columns - a
        // narrower box shows nothing worth looking at - and on a terminal where
        // even that does not fit beside the numbers the picture goes under them.
        let needed = stats
            .iter()
            .map(|row| head_width + 1 + visible_width(row.value()))
            .max()
            .unwrap_or(0);
        // Rounded up to a step of eight: a run's values creep by a character at a
        // time (a step counter, a countdown, an episode count), and a picture that
        // resized on every one of those would flicker for no reason. 18 columns of
        // picture plus its two borders and a gap is the least that is worth having.
        let wanted = ((needed + 7) / 8) * 8;
        let allotted = wanted.min(inner.saturating_sub(20));

        // Beside: the box is at most a quarter of the panel's width, and only
        // when the numbers still get a sane column of their own.
        let mut beside: Option<Vec<String>> = None;
        if let Some((label, drawn)) = picture {
            let lines = self.view_box(
                label,
                drawn,
                inner,
                height.saturating_sub(4),
                inner.saturating_sub(allotted + 3),
            );
            if let Some(lines) = lines {
                let box_width = visible_width(&lines[0]);
                if inner >= box_width + allotted {
                    beside = Some(lines);
                }
            }
        }

        let left = match &beside {
            Some(lines) => inner - visible_width(&lines[0]) - 1,
            None => inner,
        };
        let boxed_lines = beside.as_ref().map(|lines| lines.len()).unwrap_or(0);
        let lines = stats.len().max(boxed_lines);
        for index in 0..lines {
            let content = match stats.get(index) {
                Some(row) => self.row_content(row, left, label_width, head_width),
                None => String::new(),
            };
            let mut line = format!(
                "{}{}",
                self.paint(Tone::Dim, self.glyphs.vertical),
                fit(&content, left),
            );
            if let Some(boxed) = &beside {
                line.push(' ');
                match boxed.get(index) {
                    Some(right) => line.push_str(right),
                    None => line.push_str(&" ".repeat(visible_width(&boxed[0]))),
                }
            }
            line.push_str(&self.paint(Tone::Dim, self.glyphs.vertical));
            out.push_str(&line);
            out.push('\n');
        }

        // Under: only when it could not go beside, and only as tall as the rest
        // of the panel leaves room for.
        if beside.is_none()
            && let Some((label, drawn)) = picture
        {
            out.push_str(&self.view_block(
                label,
                drawn,
                label_width,
                inner,
                height.saturating_sub(stats.len() + 4),
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

    /// The content of one row, at a given width. `fit` finishes the job.
    fn row_content(&self, row: &Row, inner: usize, label_width: usize, head_width: usize) -> String {
        let head = format!(
            " {} ",
            self.paint(Tone::Dim, &fit(&format!("{} ", row.label()), label_width)),
        );
        match row {
            Row::Blank => " ".repeat(inner),
            // A view is drawn by `view_box` or `view_block`, never as one row.
            Row::View { .. } => String::new(),
            Row::Line { value, tone, .. } => {
                let space = inner.saturating_sub(head_width + 1);
                // A value that had to be cut says so: a number that stops
                // mid-word looks like a bug in the thing being measured.
                let mut painted = self.paint(*tone, value);
                if visible_width(&painted) > space && space > 1 {
                    painted = format!(
                        "{}{}",
                        truncate(&painted, space - visible_width(self.glyphs.ellipsis)),
                        self.paint(Tone::Dim, self.glyphs.ellipsis)
                    );
                }
                format!("{head}{}", fit(&painted, space))
            }
            Row::Meter {
                value,
                fraction,
                tone,
                ..
            } => {
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
                format!("{head}{value_text}{}{bar}", " ".repeat(gap))
            }
            Row::Spark {
                value, points, tone, ..
            } => {
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
        }
    }

    /// One picture at `rows` x `columns`, one character cell per pixel.
    ///
    /// The glyph says how bright the pixel is and the colour says what colour it
    /// is: the ramp alone reads well on a terminal with no colour at all, and the
    /// colour is what makes a corridor, a torch and a health bar distinguishable
    /// at 32 columns wide. `None` rather than a panic for a picture that is not
    /// the shape it claims to be: a run does not stop learning because the
    /// picture is malformed.
    fn pixel_rows(
        &self,
        picture: &Picture,
        rows: usize,
        columns: usize,
    ) -> Option<Vec<String>> {
        if picture.is_empty() || rows == 0 || columns == 0 {
            return None;
        }
        // The same range detection the fingerprint does: a frame is normally in
        // `[0, 1]`, and a frame in 0..255 must not draw as a white block.
        let peak = picture
            .rgb
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(0.0f32, f32::max);
        let scale = if peak > 1.5 { 1.0 / 255.0 } else { 1.0 };
        let sample = picture.resized(rows, columns);
        let ramp = self.glyphs.shade;
        let mut lines = Vec::with_capacity(rows);
        for row in 0..rows {
            let mut line = String::with_capacity(columns * 4);
            let mut current = String::new();
            for column in 0..columns {
                let [red, green, blue] = sample.pixel(row, column);
                let (red, green, blue) = (
                    (red * scale).clamp(0.0, 1.0),
                    (green * scale).clamp(0.0, 1.0),
                    (blue * scale).clamp(0.0, 1.0),
                );
                // With colour on, the glyph is how lit the pixel is and the
                // escape says what colour it is - so a saturated red is a solid
                // block of red, not a faint one. With colour off the glyph is the
                // only thing there is, and then it has to be luminance, which is
                // also what the model saw.
                let luma = 0.299 * red + 0.587 * green + 0.114 * blue;
                let brightness = if self.pixels == PixelColour::Off {
                    luma
                } else {
                    red.max(green).max(blue)
                };
                let index = (brightness * (ramp.len() - 1) as f32).round() as usize;
                let glyph = ramp[index.min(ramp.len() - 1)];
                if !glyph.is_empty() {
                    // One escape per colour change rather than per cell: a flat
                    // sky or a wall is most of the picture, and this is the
                    // difference between a repaint and a flood.
                    let colour = self.pixel_colour(red, green, blue);
                    if colour != current {
                        line.push_str(&colour);
                        current = colour;
                    }
                    line.push_str(glyph);
                }
            }
            if !current.is_empty() {
                line.push_str("\x1b[39m");
            }
            lines.push(line);
        }
        Some(lines)
    }

    /// The escape that makes the next glyph this colour, if colour is on.
    ///
    /// `38;2;r;g;b` is 24-bit and is what a modern terminal wants; the xterm
    /// 256-colour cube is the fallback for a terminal that says `256color` but not
    /// `truecolor`; a terminal that says neither gets no escape at all and draws
    /// the shade ramp, which is a picture rather than a blank.
    fn pixel_colour(&self, red: f32, green: f32, blue: f32) -> String {
        let to_byte = |value: f32| (value.clamp(0.0, 1.0) * 255.0).round() as u8;
        match self.pixels {
            PixelColour::Off => String::new(),
            PixelColour::True => format!(
                "\x1b[38;2;{};{};{}m",
                to_byte(red),
                to_byte(green),
                to_byte(blue)
            ),
            PixelColour::Palette => {
                format!("\x1b[38;5;{}m", ansi256(to_byte(red), to_byte(green), to_byte(blue)))
            }
        }
    }

    /// The widest the picture beside the numbers is drawn, in pixels.
    ///
    /// The observation is square and a character cell is about twice as tall as
    /// it is wide, so half as many rows as columns is the shape the bot sees.
    const VIEW_COLUMNS: usize = 32;

    /// The picture as its own little bordered box: `╭─ vision ──╮` and so on.
    ///
    /// Each line is exactly `columns + 2` wide. `None` when the panel is too
    /// small for a picture worth drawing.
    fn view_box(
        &self,
        label: &str,
        picture: &Picture,
        inner: usize,
        room: usize,
        widest: usize,
    ) -> Option<Vec<String>> {
        // A box too narrow to show anything is worse than no box: the numbers
        // keep their columns and the picture waits for a wider terminal.
        if widest < 16 {
            return None;
        }
        let mut columns = Self::VIEW_COLUMNS.min(inner / 4).min(widest).max(16);
        let mut rows = (columns / 2).max(1);
        if rows + 2 > room {
            rows = room.saturating_sub(2);
            columns = rows * 2;
        }
        if rows < 5 || columns < 16 {
            return None;
        }
        let pixels = self.pixel_rows(picture, rows, columns)?;
        let mut lines: Vec<String> = Vec::with_capacity(rows + 2);
        lines.push(
            self.border(
                self.glyphs.top_left,
                self.glyphs.top_right,
                &self.paint(Tone::Dim, label),
                "",
                columns,
            )
            .trim_end_matches('\n')
            .to_string(),
        );
        for row in pixels {
            lines.push(format!(
                "{}{}{}",
                self.paint(Tone::Dim, self.glyphs.vertical),
                row,
                self.paint(Tone::Dim, self.glyphs.vertical),
            ));
        }
        lines.push(
            self.border(
                self.glyphs.bottom_left,
                self.glyphs.bottom_right,
                "",
                "",
                columns,
            )
            .trim_end_matches('\n')
            .to_string(),
        );
        Some(lines)
    }

    /// The widest the picture under the numbers is drawn, in pixels.
    ///
    /// Narrower than the box beside them: a panel stacked like that is already
    /// as tall as the terminal can take, and the numbers come first.
    const VIEW_BELOW_COLUMNS: usize = 24;

    /// The picture as a block under the numbers, on a terminal too narrow to
    /// put it beside them: a labelled block, `inner` columns wide, `room` rows.
    fn view_block(
        &self,
        label: &str,
        picture: &Picture,
        label_width: usize,
        inner: usize,
        room: usize,
    ) -> String {
        let head_width = 1 + label_width + 1;
        let space = inner.saturating_sub(head_width + 1);
        let mut columns = space.min(Self::VIEW_BELOW_COLUMNS);
        // Half as many rows as columns, so a square frame stays square; when
        // the panel has less room than that, the picture narrows instead of
        // stretching.
        let mut rows = (columns / 2).max(1);
        if rows > room {
            rows = room;
            columns = (rows * 2).min(space);
        }
        // A view too small to read is worse than no view.
        if columns < 12 || rows < 4 {
            return String::new();
        }
        let Some(pixels) = self.pixel_rows(picture, rows, columns) else {
            return String::new();
        };
        // Centred in the space the numbers do not use: a block pinned to the
        // left of a wide empty row reads as a column of the panel, not a picture.
        let extra = inner.saturating_sub(head_width + columns + 1);
        let (left_air, right_air) = (extra / 2, extra - extra / 2);
        let mut out = String::with_capacity(rows * (inner + 16));
        for (index, row) in pixels.iter().enumerate() {
            let mut line = if index == 0 {
                format!(
                    " {} ",
                    self.paint(Tone::Dim, &fit(&format!("{label} "), label_width))
                )
            } else {
                " ".repeat(head_width)
            };
            line.push_str(&" ".repeat(left_air));
            line.push_str(row);
            line.push_str(&" ".repeat(right_air));
            out.push_str(&format!(
                "{}{}{}\n",
                self.paint(Tone::Dim, self.glyphs.vertical),
                fit(&line, inner),
                self.paint(Tone::Dim, self.glyphs.vertical),
            ));
        }
        out
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
                // One line is all a compact panel has; the picture is the
                // first thing to go.
                Row::View { .. } => {}
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

// =====================================================================
// The terminal itself
// =====================================================================

/// The terminal's own echo, which has to be off while the dashboard is up.
///
/// The hotkeys are global - F8 is pressed with this terminal in front of the
/// user - and with echo on the kernel writes the escape sequence for that key
/// into the middle of the panel, one row below wherever the cursor was left. It
/// reads as a rendering bug and it is indistinguishable from one on a screenshot.
/// The original settings are put back by `finish`, and by the panic hook when a
/// run dies.
#[cfg(unix)]
mod echo {
    use std::sync::Mutex;

    static SAVED: Mutex<Option<libc::termios>> = Mutex::new(None);

    fn attributes() -> Option<libc::termios> {
        unsafe {
            let mut term: libc::termios = std::mem::zeroed();
            // 1 is stdout: the terminal this process is drawing on.
            if libc::tcgetattr(1, &mut term) != 0 {
                None
            } else {
                Some(term)
            }
        }
    }

    /// Echo off, keeping what it was.
    pub fn off() {
        let Some(original) = attributes() else {
            return;
        };
        let mut quiet = original;
        quiet.c_lflag &= !libc::ECHO;
        // The dashboard redraws whatever is under it every 100 ms anyway, so an
        // echo is never useful - only ever damage.
        if unsafe { libc::tcsetattr(1, libc::TCSANOW, &quiet) } == 0
            && let Ok(mut saved) = SAVED.lock()
        {
            *saved = Some(original);
        }
    }

    /// Whatever `off` changed, back.
    pub fn on() {
        let Ok(mut saved) = SAVED.lock() else {
            return;
        };
        if let Some(original) = saved.take() {
            unsafe {
                libc::tcsetattr(1, libc::TCSANOW, &original);
            }
        }
    }
}

#[cfg(not(unix))]
mod echo {
    pub fn off() {}
    pub fn on() {}
}

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

/// A plausible thing to look at, for `--ui-preview`.
///
/// A sky, a ground, a row of blocks drifting past, a character and the falloff of
/// a torch: enough structure that the pixel view reads as a game rather than as
/// noise, in colours that are far enough apart to tell the parts apart at 32
/// columns wide. It moves, so the preview shows a picture that changes the way a
/// real run's does.
fn preview_picture(tick: usize, size: usize) -> Picture {
    let mut rgb = vec![0.0f32; size * size * 3];
    let body = (size / 12).max(2) as i32;
    let walker = size as i32 / 2 + ((tick as f32 * 0.35).sin() * size as f32 * 0.2) as i32;
    let horizon = size * 44 / 100;
    for y in 0..size {
        for x in 0..size {
            let fx = x as f32 / size as f32;
            let fy = y as f32 / size as f32;
            // Sky above the horizon, warm ground below it.
            let mut colour = if y < horizon {
                let depth = y as f32 / horizon.max(1) as f32;
                [
                    0.08 + 0.30 * depth,
                    0.12 + 0.42 * depth,
                    0.34 + 0.44 * depth,
                ]
            } else {
                let near = (y - horizon) as f32 / (size - horizon).max(1) as f32;
                [
                    0.42 - 0.16 * near,
                    0.34 - 0.14 * near,
                    0.22 - 0.10 * near,
                ]
            };
            // A green verge where the ground meets the sky.
            if y >= horizon && y < horizon + (size / 16).max(1) {
                colour = [0.30, 0.48, 0.20];
            }
            // Stone blocks on the ground line, scrolling left.
            if y >= horizon && y < horizon + size / 12 && (x / 7 + tick / 6) % 5 == 0 {
                colour = [0.62, 0.60, 0.56];
            }
            // A floor seam every so often, so the ground has depth.
            if y >= horizon && (y - horizon) % (size / 8).max(2) == 0 {
                colour = [colour[0] * 0.55, colour[1] * 0.55, colour[2] * 0.55];
            }
            // The character: a bright coat over dark legs.
            let dx = x as i32 - walker;
            let dy = y as i32 - (horizon as i32 - body);
            if dx.abs() < body && (0..body * 2).contains(&dy) {
                colour = if dy < body / 2 {
                    [0.92, 0.74, 0.52]
                } else {
                    [0.86, 0.28, 0.18]
                };
            }
            // A torch: warm light that falls off with distance.
            let glow = ((fx - 0.5).powi(2) + (fy - 0.55).powi(2)).sqrt();
            let falloff = (1.22 - glow).clamp(0.34, 1.0);
            let warm = (0.16 * (1.0 - glow).clamp(0.0, 1.0), 0.0);
            let index = (y * size + x) * 3;
            rgb[index] = (colour[0] * falloff + warm.0).clamp(0.0, 1.0);
            rgb[index + 1] = (colour[1] * falloff + warm.0 * 0.7).clamp(0.0, 1.0);
            rgb[index + 2] = (colour[2] * falloff).clamp(0.0, 1.0);
        }
    }
    Picture::new(size, size, rgb)
}

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
                .picture("vision", preview_picture(tick, 48))
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
    use std::collections::VecDeque;

    /// The rows a dashboard paint addressed, as (row, visible text).
    ///
    /// Parsed back out of the escape stream on purpose: what matters is where a
    /// row lands on the screen, not what the builder intended to draw there.
    fn painted_rows(paint: &str) -> Vec<(usize, String)> {
        let mut rows = Vec::new();
        let mut current: Option<(usize, usize)> = None;
        let mut index = 0;
        while let Some(found) = paint[index..].find("\x1b[") {
            let at = index + found;
            if let Some((row, payload)) = row_address(paint, at) {
                if let Some((previous, start)) = current.take() {
                    rows.push((previous, visible_text(&paint[start..at])));
                }
                current = Some((row, payload));
            }
            index = at + 2;
        }
        if let Some((row, start)) = current {
            rows.push((row, visible_text(&paint[start..])));
        }
        rows
    }

    /// `ESC [ row ; 1 H` at `at`: the row it addresses, and where its text starts.
    fn row_address(text: &str, at: usize) -> Option<(usize, usize)> {
        let after = text.get(at..)?.strip_prefix("\x1b[")?;
        let digits: String = after
            .chars()
            .take_while(|character| character.is_ascii_digit())
            .collect();
        if digits.is_empty() || !after[digits.len()..].starts_with(";1H") {
            return None;
        }
        Some((digits.parse().ok()?, at + 2 + digits.len() + 3))
    }

    /// A payload as the terminal shows it: escapes removed, trailing air trimmed.
    fn visible_text(payload: &str) -> String {
        let mut out = String::new();
        let mut characters = payload.chars();
        while let Some(character) = characters.next() {
            if character == '\x1b' {
                skip_escape(&mut characters);
            } else {
                out.push(character);
            }
        }
        out.trim_end().to_string()
    }

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
        // One column short of the terminal, every line: a wrapped panel line
        // lands on the row below it, which on the dashboard is another row of
        // the panel.
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

    /// A frame with a picture in it, for the layout tests below.
    fn frame_with_view(plane: Vec<f32>) -> Frame {
        Frame::new("Synthetic corridor")
            .badge("RUNNING", Tone::Good)
            .line("steps", "1008   167.3/s   3% in game   7 episode(s)", Tone::Value)
            .spark("reward", "+2.1051/step", vec![1.0, 2.0, 3.0], Tone::Good)
            .meter("rollout", "94%  240/256", 0.94, Tone::Accent)
            .line("actions", "no keys+turn 45%   1 key(s)+turn 32%", Tone::Body)
            .view("vision", plane, 64, 64)
            .footer("F8 pause   F9 save   F10 quit")
    }

    #[test]
    fn the_vision_box_keeps_every_line_of_the_panel_the_same_width() {
        unsafe { std::env::set_var("COLUMNS", "80") };
        let ui = Ui::fixed(true, true, true);
        ui.width.store(80, Ordering::Relaxed);
        ui.height.store(40, Ordering::Relaxed);
        let plane: Vec<f32> = (0..64 * 64).map(|index| (index % 64) as f32 / 63.0).collect();
        let composed = ui.compose(&frame_with_view(plane));
        let widths: Vec<usize> = composed.lines().map(visible_width).collect();
        assert!(
            widths.iter().all(|width| *width == 79),
            "widths {widths:?}\n{composed}"
        );
        assert!(composed.contains("vision"), "{composed}");
        // The picture is beside the numbers, so the panel is barely taller than
        // the rows it would draw without one.
        assert!(composed.lines().count() <= 20, "{composed}");
    }

    #[test]
    fn a_panel_with_no_room_for_the_picture_still_fits_the_terminal() {
        unsafe { std::env::set_var("COLUMNS", "80") };
        let ui = Ui::fixed(true, true, true);
        ui.width.store(80, Ordering::Relaxed);
        ui.height.store(12, Ordering::Relaxed);
        let plane = vec![1.0f32; 64 * 64];
        let composed = ui.compose(&frame_with_view(plane));
        let widths: Vec<usize> = composed.lines().map(visible_width).collect();
        assert!(widths.iter().all(|width| *width == 79), "{composed}");
        // The panel never draws more lines than the terminal has: a taller panel
        // scrolls the dashboard, and scrolling takes the log tail with it.
        assert!(composed.lines().count() < 12, "{composed}");
    }

    #[test]
    fn the_dashboard_paints_every_row_once_and_anchors_the_panel_to_the_bottom() {
        unsafe { std::env::set_var("COLUMNS", "80") };
        let ui = Ui::fixed(true, true, true);
        ui.width.store(80, Ordering::Relaxed);
        ui.height.store(24, Ordering::Relaxed);
        let frame = Frame::new("game")
            .badge("RUNNING", Tone::Good)
            .line("steps", "24812   41.2/s   12 episode(s)", Tone::Value)
            .meter("rollout", "58%  604/1024", 0.58, Tone::Accent)
            .footer("F8 pause   F10 quit");
        let panel = ui.compose(&frame).lines().count();
        let mut log = VecDeque::new();
        for index in 0..40 {
            log.push_back(fit(&format!("  line {index}"), 79));
        }
        let paint = ui.screen(&log, Some(&frame));
        let rows = painted_rows(&paint);
        // Every row of the terminal is painted exactly once: nothing is left
        // over from the previous frame, and nothing is painted twice.
        let addressed: Vec<usize> = rows.iter().map(|(row, _)| *row).collect();
        assert_eq!(addressed, (1..=24).collect::<Vec<usize>>(), "{addressed:?}");
        // The panel's bottom border is the last row, and its top border is
        // exactly `panel` rows above it: the log tail fills everything above.
        assert!(rows[23].1.starts_with('\u{2570}'), "{:?}", rows[23].1);
        assert!(
            rows[24 - panel].1.starts_with('\u{256d}'),
            "{:?}",
            rows[24 - panel].1
        );
        // The newest log line sits directly above the panel, not the oldest.
        assert!(rows[24 - panel - 1].1.contains("line 39"), "{rows:?}");
        // And no row is wider than the terminal, or it would wrap into the next.
        for (row, text) in &rows {
            assert!(visible_width(text) < 80, "row {row} is {} wide", visible_width(text));
        }
    }

    #[test]
    fn a_value_that_does_not_fit_ends_in_an_ellipsis() {
        unsafe { std::env::set_var("COLUMNS", "60") };
        let ui = Ui::fixed(true, true, true);
        ui.width.store(60, Ordering::Relaxed);
        ui.height.store(20, Ordering::Relaxed);
        let frame = Frame::new("game").line(
            "steps",
            "24812   41.2/s   12 episode(s)   00:31 left   and more of it",
            Tone::Value,
        );
        let composed = ui.compose(&frame);
        assert!(composed.contains('\u{2026}'), "{composed}");
        // The mark is part of the line, never past it.
        for line in composed.lines() {
            assert_eq!(visible_width(line), 59, "{line}");
        }
    }

    #[test]
    fn a_malformed_frame_draws_no_picture_and_does_not_panic() {
        let ui = Ui::fixed(false, true, true);
        // A short buffer, an empty picture, and a zero-sized request.
        let short = Picture::new(8, 8, vec![0.5; 4]);
        assert!(ui.pixel_rows(&short, 4, 8).is_none());
        assert!(ui.pixel_rows(&Picture::default(), 4, 8).is_none());
        assert!(ui.pixel_rows(&plane_picture(&[0.5; 4], 2, 2), 0, 8).is_none());
    }

    #[test]
    fn the_picture_maps_dark_to_shade_and_bright_to_solid() {
        let ui = Ui::fixed(false, true, false);
        // Two pixels: black and white, sampled to the same two columns.
        let rows = ui.pixel_rows(&plane_picture(&[0.0, 1.0, 0.0, 1.0], 2, 2), 1, 2).unwrap();
        assert_eq!(rows, vec![" █".to_string()]);
        // A frame in 0..255 is read the same way a frame in 0..1 is.
        let wide = ui
            .pixel_rows(&plane_picture(&[0.0, 255.0, 0.0, 255.0], 2, 2), 1, 2)
            .unwrap();
        assert_eq!(wide, vec![" █".to_string()]);
    }

    /// A grey plane as the picture the renderer takes.
    fn plane_picture(plane: &[f32], height: usize, width: usize) -> Picture {
        Picture::from_grey(plane, height, width)
    }

    #[test]
    fn a_colour_picture_is_drawn_in_colour() {
        // A red pixel and a blue pixel, one row.
        let picture = Picture::new(2, 1, vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let ui = Ui::fixed(true, true, false);
        let rows = ui.pixel_rows(&picture, 1, 2).unwrap();
        let line = &rows[0];
        // 24-bit foreground for each pixel, and the reset at the end of the row.
        assert!(line.starts_with("\x1b[38;2;255;0;0m"), "{line:?}");
        assert!(line.contains("\x1b[38;2;0;0;255m"), "{line:?}");
        assert!(line.ends_with("\x1b[39m"), "{line:?}");
        // With no colour the glyph is the only thing there is, so it is luminance
        // - which is what the model was handed, and a red pixel is dim in it.
        let plain = Ui::fixed(false, true, false);
        let rows = plain.pixel_rows(&picture, 1, 2).unwrap();
        assert_eq!(rows[0], "░ ");
    }

    #[test]
    fn a_256_colour_terminal_gets_the_palette_and_not_24_bit() {
        let picture = Picture::new(1, 1, vec![1.0, 0.0, 0.0]);
        let mut ui = Ui::fixed(true, true, false);
        ui.pixels = PixelColour::Palette;
        let rows = ui.pixel_rows(&picture, 1, 1).unwrap();
        assert!(rows[0].starts_with("\x1b[38;5;"), "{:?}", rows[0]);
        assert!(!rows[0].contains("38;2;"), "{:?}", rows[0]);
        // Pure red is cube index 196 (16 + 36 * 5).
        assert_eq!(ansi256(255, 0, 0), 196);
        // Mid grey is on the 24-step ramp, not the cube's diagonal.
        assert_eq!(ansi256(128, 128, 128), 244);
    }

    #[test]
    fn a_picture_narrower_than_its_box_is_stretched_not_dropped() {
        let picture = Picture::new(2, 2, vec![1.0; 12]);
        let resized = picture.resized(6, 8);
        assert_eq!((resized.height, resized.width), (6, 8));
        assert_eq!(resized.rgb.len(), 6 * 8 * 3);
        assert!(resized.rgb.iter().all(|value| (*value - 1.0).abs() < 1e-6));
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
