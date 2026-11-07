"""Zero-Dependency Streaming & Block Terminal Markdown Renderer (`crux`).

Transforms raw LLM Markdown streams into clean, high-contrast terminal typography:
- ATX headings (`#`–`######`) with clear visual hierarchy and zero raw `#` clutter
- Fenced code blocks (` ```lang ` / ` ~~~lang `) with box gutters & token-level syntax highlighting
- Inline typography (`**bold**`, `*italic*`, `` `code` ``, `~~strike~~`, `[label](url)`) with
  strict `snake_case` protection and concise `file://` basename:line formatting
- Bulleted, numbered, and task-checkbox (`- [x]` / `- [ ]`) lists with hanging-indent word wrap
- Blockquotes & GitHub Alerts (`> [!NOTE]`, `> [!TIP]`, `> [!IMPORTANT]`, `> [!WARNING]`, `> [!CAUTION]`)
- Aligned Unicode box-drawn GFM pipe tables (`╭─┬─╮`, `│ │ │`, `├─┼─┤`, `╰─┴─╯`) with cell wrapping
- Self-contained per-line ANSI SGR state so styles never bleed into the TUI Inline Dock (`INV-TUI-1`)
"""

from __future__ import annotations

import html
import pathlib
import re
import shutil
import unicodedata
import urllib.parse
from typing import Callable

from ush.cli.tui import ANSI_ESCAPE_RE, char_width, strip_ansi, vis_width

# ============================================================================
# Regex Primitives (Compiled Once)
# ============================================================================

_FENCE_OPEN_RE = re.compile(r"^(?P<indent>[ \t]{0,3})(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_HEADING_RE = re.compile(r"^[ \t]{0,3}(?P<hashes>#{1,6})[ \t]+(?P<title>.+?)[ \t]*#*[ \t]*$")
_HR_RE = re.compile(r"^[ \t]{0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$")
_BLOCKQUOTE_RE = re.compile(r"^[ \t]{0,3}>[ \t]?(?P<body>.*)$")
_ALERT_TAG_RE = re.compile(r"^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\][ \t]*(?P<rest>.*)$", re.IGNORECASE)
_TASK_ITEM_RE = re.compile(r"^(?P<indent>[ \t]*)(?:[-*+]|\d+[.)])[ \t]+\[(?P<mark>[ xX])\][ \t]+(?P<body>.*)$")
_UL_ITEM_RE = re.compile(r"^(?P<indent>[ \t]*)[-*+][ \t]+(?P<body>.*)$")
_OL_ITEM_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<num>\d+)(?P<delim>[.)])[ \t]+(?P<body>.*)$")
_BOX_ART_CHARS = frozenset("╭╮╰╯│─━┌┐└┘├┤┬┴┼╔╗╚╝║═")

# Inline code spans (supports ``code with ` inside`` and `code`)
_INLINE_CODE_RE = re.compile(r"(?P<ticks>`+)(?P<code>.+?)(?P=ticks)")
# Escaped ASCII punctuation `\*`, `\_`, `\[`, etc.
_ESCAPED_PUNCT_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|~>])")
# Markdown images & links (handles 1 level of balanced parens inside URL)
_IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<url>(?:[^()\s]|\([^()\s]*\))+)(?:[ \t]+\"[^\"]*\")?\)")
_LINK_RE = re.compile(r"\[(?P<label>[^\]]+)\]\((?P<url>(?:[^()\s]|\([^()\s]*\))+)(?:[ \t]+\"[^\"]*\")?\)")

# Typography patterns — carefully bounded so `snake_case_vars` and `a * b * c` are NEVER mangled!
_BOLD_ITALIC_AST_RE = re.compile(r"\*\*\*(?=\S)(.+?)(?<=\S)\*\*\*")
_BOLD_AST_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
_BOLD_UND_RE = re.compile(r"(?<![A-Za-z0-9_])__(?=\S)(.+?)(?<=\S)__(?![A-Za-z0-9_])")
_ITALIC_AST_RE = re.compile(r"(?<!\*)\*(?=\S)([^*\n]+?)(?<=\S)\*(?!\*)")
_ITALIC_UND_RE = re.compile(r"(?<![A-Za-z0-9_])_(?=\S)([^_\n]+?)(?<=\S)_(?![A-Za-z0-9_])")
_STRIKE_RE = re.compile(r"~~(?=\S)(.+?)(?<=\S)~~")

_HTML_ENTITY_RE = re.compile(r"&(?:#\d{1,7}|#x[0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")
_INLINE_DISPLAY_DOLLAR_RE = re.compile(r"(?<!\\)\$\$(?P<expr>.+?)(?<!\\)\$\$")
_INLINE_DISPLAY_BRACKET_RE = re.compile(r"(?<!\\)\\\[(?P<expr>.+?)(?<!\\)\\\]")
_INLINE_PAREN_MATH_RE = re.compile(r"(?<!\\)\\\((?P<expr>.+?)(?<!\\)\\\)")
_INLINE_DOLLAR_MATH_RE = re.compile(
    r"(?<![\\$A-Za-z0-9_])\$(?!\$)(?=\S)(?P<expr>[^$\n]+?)(?<!\s)(?<!\\)\$(?![$A-Za-z0-9_])"
)

# ============================================================================
# Font-Verified Terminal LaTeX & Mathematical Unicode Engine (Zero Dependencies)
# ============================================================================

# Every single target glyph below is verified 1-column wide (`east_asian_width != W/F`)
# and natively present in macOS `Menlo` / `SF Mono` + Linux `DejaVu Sans Mono` (zero `[?]` tofu!).
_SUPERSCRIPT_MAP: dict[str, str] = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "−": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
    "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ", "f": "ᶠ",
    "g": "ᵍ", "h": "ʰ", "i": "ⁱ", "j": "ʲ", "k": "ᵏ", "l": "ˡ",
    "m": "ᵐ", "n": "ⁿ", "o": "ᵒ", "p": "ᵖ", "r": "ʳ", "s": "ˢ",
    "t": "ᵗ", "u": "ᵘ", "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ", "z": "ᶻ",
    "A": "ᴬ", "B": "ᴮ", "D": "ᴰ", "E": "ᴱ", "G": "ᴳ", "H": "ᴴ",
    "I": "ᴵ", "J": "ᴶ", "K": "ᴷ", "L": "ᴸ", "M": "ᴹ", "N": "ᴺ",
    "O": "ᴼ", "P": "ᴾ", "R": "ᴿ", "T": "ᵀ", "U": "ᵁ", "V": "ⱽ", "W": "ᵂ",
    "⊤": "ᵀ", "′": "′", "*": "*", "∗": "*",
}

# Note: Unicode 6.0 subscripts `ₕₖₗₘₙₚₛₜ` (`U+2095..209C`) are intentionally omitted here
# because macOS `Menlo` lacks glyphs for `U+2095..209C`; atomic fallback keeps `x_n`, `x_k` crisp!
_SUBSCRIPT_MAP: dict[str, str] = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "+": "₊", "-": "₋", "−": "₋", "=": "₌", "(": "₍", ")": "₎",
    "a": "ₐ", "e": "ₑ", "i": "ᵢ", "j": "ⱼ", "o": "ₒ",
    "r": "ᵣ", "u": "ᵤ", "v": "ᵥ", "x": "ₓ",
}

_MENLO_MISSING_SUBSCRIPTS: dict[str, str] = {
    "ₕ": "h", "ₖ": "k", "ₗ": "l", "ₘ": "m",
    "ₙ": "n", "ₚ": "p", "ₛ": "s", "ₜ": "t",
}

_FONT_SAFE_GLYPH_FALLBACKS: dict[str, str] = {
    "‖": "||",   # U+2016 missing in Menlo -> clean double-bar norm
    "ℏ": "ħ",    # U+210F -> U+0127 (present in Menlo)
    "ℓ": "l",    # U+2113 -> ASCII l
    "∖": "\\",   # U+2216 -> backslash
    "⟶": "→",    # U+27F6 -> U+2192
    "⟵": "←",    # U+27F5 -> U+2190
    "⟷": "↔",    # U+27F7 -> U+2194
    "⟹": "⇒",    # U+27F9 -> U+21D2
    "⟸": "⇐",    # U+27F8 -> U+21D0
    "⟺": "⇔",    # U+27FA -> U+21D4
    "⟼": "↦",    # U+27FC -> U+21A6
    "≪": "<<",
    "≫": ">>",
}

_MATHBB_MAP: dict[str, str] = {
    "R": "ℝ", "Z": "ℤ", "N": "ℕ", "Q": "ℚ",
    "C": "ℂ", "P": "ℙ", "H": "ℍ",
}

_VULGAR_FRACTIONS: dict[tuple[str, str], str] = {
    ("1", "2"): "½",
    ("1", "3"): "⅓",
    ("2", "3"): "⅔",
    ("1", "4"): "¼",
    ("3", "4"): "¾",
}

_COMBINING_ACCENTS: dict[str, str] = {
    "hat": "\u0302",
    "widehat": "\u0302",
    "check": "\u030c",
    "tilde": "\u0303",
    "widetilde": "\u0303",
    "acute": "\u0301",
    "grave": "\u0300",
    "dot": "\u0307",
    "ddot": "\u0308",
    "breve": "\u0306",
    "bar": "\u0304",
    "overline": "\u0304",
    "vec": "\u20d7",
    "overrightarrow": "\u20d7",
}

# Full LaTeX symbol table (used inside `$...$`, `$$...$$`, `\(...\)`, `\[...\]`)
_LATEX_SYMBOLS: dict[str, str] = {
    # Lowercase Greek
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "vartheta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "varpi": "π", "rho": "ρ", "varrho": "ρ",
    "sigma": "σ", "varsigma": "ς", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    # Uppercase Greek
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ",
    "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    # Calculus, Logic & Set Theory
    "infty": "∞", "partial": "∂", "nabla": "∇", "grad": "∇",
    "hbar": "ħ", "ell": "l", "Re": "Re", "Im": "Im", "aleph": "ℵ",
    "forall": "∀", "exists": "∃", "nexists": "¬∃",
    "neg": "¬", "lnot": "¬", "land": "∧", "wedge": "∧", "lor": "∨", "vee": "∨",
    "emptyset": "∅", "varnothing": "∅",
    "in": "∈", "notin": "∉", "ni": "∋", "owns": "∋",
    "subset": "⊂", "supset": "⊃", "subseteq": "⊆", "supseteq": "⊇",
    "nsubseteq": "⊈", "subsetneq": "⊊",
    "cup": "∪", "cap": "∩", "bigcup": "⋃", "bigcap": "⋂",
    "setminus": "\\", "smallsetminus": "\\",
    "top": "⊤", "bot": "⊥", "perp": "⊥", "parallel": "∥",
    "vdash": "|-", "models": "|=",
    # Relations
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥",
    "neq": "≠", "ne": "≠", "equiv": "≡", "nequiv": "≢",
    "approx": "≈", "thickapprox": "≈", "sim": "∼", "simeq": "≃", "cong": "≅",
    "propto": "∝", "ll": "<<", "gg": ">>",
    "prec": "≺", "succ": "≻", "preceq": "≼", "succeq": "≽",
    "triangleq": "≜", "coloneqq": "≔", "defeq": "≔",
    "mid": "|", "nmid": "∤",
    # Binary & Big Operators
    "pm": "±", "mp": "∓", "times": "×", "div": "÷",
    "cdot": "·", "centerdot": "·", "ast": "∗", "star": "⋆",
    "circ": "∘", "bullet": "•",
    "oplus": "⊕", "ominus": "⊖", "otimes": "⊗", "oslash": "⊘", "odot": "⊙",
    "bigoplus": "⊕", "bigotimes": "⊗",
    "sum": "∑", "prod": "∏", "coprod": "∐",
    "int": "∫", "iint": "∬", "iiint": "∭", "oint": "∮",
    # Arrows
    "to": "→", "rightarrow": "→", "longrightarrow": "→",
    "gets": "←", "leftarrow": "←", "longleftarrow": "←",
    "leftrightarrow": "↔", "longleftrightarrow": "↔",
    "Rightarrow": "⇒", "Longrightarrow": "⇒", "implies": "⇒",
    "Leftarrow": "⇐", "Longleftarrow": "⇐", "impliedby": "⇐",
    "Leftrightarrow": "⇔", "Longleftrightarrow": "⇔", "iff": "⇔",
    "mapsto": "↦", "longmapsto": "↦",
    "uparrow": "↑", "downarrow": "↓", "updownarrow": "↕",
    "nearrow": "↗", "searrow": "↘", "swarrow": "↙", "nwarrow": "↖",
    "hookrightarrow": "↪", "hookleftarrow": "↩",
    # Delimiters, Dots & Misc
    "langle": "⟨", "rangle": "⟩",
    "lceil": "⌈", "rceil": "⌉", "lfloor": "⌊", "rfloor": "⌋",
    "vert": "|", "Vert": "||", "lVert": "||", "rVert": "||",
    "dots": "…", "ldots": "…", "cdots": "⋯", "dotsb": "⋯", "dotsm": "⋯", "dotsc": "…",
    "vdots": ":", "ddots": "⋱",
    "therefore": "∴", "because": "∵",
    "prime": "′", "degree": "°", "checkmark": "✓",
    "triangle": "△", "angle": "∠", "square": "□", "Box": "□",
    # Standard Named Math Functions
    "sin": "sin", "cos": "cos", "tan": "tan", "cot": "cot", "sec": "sec", "csc": "csc",
    "arcsin": "arcsin", "arccos": "arccos", "arctan": "arctan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "coth": "coth",
    "log": "log", "ln": "ln", "lg": "lg", "exp": "exp",
    "lim": "lim", "limsup": "lim sup", "liminf": "lim inf",
    "sup": "sup", "inf": "inf", "max": "max", "min": "min",
    "arg": "arg", "argmax": "argmax", "argmin": "argmin",
    "det": "det", "dim": "dim", "ker": "ker", "hom": "hom",
    "deg": "deg", "gcd": "gcd", "lcm": "lcm", "Pr": "Pr",
    "rank": "rank", "tr": "tr", "trace": "trace", "diag": "diag",
    "cov": "Cov", "var": "Var", "Var": "Var", " mod": " mod ", "bmod": " mod ",
}

# Unambiguous LaTeX symbol names safe to render even when an LLM omits `$...$` in prose
# (Excludes short English words like `in`, `to`, `div`, `top`, `sec`, `min`, `max`, `log`, `sin`, `cos`, `and`, `or`)
_PROSE_SAFE_LATEX_CMDS: frozenset[str] = frozenset(
    {
        "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta",
        "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi",
        "rho", "sigma", "varsigma", "tau", "upsilon", "phi", "varphi", "chi", "psi", "omega",
        "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma", "Phi", "Psi", "Omega",
        "infty", "partial", "nabla", "hbar", "forall", "exists", "nexists",
        "emptyset", "varnothing", "notin", "subseteq", "supseteq", "nsubseteq",
        "leq", "geq", "neq", "equiv", "nequiv", "approx", "thickapprox", "simeq", "cong", "propto",
        "triangleq", "coloneqq", "pm", "mp", "times", "cdot", "oplus", "otimes",
        "rightarrow", "longrightarrow", "leftarrow", "longleftarrow",
        "leftrightarrow", "longleftrightarrow",
        "Rightarrow", "Longrightarrow", "implies",
        "Leftarrow", "Longleftarrow", "impliedby",
        "Leftrightarrow", "Longleftrightarrow", "iff", "mapsto",
        "langle", "rangle", "lceil", "rceil", "lfloor", "rfloor",
        "ldots", "cdots",
    }
)
_PROSE_LATEX_CMD_RE = re.compile(
    r"(?<![A-Za-z0-9_./:\\~-])\\(?P<cmd>"
    + "|".join(sorted(_PROSE_SAFE_LATEX_CMDS, key=len, reverse=True))
    + r")(?![A-Za-z0-9_\\])"
)
_LATEX_CMD_TOKEN_RE = re.compile(r"\\([A-Za-z]+)")


def normalize_terminal_math_glyphs(text: str, *, use_color: bool = True) -> str:
    """Normalizes Plane-1 math alphanumerics (`U+1D400..1D7FF`) & Menlo-missing symbols so terminals never show `[?]`."""
    if not text or text.isascii():
        return text

    out: list[str] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        cp = ord(ch)
        if cp < 0x80:
            out.append(ch)
            i += 1
            continue

        # Coalesce runs of Menlo-missing Unicode 6.0 subscripts (`xₙ` -> `x_n`, `tₘₐₓ` -> `t_(max)`)
        if ch in _MENLO_MISSING_SUBSCRIPTS:
            sub_run: list[str] = []
            j = i
            while j < n:
                cj = text[j]
                if cj in _MENLO_MISSING_SUBSCRIPTS:
                    sub_run.append(_MENLO_MISSING_SUBSCRIPTS[cj])
                    j += 1
                elif cj in "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢⱼₒᵣᵤᵥₓ":
                    sub_run.append(unicodedata.normalize("NFKD", cj))
                    j += 1
                else:
                    break
            joined_sub = "".join(sub_run)
            out.append(f"_{joined_sub}" if len(joined_sub) == 1 else f"_({joined_sub})")
            i = j
            continue

        if ch in _FONT_SAFE_GLYPH_FALLBACKS:
            out.append(_FONT_SAFE_GLYPH_FALLBACKS[ch])
            i += 1
            continue

        # Plane-1 Mathematical Alphanumeric Symbols (`U+1D400..1D7FF`: `𝒪`, `𝑥`, `𝐀`, `𝔸`, `𝟙`, `𝛼`, ...)
        if 0x1D400 <= cp <= 0x1D7FF:
            base = unicodedata.normalize("NFKD", ch)
            if use_color:
                uname = unicodedata.name(ch, "")
                if "BOLD" in uname and "ITALIC" in uname:
                    out.append(f"\x1b[1;3m{base}\x1b[22;23m")
                elif "BOLD" in uname:
                    out.append(f"\x1b[1m{base}\x1b[22m")
                elif "ITALIC" in uname or "SCRIPT" in uname:
                    out.append(f"\x1b[3m{base}\x1b[23m")
                else:
                    out.append(base)
            else:
                out.append(base)
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _extract_braced(s: str, pos: int) -> tuple[str, int] | None:
    """Extracts balanced `{...}` starting at `pos` (ignoring leading spaces), returning `(inner, next_pos)`."""
    n = len(s)
    i = pos
    while i < n and s[i] in " \t":
        i += 1
    if i >= n or s[i] != "{":
        return None
    depth = 1
    i += 1
    start = i
    while i < n and depth > 0:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i], i + 1
        i += 1
    return None


def _extract_bracket_opt(s: str, pos: int) -> tuple[str, int] | None:
    """Extracts optional `[...]` argument starting at `pos` (ignoring leading spaces)."""
    n = len(s)
    i = pos
    while i < n and s[i] in " \t":
        i += 1
    if i >= n or s[i] != "[":
        return None
    depth = 1
    i += 1
    start = i
    while i < n and depth > 0:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return s[start:i], i + 1
        i += 1
    return None


def _needs_frac_parens(part: str, *, is_denom: bool = False) -> bool:
    t = part.strip()
    if len(t) <= 1:
        return False
    if t.startswith("(") and t.endswith(")"):
        # Verify outer parens stay balanced across the whole expression
        d = 0
        balanced = True
        for idx, ch in enumerate(t):
            if ch == "(":
                d += 1
            elif ch == ")":
                d -= 1
                if d == 0 and idx < len(t) - 1:
                    balanced = False
                    break
        if balanced and d == 0:
            return False
    if is_denom and t[0].isdigit() and not all(c.isdigit() or c == "." for c in t):
        return True
    ops = ("+", "-", "−", "±", "∓", " ") + (("*", "/", "·", "×", "÷") if is_denom else ())
    return any(op in t[1:] for op in ops) or (" " in t)


def _format_superscript(inner: str) -> str:
    clean = inner.strip()
    if not clean:
        return ""
    if clean in ("\\prime", "'"):
        return "′"
    if clean == "''":
        return "″"
    if all(ch in _SUPERSCRIPT_MAP for ch in clean):
        return "".join(_SUPERSCRIPT_MAP[ch] for ch in clean)
    return f"^{clean}" if len(clean) == 1 else f"^({clean})"


def _format_subscript(inner: str) -> str:
    clean = inner.strip()
    if not clean:
        return ""
    if all(ch in _SUBSCRIPT_MAP for ch in clean):
        return "".join(_SUBSCRIPT_MAP[ch] for ch in clean)
    return f"_{clean}" if len(clean) == 1 or clean.isalnum() else f"_({clean})"


def _split_latex_rows(body: str) -> list[str]:
    """Splits LaTeX environment rows on `\\\\` (ignoring optional `\\\\[4pt]` row-spacing hints)."""
    parts = re.split(r"\\\\(?:\s*\[[^\]]*\])?", body)
    rows = [p.strip() for p in parts if p.strip()]
    return rows if rows else [body.strip()]


def _cell_vis_width(s: str) -> int:
    return max((vis_width(ln) for ln in s.split("\n")), default=0)


def _restore_env_placeholders(text: str, env_tokens: list[str]) -> str:
    if "\x02ENV" not in text or not env_tokens:
        return text
    out_lines: list[str] = []
    pat = re.compile(r"\x02ENV(\d+)\x02")
    for raw_ln in text.split("\n"):
        if "\x02ENV" not in raw_ln:
            out_lines.append(raw_ln)
            continue
        # Expand placeholders left-to-right, indenting multi-line blocks to match prefix column
        cur_lines = [raw_ln]
        while any("\x02ENV" in cl for cl in cur_lines):
            next_lines: list[str] = []
            for cl in cur_lines:
                m = pat.search(cl)
                if not m:
                    next_lines.append(cl)
                    continue
                idx = int(m.group(1))
                val = env_tokens[idx] if 0 <= idx < len(env_tokens) else ""
                pfx = cl[: m.start()]
                sfx = cl[m.end() :]
                v_lines = val.split("\n")
                if len(v_lines) <= 1:
                    next_lines.append(f"{pfx}{val}{sfx}")
                else:
                    pad = " " * vis_width(pfx)
                    next_lines.append(f"{pfx}{v_lines[0]}")
                    for mid_v in v_lines[1:-1]:
                        next_lines.append(f"{pad}{mid_v}")
                    next_lines.append(f"{pad}{v_lines[-1]}{sfx}")
            cur_lines = next_lines
        out_lines.extend(cur_lines)
    return "\n".join(out_lines)


def _render_latex_environment(
    env: str,
    body: str,
    *,
    display: bool,
    env_tokens: list[str] | None = None,
) -> str:
    rows_raw = _split_latex_rows(body)
    grid = [
        [_render_latex_core(cell.strip(), display=False, _env_tokens=env_tokens) for cell in r.split("&")]
        for r in rows_raw
    ]
    ncols = max((len(r) for r in grid), default=1)
    for r in grid:
        while len(r) < ncols:
            r.append("")

    env_norm = env.rstrip("*")
    if env_norm == "cases":
        if not display or len(grid) == 1:
            items = [" ".join(c.replace("\n", " ") for c in r if c) for r in grid]
            return "{ " + ";  ".join(items) + " }"
        col0_w = max(_cell_vis_width(r[0]) for r in grid)
        rendered_rows: list[str] = []
        nrows = len(grid)
        for idx, r in enumerate(grid):
            glyph = "⎧" if idx == 0 else ("⎩" if idx == nrows - 1 else "⎨")
            c0 = r[0].replace("\n", " ")
            c1 = r[1].replace("\n", " ") if len(r) > 1 else ""
            left = c0 + (" " * max(0, col0_w - vis_width(c0)))
            right = f"   {c1}" if c1 else ""
            rendered_rows.append(f"{glyph} {left}{right}".rstrip())
        return "\n".join(rendered_rows)

    if env_norm in ("matrix", "pmatrix", "bmatrix", "Bmatrix", "vmatrix", "Vmatrix", "smallmatrix"):
        if not display or len(grid) == 1:
            row_strs = [
                ", ".join(c.replace("\n", " ") for c in r) if len(grid) > 1 else "  ".join(c.replace("\n", " ") for c in r)
                for r in grid
            ]
            open_d, close_d = {
                "pmatrix": ("(", ")"),
                "bmatrix": ("[", "]"),
                "Bmatrix": ("{", "}"),
                "vmatrix": ("|", "|"),
                "Vmatrix": ("||", "||"),
            }.get(env_norm, ("[", "]"))
            return f"{open_d}{' ; '.join(row_strs)}{close_d}"

        col_w = [max(_cell_vis_width(r[c]) for r in grid) for c in range(ncols)]
        nrows = len(grid)
        m_lines: list[str] = []
        for idx, r in enumerate(grid):
            cells_flat = [r[c].replace("\n", " ") for c in range(ncols)]
            cells_padded = [
                cells_flat[c].rjust(len(cells_flat[c]) + max(0, col_w[c] - vis_width(cells_flat[c])))
                for c in range(ncols)
            ]
            inner_row = "  ".join(cells_padded)
            if env_norm == "pmatrix":
                ld, rd = ("⎛", "⎞") if idx == 0 else (("⎝", "⎠") if idx == nrows - 1 else ("⎜", "⎟"))
            elif env_norm == "bmatrix":
                ld, rd = ("⎡", "⎤") if idx == 0 else (("⎣", "⎦") if idx == nrows - 1 else ("⎢", "⎥"))
            elif env_norm == "Bmatrix":
                ld, rd = ("⎧", "⎫") if idx == 0 else (("⎩", "⎭") if idx == nrows - 1 else ("⎨", "⎬"))
            elif env_norm == "vmatrix":
                ld, rd = ("│", "│")
            elif env_norm == "Vmatrix":
                ld, rd = ("||", "||")
            else:
                ld, rd = (" ", " ")
            m_lines.append(f"{ld} {inner_row} {rd}".rstrip())
        return "\n".join(m_lines)

    # `aligned`, `align`, `gather`, `split`, `eqnarray` (supports multi-line nested `cases`/`bmatrix` cells!)
    if not display or len(grid) == 1:
        return " ;  ".join(" ".join(c.replace("\n", " ") for c in r if c) for r in grid)
    col_w = [max(_cell_vis_width(r[c]) for r in grid) for c in range(ncols)]
    al_lines: list[str] = []
    for r in grid:
        split_cells = [r[c].split("\n") for c in range(ncols)]
        row_h = max((len(sc) for sc in split_cells), default=1)
        for sub_i in range(row_h):
            parts_r: list[str] = []
            for c in range(ncols):
                ln_c = split_cells[c][sub_i] if sub_i < len(split_cells[c]) else ""
                pad = " " * max(0, col_w[c] - vis_width(ln_c))
                # Right-align LHS column 0 in `LHS &= RHS` so `=` aligns vertically!
                parts_r.append((pad + ln_c) if c == 0 and ncols >= 2 else (ln_c + pad))
            al_lines.append(" ".join(parts_r).rstrip())
    return "\n".join(al_lines)


_INNER_ENV_BLOCK_RE = re.compile(
    r"\\begin\{(?P<env>[A-Za-z*]+)\}(?P<body>(?:(?!\\begin\{)[\s\S])*?)\\end\{(?P=env)\}"
)
_SIZING_PREFIX_RE = re.compile(
    r"\\(?:left|right|middle)\s*\."
    r"|\\(?:left|right|middle|[bB]ig(?:g|gl|gr|l|r)?|displaystyle|textstyle|scriptstyle|scriptscriptstyle|limits|nolimits)(?![A-Za-z])"
)
_ONE_OR_TWO_ARG_CMDS = frozenset(
    {
        "frac", "dfrac", "tfrac", "cfrac", "binom", "dbinom", "tbinom",
        "sqrt", "mathbb", "Bbb",
        "text", "textrm", "textup", "textbf", "textit", "emph", "texttt",
        "mathrm", "mathbf", "boldsymbol", "bm", "mathit", "mathcal", "mathscr", "mathfrak", "mathsf", "mathtt",
        "operatorname", "mbox", "hbox", "pmod", "xrightarrow", "xleftarrow",
        "overset", "stackrel", "underset", "underbrace", "overbrace",
        "phantom", "hphantom", "vphantom", "cancel", "bcancel", "xcancel",
        *_COMBINING_ACCENTS.keys(),
    }
)


def _expand_braced_commands_once(
    s: str,
    *,
    env_tokens: list[str] | None = None,
) -> tuple[str, bool]:
    r"""Expands one layer of `\cmd{...}` / `\cmd[...]{...}` using exact balanced-brace scanning."""
    out: list[str] = []
    changed = False
    i = 0
    n = len(s)
    while i < n:
        bs = s.find("\\", i)
        if bs == -1:
            out.append(s[i:])
            break
        out.append(s[i:bs])
        m = _LATEX_CMD_TOKEN_RE.match(s, bs)
        if not m:
            out.append(s[bs])
            i = bs + 1
            continue
        cmd = m.group(1)
        after_cmd = m.end()
        if cmd == "operatorname" and after_cmd < n and s[after_cmd] == "*":
            after_cmd += 1

        if cmd not in _ONE_OR_TWO_ARG_CMDS:
            out.append(s[bs:after_cmd])
            i = after_cmd
            continue

        opt_val: str | None = None
        pos = after_cmd
        if cmd == "sqrt":
            opt_res = _extract_bracket_opt(s, pos)
            if opt_res is not None:
                opt_val, pos = opt_res

        arg1_res = _extract_braced(s, pos)
        if arg1_res is None:
            out.append(s[bs:after_cmd])
            i = after_cmd
            continue
        arg1_raw, pos1 = arg1_res
        arg1 = _render_latex_core(arg1_raw, display=False, _env_tokens=env_tokens)

        if cmd in ("frac", "dfrac", "tfrac", "cfrac"):
            arg2_res = _extract_braced(s, pos1)
            if arg2_res is None:
                out.append(s[bs:after_cmd])
                i = after_cmd
                continue
            arg2_raw, pos2 = arg2_res
            arg2 = _render_latex_core(arg2_raw, display=False, _env_tokens=env_tokens)
            changed = True
            i = pos2
            vulgar = _VULGAR_FRACTIONS.get((arg1.strip(), arg2.strip()))
            if vulgar and (pos2 >= n or not s[pos2].isalnum()):
                out.append(vulgar)
            else:
                num_s = f"({arg1})" if _needs_frac_parens(arg1, is_denom=False) else arg1
                den_s = f"({arg2})" if _needs_frac_parens(arg2, is_denom=True) else arg2
                out.append(f"{num_s}/{den_s}")
            continue

        if cmd in ("binom", "dbinom", "tbinom"):
            arg2_res = _extract_braced(s, pos1)
            if arg2_res is not None:
                arg2_raw, pos2 = arg2_res
                arg2 = _render_latex_core(arg2_raw, display=False, _env_tokens=env_tokens)
                changed = True
                i = pos2
                out.append(f"C({arg1}, {arg2})")
                continue

        if cmd in ("overset", "stackrel"):
            arg2_res = _extract_braced(s, pos1)
            if arg2_res is not None:
                arg2_raw, pos2 = arg2_res
                arg2 = _render_latex_core(arg2_raw, display=False, _env_tokens=env_tokens)
                changed = True
                i = pos2
                if arg1.strip() == "def" and arg2.strip() == "=":
                    out.append("≜")
                else:
                    out.append(f"{arg2}{_format_superscript(arg1)}")
                continue

        if cmd == "underset":
            arg2_res = _extract_braced(s, pos1)
            if arg2_res is not None:
                arg2_raw, pos2 = arg2_res
                arg2 = _render_latex_core(arg2_raw, display=False, _env_tokens=env_tokens)
                changed = True
                i = pos2
                out.append(f"{arg2}{_format_subscript(arg1)}")
                continue

        changed = True
        i = pos1
        if cmd == "sqrt":
            body = arg1 if (len(arg1) <= 2 and arg1.isalnum()) else f"({arg1})"
            if opt_val:
                deg = _render_latex_core(opt_val, display=False, _env_tokens=env_tokens).strip()
                if deg == "3":
                    out.append(f"∛{body}")
                elif deg == "4":
                    out.append(f"∜{body}")
                else:
                    out.append(f"{_format_superscript(deg)}√{body}")
            else:
                out.append(f"√{body}")
        elif cmd in ("mathbb", "Bbb"):
            out.append("".join(_MATHBB_MAP.get(ch, ch) for ch in arg1))
        elif cmd == "pmod":
            out.append(f" (mod {arg1})")
        elif cmd == "xrightarrow":
            out.append(f" ─[{arg1}]→ ")
        elif cmd == "xleftarrow":
            out.append(f" ←[{arg1}]─ ")
        elif cmd in ("phantom", "hphantom", "vphantom"):
            pass
        elif cmd in _COMBINING_ACCENTS:
            comb = _COMBINING_ACCENTS[cmd]
            t_arg = arg1.strip()
            if len(t_arg) == 1:
                out.append(unicodedata.normalize("NFC", t_arg + comb))
            else:
                out.append(f"({t_arg}){comb}")
        else:
            # `\text`, `\mathrm`, `\mathbf`, `\mathcal`, `\operatorname`, `\underbrace`, `\cancel`, etc.
            out.append(arg1)

    return "".join(out), changed


def _apply_scripts(s: str, *, env_tokens: list[str] | None = None) -> str:
    """Converts braced & single-char `^` / `_` scripts into clean Unicode super/subscripts."""
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch in ("^", "_") and (i == 0 or s[i - 1] != "\\"):
            is_sup = ch == "^"
            br = _extract_braced(s, i + 1)
            if br is not None:
                raw_inner, next_i = br
                rendered_inner = _render_latex_core(raw_inner, display=False, _env_tokens=env_tokens)
                out.append(_format_superscript(rendered_inner) if is_sup else _format_subscript(rendered_inner))
                i = next_i
                continue
            if i + 1 < n and s[i + 1] == "\\":
                m = _LATEX_CMD_TOKEN_RE.match(s, i + 1)
                if m:
                    sym = _LATEX_SYMBOLS.get(m.group(1), m.group(1))
                    out.append(_format_superscript(sym) if is_sup else _format_subscript(sym))
                    i = m.end()
                    continue
            if i + 1 < n and not s[i + 1].isspace() and s[i + 1] not in "{}^_":
                atom = s[i + 1]
                out.append(_format_superscript(atom) if is_sup else _format_subscript(atom))
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _render_latex_core(
    expr: str,
    *,
    display: bool = False,
    _env_tokens: list[str] | None = None,
) -> str:
    if not expr:
        return ""
    work = expr.strip()
    env_tokens = _env_tokens if _env_tokens is not None else []

    # 1. Innermost-first environments (`\begin{cases}` inside `\begin{aligned}`, `\begin{bmatrix}`, ...)
    def _stash_env(m: re.Match[str]) -> str:
        rendered_env = _render_latex_environment(
            m.group("env"),
            m.group("body"),
            display=display,
            env_tokens=env_tokens,
        )
        idx = len(env_tokens)
        env_tokens.append(normalize_terminal_math_glyphs(rendered_env, use_color=False))
        return f"\x02ENV{idx}\x02"

    for _ in range(6):
        if "\\begin{" not in work:
            break
        new_work = _INNER_ENV_BLOCK_RE.sub(_stash_env, work)
        if new_work == work:
            break
        work = new_work

    # 2. Protect literal escaped braces `\{` and `\}` + `\|`
    work = work.replace(r"\{", "\x01LBRACE\x01").replace(r"\}", "\x01RBRACE\x01")
    work = work.replace(r"\|", "||")

    # 3. Strip sizing modifiers (`\left`, `\right`, `\left.`, `\Bigl`, `\displaystyle`, ...)
    work = _SIZING_PREFIX_RE.sub("", work)

    # 4. Expand nested braced commands (`\frac`, `\sqrt`, `\mathbb`, `\text`, `\hat`, ...)
    for _ in range(10):
        work, changed = _expand_braced_commands_once(work, env_tokens=env_tokens)
        if not changed:
            break

    # 5. Multi-command negated idioms (`\not\in`, `\not=`, `\not\equiv`, `\not\subseteq`)
    work = re.sub(r"\\not\s*\\in(?![A-Za-z])", "∉", work)
    work = re.sub(r"\\not\s*\\equiv(?![A-Za-z])", "≢", work)
    work = re.sub(r"\\not\s*\\subseteq(?![A-Za-z])", "⊈", work)
    work = re.sub(r"\\not\s*\\subset(?![A-Za-z])", "⊄", work)
    work = re.sub(r"\\not\s*=", "≠", work)

    # 6. Spacing & escaped punctuation inside math
    work = work.replace(r"\qquad", "    ").replace(r"\quad", "  ")
    work = work.replace(r"\,", " ").replace(r"\;", " ").replace(r"\:", " ").replace(r"\!", "")
    work = work.replace(r"\%", "%").replace(r"\&", "&").replace(r"\#", "#").replace(r"\$", "$").replace(r"\_", "\x01UND\x01")
    if display and r"\\" in work:
        work = re.sub(r"\\\\(?:\s*\[[^\]]*\])?", "\n", work)
    else:
        work = re.sub(r"\\\\(?:\s*\[[^\]]*\])?", "; ", work)
    work = re.sub(r"\\ ", " ", work)
    work = work.replace("~", " ")

    # 7. Superscripts & subscripts (`^{...}`, `_{...}`, `^2`, `_i`) BEFORE stripping grouping braces!
    work = _apply_scripts(work, env_tokens=env_tokens)

    # 8. Replace remaining `\symbol` commands greedily (`\alpha` -> `α`, `\leq` -> `≤`, `\int` -> `∫`)
    def _repl_sym(m: re.Match[str]) -> str:
        name = m.group(1)
        return _LATEX_SYMBOLS.get(name, name)

    work = _LATEX_CMD_TOKEN_RE.sub(_repl_sym, work)

    # 9. Remove remaining LaTeX grouping `{` and `}`, then restore literal `\{` / `\}` / `\_`
    work = work.replace("{", "").replace("}", "")
    work = work.replace("\x01LBRACE\x01", "{").replace("\x01RBRACE\x01", "}").replace("\x01UND\x01", "_")

    # 10. Clean whitespace per line, then restore pre-aligned nested environments (`cases`, `bmatrix`, `aligned`)
    cleaned_lines = [re.sub(r"[ \t]{2,}", " ", ln).strip() if not display else ln.rstrip() for ln in work.split("\n")]
    res = _restore_env_placeholders("\n".join(cleaned_lines), env_tokens)
    return normalize_terminal_math_glyphs(res, use_color=False)


def render_latex_math(expr: str, *, display: bool = False, use_color: bool = True) -> str:
    """Renders a LaTeX math expression into crisp, font-verified terminal Unicode."""
    rendered = _render_latex_core(expr, display=display)
    if not use_color or not rendered:
        return rendered
    if display:
        return "\n".join(f"\x1b[36m{ln}\x1b[0m" if ln else "" for ln in rendered.split("\n"))
    return f"\x1b[36m{rendered}\x1b[39m"


def _looks_like_currency_span(expr: str) -> bool:
    """Rejects false-positive `$10 for lunch and $` spans while keeping `$10^6$` and `$0 \\le x$`."""
    s = expr.strip()
    if not s:
        return True
    if s[0].isdigit() and " " in s and not any(ch in s for ch in "\\^_=<>+-*/()[]{}"):
        return True
    return False

# ============================================================================
# Lightweight Token-Scanner Syntax Highlighter (Zero Dependencies)
# ============================================================================

_PY_KW = frozenset(
    {
        "False", "None", "True", "and", "as", "assert", "async", "await", "break",
        "class", "continue", "def", "del", "elif", "else", "except", "finally",
        "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
        "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
        "match", "case", "type",
    }
)
_C_JS_GO_RS_KW = frozenset(
    {
        "async", "await", "break", "case", "catch", "class", "const", "continue",
        "default", "defer", "else", "enum", "export", "extends", "false", "finally",
        "fn", "for", "func", "function", "go", "if", "impl", "import", "in",
        "interface", "let", "loop", "match", "mod", "mut", "new", "nil", "null",
        "package", "priv", "pub", "return", "select", "self", "static", "struct",
        "super", "switch", "throw", "trait", "true", "try", "type", "typeof",
        "unsafe", "use", "var", "void", "where", "while", "yield",
    }
)
_SQL_KW = frozenset(
    {
        "SELECT", "FROM", "WHERE", "GROUP", "BY", "ORDER", "HAVING", "LIMIT",
        "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "FULL", "CROSS", "ON", "AS",
        "WITH", "UNION", "ALL", "INSERT", "INTO", "VALUES", "UPDATE", "SET",
        "DELETE", "CREATE", "OR", "REPLACE", "TABLE", "VIEW", "INDEX", "DROP",
        "ALTER", "AND", "NOT", "NULL", "IS", "IN", "EXISTS", "CASE", "WHEN",
        "THEN", "ELSE", "END", "DISTINCT", "COUNT", "SUM", "AVG", "MIN", "MAX",
    }
)
_SH_CMDS = frozenset(
    {
        "cd", "ls", "cp", "mv", "rm", "mkdir", "cat", "grep", "rg", "find", "sed",
        "awk", "curl", "wget", "git", "make", "python", "python3", "pip", "uv",
        "pytest", "npm", "npx", "pnpm", "yarn", "node", "cargo", "go", "docker",
        "kubectl", "gcloud", "bq", "ssh", "scp", "rsync", "export", "source",
        "echo", "printf", "chmod", "chown", "sudo", "crux", "ush", "agy",
    }
)

_PY_TOKEN_RE = re.compile(
    r"(?P<comment>#.*$)"
    r"|(?P<string>(?:[rRbBuUfF]{0,2})(?:\"\"\"[^\"]*\"\"\"|'''[^']*'''|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'))"
    r"|(?P<decorator>@[A-Za-z_][A-Za-z0-9_.]*)"
    r"|(?P<number>\b(?:0[xXoObB][0-9a-fA-F_]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b)"
    r"|(?P<ident>\b[A-Za-z_][A-Za-z0-9_]*\b)"
)
_C_TOKEN_RE = re.compile(
    r"(?P<comment>//.*$|--.*$)"
    r"|(?P<string>`(?:\\.|[^`\\])*`|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')"
    r"|(?P<number>\b(?:0[xXoObB][0-9a-fA-F_]+|\d+(?:\.\d+)?)\b)"
    r"|(?P<ident>\b[A-Za-z_][A-Za-z0-9_]*\b)"
)
_JSON_YAML_RE = re.compile(
    r"(?P<comment>#.*$)"
    r"|(?P<key>\"(?:\\.|[^\"\\])*\"(?=[ \t]*:)|^[ \t]*[A-Za-z0-9_.-]+(?=[ \t]*[:=]))"
    r"|(?P<string>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')"
    r"|(?P<lit>\b(?:true|false|null|True|False|None)\b|-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)"
)
_SH_TOKEN_RE = re.compile(
    r"(?P<comment>(?:^|(?<=[ \t]))#.*$)"
    r"|(?P<string>\"(?:\\.|[^\"\\])*\"|'[^']*')"
    r"|(?P<var>\$\{?[A-Za-z_][A-Za-z0-9_]*\}?)"
    r"|(?P<flag>(?<=[ \t])--?[A-Za-z0-9][A-Za-z0-9_-]*)"
    r"|(?P<ident>\b[A-Za-z_][A-Za-z0-9_.-]*\b)"
)


def _scan_replace(line: str, pattern: re.Pattern[str], styler: Callable[[re.Match[str]], str]) -> str:
    out: list[str] = []
    pos = 0
    for m in pattern.finditer(line):
        if m.start() > pos:
            out.append(line[pos : m.start()])
        out.append(styler(m))
        pos = m.end()
    if pos < len(line):
        out.append(line[pos:])
    return "".join(out)


def highlight_code_line(raw_line: str, lang: str = "", *, use_color: bool = True) -> str:
    """Applies fast, single-pass ANSI syntax highlighting while preserving exact code text."""
    line = raw_line.expandtabs(4)
    if not use_color or not line:
        return line

    norm = (lang or "").strip().lower().split()[0] if (lang or "").strip() else ""

    # 1. Unified diffs / patches
    if norm in ("diff", "patch"):
        if line.startswith(("+++", "---", "diff ", "index ")):
            return f"\x1b[1m{line}\x1b[0m"
        if line.startswith("@@"):
            return f"\x1b[36m{line}\x1b[0m"
        if line.startswith("+"):
            return f"\x1b[32m{line}\x1b[0m"
        if line.startswith("-"):
            return f"\x1b[31m{line}\x1b[0m"
        return line

    # 2. JSON / YAML / TOML / INI
    if norm in ("json", "jsonc", "jsonl", "yaml", "yml", "toml", "ini"):
        def _jy(m: re.Match[str]) -> str:
            if m.group("comment"):
                return f"\x1b[2m{m.group(0)}\x1b[22m"
            if m.group("key"):
                return f"\x1b[36m{m.group(0)}\x1b[39m"
            if m.group("string"):
                return f"\x1b[32m{m.group(0)}\x1b[39m"
            return f"\x1b[33m{m.group(0)}\x1b[39m"

        return _scan_replace(line, _JSON_YAML_RE, _jy) + "\x1b[0m"

    # 3. Shell / Bash / Zsh
    if norm in ("sh", "bash", "zsh", "shell", "console"):
        first_cmd_seen = False

        def _sh(m: re.Match[str]) -> str:
            nonlocal first_cmd_seen
            if m.group("comment"):
                return f"\x1b[2m{m.group(0)}\x1b[22m"
            if m.group("string"):
                return f"\x1b[32m{m.group(0)}\x1b[39m"
            if m.group("var"):
                return f"\x1b[35m{m.group(0)}\x1b[39m"
            if m.group("flag"):
                return f"\x1b[36m{m.group(0)}\x1b[39m"
            tok = m.group("ident") or ""
            if tok in _SH_CMDS and not first_cmd_seen:
                first_cmd_seen = True
                return f"\x1b[1;36m{tok}\x1b[22;39m"
            return tok

        return _scan_replace(line, _SH_TOKEN_RE, _sh) + "\x1b[0m"

    # 4. Python (and default fallback for untagged code snippets that look Pythonic)
    if norm in ("py", "python", "python3", "starlark", "bzl"):
        prev_def = False

        def _py(m: re.Match[str]) -> str:
            nonlocal prev_def
            if m.group("comment"):
                return f"\x1b[2m{m.group(0)}\x1b[22m"
            if m.group("string"):
                return f"\x1b[32m{m.group(0)}\x1b[39m"
            if m.group("decorator"):
                return f"\x1b[33m{m.group(0)}\x1b[39m"
            if m.group("number"):
                return f"\x1b[33m{m.group(0)}\x1b[39m"
            ident = m.group("ident") or ""
            if prev_def:
                prev_def = False
                return f"\x1b[1;33m{ident}\x1b[22;39m"
            if ident in ("def", "class"):
                prev_def = True
                return f"\x1b[1;35m{ident}\x1b[22;39m"
            if ident in _PY_KW:
                return f"\x1b[35m{ident}\x1b[39m"
            if ident in ("self", "cls"):
                return f"\x1b[36m{ident}\x1b[39m"
            return ident

        return _scan_replace(line, _PY_TOKEN_RE, _py) + "\x1b[0m"

    # 5. JS / TS / Go / Rust / C / C++ / Java / SQL
    if norm in (
        "js", "javascript", "ts", "typescript", "tsx", "jsx", "go", "golang",
        "rs", "rust", "c", "cpp", "c++", "cc", "h", "hpp", "java", "kotlin",
        "swift", "sql",
    ):
        is_sql = norm == "sql"

        def _cjs(m: re.Match[str]) -> str:
            if m.group("comment"):
                return f"\x1b[2m{m.group(0)}\x1b[22m"
            if m.group("string"):
                return f"\x1b[32m{m.group(0)}\x1b[39m"
            if m.group("number"):
                return f"\x1b[33m{m.group(0)}\x1b[39m"
            ident = m.group("ident") or ""
            if is_sql and ident.upper() in _SQL_KW:
                return f"\x1b[1;35m{ident}\x1b[22;39m"
            if not is_sql and ident in _C_JS_GO_RS_KW:
                return f"\x1b[35m{ident}\x1b[39m"
            return ident

        return _scan_replace(line, _C_TOKEN_RE, _cjs) + "\x1b[0m"

    return line


# ============================================================================
# Inline Markdown Renderer (`**bold**`, `*italic*`, `` `code` ``, `[link](url)`)
# ============================================================================


def _compact_link_target(label_plain: str, raw_url: str) -> str:
    """Condenses verbose `file:///.../foo.py#L10-L20` URLs into crisp `foo.py:10-20` hints."""
    url = raw_url.strip()
    lbl = label_plain.strip().strip("`")
    if not url or lbl == url or lbl == url.rstrip("/"):
        return ""

    if url.startswith("file://"):
        parsed = urllib.parse.urlparse(url)
        file_path = urllib.parse.unquote(parsed.path or "")
        base = pathlib.Path(file_path).name if file_path else url
        frag = (parsed.fragment or "").strip()
        line_sfx = ""
        if frag:
            # Convert `L401-L497` or `L42` -> `:401-497` / `:42`
            m = re.match(r"^[Ll](\d+)(?:-[Ll]?(\d+))?$", frag)
            if m:
                line_sfx = f":{m.group(1)}-{m.group(2)}" if m.group(2) else f":{m.group(1)}"
            else:
                line_sfx = f"#{frag}"
        short_ref = f"{base}{line_sfx}"
        if lbl == base:
            return line_sfx  # e.g. `[agy_acp.py](file:///.../agy_acp.py#L40)` -> append `:40`
        if lbl == short_ref or short_ref in lbl:
            return ""
        return f" ({short_ref})"

    if url.startswith("conversation://"):
        cid = url[len("conversation://") :].strip("/")
        return "" if cid in lbl else f" ({cid[:8]})"

    # Standard http(s) or relative path
    if len(url) > 76:
        url = url[:73] + "..."
    return f" ({url})"


def render_inline_markdown(
    text: str,
    *,
    use_color: bool = True,
    live_preview: bool = False,
) -> str:
    """Formats inline Markdown, LaTeX math, HTML entities & special Unicode cleanly without mangling code."""
    if not text:
        return ""

    esc_tokens: list[str] = []
    code_tokens: list[tuple[str, bool]] = []  # (code_text, inside_link)
    math_tokens: list[tuple[str, bool]] = []  # (rendered_math_plain, inside_link)

    def _stash_esc_str(val: str) -> str:
        idx = len(esc_tokens)
        esc_tokens.append(val)
        return f"\x00ESC{idx}\x00"

    # 1. Stash escaped backticks `` \` `` first so inline code scanning sees exact backtick boundaries
    work = re.sub(r"\\`", lambda _m: _stash_esc_str("`"), text)

    # 2. Extract inline code spans FIRST so `*`, `_`, `$`, `\alpha`, `&amp;` inside code are 100% untouched!
    def _stash_code(m: re.Match[str]) -> str:
        raw_c = m.group("code")
        if len(raw_c) >= 2 and raw_c.startswith(" ") and raw_c.endswith(" ") and raw_c.strip():
            raw_c = raw_c[1:-1]
        idx = len(code_tokens)
        code_tokens.append((raw_c, False))
        return f"\x00CODE{idx}\x00"

    work = _INLINE_CODE_RE.sub(_stash_code, work)

    # In live streaming preview, softly style a trailing unclosed single backtick (`foo...) without raw jitter
    if live_preview and "`" in work:
        last_bt = work.rfind("`")
        tail_code = work[last_bt + 1 :]
        if tail_code and "`" not in tail_code:
            idx = len(code_tokens)
            code_tokens.append((tail_code, False))
            work = f"{work[:last_bt]}\x00CODE{idx}\x00"

    # 3. Stash escaped `\$` BEFORE math delimiter detection so `\$10 and \$20` stays literal `$10 and $20`
    work = re.sub(r"\\\$", lambda _m: _stash_esc_str("$"), work)

    # 4. Extract & render LaTeX math spans (`$$...$$`, `\[...\]`, `\(...\)`, `$...$`) BEFORE `_ESCAPED_PUNCT_RE`!
    def _stash_math_expr(expr: str) -> str:
        rendered = _render_latex_core(expr, display=False)
        idx = len(math_tokens)
        math_tokens.append((rendered, False))
        return f"\x00MATH{idx}\x00"

    work = _INLINE_DISPLAY_DOLLAR_RE.sub(lambda m: _stash_math_expr(m.group("expr")), work)
    work = _INLINE_DISPLAY_BRACKET_RE.sub(lambda m: _stash_math_expr(m.group("expr")), work)
    work = _INLINE_PAREN_MATH_RE.sub(lambda m: _stash_math_expr(m.group("expr")), work)

    def _maybe_stash_dollar_math(m: re.Match[str]) -> str:
        expr = m.group("expr")
        if _looks_like_currency_span(expr) or "\x00CODE" in expr:
            return m.group(0)
        return _stash_math_expr(expr)

    work = _INLINE_DOLLAR_MATH_RE.sub(_maybe_stash_dollar_math, work)

    # Live streaming preview of an in-progress `$math...` or `\(math...` span containing LaTeX tokens
    if live_preview:
        open_p = work.rfind(r"\(")
        if open_p != -1 and r"\)" not in work[open_p + 2 :]:
            tail_m = work[open_p + 2 :]
            if tail_m.strip():
                work = work[:open_p] + _stash_math_expr(tail_m)
        elif "$" in work:
            last_d = work.rfind("$")
            tail_d = work[last_d + 1 :]
            if tail_d and not tail_d[0].isspace() and any(c in tail_d for c in "\\^_"):
                work = work[:last_d] + _stash_math_expr(tail_d)

    # 5. Expand un-delimited LaTeX braced formatting/math commands in prose (`\frac{a}{b}`, `\textbf{...}`, `\mathbb{R}`)
    if "\\" in work:
        # Map `\textbf{...}`, `\textit{...}`, `\emph{...}`, `\texttt{...}` into native Markdown/code styling
        def _expand_prose_braced(s: str) -> str:
            out_p: list[str] = []
            pos = 0
            n = len(s)
            while pos < n:
                bs = s.find("\\", pos)
                if bs == -1:
                    out_p.append(s[pos:])
                    break
                out_p.append(s[pos:bs])
                cm = _LATEX_CMD_TOKEN_RE.match(s, bs)
                if not cm:
                    out_p.append(s[bs])
                    pos = bs + 1
                    continue
                cname = cm.group(1)
                c_end = cm.end()
                if cname in ("textbf", "textit", "emph", "texttt"):
                    br = _extract_braced(s, c_end)
                    if br is not None:
                        inner, nxt = br
                        if cname == "textbf":
                            out_p.append(f"**{inner}**")
                        elif cname in ("textit", "emph"):
                            out_p.append(f"*{inner}*")
                        else:
                            c_idx = len(code_tokens)
                            code_tokens.append((inner, False))
                            out_p.append(f"\x00CODE{c_idx}\x00")
                        pos = nxt
                        continue
                if cname in ("frac", "dfrac", "tfrac", "binom"):
                    b1 = _extract_braced(s, c_end)
                    if b1 is not None:
                        b2 = _extract_braced(s, b1[1])
                        if b2 is not None:
                            out_p.append(_stash_math_expr(s[bs : b2[1]]))
                            pos = b2[1]
                            continue
                if cname in ("sqrt", "mathbb", "Bbb", "mathcal", "mathrm", "mathbf", "boldsymbol", *_COMBINING_ACCENTS.keys()):
                    opt_after = c_end
                    if cname == "sqrt":
                        opt = _extract_bracket_opt(s, opt_after)
                        if opt is not None:
                            opt_after = opt[1]
                    b1 = _extract_braced(s, opt_after)
                    if b1 is not None:
                        out_p.append(_stash_math_expr(s[bs : b1[1]]))
                        pos = b1[1]
                        continue
                out_p.append(s[bs:c_end])
                pos = c_end
            return "".join(out_p)

        work = _expand_prose_braced(work)
        work = _PROSE_LATEX_CMD_RE.sub(lambda m: _LATEX_SYMBOLS[m.group("cmd")], work)

    # 6. Protect remaining escaped Markdown punctuation (`\*`, `\_`, `\[`, `\|`, etc.)
    work = _ESCAPED_PUNCT_RE.sub(lambda m: _stash_esc_str(m.group(1)), work)

    # 7. Decode HTML entities (`&rarr;`, `&le;`, `&ge;`, `&ne;`, `&mdash;`, `&#x2192;`, `&nbsp;`) outside code spans
    if "&" in work:
        work = _HTML_ENTITY_RE.sub(lambda m: html.unescape(m.group(0)).replace("\u00a0", " "), work)

    # 8. Normalize Plane-1 SMP math alphanumerics (`𝒪`, `𝑥`, `𝔸`) & Menlo-missing subscripts (`xₙ`, `‖x‖`)
    work = normalize_terminal_math_glyphs(work, use_color=use_color)

    def _unstash_plain(s: str) -> str:
        out = re.sub(
            r"\x00CODE(\d+)\x00",
            lambda cm: code_tokens[int(cm.group(1))][0],
            s,
        )
        out = re.sub(
            r"\x00MATH(\d+)\x00",
            lambda mm: math_tokens[int(mm.group(1))][0],
            out,
        )
        return re.sub(
            r"\x00ESC(\d+)\x00",
            lambda em: esc_tokens[int(em.group(1))],
            out,
        )

    # 9. Images `![alt](url)` and Links `[label](url)`
    def _fmt_image(m: re.Match[str]) -> str:
        alt = _unstash_plain(m.group("alt")).strip() or "image"
        url = _unstash_plain(m.group("url"))
        badge = f"[image: {alt}]"
        sfx = _compact_link_target(alt, url)
        if not use_color:
            return f"{badge}{sfx}"
        return f"\x1b[35m{badge}\x1b[39m\x1b[2m{sfx}\x1b[22m"

    def _fmt_link(m: re.Match[str]) -> str:
        raw_lbl = m.group("label")
        for cm in re.finditer(r"\x00CODE(\d+)\x00", raw_lbl):
            c_idx = int(cm.group(1))
            code_tokens[c_idx] = (code_tokens[c_idx][0], True)
        for mm in re.finditer(r"\x00MATH(\d+)\x00", raw_lbl):
            m_idx = int(mm.group(1))
            math_tokens[m_idx] = (math_tokens[m_idx][0], True)
        plain_lbl = _unstash_plain(raw_lbl)
        url = _unstash_plain(m.group("url"))
        sfx = _compact_link_target(plain_lbl, url)
        if not use_color:
            return f"{raw_lbl}{sfx}"
        if sfx.startswith(":"):
            return f"\x1b[4;36m{raw_lbl}{sfx}\x1b[24;39m"
        dim_sfx = f"\x1b[2m{sfx}\x1b[22m" if sfx else ""
        return f"\x1b[4;36m{raw_lbl}\x1b[24;39m{dim_sfx}"

    work = _IMAGE_RE.sub(_fmt_image, work)
    work = _LINK_RE.sub(_fmt_link, work)

    # 10. Emphasis & Strikethrough (scoped SGR resets so nested styles compose cleanly!)
    if use_color:
        work = _BOLD_ITALIC_AST_RE.sub(lambda m: f"\x1b[1;3m{m.group(1)}\x1b[22;23m", work)
        work = _BOLD_AST_RE.sub(lambda m: f"\x1b[1m{m.group(1)}\x1b[22m", work)
        work = _BOLD_UND_RE.sub(lambda m: f"\x1b[1m{m.group(1)}\x1b[22m", work)
        work = _ITALIC_AST_RE.sub(lambda m: f"\x1b[3m{m.group(1)}\x1b[23m", work)
        work = _ITALIC_UND_RE.sub(lambda m: f"\x1b[3m{m.group(1)}\x1b[23m", work)
        work = _STRIKE_RE.sub(lambda m: f"\x1b[9m{m.group(1)}\x1b[29m", work)

        if live_preview and "**" in work:
            b_idx = work.rfind("**")
            tail_b = work[b_idx + 2 :]
            if tail_b.strip():
                work = f"{work[:b_idx]}\x1b[1m{tail_b}\x1b[22m"
    else:
        work = _BOLD_ITALIC_AST_RE.sub(lambda m: m.group(1), work)
        work = _BOLD_AST_RE.sub(lambda m: m.group(1), work)
        work = _BOLD_UND_RE.sub(lambda m: m.group(1), work)
        work = _ITALIC_AST_RE.sub(lambda m: m.group(1), work)
        work = _ITALIC_UND_RE.sub(lambda m: m.group(1), work)
        work = _STRIKE_RE.sub(lambda m: m.group(1), work)

    # 11. Restore inline math spans, inline code spans & escaped punctuation
    def _restore_math(m: re.Match[str]) -> str:
        m_txt, in_link = math_tokens[int(m.group(1))]
        if not use_color or in_link:
            return m_txt
        return f"\x1b[36m{m_txt}\x1b[39m"

    def _restore_code(m: re.Match[str]) -> str:
        c_txt, in_link = code_tokens[int(m.group(1))]
        c_txt = re.sub(r"\x00ESC(\d+)\x00", lambda em: esc_tokens[int(em.group(1))], c_txt)
        if not use_color:
            return c_txt if in_link else f"`{c_txt}`"
        if in_link:
            return f"\x1b[1;36m{c_txt}\x1b[22m"
        return f"\x1b[36m{c_txt}\x1b[39m"

    work = re.sub(r"\x00MATH(\d+)\x00", _restore_math, work)
    work = re.sub(r"\x00CODE(\d+)\x00", _restore_code, work)
    work = re.sub(r"\x00ESC(\d+)\x00", lambda em: esc_tokens[int(em.group(1))], work)

    if use_color and "\x1b[" in work and not work.endswith("\x1b[0m"):
        work += "\x1b[0m"
    return work


# ============================================================================
# Self-Contained Per-Line ANSI Word Wrapper (`INV-TUI-1` Safe)
# ============================================================================


def _update_sgr_state(state: dict[str, str], seq: str) -> None:
    """Tracks active SGR attributes so wrapped continuation lines re-open exact styles cleanly."""
    if not (seq.startswith("\x1b[") and seq.endswith("m")):
        return
    raw_params = seq[2:-1]
    codes = [int(p) if p.isdigit() else 0 for p in (raw_params.split(";") if raw_params else ["0"])]
    i = 0
    while i < len(codes):
        c = codes[i]
        if c == 0:
            state.clear()
        elif c == 1:
            state["weight"] = "1"
        elif c == 2:
            state["weight"] = "2"
        elif c == 22:
            state.pop("weight", None)
        elif c == 3:
            state["italic"] = "3"
        elif c == 23:
            state.pop("italic", None)
        elif c == 4:
            state["underline"] = "4"
        elif c == 24:
            state.pop("underline", None)
        elif c == 9:
            state["strike"] = "9"
        elif c == 29:
            state.pop("strike", None)
        elif 30 <= c <= 37 or 90 <= c <= 97:
            state["fg"] = str(c)
        elif c == 38 and i + 2 < len(codes) and codes[i + 1] == 5:
            state["fg"] = f"38;5;{codes[i + 2]}"
            i += 2
        elif c == 39:
            state.pop("fg", None)
        i += 1


def _sgr_prefix(state: dict[str, str]) -> str:
    if not state:
        return ""
    ordered = [state[k] for k in ("weight", "italic", "underline", "strike", "fg") if k in state]
    return f"\x1b[{';'.join(ordered)}m" if ordered else ""


def wrap_ansi_line(
    text: str,
    width: int,
    *,
    initial_indent: str = "",
    subsequent_indent: str = "",
) -> list[str]:
    """Word-wraps `text` to `width` visible columns, keeping each output line 100% ANSI-self-contained."""
    eff_w = max(20, width)
    if not text:
        return [initial_indent.rstrip()] if initial_indent else [""]

    init_w = vis_width(initial_indent)
    sub_w = vis_width(subsequent_indent)
    if init_w + vis_width(text) <= eff_w:
        full = f"{initial_indent}{text}"
        if "\x1b[" in full and not full.endswith("\x1b[0m"):
            full += "\x1b[0m"
        return [full]

    # Tokenize into `(raw_piece, vis_cols, is_space, sgr_seqs)` preserving ANSI sequences attached to words
    tokens: list[tuple[str, int, bool, list[str]]] = []
    i = 0
    n = len(text)
    cur_raw: list[str] = []
    cur_w = 0
    cur_seqs: list[str] = []
    cur_space: bool | None = None

    def _flush_tok() -> None:
        nonlocal cur_raw, cur_w, cur_seqs, cur_space
        if cur_raw:
            tokens.append(("".join(cur_raw), cur_w, bool(cur_space), cur_seqs))
            cur_raw = []
            cur_w = 0
            cur_seqs = []
            cur_space = None

    while i < n:
        if text[i] == "\x1b":
            m = ANSI_ESCAPE_RE.match(text, i)
            if m:
                seq = m.group(0)
                cur_raw.append(seq)
                cur_seqs.append(seq)
                i = m.end()
                continue
        ch = text[i]
        is_sp = ch == " "
        if cur_space is not None and is_sp != cur_space and cur_w > 0:
            _flush_tok()
        cur_space = is_sp
        cur_raw.append(ch)
        cur_w += char_width(ch)
        i += 1
    _flush_tok()

    lines: list[str] = []
    cur_indent = initial_indent
    cur_col = init_w
    line_parts: list[str] = [cur_indent]
    active_sgr: dict[str, str] = {}

    def _commit_line() -> None:
        nonlocal cur_indent, cur_col, line_parts
        joined = "".join(line_parts)
        if "\x1b[" in joined and not joined.endswith("\x1b[0m"):
            joined += "\x1b[0m"
        lines.append(joined)
        cur_indent = subsequent_indent
        cur_col = sub_w
        reopen = _sgr_prefix(active_sgr)
        line_parts = [cur_indent + reopen] if reopen else [cur_indent]

    for raw_tok, tok_w, is_sp, seqs in tokens:
        if is_sp and cur_col == (init_w if not lines else sub_w):
            # Drop leading spaces at start of wrapped continuation line (still apply any SGR sequences!)
            for sq in seqs:
                _update_sgr_state(active_sgr, sq)
                line_parts.append(sq)
            continue

        if cur_col + tok_w <= eff_w:
            line_parts.append(raw_tok)
            cur_col += tok_w
            for sq in seqs:
                _update_sgr_state(active_sgr, sq)
            continue

        if not is_sp and tok_w > (eff_w - sub_w):
            # Single token wider than viewport (e.g. long URL or path) -> break cleanly by column
            if cur_col > (init_w if not lines else sub_w):
                _commit_line()
            j = 0
            tn = len(raw_tok)
            while j < tn:
                if raw_tok[j] == "\x1b":
                    m = ANSI_ESCAPE_RE.match(raw_tok, j)
                    if m:
                        sq = m.group(0)
                        line_parts.append(sq)
                        _update_sgr_state(active_sgr, sq)
                        j = m.end()
                        continue
                cw = char_width(raw_tok[j])
                if cur_col + cw > eff_w and cur_col > sub_w:
                    _commit_line()
                line_parts.append(raw_tok[j])
                cur_col += cw
                j += 1
            continue

        _commit_line()
        if not is_sp:
            line_parts.append(raw_tok)
            cur_col += tok_w
        for sq in seqs:
            _update_sgr_state(active_sgr, sq)

    if line_parts:
        joined_last = "".join(line_parts)
        if strip_ansi(joined_last).strip() or not lines:
            if "\x1b[" in joined_last and not joined_last.endswith("\x1b[0m"):
                joined_last += "\x1b[0m"
            lines.append(joined_last)

    return lines


def _pad_ansi(text: str, target_w: int, align: str = "left") -> str:
    w = vis_width(text)
    pad = max(0, target_w - w)
    if align == "right":
        return (" " * pad) + text
    if align == "center":
        left = pad // 2
        return (" " * left) + text + (" " * (pad - left))
    return text + (" " * pad)


# ============================================================================
# GFM Pipe Table Parser & Box-Drawing Formatter
# ============================================================================


def _split_table_row(line: str) -> list[str] | None:
    """Splits a Markdown `| a | b |` row while respecting `\\|`, `` `a | b` ``, and `$|x|$` math spans."""
    s = line.strip()
    if "|" not in s:
        return None
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith(r"\|"):
        s = s[:-1]

    cells: list[str] = []
    cur: list[str] = []
    in_ticks = 0
    in_math = False
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            cur.append(s[i : i + 2])
            i += 2
            continue
        if ch == "`":
            run = 1
            while i + run < n and s[i + run] == "`":
                run += 1
            if in_ticks == 0:
                in_ticks = run
            elif in_ticks == run:
                in_ticks = 0
            cur.append(s[i : i + run])
            i += run
            continue
        if ch == "$" and in_ticks == 0:
            if in_math:
                in_math = False
            elif "$" in s[i + 1 :]:
                in_math = True
            cur.append(ch)
            i += 1
            continue
        if ch == "|" and in_ticks == 0 and not in_math:
            cells.append("".join(cur).strip())
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    cells.append("".join(cur).strip())
    return cells if len(cells) >= 2 or line.lstrip().startswith("|") else None


def _parse_delimiter_row(line: str) -> list[str] | None:
    """Returns `['left'|'center'|'right', ...]` if `line` is a valid GFM `| :--- | ---: |` separator."""
    cells = _split_table_row(line)
    if not cells:
        return None
    aligns: list[str] = []
    for c in cells:
        cs = c.strip()
        if not re.match(r"^:?-{3,}:?$", cs):
            return None
        if cs.startswith(":") and cs.endswith(":"):
            aligns.append("center")
        elif cs.endswith(":"):
            aligns.append("right")
        else:
            aligns.append("left")
    return aligns


def format_markdown_table(
    raw_lines: list[str],
    *,
    max_width: int = 100,
    use_color: bool = True,
) -> list[str] | None:
    """Formats `raw_lines` as a rounded box-drawn table (`╭─┬─╮`), or `None` if not a valid table."""
    if len(raw_lines) < 2:
        return None
    header_cells = _split_table_row(raw_lines[0])
    aligns = _parse_delimiter_row(raw_lines[1])
    if not header_cells or not aligns:
        return None

    ncols = max(len(header_cells), len(aligns))
    while len(aligns) < ncols:
        aligns.append("left")

    def _norm_row(cells: list[str]) -> list[str]:
        padded = list(cells[:ncols])
        while len(padded) < ncols:
            padded.append("")
        return padded

    hdr_fmt = [
        (f"\x1b[1;36m{render_inline_markdown(c, use_color=use_color)}\x1b[0m" if use_color else render_inline_markdown(c, use_color=False))
        for c in _norm_row(header_cells)
    ]
    body_rows: list[list[str]] = []
    for rline in raw_lines[2:]:
        rc = _split_table_row(rline)
        if rc is None:
            continue
        body_rows.append([render_inline_markdown(c, use_color=use_color) for c in _norm_row(rc)])

    # Compute natural column widths
    col_w = [max(3, vis_width(hdr_fmt[c])) for c in range(ncols)]
    for row in body_rows:
        for c in range(ncols):
            col_w[c] = max(col_w[c], vis_width(row[c]))

    # Shrink proportionally if wider than `max_width` (`2` indent + `3 * ncols + 1` borders/padding)
    overhead = 2 + (3 * ncols + 1)
    avail = max(ncols * 6, max_width - overhead)
    total_w = sum(col_w)
    if total_w > avail:
        floor_w = [min(col_w[c], max(6, avail // (ncols * 2))) for c in range(ncols)]
        rem = max(0, avail - sum(floor_w))
        extra_need = sum(max(0, col_w[c] - floor_w[c]) for c in range(ncols))
        new_w = list(floor_w)
        if extra_need > 0 and rem > 0:
            for c in range(ncols):
                share = int(rem * max(0, col_w[c] - floor_w[c]) / extra_need)
                new_w[c] += share
        col_w = [max(4, w) for w in new_w]

    def _b(s: str) -> str:
        return f"\x1b[2m{s}\x1b[0m" if use_color else s

    v_bar = _b("│")
    top = "  " + _b("╭" + "┬".join("─" * (w + 2) for w in col_w) + "╮")
    mid = "  " + _b("├" + "┼".join("─" * (w + 2) for w in col_w) + "┤")
    bot = "  " + _b("╰" + "┴".join("─" * (w + 2) for w in col_w) + "╯")

    def _render_row(cells_fmt: list[str], *, is_header: bool = False) -> list[str]:
        wrapped_cols = [wrap_ansi_line(cells_fmt[c], col_w[c]) for c in range(ncols)]
        height = max(len(wc) for wc in wrapped_cols)
        out_r: list[str] = []
        for r_idx in range(height):
            segs: list[str] = []
            for c in range(ncols):
                cell_ln = wrapped_cols[c][r_idx] if r_idx < len(wrapped_cols[c]) else ""
                if is_header and use_color and cell_ln and not cell_ln.startswith("\x1b[1;36m"):
                    cell_ln = f"\x1b[1;36m{cell_ln}\x1b[0m"
                al = "center" if (is_header and aligns[c] == "center") else ("left" if is_header else aligns[c])
                segs.append(" " + _pad_ansi(cell_ln, col_w[c], al) + " ")
            out_r.append("  " + v_bar + v_bar.join(segs) + v_bar)
        return out_r

    out_lines: list[str] = [top]
    out_lines.extend(_render_row(hdr_fmt, is_header=True))
    if body_rows:
        out_lines.append(mid)
        for row in body_rows:
            out_lines.extend(_render_row(row, is_header=False))
    out_lines.append(bot)
    return out_lines


# ============================================================================
# Stateful Streaming & Full-Document Markdown Renderer
# ============================================================================

_ALERT_STYLES: dict[str, tuple[str, str, str]] = {
    "NOTE": ("◈", "Note", "36"),        # Cyan
    "TIP": ("◆", "Tip", "32"),          # Green
    "IMPORTANT": ("▸", "Important", "35"),  # Magenta
    "WARNING": ("!", "Warning", "33"),  # Yellow
    "CAUTION": ("✗", "Caution", "31"),  # Red
}


class MarkdownStreamRenderer:
    """Line-accurate streaming Markdown formatter with live partial-line preview & full table alignment."""

    def __init__(
        self,
        *,
        use_color: bool = True,
        cols_fn: Callable[[], int] | None = None,
    ) -> None:
        self.use_color = use_color
        self._cols_fn = cols_fn or (lambda: shutil.get_terminal_size((100, 30)).columns)
        self.reset()

    def reset(self) -> None:
        self._line_buf = ""
        self._in_fence = False
        self._fence_char = ""
        self._fence_len = 0
        self._fence_lang = ""
        self._in_math_block = False
        self._math_close_kind = ""  # "$$" | "\\]" | "fence"
        self._math_buf: list[str] = []
        self._table_buf: list[str] = []
        self._active_alert: str | None = None
        self._blank_streak = 0
        self._emitted_any = False

    def _width(self) -> int:
        return max(48, self._cols_fn())

    def _prose_wrap_width(self) -> int:
        return min(self._width() - 2, 108)

    def _c(self, code: str, text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.use_color else text

    def _emit_lines(self, out_buf: list[str], lines: list[str]) -> None:
        for ln in lines:
            is_blank = not strip_ansi(ln).strip()
            if is_blank:
                if not self._emitted_any or self._blank_streak >= 1:
                    continue
                self._blank_streak += 1
                out_buf.append("")
            else:
                self._blank_streak = 0
                self._emitted_any = True
                out_buf.append(ln)

    def _flush_table_buf(self, out_buf: list[str]) -> None:
        if not self._table_buf:
            return
        rows = list(self._table_buf)
        self._table_buf.clear()
        rendered_tbl = format_markdown_table(rows, max_width=self._width() - 2, use_color=self.use_color)
        if rendered_tbl is not None:
            self._emit_lines(out_buf, rendered_tbl)
            return
        # Not a valid table (e.g. single pipe line with no `|---|---|` delimiter) -> render as normal lines
        for r in rows:
            self._format_non_table_line(r, out_buf)

    def _emit_display_math_block(self, raw_math_lines: list[str], out_buf: list[str]) -> None:
        expr = "\n".join(raw_math_lines).strip()
        if not expr:
            return
        rendered = render_latex_math(expr, display=True, use_color=self.use_color)
        indented = [f"    {ln}" if ln else "" for ln in rendered.split("\n")]
        self._emit_lines(out_buf, indented)

    def _format_single_line_preview(self, partial: str) -> list[str]:
        """Formats the active incomplete line for smooth, raw-syntax-free live dock preview."""
        if not partial:
            if self._in_math_block and self._math_buf:
                prev_m = render_latex_math("\n".join(self._math_buf[-3:]), display=True, use_color=self.use_color)
                return [f"    {ln}" for ln in prev_m.split("\n") if ln]
            if self._table_buf:
                n_r = len(self._table_buf)
                return [self._c("2", f"  ╭─ formatting table ({n_r} row{'s' if n_r != 1 else ''})…")]
            return []

        max_w = max(24, self._width() - 2)
        if self._in_math_block:
            if partial.strip() in ("$$", r"\]"):
                prev_m = render_latex_math("\n".join(self._math_buf[-3:]), display=True, use_color=self.use_color)
                return [f"    {ln}" for ln in prev_m.split("\n") if ln]
            live_lines = [*self._math_buf[-2:], partial]
            prev_m = render_latex_math("\n".join(live_lines), display=True, use_color=self.use_color)
            return [f"    {ln}" for ln in prev_m.split("\n") if ln]

        if self._in_fence:
            fm = _FENCE_OPEN_RE.match(partial)
            if fm and fm.group("fence")[0] == self._fence_char and len(fm.group("fence")) >= self._fence_len and not fm.group("info").strip():
                rule_w = min(56, max(20, max_w - 4))
                return [self._c("2", f"  ╰{'─' * rule_w}")]
            gutter = self._c("2", "  │ ")
            hl = highlight_code_line(partial, self._fence_lang, use_color=self.use_color)
            return [f"{gutter}{hl}"]

        if self._table_buf:
            return [self._c("2", f"  │ {partial.strip()}")]

        # Heading preview
        hm = _HEADING_RE.match(partial)
        if hm:
            lvl = len(hm.group("hashes"))
            title = render_inline_markdown(hm.group("title"), use_color=self.use_color, live_preview=True)
            if lvl == 1:
                return [self._c("1;36", title)]
            if lvl == 2:
                return [f"{self._c('1;36', '◆')} {self._c('1;36', title)}"]
            if lvl == 3:
                return [f"{self._c('36', '▸')} {self._c('1', title)}"]
            return [f"{self._c('2', '•')} {self._c('1', title)}"]

        # Bullet / task / numbered preview
        tm = _TASK_ITEM_RE.match(partial)
        if tm:
            depth = len(tm.group("indent").expandtabs(2)) // 2
            pad = "  " * (depth + 1)
            checked = tm.group("mark").lower() == "x"
            box = self._c("1;32", "✓") if checked else self._c("2", "○")
            body = render_inline_markdown(tm.group("body"), use_color=self.use_color, live_preview=True)
            return wrap_ansi_line(body, max_w, initial_indent=f"{pad}{box} ", subsequent_indent=f"{pad}  ")

        um = _UL_ITEM_RE.match(partial)
        if um:
            depth = len(um.group("indent").expandtabs(2)) // 2
            pad = "  " * (depth + 1)
            glyph = ("•", "◦", "▪")[min(depth, 2)]
            bul = self._c("36", glyph)
            body = render_inline_markdown(um.group("body"), use_color=self.use_color, live_preview=True)
            return wrap_ansi_line(body, max_w, initial_indent=f"{pad}{bul} ", subsequent_indent=f"{pad}  ")

        om = _OL_ITEM_RE.match(partial)
        if om:
            depth = len(om.group("indent").expandtabs(2)) // 2
            pad = "  " * (depth + 1)
            num_lbl = f"{om.group('num')}."
            pfx = f"{pad}{self._c('1;36', num_lbl)} "
            sub_pad = pad + (" " * (len(num_lbl) + 1))
            body = render_inline_markdown(om.group("body"), use_color=self.use_color, live_preview=True)
            return wrap_ansi_line(body, max_w, initial_indent=pfx, subsequent_indent=sub_pad)

        bm = _BLOCKQUOTE_RE.match(partial)
        if bm:
            bar_col = _ALERT_STYLES.get(self._active_alert or "", ("", "", "36"))[2]
            bar = f"  {self._c(bar_col, '▎')} "
            body = render_inline_markdown(bm.group("body"), use_color=self.use_color, live_preview=True)
            return wrap_ansi_line(body, max_w, initial_indent=bar, subsequent_indent=bar)

        styled = render_inline_markdown(partial, use_color=self.use_color, live_preview=True)
        return wrap_ansi_line(styled, self._prose_wrap_width())

    def _process_complete_line(self, raw_line: str, out_buf: list[str]) -> None:
        line = raw_line.rstrip("\r")

        # 0. Inside multi-line Display Math block (`$$ ... $$`, `\[ ... \]`, or ```` ```math ````)
        if self._in_math_block:
            s_trim = line.strip()
            if self._math_close_kind == "fence":
                fm = _FENCE_OPEN_RE.match(line)
                if (
                    fm
                    and fm.group("fence")[0] == self._fence_char
                    and len(fm.group("fence")) >= self._fence_len
                    and not fm.group("info").strip()
                ):
                    lines_m = list(self._math_buf)
                    self._math_buf.clear()
                    self._in_math_block = False
                    self._math_close_kind = ""
                    self._fence_char = ""
                    self._fence_len = 0
                    self._emit_display_math_block(lines_m, out_buf)
                    return
                self._math_buf.append(line)
                return

            close_tok = self._math_close_kind
            if s_trim == close_tok or (s_trim.endswith(close_tok) and not s_trim.endswith("\\" + close_tok)):
                before = s_trim[: -len(close_tok)].strip()
                if before:
                    self._math_buf.append(before)
                lines_m = list(self._math_buf)
                self._math_buf.clear()
                self._in_math_block = False
                self._math_close_kind = ""
                self._emit_display_math_block(lines_m, out_buf)
                return

            self._math_buf.append(line)
            return

        # 1. Inside fenced code block
        if self._in_fence:
            fm = _FENCE_OPEN_RE.match(line)
            if (
                fm
                and fm.group("fence")[0] == self._fence_char
                and len(fm.group("fence")) >= self._fence_len
                and not fm.group("info").strip()
            ):
                rule_w = min(60, max(24, self._width() - 6))
                self._emit_lines(out_buf, [self._c("2", f"  ╰{'─' * rule_w}")])
                self._in_fence = False
                self._fence_char = ""
                self._fence_len = 0
                self._fence_lang = ""
                return
            gutter = self._c("2", "  │ ")
            hl = highlight_code_line(line, self._fence_lang, use_color=self.use_color)
            # Never skip blank lines inside code blocks!
            self._blank_streak = 0
            self._emitted_any = True
            out_buf.append(f"{gutter}{hl}")
            return

        # 2. GFM Pipe Table buffering (`| col | col |`)
        if self._table_buf:
            if len(self._table_buf) == 1:
                if _parse_delimiter_row(line) is not None:
                    self._table_buf.append(line)
                    return
                # Line 2 wasn't a `|---|---|` separator -> flush Line 1 as normal text immediately!
                first_row = self._table_buf.pop(0)
                self._format_non_table_line(first_row, out_buf)
            else:
                if line.strip() and _split_table_row(line) is not None:
                    self._table_buf.append(line)
                    return
                self._flush_table_buf(out_buf)

        if not self._table_buf and "|" in line and _split_table_row(line) is not None and _parse_delimiter_row(line) is None:
            # Candidate table header row — wait 1 line to verify `|---|---|` separator
            s_strip = line.strip()
            if s_strip.startswith("|") or s_strip.count("|") >= 2:
                self._table_buf.append(line)
                return

        self._format_non_table_line(line, out_buf)

    def _format_non_table_line(self, line: str, out_buf: list[str]) -> None:
        stripped = line.strip()
        if not stripped:
            self._active_alert = None
            self._emit_lines(out_buf, [""])
            return

        # 0. Display Math Blocks (`$$ ... $$` and `\[ ... \]`)
        if stripped.startswith("$$"):
            if stripped != "$$" and stripped.endswith("$$") and len(stripped) >= 4 and stripped.count("$$") == 2:
                self._active_alert = None
                self._emit_display_math_block([stripped[2:-2]], out_buf)
                return
            if stripped == "$$" or "$$" not in stripped[2:]:
                self._active_alert = None
                self._in_math_block = True
                self._math_close_kind = "$$"
                self._math_buf = [stripped[2:].strip()] if stripped[2:].strip() else []
                return

        if stripped.startswith(r"\["):
            if stripped.endswith(r"\]") and len(stripped) >= 4 and r"\]" not in stripped[2:-2]:
                self._active_alert = None
                self._emit_display_math_block([stripped[2:-2]], out_buf)
                return
            if r"\]" not in stripped[2:]:
                self._active_alert = None
                self._in_math_block = True
                self._math_close_kind = r"\]"
                self._math_buf = [stripped[2:].strip()] if stripped[2:].strip() else []
                return

        # 1. Opening Fenced Code Block (```python, ~~~bash, or ```math)
        fm = _FENCE_OPEN_RE.match(line)
        if fm:
            self._active_alert = None
            info = fm.group("info").strip()
            lang_label = info.split()[0] if info else "code"
            if lang_label.lower() in ("math", "katex", "mathjax"):
                self._in_math_block = True
                self._math_close_kind = "fence"
                self._fence_char = fm.group("fence")[0]
                self._fence_len = len(fm.group("fence"))
                self._math_buf = []
                return
            self._in_fence = True
            self._fence_char = fm.group("fence")[0]
            self._fence_len = len(fm.group("fence"))
            self._fence_lang = lang_label if info else ""
            rule_w = min(60, max(24, self._width() - 6))
            dash_rem = max(4, rule_w - len(lang_label) - 3)
            hdr = (
                f"  {self._c('2', '╭─ ')}{self._c('1;36', lang_label)} "
                f"{self._c('2', '─' * dash_rem)}"
            )
            self._emit_lines(out_buf, [hdr])
            return

        # 2. Horizontal Rule (`---`, `***`, `___`)
        if _HR_RE.match(line):
            self._active_alert = None
            rule_w = min(64, max(24, self._width() - 6))
            self._emit_lines(out_buf, [self._c("2", "  " + ("─" * rule_w))])
            return

        # 3. ATX Headings (`# H1` .. `###### H6`)
        hm = _HEADING_RE.match(line)
        if hm:
            self._active_alert = None
            lvl = len(hm.group("hashes"))
            title = render_inline_markdown(hm.group("title"), use_color=self.use_color)
            if self._emitted_any and self._blank_streak == 0:
                self._emit_lines(out_buf, [""])
            if lvl == 1:
                t_styled = self._c("1;36", title)
                ul_w = min(max(4, vis_width(title)), self._width() - 4)
                self._emit_lines(out_buf, [t_styled, self._c("36", "━" * ul_w)])
            elif lvl == 2:
                self._emit_lines(out_buf, [f"{self._c('1;36', '◆')} {self._c('1;36', title)}"])
            elif lvl == 3:
                self._emit_lines(out_buf, [f"{self._c('36', '▸')} {self._c('1', title)}"])
            else:
                self._emit_lines(out_buf, [f"{self._c('2', '•')} {self._c('1', title)}"])
            return

        # 4. Blockquotes & GitHub Alerts (`> [!NOTE]`, `> text`)
        bm = _BLOCKQUOTE_RE.match(line)
        if bm:
            qbody = bm.group("body")
            am = _ALERT_TAG_RE.match(qbody.strip())
            if am:
                kind = am.group(1).upper()
                self._active_alert = kind
                glyph, label, col = _ALERT_STYLES[kind]
                bar = f"  {self._c(col, '▎')} "
                hdr_line = f"{bar}{self._c(f'1;{col}', f'{glyph} {label}')}"
                self._emit_lines(out_buf, [hdr_line])
                rest = am.group("rest").strip()
                if rest:
                    styled_r = render_inline_markdown(rest, use_color=self.use_color)
                    self._emit_lines(
                        out_buf,
                        wrap_ansi_line(styled_r, self._prose_wrap_width(), initial_indent=bar, subsequent_indent=bar),
                    )
                return

            col = _ALERT_STYLES[self._active_alert][2] if self._active_alert in _ALERT_STYLES else "36"
            bar = f"  {self._c(col, '▎')} "
            if not qbody.strip():
                self._emit_lines(out_buf, [bar.rstrip()])
                return
            styled_q = render_inline_markdown(qbody, use_color=self.use_color)
            if self._active_alert is None and self.use_color:
                styled_q = f"\x1b[3m{styled_q}\x1b[0m"
            self._emit_lines(
                out_buf,
                wrap_ansi_line(styled_q, self._prose_wrap_width(), initial_indent=bar, subsequent_indent=bar),
            )
            return

        self._active_alert = None
        wrap_w = self._prose_wrap_width()

        # 5. Task-list Checkbox Items (`- [x] Done` / `- [ ] Todo`)
        tm = _TASK_ITEM_RE.match(line)
        if tm:
            depth = len(tm.group("indent").expandtabs(2)) // 2
            pad = "  " * (depth + 1)
            checked = tm.group("mark").lower() == "x"
            box = self._c("1;32", "✓") if checked else self._c("2", "○")
            body = render_inline_markdown(tm.group("body"), use_color=self.use_color)
            self._emit_lines(
                out_buf,
                wrap_ansi_line(body, wrap_w, initial_indent=f"{pad}{box} ", subsequent_indent=f"{pad}  "),
            )
            return

        # 6. Unordered List Items (`- `, `* `, `+ `)
        um = _UL_ITEM_RE.match(line)
        if um:
            depth = len(um.group("indent").expandtabs(2)) // 2
            pad = "  " * (depth + 1)
            glyph = ("•", "◦", "▪")[min(depth, 2)]
            bul = self._c("36", glyph)
            body = render_inline_markdown(um.group("body"), use_color=self.use_color)
            self._emit_lines(
                out_buf,
                wrap_ansi_line(body, wrap_w, initial_indent=f"{pad}{bul} ", subsequent_indent=f"{pad}  "),
            )
            return

        # 7. Ordered List Items (`1. `, `2) `)
        om = _OL_ITEM_RE.match(line)
        if om:
            depth = len(om.group("indent").expandtabs(2)) // 2
            pad = "  " * (depth + 1)
            num_lbl = f"{om.group('num')}."
            pfx = f"{pad}{self._c('1;36', num_lbl)} "
            sub_pad = pad + (" " * (len(num_lbl) + 1))
            body = render_inline_markdown(om.group("body"), use_color=self.use_color)
            self._emit_lines(
                out_buf,
                wrap_ansi_line(body, wrap_w, initial_indent=pfx, subsequent_indent=sub_pad),
            )
            return

        # 8. Preserve pre-drawn ASCII / Box-Drawing Diagrams without word-wrapping
        if any(ch in _BOX_ART_CHARS for ch in line):
            self._emit_lines(out_buf, [render_inline_markdown(line, use_color=self.use_color)])
            return

        # 9. Standard Prose Paragraph Line (preserving leading indentation for list continuations!)
        lead_ws_len = len(line) - len(line.lstrip(" \t"))
        lead_pad = line[:lead_ws_len].expandtabs(2)
        styled_p = render_inline_markdown(line[lead_ws_len:], use_color=self.use_color)
        self._emit_lines(
            out_buf,
            wrap_ansi_line(styled_p, wrap_w, initial_indent=lead_pad, subsequent_indent=lead_pad),
        )

    def feed(self, chunk: str) -> tuple[str, list[str]]:
        """Ingests a streamed text delta; returns `(committed_lines_with_nl, live_preview_lines)`."""
        if not chunk:
            return "", self._format_single_line_preview(self._line_buf)

        combined = self._line_buf + chunk.replace("\r\n", "\n")
        parts = combined.split("\n")
        self._line_buf = parts[-1]

        out_lines: list[str] = []
        for complete in parts[:-1]:
            self._process_complete_line(complete, out_lines)

        committed = ("\n".join(out_lines) + "\n") if out_lines else ""
        preview = self._format_single_line_preview(self._line_buf)
        return committed, preview

    def flush(self) -> str:
        """Flushes any trailing partial line, buffered table, or open code fence at end of turn/step."""
        out_lines: list[str] = []
        if self._line_buf:
            tail = self._line_buf
            self._line_buf = ""
            self._process_complete_line(tail, out_lines)
        if self._table_buf:
            self._flush_table_buf(out_lines)
        if self._in_math_block:
            lines_m = list(self._math_buf)
            self._math_buf.clear()
            self._in_math_block = False
            self._math_close_kind = ""
            self._fence_char = ""
            self._fence_len = 0
            self._emit_display_math_block(lines_m, out_lines)
        if self._in_fence:
            rule_w = min(60, max(24, self._width() - 6))
            self._emit_lines(out_lines, [self._c("2", f"  ╰{'─' * rule_w}")])
            self._in_fence = False
            self._fence_char = ""
            self._fence_len = 0
            self._fence_lang = ""
        self._active_alert = None
        return ("\n".join(out_lines) + "\n") if out_lines else ""


def render_markdown_ansi(
    markdown_text: str,
    *,
    use_color: bool = True,
    width: int | None = None,
) -> str:
    """Renders a complete Markdown document into clean terminal ANSI text."""
    cols_fn = (lambda: width) if width is not None else None
    renderer = MarkdownStreamRenderer(use_color=use_color, cols_fn=cols_fn)
    committed, _ = renderer.feed(markdown_text)
    tail = renderer.flush()
    return (committed + tail).rstrip("\n")
