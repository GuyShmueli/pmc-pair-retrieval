"""Drawing of the retrieval figures: fonts, millimetre layout, figure types and export.

``pmc_pair_retrieval.figures`` computes the source data of the figures, writes it to CSV
files and calls the drawing functions of this module with the rows parsed from those
files, so that every drawn number is a number of the source data.

Layout. Every figure is built at its printed size, 180 mm wide (double column) and at
most 225 mm high, with a margin of ``MARGIN_MM`` on every side that no text, line or image
enters (checked for every figure). Axes and text are placed in millimetres from the top
left corner of the figure. All text is Arial at 7 pt in regular weight; only the panel
labels, lower-case letters of the same size, are bold. Colours come from the Okabe-Ito
palette, and every colour is paired with a marker shape, a line style, a position or a
text label.

Fonts. ``find_arial`` selects the Arial regular and bold files by content: the files of
the shipped figures (Arial version 2.82 of the Microsoft core fonts for the web) when
they are installed, whatever their file names, otherwise the newest regular and bold files
of one version, with a warning, because the figures then differ from the shipped ones.
``arial`` makes matplotlib use exactly these two files; text never falls back to another
font, and a missing glyph is an error.

Images. ``prepare_image`` fits an example image into its box with a 0.5 mm margin: the
whole image is shown and its aspect ratio is preserved. ``example_layout`` gives the boxes
of a figure: ``QUERY_BOX_MM`` for the query and its recorded positive and
``CANDIDATE_BOX_MM`` for the ranked candidates, or a taller top row when the query or the
positive is a portrait image that would otherwise be drawn less than ``MIN_TOP_SHORT_MM``
wide. The image is drawn at the size that fills the box, unless it would then have fewer
than ``MIN_PPI`` pixels per inch; it is then drawn smaller, at exactly ``MIN_PPI`` (images
are never upscaled). An image that would have more than ``MAX_PPI`` pixels per inch at its
drawn size is downscaled uniformly (Lanczos) to that resolution. A file that embeds an ICC
colour profile other than sRGB is first converted to sRGB (relative colorimetric intent with
black-point compensation), so that it appears as in a colour-managed viewer; no other change
is made.

Export. ``save`` writes a PDF (TrueType fonts embedded, text kept as text), an SVG (text
kept as text; images carry no rendering hint, so viewers smooth them when they scale them)
and a PNG preview at 300 dpi. No file contains a timestamp, so a regeneration with the same
inputs and software is identical byte for byte. The subset tag of each font embedded in a
PDF (the six letters before ``+`` in its name) is derived from the PostScript name and the
embedded characters, so that different subsets of the regular and bold faces never share a
tag.
"""
from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import math
import re
import string
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import matplotlib
import numpy as np
from matplotlib import font_manager, ft2font
from matplotlib.backends import backend_pdf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from PIL import Image, ImageCms

MM_PER_INCH = 25.4
POINTS_PER_INCH = 72.0
FIGURE_WIDTH_MM = 180.0
MAX_FIGURE_HEIGHT_MM = 225.0
MARGIN_MM = 2.0
PNG_DPI = 300

FONT_FAMILY = "Arial"
FONT_SIZE = 7.0
# sha256 of the Arial files of the shipped figures: version 2.82, Microsoft core fonts for the web.
ARIAL_SHA256 = {"regular": "35c0f3559d8db569e36c31095b8a60d441643d95f59139de40e23fada819b833",
                "bold": "4044aa6b5bebbc36980206b45b0aaaaa5681552a48bcadb41746d5d1d71fd7b4"}
# Distance from the baseline to the middle of the capital letters of 7 pt Arial (cap height 0.716 em), in mm.
CAP_MIDDLE_MM = 0.716 * FONT_SIZE / 2 / POINTS_PER_INCH * MM_PER_INCH
CAP_HEIGHT_MM = 2 * CAP_MIDDLE_MM
LINE_MM = 3.2  # baseline distance of consecutive text lines
# Tolerance of the margin check: matplotlib measures text with hinting, the PDF does not.
EXTENT_TOLERANCE_MM = 0.25

COLORS = {
    "pretrained": "#0072B2",    # Okabe-Ito blue
    "fine_tuned": "#D55E00",    # Okabe-Ito vermillion
    "better": "#009E73",        # Okabe-Ito bluish green
    "worse": "#CC79A7",         # Okabe-Ito reddish purple
    "positive": "#009E73",
    "text": "#000000",
    "difference": "#000000",
    "zero": "#808080",
    "overall": "#404040",
    "connector": "#999999",
    "grid": "#E3E3E3",
    "band": "#F0F0F0",
    "unannotated": "#AAAAAA",
}
CONFIGURATION_LABELS = {"image_to_image": "Image only, i2i", "image_text_to_image": "Image + text, (i+t)2i"}
MODEL_LABELS = {"pretrained": "Pretrained", "fine_tuned": "Fine-tuned"}
METRIC_LABELS = {"R@1": "Recall@1", "R@10": "Recall@10", "R@100": "Recall@100", "Hit@10": "Hit@10"}

# Example figures: ranks 1-5 of both models in two columns of landscape boxes, the query and
# its recorded positive in a row above them. Box sizes are (width, height) in mm.
RANKS_SHOWN = 5
FRAME_CLEARANCE_MM = 0.3  # keeps the outer half of the widest frame line out of the margin
CANDIDATE_COLUMN_LEFT_MM = 11.7
COLUMN_GAP_MM = 3.0
CANDIDATE_BOX_MM = ((FIGURE_WIDTH_MM - MARGIN_MM - FRAME_CLEARANCE_MM - CANDIDATE_COLUMN_LEFT_MM - COLUMN_GAP_MM) / 2,
                    27.1)
QUERY_BOX_MM = (52.0, 30.0)
# Taller top row (see example_layout): a portrait query or positive that would be drawn less than
# MIN_TOP_SHORT_MM wide in a QUERY_BOX_MM box gets a PORTRAIT_BOX_MM box instead.
MIN_TOP_SHORT_MM = 20.0
PORTRAIT_BOX_MM = (26.0, 52.0)
IMAGE_MARGIN_MM = 0.5
TAG_BAND_MM = 4.0  # space under a box for its label or tag
TAG_BASELINE_MM = 2.55  # baseline of a label or tag below the bottom edge of its box
MIN_LABEL_CLEARANCE_MM = 0.5  # ink of a label or tag to the outer edge of a frame
HEADER_MM = 6.0  # space between the top row (with its labels) and the ranked grid, holding the model headers
HEADER_ABOVE_GRID_MM = 1.0  # baseline of the model headers above the top edge of the grid
KEY_GAP_MM = 1.0  # between the tag band of the last ranked row and the key
MIN_PPI = 300
MAX_PPI = 310
# Smallest longer side (pixels) of an image that an example may show: 29 mm at 300 ppi.
MIN_LONG_PX = 343

STYLE = {
    "font.family": FONT_FAMILY,
    "font.size": FONT_SIZE,
    "font.weight": "normal",
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "text.color": COLORS["text"],
    "axes.labelcolor": COLORS["text"],
    "axes.edgecolor": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "xtick.major.pad": 1.5,
    "axes.labelpad": 2.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.unicode_minus": True,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.bbox": "standard",
    "savefig.pad_inches": 0.0,
    "lines.solid_capstyle": "butt",
    "lines.dash_capstyle": "butt",
    "image.interpolation": "none",
    "image.composite_image": False,
    "pdf.fonttype": 42,
    "pdf.compression": 6,
    "svg.fonttype": "none",
    "svg.image_inline": True,
    "svg.hashsalt": "pmc-pair-retrieval-figures",
    "text.usetex": False,
}


# --------------------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FontFile:
    """An installed font file: style (``regular`` or ``bold``), path, version string and sha256."""

    style: str
    path: str
    version: str
    sha256: str

    def describe(self) -> dict:
        """Manifest entry of the font; the directory, which depends on the system, is left out."""
        return {"family": FONT_FAMILY, "style": self.style, "file": Path(self.path).name, "version": self.version,
                "sha256": self.sha256}


def font_identity(entry: Mapping[str, str]) -> dict:
    """The content-defining fields of a manifest font entry; the file name depends on the installation."""
    return {key: entry.get(key) for key in ("family", "style", "version", "sha256")}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_number(version: str) -> float:
    match = re.search(r"\d+(?:\.\d+)?", version)
    return float(match.group(0)) if match else 0.0


def find_arial() -> tuple[dict[str, FontFile], bool]:
    """The Arial regular and bold files to draw with, and whether they are those of the shipped figures.

    Candidates are the fonts that ``matplotlib.font_manager.findSystemFonts`` lists (through
    fontconfig and the usual font directories) and the fonts of matplotlib's font cache.
    Files are recognised by their content, not by their path or name. Without the shipped
    files (sha256 in ``ARIAL_SHA256``), a regular and a bold file of the same version are
    used, the newest version first, then the first paths in sorted order. Without any Arial
    regular and bold files the figures cannot be drawn.
    """
    candidates: dict[str, list[FontFile]] = {"regular": [], "bold": []}
    seen = set()
    paths = set(font_manager.findSystemFonts(fontext="ttf"))
    paths.update(entry.fname for entry in font_manager.fontManager.ttflist)
    for path in sorted(paths):
        try:
            font = ft2font.FT2Font(path)
        except (OSError, RuntimeError):
            continue
        style = {"Regular": "regular", "Bold": "bold"}.get(font.style_name)
        if font.family_name != FONT_FAMILY or style is None:
            continue
        digest = _sha256(path)
        if (style, digest) in seen:
            continue
        seen.add((style, digest))
        names = font.get_sfnt()
        if (3, 1, 0x409, 5) in names:
            version = names[(3, 1, 0x409, 5)].decode("utf-16-be")
        else:
            version = names.get((1, 0, 0, 5), b"").decode("latin-1")
        candidates[style].append(FontFile(style, path, version.strip(), digest))
    published = {style: next((f for f in files if f.sha256 == ARIAL_SHA256[style]), None)
                 for style, files in candidates.items()}
    if all(published.values()):
        return dict(published), True
    if not candidates["regular"] or not candidates["bold"]:
        raise SystemExit("[figures] Arial regular and bold are required to draw the figures and were not found. "
                         "Install Arial version 2.82 (the Microsoft core fonts for the web, for example the Debian "
                         "or Ubuntu package ttf-mscorefonts-installer) and remove matplotlib's font cache if needed.")
    pairs = [(r, b) for r in candidates["regular"] for b in candidates["bold"] if r.version == b.version]
    if not pairs:
        pairs = [(candidates["regular"][0], candidates["bold"][0])]
    pairs.sort(key=lambda rb: (-_version_number(rb[0].version), rb[0].path, rb[1].path))
    regular, bold = pairs[0]
    warnings.warn(f"The Arial files of the published figures (version 2.82) are not installed; drawing with "
                  f"{Path(regular.path).name} ({regular.version}) and {Path(bold.path).name} ({bold.version}). "
                  "The figure files will not be identical to the published ones.", stacklevel=2)
    return {"regular": regular, "bold": bold}, False


@contextlib.contextmanager
def arial(fonts: Mapping[str, FontFile]) -> Iterator[None]:
    """Draw with exactly the given Arial files and the figure style; a missing glyph raises an error."""
    manager = font_manager.fontManager
    saved = manager.ttflist
    manager.ttflist = ([entry for entry in saved if entry.name != FONT_FAMILY]
                       + [font_manager.ttfFontProperty(ft2font.FT2Font(f.path)) for f in fonts.values()])
    manager._findfont_cached.cache_clear()
    try:
        with matplotlib.rc_context(STYLE), warnings.catch_warnings():
            warnings.filterwarnings("error", message=r"Glyph \d+")
            yield
    finally:
        manager.ttflist = saved
        manager._findfont_cached.cache_clear()


def text_width_mm(text: str, fonts: Mapping[str, FontFile], bold: bool = False) -> float:
    """Width of a 7 pt string in mm, measured like the PDF backend measures it (unhinted advances)."""
    font = font_manager.get_font(fonts["bold" if bold else "regular"].path)
    font.set_size(FONT_SIZE, POINTS_PER_INCH)
    font.set_text(text, 0.0, flags=ft2font.LOAD_NO_HINTING)
    return font.get_width_height()[0] / 64 / POINTS_PER_INCH * MM_PER_INCH


def ink_extent_mm(text: str, fonts: Mapping[str, FontFile]) -> tuple[float, float]:
    """Height of the ink above the baseline and its depth below the baseline (mm) of a 7 pt regular string."""
    font = font_manager.get_font(fonts["regular"].path)
    font.set_size(FONT_SIZE, POINTS_PER_INCH)
    font.set_text(text, 0.0, flags=ft2font.LOAD_NO_HINTING)
    height, descent = font.get_width_height()[1], font.get_descent()
    return (height - descent) / 64 / POINTS_PER_INCH * MM_PER_INCH, descent / 64 / POINTS_PER_INCH * MM_PER_INCH


def wrap_text(text: str, width_mm: float, fonts: Mapping[str, FontFile]) -> list[str]:
    """Greedy word wrap of a 7 pt string into lines of at most ``width_mm``; a longer word is an error."""
    lines: list[str] = []
    for word in text.split():
        if text_width_mm(word, fonts) > width_mm:
            raise ValueError(f"the word {word!r} is wider than {width_mm:.1f} mm")
        candidate = f"{lines[-1]} {word}" if lines else word
        if lines and text_width_mm(candidate, fonts) <= width_mm:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines


def signed(value: float, decimals: int = 2) -> str:
    """A number with an explicit sign; negative numbers take the minus sign U+2212."""
    return f"{value:+.{decimals}f}".replace("-", "−")


def nice_limit(value: float, step: float) -> float:
    """``value`` rounded away from zero to a multiple of ``step``."""
    return math.copysign(math.ceil(abs(value) / step - 1e-9) * step, value) if value else 0.0


# --------------------------------------------------------------------------------------
# Millimetre layout
# --------------------------------------------------------------------------------------
class Canvas:
    """A matplotlib figure with coordinates in millimetres from its top left corner."""

    def __init__(self, height_mm: float, fonts: Mapping[str, FontFile], width_mm: float = FIGURE_WIDTH_MM):
        if height_mm > MAX_FIGURE_HEIGHT_MM:
            raise ValueError(f"figure height {height_mm:.1f} mm exceeds {MAX_FIGURE_HEIGHT_MM:.0f} mm")
        self.width, self.height, self.fonts = width_mm, height_mm, fonts
        self.figure = Figure(figsize=(width_mm / MM_PER_INCH, height_mm / MM_PER_INCH))

    def x(self, mm: float) -> float:
        """Figure fraction of a horizontal position in mm."""
        return mm / self.width

    def y(self, mm: float) -> float:
        """Figure fraction of a vertical position in mm from the top."""
        return 1 - mm / self.height

    def axes(self, left: float, top: float, width: float, height: float):
        """Axes occupying the given rectangle (mm)."""
        return self.figure.add_axes((self.x(left), self.y(top + height), width / self.width, height / self.height))

    def text(self, x: float, baseline: float, text: str, **kwargs):
        """Text whose baseline is at ``baseline`` mm from the top."""
        kwargs.setdefault("va", "baseline")
        return self.figure.text(self.x(x), self.y(baseline), text, **kwargs)

    def row_text(self, x: float, middle: float, text: str, **kwargs):
        """Text whose capital letters are vertically centred at ``middle`` mm."""
        return self.text(x, middle + CAP_MIDDLE_MM, text, **kwargs)

    def panel_label(self, x: float, baseline: float, letter: str, title: str = "",
                    title_x: float | None = None) -> None:
        """Bold lower-case panel letter, followed by an optional title in regular weight."""
        self.text(x, baseline, letter, fontweight="bold")
        if title:
            self.text(x + 3.0 if title_x is None else title_x, baseline, title)

    def line(self, x0: float, y0: float, x1: float, y1: float, **kwargs) -> None:
        """A straight line between two points (mm)."""
        self.figure.add_artist(Line2D([self.x(x0), self.x(x1)], [self.y(y0), self.y(y1)],
                                      transform=self.figure.transFigure, **kwargs))

    def marker(self, x: float, y: float, **kwargs) -> None:
        """A single marker centred at a point (mm)."""
        self.figure.add_artist(Line2D([self.x(x)], [self.y(y)], transform=self.figure.transFigure, linestyle="",
                                      **kwargs))

    def frame(self, left: float, top: float, width: float, height: float, **kwargs) -> None:
        """An unfilled rectangle (mm)."""
        self.figure.add_artist(Rectangle((self.x(left), self.y(top + height)), width / self.width,
                                         height / self.height, transform=self.figure.transFigure, fill=False,
                                         **kwargs))

    def band(self, left: float, top: float, width: float, height: float, color: str) -> None:
        """A filled rectangle (mm) behind the axes, which must have transparent backgrounds."""
        self.figure.add_artist(Rectangle((self.x(left), self.y(top + height)), width / self.width,
                                         height / self.height, transform=self.figure.transFigure, facecolor=color,
                                         linewidth=0, zorder=-1))

    def width_of(self, text: str, bold: bool = False) -> float:
        """Width of a 7 pt string in mm."""
        return text_width_mm(text, self.fonts, bold)

    def check_extents(self, name: str) -> None:
        """Raise if any text, axes, line or rectangle reaches into the margin of ``MARGIN_MM``."""
        FigureCanvasAgg(self.figure)
        self.figure.draw_without_rendering()
        renderer = self.figure.canvas.get_renderer()
        mm_per_px = MM_PER_INCH / self.figure.dpi
        lo_x, hi_x = MARGIN_MM - EXTENT_TOLERANCE_MM, self.width - MARGIN_MM + EXTENT_TOLERANCE_MM
        lo_y, hi_y = MARGIN_MM - EXTENT_TOLERANCE_MM, self.height - MARGIN_MM + EXTENT_TOLERANCE_MM
        artists: list = list(self.figure.texts) + list(self.figure.artists) + list(self.figure.lines)
        for ax in self.figure.axes:
            artists.append(ax)
            if ax.axison:
                artists += [t for t in ax.get_xticklabels() + [ax.xaxis.label] if t.get_text()]
        for artist in artists:
            if not artist.get_visible() or (isinstance(artist, Text) and not artist.get_text()):
                continue
            box = artist.get_window_extent(renderer)
            if box.width == 0 and box.height == 0:
                continue
            left, right = box.x0 * mm_per_px, box.x1 * mm_per_px
            top, bottom = self.height - box.y1 * mm_per_px, self.height - box.y0 * mm_per_px
            if left < lo_x or right > hi_x or top < lo_y or bottom > hi_y:
                label = artist.get_text() if isinstance(artist, Text) else type(artist).__name__
                raise ValueError(f"{name}: {label!r} reaches into the {MARGIN_MM:g} mm margin (left {left:.2f}, "
                                 f"right {right:.2f}, top {top:.2f}, bottom {bottom:.2f} mm)")


def _data_to_mm(ax_left: float, ax_width: float, limits: Sequence[float], value: float) -> float:
    lo, hi = limits
    return ax_left + (value - lo) / (hi - lo) * ax_width


def _style_value_axes(ax, limits: Sequence[float], ticks: Sequence[float], n_rows: float, label: str,
                      tick_labels: Sequence[str] | None = None, grid: bool = True) -> None:
    """Horizontal value axis with rows 0..n_rows-1 from the top; no vertical axis."""
    if tick_labels is None:
        tick_labels = [f"{t:g}".replace("-", "−") for t in ticks]
    ax.set_xlim(*limits)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(list(ticks), labels=list(tick_labels))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(label)
    if grid:
        ax.xaxis.grid(True, color=COLORS["grid"], linewidth=0.4)


# --------------------------------------------------------------------------------------
# Figure: overall performance
# --------------------------------------------------------------------------------------
def draw_performance(levels: Mapping[tuple[str, str], Mapping[str, float]],
                     differences: Mapping[tuple[str, str], tuple[float, float, float]],
                     changes: Mapping[tuple[str, str], Mapping[str, float]],
                     configurations: Sequence[str], level_metrics: Sequence[str],
                     fonts: Mapping[str, FontFile]) -> Canvas:
    """Retrieval performance: levels (a), paired differences with intervals (b) and per-query change (c).

    ``levels[(configuration, model)][metric]`` are means in percent;
    ``differences[(configuration, metric)]`` is ``(difference, ci_low, ci_high)`` in
    percentage points; ``changes[(configuration, metric)]`` holds the shares (percent) of
    queries that are ``better``, ``worse`` and ``unchanged`` with the fine-tuned model.
    The difference column of panel b is set as aligned numeric sub-columns, so that the
    decimal points of the estimates and of both interval bounds line up across rows.
    """
    pitch = 3.5
    label_x, metric_right = MARGIN_MM, 31.0
    a_left, a_width = 33.0, 42.0
    b_left, b_width = 85.0, 63.0
    text_x = 151.0
    c_left, c_width = 33.0, 114.0
    unchanged_x = c_left + c_width + 4.0
    header = MARGIN_MM + 2.0

    rows_ab = []
    for conf in configurations:
        rows_ab.append((conf, None))
        rows_ab += [(conf, m) for m in level_metrics]
    change_metrics = ("R@10", "Hit@10")
    rows_c = []
    for conf in configurations:
        rows_c.append((conf, None))
        rows_c += [(conf, m) for m in change_metrics]

    top_ab = header + 3.0
    height_ab = len(rows_ab) * pitch
    header_c = top_ab + height_ab + 11.0
    top_c = header_c + 2.0
    height_c = len(rows_c) * pitch
    total = top_c + height_c + 9.5
    canvas = Canvas(total, fonts)

    # Header of panels a and b, with the key of panel a.
    canvas.panel_label(label_x, header, "a")
    key_x = a_left
    key = (("pretrained", "o", 3.6, "white"), ("fine_tuned", "s", 3.2, COLORS["fine_tuned"]))
    for model, marker, size, face in key:
        canvas.marker(key_x + 1.0, header - CAP_MIDDLE_MM, marker=marker, markersize=size, markerfacecolor=face,
                      markeredgecolor=COLORS[model], markeredgewidth=0.8)
        canvas.text(key_x + 2.6, header, MODEL_LABELS[model])
        key_x += 2.6 + canvas.width_of(MODEL_LABELS[model]) + 4.0
    canvas.panel_label(b_left - 7.0, header, "b", "Fine-tuned − pretrained")
    canvas.text(text_x, header, "Difference [95% CI]")

    # Aligned sub-columns of the difference text: the estimate, "[lower," and "upper]" are right-aligned, so
    # that the decimal points line up across rows (all values have two decimals; Arial digits share one width).
    shown = [differences[(conf, m)] for conf, m in rows_ab if m is not None]
    est_right = text_x + max(canvas.width_of(signed(d)) for d, _, _ in shown)
    lo_right = est_right + canvas.width_of(" ") + max(canvas.width_of(f"[{signed(lo)},") for _, lo, _ in shown)
    hi_right = lo_right + canvas.width_of(" ") + max(canvas.width_of(f"{signed(hi)}]") for _, _, hi in shown)
    if hi_right > FIGURE_WIDTH_MM - MARGIN_MM:
        raise ValueError("the difference column of panel b does not fit the figure width")

    ax_a = canvas.axes(a_left, top_ab, a_width, height_ab)
    ax_b = canvas.axes(b_left, top_ab, b_width, height_ab)
    level_max = max(v for vals in levels.values() for m, v in vals.items() if m in level_metrics)
    a_limits = (0.0, nice_limit(level_max, 20.0))
    _style_value_axes(ax_a, a_limits, np.arange(0, a_limits[1] + 1e-9, 20), len(rows_ab), "Mean over test queries (%)")
    diff_max = max(hi for (conf, m), (_, _, hi) in differences.items() if m in level_metrics)
    diff_min = min(lo for (conf, m), (_, lo, _) in differences.items() if m in level_metrics)
    b_limits = (min(0.0, nice_limit(diff_min, 5.0)), nice_limit(diff_max, 5.0))
    _style_value_axes(ax_b, b_limits, np.arange(b_limits[0], b_limits[1] + 1e-9, 5), len(rows_ab),
                      "Difference (percentage points)")
    ax_b.axvline(0.0, color=COLORS["zero"], linewidth=0.5, zorder=1)

    for i, (conf, metric) in enumerate(rows_ab):
        middle = top_ab + (i + 0.5) * pitch
        if metric is None:
            canvas.row_text(label_x, middle, CONFIGURATION_LABELS[conf])
            continue
        canvas.row_text(metric_right, middle, METRIC_LABELS[metric], ha="right")
        pt, ft = levels[(conf, "pretrained")][metric], levels[(conf, "fine_tuned")][metric]
        ax_a.plot([pt, ft], [i, i], color=COLORS["connector"], linewidth=0.75, zorder=2)
        ax_a.plot([ft], [i], marker="s", markersize=3.2, color=COLORS["fine_tuned"], linestyle="", zorder=3)
        ax_a.plot([pt], [i], marker="o", markersize=3.6, markerfacecolor="white", markeredgecolor=COLORS["pretrained"],
                  markeredgewidth=0.8, linestyle="", zorder=4)
        d, lo, hi = differences[(conf, metric)]
        ax_b.plot([lo, hi], [i, i], color=COLORS["difference"], linewidth=0.75, zorder=3)
        for end in (lo, hi):
            ax_b.plot([end, end], [i - 0.18, i + 0.18], color=COLORS["difference"], linewidth=0.75, zorder=3)
        ax_b.plot([d], [i], marker="o", markersize=3.2, color=COLORS["difference"], linestyle="", zorder=4)
        canvas.row_text(est_right, middle, signed(d), ha="right")
        canvas.row_text(lo_right, middle, f"[{signed(lo)},", ha="right")
        canvas.row_text(hi_right, middle, f"{signed(hi)}]", ha="right")

    # Panel c: per-query change.
    canvas.panel_label(label_x, header_c, "c", "Per-query change with fine-tuning")
    ax_c = canvas.axes(c_left, top_c, c_width, height_c)
    worse_max = max(v["worse"] for v in changes.values())
    better_max = max(v["better"] for v in changes.values())
    c_limits = (-nice_limit(worse_max, 10.0), nice_limit(better_max, 10.0))
    ticks = np.arange(c_limits[0], c_limits[1] + 1e-9, 10)
    _style_value_axes(ax_c, c_limits, ticks, len(rows_c), "Share of test queries (%)",
                      tick_labels=[f"{abs(t):g}" for t in ticks], grid=False)
    ax_c.axvline(0.0, color=COLORS["zero"], linewidth=0.5, zorder=1)
    zero_mm = _data_to_mm(c_left, c_width, c_limits, 0.0)
    canvas.text(zero_mm - 1.2, header_c, "Worse", ha="right")
    canvas.text(zero_mm + 1.2, header_c, "Better", ha="left")
    canvas.text(unchanged_x, header_c, "Unchanged")
    for i, (conf, metric) in enumerate(rows_c):
        middle = top_c + (i + 0.5) * pitch
        if metric is None:
            canvas.row_text(label_x, middle, CONFIGURATION_LABELS[conf])
            continue
        canvas.row_text(metric_right, middle, METRIC_LABELS[metric], ha="right")
        share = changes[(conf, metric)]
        ax_c.barh([i], [share["better"]], left=0.0, height=0.62, color=COLORS["better"], linewidth=0, zorder=2)
        ax_c.barh([i], [-share["worse"]], left=0.0, height=0.62, color=COLORS["worse"], linewidth=0, zorder=2)
        canvas.row_text(_data_to_mm(c_left, c_width, c_limits, share["better"]) + 0.9, middle,
                        f"{share['better']:.1f}%", ha="left")
        canvas.row_text(_data_to_mm(c_left, c_width, c_limits, -share["worse"]) - 0.9, middle,
                        f"{share['worse']:.1f}%", ha="right")
        canvas.row_text(unchanged_x, middle, f"{share['unchanged']:.1f}%")
    canvas.check_extents("retrieval_performance")
    return canvas


# --------------------------------------------------------------------------------------
# Figure: subgroups
# --------------------------------------------------------------------------------------
INTERVAL_STYLE = {
    "above": (COLORS["better"], COLORS["better"], "95% CI above zero"),
    "includes": ("#6E6E6E", "white", "95% CI includes zero"),
    "below": (COLORS["worse"], COLORS["worse"], "95% CI below zero"),
}


def interval_class(lo: float, hi: float) -> str:
    """Classify an interval relative to zero, including either endpoint at zero."""
    return "above" if lo > 0 else ("below" if hi < 0 else "includes")


def draw_subgroups(rows: Sequence[Mapping[str, object]], overall: Mapping[str, float],
                   configurations: Sequence[str], families: Sequence[tuple[str, str]],
                   fonts: Mapping[str, FontFile]) -> Canvas:
    """Paired Hit@10 difference with 95% interval per label category, one panel per configuration.

    ``rows`` are the drawn rows of the subgroup table, each with ``configuration``,
    ``family``, ``category``, ``label``, ``n_queries``, ``n_article_groups``,
    ``pretrained``, ``fine_tuned`` (mean Hit@10 in percent), ``difference``, ``ci_low``
    and ``ci_high`` (percentage points), in display order. ``overall[configuration]`` is the
    difference over all queries; ``families`` lists ``(family, heading)`` in display order.
    Alternating light bands behind every second category row of a family run across the
    label columns and both panels, so that each row of panel b can be traced to its label.
    Filled green circles mark intervals above zero and open grey circles include zero.
    """
    pitch = 2.9
    heading_x, category_x = MARGIN_MM, MARGIN_MM + 1.6
    n_right, groups_right = 45.0, 55.5
    panel_width, number_width, arrow_gap = 39.6, 5.6, 4.2
    level_width = 2 * number_width + arrow_gap
    panel_gap = 6.5
    lefts = [groups_right + 3.0]
    lefts.append(lefts[0] + panel_width + 1.5 + level_width + panel_gap)
    right_edge = lefts[1] + panel_width + 1.5 + level_width
    label_baseline, header_baseline, top = MARGIN_MM + 2.0, MARGIN_MM + 5.6, MARGIN_MM + 8.0
    if right_edge > FIGURE_WIDTH_MM - MARGIN_MM:
        raise ValueError("the subgroup panels do not fit the figure width")

    by_key = {(r["configuration"], r["family"], r["category"]): r for r in rows}
    layout: list[tuple[str, str, object]] = []  # (kind, family, category) with kind heading / category / gap
    for family, heading in families:
        categories = [r["category"] for r in rows if r["family"] == family and r["configuration"] == configurations[0]]
        if not categories:
            continue
        if layout:
            layout.append(("gap", family, None))
        layout.append(("heading", family, heading))
        layout += [("category", family, c) for c in categories]
    units = [0.55 if kind == "gap" else 1.0 for kind, _, _ in layout]
    centres = np.cumsum(units) - np.array(units) / 2
    n_units = float(sum(units))

    height = n_units * pitch
    total = top + height + 13.0
    canvas = Canvas(total, fonts)

    # Bands behind every second category row, restarted at each family heading.
    band_index = 0
    for (kind, family, category), centre in zip(layout, centres):
        if kind == "heading":
            band_index = 0
        elif kind == "category":
            if band_index % 2 == 1:
                canvas.band(MARGIN_MM, top + (centre - 0.5) * pitch, right_edge - MARGIN_MM, pitch, COLORS["band"])
            band_index += 1

    low = min(float(r["ci_low"]) for r in rows)
    high = max(float(r["ci_high"]) for r in rows)
    limits = (min(0.0, nice_limit(low, 5.0)), nice_limit(high, 5.0))
    step = 10.0 if limits[1] - limits[0] > 30 else 5.0
    ticks = np.arange(math.ceil(limits[0] / step) * step, limits[1] + 1e-9, step)

    classes_present = set()
    canvas.text(n_right, header_baseline, "n", ha="right")
    canvas.text(groups_right, header_baseline, "Groups", ha="right")
    for letter, conf, left in zip("ab", configurations, lefts):
        letter_x = heading_x if letter == "a" else left - 4.0
        canvas.panel_label(letter_x, label_baseline, letter, CONFIGURATION_LABELS[conf], title_x=left - 1.0)
        level_left = left + panel_width + 1.5
        canvas.text(level_left + level_width, header_baseline, "Hit@10 (%)", ha="right")
        ax = canvas.axes(left, top, panel_width, height)
        ax.patch.set_visible(False)
        ax.set_xlim(*limits)
        ax.set_ylim(n_units, 0.0)
        ax.set_xticks(list(ticks), labels=[f"{t:g}".replace("-", "−") for t in ticks])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.xaxis.grid(True, color=COLORS["grid"], linewidth=0.4)
        ax.set_xlabel("Hit@10 difference (percentage points)")
        ax.axvline(0.0, color=COLORS["zero"], linewidth=0.5, zorder=1)
        ax.axvline(overall[conf], color=COLORS["overall"], linewidth=0.6, linestyle=(0, (3.0, 2.0)), zorder=1)
        for (kind, family, category), centre in zip(layout, centres):
            if kind != "category":
                continue
            r = by_key[(conf, family, category)]
            d, lo, hi = float(r["difference"]), float(r["ci_low"]), float(r["ci_high"])
            cls = interval_class(lo, hi)
            classes_present.add(cls)
            edge, face, _ = INTERVAL_STYLE[cls]
            ax.plot([lo, hi], [centre, centre], color=edge, linewidth=0.75, zorder=3)
            ax.plot([d], [centre], marker="o", markersize=2.8, markerfacecolor=face, markeredgecolor=edge,
                    markeredgewidth=0.7, linestyle="", zorder=4)
            middle = top + centre * pitch
            first, second = f"{float(r['pretrained']):.1f}", f"{float(r['fine_tuned']):.1f}"
            canvas.row_text(level_left + number_width, middle, first, ha="right")
            canvas.row_text(level_left + number_width + arrow_gap / 2, middle, "→", ha="center")
            canvas.row_text(level_left + level_width, middle, second, ha="right")

    for (kind, family, category), centre in zip(layout, centres):
        middle = top + centre * pitch
        if kind == "heading":
            canvas.row_text(heading_x, middle, str(category))
        elif kind == "category":
            r = by_key[(configurations[0], family, category)]
            label, count, groups = str(r["label"]), f"{int(r['n_queries']):,}", f"{int(r['n_article_groups']):,}"
            if category_x + canvas.width_of(label) + 1.5 > n_right - canvas.width_of(count):
                raise ValueError(f"the subgroup label {label!r} does not fit next to its count")
            if n_right + 1.5 > groups_right - canvas.width_of(groups):
                raise ValueError(f"the number of article groups of {label!r} does not fit its column")
            canvas.row_text(category_x, middle, label)
            canvas.row_text(n_right, middle, count, ha="right")
            canvas.row_text(groups_right, middle, groups, ha="right")

    # Key below the axes.
    key_y = total - MARGIN_MM - 0.6
    x = lefts[0]
    entries = [INTERVAL_STYLE[cls] for cls in ("above", "includes", "below") if cls in classes_present]
    for edge, face, label in entries:
        canvas.line(x, key_y - CAP_MIDDLE_MM, x + 3.0, key_y - CAP_MIDDLE_MM, color=edge, linewidth=0.75)
        canvas.marker(x + 1.5, key_y - CAP_MIDDLE_MM, marker="o", markersize=2.8, markerfacecolor=face,
                      markeredgecolor=edge, markeredgewidth=0.7)
        canvas.text(x + 4.2, key_y, label)
        x += 4.2 + canvas.width_of(label) + 6.0
    canvas.line(x, key_y - CAP_MIDDLE_MM, x + 5.0, key_y - CAP_MIDDLE_MM, color=COLORS["overall"], linewidth=0.6,
                linestyle=(0, (3.0, 2.0)))
    canvas.text(x + 6.2, key_y, "Difference over all test queries")
    canvas.check_extents("retrieval_subgroups_hit10")
    return canvas


# --------------------------------------------------------------------------------------
# Figures: retrieval examples
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PreparedImage:
    """An example image ready to be drawn: sRGB pixels, sizes, the drawn size of one pixel (mm) and its colour profile.

    ``icc_profile`` is the description of the ICC profile embedded in the file (empty without one), and
    ``converted`` says whether the pixels were converted from that profile to sRGB.
    """

    pixels: np.ndarray
    source_size: tuple[int, int]
    size: tuple[int, int]
    mm_per_px: float
    icc_profile: str = ""
    converted: bool = False

    @property
    def resampled(self) -> bool:
        """Whether the image was downscaled."""
        return self.size != self.source_size

    @property
    def placed_mm(self) -> tuple[float, float]:
        """Width and height of the placed image in mm."""
        return self.size[0] * self.mm_per_px, self.size[1] * self.mm_per_px

    @property
    def ppi(self) -> float:
        """Pixels per inch of the placed image (equal in both directions)."""
        return MM_PER_INCH / self.mm_per_px


def fit_image(source_size: tuple[int, int], box: tuple[float, float]) -> tuple[tuple[int, int], float]:
    """Embedded size (pixels) and drawn size of one pixel (mm) of an image of ``source_size`` pixels in a box (mm).

    The image fills the box less its margin unless it would then have fewer than
    ``MIN_PPI`` pixels per inch, in which case it is drawn at exactly ``MIN_PPI``. If it
    would have more than ``MAX_PPI``, it is downscaled uniformly so that the side that
    limits its size has ``MAX_PPI`` or slightly fewer pixels per inch.
    """
    width, height = source_size
    avail = (box[0] - 2 * IMAGE_MARGIN_MM, box[1] - 2 * IMAGE_MARGIN_MM)
    axis = 0 if avail[0] / width <= avail[1] / height else 1  # the side that limits the size
    fit = avail[axis] / source_size[axis]
    if fit >= MM_PER_INCH / MIN_PPI:
        return source_size, MM_PER_INCH / MIN_PPI
    target = math.floor(MAX_PPI * avail[axis] / MM_PER_INCH)
    if source_size[axis] <= target:
        return source_size, fit
    scale = target / source_size[axis]
    size = [0, 0]
    size[axis] = target
    size[1 - axis] = max(1, round(source_size[1 - axis] * scale))
    return (size[0], size[1]), avail[axis] / target


def placed_size_mm(source_size: tuple[int, int], box: tuple[float, float]) -> tuple[float, float]:
    """Drawn width and height (mm) of an image of ``source_size`` pixels in a box (mm)."""
    size, mm_per_px = fit_image(source_size, box)
    return size[0] * mm_per_px, size[1] * mm_per_px


SRGB = ImageCms.createProfile("sRGB")


def srgb_pixels(im: Image.Image) -> tuple[Image.Image, str, bool]:
    """The image in RGB, the description of its embedded ICC profile and whether it was converted to sRGB.

    A file without a profile, or with an sRGB profile, keeps its stored values (greyscale is
    expanded to RGB). Another profile is converted to sRGB with the relative colorimetric
    intent and black-point compensation, as a colour-managed viewer displays the file.
    """
    icc = im.info.get("icc_profile")
    if not icc:
        return im.convert("RGB"), "", False
    profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
    description = profile.profile.profile_description.strip()
    if "srgb" in description.lower():
        return im.convert("RGB"), description, False
    transform = ImageCms.buildTransform(profile, SRGB, im.mode, "RGB",
                                        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                                        flags=ImageCms.Flags.BLACKPOINTCOMPENSATION)
    return ImageCms.applyTransform(im, transform), description, True


def prepare_image(path: str | Path, box: tuple[float, float]) -> PreparedImage:
    """Load an image in sRGB and size it for a box of ``box`` = (width, height) mm.

    The colours are converted to sRGB when the file embeds another ICC profile
    (``srgb_pixels``); the size follows ``fit_image``, and a downscaled image is resampled
    uniformly (Lanczos). Images with fewer than ``MIN_LONG_PX`` pixels on the longer side
    are refused; the example selection excludes them.
    """
    with Image.open(path) as im:
        source_size = im.size
        rgb, profile, converted = srgb_pixels(im)
    if max(source_size) < MIN_LONG_PX:
        raise ValueError(f"{path}: {source_size[0]} x {source_size[1]} px is smaller than {MIN_LONG_PX} px on the "
                         "longer side")
    size, mm_per_px = fit_image(source_size, box)
    if size != source_size:
        rgb = rgb.resize(size, Image.Resampling.LANCZOS)
    return PreparedImage(np.asarray(rgb), source_size, size, mm_per_px, profile, converted)


@dataclass(frozen=True)
class ExampleLayout:
    """Boxes (width, height in mm) of the query, its recorded positive and the ranked candidates of one example."""

    query_box: tuple[float, float]
    positive_box: tuple[float, float]
    candidate_box: tuple[float, float]

    def box(self, role: str) -> tuple[float, float]:
        """The box of an image in the role ``query``, ``positive`` or ``candidate``."""
        return {"query": self.query_box, "positive": self.positive_box, "candidate": self.candidate_box}[role]


def example_layout(query_size: tuple[int, int], positive_size: tuple[int, int]) -> ExampleLayout:
    """Boxes of an example figure, from the pixel sizes (width, height) of the query and its recorded positive.

    Both images have ``QUERY_BOX_MM`` boxes unless one of them is taller than wide and
    would be drawn less than ``MIN_TOP_SHORT_MM`` wide in such a box. The top row is then
    ``PORTRAIT_BOX_MM[1]`` high: that image gets a ``PORTRAIT_BOX_MM`` box and the other
    image a box of the same height and the rest of the width of the two standard boxes
    (two such images share it equally). The candidate boxes are lowered by a fifth of the
    extra height, so that the figure keeps its height.
    """
    def narrow(size: tuple[int, int]) -> bool:
        return size[1] > size[0] and placed_size_mm(size, QUERY_BOX_MM)[0] < MIN_TOP_SHORT_MM

    narrow_query, narrow_positive = narrow(query_size), narrow(positive_size)
    if not (narrow_query or narrow_positive):
        return ExampleLayout(QUERY_BOX_MM, QUERY_BOX_MM, CANDIDATE_BOX_MM)
    width, height = 2 * QUERY_BOX_MM[0], PORTRAIT_BOX_MM[1]
    if narrow_query and narrow_positive:
        widths = (width / 2, width / 2)
    elif narrow_query:
        widths = (PORTRAIT_BOX_MM[0], width - PORTRAIT_BOX_MM[0])
    else:
        widths = (width - PORTRAIT_BOX_MM[0], PORTRAIT_BOX_MM[0])
    lowered = round(CANDIDATE_BOX_MM[1] - (height - QUERY_BOX_MM[1]) / RANKS_SHOWN, 6)
    return ExampleLayout((widths[0], height), (widths[1], height), (CANDIDATE_BOX_MM[0], lowered))


FRAME_STYLES = {
    "query": {"edgecolor": COLORS["text"], "linewidth": 0.75, "linestyle": "-"},
    "positive": {"edgecolor": COLORS["positive"], "linewidth": 1.5, "linestyle": "-"},
    "same_article": {"edgecolor": COLORS["text"], "linewidth": 1.0, "linestyle": (0, (2.5, 1.5))},
    "unannotated": {"edgecolor": COLORS["unannotated"], "linewidth": 0.5, "linestyle": "-"},
}
CANDIDATE_TAGS = {"positive": "Recorded positive", "same_article": "Same article, not annotated", "unannotated": ""}
KEY_LABELS = {"positive": "Recorded positive (annotated pair with the query)",
              "same_article": "Candidate from the query’s article, not annotated",
              "unannotated": "Unannotated candidate (not a verified negative)",
              "other_unannotated": "Other unannotated candidate (not a verified negative)"}


def _draw_image_box(canvas: Canvas, left: float, top: float, box: tuple[float, float], image: PreparedImage,
                    frame: str) -> None:
    """One image centred in a box with a frame of the given style."""
    width, height = box
    ax = canvas.axes(left, top, width, height)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_axis_off()
    w, h = image.placed_mm
    if w > width - IMAGE_MARGIN_MM or h > height - IMAGE_MARGIN_MM:
        raise ValueError("an example image does not fit its box")
    x0, y0 = (width - w) / 2, (height - h) / 2
    ax.imshow(image.pixels, extent=(x0, x0 + w, y0, y0 + h), interpolation="none", aspect="auto", zorder=1)
    canvas.frame(left, top, width, height, joinstyle="miter", zorder=5, **FRAME_STYLES[frame])


def example_text_lines(panel: Mapping[str, object], width_mm: float,
                       fonts: Mapping[str, FontFile]) -> list[tuple[str, float]]:
    """Text block of an example: ``(line, extra space after it in mm)`` for each line."""
    conf = str(panel["configuration"])
    lead = ("Query text, the pair description:" if conf == "image_text_to_image"
            else "Pair description, not used by the model:")
    rank_line = (f"Rank of the recorded positive among {int(panel['ranked_gallery_size']):,} gallery images: "
                 f"pretrained {int(panel['pretrained_rank']):,}, fine-tuned {int(panel['fine_tuned_rank']):,}")
    lines = [(CONFIGURATION_LABELS[conf], 0.8), (f"Query {panel['query_id']}", 0.0),
             (f"Recorded positive {panel['positive_id']}", 0.8)]
    lines += [(line, 0.0) for line in wrap_text(lead, width_mm, fonts)]
    description = wrap_text(f"“{panel['description']}”", width_mm, fonts)
    lines += [(line, 0.8 if k == len(description) - 1 else 0.0) for k, line in enumerate(description)]
    lines += [(line, 0.0) for line in wrap_text(rank_line, width_mm, fonts)]
    return lines


TOP_ROW_LABELS = (("query", "Query"), ("positive", "Recorded positive"))


def _frame_width_mm(kind: str) -> float:
    return FRAME_STYLES[kind]["linewidth"] / POINTS_PER_INCH * MM_PER_INCH


def check_label_spacing(fonts: Mapping[str, FontFile]) -> dict[str, float]:
    """Check that every label and tag of an example figure reads as part of its own box; return the gaps (mm).

    Ink extents come from the Arial file at 7 pt. A label or tag under a box (``Query``,
    ``Recorded positive`` and the candidate tags) must be at least ``MIN_LABEL_CLEARANCE_MM``
    from the outer edge of its own frame and from whatever lies below it, and closer to its
    own frame: the next row's frame (at most the widest frame) for a candidate tag, the model
    headers for the labels of the top row. The model headers must be at least as close to
    the grid as that. The blank band between the top-row labels and the model headers must be
    at least the band between two ranked rows (thinnest frames) and at least twice each
    label's distance to its own frame, so that the top row and the grid read as two groups.
    Raises ``ValueError`` otherwise.
    """
    widest = max(_frame_width_mm(kind) for kind in FRAME_STYLES)
    gaps: dict[str, float] = {}
    for kind, tag in CANDIDATE_TAGS.items():
        if not tag:
            continue
        ascent, descent = ink_extent_mm(tag, fonts)
        own = TAG_BASELINE_MM - ascent - _frame_width_mm(kind) / 2
        below = TAG_BAND_MM - TAG_BASELINE_MM - descent - widest / 2
        if min(own, below) < MIN_LABEL_CLEARANCE_MM or own >= below:
            raise ValueError(f"the tag {tag!r} is {own:.2f} mm from its own frame and {below:.2f} mm from the next "
                             f"row; both must be at least {MIN_LABEL_CLEARANCE_MM} mm, the first the smaller")
        gaps[f"{tag}: own frame"], gaps[f"{tag}: next row"] = own, below
    header_ascent = max(ink_extent_mm(MODEL_LABELS[m], fonts)[0] for m in MODEL_LABELS)
    header_descent = max(ink_extent_mm(MODEL_LABELS[m], fonts)[1] for m in MODEL_LABELS)
    label_descent = max(ink_extent_mm(label, fonts)[1] for _, label in TOP_ROW_LABELS)
    band = TAG_BAND_MM + HEADER_MM - HEADER_ABOVE_GRID_MM - header_ascent - TAG_BASELINE_MM - label_descent
    header_gap = HEADER_ABOVE_GRID_MM - header_descent - widest / 2
    between_rows = TAG_BAND_MM - min(_frame_width_mm(kind) for kind in FRAME_STYLES)
    for kind, label in TOP_ROW_LABELS:
        own = TAG_BASELINE_MM - ink_extent_mm(label, fonts)[0] - _frame_width_mm(kind) / 2
        if own < MIN_LABEL_CLEARANCE_MM or band < 2 * own:
            raise ValueError(f"the label {label!r} is {own:.2f} mm from its frame and {band:.2f} mm from the model "
                             f"headers; at least {MIN_LABEL_CLEARANCE_MM} mm and twice that distance are required")
        gaps[f"{label}: own frame"] = own
    if band < between_rows or not MIN_LABEL_CLEARANCE_MM <= header_gap < band:
        raise ValueError(f"the top-row labels are {band:.2f} mm from the model headers (rows are {between_rows:.2f} "
                         f"mm apart) and the headers {header_gap:.2f} mm from the grid")
    gaps["top-row labels to model headers"], gaps["between ranked rows"] = band, between_rows
    gaps["model headers to grid"] = header_gap
    return gaps


def draw_example(panel: Mapping[str, object], images: Mapping[tuple[str, tuple[float, float]], PreparedImage],
                 layout: ExampleLayout, fonts: Mapping[str, FontFile], name: str) -> Canvas:
    """One retrieval example: query, recorded positive and ranks 1-5 of both models.

    ``panel`` has ``configuration``, ``query_id``, ``positive_id``, ``description``,
    ``pretrained_rank`` and ``fine_tuned_rank`` (rank of the recorded positive),
    ``ranked_gallery_size`` (number of gallery images in the query's ranking) and, per
    model (``pretrained``, ``fine_tuned``), a list ``<model>_top`` of ``(image_id, kind)``
    for ranks 1 to 5, with kind ``positive``, ``same_article`` or ``unannotated``.
    ``layout`` holds the boxes of the figure (``example_layout``) and
    ``images[(image_id, box)]`` the images sized for those boxes.
    """
    query_box, positive_box, candidate_box = layout.query_box, layout.positive_box, layout.candidate_box
    if query_box[1] != positive_box[1]:
        raise ValueError("the query and its recorded positive must have boxes of the same height")
    check_label_spacing(fonts)
    columns = [CANDIDATE_COLUMN_LEFT_MM, CANDIDATE_COLUMN_LEFT_MM + candidate_box[0] + COLUMN_GAP_MM]
    grid_right = columns[1] + candidate_box[0]
    query_columns = [CANDIDATE_COLUMN_LEFT_MM, CANDIDATE_COLUMN_LEFT_MM + query_box[0] + COLUMN_GAP_MM]
    text_left = query_columns[1] + positive_box[0] + COLUMN_GAP_MM
    lines = example_text_lines(panel, grid_right - text_left, fonts)
    text_height = CAP_HEIGHT_MM + sum(LINE_MM + extra for _, extra in lines[:-1]) + 1.0
    row_pitch = candidate_box[1] + TAG_BAND_MM
    top_row = max(query_box[1] + TAG_BAND_MM, text_height)
    key_lines = _example_key(panel, CANDIDATE_COLUMN_LEFT_MM, grid_right, fonts)
    key_height = (len(key_lines) + 1) * LINE_MM
    top = MARGIN_MM + FRAME_CLEARANCE_MM
    total = top + top_row + HEADER_MM + RANKS_SHOWN * row_pitch + KEY_GAP_MM + key_height + MARGIN_MM
    canvas = Canvas(total, fonts)
    rank_labels = [f"Rank {k + 1}" for k in range(RANKS_SHOWN)]
    if MARGIN_MM + max(canvas.width_of(label) for label in rank_labels) > columns[0] - 1.5:
        raise ValueError("the rank labels do not fit left of the image columns")

    # Row 1: query and recorded positive, with the text block to their right.
    for left, box, image_id, (frame, label) in ((query_columns[0], query_box, panel["query_id"], TOP_ROW_LABELS[0]),
                                                (query_columns[1], positive_box, panel["positive_id"],
                                                 TOP_ROW_LABELS[1])):
        _draw_image_box(canvas, left, top, box, images[(str(image_id), box)], frame)
        canvas.text(left + box[0] / 2, top + box[1] + TAG_BASELINE_MM, label, ha="center")
    baseline = top + CAP_HEIGHT_MM
    for text, extra in lines:
        canvas.text(text_left, baseline, text)
        baseline += LINE_MM + extra

    # Ranks 1-5 of each model, one column per model, with the model headers just above the grid.
    rows_top = top + top_row + HEADER_MM
    for column, model in zip(columns, ("pretrained", "fine_tuned")):
        canvas.text(column + candidate_box[0] / 2, rows_top - HEADER_ABOVE_GRID_MM, MODEL_LABELS[model], ha="center")
    for k in range(RANKS_SHOWN):
        row_top = rows_top + k * row_pitch
        canvas.row_text(columns[0] - 1.5, row_top + candidate_box[1] / 2, rank_labels[k], ha="right")
        for column, model in zip(columns, ("pretrained", "fine_tuned")):
            image_id, kind = panel[f"{model}_top"][k]
            _draw_image_box(canvas, column, row_top, candidate_box, images[(image_id, candidate_box)], kind)
            if CANDIDATE_TAGS[kind]:
                canvas.text(column + candidate_box[0] / 2, row_top + candidate_box[1] + TAG_BASELINE_MM,
                            CANDIDATE_TAGS[kind], ha="center")

    # Key: frame styles, and the displayed versus evaluated ranks.
    baseline = rows_top + RANKS_SHOWN * row_pitch + KEY_GAP_MM + CAP_HEIGHT_MM + 0.4
    for line in key_lines:
        for x, kind, label in line:
            canvas.frame(x, baseline - 2.2, 4.0, 2.8, joinstyle="miter", **FRAME_STYLES[kind])
            canvas.text(x + 5.2, baseline, label)
        baseline += LINE_MM
    canvas.text(CANDIDATE_COLUMN_LEFT_MM, baseline,
                "Ranks 1–5 of each model are shown; Recall@10 and Hit@10 count ranks 1–10.")
    canvas.check_extents(name)
    return canvas


def _example_key(panel: Mapping[str, object], left: float, right: float,
                 fonts: Mapping[str, FontFile]) -> list[list[tuple[float, str, str]]]:
    """Key entries ``(x, kind, label)`` in lines between ``left`` and ``right``; only frame styles that occur.

    The unannotated entry reads "Other unannotated candidate" only when the same-article
    entry, the other kind of unannotated candidate, is listed as well.
    """
    shown = {kind for model in ("pretrained", "fine_tuned") for _, kind in panel[f"{model}_top"]}
    shown.add("positive")
    lines: list[list[tuple[float, str, str]]] = [[]]
    x = left
    for kind in ("positive", "same_article", "unannotated"):
        if kind not in shown:
            continue
        label = KEY_LABELS["other_unannotated" if kind == "unannotated" and "same_article" in shown else kind]
        width = 5.2 + text_width_mm(label, fonts)
        if lines[-1] and x + width > right:
            lines.append([])
            x = left
        lines[-1].append((x, kind, label))
        x += width + 5.0
    return lines


# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------
SVG_IMAGE_STYLE = ' style="image-rendering:crisp-edges;image-rendering:pixelated"'


@contextlib.contextmanager
def _distinct_subset_tags() -> Iterator[None]:
    """Make matplotlib name each embedded font subset after its PostScript name and characters.

    matplotlib derives the six-letter subset tag from the character map of the whole font
    file, which is the same for Arial regular and bold, and not from the subset; tools that
    merge PDF files by font name (for example Ghostscript) can then take one subset for
    another and lose glyphs. The tag used here is the base-26 encoding of the first six
    bytes of sha256(PostScript name, NUL, embedded characters).
    """
    pdf_file = backend_pdf.PdfFile
    for method, parameters in (("embedTTF", ["self", "filename", "characters"]),
                               ("_get_subsetted_psname", ["self", "ps_name", "charmap"])):
        if list(inspect.signature(getattr(pdf_file, method)).parameters) != parameters:
            raise RuntimeError(f"matplotlib {matplotlib.__version__}: PdfFile.{method} has an unexpected signature")
    original_embed = pdf_file.embedTTF

    def embed(self, filename, characters):
        self._subset_characters = "".join(chr(c) for c in sorted(characters))
        return original_embed(self, filename, characters)

    def subset_name(self, ps_name, charmap):
        digest = hashlib.sha256(f"{ps_name}\0{self._subset_characters}".encode("utf-8")).digest()
        return "".join(string.ascii_uppercase[b % 26] for b in digest[:6]) + "+" + ps_name

    with mock.patch.object(pdf_file, "embedTTF", embed), mock.patch.object(pdf_file, "_get_subsetted_psname",
                                                                           subset_name):
        yield


def save(canvas: Canvas, directory: Path, stem: str) -> list[Path]:
    """Write ``<stem>.pdf``, ``<stem>.svg`` and ``<stem>.png`` (300 dpi preview) without timestamps."""
    fig = canvas.figure
    paths = [directory / f"{stem}.{ext}" for ext in ("pdf", "svg", "png")]
    with _distinct_subset_tags():
        fig.savefig(paths[0], format="pdf", metadata={"CreationDate": None})
    # matplotlib marks every unresampled SVG image for nearest-neighbour rendering; the hint is removed so that
    # viewers smooth the images when they scale them. The embedded pixels are not changed.
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", metadata={"Date": None})
    svg = buffer.getvalue()
    n_images = svg.count("<image ")
    if svg.count(SVG_IMAGE_STYLE) != n_images:
        raise RuntimeError(f"{stem}.svg: unexpected image elements; the rendering hint was not found on every image")
    with open(paths[1], "w", encoding="utf-8", newline="\n") as fh:
        fh.write(svg.replace(SVG_IMAGE_STYLE, ""))
    # The PDF and SVG embed every image at its own resolution; the preview smooths the images it rasterises.
    for ax in fig.axes:
        for image in ax.get_images():
            image.set_interpolation("antialiased")
    fig.savefig(paths[2], format="png", dpi=PNG_DPI, metadata={"Software": None})
    return paths
