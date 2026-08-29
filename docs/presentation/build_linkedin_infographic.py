"""
Builds a 6-slide LinkedIn carousel (portrait 1080x1350px, LinkedIn's
recommended document/carousel size) summarizing the CIB Early Warning
System project, matching the same navy/red visual language as the
dashboard and presentation deck.

Run with:
    python docs/presentation/build_linkedin_infographic.py

Output: docs/linkedin_carousel/slide_1.png .. slide_6.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "docs" / "linkedin_carousel"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1080, 1350
NAVY = "#0B2447"
NAVY_LIGHT = "#19376D"
RED = "#E4002B"
RED_DARK = "#B3001F"
WHITE = "#FFFFFF"
INK = "#0B2447"
MUTED = "#64748B"
LIGHT_BG = "#F1F5F9"

AUTHOR = "Agastya Sharma"
REPO = "github.com/agastyasharma20/cib-ews"
DEMO = "hdfc-eib-ews.streamlit.app"


def new_canvas(bg=WHITE):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.invert_yaxis()  # y=0 at top, matches how we think about slide layout
    ax.axis("off")
    ax.add_patch(mpatches.Rectangle((0, 0), W, H, color=bg, zorder=0))
    return fig, ax


def box(ax, x, y, w, h, color, radius=18, z=1):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                        linewidth=0, facecolor=color, zorder=z)
    ax.add_patch(b)
    return b


def text(ax, x, y, s, size=28, color=INK, weight="normal", ha="left", va="top", z=2, style="normal", linespacing=1.2):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            zorder=z, fontstyle=style, wrap=True, linespacing=linespacing, family="DejaVu Sans")


def draw_bank_icon(ax, cx, cy, s, color, z=3):
    """A simple hand-drawn bank pictogram (roof + pillars + base) — drawn
    with plain matplotlib shapes rather than an emoji glyph, since the
    rendering environment's font doesn't include emoji glyphs (they were
    silently rendering as blank placeholder boxes — caught by actually
    looking at the output, not assumed to work)."""
    roof = mpatches.Polygon(
        [(cx - s, cy - 0.25 * s), (cx + s, cy - 0.25 * s), (cx, cy - s)],
        closed=True, facecolor=color, edgecolor="none", zorder=z,
    )
    ax.add_patch(roof)
    base = mpatches.Rectangle((cx - s, cy + 0.55 * s), 2 * s, 0.22 * s, facecolor=color, edgecolor="none", zorder=z)
    ax.add_patch(base)
    n_pillars = 4
    pillar_w = 0.22 * s
    gap = (2 * s - n_pillars * pillar_w) / (n_pillars + 1)
    x = cx - s + gap
    for _ in range(n_pillars):
        ax.add_patch(mpatches.Rectangle((x, cy - 0.2 * s), pillar_w, 0.75 * s, facecolor=color, edgecolor="none", zorder=z))
        x += pillar_w + gap


def draw_building_icon(ax, cx, cy, s, color, z=3):
    """A simple office-building pictogram (rect + window grid) for the
    'competitor' side of a diagram — same reasoning as draw_bank_icon."""
    ax.add_patch(mpatches.Rectangle((cx - 0.7 * s, cy - s), 1.4 * s, 2 * s, facecolor=color, edgecolor="none", zorder=z))
    rows, cols = 4, 3
    win_w, win_h = 0.28 * s, 0.28 * s
    x0 = cx - 0.7 * s + 0.18 * s
    y0 = cy - s + 0.2 * s
    for r in range(rows):
        for c in range(cols):
            wx = x0 + c * (win_w + 0.15 * s)
            wy = y0 + r * (win_h + 0.15 * s)
            ax.add_patch(mpatches.Rectangle((wx, wy), win_w, win_h, facecolor=WHITE, edgecolor="none", zorder=z + 1))


def footer(ax, page, total=6):
    text(ax, 60, H - 70, f"{AUTHOR}  •  {REPO}", size=15, color=MUTED, va="bottom")
    text(ax, W - 60, H - 70, f"{page}/{total}", size=15, color=MUTED, ha="right", va="bottom")


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Saved {path}")


# ===========================================================================
# Slide 1 — Cover / Hook
# ===========================================================================
fig, ax = new_canvas(NAVY)
box(ax, 0, H - 14, W, 14, RED, radius=0)

# logo mark
box(ax, W / 2 - 70, 140, 140, 140, WHITE, radius=70)
draw_bank_icon(ax, W / 2, 210, 45, RED)

text(ax, W / 2, 380, "BANKS DON'T LOSE", size=46, color=WHITE, weight="bold", ha="center", va="top")
text(ax, W / 2, 445, "THEIR BIGGEST CUSTOMERS", size=46, color=WHITE, weight="bold", ha="center", va="top")
text(ax, W / 2, 510, "TO CLOSED ACCOUNTS.", size=46, color=RED, weight="bold", ha="center", va="top")

text(ax, W / 2, 610, "They lose them QUIETLY —", size=26, color="#CBD5E1", ha="center", va="top")
text(ax, W / 2, 650, "one transaction at a time —", size=26, color="#CBD5E1", ha="center", va="top")
text(ax, W / 2, 690, "while the account stays open.", size=26, color="#CBD5E1", ha="center", va="top")

box(ax, W / 2 - 260, 820, 520, 130, NAVY_LIGHT, radius=16)
text(ax, W / 2, 855, "I built an AI system that", size=22, color=WHITE, ha="center", va="top")
text(ax, W / 2, 890, "catches it 30-90 DAYS EARLY.", size=24, color="#FFC1CC", weight="bold", ha="center", va="top")

text(ax, W / 2, 1050, "Data Science Internship Project", size=20, color=MUTED, ha="center", va="top")
text(ax, W / 2, 1085, "HDFC Bank  •  100% Synthetic Data", size=18, color=MUTED, ha="center", va="top")

text(ax, W / 2, 1230, "SWIPE ->", size=24, color=RED, weight="bold", ha="center", va="top")
save(fig, "slide_1.png")


# ===========================================================================
# Slide 2 — The Problem
# ===========================================================================
fig, ax = new_canvas(WHITE)
box(ax, 0, 0, W, 140, NAVY)
text(ax, 60, 70, "THE PROBLEM", size=34, color=WHITE, weight="bold", va="center")

text(ax, 60, 220, "A large customer stays with the bank...", size=24, color=INK, weight="bold")

# Two boxes: HDFC (shrinking) vs Competitor (growing) with an arrow between
box(ax, 60, 320, 420, 260, LIGHT_BG, radius=16)
draw_bank_icon(ax, 270, 375, 34, NAVY)
text(ax, 270, 440, "Account stays", size=20, color=INK, ha="center")
text(ax, 270, 470, "OPEN", size=26, color=NAVY, weight="bold", ha="center")
text(ax, 270, 520, "Balance looks fine", size=16, color=MUTED, ha="center")
text(ax, 270, 545, "on paper...", size=16, color=MUTED, ha="center")

text(ax, 540, 450, "➜", size=44, color=RED, ha="center", va="center")

box(ax, 600, 320, 420, 260, "#FFF1F2", radius=16)
draw_building_icon(ax, 810, 375, 34, RED_DARK)
text(ax, 810, 440, "...but payroll, trade,", size=18, color=INK, ha="center")
text(ax, 810, 465, "and transactions", size=18, color=INK, ha="center")
text(ax, 810, 495, "QUIETLY MOVE", size=24, color=RED_DARK, weight="bold", ha="center")
text(ax, 810, 525, "to a competitor.", size=18, color=INK, ha="center")

box(ax, 60, 650, 960, 260, NAVY_LIGHT, radius=16)
text(ax, 100, 700, "By the time the balance visibly drops...", size=22, color=WHITE, weight="bold")
text(ax, 100, 750, "...most of the relationship is", size=22, color="#CBD5E1")
text(ax, 100, 790, "already gone.", size=30, color=RED, weight="bold")
text(ax, 100, 850, "Today's process only reacts AFTER that point.", size=18, color="#94A3B8")

text(ax, 60, 1000, "The real question isn't", size=22, color=INK)
text(ax, 60, 1035, "\"who closed their account?\"", size=22, color=MUTED, style="italic")
text(ax, 60, 1090, "It's \"who's quietly leaving", size=26, color=INK, weight="bold")
text(ax, 60, 1130, "right now?\"", size=26, color=RED, weight="bold")

footer(ax, 2)
save(fig, "slide_2.png")


# ===========================================================================
# Slide 3 — What I Built (pipeline)
# ===========================================================================
fig, ax = new_canvas(WHITE)
box(ax, 0, 0, W, 140, NAVY)
text(ax, 60, 70, "WHAT I BUILT", size=34, color=WHITE, weight="bold", va="center")

steps = [
    ("1", "Synthetic Data", "5,000 customers, 18 months, planted attrition patterns"),
    ("2", "Risk Index", "Ranks each customer vs. PEERS, not a flat threshold"),
    ("3", "XGBoost Model", "Tuned + compared against a baseline, honestly"),
    ("4", "SHAP Explainability", "Not just \"risky\" — WHY, in plain English"),
    ("5", "Risk Sizing + Actions", "₹ at stake + a specific RM recommendation"),
    ("6", "Survival Model + Dashboard", "Estimates WHEN, plus a live RM tool"),
]
y = 210
for num, title_, desc in steps:
    box(ax, 60, y, 960, 155, LIGHT_BG, radius=14)
    box(ax, 90, y + 35, 85, 85, RED, radius=42)
    text(ax, 132, y + 77, num, size=32, color=WHITE, weight="bold", ha="center", va="center")
    text(ax, 210, y + 40, title_, size=24, color=INK, weight="bold")
    text(ax, 210, y + 82, desc, size=16, color=MUTED)
    y += 178

footer(ax, 3)
save(fig, "slide_3.png")


# ===========================================================================
# Slide 4 — The Results
# ===========================================================================
fig, ax = new_canvas(NAVY)
box(ax, 0, H - 14, W, 14, RED, radius=0)
text(ax, 60, 70, "THE RESULTS", size=34, color=WHITE, weight="bold")

stats = [
    ("0.958", "ROC-AUC", "core model, 90-day horizon"),
    ("6.3x", "TOP-DECILE LIFT", "vs. random chance"),
    ("98.9%", "LABEL PRECISION", "validated vs. hidden ground truth"),
    ("0.659", "CONCORDANCE INDEX", "time-to-deterioration model"),
]
y = 220
for value, label, sub in stats:
    box(ax, 60, y, 960, 235, NAVY_LIGHT, radius=16)
    text(ax, 100, y + 40, value, size=56, color=RED, weight="bold")
    text(ax, 100, y + 130, label, size=22, color=WHITE, weight="bold")
    text(ax, 100, y + 168, sub, size=16, color="#94A3B8")
    y += 260

footer(ax, 4)
save(fig, "slide_4.png")


# ===========================================================================
# Slide 5 — The Honest Miss
# ===========================================================================
fig, ax = new_canvas(WHITE)
box(ax, 0, 0, W, 140, RED_DARK)
text(ax, 60, 70, "THE PART I'M MOST PROUD OF", size=28, color=WHITE, weight="bold", va="center")

text(ax, 60, 220, "I tested a graph-based feature", size=28, color=INK, weight="bold")
text(ax, 60, 262, "I was SURE would improve the model.", size=28, color=INK, weight="bold")

box(ax, 60, 350, 960, 160, "#FFF1F2", radius=16)
text(ax, 100, 390, "It didn't.", size=44, color=RED_DARK, weight="bold")
text(ax, 100, 460, "ROC-AUC went from 0.9585 -> 0.9580.", size=18, color=INK)

text(ax, 60, 570, "I could have hidden that.", size=22, color=MUTED, style="italic")

box(ax, 60, 650, 960, 300, LIGHT_BG, radius=16)
text(ax, 100, 690, "Instead I dug in and found the exact", size=20, color=INK)
text(ax, 100, 725, "reason: the new feature correlated", size=20, color=INK)
text(ax, 100, 760, "0.88-0.91 with one I already had.", size=20, color=INK)
text(ax, 100, 815, "Same underlying signal,", size=20, color=INK, weight="bold")
text(ax, 100, 850, "computed a different way.", size=20, color=INK, weight="bold")
text(ax, 100, 900, "Nothing new to learn from it.", size=18, color=MUTED)

text(ax, 60, 1020, "In real data science,", size=26, color=INK)
text(ax, 60, 1062, "a negative result is still a result.", size=26, color=RED, weight="bold")

footer(ax, 5)
save(fig, "slide_5.png")


# ===========================================================================
# Slide 6 — CTA
# ===========================================================================
fig, ax = new_canvas(NAVY)
box(ax, 0, H - 14, W, 14, RED, radius=0)

box(ax, W / 2 - 60, 130, 120, 120, WHITE, radius=60)
draw_bank_icon(ax, W / 2, 190, 38, RED)

text(ax, W / 2, 320, "TRY IT YOURSELF", size=40, color=WHITE, weight="bold", ha="center")

box(ax, W / 2 - 440, 430, 880, 140, NAVY_LIGHT, radius=16)
text(ax, W / 2, 465, "LIVE DEMO", size=18, color="#94A3B8", ha="center")
text(ax, W / 2, 500, DEMO, size=24, color=RED, weight="bold", ha="center")

box(ax, W / 2 - 440, 610, 880, 140, NAVY_LIGHT, radius=16)
text(ax, W / 2, 645, "CODE + FULL WRITEUP", size=18, color="#94A3B8", ha="center")
text(ax, W / 2, 680, REPO, size=24, color=RED, weight="bold", ha="center")

text(ax, W / 2, 850, "If you build fintech risk models —", size=20, color="#CBD5E1", ha="center")
text(ax, W / 2, 885, "what would YOU have done differently?", size=20, color="#CBD5E1", ha="center")

text(ax, W / 2, 1010, f"{AUTHOR}", size=26, color=WHITE, weight="bold", ha="center")
text(ax, W / 2, 1050, "Data Science Intern, HDFC Bank", size=18, color=MUTED, ha="center")

text(ax, W / 2, 1180, "♻ Repost if banks should catch this", size=17, color="#94A3B8", ha="center")
text(ax, W / 2, 1210, "BEFORE the balance drops.", size=17, color="#94A3B8", ha="center")

save(fig, "slide_6.png")

print(f"\nAll slides saved to {OUT_DIR}")
