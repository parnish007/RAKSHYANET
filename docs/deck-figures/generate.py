"""RakshyaNet deck figures — Claude Design "Organic" system.

Warm ivory ground, terracotta and sage accents, Fraunces display over Figtree
body, IBM Plex Mono for every number. Editorial data-visualisation conventions:
no chart junk, no boxed axes, gridlines that recede, labels on the data rather
than in a legend where possible.
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = r"C:\Users\AB\Desktop\rakshyanet"
FONTS = r"D:\AI\Temp\claude\C--Users-AB-Desktop-rakshyanet\f738650d-1970-416f-b7f9-4271da2f38ba\scratchpad\fonts"
OUT = os.path.join(ROOT, "docs", "deck-figures")
os.makedirs(OUT, exist_ok=True)

# ── Organic palette ───────────────────────────────────────────────────────
IVORY   = "#FBF6EE"   # page
SAND    = "#F3E8D8"   # secondary surface
CLAY    = "#EADCC7"   # raised surface / gridline base
INK     = "#2E2019"   # primary text, deep warm brown
MUTED   = "#6B584A"
SUBTLE  = "#9C8A79"
TERRA   = "#C2673B"   # primary accent
EMBER   = "#DE8A4F"   # lighter accent
RUST    = "#8F4426"   # deep accent
SAGE    = "#7D8A6A"   # secondary accent
OLIVE   = "#5B6A4B"
GOOD    = "#6E8C5A"
BAD     = "#A8402C"

for f in os.listdir(FONTS):
    if f.endswith(".ttf"):
        font_manager.fontManager.addfont(os.path.join(FONTS, f))

names = {f.name for f in font_manager.fontManager.ttflist}
DISPLAY = "Fraunces" if "Fraunces" in names else ("Playfair Display" if "Playfair Display" in names else "Georgia")
BODY    = "Figtree" if "Figtree" in names else "Segoe UI"
MONO    = "IBM Plex Mono" if "IBM Plex Mono" in names else "Consolas"

plt.rcParams.update({
    "font.family": BODY,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "xtick.color": SUBTLE,
    "ytick.color": SUBTLE,
    "figure.facecolor": IVORY,
    "axes.facecolor": IVORY,
    "savefig.facecolor": IVORY,
    "axes.edgecolor": CLAY,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=210, bbox_inches="tight", pad_inches=0.42)
    plt.close(fig)
    print(f"{name:32s} {os.path.getsize(p)/1024:6.0f} KB")


def title(ax, text, sub=None, x=0.0, y=1.0):
    ax.text(x, y, text, transform=ax.transAxes, fontfamily=DISPLAY,
            fontsize=26, color=INK, va="bottom", ha="left")
    if sub:
        ax.text(x, y - 0.075, sub, transform=ax.transAxes, fontfamily=BODY,
                fontsize=13, color=MUTED, va="bottom", ha="left")


def card(ax, x, y, w, h, edge=CLAY, face=SAND, lw=1.2, r=0.22):
    """Warm card with a soft offset shadow, the Organic depth cue."""
    ax.add_patch(FancyBboxPatch(
        (x + 0.045, y - 0.055), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        linewidth=0, facecolor=CLAY, alpha=0.55, zorder=1))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=2))


# ── f1 · the pipeline ─────────────────────────────────────────────────────
def pipeline():
    fig, ax = plt.subplots(figsize=(15.5, 5.2))
    ax.set_xlim(0, 15.5); ax.set_ylim(0, 5.2); ax.axis("off")
    title(ax, "Four stages, one direction",
          "Gemma interprets the evidence and chooses the computation. It cannot allocate, route, approve, or dispatch.",
          y=0.86)

    stages = [
        ("01", "Evidence",  "three field reports\nthat disagree",      SUBTLE),
        ("02", "Gemma",     "reads · cites · calls\nthe engine",       TERRA),
        ("03", "Engine",    "urgency · Dijkstra ·\nallocation",        SAGE),
        ("04", "Human",     "authorises,\nor refuses",                 RUST),
    ]
    w, h, y = 3.24, 2.5, 1.05
    for i, (num, name, sub, colour) in enumerate(stages):
        x = 0.12 + i * 3.82
        card(ax, x, y, w, h, edge=colour, face=IVORY, lw=1.6)
        ax.add_patch(Circle((x + 0.46, y + h - 0.46), 0.245,
                            facecolor=colour, edgecolor="none", zorder=3))
        ax.text(x + 0.46, y + h - 0.46, num, ha="center", va="center",
                fontsize=10.5, color=IVORY, fontfamily=MONO, zorder=4)
        ax.text(x + 0.92, y + h - 0.5, name, ha="left", va="center",
                fontsize=21, color=INK, fontfamily=DISPLAY, zorder=4)
        ax.text(x + 0.46, y + 0.72, sub, ha="left", va="center",
                fontsize=13, color=MUTED, linespacing=1.6, zorder=4)

    for i in range(3):
        x = 0.12 + w + i * 3.82 + 0.06
        ax.add_patch(FancyArrowPatch(
            (x, y + h / 2), (x + 0.46, y + h / 2), arrowstyle="-|>",
            mutation_scale=20, linewidth=1.6, color=SUBTLE, zorder=1))
    save(fig, "f1-pipeline.png")


# ── f2 · urgency curve ────────────────────────────────────────────────────
def urgency():
    fig, ax = plt.subplots(figsize=(11, 6.4))
    t = [i * 0.05 for i in range(0, 241)]
    exp = [1 + 0.5 * (math.exp(0.3 * x) - 1) for x in t]
    lin = [1 + 0.2 * x for x in t]

    ax.fill_between(t, exp, color=TERRA, alpha=0.10, zorder=1)
    ax.plot(t, lin, color=SUBTLE, linewidth=1.5, linestyle=(0, (5, 4)), zorder=2)
    ax.plot(t, exp, color=TERRA, linewidth=3.4, zorder=4, solid_capstyle="round")

    # Annotations sit in the empty upper-left field, not on top of the curves
    # they describe, with a thin leader line to the data.
    ax.text(0.95, 15.9, "what the survival curve\nactually looks like",
            fontsize=13, color=RUST, ha="left", va="top", linespacing=1.5)
    ax.annotate("", xy=(9.7, 13.2), xytext=(3.5, 14.5),
                arrowprops=dict(arrowstyle="-", color=RUST, linewidth=1.0,
                                connectionstyle="arc3,rad=-0.24"), zorder=5)
    ax.text(11.8, 2.55, "if urgency grew linearly", fontsize=12, color=SUBTLE,
            ha="right", va="top", style="italic")

    # Offsets chosen per point so no label crosses a line.
    label_offsets = {2: (-16, 18), 4: (-14, 24), 8: (16, -6)}
    aligns = {2: "right", 4: "right", 8: "left"}
    for hour in (2, 4, 8):
        v = 1 + 0.5 * (math.exp(0.3 * hour) - 1)
        ax.plot([hour], [v], "o", color=TERRA, markersize=9,
                markeredgecolor=IVORY, markeredgewidth=2.2, zorder=6)
        ax.annotate(f"T({hour}) = {v:.3f}", (hour, v),
                    textcoords="offset points", xytext=label_offsets[hour],
                    ha=aligns[hour], fontsize=12, color=INK,
                    fontfamily=MONO, zorder=6)

    ax.set_xlim(0, 12); ax.set_ylim(0, 19)
    ax.set_xlabel("hours since the incident began", fontsize=13, labelpad=10)
    ax.grid(True, axis="y", color=CLAY, linewidth=1.1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(CLAY)
    title(ax, "Hour eight is not four times worse than hour two",
          r"$T(t) = 1 + 0.5\,(e^{0.3t} - 1)$   ·   why the time factor is exponential, not linear",
          y=1.04)
    save(fig, "f2-urgency-curve.png")


# ── f3 · baseline dumbbell ────────────────────────────────────────────────
def baseline():
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.6); ax.axis("off")
    title(ax, "What survives when a corridor closes",
          "closure of east_west_bharatpur_nepalgunj — identical inputs to both planners",
          y=0.90)

    rows = [
        ("Naive planner", 4, BAD,  "5 routes still run through the closed road"),
        ("RakshyaNet",    9, GOOD, "re-planned around it — every route drivable"),
    ]
    x0, span = 3.5, 6.4
    for i, (label, ok, colour, note) in enumerate(rows):
        y = 4.15 - i * 1.75
        ax.text(3.2, y, label, fontsize=15.5, color=INK, ha="right", va="center",
                fontfamily=BODY)
        ax.plot([x0, x0 + span], [y, y], color=CLAY, linewidth=7,
                solid_capstyle="round", zorder=1)
        ax.plot([x0, x0 + span * ok / 9], [y, y], color=colour, linewidth=7,
                solid_capstyle="round", zorder=2)
        ax.plot([x0 + span * ok / 9], [y], "o", color=colour, markersize=15,
                markeredgecolor=IVORY, markeredgewidth=2.5, zorder=3)
        ax.text(x0 + span + 0.35, y, f"{ok}/9", fontsize=25, color=colour,
                va="center", ha="left", fontfamily=MONO)
        ax.text(x0, y - 0.52, note, fontsize=12.5, color=MUTED, va="center")

    ax.text(3.5, 5.02, "executable routes, out of nine assigned",
            fontsize=11.5, color=SUBTLE, va="center", ha="left")

    card(ax, 3.5, 0.35, 7.9, 1.0, edge=CLAY, face=SAND, lw=1.1)
    ax.text(3.85, 0.85,
            "Measured limitation:  terrain weighting alone changes no path on this network.",
            fontsize=12.5, color=INK, va="center")
    ax.text(3.85, 0.55,
            "The entire advantage is closure-aware re-planning. We measured it and we say so.",
            fontsize=12, color=MUTED, va="center")
    ax.text(0.15, 1.55, "+8.0%", fontsize=22, color=MUTED, fontfamily=MONO, va="center")
    ax.text(0.15, 1.15, "more distance,\nthe price of a\nvalid plan", fontsize=11.5,
            color=SUBTLE, va="top", linespacing=1.55)
    save(fig, "f3-baseline.png")


# ── f4 · corridor map ─────────────────────────────────────────────────────
def corridors():
    g = json.load(open(os.path.join(ROOT, "backend", "data", "terrain_graph.json"),
                        encoding="utf-8"))
    nodes = {n["id"]: n for n in g["nodes"]}
    closed = "east_west_bharatpur_nepalgunj"

    fig, ax = plt.subplots(figsize=(13.5, 8.2))
    ax.axis("off")

    offsets = {
        "depot": (0.0, 0.26), "jumla": (0.0, 0.24), "mahendranagar": (0.0, 0.24),
        "nepalgunj": (-0.12, -0.34), "pokhara": (0.0, 0.24),
        "bharatpur": (0.28, -0.30), "janakpur": (0.0, -0.34),
        "dharan": (0.26, -0.30), "taplejung": (0.0, 0.24),
    }
    for e in g["edges"]:
        a, b = nodes.get(e["from"]), nodes.get(e["to"])
        if not a or not b:
            continue
        shut = e["id"] == closed
        vuln = e.get("vulnerable_to_landslide")
        ax.plot([a["lng"], b["lng"]], [a["lat"], b["lat"]],
                color=BAD if shut else (EMBER if vuln else SAGE),
                linewidth=4.0 if shut else 1.9,
                linestyle=(0, (5, 3.5)) if shut else "solid",
                alpha=1.0 if shut else (0.85 if vuln else 0.6),
                zorder=4 if shut else 2, solid_capstyle="round")

    a, b = nodes["bharatpur"], nodes["nepalgunj"]
    ax.annotate("CLOSED", xy=((a["lng"] + b["lng"]) / 2, (a["lat"] + b["lat"]) / 2),
                xytext=(-58, -34), textcoords="offset points",
                fontsize=12, color=BAD, fontfamily=MONO,
                arrowprops=dict(arrowstyle="-", color=BAD, linewidth=1.1), zorder=7)

    for n in g["nodes"]:
        hub = n["id"] == "depot"
        ax.plot([n["lng"]], [n["lat"]], "o",
                markersize=17 if hub else 11,
                color=RUST if hub else INK,
                markeredgecolor=IVORY, markeredgewidth=2.6, zorder=6)
        dx, dy = offsets.get(n["id"], (0.0, 0.24))
        ax.text(n["lng"] + dx, n["lat"] + dy,
                "Kathmandu hub" if hub else n.get("name", n["id"]),
                ha="center", fontsize=11.5,
                color=INK if hub else MUTED, zorder=7,
                fontfamily=BODY)

    ax.margins(x=0.10, y=0.17)
    title(ax, "Every ground route crosses one corridor",
          "13 corridors · amber marks the 8 that are landslide-vulnerable · closing the busiest strands five trucks",
          y=1.02)
    save(fig, "f4-corridors.png")


# ── f5 · bounded influence ────────────────────────────────────────────────
def bounded():
    fig, ax = plt.subplots(figsize=(11.5, 5.0)); ax.axis("off")
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 5.0)
    title(ax, "How far the model can move a priority",
          "Gemma's entire influence on ranking is one bounded scalar, drawn here to scale",
          y=0.90)

    scale = 0.93
    ax.plot([0.12, 0.12 + 10.0 * scale], [1.55, 1.55], color=CLAY, linewidth=26,
            solid_capstyle="butt", zorder=1)
    ax.plot([0.12, 0.12 + 1.0 * scale], [1.55, 1.55], color=TERRA, linewidth=26,
            solid_capstyle="butt", zorder=2)

    ax.text(0.12 + 1.0 * scale + 0.22, 2.30, "≤ 1.0", fontsize=19, color=TERRA,
            fontfamily=MONO, va="center")
    ax.text(0.12 + 1.0 * scale + 0.22, 1.92, "maximum Gemma contribution",
            fontsize=12.5, color=TERRA, va="center")
    ax.text(0.12 + 10.0 * scale, 0.88, "10.0", fontsize=19, color=MUTED,
            fontfamily=MONO, ha="right", va="center")
    ax.text(0.12 + 10.0 * scale, 0.52, "deterministic survival-threshold penalty",
            fontsize=12.5, color=MUTED, ha="right", va="center")
    ax.text(0.12, 0.12,
            "B = round( max(severity, medical_urgency, accessibility_risk) × system_confidence , 4 )",
            fontsize=11.5, color=SUBTLE, fontfamily=MONO, va="center")
    save(fig, "f5-bounded-influence.png")


# ── f6 · the disagreement ─────────────────────────────────────────────────
def sources():
    fig, ax = plt.subplots(figsize=(14.5, 6.2))
    ax.set_xlim(0, 14.5); ax.set_ylim(0, 6.2); ax.axis("off")
    title(ax, "Three sources. They disagree.",
          "Nobody resolves the contradiction for the dispatcher — and there are four helicopters for the country",
          y=0.90)

    cards = [
        ("NEPAL POLICE", "Heavy vehicles cannot pass\nthe primary approach.", TERRA),
        ("MUNICIPALITY", "180–340 residents isolated.\nMedical supplies requested.", SAGE),
        ("DHM WEATHER",  "Continued rainfall. Risk of\nsecondary landslides.", RUST),
    ]
    for i, (src, claim, colour) in enumerate(cards):
        x = 0.15 + i * 4.82
        card(ax, x, 2.05, 4.42, 2.55, edge=colour, face=IVORY, lw=1.5)
        ax.plot([x + 0.38, x + 0.38 + 0.62], [4.18, 4.18], color=colour,
                linewidth=3, solid_capstyle="round", zorder=4)
        ax.text(x + 0.38, 3.83, src, fontsize=10.5, color=colour,
                fontfamily=MONO, va="center", zorder=4)
        ax.text(x + 0.38, 2.92, claim, fontsize=14, color=INK, va="center",
                linespacing=1.62, zorder=4, fontfamily=BODY)

    ax.text(7.25, 1.35,
            "Both claims are kept, with their citations.",
            fontsize=14.5, color=INK, ha="center", va="center", fontfamily=DISPLAY)
    ax.text(7.25, 0.78,
            "The system does not pick a winner — source credibility is a human judgement, not a model output.",
            fontsize=12.5, color=MUTED, ha="center", va="center")
    save(fig, "f6-sources.png")


for fn in (pipeline, urgency, baseline, corridors, bounded, sources):
    fn()
print(f"\ndisplay={DISPLAY}  body={BODY}  mono={MONO}")
print("out:", OUT)
