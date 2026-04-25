"""
Interactive trade-detail visualization for the AS market-making strategy.

Loads a trade-detail CSV (produced by paper_experiments.py) and draws a
three-panel chart:

    Panel 1  Mid-price + bid/ask quotes + event markers
    Panel 2  Inventory (shares)
    Panel 3  Mark-to-market P&L

Event markers
-------------
    ▽ PLACE_BID   – green, triangle-down
    △ PLACE_ASK   – red, triangle-up
    ✕ CANCEL_BID  – dark-green, ×
    ✕ CANCEL_ASK  – dark-red, ×
    ◆ FILL_BID    – blue, diamond (filled, we bought)
    ◆ FILL_ASK    – orange, diamond (filled, we sold)

Usage
-----
    # Interactive (full day, range slider to zoom)
    python trade_detail_viz.py

    # Specify a different trade-log file
    python trade_detail_viz.py --file data/result/trade_detail_20260420_gamma0.1.csv

    # Filter to a wall-clock time range (HH:MM:SS)
    python trade_detail_viz.py --start 09:30:00 --end 10:30:00

    # Filter by normalised time [0, 1]
    python trade_detail_viz.py --t-start 0.0 --t-end 0.25

    # Save static PNG instead of showing an interactive window
    python trade_detail_viz.py --start 09:30:00 --end 10:30:00 --save out.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.widgets import RangeSlider


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_FILE = "data/result/trade_detail_20260420_gamma0.1.csv"

# Marker style per event type
_EVENT_STYLES = {
    "PLACE_BID":   dict(marker="v",  color="green",     alpha=0.35, s=18, zorder=3),
    "PLACE_ASK":   dict(marker="^",  color="red",       alpha=0.35, s=18, zorder=3),
    "CANCEL_BID":  dict(marker="x",  color="darkgreen", alpha=0.6,  s=28, zorder=4),
    "CANCEL_ASK":  dict(marker="x",  color="darkred",   alpha=0.6,  s=28, zorder=4),
    "FILL_BID":    dict(marker="D",  color="blue",      alpha=0.90, s=55, zorder=5),
    "FILL_ASK":    dict(marker="D",  color="darkorange",alpha=0.90, s=55, zorder=5),
}

# Human-readable Chinese label for the legend
_EVENT_LABELS = {
    "PLACE_BID":  "挂单-买 (PLACE_BID)",
    "PLACE_ASK":  "挂单-卖 (PLACE_ASK)",
    "CANCEL_BID": "撤单-买 (CANCEL_BID)",
    "CANCEL_ASK": "撤单-卖 (CANCEL_ASK)",
    "FILL_BID":   "成交-买 (FILL_BID)",
    "FILL_ASK":   "成交-卖 (FILL_ASK)",
}

TRADING_SECONDS = 14_400   # total continuous-trading seconds per day


# ── Time helpers ──────────────────────────────────────────────────────────────

def _time_str_to_seconds(ts: str) -> float:
    """'HH:MM:SS.mmm' → seconds since midnight."""
    try:
        parts = ts.split(":")
        h = int(parts[0])
        m = int(parts[1])
        s_ms = parts[2].split(".")
        s = int(s_ms[0])
        ms = int(s_ms[1]) if len(s_ms) > 1 else 0
        return h * 3600 + m * 60 + s + ms / 1000
    except Exception:
        return float("nan")


def _seconds_to_hhmm(sec: float) -> str:
    """Seconds from midnight → 'HH:MM' string (for axis tick labels)."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    return f"{h:02d}:{m:02d}"


def _norm_to_wallclock(t_norm: float) -> float:
    """Normalised time [0,1] → seconds since midnight (skipping lunch break)."""
    morning_open = 9 * 3600 + 30 * 60    # 34 200 s
    morning_close = 11 * 3600 + 30 * 60  # 41 400 s
    afternoon_open = 13 * 3600            # 46 800 s
    elapsed = t_norm * TRADING_SECONDS
    if elapsed <= (morning_close - morning_open):
        return morning_open + elapsed
    else:
        return afternoon_open + (elapsed - (morning_close - morning_open))


def _wallclock_to_norm(sec: float) -> float:
    """Wall-clock seconds-from-midnight → normalised time [0,1]."""
    morning_open = 9 * 3600 + 30 * 60
    morning_close = 11 * 3600 + 30 * 60
    afternoon_open = 13 * 3600
    if sec <= morning_close:
        elapsed = sec - morning_open
    else:
        elapsed = (morning_close - morning_open) + (sec - afternoon_open)
    return elapsed / TRADING_SECONDS


# ── Data loading & filtering ──────────────────────────────────────────────────

def load_trade_log(path: str) -> pd.DataFrame:
    """Load and lightly validate the trade-detail CSV."""
    df = pd.read_csv(path)
    required = {"time_str", "time_norm", "event_type", "price",
                "volume", "mid_price", "inventory", "cash", "pnl"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trade log is missing columns: {missing}")
    df = df.sort_values("time_norm").reset_index(drop=True)
    return df


def filter_by_time(
    df: pd.DataFrame,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> pd.DataFrame:
    """Filter trade log to a normalised-time window [t_start, t_end]."""
    if t_start is not None:
        df = df[df["time_norm"] >= t_start]
    if t_end is not None:
        df = df[df["time_norm"] <= t_end]
    return df


# ── Core plot builder ──────────────────────────────────────────────────────────

def build_figure(
    df: pd.DataFrame,
    t_start: float,
    t_end: float,
    title: str = "AS Strategy — Trade Detail",
) -> tuple:
    """
    Build a 3-panel figure from the trade log.

    Returns (fig, axes, scatter_artists) so that callers can update limits
    without rebuilding from scratch.
    """
    matplotlib.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1.5]},
    )
    fig.suptitle(title, fontsize=13)

    view = filter_by_time(df, t_start, t_end)

    # ── Panel 1: prices + event markers ──────────────────────────────────────
    ax1 = axes[0]

    # Continuous price lines from the PLACE_BID rows (one per snapshot)
    price_rows = view[view["event_type"] == "PLACE_BID"].drop_duplicates(
        subset="time_norm"
    )
    if not price_rows.empty:
        ax1.plot(
            price_rows["time_norm"], price_rows["mid_price"],
            color="black", lw=0.8, label="Mid-price $S(t)$", zorder=2,
        )

    bid_rows = view[view["event_type"] == "PLACE_BID"]
    ask_rows = view[view["event_type"] == "PLACE_ASK"]
    if not bid_rows.empty:
        ax1.plot(
            bid_rows["time_norm"], bid_rows["price"],
            color="green", lw=0.5, alpha=0.5, label="Bid quote", zorder=2,
        )
    if not ask_rows.empty:
        ax1.plot(
            ask_rows["time_norm"], ask_rows["price"],
            color="red", lw=0.5, alpha=0.5, label="Ask quote", zorder=2,
        )

    # Event scatter markers
    scatter_artists: dict = {}
    for etype, style in _EVENT_STYLES.items():
        sub = view[view["event_type"] == etype]
        if sub.empty:
            continue
        sc = ax1.scatter(
            sub["time_norm"], sub["price"],
            marker=style["marker"],
            c=style["color"],
            alpha=style["alpha"],
            s=style["s"],
            zorder=style["zorder"],
            label=_EVENT_LABELS[etype],
        )
        scatter_artists[etype] = sc

    ax1.set_ylabel("Price (CNY)")
    ax1.set_title("Panel 1 — Quotes & Strategy Events")
    _add_legend(ax1)

    # ── Panel 2: inventory ────────────────────────────────────────────────────
    ax2 = axes[1]
    # Use last inventory reading at each time_norm
    inv_view = view.drop_duplicates(subset="time_norm", keep="last")
    ax2.step(
        inv_view["time_norm"], inv_view["inventory"],
        where="post", color="purple", lw=0.8,
    )
    ax2.axhline(0, color="gray", lw=0.5, ls=":")
    ax2.set_ylabel("Inventory (shares)")
    ax2.set_title("Panel 2 — Inventory")

    # ── Panel 3: PnL ─────────────────────────────────────────────────────────
    ax3 = axes[2]
    pnl_view = view.drop_duplicates(subset="time_norm", keep="last")
    ax3.plot(
        pnl_view["time_norm"], pnl_view["pnl"],
        color="darkorange", lw=0.8,
    )
    ax3.axhline(0, color="gray", lw=0.5, ls=":")
    ax3.set_ylabel("P&L (CNY)")
    ax3.set_title("Panel 3 — Mark-to-Market P&L")

    # ── X-axis: normalised time with HH:MM labels ─────────────────────────────
    _format_time_axis(ax3, t_start, t_end)

    plt.tight_layout()
    return fig, axes, scatter_artists


def _add_legend(ax: plt.Axes) -> None:
    """Add a compact legend that includes both line and scatter handles."""
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        fontsize=7, loc="upper left",
        ncol=2, framealpha=0.7,
        markerscale=1.4,
    )


def _format_time_axis(ax: plt.Axes, t_start: float, t_end: float) -> None:
    """Replace normalised-time ticks with wall-clock HH:MM labels."""
    n_ticks = 8
    tick_norms = np.linspace(t_start, t_end, n_ticks)
    tick_labels = [
        _seconds_to_hhmm(_norm_to_wallclock(tn)) for tn in tick_norms
    ]
    ax.set_xticks(tick_norms)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_xlabel("Wall-clock time")


# ── Static plot (fixed time window) ───────────────────────────────────────────

def plot_static(
    df: pd.DataFrame,
    t_start: float,
    t_end: float,
    title: str = "AS Strategy — Trade Detail",
    save_path: Optional[str] = None,
) -> None:
    """Render a static, non-interactive chart."""
    fig, axes, _ = build_figure(df, t_start, t_end, title=title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Interactive plot (range slider) ───────────────────────────────────────────

def plot_interactive(
    df: pd.DataFrame,
    title: str = "AS Strategy — Trade Detail  [drag slider to zoom]",
) -> None:
    """
    Full-day chart with a RangeSlider at the bottom.

    Drag the slider handles to zoom into any time window.
    """
    t_min = float(df["time_norm"].min())
    t_max = float(df["time_norm"].max())

    fig, axes, _ = build_figure(df, t_min, t_max, title=title)

    # Reserve space at the bottom for the slider
    fig.subplots_adjust(bottom=0.18)
    ax_slider = fig.add_axes([0.12, 0.04, 0.78, 0.04])

    slider = RangeSlider(
        ax_slider, "Time window",
        valmin=t_min, valmax=t_max,
        valinit=(t_min, t_max),
    )

    # Format slider value labels as HH:MM
    def _fmt(val: float) -> str:
        return _seconds_to_hhmm(_norm_to_wallclock(val))

    slider.valtext.set_text(
        f"{_fmt(t_min)} – {_fmt(t_max)}"
    )

    def on_slider_change(val: tuple) -> None:
        lo, hi = slider.val
        if hi <= lo:
            return

        slider.valtext.set_text(f"{_fmt(lo)} – {_fmt(hi)}")

        # Rebuild figure content
        for ax in axes:
            ax.cla()

        view = filter_by_time(df, lo, hi)
        ax1, ax2, ax3 = axes

        # Panel 1
        price_rows = view[view["event_type"] == "PLACE_BID"].drop_duplicates(
            subset="time_norm"
        )
        if not price_rows.empty:
            ax1.plot(
                price_rows["time_norm"], price_rows["mid_price"],
                color="black", lw=0.8, label="Mid-price $S(t)$", zorder=2,
            )
        bid_r = view[view["event_type"] == "PLACE_BID"]
        ask_r = view[view["event_type"] == "PLACE_ASK"]
        if not bid_r.empty:
            ax1.plot(bid_r["time_norm"], bid_r["price"],
                     color="green", lw=0.5, alpha=0.5, label="Bid quote", zorder=2)
        if not ask_r.empty:
            ax1.plot(ask_r["time_norm"], ask_r["price"],
                     color="red", lw=0.5, alpha=0.5, label="Ask quote", zorder=2)
        for etype, style in _EVENT_STYLES.items():
            sub = view[view["event_type"] == etype]
            if sub.empty:
                continue
            ax1.scatter(
                sub["time_norm"], sub["price"],
                marker=style["marker"], c=style["color"],
                alpha=style["alpha"], s=style["s"], zorder=style["zorder"],
                label=_EVENT_LABELS[etype],
            )
        ax1.set_ylabel("Price (CNY)")
        ax1.set_title("Panel 1 — Quotes & Strategy Events")
        _add_legend(ax1)

        # Panel 2
        inv_v = view.drop_duplicates(subset="time_norm", keep="last")
        ax2.step(inv_v["time_norm"], inv_v["inventory"],
                 where="post", color="purple", lw=0.8)
        ax2.axhline(0, color="gray", lw=0.5, ls=":")
        ax2.set_ylabel("Inventory (shares)")
        ax2.set_title("Panel 2 — Inventory")

        # Panel 3
        pnl_v = view.drop_duplicates(subset="time_norm", keep="last")
        ax3.plot(pnl_v["time_norm"], pnl_v["pnl"],
                 color="darkorange", lw=0.8)
        ax3.axhline(0, color="gray", lw=0.5, ls=":")
        ax3.set_ylabel("P&L (CNY)")
        ax3.set_title("Panel 3 — Mark-to-Market P&L")
        _format_time_axis(ax3, lo, hi)

        fig.canvas.draw_idle()

    slider.on_changed(on_slider_change)
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive trade-detail visualizer for the AS strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--file", default=DEFAULT_FILE,
        help=f"Path to trade detail CSV (default: {DEFAULT_FILE})",
    )
    parser.add_argument(
        "--start", metavar="HH:MM:SS",
        help="Start time filter (wall-clock, e.g. 09:30:00)",
    )
    parser.add_argument(
        "--end", metavar="HH:MM:SS",
        help="End time filter (wall-clock, e.g. 11:00:00)",
    )
    parser.add_argument(
        "--t-start", type=float, default=None, dest="t_start",
        help="Start filter as normalised time [0,1]",
    )
    parser.add_argument(
        "--t-end", type=float, default=None, dest="t_end",
        help="End filter as normalised time [0,1]",
    )
    parser.add_argument(
        "--save", metavar="PATH",
        help="Save static PNG to this path instead of showing an interactive window",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        print(f"Error: trade log file not found: {args.file}", file=sys.stderr)
        print(
            "Tip: run  python paper_experiments.py  first to generate the file.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = load_trade_log(args.file)
    t_min = float(df["time_norm"].min())
    t_max = float(df["time_norm"].max())

    # Resolve time range
    t_start = args.t_start
    t_end = args.t_end

    if args.start:
        t_start = _wallclock_to_norm(_time_str_to_seconds(args.start))
    if args.end:
        t_end = _wallclock_to_norm(_time_str_to_seconds(args.end))

    t_start = t_start if t_start is not None else t_min
    t_end = t_end if t_end is not None else t_max
    t_start = max(t_start, t_min)
    t_end = min(t_end, t_max)

    date_tag = os.path.basename(args.file).replace("trade_detail_", "").replace(".csv", "")
    title = f"AS Strategy — Trade Detail  [{date_tag}]"

    if args.save or (args.start or args.end or args.t_start is not None or args.t_end is not None):
        # Static mode
        plot_static(df, t_start, t_end, title=title, save_path=args.save)
    else:
        # Interactive mode: show full day with range slider
        matplotlib.use("TkAgg") if os.environ.get("DISPLAY") else None
        plot_interactive(df, title=title)


if __name__ == "__main__":
    main()
