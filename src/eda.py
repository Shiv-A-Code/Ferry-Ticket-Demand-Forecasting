"""
eda.py
------
Exploratory Data Analysis for the Toronto Island Ferry ticket series.
Produces PNG charts (assets/) and prints summary stats used to write
eda_report.md.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_prep import load_raw, build_regular_grid, _PROJECT_ROOT

plt.rcParams["figure.figsize"] = (11, 4.5)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

OUT = os.path.join(_PROJECT_ROOT, "assets")


def main():
    raw = load_raw()
    grid = build_regular_grid(raw)

    print("=== Basic shape ===")
    print("Raw rows:", len(raw))
    print("Regular 15-min grid rows:", len(grid))
    print("Date range:", grid.index.min(), "->", grid.index.max())
    print("Share interpolated:", grid['is_interpolated'].mean())
    print("Share non-operating (masked to 0):", grid['is_non_operating'].mean())

    print("\n=== Sales Count distribution ===")
    print(grid["Sales Count"].describe())
    print("\n=== Redemption Count distribution ===")
    print(grid["Redemption Count"].describe())

    # 1. Full history (weekly resample for readability)
    weekly = grid[["Sales Count", "Redemption Count"]].resample("W").sum()
    fig, ax = plt.subplots()
    ax.plot(weekly.index, weekly["Sales Count"], label="Sales (weekly total)")
    ax.plot(weekly.index, weekly["Redemption Count"], label="Redemptions (weekly total)", alpha=0.8)
    ax.set_title("Weekly Ticket Volume, 2015-2025")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/01_weekly_history.png", dpi=110)
    plt.close(fig)

    # 2. Hour-of-day profile
    hourly = grid.copy()
    hourly["hour"] = hourly.index.hour
    prof = hourly.groupby("hour")[["Sales Count", "Redemption Count"]].mean()
    fig, ax = plt.subplots()
    ax.bar(prof.index - 0.15, prof["Sales Count"], width=0.3, label="Sales")
    ax.bar(prof.index + 0.15, prof["Redemption Count"], width=0.3, label="Redemptions")
    ax.set_title("Average Ticket Activity by Hour of Day")
    ax.set_xlabel("Hour")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/02_hourly_profile.png", dpi=110)
    plt.close(fig)

    # 3. Day-of-week profile
    dow = grid.copy()
    dow["dow"] = dow.index.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    prof_dow = dow.groupby("dow")[["Sales Count", "Redemption Count"]].mean().reindex(order)
    fig, ax = plt.subplots()
    prof_dow.plot(kind="bar", ax=ax)
    ax.set_title("Average Ticket Activity by Day of Week")
    fig.tight_layout()
    fig.savefig(f"{OUT}/03_dow_profile.png", dpi=110)
    plt.close(fig)

    # 4. Monthly / seasonal profile
    mon = grid.copy()
    mon["month"] = mon.index.month
    prof_m = mon.groupby("month")[["Sales Count", "Redemption Count"]].mean()
    fig, ax = plt.subplots()
    prof_m.plot(kind="bar", ax=ax)
    ax.set_title("Average Ticket Activity by Month (Seasonality)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/04_monthly_profile.png", dpi=110)
    plt.close(fig)

    # 5. One representative summer week, fine-grained
    window = grid.loc["2024-07-01":"2024-07-07"]
    fig, ax = plt.subplots()
    ax.plot(window.index, window["Sales Count"], label="Sales")
    ax.plot(window.index, window["Redemption Count"], label="Redemptions", alpha=0.8)
    ax.set_title("Sample Week (July 2024) — 15-min Resolution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/05_sample_week.png", dpi=110)
    plt.close(fig)

    # 6. Distribution (log scale) of non-zero sales
    fig, ax = plt.subplots()
    nz = grid.loc[grid["Sales Count"] > 0, "Sales Count"]
    ax.hist(nz, bins=60, log=True)
    ax.set_title("Distribution of Sales Count (non-zero intervals, log y-scale)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/06_sales_distribution.png", dpi=110)
    plt.close(fig)

    # 7. Peak-day check: top 10 highest single-interval sales spikes
    top_spikes = grid["Sales Count"].sort_values(ascending=False).head(10)
    print("\n=== Top 10 single-interval sales spikes ===")
    print(top_spikes)

    print("\nSaved 6 charts to assets/")


if __name__ == "__main__":
    main()
