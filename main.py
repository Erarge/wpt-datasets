import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# ----------------- Configuration -----------------
DATA_FOLDER = Path(__file__).parent / "data"
PLOT_FOLDER = Path(__file__).parent / "plots"

PLOT_BOXPLOT = False
PLOT_HISTOGRAM = False
PLOT_KDE_HEATMAP = False
PLOT_SCATTER_PLOTS = False
PLOT_POWER_DISTRIBUTIONS = False
PLOT_TEMP_AND_POWER = False
PLOT_MAX_TEMP_AND_POWER = True
SUMMARY = True
# --------------------------------------------------


def load_all_csvs(folder_path: Path) -> pd.DataFrame:
    """Recursively load all CSV files except session_parameters.csv and combine them."""
    all_files = folder_path.rglob("*.csv")
    dfs = []
    for file in all_files:
        if file.name == "session_parameters.csv":
            continue
        try:
            df = pd.read_csv(file)
            df["source_file"] = file.name
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    return pd.concat(dfs, ignore_index=True)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and compute required fields in the DataFrame."""
    df = df.copy()
    df["soc"] = df["soc"].astype(str).str.replace('%', '', regex=False).astype(int)
    df["distance"] = df["distance"].astype(str) + "mm"
    df["p_pri"] = df["v_pri_v"] * df["a_pri_a"]
    df["p_sin"] = df["v_sin_v"] * df["a_sin_a"]
    df["p_sout"] = df["v_sout_v"] * df["a_sout_a"]
    df["efficiency"] = df["p_sin"] / df["p_pri"]
    print(f"Preprocessed DataFrame with {len(df[df['efficiency'] > 1.0])} unphysical entries.")
    df = df[df["efficiency"] <= 1.0]  # Drop unphysical entries
    max_eff = df["efficiency"].max()
    df["eff_norm"] = df["efficiency"] / max_eff
    df["energy_loss"] = df["p_pri"] - df["p_sin"]
    return df


def session_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize each session by max t_coil, mean p_sin and efficiency."""
    return df.groupby("source_file").agg({
        "t_coil_c": "max",
        "p_sin": "mean",
        "eff_norm": "mean",
        "soc": "first",
        "distance": "first"
    }).reset_index()


def plot_efficiency_histogram(df: pd.DataFrame, output_dir: Path):
    """Create and save histogram per (SoC, distance) group."""
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = df.groupby(["soc", "distance"])

    for (soc, dist), group in grouped:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=group, x="eff_norm", bins=50, kde=True)
        plt.title(f"Norm. Efficiency Distribution - SoC: {soc}%, Distance: {dist}")
        plt.xlabel("Norm. Efficiency")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / f"hist_eff_soc{soc}_dist{dist}.png")
        plt.close()



def plot_efficiency_boxplot(df: pd.DataFrame, output_dir: Path):
    """Create and save boxplot per (SoC, distance) group."""
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = df.groupby(["soc", "distance"])

    for (soc, dist), group in grouped:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=group, x="source_file", y="eff_norm")
        plt.title(f"Norm. Efficiency by File - SoC: {soc}%, Distance: {dist}")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(output_dir / f"boxplot_eff_soc{soc}_dist{dist}.png")
        plt.close()


def plot_kde_(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. KDE HEATMAP: efficiency vs distancve
    plt.figure(figsize=(10, 8))
    sns.kdeplot(
        data=df,
        x="eff_norm", y="p_sin",
        fill=True,
        cmap="magma",
        levels=100,
        thresh=0.02
    )
    plt.title("KDE Heatmap of Norm. Efficiency vs p_sin")
    plt.xlabel("Norm. Efficiency")
    plt.ylabel("p_sin")
    plt.tight_layout()
    plt.savefig(output_dir / "kde_heatmap_eff_vs_psin.png")
    plt.close()


def plot_scatter(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. SCATTER PLOT: SoC 20%
    df_20 = df[df["soc"] == 20]
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_20,
        x="p_sin", y="eff_norm",
        hue="distance",
        palette="viridis",
        alpha=0.6,
        edgecolor=None
    )
    plt.title("Norm. Efficiency vs Secondary Power (p_sin) - SoC 20%")
    plt.xlabel("p_sin [W]")
    plt.ylabel("Norm. Efficiency")
    plt.legend(title="Distance")
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_eff_vs_psin_soc20.png")
    plt.close()

    # 3. SCATTER PLOT: SoC 80%
    df_80 = df[df["soc"] == 80]
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_80,
        x="p_sin", y="eff_norm",
        hue="distance",
        palette="plasma",
        alpha=0.6,
        edgecolor=None
    )
    plt.title("Norm. Efficiency vs Secondary Power (p_sin) - SoC 80%")
    plt.xlabel("p_sin [W]")
    plt.ylabel("Norm. Efficiency")
    plt.legend(title="Distance")
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_eff_vs_psin_soc80.png")
    plt.close()


def plot_power_distributions(df: pd.DataFrame, output_dir: Path):
    """Create and save histogram of p_sin per (SoC, distance) group."""
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = df.groupby(["soc", "distance"])

    for (soc, dist), group in grouped:
        plt.figure(figsize=(12, 6))
        sns.histplot(data=group, x="p_sin", bins=30, kde=True)
        plt.title(f"Power Transferred (p_sin) - SoC: {soc}%, Distance: {dist}")
        plt.xlabel("Primary Power (p_sin) [W]")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / f"hist_p_sin_soc{soc}_dist{dist}.png")
        plt.close()


def plot_temp_vs_power_and_eff(df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Line Plot of t_coil_c vs time per session
    plt.figure(figsize=(12, 8))
    for name, group in df.groupby("source_file"):
        sns.lineplot(data=group, x="timestamp", y="t_coil_c", label=name, alpha=0.5)
    plt.title("t_coil_c over Time per Session")
    plt.xlabel("Time")
    plt.ylabel("Coil Temperature [°C]")
    plt.legend(title="Session", bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "lineplot_t_coil_vs_time.png")
    plt.close()

    # 2. KDE Heatmap: t_coil_c vs p_sin
    plt.figure(figsize=(10, 8))
    sns.kdeplot(data=df, x="t_coil_c", y="p_sin", fill=True, cmap="magma", levels=100)
    plt.title("2D KDE: Coil Temperature vs Primary Power")
    plt.xlabel("Coil Temperature [°C]")
    plt.ylabel("Primary Power [W]")
    plt.tight_layout()
    plt.savefig(output_dir / "kde_t_coil_vs_p_sin.png")
    plt.close()

    # 3. Summary Scatter: max t_coil vs mean efficiency
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=summary_df, x="t_coil_c", y="eff_norm",
                    hue="distance", style="soc", s=100)
    plt.title("Session Summary: Norm. Efficiency vs Max Coil Temperature")
    plt.xlabel("Max Coil Temperature [°C]")
    plt.ylabel("Mean Normalized Efficiency")
    plt.legend(title="Distance / SoC", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "summary_eff_vs_t_coil.png")
    plt.close()


def plot_max_temp_vs_power(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate session summaries
    summary = df.groupby("source_file").agg({
        "energy_loss": "mean",
        "p_sin": "mean",
        "t_coil_c": "max",
        "soc": "first",
        "distance": "first"
    }).reset_index()

    # 1. Scatter Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=summary,
        x="p_sin", y="t_coil_c",
        hue="distance", style="soc", s=100
    )
    plt.title("Max Coil Temperature vs Mean Primary Power (Per Session)")
    plt.xlabel("Mean Primary Power (p_sin) [W]")
    plt.ylabel("Max Coil Temperature (t_coil_c) [°C]")
    plt.legend(title="Distance / SoC", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "max_t_coil_vs_mean_p_sin.png")
    plt.close()

    # 2. Binned Boxplot
    bin_edges = list(range(0, int(summary["p_sin"].max()) + 100, 100))
    summary["power_bin"] = pd.cut(summary["p_sin"], bins=bin_edges)

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=summary, x="power_bin", y="t_coil_c", hue="distance")
    plt.title("Max Coil Temperature by Binned Primary Power (Per Session)")
    plt.xlabel("Mean Primary Power (p_sin) Bins [W]")
    plt.ylabel("Max Coil Temperature (t_coil_c) [°C]")
    plt.legend(title="Distance", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "boxplot_t_coil_vs_power_bins.png")
    plt.close()

     # 3. Summary Scatter: max t_coil vs mean efficiency
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=summary, x="t_coil_c", y="energy_loss",
                    hue="distance", style="soc", s=100)
    plt.title("Session Summary: Energy Loss vs Max Coil Temperature")
    plt.xlabel("Max Coil Temperature [°C]")
    plt.ylabel("Mean Energy Loss [W]")
    plt.legend(title="Distance / SoC", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "summary_energy_loss_vs_t_coil.png")
    plt.close()


def summarize_wpt_dataset(df):
    """
    Summarize key metrics from a preprocessed WPT dataset DataFrame.
    Assumes df already contains: p_sin, efficiency, t_coil_c, t_ambiant_c, soc, distance.
    """

    # Calculate session durations (sampling period is 5 seconds)
    session_durations = df.groupby("source_file").size() * 5 / 3600  # Convert to hours
    print("Session durations (hours):")
    for session, duration in session_durations.items():
        print(f"  {session}: {duration:.3f} hours")
    
    print(f"\nTotal duration across all sessions: {session_durations.sum():.3f} hours")
    print(f"Average session duration: {session_durations.mean():.3f} hours")
    print(f"Min session duration: {session_durations.min():.3f} hours")
    print(f"Max session duration: {session_durations.max():.3f} hours")

    print(f"\nEarliest record timestamp: {df['timestamp'].min()}")
    print(f"Latest record timestamp: {df['timestamp'].max()}")
    # Max instantaneous power transferred by group
    if {'soc', 'distance', 'p_sin'}.issubset(df.columns):
        print("\nMax instantaneous power (p_sin) by (SoC, distance):")
        max_power_groups = df.groupby(['soc', 'distance'])['p_sin'].max()
        for (soc, dist), pwr in max_power_groups.items():
            print(f"  SoC={soc}, Distance={dist}: {pwr:.2f} W")

    # Max coil temperature
    if 't_coil_c' in df.columns:
        print(f"\nMax coil temperature: {df['t_coil_c'].max():.2f} °C")

    # Min/Max ambient temperature
    if 't_ambiant_c' in df.columns:
        print(f"Min ambient temperature: {df['t_ambiant_c'].min():.2f} °C")
        print(f"Max ambient temperature: {df['t_ambiant_c'].max():.2f} °C")

    df["energy_Wh"] = df["p_sout"] * 5 / 3600

    total_energy_per_session = df.groupby("source_file")["energy_Wh"].sum().reset_index()

    print("\nTotal energy transferred per session (Wh):")
    print(total_energy_per_session)
    
    if 20 in df['soc'].unique():
        soc20 = df[df['soc'] == 20].copy()
        mean_eff_soc20 = soc20.groupby('distance')['efficiency'].mean()
        std_eff_soc20 = soc20.groupby('distance')['efficiency'].std()
        mean_psin_soc20 = soc20.groupby('distance')['p_sin'].mean()
        std_psin_soc20 = soc20.groupby('distance')['p_sin'].std()

        print("\nMean secondary input power by distance for SoC=20:")
        for dist, pwr in mean_psin_soc20.items():
            print(f"  SoC=20, Distance={dist}: {pwr:.4f} Standard Deviation={std_psin_soc20[dist]:.4f}")

        print("\nMean efficiency by distance for SoC=20:")
        for dist, eff in mean_eff_soc20.items():
            print(f"  SoC=20, Distance={dist}: {eff:.4f} Standard Deviation={std_eff_soc20[dist]:.4f}")

    # Mean efficiency by SoC and distance
    if 80 in df['soc'].unique():
        soc80 = df[df['soc'] == 80].copy()

        pmax = soc80['p_sin'].max(skipna=True)
        if pd.isna(pmax):
            print("\nSoC=80 has no p_sin values; skipping power-bin breakdown.")
        else:
            base_edges = np.array([0, 500, 1000, 2000, 3000], dtype=float)

            # keep only edges strictly below pmax, then append pmax as the final edge
            edges = base_edges[base_edges < pmax].tolist() + [float(pmax)]

            # ensure we have at least two edges; fall back to [0, pmax] (or a tiny span if pmax==0)
            if len(edges) < 2:
                edges = [0.0, pmax if pmax > 0 else (pmax + 1e-9)]

            # make absolutely sure edges are strictly increasing and unique
            edges = np.unique(edges)
            if np.any(np.diff(edges) <= 0):
                # last resort: jitter any equal edges by a tiny epsilon
                eps = 1e-9
                for i in range(1, len(edges)):
                    if edges[i] <= edges[i-1]:
                        edges[i] = edges[i-1] + eps


            edges = np.array(edges, dtype=float)
            soc80['power_bin'] = pd.cut(soc80['p_sin'], bins=list(edges), include_lowest=True)  # optional: left-closed bins [a, b))

            print("\nSoC=80 mean efficiency by distance & power bin:")
            mean_eff_soc80 = soc80.groupby(['distance', 'power_bin'], observed=True)['efficiency'].mean()
            for (dist, pbin), eff in mean_eff_soc80.items():
                print(f"  Distance={dist}, Power={pbin}: {eff:.4f}")


def main():
    df = load_all_csvs(DATA_FOLDER)
    df = preprocess_dataframe(df)
    summary_df = session_summary(df)
    print(f"Loaded {len(df)} rows from CSV files.")

    # Generate Plots based on configuration
    if not PLOT_FOLDER.exists():
        PLOT_FOLDER.mkdir(parents=True)

    if PLOT_HISTOGRAM:
        plot_efficiency_histogram(df, PLOT_FOLDER)

    if PLOT_BOXPLOT:
        plot_efficiency_boxplot(df, PLOT_FOLDER)

    if PLOT_KDE_HEATMAP:
        plot_kde_(df, PLOT_FOLDER)

    if PLOT_SCATTER_PLOTS:
        plot_scatter(df, PLOT_FOLDER)
    
    if PLOT_POWER_DISTRIBUTIONS:
        plot_power_distributions(df, PLOT_FOLDER)

    if PLOT_TEMP_AND_POWER:
        plot_temp_vs_power_and_eff(df, summary_df, PLOT_FOLDER)

    if PLOT_MAX_TEMP_AND_POWER:
        plot_max_temp_vs_power(df, PLOT_FOLDER)

    
    if SUMMARY:
        summarize_wpt_dataset(df)
        

    print("Plots generated successfully.")


if __name__ == "__main__":
    main()