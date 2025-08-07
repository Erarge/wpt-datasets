import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


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
    df["efficiency"] = df["p_sin"] / df["p_pri"]
    return df


def plot_efficiency_distributions(df: pd.DataFrame, output_dir: Path):
    """Create and save histogram and boxplot per (SoC, distance) group."""
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = df.groupby(["soc", "distance"])

    for (soc, dist), group in grouped:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=group, x="efficiency", bins=50, kde=True)
        plt.title(f"Efficiency Distribution - SoC: {soc}%, Distance: {dist}")
        plt.xlabel("Efficiency")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / f"hist_eff_soc{soc}_dist{dist}.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=group, x="source_file", y="efficiency")
        plt.title(f"Efficiency by File - SoC: {soc}%, Distance: {dist}")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(output_dir / f"boxplot_eff_soc{soc}_dist{dist}.png")
        plt.close()


def plot_power_distributions(df: pd.DataFrame, output_dir: Path):
    """Create and save histogram of p_sin per (SoC, distance) group."""
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = df.groupby(["soc", "distance"])

    for (soc, dist), group in grouped:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=group, x="p_sin", bins=50, kde=True)
        plt.title(f"Power Transferred (p_sin) - SoC: {soc}%, Distance: {dist}")
        plt.xlabel("Primary Power (p_sin) [W]")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / f"hist_p_sin_soc{soc}_dist{dist}.png")
        plt.close()


def main():
    input_dir = Path("C:\\Users\\MONSTER\\dev\\projects\\wpt\\wpt-datasets\\data")
    output_dir = Path("C:\\Users\\MONSTER\\dev\\projects\\wpt\\wpt-datasets\\plots")

    df = load_all_csvs(input_dir)
    df = preprocess_dataframe(df)
    plot_efficiency_distributions(df, output_dir)
    plot_power_distributions(df, output_dir)

    print("Plots generated successfully.")


if __name__ == "__main__":
    main()
