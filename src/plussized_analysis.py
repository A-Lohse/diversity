import sys
from pathlib import Path

# Add the project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Imports ---
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
import statsmodels.formula.api as smf  # unused here, but kept if you need it elsewhere
from tqdm import trange

# -------------------------
# Helpers from your project
# -------------------------
from src.paths import (
    DATA_DIR, FIGURES_DIR, TABLES_DIR, ensure_directories_exist
)

from src.analysis import (
    load_core_datasets,
    create_master_dataset,
    prepare_shows_and_measurements_data,
)

# --------------------------------
# Data loading / preprocessing
# --------------------------------

def load_female_model_data():
    """Load core + profile-pic race info; keep females, drop unknown race, flag White."""
    core = load_core_datasets()
    df = create_master_dataset(core)

    skincolor_data = pd.read_csv(DATA_DIR / "model_info_from_profilepic.csv")
    df["model_name"] = df["filename"].apply(lambda x: x.split(".")[0])
    skincolor_data["model_name"] = skincolor_data["image_file"].apply(lambda x: x.split(".")[0])

    df = df.merge(skincolor_data, on="model_name")
    df = df.loc[df["face_detected"]]
    df = df.loc[df["predicted_race"] != "Unknown"]
    df["is_white"] = (df["predicted_race"] == "White").astype(int)

    return df[df["gender_consensus"] == "female"]

# EU → US dress mapping
EU_TO_US_DRESS = {
    30: 0, 32: 2, 34: 4, 36: 6, 38: 8, 40: 10, 42: 12,
    44: 14, 46: 16, 48: 18, 50: 20, 52: 22, 54: 24
}

def parse_eu_dress_to_us(val):
    """Convert EU dress size (or range) to US equivalent."""
    if pd.isnull(val):
        return None
    try:
        cleaned = re.sub(r"[^\d\-]", "", str(val))
        if '-' in cleaned:
            parts = cleaned.split('-')
            nums = []
            for p in parts:
                try:
                    nums.append(int(p))
                except ValueError:
                    continue
            if len(nums) == 2:
                avg = round(np.mean(nums))
                return EU_TO_US_DRESS.get(avg)
        else:
            return EU_TO_US_DRESS.get(int(cleaned))
    except Exception:
        return None

def preprocess_model_data(df):
    """
    Prepare model-level data:
    - Parse dress-eu -> US
    - Drop rows without valid sizes
    - Create plus_sized label (US >= 12)
    """
    df = df.copy()
    df['dress-us_clean'] = df['dress-eu'].apply(parse_eu_dress_to_us)
    df = df.dropna(subset=['dress-us_clean'])
    df['plus_sized'] = (df['dress-us_clean'] >= 12).astype(int)
    return df

def assign_year_bin(year):
    if 2011 <= year <= 2013:
        return "2011–2013"
    elif 2014 <= year <= 2016:
        return "2014–2016"
    elif 2017 <= year <= 2019:
        return "2017–2019"
    elif 2020 <= year <= 2024:
        return "2020–2024"
    else:
        return None

def enrich_shows_with_model_data(core_data, model_data):
    """
    Build show-level data, keep 2011–2024, map model attributes.
    """
    shows_data = prepare_shows_and_measurements_data(core_data)
    shows_data = shows_data[(shows_data["year"] >= 2011) & (shows_data["year"] < 2025)]
    shows_data['year_bin'] = shows_data['year'].apply(assign_year_bin)

    shows_data = shows_data[shows_data["model_id"].isin(model_data["model_id"].unique())]

    # map attributes from model_id
    for col in ["plus_sized", "is_white", "predicted_race"]:
        shows_data[col] = shows_data["model_id"].map(dict(zip(model_data["model_id"], model_data[col])))

    shows_data["plus_sized"] = shows_data["plus_sized"].astype(int)
    shows_data["is_white"] = shows_data["is_white"].astype(int)
    return shows_data

# ---------- Plotting helpers (integrated) ----------

def _agg_share(d, col):
    return (
        d.groupby("year")[col]
         .agg(["mean", "count"])
         .rename(columns={"mean": "share", "count": "n"})
         .reset_index()
         .sort_values("year")
    )

def _exp_growth(t, a, b):
    return a * np.exp(b * t)

def _fit_exp_with_stats(years, y):
    # Fit y = a * exp(b * (year - min_year))
    t = years - years.min()
    y = np.clip(y, 1e-9, 1-1e-9)
    popt, pcov = curve_fit(_exp_growth, t, y, maxfev=10000)
    a, b = popt
    a_se, b_se = np.sqrt(np.diag(pcov))
    yhat = _exp_growth(t, a, b)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    gpy = (np.exp(b) - 1) * 100
    dbl = np.log(2)/b if b > 0 else np.nan
    return dict(a=a, b=b, a_se=a_se, b_se=b_se, R2=R2, gpy=gpy, dbl=dbl, pcov=pcov)

def _pred_ci_band(years_fit, years_min, params):
    # Delta-method CI for fitted mean curve
    a, b, pcov = params["a"], params["b"], params["pcov"]
    t = years_fit - years_min
    yhat = _exp_growth(t, a, b)
    eps = 1e-12
    dy_da = yhat / max(a, eps)    # dy/da = exp(b t) = yhat/a
    dy_db = t * yhat              # dy/db = a t exp(b t) = t*yhat
    J = np.vstack([dy_da, dy_db]).T
    var_pred = np.einsum("ni,ij,nj->n", J, pcov, J)
    se = np.sqrt(np.clip(var_pred, 0, None))
    lo = np.clip(yhat - 1.96*se, 0, 1)
    hi = np.clip(yhat + 1.96*se, 0, 1)
    return yhat, lo, hi

def _fmt_box(name, f):
    if np.isfinite(f["dbl"]):
        return (
            f"{name}:\n"
            f"  a={f['a']:.4f}±{f['a_se']:.4f}\n"
            f"  b={f['b']:.4f}±{f['b_se']:.4f}\n"
            f"  R²={f['R2']:.3f}\n"
            f"  growth≈{f['gpy']:.2f}%/yr\n"
            f"  double≈{f['dbl']:.2f} yrs"
        )
    else:
        return (
            f"{name}:\n"
            f"  a={f['a']:.4f}±{f['a_se']:.4f}\n"
            f"  b={f['b']:.4f}±{f['b_se']:.4f}\n"
            f"  R²={f['R2']:.3f}\n"
            f"  growth≈{f['gpy']:.2f}%/yr\n"
            f"  double=—"
        )

def _plot_exponential_grid(shows_data, savepath=None, tight_layout=True):
    """Create the 2x2 grid:
       - Left column (spans rows): joint Black vs White (plus_sized)
       - Top-right: share Non-White (exp fit)
       - Bottom-right: share Plus-Sized (exp fit)
    """
    # Prep
    df = shows_data.copy()
    df["is_white"] = df["is_white"].astype(bool)
    df["plus_sized"] = df["plus_sized"].astype(bool)
    df["is_nonwhite"] = ~df["is_white"]

    nonwhite = _agg_share(df, "is_nonwhite")
    plus = _agg_share(df, "plus_sized")

    # Joint Black/White plus-sized
    agg_joint = (
        df.groupby(["year", "is_white"])["plus_sized"]
          .agg(["mean", "count"])
          .reset_index()
          .rename(columns={"mean": "share", "count": "n"})
    )
    pivoted = agg_joint.pivot(index="year", columns="is_white", values="share").sort_index()
    pivoted.columns = ["Black", "White"]  # False -> Black, True -> White
    years_joint = pivoted.index.values.astype(float)
    fits_joint = {grp: _fit_exp_with_stats(years_joint, pivoted[grp].values) for grp in ["Black", "White"]}
    t_fit_joint = np.linspace(0, years_joint.max()-years_joint.min(), 400)
    years_fit_joint = years_joint.min() + t_fit_joint
    bands_joint = {lbl: _pred_ci_band(years_fit_joint, years_joint.min(), fits_joint[lbl]) for lbl in ["Black", "White"]}

    # Non-White
    yrs_nw = nonwhite["year"].to_numpy(float)
    fit_nw = _fit_exp_with_stats(yrs_nw, nonwhite["share"].to_numpy())
    t_fit_nw = np.linspace(0, yrs_nw.max()-yrs_nw.min(), 400)
    years_fit_nw = yrs_nw.min() + t_fit_nw
    yhat_nw, ylo_nw, yhi_nw = _pred_ci_band(years_fit_nw, yrs_nw.min(), fit_nw)

    # Plus-Sized
    yrs_ps = plus["year"].to_numpy(float)
    fit_ps = _fit_exp_with_stats(yrs_ps, plus["share"].to_numpy())
    t_fit_ps = np.linspace(0, yrs_ps.max()-yrs_ps.min(), 400)
    years_fit_ps = yrs_ps.min() + t_fit_ps
    yhat_ps, ylo_ps, yhi_ps = _pred_ci_band(years_fit_ps, yrs_ps.min(), fit_ps)

    # Layout
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2, 1])
    ax_joint = fig.add_subplot(gs[:, 0])
    ax_nw = fig.add_subplot(gs[0, 1])
    ax_ps = fig.add_subplot(gs[1, 1])

    # Consistent colors
    color_black = "tab:blue"
    color_white = "tab:orange"

    # Joint plot (markers only for data; dashed fit; shaded CI)
    for is_w, label, color in [(False, "Non-white models", color_black),
                           (True, "White models", color_white)]:
        sub = agg_joint[agg_joint["is_white"] == is_w].sort_values("year")
        ax_joint.plot(sub["year"], sub["share"], "o", linestyle="None", color=color, label=f"{label} data")


    for label, color, pretty in [("Black", color_black, "Non-white Models"),
                             ("White", color_white, "White Models")]:
        yhat, ylo, yhi = bands_joint[label]
        ax_joint.plot(years_fit_joint, yhat, "--", lw=2, color=color, label=f"{pretty} (Exp. fit)")
        ax_joint.fill_between(years_fit_joint, ylo, yhi, color=color, alpha=0.15)

    ax_joint.set_title("Plus-Size Share by Race")
    ax_joint.set_xlabel("Year")
    ax_joint.set_ylabel("Share Plus-Sized")
    ax_joint.grid(alpha=0.3)
    ax_joint.legend(loc="upper left")

    txt_joint = _fmt_box("Non-white Models", fits_joint["Black"]) + "\n\n\n" + _fmt_box("White Models", fits_joint["White"])
    ax_joint.text(0.02, 0.15, txt_joint,
                  transform=ax_joint.transAxes, fontsize=9,
                  va="bottom", ha="left", family="monospace",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85))

    # Non-White subplot
    ax_nw.plot(nonwhite["year"], nonwhite["share"], "o", linestyle="None", color=color_black, label="Non-White Models")
    ax_nw.plot(years_fit_nw, yhat_nw, "--", lw=2, color=color_black, label="Exp. fit")
    ax_nw.fill_between(years_fit_nw, ylo_nw, yhi_nw, color=color_black, alpha=0.15)
    ax_nw.set_title("Non-White Models")
    ax_nw.set_xlabel("Year"); ax_nw.set_ylabel("Share")
    ax_nw.grid(alpha=0.3); ax_nw.legend(loc="upper left")
    ax_nw.text(0.02, 0.15, _fmt_box("Non-White Models", fit_nw),
               transform=ax_nw.transAxes, fontsize=9, va="bottom", ha="left", family="monospace",
               bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85))

    # Plus-Sized subplot
    ax_ps.plot(plus["year"], plus["share"], "o", linestyle="None", color=color_white, label="Plus-sized Models")
    ax_ps.plot(years_fit_ps, yhat_ps, "--", lw=2, color=color_white, label="Exp. Fit")
    ax_ps.fill_between(years_fit_ps, ylo_ps, yhi_ps, color=color_white, alpha=0.15)
    ax_ps.set_title("Share Plus-Sized Models")
    ax_ps.set_xlabel("Year"); ax_ps.set_ylabel("Share")
    ax_ps.grid(alpha=0.3); ax_ps.legend(loc="upper left")
    ax_ps.text(0.02, 0.15, _fmt_box("Plus-sized Models", fit_ps),
               transform=ax_ps.transAxes, fontsize=9, va="bottom", ha="left", family="monospace",
               bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.85))

    if tight_layout:
        plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    # Return fits for table building
    return {
        "Non-white models": fit_nw,
        "White models":      fits_joint["White"],
        "Plus-size models":  fit_ps,
    }



def make_exp_fit_table(fits: dict, outpath: Path, caption="Exponential growth model fits for model shares"):
    rows = []
    for label, f in fits.items():
        def fmt_num_se(val, se):
            return f"${val:.4f}\\ ({se:.4f})$"
        rows.append({
            "Group": label,
            "$a$": fmt_num_se(f["a"], f["a_se"]),
            "$b$": fmt_num_se(f["b"], f["b_se"]),
            "$R^2$": f"${f['R2']:.3f}$",
            "Growth/yr": f"${f['gpy']:.2f}\\%$",
            "Doubling time": f"${f['dbl']:.2f}$" if np.isfinite(f["dbl"]) else "$\\text{—}$",
        })
    df = pd.DataFrame(rows, columns=["Group", "$a$", "$b$", "$R^2$", "Growth/yr", "Doubling time"])

    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        "\\label{tab:expfits}\n"
        "\\begin{tabular}{llllll}\n"
        "\\toprule\n"
        + " & ".join(df.columns) + " \\\\\n"
        "\\midrule\n"
        + "\n".join(" & ".join(map(str, row)) + " \\\\" for row in df.values)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    outpath.write_text(latex)
    print(f"Wrote {outpath}")
    return df


# ----------- MAIN -----------

def main():
    # Setup
    ensure_directories_exist()

    # Load & preprocess model-level data (female only)
    print("Loading & preprocessing model data...")
    model_data = load_female_model_data()
    model_data = preprocess_model_data(model_data)

    # Build show-level dataset with mapped attributes
    print("Preparing show-level dataset...")
    core_data = load_core_datasets()
    shows_data = enrich_shows_with_model_data(core_data, model_data)

    # --- Plot & save ---
    fig_path = FIGURES_DIR / "exponential_grid_joint_nonwhite_plussized.png"
    print(f"Creating plots → {fig_path}")
    fits_summary = _plot_exponential_grid(shows_data, savepath=fig_path)

    # (Optional) All-races plot — uncomment if you want the per-race figure & fits
    # fig_path_allraces = FIGURES_DIR / "share_by_race_all_groups.png"
    # print(f"Creating plots → {fig_path_allraces}")
    # fits_all_races = _plot_all_races(shows_data, savepath=fig_path_allraces)
    # fits_for_table = {**fits_summary, **{f"Race: {k}": v for k, v in fits_all_races.items()}}

    # If you’re not plotting all races, just use fits_summary:
    fits_for_table = fits_summary

    # Save table
    make_exp_fit_table(
        fits_for_table,
        TABLES_DIR / "exp_growth_models.tex",
        caption="Exponential growth model fits for model shares"
    )

    # Save table
    make_exp_fit_table(fits_summary, TABLES_DIR / "exp_growth_models.tex",
                    caption="Exponential growth model fits for model shares")


    print("Done.")

if __name__ == "__main__":
    _ = main()
