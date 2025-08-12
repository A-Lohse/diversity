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
import statsmodels.formula.api as smf
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

# --------------------------------
# Uncertainty via counts (shares)
# --------------------------------

def _wilson_ci(k, n, z=1.96):
    """Wilson score interval for binomial proportion; returns (low, high) in [0,1]."""
    if n is None or n == 0:
        return (np.nan, np.nan)
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2.0 * n)
    adj = z * np.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n)
    low = (center - adj) / denom
    high = (center + adj) / denom
    return (max(0.0, low), min(1.0, high))

def shares_with_binomial_ci(d: pd.DataFrame, years, z=1.96) -> pd.DataFrame:
    """
    Compute % plus-sized with Wilson CIs by year: overall, white, non-white.
    Returns DF indexed by year with columns:
      pct_overall, pct_overall_ci_low, pct_overall_ci_high,
      pct_white,   pct_white_ci_low,   pct_white_ci_high,
      pct_nonwhite, pct_nonwhite_ci_low, pct_nonwhite_ci_high
    Percentages are 0–100.
    """
    # overall
    g = d.groupby('year').agg(total=('model_id', 'count'),
                              plus=('plus_sized', 'sum'))
    g['pct_overall'] = (g['plus'] / g['total']) * 100.0
    lows, highs = [], []
    for k, n in zip(g['plus'].fillna(0).astype(float), g['total'].fillna(0).astype(float)):
        lo, hi = _wilson_ci(k, n, z=z)
        lows.append(lo * 100.0)
        highs.append(hi * 100.0)
    g['pct_overall_ci_low'] = lows
    g['pct_overall_ci_high'] = highs

    # white
    w = d[d['is_white'] == 1].groupby('year').agg(tw=('model_id', 'count'),
                                                  pw=('plus_sized', 'sum'))
    w['pct_white'] = (w['pw'] / w['tw']) * 100.0
    lows, highs = [], []
    for k, n in zip(w['pw'].fillna(0).astype(float), w['tw'].fillna(0).astype(float)):
        lo, hi = _wilson_ci(k, n, z=z)
        lows.append(lo * 100.0)
        highs.append(hi * 100.0)
    w['pct_white_ci_low'] = lows
    w['pct_white_ci_high'] = highs

    # non-white
    nw = d[d['is_white'] == 0].groupby('year').agg(tn=('model_id', 'count'),
                                                   pn=('plus_sized', 'sum'))
    nw['pct_nonwhite'] = (nw['pn'] / nw['tn']) * 100.0
    lows, highs = [], []
    for k, n in zip(nw['pn'].fillna(0).astype(float), nw['tn'].fillna(0).astype(float)):
        lo, hi = _wilson_ci(k, n, z=z)
        lows.append(lo * 100.0)
        highs.append(hi * 100.0)
    nw['pct_nonwhite_ci_low'] = lows
    nw['pct_nonwhite_ci_high'] = highs

    out = (g[['pct_overall', 'pct_overall_ci_low', 'pct_overall_ci_high']]
           .join(w[['pct_white', 'pct_white_ci_low', 'pct_white_ci_high']], how='outer')
           .join(nw[['pct_nonwhite', 'pct_nonwhite_ci_low', 'pct_nonwhite_ci_high']], how='outer'))
    return out.reindex(years)

# --------------------------------
# Odds ratio helpers (model + OR)
# --------------------------------

def _extract_nonwhite_year_coefs(fitted_model):
    """Return (b0, b1) for non-white main effect and its interaction with year."""
    names = pd.Index(fitted_model.params.index)
    base = names[names.str.contains(r"C\(is_white.*\)\[T\.0\]$", regex=True)]
    inter = names[names.str.contains(r"C\(is_white.*\)\[T\.0\]:year$", regex=True)]
    if len(base) != 1 or len(inter) != 1:
        raise ValueError(f"Could not identify coefficient names. Found base={list(base)}, inter={list(inter)}")
    return fitted_model.params[base[0]], fitted_model.params[inter[0]]

def _or_by_year_from_model(model, years):
    b0, b1 = _extract_nonwhite_year_coefs(model)
    return pd.Series({y: np.exp(b0 + b1*y) for y in years})

def _or_by_year_from_df(df, years, maxiter=100, use_regularized_fallback=True):
    """
    Fit the dynamic model on df and return OR(non-white vs white) by year.
    No robust SEs needed here — we're using bootstrap percentiles for CIs.
    """
    try:
        m = smf.logit(
            formula="plus_sized ~ C(is_white, Treatment(reference=1)) * year",
            data=df
        ).fit(maxiter=maxiter, disp=0)
    except Exception:
        if not use_regularized_fallback:
            raise
        # mild fallback if separation/convergence issues arise
        m = smf.logit(
            formula="plus_sized ~ C(is_white, Treatment(reference=1)) * year",
            data=df
        ).fit_regularized(method='l1', maxiter=maxiter, alpha=1e-6, disp=0)
    return _or_by_year_from_model(m, years)

# --------------------------------
# Cluster bootstrap for OR with caching
# --------------------------------

def _sample_within_year_cluster(df, years, rng):
    """
    Within-year cluster bootstrap on model_id; returns resampled DataFrame.
    Resamples model_id clusters with replacement within each year and
    re-concatenates their rows.
    """
    parts = []
    for y in years:
        block = df.loc[df['year'] == y]
        clust = block['model_id'].dropna().unique()
        if len(clust) == 0:
            return None
        sampled = rng.choice(clust, size=len(clust), replace=True)
        parts.append(pd.concat([block.loc[block['model_id'] == cid] for cid in sampled], axis=0))
    return pd.concat(parts, axis=0)

def bootstrap_or_only_cached(
    data: pd.DataFrame,
    years: np.ndarray,
    n_boot: int = 1000,
    random_state: int = 42,
    maxiter: int = 100,
    cache_dir: Path | str | None = None,
    use_cache: bool = True,
    save_cache: bool = True,
    save_draws: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Cluster bootstrap (within-year over model_id) to get percentile CIs for OR(t),
    with optional caching of results and raw draws.

    Returns DataFrame with columns:
      year, odds_ratio, ci_lower, ci_upper, n_boot_kept
    """
    cache_path = Path(cache_dir) if cache_dir is not None else None
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
        or_df_path = cache_path / "or_df.csv"
        or_draws_path = cache_path / "boot_draws_or.npz"

    # Try load cache
    if use_cache and cache_path is not None and or_df_path.exists():
        if verbose:
            print(f"[bootstrap_or_only_cached] Loading cached OR from {or_df_path}")
        or_df = pd.read_csv(or_df_path)
        cached_years = or_df['year'].to_numpy()
        if np.array_equal(np.sort(cached_years), np.sort(years)):
            return or_df
        else:
            if verbose:
                print("[bootstrap_or_only_cached] Cached years differ from current years; recomputing.")

    rng = np.random.default_rng(random_state)

    # point estimate on full data
    or_point = _or_by_year_from_df(data, years, maxiter=maxiter)

    # bootstrap replicates
    or_tables = []
    kept = 0
    for _ in trange(n_boot, desc="Bootstrapping OR", unit="rep", disable=not verbose):
        sample_df = _sample_within_year_cluster(data, years, rng)
        if sample_df is None:
            continue
        try:
            or_tables.append(_or_by_year_from_df(sample_df, years, maxiter))
            kept += 1
        except Exception:
            continue

    if kept == 0:
        raise RuntimeError("All bootstrap replicates failed; check data sufficiency per year.")

    # stack and percentile CIs
    or_stack = np.stack([ser.reindex(years).to_numpy(dtype=float) for ser in or_tables])  # (kept, len(years))
    or_lo = np.nanpercentile(or_stack, 2.5, axis=0)
    or_hi = np.nanpercentile(or_stack, 97.5, axis=0)

    or_df = pd.DataFrame({
        'year': years,
        'odds_ratio': or_point.values,
        'ci_lower': or_lo,
        'ci_upper': or_hi,
        'n_boot_kept': kept
    })

    # Save cache
    if save_cache and cache_path is not None:
        if verbose:
            print(f"[bootstrap_or_only_cached] Saving OR results to {cache_path}")
        or_df.to_csv(or_df_path, index=False)
        if save_draws:
            np.savez_compressed(
                or_draws_path,
                draws=or_stack,
                years=years,
                meta=np.array([('n_boot_requested', n_boot),
                               ('n_boot_kept', kept),
                               ('random_state', random_state)], dtype=object)
            )

    return or_df

# -----------------
# Plotting
# -----------------

def plot_race_shares_and_effect_panel(df_share, effect_df, save_path=None, scale='or'):
    """
    Create a 2-panel figure:
      Left: % Plus-Sized (Overall, White, Non-White) with 95% Wilson CI bands
      Right: Effect over time; by default, odds ratio with bootstrap percentile CI

    effect_df expects:
        - scale='or'   : ['year','odds_ratio','ci_lower','ci_upper']
        - scale='logit': ['year','logit','ci_lower','ci_upper']  (not used here)
    """

    def _get_series(df, names):
        for n in names:
            if n in df.columns:
                return df[n]
        return None

    years = df_share.index

    # Resolve share columns
    pct_overall = _get_series(df_share, ["pct_overall", "overall_pct"])
    lo_overall  = _get_series(df_share, ["pct_overall_ci_low", "overall_ci_low"])
    hi_overall  = _get_series(df_share, ["pct_overall_ci_high", "overall_ci_high"])

    pct_white = _get_series(df_share, ["pct_white", "white_pct"])
    lo_white  = _get_series(df_share, ["pct_white_ci_low", "white_ci_low"])
    hi_white  = _get_series(df_share, ["pct_white_ci_high", "white_ci_high"])

    pct_nonwhite = _get_series(df_share, ["pct_nonwhite", "nonwhite_pct"])
    lo_nonwhite  = _get_series(df_share, ["pct_nonwhite_ci_low", "nonwhite_ci_low"])
    hi_nonwhite  = _get_series(df_share, ["pct_nonwhite_ci_high", "nonwhite_ci_high"])

    # --- PLOT ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=False)

    # LEFT PANEL: shares
    ax = axes[0]
    if pct_overall is not None:
        ax.plot(years, pct_overall, marker='o', label='% Plus-Sized (Overall)')
        if lo_overall is not None and hi_overall is not None:
            ax.fill_between(years, lo_overall, hi_overall, alpha=0.2)
    if pct_white is not None:
        ax.plot(years, pct_white, marker='s', label='% Plus-Sized (White)')
        if lo_white is not None and hi_white is not None:
            ax.fill_between(years, lo_white, hi_white, alpha=0.2)
    if pct_nonwhite is not None:
        ax.plot(years, pct_nonwhite, marker='x', linestyle='--', label='% Plus-Sized (Non-White)')
        if lo_nonwhite is not None and hi_nonwhite is not None:
            ax.fill_between(years, lo_nonwhite, hi_nonwhite, alpha=0.2)

    ax.set_title("% Plus-Sized Models Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Percentage (%)")
    ax.legend()
    ax.grid(True, axis='y', alpha=0.4)

    # RIGHT PANEL: odds ratio + CI band
    ax = axes[1]
    ax.plot(effect_df['year'], effect_df['odds_ratio'], marker='o',
            label='Odds Ratio (Non-white vs White)')
    if {'ci_lower', 'ci_upper'}.issubset(effect_df.columns):
        ax.fill_between(effect_df['year'], effect_df['ci_lower'], effect_df['ci_upper'],
                        alpha=0.2, label='95% CI (bootstrap)')
    ax.axhline(1.0, linestyle='--')
    ax.set_ylabel("Odds Ratio")
    ax.set_title("Odds Ratio Over Time")

    ax.set_xlabel("Year")
    ax.legend()
    ax.grid(True, axis='y', alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

# -----------------
# Regression tables
# -----------------

def save_logit_tables(
    data,
    tables_dir,
    cluster_col="model_id",
    maxiter=100,
    ref_level=1,
    static_name="logit_static.tex",
    dynamic_name="logit_dynamic.tex",
    verbose=True
):
    """
    Fit two clustered logit models and save LaTeX summaries:
      - Static:  plus_sized ~ C(is_white, Treatment(reference=ref_level))
      - Dynamic: plus_sized ~ C(is_white, Treatment(reference=ref_level)) * year
    """
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Static
    model_static = smf.logit(
        formula=f"plus_sized ~ C(is_white, Treatment(reference={ref_level}))",
        data=data
    ).fit(maxiter=maxiter, disp=0, cov_type='cluster',
          cov_kwds={'groups': data[cluster_col]})

    # Dynamic (interaction with year)
    model_dynamic = smf.logit(
        formula=f"plus_sized ~ C(is_white, Treatment(reference={ref_level})) * year",
        data=data
    ).fit(maxiter=maxiter, disp=0, cov_type='cluster',
          cov_kwds={'groups': data[cluster_col]})

    # Save LaTeX
    static_path = Path(tables_dir) / static_name
    dynamic_path = Path(tables_dir) / dynamic_name
    with open(static_path, "w", encoding="utf-8") as f:
        f.write(model_static.summary().as_latex())
    with open(dynamic_path, "w", encoding="utf-8") as f:
        f.write(model_dynamic.summary().as_latex())

    if verbose:
        print(f"Saved: {static_path}")
        print(f"Saved: {dynamic_path}")

    return model_static, model_dynamic

# -----------
# MAIN
# -----------

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

    # Regression tables (cluster-robust, saved to LaTeX)
    print("Saving regression tables...")
    _, _ = save_logit_tables(
        data=shows_data,
        tables_dir=TABLES_DIR,
        cluster_col="model_id",
        maxiter=100,
        ref_level=1,
        static_name="logit_static.tex",
        dynamic_name="logit_dynamic.tex",
        verbose=True
    )

    # Years to evaluate
    years = np.sort(shows_data['year'].unique())

    # Shares with Wilson CIs (from counts)
    df_share = shares_with_binomial_ci(shows_data, years, z=1.96)

    # Odds ratio over time with bootstrap percentile CIs + caching
    print("Bootstrapping odds ratios (with cache)...")
    cache_dir = DATA_DIR / "bootstrap_cache/race"
    or_df = bootstrap_or_only_cached(
        data=shows_data,
        years=years,
        n_boot=1000,
        random_state=42,
        maxiter=100,
        cache_dir=cache_dir,
        use_cache=False,      # set False to force recompute
        save_cache=True,
        save_draws=True,
        verbose=True
    )

    # Plot panel (shares + OR curve)
    print("Saving panel plot...")
    plot_race_shares_and_effect_panel(
        df_share=df_share,
        effect_df=or_df,
        save_path=FIGURES_DIR / "race_panel_bootstrap_OR.png",
        scale='or'
    )

    print("Done.")

if __name__ == "__main__":
    main()
