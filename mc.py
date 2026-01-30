import pandas as pd
import numpy as np
import re
from collections import defaultdict

DATA_PATH = "2026_MCM_Problem_C_Data.csv"

# -------------------------
# Helpers: parsing + season slicing
# -------------------------

# loads data set
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

# returns data from problemdata
def season_df(df, season_num: int):
    s = df[df["season"] == season_num].copy()
    # Make sure numeric columns are numeric
    for c in s.columns:
        if c.startswith("week") and "_judge" in c:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    return s.reset_index(drop=True)

def week_judge_cols(week: int):
    return [f"week{week}_judge{i}_score" for i in range(1, 5)]  # judge1..judge4

def judge_total_for_week(sdf, week: int):
    cols = [c for c in week_judge_cols(week) if c in sdf.columns]
    # Sum across judges; NaNs ignored
    return sdf[cols].sum(axis=1, skipna=True)

def active_mask(sdf, week: int):
    # Contestants "active" in week have positive judge total that week
    # (after elimination they become 0s)
    totals = judge_total_for_week(sdf, week)
    return totals > 0

def eliminated_in_week(sdf, week: int):
    # Uses the 'results' string to find who is eliminated in that week
    pat = rf"Eliminated Week {week}\b"
    return set(sdf.loc[sdf["results"].astype(str).str.contains(pat, regex=True, na=False),
                       "celebrity_name"].tolist())

# -------------------------
# Monte Carlo for one week (percent-based)
# -------------------------

def sample_fan_shares(n, alpha=1.0, rng=None):
    """
    Sample a random fan-share vector of length n that sums to 1.
    alpha=1 => uniform on simplex; alpha<1 spikier; alpha>1 smoother.
    """
    if rng is None:
        rng = np.random.default_rng()
    a = np.full(n, alpha, dtype=float)
    return rng.dirichlet(a)

def mc_week_percent(sdf, week: int, n_keep=2000, max_tries=2_000_000, alpha=1.0, seed=0):
    """
    Returns a list of kept fan-share samples, each aligned to active contestants list order.
    Only keeps samples that reproduce the observed elimination set for that week
    under percent-combining rule: combined = judge_share + fan_share.
    """
    rng = np.random.default_rng(seed)

    obs_elim = eliminated_in_week(sdf, week)
    if len(obs_elim) == 0:
        return None, None, None  # no elimination info this week

    act = sdf.loc[active_mask(sdf, week), ["celebrity_name"]].copy()
    names = act["celebrity_name"].tolist()

    # If someone is listed as eliminated but not active by scores, data inconsistency -> skip safely
    if not obs_elim.issubset(set(names)):
        return None, names, {"status": "skip", "reason": "observed eliminated not active by scores"}

    # Determine how many eliminations happened this week
    k = len(obs_elim)

    # Judge shares
    totals = judge_total_for_week(sdf.loc[active_mask(sdf, week)], week).to_numpy()
    judge_sum = totals.sum()
    if judge_sum <= 0:
        return None, names, {"status": "skip", "reason": "no judge points this week"}

    judge_share = totals / judge_sum  # sums to 1

    kept = []
    tries = 0
    while len(kept) < n_keep and tries < max_tries:
        tries += 1
        fan_share = sample_fan_shares(len(names), alpha=alpha, rng=rng)
        combined = judge_share + fan_share

        # Find bottom-k by combined score
        bottom_idx = np.argsort(combined)[:k]
        pred_elim = set(names[i] for i in bottom_idx)

        if pred_elim == obs_elim:
            kept.append(fan_share)

    info = {
        "status": "ok" if len(kept) > 0 else "fail",
        "week": week,
        "active_n": len(names),
        "elim_k": k,
        "tries": tries,
        "kept": len(kept),
        "accept_rate": (len(kept) / tries) if tries else 0.0
    }
    return np.array(kept), names, info

# -------------------------
# Monte Carlo for a whole season
# -------------------------

def mc_season_percent(season_num: int, n_keep_per_week=2000, alpha=1.0, seed=0):
    df = load_data()
    sdf = season_df(df, season_num)

    week_results = {}
    for week in range(1, 12):  # dataset has weeks 1..11
        samples, names, info = mc_week_percent(
            sdf, week,
            n_keep=n_keep_per_week,
            alpha=alpha,
            seed=seed + week
        )
        week_results[week] = {"samples": samples, "names": names, "info": info}
    return week_results

def summarize_week_samples(samples, names):
    """
    Returns mean and 10-90% interval for each contestant's fan share that week.
    """
    if samples is None or len(samples) == 0:
        return None
    mean = samples.mean(axis=0)
    lo = np.quantile(samples, 0.10, axis=0)
    hi = np.quantile(samples, 0.90, axis=0)
    out = pd.DataFrame({"celebrity_name": names, "mean_fan_share": mean, "p10": lo, "p90": hi})
    return out.sort_values("mean_fan_share", ascending=False).reset_index(drop=True)

# -------------------------
# Monte Carlo for final week (percent-based)
# -------------------------

def finalists_for_season(sdf):
    # finalists are those with placement 1..k where k is number of finalists in that season
    # simplest: take the minimum placements present (usually 1,2,3)
    fin = sdf[sdf["placement"].isin([1,2,3])].copy()
    # If some seasons have 4 finalists, include 4 too if it exists
    if (sdf["placement"] == 4).any() and len(fin) < 4:
        fin = sdf[sdf["placement"].isin([1,2,3,4])].copy()
    return fin

def final_week_index(sdf):
    # Determine the last week that has any nonzero judge totals among finalists
    fin = finalists_for_season(sdf)
    last_week = None
    for w in range(1, 12):
        totals = judge_total_for_week(fin, w)
        if (totals > 0).any():
            last_week = w
    return last_week

def mc_finals_percent(sdf, n_keep=5000, max_tries=2_000_000, alpha=1.0, seed=0):
    rng = np.random.default_rng(seed)

    fin = finalists_for_season(sdf)
    if fin.empty:
        return None, None, {"status": "skip", "reason": "no finalists detected"}

    w = final_week_index(sdf)
    if w is None:
        return None, None, {"status": "skip", "reason": "cannot find final week"}

    names = fin["celebrity_name"].tolist()
    placements = fin["placement"].to_numpy()

    # Judge shares among finalists in final week
    totals = judge_total_for_week(fin, w).to_numpy()
    if totals.sum() <= 0:
        return None, names, {"status": "skip", "reason": "no judge points in finals"}

    judge_share = totals / totals.sum()

    # Observed order: placement 1 is best.
    # We want combined score ordering descending to match placement ascending.
    obs_order = np.argsort(placements)  # indices in winner->... order

    kept = []
    tries = 0
    while len(kept) < n_keep and tries < max_tries:
        tries += 1
        fan_share = rng.dirichlet(np.full(len(names), alpha))
        combined = judge_share + fan_share

        pred_order = np.argsort(-combined)  # descending combined
        if np.array_equal(pred_order, obs_order):
            kept.append(fan_share)

    info = {
        "status": "ok" if len(kept) > 0 else "fail",
        "final_week": w,
        "finalists_n": len(names),
        "tries": tries,
        "kept": len(kept),
        "accept_rate": (len(kept) / tries) if tries else 0.0
    }
    return np.array(kept), names, info

def summarize_finals(samples, names):
    if samples is None or len(samples) == 0:
        return None
    mean = samples.mean(axis=0)
    lo = np.quantile(samples, 0.10, axis=0)
    hi = np.quantile(samples, 0.90, axis=0)
    return (pd.DataFrame({"celebrity_name": names, "mean_fan_share": mean, "p10": lo, "p90": hi})
              .sort_values("mean_fan_share", ascending=False)
              .reset_index(drop=True))

def save_finals_samples_csv(samples, names, season_num, final_week, out_csv):
    """
    Save accepted finals fan-share samples to CSV.
    Each row = one accepted fan-vote vector (shares sum to 1).
    Columns are finalist names.
    """
    if samples is None or len(samples) == 0:
        print("No samples to save.")
        return

    df_out = pd.DataFrame(samples, columns=names)
    df_out.insert(0, "season", season_num)
    df_out.insert(1, "final_week", final_week)
    df_out.insert(2, "sample_id", np.arange(len(df_out)))
    df_out.to_csv(out_csv, index=False)
    print(f"Saved {len(df_out)} accepted finals samples to {out_csv}")


def main():
    season_num = 3
    n_keep = 2000          # how many accepted samples you want to TRY to collect
    max_tries = 500000     # finite random generations (cap)
    alpha = 0.7
    seed = 200

    df = load_data()
    sdf = season_df(df, season_num)

    samples, names, info = mc_finals_percent(
        sdf,
        n_keep=n_keep,
        max_tries=max_tries,
        alpha=alpha,
        seed=seed
    )

    print("Finals info:", info)

    # If the finals function succeeded, info has final_week
    final_week = info.get("final_week", None) if isinstance(info, dict) else None

    out_csv = f"season_{season_num}_final_week_{final_week}_fan_samples.csv"
    save_finals_samples_csv(samples, names, season_num, final_week, out_csv)



# # Example usage:
# def main():
#     results = mc_season_percent(season_num=3, n_keep_per_week=1000, alpha=1.0, seed=123)

#     # Print info for all weeks
#     for week in range(1, 12):
#         wk = results[week]
#         print(f"Week {week} info:", wk["info"])

#     # Find the first week with accepted samples and summarize it
#     for week in range(1, 12):
#         wk = results[week]
#         if wk["samples"] is not None and len(wk["samples"]) > 0:
#             summary = summarize_week_samples(wk["samples"], wk["names"])
#             print("\nShowing summary for week", week)
#             print(summary.head(10))
#             break
#     else:
#         print("\nNo week produced accepted samples. Try alpha=0.5 or reduce n_keep_per_week.")


if __name__ == "__main__":
    main()