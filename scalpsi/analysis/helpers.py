import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata, f as f_dist
from scipy.spatial.distance import cosine as cosine_dist, pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from itertools import combinations


def get_common_perturbations(df):
    """Filter df to perturbations present in all cell lines.

    Returns (df_common, datasets, pairs, common_perturbs).
    """
    datasets = sorted(df['DataSet'].unique())
    pairs = list(combinations(datasets, 2))
    perturb_by_ds = {ds: set(df[df['DataSet'] == ds]['perturb'].unique()) for ds in datasets}
    common_perturbs = set.intersection(*perturb_by_ds.values())
    df_common = df[df['perturb'].isin(common_perturbs)].copy()
    print(f"Cell lines: {datasets}")
    print(f"Common perturbations: {len(common_perturbs)}")
    return df_common, datasets, pairs, common_perturbs


def _pairwise_correlations(pivot, pairs, label=None, verbose=False):
    """Compute Spearman and Pearson for all cell-line pairs on a pivoted df.

    Only uses rows (perturbations) present in ALL cell lines (no NAs).
    """
    # keep only perturbations present in all cell lines
    pivot = pivot.dropna()
    if verbose and label:
        print(f"  [{label}] perturbations in all cell lines: {len(pivot)}")
    if len(pivot) < 3:
        return []
    records = []
    for ds1, ds2 in pairs:
        rho, p_sp = spearmanr(pivot[ds1], pivot[ds2])
        r, p_pe = pearsonr(pivot[ds1], pivot[ds2])
        records.append({
            'pair': f"{ds1} vs {ds2}",
            'spearman_rho': rho, 'spearman_p': p_sp,
            'pearson_r': r, 'pearson_p': p_pe,
            'n_perturbs': len(pivot)
        })
    return records


# ─── Analysis 1: per metric (avg across DEG, method) ────────────────

def crosscell_per_metric(df, verbose=True):
    """Cross-cell-line consistency per metric.

    Averages across DEG and method.
    """
    df_common, datasets, pairs, _ = get_common_perturbations(df)
    results = []
    for metric in sorted(df_common['metric'].unique()):
        sub = df_common[df_common['metric'] == metric]
        agg = sub.groupby(['DataSet', 'perturb'])['performance'].mean().reset_index()
        pivot = agg.pivot(index='perturb', columns='DataSet', values='performance')
        recs = _pairwise_correlations(pivot, pairs)
        for rec in recs:
            rec['metric'] = metric
            results.append(rec)
        if verbose and recs:
            print(f"\n-- {metric} (n={recs[0]['n_perturbs']}) --")
            for rec in recs:
                print(f"  {rec['pair']}:  Spearman rho={rec['spearman_rho']:.4f} (p={rec['spearman_p']:.2e})"
                      f"  |  Pearson r={rec['pearson_r']:.4f} (p={rec['pearson_p']:.2e})")
    res = pd.DataFrame(results)
    if verbose and len(res) > 0:
        print("\n-- Summary (Spearman rho) --")
        print(res.pivot(index='metric', columns='pair', values='spearman_rho').round(4).to_string())
    return res


# ─── Analysis 2: per DEG (avg across metric, method) ────────────────

def crosscell_per_deg(df, verbose=True):
    """Cross-cell-line consistency per DEG level.

    Averages across all metrics and method.
    """
    df_common, datasets, pairs, _ = get_common_perturbations(df)
    results = []
    for deg in sorted(df_common['DEG'].unique()):
        sub = df_common[df_common['DEG'] == deg]
        agg = sub.groupby(['DataSet', 'perturb'])['performance'].mean().reset_index()
        pivot = agg.pivot(index='perturb', columns='DataSet', values='performance')
        recs = _pairwise_correlations(pivot, pairs)
        for rec in recs:
            rec['DEG'] = deg
            results.append(rec)
        if verbose and recs:
            print(f"\n-- DEG={deg} (n={recs[0]['n_perturbs']}) --")
            for rec in recs:
                print(f"  {rec['pair']}:  Spearman rho={rec['spearman_rho']:.4f} (p={rec['spearman_p']:.2e})"
                      f"  |  Pearson r={rec['pearson_r']:.4f} (p={rec['pearson_p']:.2e})")
    res = pd.DataFrame(results)
    if verbose and len(res) > 0:
        print("\n-- Summary (Spearman rho) --")
        print(res.pivot(index='DEG', columns='pair', values='spearman_rho').round(4).to_string())
    return res


# ─── Analysis 3a: per method (avg across metric, DEG) ───────────────

def crosscell_per_method(df, verbose=True):
    """Cross-cell-line consistency per method.

    Averages across metric and DEG.
    """
    df_common, datasets, pairs, _ = get_common_perturbations(df)
    results = []
    for method in sorted(df_common['method'].unique()):
        sub = df_common[df_common['method'] == method]
        agg = sub.groupby(['DataSet', 'perturb'])['performance'].mean().reset_index()
        pivot = agg.pivot(index='perturb', columns='DataSet', values='performance')
        recs = _pairwise_correlations(pivot, pairs)
        for rec in recs:
            rec['method'] = method
            results.append(rec)
        if verbose and recs:
            print(f"\n-- {method} (n={recs[0]['n_perturbs']}) --")
            for rec in recs:
                print(f"  {rec['pair']}:  Spearman rho={rec['spearman_rho']:.4f} (p={rec['spearman_p']:.2e})"
                      f"  |  Pearson r={rec['pearson_r']:.4f} (p={rec['pearson_p']:.2e})")
    res = pd.DataFrame(results)
    if verbose and len(res) > 0:
        print("\n-- Summary (Spearman rho) --")
        print(res.pivot(index='method', columns='pair', values='spearman_rho').round(4).to_string())
    return res


# ─── Analysis 3b: per method x metric (avg across DEG) ──────────────

def crosscell_per_method_metric(df, verbose=True):
    """Cross-cell-line consistency per method x metric.

    Averages across DEG.
    """
    df_common, datasets, pairs, _ = get_common_perturbations(df)
    results = []
    for method in sorted(df_common['method'].unique()):
        for metric in sorted(df_common['metric'].unique()):
            sub = df_common[(df_common['method'] == method) & (df_common['metric'] == metric)]
            agg = sub.groupby(['DataSet', 'perturb'])['performance'].mean().reset_index()
            pivot = agg.pivot(index='perturb', columns='DataSet', values='performance')
            recs = _pairwise_correlations(pivot, pairs)
            for rec in recs:
                rec['method'] = method
                rec['metric'] = metric
                results.append(rec)
    res = pd.DataFrame(results)
    if verbose and len(res) > 0:
        print("\n-- Mean Spearman rho across cell-line pairs (method x metric) --")
        avg = res.groupby(['method', 'metric'])['spearman_rho'].mean().unstack('metric')
        print(avg.round(4).to_string())
    return res


# ─── Analysis 3c: per method x DEG (avg across metric) ──────────────

def crosscell_per_method_deg(df, verbose=True):
    """Cross-cell-line consistency per method x DEG.

    Averages across metric.
    """
    df_common, datasets, pairs, _ = get_common_perturbations(df)
    results = []
    for method in sorted(df_common['method'].unique()):
        for deg in sorted(df_common['DEG'].unique()):
            sub = df_common[(df_common['method'] == method) & (df_common['DEG'] == deg)]
            agg = sub.groupby(['DataSet', 'perturb'])['performance'].mean().reset_index()
            pivot = agg.pivot(index='perturb', columns='DataSet', values='performance')
            recs = _pairwise_correlations(pivot, pairs)
            for rec in recs:
                rec['method'] = method
                rec['DEG'] = deg
                results.append(rec)
    res = pd.DataFrame(results)
    if verbose and len(res) > 0:
        print("\n-- Mean Spearman rho across cell-line pairs (method x DEG) --")
        avg = res.groupby(['method', 'DEG'])['spearman_rho'].mean().unstack('DEG')
        print(avg.round(4).to_string())
    return res


# ═══════════════════════════════════════════════════════════════════════
# Robustness checks
# ═══════════════════════════════════════════════════════════════════════

def permutation_test(df, n_perm=1000, seed=42, verbose=True):
    """Permutation test: shuffle perturbation labels within each cell line.

    Computes a null distribution of Spearman rho to compare against observed.
    Uses the overall average (across metric, DEG, method) per perturbation.
    Returns dict with observed rho, null distribution, and p-values per pair.
    """
    rng = np.random.default_rng(seed)
    df_common, datasets, pairs, _ = get_common_perturbations(df)

    # observed: avg across metric, DEG, method
    agg = df_common.groupby(['DataSet', 'perturb'])['performance'].mean().reset_index()
    pivot = agg.pivot(index='perturb', columns='DataSet', values='performance').dropna()

    observed = {}
    for ds1, ds2 in pairs:
        rho, _ = spearmanr(pivot[ds1], pivot[ds2])
        observed[f"{ds1} vs {ds2}"] = rho

    # null distribution: shuffle perturb labels within each cell line independently
    null_dist = {p: [] for p in observed}
    for _ in range(n_perm):
        shuffled = pivot.copy()
        for col in shuffled.columns:
            shuffled[col] = rng.permutation(shuffled[col].values)
        for ds1, ds2 in pairs:
            rho, _ = spearmanr(shuffled[ds1], shuffled[ds2])
            null_dist[f"{ds1} vs {ds2}"].append(rho)

    # p-values
    results = {}
    for pair_name in observed:
        null = np.array(null_dist[pair_name])
        obs = observed[pair_name]
        p = (np.sum(np.abs(null) >= np.abs(obs)) + 1) / (n_perm + 1)
        results[pair_name] = {'observed_rho': obs, 'perm_p': p,
                              'null_mean': null.mean(), 'null_std': null.std()}
        if verbose:
            print(f"{pair_name}:  observed rho={obs:.4f}  |  null mean={null.mean():.4f} +/- {null.std():.4f}  |  perm p={p:.4f}")

    return results, null_dist


def robustness_drop_top_perturbations(df, drop_quantile=0.9, verbose=True):
    """Re-run correlation after dropping perturbations with largest effects.

    Removes perturbations above drop_quantile of mean performance (across
    all cell lines, metrics, methods, DEGs). Tests whether a few extreme
    perturbations are driving the correlation.
    """
    df_common, datasets, pairs, _ = get_common_perturbations(df)

    # overall mean per perturbation (across everything)
    perturb_mean = df_common.groupby('perturb')['performance'].mean()
    threshold = perturb_mean.quantile(drop_quantile)
    keep = perturb_mean[perturb_mean <= threshold].index
    dropped = perturb_mean[perturb_mean > threshold].index

    if verbose:
        print(f"Dropping {len(dropped)} perturbations above {drop_quantile:.0%} quantile (threshold={threshold:.2f})")
        print(f"Remaining: {len(keep)} perturbations")

    df_trimmed = df_common[df_common['perturb'].isin(keep)]

    # re-compute overall correlation
    agg = df_trimmed.groupby(['DataSet', 'perturb'])['performance'].mean().reset_index()
    pivot = agg.pivot(index='perturb', columns='DataSet', values='performance').dropna()

    if verbose:
        print(f"\nAfter dropping top {1 - drop_quantile:.0%}:")
    results = []
    for ds1, ds2 in pairs:
        rho, p = spearmanr(pivot[ds1], pivot[ds2])
        results.append({'pair': f"{ds1} vs {ds2}", 'spearman_rho': rho, 'spearman_p': p})
        if verbose:
            print(f"  {ds1} vs {ds2}:  Spearman rho={rho:.4f} (p={p:.2e})")

    return pd.DataFrame(results), dropped


def robustness_per_split(df, verbose=True):
    """Run correlation separately for each split.

    If correlations are stable across splits, the result is not an
    artifact of a particular train/test partition.
    """
    df_common, datasets, pairs, _ = get_common_perturbations(df)
    splits = sorted(df_common['split'].unique())
    results = []

    for split in splits:
        sub = df_common[df_common['split'] == split]
        agg = sub.groupby(['DataSet', 'perturb'])['performance'].mean().reset_index()
        pivot = agg.pivot(index='perturb', columns='DataSet', values='performance').dropna()
        if verbose:
            print(f"\n-- split={split} (n={len(pivot)} perturbations) --")
        for ds1, ds2 in pairs:
            if ds1 in pivot.columns and ds2 in pivot.columns:
                rho, p = spearmanr(pivot[ds1], pivot[ds2])
                results.append({'split': split, 'pair': f"{ds1} vs {ds2}",
                                'spearman_rho': rho, 'spearman_p': p, 'n': len(pivot)})
                if verbose:
                    print(f"  {ds1} vs {ds2}:  Spearman rho={rho:.4f} (p={p:.2e})  n={len(pivot)}")

    res = pd.DataFrame(results)
    if verbose and len(res) > 0:
        print("\n-- Summary across splits --")
        summary = res.groupby('pair')['spearman_rho'].agg(['mean', 'std', 'min', 'max'])
        print(summary.round(4).to_string())
    return res


# ═══════════════════════════════════════════════════════════════════════
# Alternative agreement measures
# ═══════════════════════════════════════════════════════════════════════

def _make_pivot(df, groupby_cols=None):
    """Helper: aggregate df and pivot to perturbation x cell line."""
    df_common, datasets, pairs, _ = get_common_perturbations(df)
    agg = df_common.groupby(['DataSet', 'perturb'])['performance'].mean().reset_index()
    pivot = agg.pivot(index='perturb', columns='DataSet', values='performance').dropna()
    return pivot, datasets, pairs


def icc_agreement(df, verbose=True):
    """Intraclass Correlation Coefficient ICC(3,1) — consistency.

    Treats cell lines as fixed raters and perturbations as random subjects.
    A single number summarizing agreement across ALL cell lines simultaneously
    (not pairwise). ICC=1 means perfect agreement, ICC=0 means no agreement.

    Can be run on the full df (averages across metric, DEG, method) or on
    a pre-filtered df.
    """
    pivot, datasets, _ = _make_pivot(df)
    n = len(pivot)       # number of perturbations (subjects)
    k = len(datasets)    # number of cell lines (raters)
    X = pivot.values     # n x k matrix

    # Two-way ANOVA decomposition
    row_means = X.mean(axis=1)
    col_means = X.mean(axis=0)
    grand_mean = X.mean()

    # Sum of squares
    SS_row = k * np.sum((row_means - grand_mean) ** 2)        # between subjects
    SS_col = n * np.sum((col_means - grand_mean) ** 2)        # between raters
    SS_total = np.sum((X - grand_mean) ** 2)
    SS_error = SS_total - SS_row - SS_col                      # residual

    # Mean squares
    MS_row = SS_row / (n - 1)
    MS_error = SS_error / ((n - 1) * (k - 1))

    # ICC(3,1) — two-way mixed, consistency
    icc = (MS_row - MS_error) / (MS_row + (k - 1) * MS_error)

    # F-test for ICC significance
    F_val = MS_row / MS_error
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    p_val = 1 - f_dist.cdf(F_val, df1, df2)

    if verbose:
        print(f"ICC(3,1) = {icc:.4f}  (F={F_val:.2f}, df1={df1}, df2={df2}, p={p_val:.2e})")
        print(f"  n_perturbations={n}, n_cell_lines={k}")
        if icc < 0.5:
            print("  Interpretation: poor agreement")
        elif icc < 0.75:
            print("  Interpretation: moderate agreement")
        elif icc < 0.9:
            print("  Interpretation: good agreement")
        else:
            print("  Interpretation: excellent agreement")

    return {'icc': icc, 'F': F_val, 'df1': df1, 'df2': df2, 'p': p_val,
            'n_perturbs': n, 'n_celllines': k}


def kendall_w(df, verbose=True):
    """Kendall's W (coefficient of concordance).

    Measures agreement among multiple raters (cell lines) ranking the
    same items (perturbations). W=1 means perfect agreement in rankings,
    W=0 means no agreement. Based on ranks, so robust to outliers.
    """
    pivot, datasets, _ = _make_pivot(df)
    n = len(pivot)       # subjects
    k = len(datasets)    # raters

    # Rank within each cell line
    ranks = np.apply_along_axis(rankdata, 0, pivot.values)
    rank_sums = ranks.sum(axis=1)  # sum of ranks per perturbation

    # Kendall's W
    mean_rank_sum = rank_sums.mean()
    SS = np.sum((rank_sums - mean_rank_sum) ** 2)
    W = 12 * SS / (k ** 2 * (n ** 3 - n))

    # Chi-squared test for significance
    chi2 = k * (n - 1) * W
    df_chi = n - 1
    from scipy.stats import chi2 as chi2_dist
    p_val = 1 - chi2_dist.cdf(chi2, df_chi)

    if verbose:
        print(f"Kendall's W = {W:.4f}  (chi2={chi2:.2f}, df={df_chi}, p={p_val:.2e})")
        print(f"  n_perturbations={n}, n_cell_lines={k}")
        if W < 0.3:
            print("  Interpretation: weak concordance")
        elif W < 0.6:
            print("  Interpretation: moderate concordance")
        else:
            print("  Interpretation: strong concordance")

    return {'W': W, 'chi2': chi2, 'df': df_chi, 'p': p_val,
            'n_perturbs': n, 'n_celllines': k}


def topk_overlap(df, k_values=None, verbose=True):
    """Top-k and bottom-k overlap across cell lines.

    For each cell line, rank perturbations by performance (high = hard).
    Then check: do the top-k hardest (and bottom-k easiest) perturbations
    agree across cell lines?

    Reports Jaccard index for each pair and the k-way intersection size.
    """
    if k_values is None:
        k_values = [10, 20, 50]

    pivot, datasets, pairs = _make_pivot(df)
    n = len(pivot)
    results = []

    for k_val in k_values:
        if k_val > n:
            continue

        if verbose:
            print(f"\n-- k={k_val} (out of {n} perturbations) --")

        for direction, label in [('hardest', False), ('easiest', True)]:
            # rank perturbations: ascending=True means lowest value first (easiest)
            topk_sets = {}
            for ds in datasets:
                ranked = pivot[ds].sort_values(ascending=(direction == 'easiest'))
                topk_sets[ds] = set(ranked.index[:k_val])

            # all-way intersection
            all_intersection = set.intersection(*topk_sets.values())
            all_union = set.union(*topk_sets.values())
            all_jaccard = len(all_intersection) / len(all_union) if all_union else 0

            # expected overlap under random (hypergeometric)
            expected_pair = k_val * k_val / n
            expected_all = k_val * (k_val / n) ** (len(datasets) - 1)

            if verbose:
                print(f"  {label}: all-{len(datasets)} intersection={len(all_intersection)}/{k_val}  "
                      f"Jaccard={all_jaccard:.3f}  (expected random pair overlap ~{expected_pair:.1f})")

            # pairwise Jaccard
            for ds1, ds2 in pairs:
                inter = len(topk_sets[ds1] & topk_sets[ds2])
                union = len(topk_sets[ds1] | topk_sets[ds2])
                jaccard = inter / union if union else 0
                results.append({
                    'k': k_val, 'direction': direction, 'pair': f"{ds1} vs {ds2}",
                    'intersection': inter, 'jaccard': jaccard,
                    'expected_random': expected_pair
                })
                if verbose:
                    print(f"    {ds1} vs {ds2}: overlap={inter}/{k_val}  Jaccard={jaccard:.3f}")

    return pd.DataFrame(results)


def variance_decomposition(df, verbose=True):
    """Two-way ANOVA-style variance decomposition.

    Decomposes total variance in performance into:
      - perturbation effect (what we care about — shared signal)
      - cell line effect (systematic differences between cell lines)
      - residual (interaction + noise — the disagreement)

    High perturbation % = perturbations behave consistently across cell lines.
    High residual % = cell lines disagree on which perturbations are hard.
    """
    pivot, datasets, _ = _make_pivot(df)
    n = len(pivot)       # perturbations
    k = len(datasets)    # cell lines
    X = pivot.values

    grand_mean = X.mean()
    row_means = X.mean(axis=1)  # per perturbation
    col_means = X.mean(axis=0)  # per cell line

    SS_total = np.sum((X - grand_mean) ** 2)
    SS_perturb = k * np.sum((row_means - grand_mean) ** 2)
    SS_cellline = n * np.sum((col_means - grand_mean) ** 2)
    SS_residual = SS_total - SS_perturb - SS_cellline

    pct_perturb = 100 * SS_perturb / SS_total
    pct_cellline = 100 * SS_cellline / SS_total
    pct_residual = 100 * SS_residual / SS_total

    if verbose:
        print(f"Variance decomposition (n={n} perturbations, k={k} cell lines):")
        print(f"  Perturbation effect:  {pct_perturb:6.2f}%  (shared signal across cell lines)")
        print(f"  Cell line effect:     {pct_cellline:6.2f}%  (systematic cell line differences)")
        print(f"  Residual:             {pct_residual:6.2f}%  (disagreement / interaction)")

    return {'pct_perturbation': pct_perturb, 'pct_cellline': pct_cellline,
            'pct_residual': pct_residual, 'SS_perturbation': SS_perturb,
            'SS_cellline': SS_cellline, 'SS_residual': SS_residual,
            'n_perturbs': n, 'n_celllines': k}


# ═══════════════════════════════════════════════════════════════════════
# Perturbation-level analysis
# ═══════════════════════════════════════════════════════════════════════

def perturbation_analysis(df, method, metric, deg, top_n=20, agg_func='mean',
                          show_variable=False, verbose=True, cell_types=None):
    """Comprehensive perturbation-level analysis for a given method/metric/DEG.

    Returns a DataFrame (one row per perturbation) with:
      - per-cell-line performance
      - mean, std, range across cell lines
      - per-cell-line ranks
      - max rank difference
      - cell-line-specific delta (deviation from the other two)
      - which cell line is the outlier

    Parameters
    ----------
    cell_types : list of str or None
        If None (default), uses all cell types and restricts to common
        perturbations (original behavior). If a list is provided, restricts
        to those cell types and includes only perturbations present in all
        of the specified cell types.
    """
    sub = df[(df['method'] == method) & (df['metric'] == metric) & (df['DEG'] == deg)]

    if cell_types is not None:
        sub = sub[sub['DataSet'].isin(cell_types)]
        datasets = sorted(cell_types)
    else:
        datasets = None

    if datasets is not None:
        agg = sub.groupby(['DataSet', 'perturb'])['performance'].agg(agg_func).reset_index()
        pivot = agg.pivot(index='perturb', columns='DataSet', values='performance')
        # Keep only perturbations present in all specified cell types
        pivot = pivot.dropna(subset=datasets)
    else:
        df_common, datasets, pairs, _ = get_common_perturbations(sub)
        agg = df_common.groupby(['DataSet', 'perturb'])['performance'].agg(agg_func).reset_index()
        pivot = agg.pivot(index='perturb', columns='DataSet', values='performance').dropna()

    # summary stats
    pivot['mean_perf'] = pivot[datasets].mean(axis=1)
    pivot['median_perf'] = pivot[datasets].median(axis=1)
    pivot['std_perf'] = pivot[datasets].std(axis=1)
    pivot['range_perf'] = pivot[datasets].max(axis=1) - pivot[datasets].min(axis=1)

    # ranks within each cell line (1 = best/easiest)
    for ds in datasets:
        pivot[f'rank_{ds}'] = pivot[ds].rank()
    rank_cols = [f'rank_{ds}' for ds in datasets]
    pivot['max_rank_diff'] = (pivot[rank_cols].max(axis=1) -
                              pivot[rank_cols].min(axis=1))

    # cell-line-specific deltas
    for ds in datasets:
        others = [d for d in datasets if d != ds]
        pivot[f'delta_{ds}'] = pivot[ds] - pivot[others].mean(axis=1)
    delta_cols = [f'delta_{ds}' for ds in datasets]
    pivot['max_outlier_cellline'] = (pivot[delta_cols].abs().idxmax(axis=1)
                                     .str.replace('delta_', ''))
    pivot['max_outlier_value'] = pivot[delta_cols].abs().max(axis=1)

    if verbose:
        print(f"Method={method}, Metric={metric}, DEG={deg}")
        print(f"Cell lines: {datasets}")
        print(f"Lower = better | n={len(pivot)} common perturbations\n")

        # best
        print(f"{'='*70}")
        print(f"TOP {top_n} BEST PREDICTED (lowest {metric})")
        print(f"{'='*70}")
        print(pivot.sort_values('mean_perf')
              .head(top_n)[datasets + ['mean_perf']].round(4).to_string())

        # worst
        print(f"\n{'='*70}")
        print(f"TOP {top_n} WORST PREDICTED (highest {metric})")
        print(f"{'='*70}")
        print(pivot.sort_values('mean_perf', ascending=False)
              .head(top_n)[datasets + ['mean_perf']].round(4).to_string())

        if show_variable:
            # most variable
            print(f"\n{'='*70}")
            print(f"TOP {top_n} MOST VARIABLE ACROSS CELL LINES (by std)")
            print(f"{'='*70}")
            print(pivot.sort_values('std_perf', ascending=False)
                  .head(top_n)[datasets + ['mean_perf', 'std_perf', 'range_perf']
                               + rank_cols].round(4).to_string())

            # most variable by rank
            print(f"\n{'='*70}")
            print(f"TOP {top_n} MOST VARIABLE BY RANK DIFFERENCE")
            print(f"{'='*70}")
            print(pivot.sort_values('max_rank_diff', ascending=False)
                  .head(top_n)[datasets + ['mean_perf', 'max_rank_diff']
                               + rank_cols].round(4).to_string())

            # most consistent
            print(f"\n{'='*70}")
            print(f"TOP {top_n} MOST CONSISTENT ACROSS CELL LINES (lowest std)")
            print(f"{'='*70}")
            print(pivot.sort_values('std_perf')
                  .head(top_n)[datasets + ['mean_perf', 'std_perf']
                               + rank_cols].round(4).to_string())

            # cell-line-specific outliers
            print(f"\n{'='*70}")
            print(f"TOP {top_n} CELL-LINE-SPECIFIC OUTLIERS")
            print(f"{'='*70}")
            print(pivot.sort_values('max_outlier_value', ascending=False)
                  .head(top_n)[datasets + ['mean_perf', 'max_outlier_cellline',
                               'max_outlier_value'] + delta_cols].round(4).to_string())

            # per cell line: uniquely hard/easy
            for ds in datasets:
                delta_col = f'delta_{ds}'
                print(f"\n{'─'*70}")
                print(f"  {ds.upper()}: top 5 uniquely HARDER")
                print(pivot.sort_values(delta_col, ascending=False)
                      .head(5)[datasets + [delta_col] + rank_cols].round(4).to_string())
                print(f"\n  {ds.upper()}: top 5 uniquely EASIER")
                print(pivot.sort_values(delta_col)
                      .head(5)[datasets + [delta_col] + rank_cols].round(4).to_string())

    return pivot


# ═══════════════════════════════════════════════════════════════════════
# Method performance overview
# ═══════════════════════════════════════════════════════════════════════

def _convert_higher_to_lower(df, higher_is_better_metrics=None, verbose=True):
    """Convert higher-is-better metrics into lower-is-better distances.

    Uses distance = 1 - score for each metric listed in higher_is_better_metrics.
    This is suitable for bounded concordance/correlation-style scores.
    """
    if higher_is_better_metrics is None:
        higher_is_better_metrics = {'ccc_delta'}

    out = df.copy()
    converted = []
    missing = []

    for metric in higher_is_better_metrics:
        mask = out['metric'] == metric
        if not mask.any():
            missing.append(metric)
            continue

        out.loc[mask, 'performance'] = 1.0 - out.loc[mask, 'performance']
        new_name = metric.replace('ccc_', 'ccc_distance_') if metric.startswith('ccc_') else f"{metric}_distance"
        out.loc[mask, 'metric'] = new_name
        converted.append((metric, new_name))

    if verbose and converted:
        print("Converted higher-is-better metrics to lower-is-better:")
        for old, new in converted:
            print(f"  {old} -> {new} (performance = 1 - value)")
    if verbose and missing:
        print(f"No rows found for metrics marked as higher-is-better: {sorted(missing)}")

    return out


def method_performance_overview(df, common_only=False, verbose=True,
                                convert_higher_to_lower=False,
                                higher_is_better_metrics=None):
    """Overview of method performance across cell lines, metrics, and DEG levels.

    Args:
        df: performance DataFrame.
        common_only: if True, filter to perturbations present in ALL cell lines.
        verbose: print results.
        convert_higher_to_lower: if True, transform higher-is-better metrics
            to lower-is-better (distance = 1 - score) before aggregation/ranking.
        higher_is_better_metrics: iterable of metric names to transform when
            convert_higher_to_lower=True. Default {'ccc_delta'}.

    Prints:
      1. Mean performance per method (averaged across everything)
      2. Method x cell line (averaged across metric, DEG)
      3. Method x metric (averaged across cell line, DEG)
      4. Method x DEG (averaged across cell line, metric)
      5. Method ranking per metric (which method wins on each metric?)
      6. Method ranking per cell line (which method wins per cell line?)
      7. Best method per metric x cell line

    Returns dict of DataFrames.
    """
    if convert_higher_to_lower:
        df = _convert_higher_to_lower(
            df,
            higher_is_better_metrics=higher_is_better_metrics,
            verbose=verbose,
        )

    if common_only:
        datasets = sorted(df['DataSet'].unique())
        perturb_by_ds = {ds: set(df[df['DataSet'] == ds]['perturb'].unique())
                         for ds in datasets}
        common = set.intersection(*perturb_by_ds.values())
        df = df[df['perturb'].isin(common)]
        if verbose:
            print(f"Filtered to {len(common)} common perturbations across {datasets}\n")

    methods = sorted(df['method'].unique())
    metrics = sorted(df['metric'].unique())
    datasets = sorted(df['DataSet'].unique())
    deg_values = sorted(df['DEG'].unique())

    # 0. Perturbation counts
    n_perturbs = df.groupby(['method', 'DataSet'])['perturb'].nunique().unstack('DataSet')
    if verbose:
        print("="*70)
        print("0. NUMBER OF PERTURBATIONS PER METHOD x CELL LINE")
        print("="*70)
        print(n_perturbs.to_string())

    # 1. Overall: method mean
    overall = df.groupby('method')['performance'].mean().sort_values()
    if verbose:
        print(f"\n{'='*70}")
        print("1. OVERALL MEAN PERFORMANCE (lower = better)")
        print("="*70)
        print(overall.round(4).to_string())

    # 2. Method x cell line
    method_cellline = df.groupby(['method', 'DataSet'])['performance'].mean().unstack('DataSet')
    method_cellline['mean'] = method_cellline.mean(axis=1)
    method_cellline = method_cellline.sort_values('mean')
    if verbose:
        print(f"\n{'='*70}")
        print("2. METHOD x CELL LINE (avg across metric, DEG)")
        print("="*70)
        print(method_cellline.round(4).to_string())

    # 3. Method x metric
    method_metric = df.groupby(['method', 'metric'])['performance'].mean().unstack('metric')
    if verbose:
        print(f"\n{'='*70}")
        print("3. METHOD x METRIC (avg across cell line, DEG)")
        print("="*70)
        print(method_metric.round(4).to_string())

    # 4. Method x DEG
    method_deg = df.groupby(['method', 'DEG'])['performance'].mean().unstack('DEG')
    if verbose:
        print(f"\n{'='*70}")
        print("4. METHOD x DEG (avg across cell line, metric)")
        print("="*70)
        print(method_deg.round(4).to_string())

    # 5. Rank methods per metric (1 = best)
    ranks_by_metric = method_metric.rank(axis=0)
    if verbose:
        print(f"\n{'='*70}")
        print("5. METHOD RANK PER METRIC (1 = best)")
        print("="*70)
        print(ranks_by_metric.round(1).to_string())
        print(f"\nMean rank across metrics:")
        print(ranks_by_metric.mean(axis=1).sort_values().round(2).to_string())

    # 6. Rank methods per cell line (1 = best)
    ranks_by_cellline = method_cellline[datasets].rank(axis=0)
    if verbose:
        print(f"\n{'='*70}")
        print("6. METHOD RANK PER CELL LINE (1 = best)")
        print("="*70)
        print(ranks_by_cellline.round(1).to_string())
        print(f"\nMean rank across cell lines:")
        print(ranks_by_cellline.mean(axis=1).sort_values().round(2).to_string())

    # 7. Method x metric x cell line — who wins where?
    if verbose:
        print(f"\n{'='*70}")
        print("7. BEST METHOD PER METRIC x CELL LINE")
        print("="*70)
        best = df.groupby(['metric', 'DataSet', 'method'])['performance'].mean().reset_index()
        winners = best.loc[best.groupby(['metric', 'DataSet'])['performance'].idxmin()]
        winner_pivot = winners.pivot(index='metric', columns='DataSet', values='method')
        print(winner_pivot.to_string())

    # 8. Balanced rank across (cell line x metric), robust to metric scale
    method_dataset_metric = (
        df.groupby(['method', 'DataSet', 'metric'])['performance']
          .mean()
          .reset_index()
    )
    method_dataset_metric['rank'] = (
        method_dataset_metric.groupby(['DataSet', 'metric'])['performance']
        .rank(method='average')
    )
    balanced_rank = (
        method_dataset_metric.groupby('method')['rank']
        .mean()
        .sort_values()
    )
    if verbose:
        print(f"\n{'='*70}")
        print("8. BALANCED MEAN RANK (avg over cell line x metric, 1 = best)")
        print("="*70)
        print(balanced_rank.round(2).to_string())

    return {
        'overall': overall,
        'method_cellline': method_cellline,
        'method_metric': method_metric,
        'method_deg': method_deg,
        'ranks_by_metric': ranks_by_metric,
        'ranks_by_cellline': ranks_by_cellline,
        'balanced_rank': balanced_rank,
    }


# ═══════════════════════════════════════════════════════════════════════
# Metric recovery / validation from gene_perf
# ═══════════════════════════════════════════════════════════════════════

def recover_metrics_from_geneperf(gene_perf, deg_levels=None):
    """Recompute pseudo-bulk metrics from gene-level data.

    From gene_perf (one row per gene per perturbation), recomputes:
      - root_mean_squared_error: sqrt(mean(squared_error))
      - mean_absolute_error: mean(abs_error)
      - pearson_distance: 1 - pearson(mean_pred_delta, mean_true_delta)
      - spearman_distance: 1 - spearman(mean_pred, mean_true)
      - cosine_distance: 1 - cosine_similarity(mean_pred, mean_true)

    Args:
        gene_perf: DataFrame with columns dataset, method, split, perturbation,
                   gene, mean_pred, mean_true, mean_pred_delta, mean_true_delta,
                   abs_error, squared_error, deg_rank
        deg_levels: list of DEG thresholds to compute (default [100, 1000, 2000]).
                    gene_perf only has top 2000 DEGs, so max is 2000.

    Returns:
        DataFrame matching the structure of the summary df, with columns:
        DataSet, method, split, perturb, metric, DEG, performance
    """
    if deg_levels is None:
        deg_levels = [100, 1000, 2000]

    max_rank = gene_perf['deg_rank'].max()
    deg_levels = [d for d in deg_levels if d <= max_rank]

    records = []
    groups = gene_perf.groupby(['dataset', 'method', 'split', 'perturbation'])

    for (ds, method, split, perturb), g in groups:
        for deg in deg_levels:
            sub = g[g['deg_rank'] <= deg]
            if len(sub) < 2:
                continue

            pred = sub['mean_pred'].values
            true = sub['mean_true'].values
            pred_d = sub['mean_pred_delta'].values
            true_d = sub['mean_true_delta'].values

            base = {'DataSet': ds, 'method': method, 'split': split,
                    'perturb': perturb, 'DEG': deg}

            # root_mean_squared_error — pertpy computes Euclidean distance:
            # sqrt(SUM((x-y)^2)), NOT sqrt(MEAN((x-y)^2))
            euclidean = np.sqrt(sub['squared_error'].sum())
            records.append({**base, 'metric': 'root_mean_squared_error',
                            'performance': round(euclidean, 4)})

            # MAE
            mae = sub['abs_error'].mean()
            records.append({**base, 'metric': 'mean_absolute_error',
                            'performance': round(mae, 4)})

            # Pearson distance (on deltas)
            if np.std(pred_d) > 0 and np.std(true_d) > 0:
                r, _ = pearsonr(pred_d, true_d)
                records.append({**base, 'metric': 'pearson_distance',
                                'performance': round(1 - r, 4)})

            # Spearman distance (on raw)
            if np.std(pred) > 0 and np.std(true) > 0:
                rho, _ = spearmanr(pred, true)
                records.append({**base, 'metric': 'spearman_distance',
                                'performance': round(1 - rho, 4)})

            # Cosine distance (on raw)
            if np.any(pred != 0) and np.any(true != 0):
                cd = cosine_dist(pred, true)
                records.append({**base, 'metric': 'cosine_distance',
                                'performance': round(cd, 4)})

    return pd.DataFrame(records)


def validate_geneperf_vs_summary(gene_perf, summary_df, deg_levels=None,
                                  verbose=True):
    """Validate that metrics recomputed from gene_perf match the summary df.

    Recomputes pseudo-bulk metrics from gene_perf and compares them to
    the pre-computed values in summary_df.

    Args:
        gene_perf: gene-level DataFrame (from geneperf parquet)
        summary_df: summary performance DataFrame (from performance TSV)
        deg_levels: DEG levels to check (default [100, 1000, 2000])
        verbose: print comparison results

    Returns:
        DataFrame with columns: DataSet, method, split, perturb, metric, DEG,
        perf_summary, perf_recovered, abs_diff, rel_diff
    """
    recoverable = ['root_mean_squared_error', 'mean_absolute_error',
                    'pearson_distance', 'spearman_distance', 'cosine_distance']

    if deg_levels is None:
        deg_levels = [100, 1000, 2000]

    if verbose:
        print("Recomputing pseudo-bulk metrics from gene_perf...")
    recovered = recover_metrics_from_geneperf(gene_perf, deg_levels)
    if verbose:
        print(f"  Recovered {len(recovered)} metric values")

    # Filter summary to recoverable metrics and DEG levels
    summary_sub = summary_df[
        (summary_df['metric'].isin(recoverable)) &
        (summary_df['DEG'].isin(deg_levels))
    ].copy()

    if verbose:
        print(f"  Summary df has {len(summary_sub)} matching entries")

    # Merge
    merge_keys = ['DataSet', 'method', 'split', 'perturb', 'metric', 'DEG']
    # Ensure matching dtypes
    for col in ['split', 'DEG']:
        recovered[col] = recovered[col].astype(int)
        summary_sub[col] = summary_sub[col].astype(int)

    merged = summary_sub.merge(
        recovered,
        on=merge_keys,
        how='inner',
        suffixes=('_summary', '_recovered')
    )

    if verbose:
        print(f"  Matched {len(merged)} entries after merge")

    merged['abs_diff'] = (merged['performance_summary'] -
                          merged['performance_recovered']).abs()
    merged['rel_diff'] = merged['abs_diff'] / (merged['performance_summary'].abs() + 1e-10)

    if verbose:
        print(f"\n{'='*70}")
        print("VALIDATION RESULTS")
        print(f"{'='*70}")

        for metric in recoverable:
            m = merged[merged['metric'] == metric]
            if len(m) == 0:
                print(f"\n  {metric}: no matched entries")
                continue

            print(f"\n  {metric} ({len(m)} entries):")
            print(f"    Mean abs diff:   {m['abs_diff'].mean():.6f}")
            print(f"    Median abs diff: {m['abs_diff'].median():.6f}")
            print(f"    Max abs diff:    {m['abs_diff'].max():.6f}")
            print(f"    Mean rel diff:   {m['rel_diff'].mean():.4%}")

            close = (m['abs_diff'] < 0.001).sum()
            print(f"    Within 0.001:    {close}/{len(m)} ({close/len(m):.1%})")

            close2 = (m['abs_diff'] < 0.01).sum()
            print(f"    Within 0.01:     {close2}/{len(m)} ({close2/len(m):.1%})")

        # per DEG level
        print(f"\n{'='*70}")
        print("BY DEG LEVEL")
        print(f"{'='*70}")
        for deg in sorted(merged['DEG'].unique()):
            m = merged[merged['DEG'] == deg]
            print(f"\n  DEG={deg} ({len(m)} entries):")
            for metric in recoverable:
                mm = m[m['metric'] == metric]
                if len(mm) > 0:
                    print(f"    {metric:30s}  mean_diff={mm['abs_diff'].mean():.6f}  "
                          f"max_diff={mm['abs_diff'].max():.6f}")

        # Show worst mismatches
        worst = merged.nlargest(10, 'abs_diff')
        print(f"\n{'='*70}")
        print("TOP 10 WORST MISMATCHES")
        print(f"{'='*70}")
        print(worst[merge_keys + ['performance_summary', 'performance_recovered',
                                   'abs_diff']].to_string(index=False))

    return merged


def compute_new_corr_metrics(
    gene_perf,
    deg=2000,
    verbose=True,
    sign_tau=0.0,
    updown_tau=0.0,
    de_frac=0.1,
    topk_values=(100,),
    include_metrics=None,
):
    """Compute additional distance metrics from gene_perf.

    Existing metrics in df:
      - pearson_distance:  on deltas (control-subtracted)
      - spearman_distance: on raw expression

    This function computes:
      - pearson_distance_raw:    1 - pearson(mean_pred, mean_true)
      - spearman_distance_raw:   1 - spearman(mean_pred, mean_true)
      - pearson_distance_delta:  1 - pearson(mean_pred_delta, mean_true_delta)
      - spearman_distance_delta: 1 - spearman(mean_pred_delta, mean_true_delta)
      - mean_squared_error_delta: mean((pred_delta - true_delta)^2)
      - cosine_distance_delta:   cosine distance on delta vectors
      - sign_error_delta:        1 - sign_accuracy_delta (optionally |true_delta| > sign_tau)
      - updown_f1_distance_delta: 1 - macro-F1 on {-1,0,+1} labels from deltas
      - de_auroc_distance_delta: 1 - AUROC for DE detection using |pred_delta|
      - de_auprc_distance_delta: 1 - AUPRC for DE detection using |pred_delta|
      - topk_jaccard_distance_delta_k{K}: 1 - Jaccard(top-K |delta| genes)
      - topk_precision_distance_delta_k{K}: 1 - Precision@K(top-K |delta| genes)
      - ccc_delta:               Concordance Correlation Coefficient on deltas

    Args:
      gene_perf: gene-level performance DataFrame.
      deg: DEG threshold(s) to use. Can be:
        - int (e.g., 2000)
        - iterable of ints (e.g., [100, 1000, 2000, 5000])
        - 'all' to use all unique available deg_rank cutoffs
      verbose: print summary counts
      sign_tau: threshold for sign metric mask; only genes with
          |true_delta| > sign_tau are used in sign_error_delta.
      updown_tau: threshold for 3-class labels in up/down F1.
          delta > +tau => up, delta < -tau => down, otherwise unchanged.
      de_frac: fraction of genes labeled DE-positive (top |true_delta|) for
          AUROC/AUPRC metrics. Must be in (0, 1).
      topk_values: iterable of K values for top-K overlap metrics.
      include_metrics: optional metric filter. If None, compute all metrics.
          Can include exact metric names (e.g., 'pearson_distance_delta'),
          top-K names (e.g., 'topk_jaccard_distance_delta_k100'), or aliases:
          'pearson', 'spearman', 'raw_corr', 'delta_corr', 'de', 'topk', 'all'.

    Returns DataFrame with same columns as df (DataSet, method, split, perturb,
    metric, DEG, performance, Nstimulated, Nimputed) ready to pd.concat onto df.
    """
    if isinstance(deg, str):
        if deg.lower() == 'all':
            deg_levels = sorted(gene_perf['deg_rank'].dropna().astype(int).unique().tolist())
        else:
            raise ValueError("deg must be an int, iterable of ints, or 'all'")
    elif np.isscalar(deg):
        deg_levels = [int(deg)]
    else:
        deg_levels = sorted({int(d) for d in deg})

    if len(deg_levels) == 0:
        raise ValueError("No DEG levels provided")

    max_rank = int(gene_perf['deg_rank'].max())
    valid_deg_levels = [d for d in deg_levels if d <= max_rank]
    dropped_deg_levels = [d for d in deg_levels if d > max_rank]
    if len(valid_deg_levels) == 0:
        raise ValueError(f"All requested DEG levels exceed max deg_rank={max_rank}: {deg_levels}")
    if not (0 < float(de_frac) < 1):
        raise ValueError(f"de_frac must be in (0, 1), got {de_frac}")

    topk_levels = sorted({int(k) for k in topk_values if int(k) > 0})
    if len(topk_levels) == 0:
        raise ValueError("topk_values must contain at least one positive integer")

    topk_jaccard_metrics = {f'topk_jaccard_distance_delta_k{k}' for k in topk_levels}
    topk_precision_metrics = {f'topk_precision_distance_delta_k{k}' for k in topk_levels}
    fixed_metrics = {
        'pearson_distance_raw',
        'spearman_distance_raw',
        'pearson_distance_delta',
        'spearman_distance_delta',
        'mean_absolute_error_delta',
        'mean_squared_error_delta',
        'root_mean_squared_error_delta',
        'cosine_distance_delta',
        'sign_error_delta',
        'updown_f1_distance_delta',
        'de_auroc_distance_delta',
        'de_auprc_distance_delta',
        'ccc_delta',
    }
    all_metric_names = fixed_metrics | topk_jaccard_metrics | topk_precision_metrics
    alias_map = {
        'all': all_metric_names,
        'pearson': {'pearson_distance_raw', 'pearson_distance_delta'},
        'spearman': {'spearman_distance_raw', 'spearman_distance_delta'},
        'raw_corr': {'pearson_distance_raw', 'spearman_distance_raw'},
        'delta_corr': {'pearson_distance_delta', 'spearman_distance_delta'},
        'de': {'de_auroc_distance_delta', 'de_auprc_distance_delta'},
        'topk': topk_jaccard_metrics | topk_precision_metrics,
        'topk_jaccard_distance_delta': topk_jaccard_metrics,
        'topk_precision_distance_delta': topk_precision_metrics,
    }

    if include_metrics is None:
        enabled_metrics = set(all_metric_names)
    else:
        requested = [include_metrics] if isinstance(include_metrics, str) else list(include_metrics)
        if len(requested) == 0:
            raise ValueError("include_metrics must be None, a string, or a non-empty iterable")
        enabled_metrics = set()
        unknown = []
        for metric_name in requested:
            key = str(metric_name).strip()
            if key in all_metric_names:
                enabled_metrics.add(key)
            elif key in alias_map:
                enabled_metrics.update(alias_map[key])
            else:
                unknown.append(key)
        if unknown:
            valid_options = sorted(all_metric_names | set(alias_map.keys()))
            raise ValueError(
                f"Unknown include_metrics entries: {unknown}. "
                f"Valid metric names/aliases: {valid_options}"
            )
        if len(enabled_metrics) == 0:
            raise ValueError(
                "include_metrics resolved to an empty metric set. "
                "If selecting top-K metrics, ensure topk_values contains matching K."
            )

    enabled_topk_jaccard = [k for k in topk_levels if f'topk_jaccard_distance_delta_k{k}' in enabled_metrics]
    enabled_topk_precision = [k for k in topk_levels if f'topk_precision_distance_delta_k{k}' in enabled_metrics]
    need_sign = 'sign_error_delta' in enabled_metrics
    need_updown = 'updown_f1_distance_delta' in enabled_metrics
    need_de = ('de_auroc_distance_delta' in enabled_metrics) or ('de_auprc_distance_delta' in enabled_metrics)
    need_any_topk = bool(enabled_topk_jaccard) or bool(enabled_topk_precision)
    need_ccc = 'ccc_delta' in enabled_metrics

    groups = gene_perf.groupby(['dataset', 'method', 'split', 'perturbation'])

    records = []
    n_skip_praw = 0
    n_skip_sraw = 0
    n_skip_pdelta = 0
    n_skip_sdelta = 0
    n_skip_ccc_delta = 0
    n_skip_sign = 0
    n_skip_de_auc = 0
    n_skip_de_auprc = 0

    def _macro_f1_three_class(y_true, y_pred, labels=(-1, 0, 1)):
        f1s = []
        for label in labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fp = np.sum((y_true != label) & (y_pred == label))
            fn = np.sum((y_true == label) & (y_pred != label))
            denom = (2 * tp + fp + fn)
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            f1s.append(f1)
        return float(np.mean(f1s))

    def _binary_auroc(y_true, scores):
        y_true = np.asarray(y_true, dtype=int)
        scores = np.asarray(scores, dtype=float)
        pos = (y_true == 1)
        neg = (y_true == 0)
        n_pos = int(pos.sum())
        n_neg = int(neg.sum())
        if n_pos == 0 or n_neg == 0:
            return None
        ranks = rankdata(scores, method='average')
        sum_ranks_pos = ranks[pos].sum()
        auroc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return float(auroc)

    def _binary_auprc(y_true, scores):
        y_true = np.asarray(y_true, dtype=int)
        scores = np.asarray(scores, dtype=float)
        n_pos = int((y_true == 1).sum())
        if n_pos == 0:
            return None
        order = np.argsort(scores)[::-1]
        y_sorted = y_true[order]
        tp = np.cumsum(y_sorted == 1)
        fp = np.cumsum(y_sorted == 0)
        precision = tp / np.maximum(tp + fp, 1)
        ap = precision[y_sorted == 1].sum() / n_pos
        return float(ap)

    for (ds, method, split, perturb), g_all in groups:
        n_stim = g_all['n_stim_cells'].iloc[0]
        n_imp = g_all['n_imp_cells'].iloc[0]

        for deg_level in valid_deg_levels:
            g = g_all[g_all['deg_rank'] <= deg_level]
            if g.shape[0] == 0:
                continue

            pred = g['mean_pred'].values
            true = g['mean_true'].values
            pred_d = g['mean_pred_delta'].values
            true_d = g['mean_true_delta'].values

            base = {'DataSet': ds, 'method': method, 'split': int(split),
                    'perturb': perturb, 'DEG': int(deg_level),
                    'Nstimulated': int(n_stim), 'Nimputed': int(n_imp)}

            # pearson on raw (no delta)
            if 'pearson_distance_raw' in enabled_metrics:
                if np.std(pred) > 0 and np.std(true) > 0:
                    r, _ = pearsonr(pred, true)
                    records.append({**base, 'metric': 'pearson_distance_raw',
                                    'performance': round(1 - r, 4)})
                else:
                    n_skip_praw += 1

            # spearman on raw (no delta)
            if 'spearman_distance_raw' in enabled_metrics:
                if np.std(pred) > 0 and np.std(true) > 0:
                    rho_raw, _ = spearmanr(pred, true)
                    records.append({**base, 'metric': 'spearman_distance_raw',
                                    'performance': round(1 - rho_raw, 4)})
                else:
                    n_skip_sraw += 1

            # pearson on delta
            if 'pearson_distance_delta' in enabled_metrics:
                if np.std(pred_d) > 0 and np.std(true_d) > 0:
                    r_d, _ = pearsonr(pred_d, true_d)
                    records.append({**base, 'metric': 'pearson_distance_delta',
                                    'performance': round(1 - r_d, 4)})
                else:
                    n_skip_pdelta += 1

            # spearman on delta
            if 'spearman_distance_delta' in enabled_metrics:
                if np.std(pred_d) > 0 and np.std(true_d) > 0:
                    rho, _ = spearmanr(pred_d, true_d)
                    records.append({**base, 'metric': 'spearman_distance_delta',
                                    'performance': round(1 - rho, 4)})
                else:
                    n_skip_sdelta += 1

            # MAE on delta
            if 'mean_absolute_error_delta' in enabled_metrics:
                mae_delta = np.mean(np.abs(pred_d - true_d))
                records.append({**base, 'metric': 'mean_absolute_error_delta',
                                'performance': round(mae_delta, 4)})

            # MSE on delta
            if 'mean_squared_error_delta' in enabled_metrics:
                mse_delta = np.mean((pred_d - true_d) ** 2)
                records.append({**base, 'metric': 'mean_squared_error_delta',
                                'performance': round(mse_delta, 4)})

            # RMSE on delta
            if 'root_mean_squared_error_delta' in enabled_metrics:
                rmse_delta = np.sqrt(np.mean((pred_d - true_d) ** 2))
                records.append({**base, 'metric': 'root_mean_squared_error_delta',
                                'performance': round(rmse_delta, 4)})

            # Cosine distance on delta
            if 'cosine_distance_delta' in enabled_metrics:
                if np.any(pred_d != 0) and np.any(true_d != 0):
                    cos_d = cosine_dist(pred_d, true_d)
                elif np.allclose(pred_d, 0) and np.allclose(true_d, 0):
                    cos_d = 0.0
                else:
                    cos_d = 1.0
                records.append({**base, 'metric': 'cosine_distance_delta',
                                'performance': round(float(cos_d), 4)})

            # Sign error on delta (optionally filtered by |true_delta| > sign_tau)
            if need_sign:
                if sign_tau > 0:
                    mask = np.abs(true_d) > sign_tau
                else:
                    mask = np.ones_like(true_d, dtype=bool)
                if mask.sum() > 0:
                    sign_acc = np.mean(np.sign(pred_d[mask]) == np.sign(true_d[mask]))
                    records.append({**base, 'metric': 'sign_error_delta',
                                    'performance': round(float(1 - sign_acc), 4)})
                else:
                    n_skip_sign += 1

            # Up/down/unchanged macro-F1 on delta (reported as distance)
            if need_updown:
                pred_cls = np.where(pred_d > updown_tau, 1, np.where(pred_d < -updown_tau, -1, 0))
                true_cls = np.where(true_d > updown_tau, 1, np.where(true_d < -updown_tau, -1, 0))
                f1_macro = _macro_f1_three_class(true_cls, pred_cls, labels=(-1, 0, 1))
                records.append({**base, 'metric': 'updown_f1_distance_delta',
                                'performance': round(float(1 - f1_macro), 4)})

            if need_de or need_any_topk:
                abs_true = np.abs(true_d)
                abs_pred = np.abs(pred_d)
                n_genes = len(abs_true)

                if need_de:
                    # DE detection metrics: positives = top de_frac by |true_delta|, scores = |pred_delta|
                    n_pos = int(np.ceil(float(de_frac) * n_genes))
                    n_pos = max(1, min(n_pos, n_genes - 1))
                    pos_idx = np.argpartition(abs_true, -n_pos)[-n_pos:]
                    y_de = np.zeros(n_genes, dtype=int)
                    y_de[pos_idx] = 1

                    if 'de_auroc_distance_delta' in enabled_metrics:
                        auroc = _binary_auroc(y_de, abs_pred)
                        if auroc is not None:
                            records.append({**base, 'metric': 'de_auroc_distance_delta',
                                            'performance': round(float(1 - auroc), 4)})
                        else:
                            n_skip_de_auc += 1

                    if 'de_auprc_distance_delta' in enabled_metrics:
                        auprc = _binary_auprc(y_de, abs_pred)
                        if auprc is not None:
                            records.append({**base, 'metric': 'de_auprc_distance_delta',
                                            'performance': round(float(1 - auprc), 4)})
                        else:
                            n_skip_de_auprc += 1

                if need_any_topk:
                    # Top-K overlap metrics on |delta| (reported as distances)
                    true_rank = np.argsort(abs_true)[::-1]
                    pred_rank = np.argsort(abs_pred)[::-1]

                    for k in enabled_topk_jaccard:
                        k_eff = min(k, n_genes)
                        true_top = set(true_rank[:k_eff].tolist())
                        pred_top = set(pred_rank[:k_eff].tolist())
                        inter = len(true_top & pred_top)
                        union = len(true_top | pred_top)
                        jaccard = (inter / union) if union > 0 else 0.0
                        records.append({**base, 'metric': f'topk_jaccard_distance_delta_k{k}',
                                        'performance': round(float(1 - jaccard), 4)})

                    for k in enabled_topk_precision:
                        k_eff = min(k, n_genes)
                        true_top = set(true_rank[:k_eff].tolist())
                        pred_top = set(pred_rank[:k_eff].tolist())
                        inter = len(true_top & pred_top)
                        precision_k = inter / k_eff
                        records.append({**base, 'metric': f'topk_precision_distance_delta_k{k}',
                                        'performance': round(float(1 - precision_k), 4)})

            # CCC on delta
            if need_ccc:
                mu_pred_d = np.mean(pred_d)
                mu_true_d = np.mean(true_d)
                var_pred_d = np.var(pred_d)
                var_true_d = np.var(true_d)
                cov_pred_true_d = np.mean((pred_d - mu_pred_d) * (true_d - mu_true_d))
                ccc_den = var_pred_d + var_true_d + (mu_pred_d - mu_true_d) ** 2
                if ccc_den > 0:
                    ccc_delta = (2 * cov_pred_true_d) / ccc_den
                    ccc_delta = float(np.clip(ccc_delta, -1.0, 1.0))
                    records.append({**base, 'metric': 'ccc_delta',
                                    'performance': round(ccc_delta, 4)})
                elif np.allclose(pred_d, true_d):
                    # Degenerate identical vectors: perfect concordance.
                    records.append({**base, 'metric': 'ccc_delta',
                                    'performance': 1.0})
                else:
                    n_skip_ccc_delta += 1

    result = pd.DataFrame(records)

    if verbose:
        print(f"Computed from gene_perf (DEG levels={valid_deg_levels})")
        if dropped_deg_levels:
            print(f"  Skipped DEG levels > max deg_rank ({max_rank}): {dropped_deg_levels}")
        print(f"  include_metrics={sorted(enabled_metrics)}")
        print(f"  sign_tau={sign_tau}, updown_tau={updown_tau}, de_frac={de_frac}, topk_values={topk_levels}")
        if result.empty:
            print("  No metric rows were produced.")
        else:
            if 'pearson_distance_raw' in enabled_metrics:
                print(f"  pearson_distance_raw:    {(result['metric']=='pearson_distance_raw').sum()} entries"
                      f"  (skipped {n_skip_praw} due to zero std)")
            if 'spearman_distance_raw' in enabled_metrics:
                print(f"  spearman_distance_raw:   {(result['metric']=='spearman_distance_raw').sum()} entries"
                      f"  (skipped {n_skip_sraw} due to zero std)")
            if 'pearson_distance_delta' in enabled_metrics:
                print(f"  pearson_distance_delta:  {(result['metric']=='pearson_distance_delta').sum()} entries"
                      f"  (skipped {n_skip_pdelta} due to zero std)")
            if 'spearman_distance_delta' in enabled_metrics:
                print(f"  spearman_distance_delta: {(result['metric']=='spearman_distance_delta').sum()} entries"
                      f"  (skipped {n_skip_sdelta} due to zero std)")
            if 'mean_absolute_error_delta' in enabled_metrics:
                print(f"  mean_absolute_error_delta: {(result['metric']=='mean_absolute_error_delta').sum()} entries")
            if 'cosine_distance_delta' in enabled_metrics:
                print(f"  cosine_distance_delta: {(result['metric']=='cosine_distance_delta').sum()} entries")
            if 'sign_error_delta' in enabled_metrics:
                print(f"  sign_error_delta: {(result['metric']=='sign_error_delta').sum()} entries"
                      f"  (skipped {n_skip_sign} due to empty |true_delta| > sign_tau mask)")
            if 'updown_f1_distance_delta' in enabled_metrics:
                print(f"  updown_f1_distance_delta: {(result['metric']=='updown_f1_distance_delta').sum()} entries")
            if 'de_auroc_distance_delta' in enabled_metrics:
                print(f"  de_auroc_distance_delta: {(result['metric']=='de_auroc_distance_delta').sum()} entries"
                      f"  (skipped {n_skip_de_auc})")
            if 'de_auprc_distance_delta' in enabled_metrics:
                print(f"  de_auprc_distance_delta: {(result['metric']=='de_auprc_distance_delta').sum()} entries"
                      f"  (skipped {n_skip_de_auprc})")
            for k in enabled_topk_jaccard:
                print(f"  topk_jaccard_distance_delta_k{k}: {(result['metric']==f'topk_jaccard_distance_delta_k{k}').sum()} entries")
            for k in enabled_topk_precision:
                print(f"  topk_precision_distance_delta_k{k}: {(result['metric']==f'topk_precision_distance_delta_k{k}').sum()} entries")
            if 'ccc_delta' in enabled_metrics:
                print(f"  ccc_delta: {(result['metric']=='ccc_delta').sum()} entries"
                      f"  (skipped {n_skip_ccc_delta} due to undefined denominator)")

    return result


# ═══════════════════════════════════════════════════════════════════════
# Cell-line clustering by error profile
# ═══════════════════════════════════════════════════════════════════════

def cellline_clustering(df, metric_name='mean_absolute_error', verbose=True,
                        plot=True):
    """Cluster cell lines by their perturbation error vectors.

    For each cell line, builds a vector of mean performance across all common
    perturbations, then computes pairwise distances (correlation & cosine)
    and hierarchical clustering.

    Args:
        df: DataFrame with columns DataSet, perturb, metric, performance.
        metric_name: which metric to use for the error vectors.
        verbose: print stats and distance matrices.
        plot: show dendrograms.

    Returns:
        dict with keys: pivot, stats, corr, dist_corr, dist_cosine
    """
    sub = df[df['metric'] == metric_name]
    agg = sub.groupby(['DataSet', 'perturb'])['performance'].mean().reset_index()
    pivot = agg.pivot(index='perturb', columns='DataSet', values='performance').dropna()

    if verbose:
        print(f"Metric: {metric_name}")
        print(f"{len(pivot)} common perturbations across {pivot.shape[1]} cell lines")

    # Per-cell-line summary
    stats = pd.DataFrame({
        'mean': pivot.mean(),
        'median': pivot.median(),
        'std': pivot.std(),
        'cv': pivot.std() / pivot.mean(),
    }).sort_values('mean')

    if verbose:
        print("\n--- Per-cell-line stats ---")
        print(stats.round(4))

    # Spearman correlation of error vectors
    corr = pivot.corr(method='spearman')
    if verbose:
        print("\n--- Spearman correlation of error vectors ---")
        print(corr.round(3))

    # Distance matrices
    dist_corr_vec = pdist(pivot.T.values, metric='correlation')
    dist_cosine_vec = pdist(pivot.T.values, metric='cosine')
    labels = pivot.columns.tolist()

    dist_corr = pd.DataFrame(squareform(dist_corr_vec),
                              index=labels, columns=labels)
    dist_cosine = pd.DataFrame(squareform(dist_cosine_vec),
                                index=labels, columns=labels)

    if verbose:
        print("\n--- Pairwise correlation distance ---")
        print(dist_corr.round(3))
        print("\n--- Pairwise cosine distance ---")
        print(dist_cosine.round(3))

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, dist_vec, label in [
            (axes[0], dist_corr_vec, 'Correlation distance'),
            (axes[1], dist_cosine_vec, 'Cosine distance'),
        ]:
            Z = linkage(dist_vec, method='average')
            dendrogram(Z, labels=labels, ax=ax)
            ax.set_title(f'{label} ({metric_name})')
            ax.set_ylabel('Distance')
        plt.tight_layout()
        plt.show()

    return {
        'pivot': pivot, 'stats': stats, 'corr': corr,
        'dist_corr': dist_corr, 'dist_cosine': dist_cosine,
    }
