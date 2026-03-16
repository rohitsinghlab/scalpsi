"""
Stage 1: SVD on the residual (interaction) matrix.

Workflow (from notebook):
    from results_modeling_helper import *

    # 1. Build residual matrices
    residuals = build_residual_matrices(gene_perf, method='trainMean', deg=5000)

    # 2. Scree plots  →  pick k
    plot_scree(residuals)

    # 3. Factor loadings
    plot_factor_loadings(residuals, cell_type='K562', top_n=20)

    # 4. Reconstruction-error hardness
    hardness = reconstruction_hardness(residuals, k=10)

    # 5. Compare with specificity (pass pert_features from notebook cell 5)
    compare_hardness(hardness, pivot, pert_features=pert_features)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# ═══════════════════════════════════════════════════════════════════════
# 1. Build delta & residual matrices
# ═══════════════════════════════════════════════════════════════════════

def build_residual_matrices(gene_perf, method="trainMean", deg=5000,
                            common_perts=None):
    """Build additive-model residual matrices R per cell type.

    For each cell type:
        delta_mat : genes × perturbations  (δ_{ij})
        R         : genes × perturbations  (δ_{ij} - μ - α_i - β_j)

    Parameters
    ----------
    gene_perf : DataFrame
        Gene-level performance with columns: dataset, method, split,
        perturbation, gene, mean_true_delta, deg_rank.
    method : str
        Which prediction method to use (default 'trainMean').
    deg : int
        Filter to genes with deg_rank <= deg.
    common_perts : set or None
        If given, restrict to these perturbations.  If None, uses the
        intersection of perturbations across all cell types.

    Returns
    -------
    dict  {cell_type: dict} with keys per cell type:
        'delta_mat'   : DataFrame (genes × perturbations)
        'R'           : ndarray   (genes × perturbations)  residual matrix
        'mu'          : float     grand mean
        'alpha'       : ndarray   gene main effects  (n_genes,)
        'beta'        : ndarray   perturbation main effects  (n_perts,)
        'genes'       : Index     gene labels
        'perts'       : Index     perturbation labels
        'U', 'S', 'Vt': SVD of R  (computed eagerly — cheap at 5000×284)
        'var_explained': ndarray  fraction of variance per component
    """
    sub = gene_perf[
        (gene_perf["method"] == method) & (gene_perf["deg_rank"] <= deg)
    ]

    # Find common perturbations
    if common_perts is None:
        perts_by_cell = {
            c: set(g["perturbation"].unique())
            for c, g in sub.groupby("dataset")
        }
        common_perts = set.intersection(*perts_by_cell.values())
    print(f"Using {len(common_perts)} common perturbations")

    sub = sub[sub["perturbation"].isin(common_perts)]

    results = {}
    for cell_type in sorted(sub["dataset"].unique()):
        ct = sub[sub["dataset"] == cell_type]

        # Average across splits → one value per (gene, perturbation)
        avg = (
            ct.groupby(["gene", "perturbation"])["mean_true_delta"]
            .mean()
            .reset_index()
        )
        delta_mat = avg.pivot(
            index="gene", columns="perturbation", values="mean_true_delta"
        )
        # Some genes are not in top-DEG for every perturbation → NaN.
        # Fill with 0 (negligible delta for that gene under that perturbation).
        n_nan = delta_mat.isna().sum().sum()
        n_total = delta_mat.shape[0] * delta_mat.shape[1]
        if n_nan > 0:
            print(f"    ({cell_type}: filled {n_nan}/{n_total} NaN = "
                  f"{n_nan/n_total:.1%} with 0)")
        delta_mat = delta_mat.fillna(0.0)

        X = delta_mat.values  # genes × perturbations
        mu = X.mean()
        alpha = X.mean(axis=1)  # gene main effect
        beta = X.mean(axis=0)   # perturbation main effect

        # Additive model residual
        R = X - mu - (alpha[:, None] - mu) - (beta[None, :] - mu)
        # Simplifies to: R = X - alpha[:, None] - beta[None, :] + mu

        # Full SVD of R (numpy is more robust than scipy for edge cases)
        U, S, Vt = np.linalg.svd(R, full_matrices=False)

        # Variance explained per component
        total_var = np.sum(S ** 2)
        var_explained = S ** 2 / total_var

        results[cell_type] = {
            "delta_mat": delta_mat,
            "R": R,
            "mu": mu,
            "alpha": alpha,
            "beta": beta,
            "genes": delta_mat.index,
            "perts": delta_mat.columns,
            "U": U,
            "S": S,
            "Vt": Vt,
            "var_explained": var_explained,
        }
        print(
            f"  {cell_type}: {delta_mat.shape[0]} genes × {delta_mat.shape[1]} perts, "
            f"top-1 var explained = {var_explained[0]:.1%}, "
            f"top-10 cumul = {var_explained[:10].sum():.1%}"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════
# 2. Scree plots
# ═══════════════════════════════════════════════════════════════════════

def plot_scree(residuals, max_k=50):
    """Scree plots: variance explained per component and cumulative.

    Parameters
    ----------
    residuals : dict from build_residual_matrices
    max_k : int
        Show up to this many components.
    """
    cell_types = sorted(residuals.keys())
    n = len(cell_types)

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7), squeeze=False)

    for col, ct in enumerate(cell_types):
        ve = residuals[ct]["var_explained"][:max_k]
        cum = np.cumsum(ve)
        x = np.arange(1, len(ve) + 1)

        # Per-component
        ax = axes[0, col]
        ax.bar(x, ve, color="steelblue", alpha=0.7)
        ax.set_title(ct)
        ax.set_xlabel("Component")
        if col == 0:
            ax.set_ylabel("Fraction variance explained")

        # Cumulative
        ax = axes[1, col]
        ax.plot(x, cum, "o-", ms=3, color="steelblue")
        ax.axhline(0.5, ls="--", color="gray", lw=0.8)
        ax.axhline(0.8, ls="--", color="gray", lw=0.8)
        ax.set_xlabel("Component")
        ax.set_ylim(0, 1)
        if col == 0:
            ax.set_ylabel("Cumulative variance explained")

        # Print rank at 50% and 80%
        k50 = np.searchsorted(cum, 0.5) + 1
        k80 = np.searchsorted(cum, 0.8) + 1
        ax.annotate(f"50%: k={k50}", xy=(k50, 0.5), fontsize=8,
                    xytext=(k50 + 3, 0.45), arrowprops=dict(arrowstyle="->", lw=0.5))
        ax.annotate(f"80%: k={k80}", xy=(k80, 0.8), fontsize=8,
                    xytext=(k80 + 3, 0.75), arrowprops=dict(arrowstyle="->", lw=0.5))

    fig.suptitle("Scree plots of residual (interaction) matrix R", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()

    # Summary table
    rows = []
    for ct in cell_types:
        ve = residuals[ct]["var_explained"]
        cum = np.cumsum(ve)
        rows.append({
            "cell_type": ct,
            "top1_%": f"{ve[0]:.2%}",
            "top5_%": f"{cum[4]:.2%}",
            "top10_%": f"{cum[9]:.2%}",
            "top20_%": f"{cum[19]:.2%}",
            "k_for_50%": int(np.searchsorted(cum, 0.5) + 1),
            "k_for_80%": int(np.searchsorted(cum, 0.8) + 1),
        })
    summary = pd.DataFrame(rows)
    print("\nVariance explained summary:")
    print(summary.to_string(index=False))
    return summary


# ═══════════════════════════════════════════════════════════════════════
# 3. Factor loadings
# ═══════════════════════════════════════════════════════════════════════

def plot_factor_loadings(residuals, cell_type, factors=None, top_n=15):
    """Show top gene and perturbation loadings for selected factors.

    Parameters
    ----------
    residuals : dict from build_residual_matrices
    cell_type : str
    factors : list of int (0-indexed) or None (default [0,1,2])
    top_n : int
        Show top/bottom N loadings per factor.
    """
    if factors is None:
        factors = [0, 1, 2]

    r = residuals[cell_type]
    U, S, Vt = r["U"], r["S"], r["Vt"]
    genes = r["genes"]
    perts = r["perts"]

    n_factors = len(factors)
    fig, axes = plt.subplots(n_factors, 2, figsize=(14, 4 * n_factors))
    if n_factors == 1:
        axes = axes[np.newaxis, :]

    for row, k in enumerate(factors):
        ve = r["var_explained"][k]

        # Gene loadings (U[:, k] scaled by S[k])
        gene_loading = U[:, k] * S[k]
        gene_df = pd.Series(gene_loading, index=genes).sort_values()

        ax = axes[row, 0]
        top_genes = pd.concat([gene_df.head(top_n), gene_df.tail(top_n)])
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in top_genes.values]
        ax.barh(range(len(top_genes)), top_genes.values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_genes)))
        ax.set_yticklabels(top_genes.index, fontsize=7)
        ax.set_title(f"Factor {k+1} gene loadings ({ve:.1%} var)")
        ax.axvline(0, color="black", lw=0.5)

        # Perturbation loadings (Vt[k, :] scaled by S[k])
        pert_loading = Vt[k, :] * S[k]
        pert_df = pd.Series(pert_loading, index=perts).sort_values()

        ax = axes[row, 1]
        top_perts = pd.concat([pert_df.head(top_n), pert_df.tail(top_n)])
        colors = ["#d62728" if v < 0 else "#2ca02c" for v in top_perts.values]
        ax.barh(range(len(top_perts)), top_perts.values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_perts)))
        ax.set_yticklabels(top_perts.index, fontsize=7)
        ax.set_title(f"Factor {k+1} perturbation loadings ({ve:.1%} var)")
        ax.axvline(0, color="black", lw=0.5)

    fig.suptitle(f"{cell_type}: Top factor loadings", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()


def get_factor_loadings(residuals, cell_type, factor=0):
    """Return gene and perturbation loadings as sorted DataFrames.

    Useful for downstream enrichment analysis (e.g. pass gene_loadings
    to GSEA as a pre-ranked gene list).

    Returns
    -------
    gene_loadings : Series  (gene → loading, sorted by absolute value)
    pert_loadings : Series  (perturbation → loading, sorted by absolute value)
    """
    r = residuals[cell_type]
    U, S, Vt = r["U"], r["S"], r["Vt"]

    gene_loadings = pd.Series(
        U[:, factor] * S[factor], index=r["genes"], name=f"factor_{factor+1}"
    ).sort_values(key=abs, ascending=False)

    pert_loadings = pd.Series(
        Vt[factor, :] * S[factor], index=r["perts"], name=f"factor_{factor+1}"
    ).sort_values(key=abs, ascending=False)

    return gene_loadings, pert_loadings


# ═══════════════════════════════════════════════════════════════════════
# 4. Reconstruction error → refined hardness score
# ═══════════════════════════════════════════════════════════════════════

def reconstruction_hardness(residuals, k=10):
    """Per-perturbation reconstruction error at rank k.

    For each perturbation j, the reconstruction error is:
        recon_err_j = ||r_{·j} - R̂_{·j}||² / ||r_{·j}||²

    where R̂ is the rank-k approximation of R.

    High recon_err → perturbation j has interaction structure NOT captured
    by the top k shared factors → harder to predict even with structured models.

    Parameters
    ----------
    residuals : dict from build_residual_matrices
    k : int
        Rank for the low-rank approximation.

    Returns
    -------
    DataFrame with columns: perturbation, cell_type, recon_err,
        total_interaction_var, structured_var, idiosyncratic_var
    """
    records = []
    for ct, r in residuals.items():
        U, S, Vt, R = r["U"], r["S"], r["Vt"], r["R"]
        perts = r["perts"]

        # Rank-k approximation
        R_hat = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        E = R - R_hat  # residual after removing low-rank structure

        for j, pert in enumerate(perts):
            col_norm_sq = np.sum(R[:, j] ** 2)
            err_sq = np.sum(E[:, j] ** 2)
            structured_sq = np.sum(R_hat[:, j] ** 2)

            records.append({
                "perturbation": pert,
                "cell_type": ct,
                "recon_err": err_sq / col_norm_sq if col_norm_sq > 0 else np.nan,
                "total_interaction_var": col_norm_sq,
                "structured_var": structured_sq,
                "idiosyncratic_var": err_sq,
            })

    df = pd.DataFrame(records)

    # Also compute mean across cell types
    avg = (
        df.groupby("perturbation")[["recon_err", "total_interaction_var",
                                     "structured_var", "idiosyncratic_var"]]
        .mean()
        .reset_index()
    )
    avg["cell_type"] = "mean"
    df = pd.concat([df, avg], ignore_index=True)

    print(f"Reconstruction hardness at rank k={k}:")
    print(f"  Mean recon_err across perturbations (per cell type):")
    for ct in sorted(residuals.keys()):
        vals = df[df["cell_type"] == ct]["recon_err"]
        print(f"    {ct}: {vals.mean():.3f} ± {vals.std():.3f}")
    vals = df[df["cell_type"] == "mean"]["recon_err"]
    print(f"    MEAN:  {vals.mean():.3f} ± {vals.std():.3f}")

    return df


def reconstruction_hardness_sweep(residuals, k_values=None):
    """Compute reconstruction error for multiple k values.

    Returns a DataFrame: perturbation × k → recon_err (averaged across cell types).
    Useful for seeing how hardness rankings change with k.
    """
    if k_values is None:
        k_values = [1, 2, 5, 10, 20, 50]

    all_results = {}
    for k in k_values:
        df = reconstruction_hardness.__wrapped__(residuals, k) if hasattr(
            reconstruction_hardness, '__wrapped__') else _recon_hardness_quiet(residuals, k)
        avg = df[df["cell_type"] == "mean"][["perturbation", "recon_err"]]
        all_results[k] = avg.set_index("perturbation")["recon_err"]

    sweep = pd.DataFrame(all_results)
    sweep.columns = [f"k={k}" for k in k_values]
    return sweep


def _recon_hardness_quiet(residuals, k):
    """Same as reconstruction_hardness but without printing."""
    records = []
    for ct, r in residuals.items():
        U, S, Vt, R = r["U"], r["S"], r["Vt"], r["R"]
        perts = r["perts"]
        R_hat = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        E = R - R_hat
        for j, pert in enumerate(perts):
            col_norm_sq = np.sum(R[:, j] ** 2)
            err_sq = np.sum(E[:, j] ** 2)
            records.append({
                "perturbation": pert,
                "cell_type": ct,
                "recon_err": err_sq / col_norm_sq if col_norm_sq > 0 else np.nan,
                "total_interaction_var": col_norm_sq,
                "structured_var": np.sum(R_hat[:, j] ** 2),
                "idiosyncratic_var": err_sq,
            })
    df = pd.DataFrame(records)
    avg = (
        df.groupby("perturbation")[["recon_err", "total_interaction_var",
                                     "structured_var", "idiosyncratic_var"]]
        .mean()
        .reset_index()
    )
    avg["cell_type"] = "mean"
    return pd.concat([df, avg], ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════
# 5. Compare with existing specificity / prediction error
# ═══════════════════════════════════════════════════════════════════════

def compare_hardness(hardness_df, pivot, pert_features=None):
    """Compare reconstruction error with specificity and prediction error.

    Parameters
    ----------
    hardness_df : DataFrame from reconstruction_hardness (single k)
    pivot : DataFrame from perturbation_analysis with 'mean_perf' column
        (index = perturbation names as 'perturb')
    pert_features : DataFrame or None
        The pert_features DataFrame from the notebook (cell 5), which contains
        the correctly computed specificity column.  If provided, uses
        pert_features.groupby('perturbation')['specificity'].mean() directly.
        If None, falls back to the SVD-derived proxy (recon_err based).

    Prints Spearman correlations and makes scatter plots.
    """
    # Get the mean-across-cell-types reconstruction error
    recon = hardness_df[hardness_df["cell_type"] == "mean"][
        ["perturbation", "recon_err"]
    ].set_index("perturbation")

    # Get prediction error from pivot
    perf = pivot.reset_index()[["perturb", "mean_perf"]].set_index("perturb")
    perf.index.name = "perturbation"

    # Merge
    merged = recon.join(perf, how="inner")

    rho, p = spearmanr(merged["recon_err"], merged["mean_perf"])
    print(f"Spearman(recon_err, prediction_error): rho={rho:.4f}, p={p:.2e}")
    print(f"  n={len(merged)} perturbations")

    # Specificity: use pert_features if provided, else fall back to SVD proxy
    if pert_features is not None:
        spec_series = (
            pert_features.groupby("perturbation")["specificity"]
            .mean()
            .rename("specificity")
        )
        specificity = spec_series.to_frame()
    else:
        specificity = _compute_specificity(hardness_df)
    merged = merged.join(specificity, how="inner")

    rho_s, p_s = spearmanr(merged["specificity"], merged["mean_perf"])
    rho_r, p_r = spearmanr(merged["recon_err"], merged["mean_perf"])
    rho_rs, p_rs = spearmanr(merged["recon_err"], merged["specificity"])

    print(f"\nCorrelation summary:")
    print(f"  specificity  ↔ prediction_error:  rho={rho_s:.4f}  (p={p_s:.2e})")
    print(f"  recon_err    ↔ prediction_error:  rho={rho_r:.4f}  (p={p_r:.2e})")
    print(f"  recon_err    ↔ specificity:       rho={rho_rs:.4f}  (p={p_rs:.2e})")

    # Scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.scatter(merged["specificity"], merged["mean_perf"], alpha=0.4, s=15)
    ax.set_xlabel("Specificity (original)")
    ax.set_ylabel("Prediction error (mean_perf)")
    ax.set_title(f"Specificity → Error\nrho={rho_s:.3f}")

    ax = axes[1]
    ax.scatter(merged["recon_err"], merged["mean_perf"], alpha=0.4, s=15)
    ax.set_xlabel("Reconstruction error (SVD)")
    ax.set_ylabel("Prediction error (mean_perf)")
    ax.set_title(f"Recon error → Error\nrho={rho_r:.3f}")

    ax = axes[2]
    ax.scatter(merged["specificity"], merged["recon_err"], alpha=0.4, s=15)
    ax.set_xlabel("Specificity (original)")
    ax.set_ylabel("Reconstruction error (SVD)")
    ax.set_title(f"Specificity ↔ Recon error\nrho={rho_rs:.3f}")

    plt.tight_layout()
    plt.show()

    # Show perturbations where specificity and recon_err disagree most
    merged["rank_specificity"] = merged["specificity"].rank()
    merged["rank_recon_err"] = merged["recon_err"].rank()
    merged["rank_diff"] = (merged["rank_specificity"] - merged["rank_recon_err"]).abs()

    print("\nTop 15 perturbations where specificity & recon_err disagree most:")
    top_disagree = merged.sort_values("rank_diff", ascending=False).head(15)
    print(
        top_disagree[["specificity", "recon_err", "mean_perf",
                       "rank_specificity", "rank_recon_err", "rank_diff"]]
        .round(4)
        .to_string()
    )

    return merged


def _compute_specificity(hardness_df):
    """Proxy specificity from SVD decomposition outputs.

    specificity_j = idiosyncratic_var / total_interaction_var
    Averaged across cell types.
    """
    avg = hardness_df[hardness_df["cell_type"] != "mean"].groupby("perturbation").agg(
        total_int=("total_interaction_var", "mean"),
        idio=("idiosyncratic_var", "mean"),
    )
    avg["specificity"] = avg["idio"] / avg["total_int"]
    return avg[["specificity"]]


def compare_hardness_sweep(residuals, pivot, k_values=None):
    """Compare reconstruction hardness at multiple k values with prediction error.

    Shows how the correlation between recon_err and prediction_error
    changes as k increases (more structure removed → purer idiosyncratic signal).
    """
    if k_values is None:
        k_values = [1, 2, 3, 5, 10, 15, 20, 30, 50]

    perf = pivot.reset_index()[["perturb", "mean_perf"]].set_index("perturb")
    perf.index.name = "perturbation"

    results = []
    for k in k_values:
        df = _recon_hardness_quiet(residuals, k)
        recon = df[df["cell_type"] == "mean"][
            ["perturbation", "recon_err"]
        ].set_index("perturbation")
        merged = recon.join(perf, how="inner")
        rho, p = spearmanr(merged["recon_err"], merged["mean_perf"])
        results.append({"k": k, "rho": rho, "p": p, "n": len(merged)})

    res = pd.DataFrame(results)
    print("Spearman(recon_err @ rank k, prediction_error):")
    print(res.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(res["k"], res["rho"], "o-", color="steelblue", ms=6)
    ax.set_xlabel("Rank k (number of shared factors removed)")
    ax.set_ylabel("Spearman rho with prediction error")
    ax.set_title("How well does residual interaction predict hardness?")
    ax.axhline(0, ls="--", color="gray", lw=0.5)

    # Add annotation at the peak
    best = res.loc[res["rho"].abs().idxmax()]
    ax.annotate(
        f"best: k={int(best['k'])}, rho={best['rho']:.3f}",
        xy=(best["k"], best["rho"]),
        xytext=(best["k"] + 5, best["rho"] - 0.05),
        arrowprops=dict(arrowstyle="->", lw=0.5),
        fontsize=9,
    )
    plt.tight_layout()
    plt.show()

    return res


# ═══════════════════════════════════════════════════════════════════════
# 6. Cross-cell-type factor comparison
# ═══════════════════════════════════════════════════════════════════════

def cross_celltype_factor_similarity(residuals, n_factors=10):
    """Compare whether the same latent factors appear across cell types.

    Uses absolute cosine similarity between perturbation loading vectors
    (columns of Vt) across cell types.

    This answers: are the shared interaction programs conserved?
    """
    cell_types = sorted(residuals.keys())
    from itertools import combinations

    # Build perturbation loading matrices (n_factors × n_perts) per cell type
    # Align to same perturbation ordering
    ref_perts = residuals[cell_types[0]]["perts"]
    V_dict = {}
    for ct in cell_types:
        r = residuals[ct]
        # Ensure same perturbation ordering
        perm = [list(r["perts"]).index(p) for p in ref_perts]
        Vt_aligned = r["Vt"][:n_factors, :][:, perm]
        V_dict[ct] = Vt_aligned  # (n_factors × n_perts)

    # For each pair of cell types, compute best-match cosine similarity
    # between their factors (using Hungarian matching)
    from scipy.optimize import linear_sum_assignment

    pair_results = []
    for ct1, ct2 in combinations(cell_types, 2):
        V1 = V_dict[ct1]
        V2 = V_dict[ct2]

        # Cosine similarity matrix (n_factors × n_factors)
        # Rows = ct1 factors, Cols = ct2 factors
        cos_sim = np.abs(V1 @ V2.T)  # abs because sign is arbitrary in SVD
        # Normalize
        norms1 = np.linalg.norm(V1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(V2, axis=1, keepdims=True)
        cos_sim = cos_sim / (norms1 @ norms2.T + 1e-10)

        # Hungarian matching to find best 1-to-1 factor alignment
        row_ind, col_ind = linear_sum_assignment(-cos_sim)
        matched_sims = cos_sim[row_ind, col_ind]

        pair_results.append({
            "pair": f"{ct1} vs {ct2}",
            "mean_matched_cosine": matched_sims.mean(),
            "min_matched_cosine": matched_sims.min(),
            "max_matched_cosine": matched_sims.max(),
            "cos_sim_matrix": cos_sim,
            "matching": list(zip(row_ind, col_ind, matched_sims)),
        })

    # Print summary
    print(f"Cross-cell-type factor similarity (top {n_factors} factors):")
    print(f"{'Pair':<25s}  {'Mean cos':>8s}  {'Min':>6s}  {'Max':>6s}")
    for r in pair_results:
        print(f"  {r['pair']:<23s}  {r['mean_matched_cosine']:8.3f}  "
              f"{r['min_matched_cosine']:6.3f}  {r['max_matched_cosine']:6.3f}")

    # Heatmap of one example pair
    fig, axes = plt.subplots(1, min(3, len(pair_results)),
                              figsize=(5 * min(3, len(pair_results)), 4))
    if len(pair_results) == 1:
        axes = [axes]
    for ax, r in zip(axes, pair_results[:3]):
        im = ax.imshow(r["cos_sim_matrix"], cmap="Blues", vmin=0, vmax=1)
        ax.set_xlabel(r["pair"].split(" vs ")[1] + " factors")
        ax.set_ylabel(r["pair"].split(" vs ")[0] + " factors")
        ax.set_title(f"{r['pair']}\nmean={r['mean_matched_cosine']:.3f}")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.show()

    return pair_results


# ═══════════════════════════════════════════════════════════════════════
# 7. Gene error vs perturbation hardness — four-metric comparison
# ═══════════════════════════════════════════════════════════════════════

def plot_gene_error_vs_hardness(gene_perf, pivot, cell_type,
                                 method="trainMean", deg=5000,
                                 norm_epsilon=0.1, perf_col=None,
                                 compute_baseline_norm=True):
    """2×2 scatter: per-perturbation mean gene error vs perturbation hardness.

    Four metrics in a 2×2 grid:
        [0,0] mean abs_error_delta       (unsigned magnitude)
        [0,1] mean error_delta           (signed: pred_delta − true_delta)
        [1,0] norm abs_error_delta       (÷ |true_delta| + ε)
        [1,1] norm error_delta           (signed, normalized)

    Signed error_delta shows *bias*: positive → model over-predicted the delta,
    negative → model under-predicted.  Unsigned shows raw magnitude.

    Each point = one perturbation, aggregated (mean) over its top-DEG genes.
    x-axis = mean gene error metric   |   y-axis = perturbation prediction error

    Parameters
    ----------
    gene_perf : DataFrame
    pivot : DataFrame from perturbation_analysis (has mean_perf + per-cell columns)
    cell_type : str
    method : str
    deg : int
    norm_epsilon : float
        Added to denominator to prevent division by ~0.
        Default 0.1 (in log1p units — roughly a 10 % expression change).
    perf_col : str or None
        Column from pivot to use as perturbation prediction error (y-axis).
        If None (default), uses cell_type column if present in pivot,
        otherwise falls back to 'mean_perf' (cross-cell-type average).
    compute_baseline_norm : bool
        If True (default), add 'norm_by_baseline' column to avg:
            abs_error_delta / (|baseline_expr| + ε)
        where baseline_expr = mean_true − mean_true_delta (control expression).
        More stable than norm_abs_error_delta when mean_true_delta ≈ 0.
        Set False to skip (saves a tiny bit of time; omits the column).

    Returns
    -------
    per_pert : DataFrame  (one row per perturbation)
    avg      : DataFrame  (one row per gene×perturbation)
    """
    # ── 1. Filter and average across splits ─────────────────────────────
    pivot_perts = set(pivot.reset_index()["perturb"])
    sub = gene_perf[
        (gene_perf["dataset"] == cell_type) &
        (gene_perf["method"] == method) &
        (gene_perf["deg_rank"] <= deg) &
        (gene_perf["perturbation"].isin(pivot_perts))
    ]

    avg = (
        sub.groupby(["gene", "perturbation"])
        .agg(
            abs_error_delta=("abs_error_delta", "mean"),
            mean_pred_delta=("mean_pred_delta", "mean"),
            mean_true_delta=("mean_true_delta", "mean"),
            mean_true       =("mean_true",       "mean"),
        )
        .reset_index()
    )

    # Signed error delta: positive = model over-predicted
    avg["error_delta"] = avg["mean_pred_delta"] - avg["mean_true_delta"]

    # ── 2. Normalized versions ───────────────────────────────────────────
    denom = avg["mean_true_delta"].abs() + norm_epsilon
    avg["norm_abs_error_delta"] = avg["abs_error_delta"] / denom
    avg["norm_error_delta"]     = avg["error_delta"]     / denom

    # Optional: normalise by baseline (control) expression instead of delta.
    # baseline_expr = unperturbed expression = mean_true − mean_true_delta.
    # More stable denominator than |mean_true_delta| (which is ~0 for most
    # genes under most perturbations, collapsing to ε and reintroducing bias).
    if compute_baseline_norm:
        baseline_expr = avg["mean_true"] - avg["mean_true_delta"]
        avg["norm_by_baseline"] = avg["abs_error_delta"] / (baseline_expr.abs() + norm_epsilon)

    # ── 3. Aggregate to perturbation level ───────────────────────────────
    per_pert = (
        avg.groupby("perturbation")
        .agg(
            mean_abs_error_delta=("abs_error_delta",      "mean"),
            mean_error_delta    =("error_delta",          "mean"),
            mean_norm_abs_error =("norm_abs_error_delta", "mean"),
            mean_norm_error     =("norm_error_delta",     "mean"),
            n_genes             =("gene",                 "count"),
        )
        .reset_index()
    )

    # ── 4. Merge with perturbation prediction error ──────────────────────
    # Auto-detect: use cell-type-specific column if available
    if perf_col is None:
        perf_col = cell_type if cell_type in pivot.columns else "mean_perf"
    print(f"Using prediction error column: '{perf_col}'")

    perf = pivot.reset_index()[["perturb", perf_col]].rename(
        columns={"perturb": "perturbation", perf_col: "mean_perf"}
    )
    per_pert = per_pert.merge(perf, on="perturbation", how="inner")
    print(f"{cell_type}: {len(per_pert)} perturbations, "
          f"{avg['gene'].nunique()} genes, "
          f"{len(avg)} gene×pert pairs")

    # ── 5. 2×2 scatter plots ─────────────────────────────────────────────
    metrics = [
        ("mean_abs_error_delta", "Mean |abs_error_delta|\n(unsigned)"),
        ("mean_error_delta",     "Mean error_delta\n(signed: pred − true)"),
        ("mean_norm_abs_error",  "Mean norm |abs_error_delta|\n(÷ |true_delta| + ε)"),
        ("mean_norm_error",      "Mean norm error_delta\n(signed, ÷ |true_delta| + ε)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, metrics):
        x = per_pert[col]
        y = per_pert["mean_perf"]
        rho, p = spearmanr(x, y)

        ax.scatter(x, y, alpha=0.5, s=18, color="steelblue")
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Perturbation prediction error (mean_perf)", fontsize=9)
        ax.set_title(f"rho={rho:.3f}  p={p:.1e}", fontsize=10)
        # Mark zero for signed plots
        if "signed" in label:
            ax.axvline(0, color="gray", lw=0.6, ls="--")

    fig.suptitle(
        f"{cell_type} — mean gene error vs perturbation hardness\n"
        f"(each point = 1 perturbation, averaged over ~{int(per_pert['n_genes'].mean())} genes, "
        f"ε={norm_epsilon})",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()

    # ── 6. Print summary correlations ────────────────────────────────────
    print("\nSpearman(mean gene error, pert prediction error):")
    for col, label in metrics:
        rho, p = spearmanr(per_pert[col], per_pert["mean_perf"])
        print(f"  {label.split(chr(10))[0]:42s}: rho={rho:.4f}  p={p:.2e}")

    # Return both: per_pert (one row per pert) and avg (one row per gene×pert)
    return per_pert, avg


# ═══════════════════════════════════════════════════════════════════════
# 8. Quadrant analysis — easy/hard pert × easy/hard gene
# ═══════════════════════════════════════════════════════════════════════

def gene_pert_quadrants(avg, per_pert,
                        pert_top_frac=0.33,
                        gene_top_frac=0.33,
                        error_col="abs_error_delta",
                        top_n=20,
                        show_hard_only=False):
    """Classify gene×pert pairs into 4 quadrants.

    Quadrant definitions (based on global quantile thresholds):
        hard_pert / hard_gene  : pert in top pert_top_frac mean_perf
                                 AND gene error in top gene_top_frac
        hard_pert / easy_gene  : pert in top pert_top_frac mean_perf
                                 AND gene error in bottom gene_top_frac
        easy_pert / hard_gene  : pert in bottom pert_top_frac mean_perf
                                 AND gene error in top gene_top_frac
        easy_pert / easy_gene  : pert in bottom pert_top_frac mean_perf
                                 AND gene error in bottom gene_top_frac
        middle                 : neither extreme

    The most biologically interesting quadrant is **easy_pert/hard_gene**:
    the perturbation is globally easy to predict, but this specific gene
    shows an unexpected/idiosyncratic response.

    Parameters
    ----------
    avg : DataFrame
        Gene×pert level data returned by plot_gene_error_vs_hardness
        (columns: gene, perturbation, abs_error_delta, error_delta, …).
    per_pert : DataFrame
        Perturbation level data returned by plot_gene_error_vs_hardness
        (columns: perturbation, mean_perf, …).
    pert_top_frac : float
        Fraction of perturbations to label "hard" (top) and "easy" (bottom).
        Default 0.33 → top/bottom tercile.
    gene_top_frac : float
        Fraction of gene errors to label "hard" (top) and "easy" (bottom).
        Global threshold across all gene×pert pairs.
    error_col : str
        Gene-level error column to use for gene hardness.
    top_n : int
        Number of top genes/perturbations to print per quadrant.

    Returns
    -------
    classified : DataFrame with added 'quadrant' column (one row per gene×pert)
    """
    # ── Merge gene-level errors with perturbation hardness ───────────────
    merged = avg.merge(
        per_pert[["perturbation", "mean_perf"]], on="perturbation", how="inner"
    )

    # ── Thresholds ───────────────────────────────────────────────────────
    pert_thresh_hard = merged["mean_perf"].quantile(1 - pert_top_frac)
    pert_thresh_easy = merged["mean_perf"].quantile(pert_top_frac)
    gene_thresh_hard = merged[error_col].quantile(1 - gene_top_frac)
    gene_thresh_easy = merged[error_col].quantile(gene_top_frac)

    print(f"Thresholds (top/bottom {pert_top_frac:.0%}):")
    print(f"  Pert mean_perf  — hard ≥ {pert_thresh_hard:.4f},  "
          f"easy ≤ {pert_thresh_easy:.4f}")
    print(f"  Gene {error_col} — hard ≥ {gene_thresh_hard:.4f},  "
          f"easy ≤ {gene_thresh_easy:.4f}")

    # ── Vectorised quadrant assignment ───────────────────────────────────
    pert_hard = merged["mean_perf"] >= pert_thresh_hard
    pert_easy = merged["mean_perf"] <= pert_thresh_easy
    gene_hard = merged[error_col]   >= gene_thresh_hard
    gene_easy = merged[error_col]   <= gene_thresh_easy

    merged["quadrant"] = "middle"
    merged.loc[pert_hard & gene_hard, "quadrant"] = "hard_pert/hard_gene"
    merged.loc[pert_hard & gene_easy, "quadrant"] = "hard_pert/easy_gene"
    merged.loc[pert_easy & gene_hard, "quadrant"] = "easy_pert/hard_gene"
    merged.loc[pert_easy & gene_easy, "quadrant"] = "easy_pert/easy_gene"

    # ── Summary table ────────────────────────────────────────────────────
    counts = merged["quadrant"].value_counts()
    print(f"\nQuadrant counts (total {len(merged):,} gene×pert pairs):")
    quad_order = [
        "hard_pert/hard_gene", "hard_pert/easy_gene",
        "easy_pert/hard_gene", "easy_pert/easy_gene", "middle",
    ]
    for q in quad_order:
        n = counts.get(q, 0)
        print(f"  {q:<25s}: {n:6,}  ({n/len(merged):.1%})")

    # ── Per-gene quadrant fraction breakdown ─────────────────────────────
    # For each gene: fraction of its 284 pairs that fall in each quadrant.
    # This avoids the "universally high error" confound — a gene ranked high
    # by mean error in easy/hard might just have high error everywhere.
    gene_counts = (
        merged.groupby(["gene", "quadrant"])
        .size()
        .unstack(fill_value=0)
    )
    gene_total = merged.groupby("gene").size().rename("n_total")
    gene_counts = gene_counts.join(gene_total)

    # Fraction of pairs per quadrant
    frac_cols = [c for c in gene_counts.columns if c != "n_total"]
    gene_frac = gene_counts[frac_cols].div(gene_counts["n_total"], axis=0)
    gene_frac["n_total"] = gene_counts["n_total"]

    print(f"\nTop {top_n} genes by FRACTION of pairs in each quadrant:")
    quad_labels = [
        ("easy_pert/hard_gene", "idiosyncratic — hard error in easy perts"),
        ("hard_pert/hard_gene", "universally hard"),
    ] if show_hard_only else [
        ("easy_pert/hard_gene", "idiosyncratic — hard error in easy perts"),
        ("hard_pert/hard_gene", "universally hard"),
        ("hard_pert/easy_gene", "robust       — low error even in hard perts"),
        ("easy_pert/easy_gene", "universally easy"),
    ]
    for q, label in quad_labels:
        if q not in gene_frac.columns:
            continue
        top = gene_frac.nlargest(top_n, q)
        print(f"\n  [{q}] — {label}")
        for gene, row in top.iterrows():
            print(f"    {gene:22s}: {row[q]:.0%} of pairs  (n_total={int(row['n_total'])})")

    # ── Scatter: gene error vs pert hardness, coloured by quadrant ───────
    quad_colors = {
        "hard_pert/hard_gene":  "#d62728",   # red
        "hard_pert/easy_gene":  "#ff7f0e",   # orange
        "easy_pert/hard_gene":  "#1f77b4",   # blue
        "easy_pert/easy_gene":  "#2ca02c",   # green
        "middle":               "#cccccc",   # grey
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for q, color in quad_colors.items():
        sub = merged[merged["quadrant"] == q]
        if len(sub) == 0:
            continue
        # Downsample for speed if large
        if len(sub) > 5000:
            sub = sub.sample(5000, random_state=42)
        ax.scatter(
            sub["mean_perf"], sub[error_col],
            s=4, alpha=0.3, color=color, label=q, rasterized=True
        )

    # Threshold lines
    ax.axvline(pert_thresh_hard, color="red",   lw=0.8, ls="--", alpha=0.6)
    ax.axvline(pert_thresh_easy, color="green", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(gene_thresh_hard, color="red",   lw=0.8, ls="--", alpha=0.6)
    ax.axhline(gene_thresh_easy, color="green", lw=0.8, ls="--", alpha=0.6)

    ax.set_xlabel("Perturbation hardness (mean_perf, higher = harder)", fontsize=11)
    ax.set_ylabel(f"Gene error ({error_col})", fontsize=11)
    ax.set_title(
        f"Gene×pert quadrant classification\n"
        f"(top/bottom {pert_top_frac:.0%} pert × top/bottom {gene_top_frac:.0%} gene error)",
        fontsize=11,
    )
    ax.legend(markerscale=3, fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.show()

    # merged  : one row per gene×pert, with 'quadrant' column
    # gene_frac: one row per gene, fraction of pairs in each quadrant
    return merged, gene_frac


# ═══════════════════════════════════════════════════════════════════════
# 9. Method complementarity analysis
# ═══════════════════════════════════════════════════════════════════════

def method_complementarity(df, metric="spearman_distance_delta", deg=10000,
                           agg_func='median', baseline='trainMean',
                           exclude_methods=None):
    """Analyze whether methods have complementary strengths across perturbations.

    Part 1: Pairwise Spearman correlation of per-perturbation performance
        across methods (high = methods struggle on the same perturbations).
    Part 2: Oracle "best-of" ensemble — for each perturbation, pick the
        method with the best (lowest) performance and compare distributions.

    Parameters
    ----------
    df : DataFrame
        Long-form performance data with columns: metric, DEG, DataSet,
        method, perturb, performance, split.
    metric : str
    deg : int
    agg_func : str
        How to aggregate across splits ('median' or 'mean').
    baseline : str
        Method to compare against individually (default 'trainMean').
    exclude_methods : list or None
        Methods to exclude (e.g. ['controlMean']).

    Returns
    -------
    pivot_all : DataFrame  (index = DataSet × perturb, columns = methods + oracle_best)
    """
    import itertools
    from scipy.stats import spearmanr as _spearmanr, wilcoxon

    if exclude_methods is None:
        exclude_methods = ['controlMean']

    sub = df[(df.metric == metric) & (df.DEG == deg)]

    # Restrict to shared perturbations across all cell types
    all_datasets = sub['DataSet'].unique()
    perturb_counts = sub.groupby('perturb')['DataSet'].nunique()
    shared_perturbs = perturb_counts[perturb_counts == len(all_datasets)].index
    sub = sub[sub['perturb'].isin(shared_perturbs)]

    agg = sub.groupby(['DataSet', 'perturb', 'method'])['performance'].agg(agg_func).reset_index()
    pivot_all = agg.pivot_table(index=['DataSet', 'perturb'], columns='method', values='performance')

    methods = [m for m in pivot_all.columns if m not in exclude_methods]
    pivot_all = pivot_all[methods]

    # ── Part 1: Pairwise correlations ────────────────────────────────────
    print("=" * 60)
    print(f"Method complementarity ({metric}, DEG={deg}, agg={agg_func})")
    print(f"  {len(shared_perturbs)} shared perturbations across {len(all_datasets)} cell types")
    print("=" * 60)

    print(f"\nPairwise Spearman correlation of per-perturbation performance:")
    corr_records = []
    for m1, m2 in itertools.combinations(methods, 2):
        valid = pivot_all[[m1, m2]].dropna()
        r, p = _spearmanr(valid[m1], valid[m2])
        corr_records.append({'method_1': m1, 'method_2': m2, 'rho': r, 'p': p, 'n': len(valid)})
        print(f"  {m1:>12s} vs {m2:<12s}: rho={r:.3f} (p={p:.2e}, n={len(valid)})")

    # ── Part 2: Oracle best-of ensemble ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Oracle best-of ensemble vs individual methods")
    print("=" * 60)

    pivot_all['oracle_best'] = pivot_all[methods].min(axis=1)
    pivot_all['best_method'] = pivot_all[methods].idxmin(axis=1)

    print("\nHow often each method is the best:")
    counts = pivot_all['best_method'].value_counts()
    for m, c in counts.items():
        print(f"  {m:>12s}: {c:4d}  ({c / len(pivot_all):.1%})")

    plot_methods = methods + ['oracle_best']
    print(f"\nMedian {metric} (lower is better):")
    for m in plot_methods:
        print(f"  {m:>15s}: {pivot_all[m].median():.4f}")

    print(f"\nWilcoxon signed-rank test (oracle_best vs each method):")
    for m in methods:
        valid = pivot_all[[m, 'oracle_best']].dropna()
        stat, p = wilcoxon(valid[m], valid['oracle_best'])
        median_diff = (valid[m] - valid['oracle_best']).median()
        print(f"  oracle vs {m:>12s}: median improvement={median_diff:.4f}, p={p:.2e}")

    # ── Plots ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    box_data = [pivot_all[m].dropna().values for m in plot_methods]
    axes[0].boxplot(box_data, labels=plot_methods, vert=True)
    axes[0].set_ylabel(metric)
    axes[0].set_title('Per-perturbation performance (lower is better)')
    axes[0].tick_params(axis='x', rotation=45)

    # CDF plot
    for m in plot_methods:
        vals = np.sort(pivot_all[m].dropna().values)
        axes[1].plot(vals, np.linspace(0, 1, len(vals)), label=m,
                     linewidth=3 if m == 'oracle_best' else 1.5,
                     linestyle='--' if m == 'oracle_best' else '-')
    axes[1].set_xlabel(metric)
    axes[1].set_ylabel('CDF')
    axes[1].set_title('CDF of performance')
    axes[1].legend()

    fig.suptitle(f"Method complementarity — {metric} (DEG={deg})", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()

    return pivot_all
