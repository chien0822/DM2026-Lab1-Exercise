from typing import Optional
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import VarianceThreshold
from langchain_core.tools import tool


def _compute_feature_correlations(
    feature_matrix,
    feature_names,
    labels,
    target_class,
    method="pearson",
    max_terms=None,
):
    """Helper function to compute feature-label correlations (Pearson or Spearman) for target_class (one-vs-rest).

    Supports max_terms pre-filtering by variance (deterministic tie-breaking).
    Returns (sorted_records, n_zero_variance, n_considered, error_message).
    """
    if labels is None or len(labels) == 0:
        return (
            None,
            0,
            0,
            "No labels found in session. Please load dataset and build DTM first.",
        )

    # Binary one-vs-rest array
    y = (labels == target_class).astype(int)
    n_unique = len(np.unique(y))
    if n_unique < 2:
        return (
            None,
            0,
            0,
            f"Target class '{target_class}' not valid (y has {n_unique} unique values).",
        )

    n_docs = feature_matrix.shape[0]

    # Calculate variances directly on sparse matrix for pre-filtering and zero-variance check
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(feature_matrix)
    variances = selector.variances_

    # Pair indices, terms, variances
    indexed_variances = [
        (i, name, float(v)) for i, (name, v) in enumerate(zip(feature_names, variances))
    ]

    # Exclude zero variance
    active_terms = [(i, name, v) for i, name, v in indexed_variances if v > 0]
    n_zero_variance = len(indexed_variances) - len(active_terms)

    if not active_terms:
        return [], n_zero_variance, 0, None

    # Pre-filter by max_terms highest variance (deterministic tie-breaking by term name)
    active_terms_sorted = sorted(active_terms, key=lambda x: (-x[2], x[1]))
    if max_terms is not None and max_terms > 0 and len(active_terms_sorted) > max_terms:
        selected_terms = active_terms_sorted[:max_terms]
    else:
        selected_terms = active_terms_sorted

    n_considered = len(selected_terms)
    selected_indices = [idx for idx, _, _ in selected_terms]
    selected_names = [name for _, name, _ in selected_terms]

    # Slice feature matrix columns for selected terms
    X_sub = feature_matrix[:, selected_indices]

    if method == "pearson":
        # Vectorized Pearson correlation on sparse matrix X and binary y
        y_mean = y.mean()
        y_std = y.std(ddof=0)

        X_means = np.array(X_sub.mean(axis=0)).ravel()
        X_sq_means = np.array(X_sub.power(2).mean(axis=0)).ravel()
        X_stds = np.sqrt(np.maximum(0.0, X_sq_means - X_means ** 2))

        X_y_means = np.array(X_sub.T.dot(y)).ravel() / n_docs
        covs = X_y_means - X_means * y_mean

        with np.errstate(divide="ignore", invalid="ignore"):
            r_all = covs / (X_stds * y_std)

        df = n_docs - 2
        records = []
        for term, r, std in zip(selected_names, r_all, X_stds):
            if std == 0 or np.isnan(r):
                continue

            r = float(np.clip(r, -1.0, 1.0))
            if abs(r) >= 1.0:
                p = 0.0
            else:
                t_stat = r * np.sqrt(df / (1.0 - r ** 2))
                p = float(2.0 * stats.t.sf(np.abs(t_stat), df))

            records.append((term, r, p, abs(r)))

    elif method == "spearman":
        # Spearman correlation = Pearson correlation on ranked values
        X_dense = X_sub.toarray()
        X_rank = stats.rankdata(X_dense, axis=0)

        y_mean = y.mean()
        y_std = y.std(ddof=0)

        X_rank_means = X_rank.mean(axis=0)
        X_rank_stds = X_rank.std(axis=0, ddof=0)

        covs = (X_rank.T @ y) / n_docs - X_rank_means * y_mean

        with np.errstate(divide="ignore", invalid="ignore"):
            r_all = covs / (X_rank_stds * y_std)

        df = n_docs - 2
        records = []
        for term, r, std in zip(selected_names, r_all, X_rank_stds):
            if std == 0 or np.isnan(r):
                continue

            r = float(np.clip(r, -1.0, 1.0))
            if abs(r) >= 1.0:
                p = 0.0
            else:
                t_stat = r * np.sqrt(df / (1.0 - r ** 2))
                p = float(2.0 * stats.t.sf(np.abs(t_stat), df))

            records.append((term, r, p, abs(r)))
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    # Sort by absolute correlation |r| descending, breaking ties alphabetically by term
    sorted_records = sorted(records, key=lambda x: (-x[3], x[0]))
    return sorted_records, n_zero_variance, n_considered, None


def make_tools(session):
    @tool
    def variance_filter_tool(threshold: float = 0.15) -> dict:
        """Filters features based on variance across all documents in the DTM.

        Requires build_dtm_tool to have run first to populate session.feature_matrix and session.feature_names.
        Filters out terms with variance less than or equal to threshold (keeps terms with variance > threshold).
        Does not modify session.feature_matrix in place.

        Args:
            threshold: Variance threshold cutoff (default 0.15, matching Master notebook). Terms with variance > threshold are kept.

        Returns:
            dict: Summary containing kept/removed term counts, variance samples, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No DTM found in session. Please run build_dtm_tool first."
            }

        # Compute variances directly on sparse matrix using scikit-learn's VarianceThreshold (ddof=0)
        selector = VarianceThreshold(threshold=0.0)
        selector.fit(session.feature_matrix)
        variances = selector.variances_

        # Pair terms with calculated variance
        term_variances = [
            (term, float(v))
            for term, v in zip(session.feature_names, variances)
        ]

        # Sort all terms by variance descending, breaking ties alphabetically
        sorted_terms = sorted(term_variances, key=lambda x: (-x[1], x[0]))

        # Prepare full_report DataFrame for visualize_result_tool
        full_report = pd.DataFrame(
            sorted_terms, columns=["term", "variance"]
        )

        # Categorize into kept and removed sets based on strict inequality (> threshold)
        kept = [(t, v) for t, v in sorted_terms if v > threshold]
        removed = [(t, v) for t, v in sorted_terms if v <= threshold]

        # Highest-variance kept terms (top 20)
        kept_sample = [t for t, v in kept[:20]]

        # Lowest-variance removed terms (bottom 20, sorted ascending by variance)
        removed_sorted_asc = sorted(removed, key=lambda x: (x[1], x[0]))
        removed_sample = [t for t, v in removed_sorted_asc[:20]]

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "n_kept": len(kept),
            "n_removed": len(removed),
            "kept_terms_sample": kept_sample,
            "removed_terms_sample": removed_sample,
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=full_report,
        )

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str, top_n: int = 15
    ) -> dict:
        """Computes Pearson correlation (r and p-value) between each feature's count and target_class (one-vs-rest).

        Requires build_dtm_tool to have run first to populate session.feature_matrix, session.feature_names, and session.labels.
        Excludes zero-variance terms from ranking. Ranks remaining terms by absolute correlation |r| descending.

        Args:
            target_class: Name of target category label to correlate features against (one-vs-rest).
            top_n: Number of top correlated terms by |r| to include in summary (default 15, matching Master notebook).

        Returns:
            dict: Summary containing target class, total terms, zero variance excluded count, top_n correlations, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No DTM found in session. Please run build_dtm_tool first."
            }

        if top_n < 1:
            return {"error": "top_n must be an integer >= 1."}

        sorted_records, n_zero_var, _, err = _compute_feature_correlations(
            session.feature_matrix,
            session.feature_names,
            session.labels,
            target_class,
            method="pearson",
        )
        if err:
            return {"error": err}

        # Prepare full_report DataFrame for visualize_result_tool
        report_rows = [(term, r, p) for term, r, p, _ in sorted_records]
        full_report = pd.DataFrame(
            report_rows, columns=["term", "pearson_r", "p_value"]
        )

        # Format p-values with 3 significant digits for JSON summary slice
        top_slice = sorted_records[:top_n]
        top_correlations = {
            term: {"r": round(r, 4), "p": float(f"{p:.3g}")}
            for term, r, p, _ in top_slice
        }

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "total_terms": len(session.feature_names),
            "ranked_terms": len(sorted_records),
            "zero_variance_excluded": n_zero_var,
            "top_n": top_n,
            "top_correlations": top_correlations,
        }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={
                "target_class": target_class,
                "top_n": top_n,
            },
            summary=summary,
            full_report=full_report,
        )

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str, top_n: int = 15, max_terms: Optional[int] = 1500
    ) -> dict:
        """Computes Spearman rank correlation (rho and p-value) between feature counts and target_class (one-vs-rest).

        Requires build_dtm_tool to have run first to populate session.feature_matrix, session.feature_names, and session.labels.
        Pre-filters top max_terms by variance (default 1500, matching Master notebook; None for all terms), then ranks terms by absolute correlation |rho| descending.

        Args:
            target_class: Name of target category label to correlate features against (one-vs-rest).
            top_n: Number of top correlated terms by |rho| to include in summary (default 15, matching Master notebook).
            max_terms: Maximum number of highest-variance terms to pre-select before computing rank correlation (default 1500, None for all terms).

        Returns:
            dict: Summary containing target class, total terms, terms considered, top_n correlations, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No DTM found in session. Please run build_dtm_tool first."
            }

        if top_n < 1:
            return {"error": "top_n must be an integer >= 1."}

        sorted_records, n_zero_var, n_considered, err = _compute_feature_correlations(
            session.feature_matrix,
            session.feature_names,
            session.labels,
            target_class,
            method="spearman",
            max_terms=max_terms,
        )
        if err:
            return {"error": err}

        # Prepare full_report DataFrame for visualize_result_tool
        report_rows = [(term, r, p) for term, r, p, _ in sorted_records]
        full_report = pd.DataFrame(
            report_rows, columns=["term", "spearman_r", "p_value"]
        )

        # Format p-values with 3 significant digits for JSON summary slice
        top_slice = sorted_records[:top_n]
        top_correlations = {
            term: {"r": round(r, 4), "p": float(f"{p:.3g}")}
            for term, r, p, _ in top_slice
        }

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "total_terms": len(session.feature_names),
            "terms_considered": n_considered,
            "zero_variance_excluded": n_zero_var,
            "top_n": top_n,
            "top_correlations": top_correlations,
        }

        session.store_result(
            tool_name="spearman_filter_tool",
            args={
                "target_class": target_class,
                "top_n": top_n,
                "max_terms": max_terms,
            },
            summary=summary,
            full_report=full_report,
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
