from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool

def make_tools(session):
    """
    Factory function to instantiate tools bound to the given pipeline session.
    """

    @tool
    def cosine_similarity_tool(
        doc_index_1: Optional[int] = None,
        doc_index_2: Optional[int] = None,
        text_1: Optional[str] = None,
        text_2: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Computes the cosine similarity between two documents.

        Can operate either on existing documents by 0-based positional row indices
        in the Document-Term Matrix (doc_index_1, doc_index_2) or on arbitrary raw
        text strings (text_1, text_2) vectorized using the fitted CountVectorizer
        from session.artifacts['count_vectorizer'].

        Args:
            doc_index_1: 0-based positional row index of the first document.
            doc_index_2: 0-based positional row index of the second document.
            text_1: First raw text string to compare.
            text_2: Second raw text string to compare.

        Returns:
            Dict containing result_id, cosine_similarity, mode, categories (for index mode),
            zero_vector flag, and optionally note or error description.
        """
        result_id = session.next_result_id("cosine_sim")

        has_indices = doc_index_1 is not None and doc_index_2 is not None
        has_texts = text_1 is not None and text_2 is not None

        # Validate input modes
        if (has_indices and has_texts) or (not has_indices and not has_texts) or (doc_index_1 is not None and doc_index_2 is None) or (doc_index_1 is None and doc_index_2 is not None) or (text_1 is not None and text_2 is None) or (text_1 is None and text_2 is not None):
            summary = {
                "result_id": result_id,
                "error": "Provide either both doc_index_1 and doc_index_2, OR both text_1 and text_2."
            }
            session.store_result("cosine_similarity_tool", {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2, "text_1": text_1, "text_2": text_2}, summary)
            return summary

        if has_indices:
            if session.feature_matrix is None:
                summary = {
                    "result_id": result_id,
                    "error": "Document-term matrix has not been built yet. Call build_dtm_tool first."
                }
                session.store_result("cosine_similarity_tool", {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2}, summary)
                return summary

            n_rows = session.feature_matrix.shape[0]
            if doc_index_1 < 0 or doc_index_1 >= n_rows or doc_index_2 < 0 or doc_index_2 >= n_rows:
                summary = {
                    "result_id": result_id,
                    "error": f"Document index out of bounds (0 to {n_rows - 1}). Received doc_index_1={doc_index_1}, doc_index_2={doc_index_2}."
                }
                session.store_result("cosine_similarity_tool", {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2}, summary)
                return summary

            vec1 = session.feature_matrix[doc_index_1].toarray().ravel()
            vec2 = session.feature_matrix[doc_index_2].toarray().ravel()
            mode = "index"

        else:
            vectorizer = session.artifacts.get("count_vectorizer") if session.artifacts else None
            if vectorizer is None:
                summary = {
                    "result_id": result_id,
                    "error": "CountVectorizer not found in session.artifacts['count_vectorizer']. Call build_dtm_tool first."
                }
                session.store_result("cosine_similarity_tool", {"text_1": text_1, "text_2": text_2}, summary)
                return summary

            matrix = vectorizer.transform([text_1, text_2])
            vec1 = matrix[0].toarray().ravel()
            vec2 = matrix[1].toarray().ravel()
            mode = "text"

        # Compute dot product and vector norms
        dot_product = float(np.dot(vec1, vec2))
        norm1 = float(np.linalg.norm(vec1))
        norm2 = float(np.linalg.norm(vec2))

        # Handle zero vectors (e.g. empty text or terms outside vocabulary)
        zero_vec_1 = norm1 == 0.0
        zero_vec_2 = norm2 == 0.0

        if zero_vec_1 or zero_vec_2:
            similarity = 0.0
            is_zero_vector = True

            zero_inputs = []
            if zero_vec_1:
                zero_inputs.append(f"doc 1 (index {doc_index_1})" if mode == "index" else "text_1")
            if zero_vec_2:
                zero_inputs.append(f"doc 2 (index {doc_index_2})" if mode == "index" else "text_2")

            note = f"Zero vector detected for {', '.join(zero_inputs)}. Returned cosine similarity 0.0."
        else:
            similarity = dot_product / (norm1 * norm2)
            is_zero_vector = False
            note = None

        summary = {
            "result_id": result_id,
            "cosine_similarity": round(float(similarity), 4),
            "mode": mode,
            "zero_vector": is_zero_vector,
        }

        if note is not None:
            summary["note"] = note

        if mode == "index":
            summary["doc_index_1"] = doc_index_1
            summary["doc_index_2"] = doc_index_2
            if session.labels is not None and len(session.labels) > max(doc_index_1, doc_index_2):
                summary["category_1"] = str(session.labels[doc_index_1])
                summary["category_2"] = str(session.labels[doc_index_2])

        session.store_result(
            "cosine_similarity_tool",
            {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2, "text_1": text_1, "text_2": text_2},
            summary
        )
        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> Dict[str, Any]:
        """
        Computes the Pearson correlation matrix across top terms by variance in the global DTM.

        Selects up to top_n non-zero variance terms from session.feature_matrix across all
        documents, computes pairwise Pearson correlation, builds a heatmap visualization on
        session.pending_figure, and stores the full report.

        Args:
            top_n: Maximum number of top terms by variance to include (default 20).

        Returns:
            Dict containing result_id, terms, variances, correlation_matrix,
            and zero_variance_excluded count.
        """
        result_id = session.next_result_id("feature_corr")

        if top_n < 1:
            summary = {
                "result_id": result_id,
                "error": "top_n must be at least 1."
            }
            session.store_result("feature_correlation_matrix_tool", {"top_n": top_n}, summary)
            return summary

        X = session.feature_matrix
        feature_names = session.feature_names

        if X is None or feature_names is None:
            summary = {
                "result_id": result_id,
                "error": "Document-term matrix has not been built yet. Call build_dtm_tool first."
            }
            session.store_result("feature_correlation_matrix_tool", {"top_n": top_n}, summary)
            return summary

        # Sparse variance calculation across all documents (ddof=0)
        means = np.asarray(X.mean(axis=0)).ravel()
        means_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
        variances = means_sq - (means ** 2)

        # Clip negative small floats at 0.0 due to numerical precision
        variances = np.maximum(variances, 0.0)

        # Filter out zero variance terms (< 1e-12)
        valid_mask = variances >= 1e-12
        zero_variance_excluded = int(np.sum(~valid_mask))

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            summary = {
                "result_id": result_id,
                "error": "No terms with non-zero variance available to compute correlation.",
                "zero_variance_excluded": zero_variance_excluded
            }
            session.store_result("feature_correlation_matrix_tool", {"top_n": top_n}, summary)
            return summary

        # Sort valid terms by variance descending, break ties deterministically by feature name
        valid_variances = variances[valid_indices]
        valid_feature_names = [feature_names[i] for i in valid_indices]

        # Sort key: (-variance, feature_name)
        sorted_pairs = sorted(
            zip(valid_indices, valid_feature_names, valid_variances),
            key=lambda item: (-item[2], item[1])
        )

        # Slice top_n
        top_pairs = sorted_pairs[:top_n]
        selected_indices = [p[0] for p in top_pairs]
        selected_terms = [p[1] for p in top_pairs]
        selected_variances = [round(float(p[2]), 4) for p in top_pairs]

        # Extract selected columns and compute Pearson correlation matrix
        sub_X = X[:, selected_indices].toarray()
        corr_matrix = np.corrcoef(sub_X, rowvar=False)
        corr_matrix = np.atleast_2d(corr_matrix)

        # Format correlation matrix to nested list rounded to 4 decimals
        corr_matrix_list = [
            [round(float(val), 4) for val in row]
            for row in corr_matrix
        ]

        # Build heatmap figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            xticklabels=selected_terms,
            yticklabels=selected_terms,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            ax=ax
        )
        ax.set_title("Top Term Feature Correlation Matrix")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        # Full report DataFrame (sorted by variance descending) for visualize_result_tool
        full_report_df = pd.DataFrame({
            "term": selected_terms,
            "variance": selected_variances
        })

        summary = {
            "result_id": result_id,
            "terms": selected_terms,
            "variances": selected_variances,
            "correlation_matrix": corr_matrix_list,
            "zero_variance_excluded": zero_variance_excluded
        }

        session.store_result(
            "feature_correlation_matrix_tool",
            {"top_n": top_n},
            summary,
            full_report=full_report_df
        )
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
