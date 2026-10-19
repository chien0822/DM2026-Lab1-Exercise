from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        max_features: Optional[int] = None,
        min_df: int = 1,
        max_df: float = 1.0,
    ) -> dict:
        """Builds a Document-Term Matrix (DTM) from dataset text.

        Requires load_dataset_tool to have run first to populate session.dataframe.
        Vectorizes text in session.dataframe["text"] using CountVectorizer,
        populates session.feature_matrix and session.feature_names, sets
        labels, and stores the fitted vectorizer in session.artifacts["count_vectorizer"].

        Args:
            ngram_range: Lower and upper boundary of range of n-values for n-grams, e.g., [1, 1] for unigrams.
            max_features: Maximum number of vocabulary terms to retain by frequency. None for all.
            min_df: Minimum document count threshold (integer >= 1). Terms appearing in fewer documents are ignored.
            max_df: Maximum document frequency proportion (float in (0.0, 1.0]). Terms appearing in a larger proportion of docs are ignored.

        Returns:
            dict: Summary containing matrix shape, non-zero element counts, sparsity percentage, and vocabulary sample.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {
                "error": "No dataset loaded in session. Please load a dataset first."
            }

        text_series = session.dataframe["text"].fillna("")

        # Convert ngram_range list to tuple for CountVectorizer
        ngram_tuple = tuple(ngram_range)

        try:
            vectorizer = CountVectorizer(
                ngram_range=ngram_tuple,
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
            )
            feature_matrix = vectorizer.fit_transform(text_series)
        except ValueError as e:
            return {"error": f"Failed to build DTM: {str(e)}"}

        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names

        if "category_name" in session.dataframe.columns:
            session.set_labels(
                session.dataframe["category_name"].to_numpy()
            )

        # Store vectorizer artifact for downstream tools like cosine_similarity_tool
        session.artifacts["count_vectorizer"] = vectorizer

        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = (
            100.0 * (1.0 - non_zero / total_elements)
            if total_elements > 0
            else 0.0
        )

        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "vocabulary_sample": feature_names[:10],
        }

        session.store_result(
            tool_name="build_dtm_tool",
            args={
                "ngram_range": ngram_range,
                "max_features": max_features,
                "min_df": min_df,
                "max_df": max_df,
            },
            summary=summary,
        )

        return summary

    @tool
    def term_frequency_tool(top_n: int = 20) -> dict:
        """Aggregates term frequencies across all documents in the Document-Term Matrix (DTM).

        Requires build_dtm_tool to have run first to populate session.feature_matrix and session.feature_names.

        Args:
            top_n: Number of top frequent terms to return in the summary (default 20, must be >= 1).

        Returns:
            dict: Summary containing total term count, vocabulary size, top_n term frequencies, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No DTM found in session. Please run build_dtm_tool first."
            }

        if top_n < 1:
            return {"error": "top_n must be an integer >= 1."}

        # Sum term counts across all documents (columns of DTM)
        counts = session.feature_matrix.sum(axis=0).A1

        # Pair terms with counts and sort descending by frequency, breaking ties alphabetically
        term_counts = [
            (term, int(count))
            for term, count in zip(session.feature_names, counts)
        ]
        sorted_terms = sorted(term_counts, key=lambda x: (-x[1], x[0]))

        # Prepare full_report DataFrame for visualize_result_tool
        full_report = pd.DataFrame(sorted_terms, columns=["term", "frequency"])

        # Slice top_n terms for JSON summary
        top_terms = sorted_terms[:top_n]
        top_frequencies = {term: freq for term, freq in top_terms}

        result_id = session.next_result_id("term_freq")
        summary = {
            "result_id": result_id,
            "total_term_count": int(counts.sum()),
            "total_terms_vocabulary": len(session.feature_names),
            "top_n": top_n,
            "top_frequencies": top_frequencies,
        }

        session.store_result(
            tool_name="term_frequency_tool",
            args={"top_n": top_n},
            summary=summary,
            full_report=full_report,
        )

        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> dict:
        """Generates a heatmap of a raw slice of the document-term matrix.

        Requires build_dtm_tool to have run first to populate session.feature_matrix and session.feature_names.

        Args:
            n_terms: Number of terms (columns) to include in existing vocabulary order (default 20).
            n_documents: Number of documents (rows) to include in existing row order (default 20).

        Returns:
            dict: Summary containing sliced matrix counts, term names, document indices, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No DTM found in session. Please run build_dtm_tool first."
            }

        if n_terms < 1 or n_documents < 1:
            return {
                "error": "n_terms and n_documents must both be integers >= 1."
            }

        # Positional slice of the DTM
        sub_matrix = session.feature_matrix[
            :n_documents, :n_terms
        ].toarray()
        sliced_terms = session.feature_names[:n_terms]

        # Extract document index values and category names from dataframe
        if session.dataframe is not None and not session.dataframe.empty:
            sub_df = session.dataframe.iloc[: sub_matrix.shape[0]]
            doc_indices = [int(idx) for idx in sub_df.index]
            if "category_name" in sub_df.columns:
                row_labels = [
                    f"Doc {idx} ({cat})"
                    for idx, cat in zip(doc_indices, sub_df["category_name"])
                ]
            else:
                row_labels = [f"Doc {idx}" for idx in doc_indices]
        else:
            doc_indices = list(range(sub_matrix.shape[0]))
            row_labels = [f"Doc {idx}" for idx in doc_indices]

        # Convert sliced matrix to nested Python list of ints
        sliced_matrix_list = [
            [int(val) for val in row] for row in sub_matrix
        ]

        # Build Seaborn heatmap figure
        fig, ax = plt.subplots(
            figsize=(
                max(6, len(sliced_terms) * 0.8),
                max(4, len(row_labels) * 0.5),
            )
        )
        sns.heatmap(
            sub_matrix,
            annot=True,
            fmt="d",
            xticklabels=sliced_terms,
            yticklabels=row_labels,
            ax=ax,
            cmap="YlGnBu",
        )
        ax.set_title(
            f"DTM Heatmap ({sub_matrix.shape[0]} Docs x {len(sliced_terms)} Terms)"
        )
        fig.tight_layout()

        # Close figure to prevent double-rendering / memory leaks; Figure object remains valid
        plt.close(fig)
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents": sub_matrix.shape[0],
            "n_terms": len(sliced_terms),
            "document_indices": doc_indices,
            "term_names": sliced_terms,
            "matrix": sliced_matrix_list,
        }

        session.store_result(
            tool_name="dtm_heatmap_tool",
            args={"n_terms": n_terms, "n_documents": n_documents},
            summary=summary,
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
