from typing import Any, Dict, List, Literal, Optional
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import seaborn as sns
from langchain_core.tools import tool
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer


def make_tools(session: Any) -> List[Any]:
    """Factory function that creates tools bound to the provided session instance."""

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = None,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
    ) -> Dict[str, Any]:
        """Reduces the dimensionality of session.feature_matrix using PCA, t-SNE, or UMAP.

        Must produce a 2D scatter plot colored by document category and store it
        on session.pending_figure. When session.feature_matrix is sparse, t-SNE
        uses init="random" because default PCA initialization is not supported for
        sparse inputs in scikit-learn.

        Args:
            method: Reduction algorithm to use ("pca", "tsne", or "umap"). Default "pca".
            n_components: Target dimensions for reduction. Must be >= 2 for plotting. Default 2.
            random_state: Optional seed for reproducible results.
            perplexity: Perplexity parameter for t-SNE (automatically reduced if too large for sample size). Default 30.0.
            n_neighbors: Number of neighbors parameter for UMAP (automatically reduced if too large for sample size). Default 15.

        Returns:
            Dict containing result_id, method, n_components, n_documents, coordinates preview,
            and explained_variance_ratio (if method is 'pca').
        """
        if session.feature_matrix is None:
            return {"error": "No feature_matrix found in session. Please run build_dtm_tool first."}

        if n_components < 2:
            return {"error": "n_components must be at least 2 to build the required scatter plot."}

        X = session.feature_matrix
        n_samples = X.shape[0]

        if n_samples < 2:
            return {"error": "Need at least 2 documents to perform dimensionality reduction."}

        adjusted = False
        effective_perplexity = perplexity
        effective_n_neighbors = n_neighbors

        # Perform reduction based on selected method
        if method == "pca":
            model = PCA(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(X)
            explained_variance = [float(round(v, 4)) for v in model.explained_variance_ratio_]
        elif method == "tsne":
            if perplexity >= n_samples:
                effective_perplexity = max(1.0, float(n_samples - 1))
                adjusted = True
            
            init_param = "random" if sp.issparse(X) else "pca"
            model = TSNE(
                n_components=n_components,
                perplexity=effective_perplexity,
                random_state=random_state,
                init=init_param,
            )
            coords = model.fit_transform(X)
            explained_variance = None
        elif method == "umap":
            try:
                import umap
            except ImportError:
                return {"error": "umap-learn package is not installed."}

            if n_neighbors >= n_samples:
                effective_n_neighbors = max(2, n_samples - 1)
                adjusted = True
            model = umap.UMAP(
                n_components=n_components,
                n_neighbors=effective_n_neighbors,
                random_state=random_state,
            )
            coords = model.fit_transform(X)
            explained_variance = None
        else:
            return {"error": f"Unknown reduction method: {method}"}

        # Store full output in artifacts
        coords_array = np.asarray(coords)
        session.artifacts[f"reduced_{method}"] = coords_array
        session.artifacts["reduced_coordinates"] = coords_array

        # Color scatter plot by session.labels (fallback to categories/dataframe if missing)
        if session.labels is not None:
            hue_labels = session.labels
        elif session.dataframe is not None and "category_name" in session.dataframe.columns:
            hue_labels = session.dataframe["category_name"].values
        elif session.dataframe is not None and "category" in session.dataframe.columns:
            hue_labels = session.dataframe["category"].values
        else:
            hue_labels = ["default"] * n_samples

        # Plot 2D scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=coords_array[:, 0],
            y=coords_array[:, 1],
            hue=hue_labels,
            palette="tab10",
            ax=ax,
            s=60,
        )
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1" if method == "pca" else "Dim 1")
        ax.set_ylabel("Component 2" if method == "pca" else "Dim 2")
        ax.grid(True, linestyle="--", alpha=0.5)

        session.pending_figure = fig
        plt.close(fig)

        # Store result summary
        result_id = session.next_result_id("reduce_dimensions")
        preview_coords = [[float(round(c, 4)) for c in row[:2]] for row in coords_array[:5]]

        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "n_documents": n_samples,
            "coordinates_preview": preview_coords,
            "adjusted": adjusted,
        }

        if method == "pca":
            summary["explained_variance_ratio"] = explained_variance

        if method == "tsne" and adjusted:
            summary["effective_perplexity"] = effective_perplexity

        if method == "umap" and adjusted:
            summary["effective_n_neighbors"] = effective_n_neighbors

        args = {
            "method": method,
            "n_components": n_components,
            "random_state": random_state,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors,
        }

        session.store_result("reduce_dimensions_tool", args, summary)
        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """Binarizes document category labels using sklearn LabelBinarizer.

        Reads session.dataframe['category_name'], one-hot encodes or binary encodes
        the string labels, stores the result in session.dataframe['bin_category'],
        and saves artifacts for downstream ML tools.

        Returns:
            Dict containing result_id, classes, shape, counts_per_class, and preview_rows.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Please load a dataset first."}

        if "category_name" not in session.dataframe.columns:
            return {"error": "category_name column missing from session.dataframe."}

        labels = session.dataframe["category_name"]
        lb = LabelBinarizer()
        binarized = lb.fit_transform(labels)

        # Store in session.dataframe as list of lists
        bin_list = binarized.tolist()
        session.dataframe["bin_category"] = bin_list

        # Store artifacts for downstream tools
        session.artifacts["label_binarizer"] = lb
        session.artifacts["binarized_labels"] = binarized

        # Clean Python types for summary
        classes_list = [str(c) for c in lb.classes_]
        n_classes = len(classes_list)
        shape_list = [int(s) for s in binarized.shape]
        
        # Count per class from original category_name series
        val_counts = labels.value_counts()
        counts_per_class = {str(k): int(v) for k, v in val_counts.items()}

        preview_rows = [[int(val) for val in row] for row in binarized[:5]]

        result_id = session.next_result_id("binarize_labels")

        summary = {
            "result_id": result_id,
            "classes": classes_list,
            "n_classes": n_classes,
            "shape": shape_list,
            "counts_per_class": counts_per_class,
            "preview_rows": preview_rows,
        }

        # Add binary encoding details if exactly 2 classes
        if n_classes == 2:
            summary["positive_class"] = classes_list[1]
            summary["binary_encoding_note"] = f"1 = {classes_list[1]}, 0 = {classes_list[0]}"

        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
