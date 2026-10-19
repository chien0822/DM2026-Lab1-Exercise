from typing import List, Dict, Any, Optional
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from langchain_core.tools import tool


def _clean_stats_dict(d: dict) -> dict:
    """Converts numpy numbers and replaces NaN/Inf values with None for JSON compatibility."""
    cleaned = {}
    for k, v in d.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            cleaned[str(k)] = None
        elif isinstance(v, (np.integer, np.floating)):
            val = float(v) if isinstance(v, np.floating) else int(v)
            cleaned[str(k)] = None if math.isnan(val) or math.isinf(val) else val
        else:
            cleaned[str(k)] = v
    return cleaned


def _resolve_and_validate_path(subdirectory: str) -> Optional[str]:
    """
    Resolves subdirectory relative to repo root (os.getcwd()) and verifies it stays strictly within the repo root.
    Returns real absolute path if valid, or None if outside repo root.
    """
    repo_root = os.path.realpath(os.getcwd())
    target_path = os.path.realpath(os.path.join(repo_root, subdirectory))
    
    if target_path == repo_root or target_path.startswith(repo_root + os.sep):
        return target_path
    return None


def make_tools(session):
    @tool
    def check_missing_tool(drop: bool = False) -> Dict[str, Any]:
        """
        Checks for missing or empty/whitespace-only text in the loaded dataset, and optionally drops them.

        Note: When drop=True and missing rows are actually removed, any existing DTM 
        (session.feature_matrix, feature_names, labels) is invalidated and cleared. 
        You must re-run build_dtm_tool afterwards.

        Args:
            drop: If True, drops rows with missing/empty text from session.dataframe while preserving original index labels. Default is False.

        Returns:
            Dict summary containing missing_count, total_rows, missing_indices (up to 10), action_taken, and optional note.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session. Please run load_dataset_tool first."}

        if "text" not in session.dataframe.columns:
            return {"status": "error", "message": "Loaded dataset does not contain a 'text' column."}

        df = session.dataframe

        # Identify rows where text is NaN or empty/whitespace-only
        missing_mask = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
        missing_rows = df[missing_mask]
        missing_count = int(len(missing_rows))
        total_rows = int(len(df))

        # Preserve original index labels (converted to python ints)
        missing_indices = [int(idx) for idx in missing_rows.index[:10].tolist()]

        result_id = session.next_result_id("missing")

        if drop and missing_count > 0:
            session.dataframe.drop(index=missing_rows.index, inplace=True)
            
            # Rows were actually dropped -> invalidate existing feature matrix and labels
            session.feature_matrix = None
            session.feature_names = None
            session.labels = None

            rows_after = int(len(session.dataframe))
            summary = {
                "result_id": result_id,
                "missing_count": missing_count,
                "total_rows": total_rows,
                "rows_after": rows_after,
                "missing_indices": missing_indices,
                "action_taken": "dropped",
                "note": "Rows dropped. Existing DTM and labels invalidated; please re-run build_dtm_tool."
            }
        else:
            summary = {
                "result_id": result_id,
                "missing_count": missing_count,
                "total_rows": total_rows,
                "missing_indices": missing_indices,
                "action_taken": "dropped" if drop else "reported",
                "note": "No rows were dropped." if drop else "Missing rows reported."
            }

        session.store_result("check_missing_tool", {"drop": drop}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """
        Checks for duplicate rows in session.dataframe across all columns, and optionally drops them.

        Note: When drop=True, dropping is done with keep=False (dropping EVERY copy of a duplicated row),
        matching Master's X.drop_duplicates(keep=False, inplace=True). If rows are dropped, any existing
        DTM (session.feature_matrix, feature_names, labels) is invalidated and cleared. You must re-run build_dtm_tool afterwards.

        Args:
            drop: If True, drops all copies of duplicated rows from session.dataframe while preserving original index labels. Default is False.

        Returns:
            Dict summary containing duplicate_count, rows_removed, total_rows, rows_after, duplicate_indices (up to 10), action_taken, and optional note.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session. Please run load_dataset_tool first."}

        df = session.dataframe
        total_rows = int(len(df))

        # Create a temporary copy converting list-valued columns to tuples to avoid "unhashable type: list"
        df_temp = df.copy()
        for col in df_temp.columns:
            if df_temp[col].apply(lambda x: isinstance(x, list)).any():
                df_temp[col] = df_temp[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)

        # duplicate_count is extra copies count (keep="first")
        dup_mask_first = df_temp.duplicated(keep="first")
        duplicate_count = int(dup_mask_first.sum())

        # For reporting indices: indices of extra copies (keep="first")
        dup_indices_first = [int(idx) for idx in df_temp[dup_mask_first].index[:10].tolist()]

        result_id = session.next_result_id("duplicates")

        if drop and duplicate_count > 0:
            # When dropping, use keep=False to drop ALL copies of duplicated rows
            dup_mask_all = df_temp.duplicated(keep=False)
            rows_to_drop_indices = df_temp[dup_mask_all].index
            rows_removed = int(len(rows_to_drop_indices))

            session.dataframe.drop(index=rows_to_drop_indices, inplace=True)

            # Rows were actually dropped -> invalidate existing feature matrix and labels
            session.feature_matrix = None
            session.feature_names = None
            session.labels = None

            rows_after = int(len(session.dataframe))
            summary = {
                "result_id": result_id,
                "duplicate_count": duplicate_count,
                "rows_removed": rows_removed,
                "total_rows": total_rows,
                "rows_after": rows_after,
                "duplicate_indices": [int(idx) for idx in rows_to_drop_indices[:10].tolist()],
                "action_taken": "dropped",
                "note": "Rows dropped with keep=False (all copies removed). Existing DTM and labels invalidated; please re-run build_dtm_tool."
            }
        else:
            summary = {
                "result_id": result_id,
                "duplicate_count": duplicate_count,
                "rows_removed": 0,
                "total_rows": total_rows,
                "duplicate_indices": dup_indices_first,
                "action_taken": "dropped" if drop else "reported",
                "note": "No rows were dropped." if drop else "Duplicates reported."
            }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(n: int, random_state: int = 42) -> Dict[str, Any]:
        """
        Subsamples session.dataframe to n rows using pandas DataFrame.sample.

        Note: Replaces session.dataframe with the sampled subset while preserving original row index labels.
        The original full dataset is not kept; to get it back, re-run load_dataset_tool.
        Any existing DTM (session.feature_matrix, feature_names, labels) is invalidated and cleared.
        You must re-run build_dtm_tool afterwards.

        Args:
            n: Number of rows to sample. Must be between 1 and total rows.
            random_state: Seed for reproducible random sampling. Default is 42.

        Returns:
            Dict summary containing rows_before, rows_after, sampled_indices (up to 10), counts_per_category, and a note.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session. Please run load_dataset_tool first."}

        total_rows = int(len(session.dataframe))

        if n < 1 or n > total_rows:
            return {
                "status": "error",
                "message": f"Sample size n={n} is invalid. Must be between 1 and total_rows={total_rows}."
            }

        rows_before = total_rows
        sampled_df = session.dataframe.sample(n=n, random_state=random_state)
        session.dataframe = sampled_df

        # Invalidate existing DTM and labels
        session.feature_matrix = None
        session.feature_names = None
        session.labels = None

        sampled_indices = [int(idx) for idx in sampled_df.index[:10].tolist()]

        counts_per_category = {}
        if "category_name" in sampled_df.columns:
            counts_per_category = {str(k): int(v) for k, v in sampled_df["category_name"].value_counts().items()}

        result_id = session.next_result_id("sample")
        summary = {
            "result_id": result_id,
            "rows_before": rows_before,
            "rows_after": int(len(sampled_df)),
            "sampled_indices": sampled_indices,
            "counts_per_category": counts_per_category,
            "note": "Dataset subsampled. Original dataset not kept (re-run load_dataset_tool to restore). Existing DTM and labels invalidated; please re-run build_dtm_tool."
        }

        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """
        Computes summary statistics on document text length (character count, not word count) overall and per category, and generates a boxplot.

        Requires load_dataset_tool to be run first. This tool is read-only and does not modify session.dataframe,
        session.feature_matrix, or session.labels. It describes the current state of session.dataframe, so after
        sampling or dropping duplicate/missing rows, statistics reflect that subset.

        Sets the generated boxplot figure on session.pending_figure.

        Returns:
            Dict summary containing overall_stats and per_category_stats.
        """
        if session.dataframe is None or len(session.dataframe) == 0:
            return {"status": "error", "message": "No dataset loaded in session or dataset is empty. Please run load_dataset_tool first."}

        if "text" not in session.dataframe.columns:
            return {"status": "error", "message": "Loaded dataset does not contain a 'text' column."}

        df = session.dataframe

        # Compute text length locally (character count) without modifying session.dataframe
        text_length = df["text"].fillna("").astype(str).apply(len)

        # Overall summary stats
        overall_stats = _clean_stats_dict(text_length.describe().to_dict())

        # Category column selection
        cat_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)

        per_category_stats = {}
        sorted_categories = []
        if cat_col:
            grouped = text_length.groupby(df[cat_col])
            sorted_categories = sorted([str(cat) for cat in grouped.groups.keys()])
            for cat in sorted_categories:
                # Find matching group key
                matching_key = next((k for k in grouped.groups.keys() if str(k) == cat), cat)
                group_series = grouped.get_group(matching_key)
                per_category_stats[cat] = _clean_stats_dict(group_series.describe().to_dict())

        # Generate boxplot figure
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df = pd.DataFrame({"text_length": text_length})
        if cat_col:
            plot_df[cat_col] = df[cat_col].astype(str)
            sns.boxplot(data=plot_df, x=cat_col, y="text_length", order=sorted_categories, ax=ax)
            ax.set_title(f"Document Text Length by {cat_col}")
            ax.set_xlabel(cat_col)
        else:
            sns.boxplot(data=plot_df, y="text_length", ax=ax)
            ax.set_title("Document Text Length Distribution")

        ax.set_ylabel("Text Length (characters)")
        fig.tight_layout()
        plt.close(fig)

        # Set figure on session.pending_figure
        session.pending_figure = fig

        result_id = session.next_result_id("describe")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    @tool
    def tokenize_tool(lowercase: bool = True) -> Dict[str, Any]:
        """
        Tokenizes each document's text into unigrams using nltk.word_tokenize.

        Note: word_tokenize separates punctuation into independent tokens and splits contractions (e.g., "don't" -> ["do", "n't"]).
        Stores unigram lists in session.dataframe["unigrams"] per row.

        Args:
            lowercase: If True, lowercases text before tokenizing. Default is True.

        Returns:
            Dict summary containing n_documents, total_tokens, mean_tokens_per_document, n_empty_documents, preview (first 3 docs, up to 10 tokens each), and a note.
        """
        if session.dataframe is None or len(session.dataframe) == 0:
            return {"status": "error", "message": "No dataset loaded in session or dataset is empty. Please run load_dataset_tool first."}

        if "text" not in session.dataframe.columns:
            return {"status": "error", "message": "Loaded dataset does not contain a 'text' column."}

        df = session.dataframe

        def _tokenize_row(val: Any) -> List[str]:
            if pd.isna(val) or val is None:
                return []
            text_str = str(val).strip()
            if not text_str:
                return []
            if lowercase:
                text_str = text_str.lower()
            return word_tokenize(text_str)

        try:
            unigrams_series = df["text"].apply(_tokenize_row)
        except LookupError:
            return {
                "status": "error",
                "message": "NLTK tokenizer resource missing. Please run nltk.download('punkt_tab') or nltk.download('punkt')."
            }

        session.dataframe["unigrams"] = unigrams_series

        n_documents = int(len(df))
        total_tokens = int(sum(len(u) for u in unigrams_series))
        mean_tokens_per_document = float(total_tokens / n_documents) if n_documents > 0 else 0.0
        n_empty_documents = int(sum(1 for u in unigrams_series if len(u) == 0))

        # Preview: first 3 documents, up to 10 tokens each
        preview = [u[:10] for u in unigrams_series.iloc[:3].tolist()]

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "total_tokens": total_tokens,
            "mean_tokens_per_document": mean_tokens_per_document,
            "n_empty_documents": n_empty_documents,
            "preview": preview,
            "note": "Text tokenized into unigrams and stored in session.dataframe['unigrams']."
        }

        session.store_result("tokenize_tool", {"lowercase": lowercase}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> Dict[str, Any]:
        """
        Lists files and directories in a subdirectory relative to the repo root.

        This tool is read-only and does not touch session.dataframe.

        Args:
            subdirectory: Path relative to the repo root. Default "" refers to the repo root.

        Returns:
            Dict summary containing directories and files lists (sorted, excluding hidden items and __pycache__), capped at 200 items each.
        """
        target_path = _resolve_and_validate_path(subdirectory)
        if target_path is None:
            return {
                "status": "error",
                "message": f"Access denied: Path '{subdirectory}' resolves outside the repository root."
            }

        if not os.path.exists(target_path):
            return {
                "status": "error",
                "message": f"Path '{subdirectory}' does not exist."
            }

        if not os.path.isdir(target_path):
            return {
                "status": "error",
                "message": f"Path '{subdirectory}' is a file, not a directory."
            }

        dirs = []
        files = []

        try:
            entries = os.listdir(target_path)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to list directory '{subdirectory}': {str(e)}"
            }

        for name in entries:
            if name.startswith(".") or name == "__pycache__":
                continue
            full_item_path = os.path.join(target_path, name)
            if os.path.isdir(full_item_path):
                dirs.append(name)
            elif os.path.isfile(full_item_path):
                files.append(name)

        dirs.sort()
        files.sort()

        total_dirs = len(dirs)
        total_files = len(files)

        is_truncated = (total_dirs > 200) or (total_files > 200)
        dirs_capped = dirs[:200]
        files_capped = files[:200]

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "directories": dirs_capped,
            "files": files_capped,
            "subdirectory": subdirectory
        }

        if is_truncated:
            summary["truncated"] = True
            summary["total_directories"] = total_dirs
            summary["total_files"] = total_files

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(n: int = 5, max_chars: int = 200) -> Dict[str, Any]:
        """
        Inspects the top n rows of session.dataframe and provides column metadata and row previews.

        This tool is read-only and does not modify session.dataframe.

        Args:
            n: Number of top rows to inspect. Default is 5. Must be at least 1.
            max_chars: Maximum character length for string previews before truncating with '...'. Default is 200. Must be at least 1.

        Returns:
            Dict summary containing n_requested, n_returned, n_rows, clamped, n_columns, columns, dtypes, and preview records.
        """
        if session.dataframe is None or len(session.dataframe) == 0:
            return {"status": "error", "message": "No dataset loaded in session or dataset is empty. Please run load_dataset_tool first."}

        if n < 1:
            return {"status": "error", "message": f"Parameter n must be at least 1, got n={n}."}

        if max_chars < 1:
            return {"status": "error", "message": f"Parameter max_chars must be at least 1, got max_chars={max_chars}."}

        df = session.dataframe
        n_rows = int(len(df))
        n_columns = int(len(df.columns))
        columns = list(df.columns)
        dtypes = {str(col): str(df[col].dtype) for col in df.columns}

        n_returned = min(n, n_rows)
        clamped = (n > n_rows)

        head_df = df.head(n_returned)

        def _format_cell(val: Any) -> Any:
            if isinstance(val, (list, tuple, np.ndarray)):
                items_list = list(val)
                first_items = items_list[:5]
                formatted_items = [_format_cell(item) for item in first_items]
                return {"length": len(items_list), "first": formatted_items}
            if pd.isna(val) or val is None:
                return None
            if isinstance(val, (bool, np.bool_)):
                return bool(val)
            if isinstance(val, (np.integer, np.floating)):
                fval = float(val) if isinstance(val, np.floating) else int(val)
                if math.isnan(fval) or math.isinf(fval):
                    return None
                return fval
            if isinstance(val, str):
                if len(val) > max_chars:
                    return val[:max_chars] + "..."
                return val
            return val

        preview = []
        for idx, row in head_df.iterrows():
            record = {"index": int(idx)}
            for col in columns:
                record[col] = _format_cell(row[col])
            preview.append(record)

        result_id = session.next_result_id("inspect")
        summary = {
            "result_id": result_id,
            "n_requested": n,
            "n_returned": n_returned,
            "n_rows": n_rows,
            "clamped": clamped,
            "n_columns": n_columns,
            "columns": columns,
            "dtypes": dtypes,
            "preview": preview
        }

        session.store_result("inspect_data_tool", {"n": n, "max_chars": max_chars}, summary)
        return summary

    return [
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
        tokenize_tool,
        list_files_tool,
        inspect_data_tool
    ]
