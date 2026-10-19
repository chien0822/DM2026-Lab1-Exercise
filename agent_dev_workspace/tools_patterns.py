import os
import tempfile
from typing import Any, Dict, List, Literal, Optional
from langchain_core.tools import tool
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def make_tools(session: Any) -> List[Any]:
    """Factory function that creates tools bound to the provided session instance."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Mines frequent itemsets/patterns from documents belonging to a specific category.

        Builds a fresh category-local document-term matrix with English stop words removed,
        applies vocabulary filtering (variance, tfidf, or term_frequency), and mines patterns
        using PAMI algorithms (fpgrowth, topk, or maxfpgrowth).

        Args:
            category_name: The target category name to mine (must exist in session.dataframe['category_name']).
            filtering_method: Method used to filter terms ("variance", "tfidf", or "term_frequency").
            algorithm: PAMI algorithm ("fpgrowth", "topk", or "maxfpgrowth"). Default "fpgrowth".
            min_sup: Minimum support threshold (int) for "fpgrowth" and "maxfpgrowth".
            k: Top k patterns count (int) for "topk".

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm, n_patterns,
            and top patterns preview.
        """
        if session.dataframe is None:
            return {"error": "No dataframe found in session. Please load a dataset first."}

        if "category_name" not in session.dataframe.columns:
            return {"error": "category_name column missing from session.dataframe."}

        available_categories = session.dataframe["category_name"].dropna().unique().tolist()
        if category_name not in available_categories:
            return {"error": f"Category '{category_name}' not found in session.dataframe['category_name']. Available: {available_categories}"}

        # Validate algorithm parameters
        if algorithm in ("fpgrowth", "maxfpgrowth") and min_sup is None:
            return {"error": f"Parameter 'min_sup' (int) is required for algorithm '{algorithm}'."}

        if algorithm == "topk" and k is None:
            return {"error": "Parameter 'k' (int) is required for algorithm 'topk'."}

        args_dict = {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k,
        }

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"

        def _early_exit(note_msg: str) -> Dict[str, Any]:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": note_msg,
            }
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", args_dict, summary)
            return summary

        # 1. Filter dataframe to category without dropping empty rows
        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        text_series = cat_df["text"].fillna("")

        # 2. Vectorize with English stop words removed
        vectorizer = CountVectorizer(stop_words="english")
        try:
            counts_matrix = vectorizer.fit_transform(text_series)
        except ValueError:
            # Handles empty vocabulary after stopword removal
            return _early_exit("vocabulary empty after stopword removal")

        feature_names = list(vectorizer.get_feature_names_out())
        if not feature_names:
            return _early_exit("vocabulary empty after stopword removal")

        df_counts = pd.DataFrame(counts_matrix.toarray(), columns=feature_names)

        # 3. Apply vocabulary filtering
        if filtering_method == "variance":
            metric = df_counts.var(axis=0)
            lower = np.percentile(metric, 5)
            upper = np.percentile(metric, 95)
            kept_terms = metric[(metric >= lower) & (metric <= upper)].index.tolist()
        elif filtering_method == "term_frequency":
            metric = df_counts.sum(axis=0)
            lower = np.percentile(metric, 5)
            upper = np.percentile(metric, 95)
            kept_terms = metric[(metric >= lower) & (metric <= upper)].index.tolist()
        elif filtering_method == "tfidf":
            tfidf_transformer = TfidfTransformer()
            tfidf_matrix = tfidf_transformer.fit_transform(df_counts.values)
            tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            metric = pd.Series(tfidf_scores, index=df_counts.columns)
            lower = np.percentile(metric, 20)
            kept_terms = metric[metric >= lower].index.tolist()
        else:
            return {"error": f"Unknown filtering_method: {filtering_method}"}

        if not kept_terms:
            return _early_exit("no terms survived filtering")

        # 4. Filter dataframe to kept terms and rows with non-zero counts
        df_filtered = df_counts[kept_terms]
        row_sums = df_filtered.sum(axis=1)
        df_txns = df_filtered[row_sums > 0]

        if df_txns.empty:
            return _early_exit("no non-empty transactions survived")

        # 5. Convert to transactional database tempfile via PAMI DF2DB
        temp_db_file = None
        raw_patterns_dict = {}

        try:
            import PAMI.extras.convert.DF2DB as db_converter

            temp_fd, temp_db_file = tempfile.mkstemp(suffix=".txt")
            os.close(temp_fd)

            # Convert count dataframe to transactional database file matching PAMI 0.9.2 signature
            conv = db_converter.DF2DB(df_txns, DFtype="dense")
            conv.convert2TransactionalDatabase(temp_db_file, ">=", 1)

            # 6. Mine patterns using requested PAMI algorithm
            if algorithm == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth
                miner = FPGrowth.FPGrowth(temp_db_file, minSup=min_sup)
            elif algorithm == "topk":
                from PAMI.frequentPattern.topk import FAE
                miner = FAE.FAE(temp_db_file, k=k)
            elif algorithm == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                miner = MaxFPGrowth.MaxFPGrowth(temp_db_file, minSup=min_sup)
            else:
                return {"error": f"Unknown algorithm: {algorithm}"}

            miner.startMine()
            raw_patterns_dict = miner.getPatterns()
        except ImportError as e:
            return {"error": f"PAMI library or required submodule not found: {str(e)}"}
        finally:
            if temp_db_file and os.path.exists(temp_db_file):
                try:
                    os.remove(temp_db_file)
                except OSError:
                    pass

        # 7. Normalize, parse, and sort mined patterns
        parsed_patterns = []
        if raw_patterns_dict:
            for pat_key, sup_val in raw_patterns_dict.items():
                if isinstance(pat_key, str):
                    # PAMI returns tab-delimited or space-delimited string of items
                    items = [it.strip() for it in pat_key.replace("\t", " ").split(" ") if it.strip()]
                elif isinstance(pat_key, (tuple, list)):
                    items = [str(it).strip() for it in pat_key]
                else:
                    items = [str(pat_key)]

                parsed_patterns.append({
                    "items": sorted(items),
                    "support": int(sup_val),
                })

        # Sort patterns by support descending, then items lexicographically for deterministic order
        parsed_patterns.sort(key=lambda p: (-p["support"], p["items"]))

        result_id = session.next_result_id("mine_patterns")
        session.artifacts[artifact_key] = parsed_patterns

        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns": parsed_patterns[:20],  # Preview top 20
            "n_patterns": len(parsed_patterns),
        }

        session.store_result("mine_patterns_tool", args_dict, summary)
        return summary

    return [mine_patterns_tool]
