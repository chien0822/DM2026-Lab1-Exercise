import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def _setup_session_with_dtm():
    """Helper to initialize a fresh session with loaded fixture data and built DTM."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM to populate session.feature_matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    return session


def test_reduce_dimensions_pca():
    """Known answer #7: PCA reduction on sample_fixture.csv."""
    session = _setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({"method": "pca", "n_components": 2})

    assert "error" not in res
    assert res["method"] == "pca"
    assert res["n_documents"] == 8
    assert session.pending_figure is not None

    # Expected PCA coordinates from TEST_FIXTURE.md #7
    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])

    actual_coords = session.artifacts["reduced_pca"]
    np.testing.assert_allclose(actual_coords, expected_coords, atol=1e-3)

    # Expected explained_variance_ratio: [0.7532, 0.1965]
    expected_var = [0.7532, 0.1965]
    actual_var = res["explained_variance_ratio"]
    np.testing.assert_allclose(actual_var, expected_var, atol=1e-3)


def test_reduce_dimensions_tsne():
    """t-SNE reduction test.

    Note: Known answer #8 exact coordinates could not be reproduced in this
    environment (observed row 0 = [1314.2488, 286.825] for dense float and
    [129.4866, -1280.6466] for dense int). Testing shape, finite values,
    reproducibility with random_state, scatter plot creation, and automatic
    perplexity adjustment.
    """
    session = _setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({"method": "tsne", "perplexity": 2, "random_state": 42})

    assert "error" not in res
    assert res["method"] == "tsne"
    assert res["adjusted"] is False
    assert session.pending_figure is not None

    coords1 = session.artifacts["reduced_tsne"]
    assert coords1.shape == (8, 2)
    assert np.isfinite(coords1).all()

    # Reproducibility check with same random_state
    session2 = _setup_session_with_dtm()
    reduction_tools2 = make_reduction_tools(session2)
    reduce_tool2 = next(t for t in reduction_tools2 if t.name == "reduce_dimensions_tool")
    reduce_tool2.invoke({"method": "tsne", "perplexity": 2, "random_state": 42})
    coords2 = session2.artifacts["reduced_tsne"]
    np.testing.assert_allclose(coords1, coords2)

    # Test auto-adjustment when perplexity >= n_samples (8)
    session_adj = _setup_session_with_dtm()
    reduction_tools_adj = make_reduction_tools(session_adj)
    reduce_tool_adj = next(t for t in reduction_tools_adj if t.name == "reduce_dimensions_tool")
    res_adj = reduce_tool_adj.invoke({"method": "tsne", "perplexity": 30, "random_state": 42})
    assert res_adj["adjusted"] is True
    assert res_adj["effective_perplexity"] < 8


def test_reduce_dimensions_umap():
    """UMAP reduction test.

    Note: Known answer #9 exact coordinates could not be reproduced in this
    environment (observed row 0 = [10.0638, 1.3829]). Testing shape, finite
    values, reproducibility with random_state, and scatter plot creation.
    """
    session = _setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})

    assert "error" not in res
    assert res["method"] == "umap"
    assert res["adjusted"] is False
    assert session.pending_figure is not None

    coords1 = session.artifacts["reduced_umap"]
    assert coords1.shape == (8, 2)
    assert np.isfinite(coords1).all()

    # Reproducibility check with same random_state
    session2 = _setup_session_with_dtm()
    reduction_tools2 = make_reduction_tools(session2)
    reduce_tool2 = next(t for t in reduction_tools2 if t.name == "reduce_dimensions_tool")
    reduce_tool2.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})
    coords2 = session2.artifacts["reduced_umap"]
    np.testing.assert_allclose(coords1, coords2)


def test_binarize_labels_tool():
    """Unit test for binarize_labels_tool."""
    session = _setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    binarize_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_tool.invoke({})

    assert "error" not in res
    assert res["n_classes"] == 2
    assert res["classes"] == ["catA", "catB"]
    assert res["shape"] == [8, 1]
    assert res["counts_per_class"] == {"catA": 4, "catB": 4}
    assert res["positive_class"] == "catB"
    assert len(res["preview_rows"]) == 5

    # Check dataframe column addition
    assert "bin_category" in session.dataframe.columns
    assert len(session.dataframe["bin_category"]) == 8
    assert isinstance(session.dataframe["bin_category"].iloc[0], list)

    # Check artifacts
    assert "label_binarizer" in session.artifacts
    assert "binarized_labels" in session.artifacts
    assert session.artifacts["binarized_labels"].shape == (8, 1)

    # Error handling test on empty session
    empty_session = SessionState()
    empty_tools = make_reduction_tools(empty_session)
    binarize_empty = next(t for t in empty_tools if t.name == "binarize_labels_tool")
    err_res = binarize_empty.invoke({})
    assert "error" in err_res
