import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools

def setup_fixture_session():
    """
    Helper function to load sample_fixture.csv and build the Document-Term Matrix
    using build_dtm_tool, ensuring session.feature_matrix, session.labels,
    and session.artifacts['count_vectorizer'] are set by the actual pipeline tools.
    """
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    return session

def test_cosine_similarity_known_answer_16():
    """
    Tests cosine_similarity_tool against Known Answer #16 in TEST_FIXTURE.md:
    - 0 vs 1 (both catA, near-identical text): 0.9847
    - 0 vs 3 (catA vs catB, share only 'always'): 0.0909
    - 3 vs 4 (both catB, near-identical text): 0.9847
    - 0 vs 7 (catA vs empty-text row): 0.0
    """
    session = setup_fixture_session()
    tools = make_tools(session)
    cosine_similarity_tool = next(t for t in tools if t.name == "cosine_similarity_tool")

    # Pair 0 vs 1
    res_0_1 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert res_0_1["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)
    assert res_0_1["mode"] == "index"
    assert res_0_1["zero_vector"] is False

    # Pair 0 vs 3
    res_0_3 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert res_0_3["cosine_similarity"] == pytest.approx(0.0909, abs=1e-3)
    assert res_0_3["zero_vector"] is False

    # Pair 3 vs 4
    res_3_4 = cosine_similarity_tool.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert res_3_4["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)
    assert res_3_4["zero_vector"] is False

    # Pair 0 vs 7 (empty row)
    res_0_7 = cosine_similarity_tool.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert res_0_7["cosine_similarity"] == 0.0
    assert res_0_7["zero_vector"] is True
    assert "note" in res_0_7
    assert "7" in res_0_7["note"]

def test_cosine_similarity_raw_text_mode():
    """
    Tests cosine_similarity_tool operating on raw text strings using the fitted CountVectorizer.
    """
    session = setup_fixture_session()
    tools = make_tools(session)
    cosine_similarity_tool = next(t for t in tools if t.name == "cosine_similarity_tool")

    # Compare text matching doc 0 vs doc 1
    res = cosine_similarity_tool.invoke({
        "text_1": "always alpha alpha alpha beta",
        "text_2": "always alpha alpha beta"
    })
    assert res["mode"] == "text"
    assert res["cosine_similarity"] == pytest.approx(0.9847, abs=1e-3)
    assert res["zero_vector"] is False

def test_feature_correlation_matrix_known_answer_19():
    """
    Tests feature_correlation_matrix_tool against Known Answer #19 in TEST_FIXTURE.md:
    - Terms by variance descending: ['alpha', 'gamma', 'delta', 'beta', 'always']
    - Variances: alpha=1.1875, gamma=1.1094, delta=0.25, beta=0.2344, always=0.1094
    - Full 5x5 correlation matrix checked with tolerance abs=1e-3.
    - session.pending_figure is populated with a matplotlib figure.
    """
    session = setup_fixture_session()
    tools = make_tools(session)
    feature_corr_tool = next(t for t in tools if t.name == "feature_correlation_matrix_tool")

    res = feature_corr_tool.invoke({"top_n": 20})

    # Expected term order by variance descending
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms

    # Expected variances
    expected_variances = [1.1875, 1.1094, 0.25, 0.2344, 0.1094]
    assert res["variances"] == pytest.approx(expected_variances, abs=1e-3)

    # Known Answer #19 Correlation Matrix
    expected_matrix = [
        [ 1.0000, -0.5718, -0.6882,  0.8885,  0.2601],
        [-0.5718,  1.0000,  0.8307, -0.6435,  0.3140],
        [-0.6882,  0.8307,  1.0000, -0.7746,  0.3780],
        [ 0.8885, -0.6435, -0.7746,  1.0000,  0.2928],
        [ 0.2601,  0.3140,  0.3780,  0.2928,  1.0000],
    ]

    actual_matrix = res["correlation_matrix"]
    assert len(actual_matrix) == 5
    for i in range(5):
        assert actual_matrix[i] == pytest.approx(expected_matrix[i], abs=1e-3)

    # Verify session.pending_figure is set
    assert session.pending_figure is not None

    # Verify zero_variance_excluded is 0
    assert res["zero_variance_excluded"] == 0
