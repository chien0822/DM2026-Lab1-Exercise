import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool():
    # Single session reused across loading and testing
    session = SessionState()

    # Load dataset using premade_tools
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Get student's build_dtm_tool
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    # Invoke build_dtm_tool
    result = build_dtm_tool.invoke({})

    # Assert known answer 4 numbers from TEST_FIXTURE.md
    assert result["n_documents"] == 8
    assert result["n_terms"] == 5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)
    assert session.feature_names == [
        "alpha",
        "always",
        "beta",
        "delta",
        "gamma",
    ]

    # Assert count_vectorizer artifact is stored
    assert "count_vectorizer" in session.artifacts
    assert session.artifacts["count_vectorizer"] is not None


def test_term_frequency_tool():
    # Fresh session for term_frequency_tool test
    session = SessionState()

    # Load dataset using premade_tools
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Build DTM first
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(
        t for t in tools if t.name == "term_frequency_tool"
    )

    build_dtm_tool.invoke({})

    # Invoke term_frequency_tool
    result = term_frequency_tool.invoke({"top_n": 20})

    # Assert known answer 14 numbers from TEST_FIXTURE.md
    top_freqs = result["top_frequencies"]
    assert top_freqs["alpha"] == 6
    assert top_freqs["always"] == 7
    assert top_freqs["beta"] == 3
    assert top_freqs["delta"] == 4
    assert top_freqs["gamma"] == 7

    assert result["total_term_count"] == 27
    assert result["total_terms_vocabulary"] == 5

    # Assert exact deterministic sorting (frequency descending, ties alphabetical)
    expected_order = ["always", "gamma", "alpha", "delta", "beta"]
    assert list(top_freqs.keys()) == expected_order


def test_dtm_heatmap_tool():
    # Fresh session for dtm_heatmap_tool test
    session = SessionState()

    # Load dataset using premade_tools
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Build DTM first
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(
        t for t in tools if t.name == "dtm_heatmap_tool"
    )

    build_dtm_tool.invoke({})

    # Test 1: Defaults (returns full 8x5 matrix)
    default_res = dtm_heatmap_tool.invoke({})
    assert default_res["n_documents"] == 8
    assert default_res["n_terms"] == 5
    assert default_res["document_indices"] == list(range(8))
    assert default_res["term_names"] == [
        "alpha",
        "always",
        "beta",
        "delta",
        "gamma",
    ]

    # Full 8x5 matrix check from TEST_FIXTURE.md known answer 21
    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    assert default_res["matrix"] == expected_full_matrix
    assert session.pending_figure is not None

    # Test 2: Sliced call with n_terms=2, n_documents=3
    session.pending_figure = None  # reset pending figure
    sliced_res = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert sliced_res["n_documents"] == 3
    assert sliced_res["n_terms"] == 2
    assert sliced_res["document_indices"] == [0, 1, 2]
    assert sliced_res["term_names"] == ["alpha", "always"]
    assert sliced_res["matrix"] == [[3, 1], [2, 1], [1, 1]]
    assert session.pending_figure is not None
