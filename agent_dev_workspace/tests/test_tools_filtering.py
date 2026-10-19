import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool():
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

    # Build DTM first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Get variance_filter_tool
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(
        t for t in filtering_tools if t.name == "variance_filter_tool"
    )

    # Invoke variance_filter_tool with threshold=0.15
    result = variance_filter_tool.invoke({"threshold": 0.15})

    # Assert known answer 5 summary numbers from TEST_FIXTURE.md
    assert result["total_terms"] == 5
    assert result["n_kept"] == 4
    assert result["n_removed"] == 1

    # Check kept vs removed term sets
    assert set(result["kept_terms_sample"]) == {"alpha", "beta", "delta", "gamma"}
    assert result["removed_terms_sample"] == ["always"]

    # Verify per-term variances stored in full_report DataFrame
    stored = session.results_store[result["result_id"]]
    full_report = stored["full_report"]
    variances = dict(zip(full_report["term"], full_report["variance"]))

    assert variances["alpha"] == pytest.approx(1.1875, abs=1e-4)
    assert variances["always"] == pytest.approx(0.109375, abs=1e-4)
    assert variances["beta"] == pytest.approx(0.234375, abs=1e-4)
    assert variances["delta"] == pytest.approx(0.25, abs=1e-4)
    assert variances["gamma"] == pytest.approx(1.109375, abs=1e-4)


def test_pearson_filter_tool():
    # Fresh session for pearson_filter_tool test
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

    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Get pearson_filter_tool
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(
        t for t in filtering_tools if t.name == "pearson_filter_tool"
    )

    # Invoke pearson_filter_tool for target_class="catB"
    result = pearson_filter_tool.invoke({"target_class": "catB", "top_n": 15})

    assert result["target_class"] == "catB"
    assert result["total_terms"] == 5
    assert result["ranked_terms"] == 5
    assert result["zero_variance_excluded"] == 0

    # Verify Pearson r values against TEST_FIXTURE.md known answer 6
    top_corrs = result["top_correlations"]
    assert top_corrs["alpha"]["r"] == pytest.approx(-0.6882, abs=1e-3)
    assert top_corrs["always"]["r"] == pytest.approx(0.3780, abs=1e-3)
    assert top_corrs["beta"]["r"] == pytest.approx(-0.7746, abs=1e-3)
    assert top_corrs["delta"]["r"] == pytest.approx(1.0000, abs=1e-3)
    assert top_corrs["gamma"]["r"] == pytest.approx(0.8307, abs=1e-3)

    # Verify ranking by |r| descending: delta (1.0), gamma (0.8307), beta (-0.7746), alpha (-0.6882), always (0.3780)
    expected_rank = ["delta", "gamma", "beta", "alpha", "always"]
    assert list(top_corrs.keys()) == expected_rank

    # Verify full_report columns
    stored = session.results_store[result["result_id"]]
    full_report = stored["full_report"]
    assert list(full_report.columns) == ["term", "pearson_r", "p_value"]


def test_spearman_filter_tool():
    # Fresh session for spearman_filter_tool test
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

    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Get spearman_filter_tool
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(
        t for t in filtering_tools if t.name == "spearman_filter_tool"
    )

    # Invoke spearman_filter_tool for target_class="catB"
    result = spearman_filter_tool.invoke({"target_class": "catB", "top_n": 15})

    assert result["target_class"] == "catB"
    assert result["total_terms"] == 5
    assert result["terms_considered"] == 5
    assert result["zero_variance_excluded"] == 0

    # Verify Spearman rho values against TEST_FIXTURE.md known answer 6
    top_corrs = result["top_correlations"]
    assert top_corrs["alpha"]["r"] == pytest.approx(-0.7500, abs=1e-3)
    assert top_corrs["always"]["r"] == pytest.approx(0.3780, abs=1e-3)
    assert top_corrs["beta"]["r"] == pytest.approx(-0.7746, abs=1e-3)
    assert top_corrs["delta"]["r"] == pytest.approx(1.0000, abs=1e-3)
    assert top_corrs["gamma"]["r"] == pytest.approx(0.9363, abs=1e-3)

    # Verify ranking by |rho| descending: delta (1.0), gamma (0.9363), beta (-0.7746), alpha (-0.7500), always (0.3780)
    expected_rank = ["delta", "gamma", "beta", "alpha", "always"]
    assert list(top_corrs.keys()) == expected_rank

    # Verify full_report columns
    stored = session.results_store[result["result_id"]]
    full_report = stored["full_report"]
    assert list(full_report.columns) == ["term", "spearman_r", "p_value"]
