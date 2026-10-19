import pytest
import numpy as np
import matplotlib.figure
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_check_missing_tool_known_answer_2():
    """
    Tests check_missing_tool against known answer #2 in TEST_FIXTURE.md:
    sample_fixture.csv has 8 rows, and exactly 1 row (row index 7) has missing/empty text.
    """
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample fixture dataset
    load_result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_result.get("n_documents") == 8

    # Get student's check_missing_tool
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    # Test report mode (drop=False)
    res_report = check_missing_tool.invoke({"drop": False})
    assert res_report["missing_count"] == 1
    assert res_report["total_rows"] == 8
    assert res_report["missing_indices"] == [7]
    assert res_report["action_taken"] == "reported"

    # Test drop mode (drop=True)
    res_drop = check_missing_tool.invoke({"drop": True})
    assert res_drop["missing_count"] == 1
    assert res_drop["total_rows"] == 8
    assert res_drop["rows_after"] == 7
    assert res_drop["missing_indices"] == [7]
    assert res_drop["action_taken"] == "dropped"
    assert len(session.dataframe) == 7
    assert 7 not in session.dataframe.index


def test_check_duplicates_tool_known_answer_3():
    """
    Tests check_duplicates_tool against known answer #3 in TEST_FIXTURE.md:
    sample_fixture.csv has 8 rows, with exactly 1 duplicate copy (row 6 is duplicate of row 5).
    When drop=True, keep=False removes both rows 5 and 6, leaving 6 rows.
    """
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample fixture dataset
    load_result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_result.get("n_documents") == 8

    # Get student's check_duplicates_tool
    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # Test report mode (drop=False)
    res_report = check_duplicates_tool.invoke({"drop": False})
    assert res_report["duplicate_count"] == 1
    assert res_report["rows_removed"] == 0
    assert res_report["total_rows"] == 8
    assert res_report["duplicate_indices"] == [6]
    assert res_report["action_taken"] == "reported"

    # Test drop mode (drop=True) with keep=False behavior
    res_drop = check_duplicates_tool.invoke({"drop": True})
    assert res_drop["duplicate_count"] == 1
    assert res_drop["rows_removed"] == 2
    assert res_drop["total_rows"] == 8
    assert res_drop["rows_after"] == 6
    assert set(res_drop["duplicate_indices"]) == {5, 6}
    assert res_drop["action_taken"] == "dropped"
    assert len(session.dataframe) == 6
    assert 5 not in session.dataframe.index
    assert 6 not in session.dataframe.index


def test_sample_data_tool_known_answer_13():
    """
    Tests sample_data_tool against known answer #13 in TEST_FIXTURE.md:
    Subsampling with n=4, random_state=42 returns exactly rows [1, 5, 0, 7].
    """
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample fixture dataset
    load_result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_result.get("n_documents") == 8

    # Get student's sample_data_tool
    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    # Sample n=4 with random_state=42
    res_sample = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert res_sample["rows_before"] == 8
    assert res_sample["rows_after"] == 4
    assert res_sample["sampled_indices"] == [1, 5, 0, 7]
    assert list(session.dataframe.index) == [1, 5, 0, 7]
    assert len(session.dataframe) == 4


def test_describe_data_tool_known_answer_18():
    """
    Tests describe_data_tool against known answer #18 in TEST_FIXTURE.md:
    Character count text_lengths: [29, 23, 17, 30, 24, 18, 18, 0]
    Overall: count=8, mean=19.875, std=9.4330, min=0, 25%=17.75, 50%=20.5, 75%=25.25, max=30
    catA: count=4, mean=17.25, std=12.5, min=0, 25%=12.75, 50%=20.0, 75%=24.5, max=29
    catB: count=4, mean=22.5, std=5.7446, min=18, 25%=18.0, 50%=21.0, 75%=25.5, max=30
    """
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample fixture dataset
    load_result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_result.get("n_documents") == 8

    # Get student's describe_data_tool
    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({})

    # Check figure set on session.pending_figure
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    # Check overall stats against known answer #18
    overall = res["overall_stats"]
    assert overall["count"] == 8
    assert np.isclose(overall["mean"], 19.875, atol=1e-3)
    assert np.isclose(overall["std"], 9.4330, atol=1e-3)
    assert overall["min"] == 0
    assert np.isclose(overall["25%"], 17.75, atol=1e-3)
    assert np.isclose(overall["50%"], 20.5, atol=1e-3)
    assert np.isclose(overall["75%"], 25.25, atol=1e-3)
    assert overall["max"] == 30

    # Check per-category stats
    catA = res["per_category_stats"]["catA"]
    assert catA["count"] == 4
    assert np.isclose(catA["mean"], 17.25, atol=1e-3)
    assert np.isclose(catA["std"], 12.5, atol=1e-3)
    assert catA["min"] == 0
    assert np.isclose(catA["25%"], 12.75, atol=1e-3)
    assert np.isclose(catA["50%"], 20.0, atol=1e-3)
    assert np.isclose(catA["75%"], 24.5, atol=1e-3)
    assert catA["max"] == 29

    catB = res["per_category_stats"]["catB"]
    assert catB["count"] == 4
    assert np.isclose(catB["mean"], 22.5, atol=1e-3)
    assert np.isclose(catB["std"], 5.7446, atol=1e-3)
    assert catB["min"] == 18
    assert np.isclose(catB["25%"], 18.0, atol=1e-3)
    assert np.isclose(catB["50%"], 21.0, atol=1e-3)
    assert np.isclose(catB["75%"], 25.5, atol=1e-3)
    assert catB["max"] == 30


def test_tokenize_tool_known_answer_10():
    """
    Tests tokenize_tool against known answer #10 in TEST_FIXTURE.md:
    Asserts expected tokens for all 8 rows of sample_fixture.csv in lowercased word_tokenize order.
    Row 7 (empty text) produces [].
    """
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample fixture dataset
    load_result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_result.get("n_documents") == 8

    # Get student's tokenize_tool
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({"lowercase": True})

    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]

    expected_total = sum(len(t) for t in expected_tokens)

    assert res["n_documents"] == 8
    assert res["n_empty_documents"] == 1
    assert res["total_tokens"] == expected_total
    assert res["mean_tokens_per_document"] == pytest.approx(expected_total / 8)

    # Check exact tokens stored per document in session.dataframe["unigrams"]
    unigrams = session.dataframe["unigrams"].tolist()
    assert unigrams == expected_tokens


def test_list_files_tool_known_answer_11():
    """
    Tests list_files_tool against known answer #11 in TEST_FIXTURE.md:
    Listing "newdataset" subdirectory should return directories: [] and files: ["Reddit-stock-sentiment.csv"].
    """
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]
    assert res["subdirectory"] == "newdataset"


def test_inspect_data_tool_custom():
    """
    Tests inspect_data_tool on sample_fixture.csv:
    Checks top n=3 row previews, column metadata, clamped flag, string truncation (max_chars),
    and unigram list formatting after tokenization.
    """
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_result.get("n_documents") == 8

    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    # Tokenize first so unigrams column exists
    tokenize_tool.invoke({})

    # Test n=3, max_chars=10 inspection
    res = inspect_data_tool.invoke({"n": 3, "max_chars": 10})

    assert res["n_requested"] == 3
    assert res["n_returned"] == 3
    assert res["n_rows"] == 8
    assert res["clamped"] is False
    assert "text" in res["columns"]
    assert "unigrams" in res["columns"]
    assert len(res["preview"]) == 3

    # Check index label and cell formatting
    rec0 = res["preview"][0]
    assert rec0["index"] == 0
    assert rec0["text"] == "always alp..."  # truncated at max_chars=10
    assert isinstance(rec0["unigrams"], dict)
    assert rec0["unigrams"]["length"] == 5
    assert rec0["unigrams"]["first"] == ["always", "alpha", "alpha", "alpha", "beta"]

    # Test clamping when n > total rows
    res_clamped = inspect_data_tool.invoke({"n": 20})
    assert res_clamped["n_requested"] == 20
    assert res_clamped["n_returned"] == 8
    assert res_clamped["clamped"] is True
