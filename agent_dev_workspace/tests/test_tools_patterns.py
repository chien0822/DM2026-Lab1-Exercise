import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def _setup_session():
    """Helper to initialize a fresh session with sample_fixture.csv loaded."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    return session


def test_mine_patterns_fpgrowth():
    """Known answer #17: fpgrowth mining on sample_fixture.csv."""
    session = _setup_session()
    tools = make_patterns_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # 1. Variance filtering keeps zero terms on sample_fixture
    res_var = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert "error" not in res_var
    assert res_var["n_patterns"] == 0
    assert "no terms survived filtering" in res_var.get("note", "")

    # 2. TF-IDF filtering on catA -> keeps alpha only, support 3
    res_catA = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert "error" not in res_catA
    assert res_catA["n_patterns"] == 1
    assert res_catA["patterns"][0] == {"items": ["alpha"], "support": 3}
    assert session.artifacts["patterns_catA_tfidf_fpgrowth"] == [{"items": ["alpha"], "support": 3}]

    # 3. TF-IDF filtering on catB -> keeps gamma only, support 4
    res_catB = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert "error" not in res_catB
    assert res_catB["n_patterns"] == 1
    assert res_catB["patterns"][0] == {"items": ["gamma"], "support": 4}
    assert session.artifacts["patterns_catB_tfidf_fpgrowth"] == [{"items": ["gamma"], "support": 4}]


def test_mine_patterns_topk_and_maxfpgrowth():
    """Known answer #20: topk and maxfpgrowth mining on sample_fixture.csv."""
    session = _setup_session()
    tools = make_patterns_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # 1. topk (k=1) on catA -> returns {alpha} support 3
    res_topk_a = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert "error" not in res_topk_a
    assert res_topk_a["n_patterns"] == 1
    assert res_topk_a["patterns"][0] == {"items": ["alpha"], "support": 3}

    # 2. topk (k=1) on catB -> returns {gamma} support 4
    res_topk_b = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert "error" not in res_topk_b
    assert res_topk_b["n_patterns"] == 1
    assert res_topk_b["patterns"][0] == {"items": ["gamma"], "support": 4}

    # 3. maxfpgrowth returns 0 patterns for single-surviving term dataset (expected PAMI behavior)
    res_max_a = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert "error" not in res_max_a
    assert res_max_a["n_patterns"] == 0

    res_max_b = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert "error" not in res_max_b
    assert res_max_b["n_patterns"] == 0


def test_mine_patterns_parameter_validation():
    """Test parameter validation and error handling in mine_patterns_tool."""
    session = _setup_session()
    tools = make_patterns_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Invalid category
    err_cat = mine_tool.invoke({"category_name": "nonexistent", "filtering_method": "tfidf"})
    assert "error" in err_cat

    # Missing min_sup for fpgrowth
    err_sup = mine_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth"})
    assert "error" in err_sup

    # Missing k for topk
    err_k = mine_tool.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk"})
    assert "error" in err_k
