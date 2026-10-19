from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


def test_load_dataset_tool():
    # Instantiate session state
    session = SessionState()

    # Load premade tools using workspace loader
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # Invoke load_dataset_tool on sample_fixture.csv
    result = load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Assert expected outputs based on TEST_FIXTURE.md known answer 1
    assert result["n_documents"] == 8
    assert result["counts_per_category"] == {"catA": 4, "catB": 4}
