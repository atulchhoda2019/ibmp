"""Kill drill: the pending proposal survives losing the process, and still needs its nonce."""
import pytest
from langgraph.types import Command

from app.graph_build import build_graph
from app.mocks import odl
from tests.conftest import config_for, make_state, run_turn

CHANGE = "change my 401k contribution to 8 percent"


def test_a_pending_proposal_resumes_in_a_new_process(tmp_path):
    path = str(tmp_path / "checkpoints.sqlite")
    state = make_state(CHANGE)
    config = config_for(state)

    first = run_turn(build_graph(path), state)          # process A
    proposal = first["response"]["proposal"]

    revived = build_graph(path)                          # process B, same sqlite file
    snapshot = revived.get_state(config)
    assert snapshot.next == ("corridor_wait_confirmation",)
    assert snapshot.values["proposal"].proposal_id == proposal["proposal_id"]

    final = revived.invoke(
        Command(resume={"proposal_id": proposal["proposal_id"], "nonce": proposal["nonce"]}), config)
    assert final["response"]["kind"] == "receipt"
    assert odl.get_elections("P-1001")["payload"]["rate"] == "0.08"


def test_the_revived_process_still_rejects_a_bad_nonce(tmp_path):
    path = str(tmp_path / "checkpoints.sqlite")
    state = make_state(CHANGE)
    config = config_for(state)
    run_turn(build_graph(path), state)

    revived = build_graph(path)
    with pytest.raises(ValueError):
        revived.invoke(Command(resume={"proposal_id": "PRP-nope", "nonce": "nope"}), config)
    assert odl.get_elections("P-1001")["payload"]["rate"] == "0.06"
