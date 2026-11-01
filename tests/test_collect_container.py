import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "collect_container", Path(__file__).resolve().parents[1] / "docker" / "collect_container.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)


def receipt(**changes):
    boards = {m: {"state": "COMPLETE", "ref": 100 + k, "error": ""} for k, m in enumerate(collect.METRICS)}
    return {"track": "track_1", "file": "/x/track1_cand_A.parquet", "submissions": boards, **changes}


def test_complete_receipt_passes():
    assert collect.receipt_errors(receipt(), "track1_cand_A.parquet") == []


def test_receipt_for_another_file_or_track_fails():
    assert collect.receipt_errors(receipt(), "track1_cand_B.parquet")
    assert collect.receipt_errors(receipt(track="track_3"), "track1_cand_A.parquet")


def test_missing_or_unscored_board_fails():
    gone = receipt()
    del gone["submissions"]["acc_o_cm"]
    assert any("acc_o_cm" in e for e in collect.receipt_errors(gone, "track1_cand_A.parquet"))
    pending = receipt()
    pending["submissions"]["cd_h_cm"]["state"] = "PENDING"
    assert any("cd_h_cm" in e for e in collect.receipt_errors(pending, "track1_cand_A.parquet"))


def test_two_commits_in_one_file_stop_the_build():
    pd = pytest.importorskip("pandas")
    base = "https://github.com/o/r/commit/"
    with pytest.raises(SystemExit, match="2 commits"):
        collect.cited_commit(pd.DataFrame({"code_commit_url": [base + "a" * 40, base + "b" * 40]}))
    assert collect.cited_commit(pd.DataFrame({"code_commit_url": [base + "a" * 40]}))[1] == "a" * 40
