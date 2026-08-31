from __future__ import annotations

from rag.protocol_metrics import ProtocolWindowMonitor, compute_protocol_metrics


def _rollout(*, error: str | None = None, final_ok: bool = True) -> dict:
    raw = (
        "plain text"
        if error and not error.startswith("final_answer_required:")
        else f'<answer>{{"can_answer":{str(final_ok).lower()},"answer":null}}</answer>'
    )
    return {
        "parse_errors": [error] if error else [],
        "trajectory": [
            {
                "force_final_answer": True,
                "raw_responses": {"answer_generator": raw},
                "answer": {"can_answer": final_ok, "answer": "x" if final_ok else None},
            }
        ],
    }


def test_protocol_metrics_distinguish_missing_answer_tag() -> None:
    metrics = compute_protocol_metrics([_rollout(), _rollout(error="Missing required tag: answer")])
    assert metrics["parse_failure_rate"] == 0.5
    assert metrics["missing_answer_tag_rate"] == 0.5
    assert metrics["final_compliance_rate"] == 0.5
    assert metrics["checkpoint_eligible"] is False


def test_final_answer_required_is_not_a_parse_failure() -> None:
    metrics = compute_protocol_metrics(
        [
            _rollout(
                error="final_answer_required: answer.can_answer must be true in the final round",
                final_ok=False,
            )
        ]
    )
    assert metrics["parse_failure_rate"] == 0.0
    assert metrics["missing_answer_tag_rate"] == 0.0
    assert metrics["final_compliance_rate"] == 0.0
    assert metrics["checkpoint_eligible"] is False


def test_mixed_final_requirement_and_structural_error_is_one_parse_failure() -> None:
    rollout = _rollout(
        error="final_answer_required: answer.can_answer must be true in the final round",
        final_ok=False,
    )
    rollout["parse_errors"].append("Invalid JSON payload")

    metrics = compute_protocol_metrics([rollout])

    assert metrics["parse_failure_rate"] == 1.0
    assert metrics["final_compliance_rate"] == 0.0


def test_protocol_monitor_warns_after_two_bad_nonoverlapping_windows() -> None:
    monitor = ProtocolWindowMonitor(
        window_size=2,
        max_parse_failure_rate=0.02,
        bad_windows_to_warn=2,
    )
    first = monitor.add([_rollout(error="bad"), _rollout()])
    second = monitor.add([_rollout(error="bad"), _rollout()])
    assert first["should_warn"] is False
    assert second["should_warn"] is True
    assert "should_stop" not in first
    assert "should_stop" not in second


def test_protocol_monitor_warning_is_edge_triggered_between_window_completions() -> None:
    monitor = ProtocolWindowMonitor(
        window_size=2,
        max_parse_failure_rate=0.02,
        bad_windows_to_warn=2,
    )
    bad_window = [_rollout(error="bad"), _rollout()]
    good_window = [_rollout(), _rollout()]

    assert monitor.add(bad_window)["should_warn"] is False
    assert monitor.add(bad_window)["should_warn"] is True
    assert monitor.add([_rollout(error="bad")])["should_warn"] is False
    assert monitor.add([_rollout()])["should_warn"] is False
    assert monitor.add(good_window)["should_warn"] is False
    assert monitor.add(bad_window)["should_warn"] is False
    assert monitor.add(bad_window)["should_warn"] is True
