import json
import tempfile
import warnings
from pathlib import Path

import pytest
from datasets import Dataset

from bespokelabs.curator.types.curator_response import (
    CostInfo,
    CuratorResponse,
    PerformanceStats,
    RequestStats,
    TokenUsage,
)


def test_token_usage_add():
    a = TokenUsage(input=10, output=20, total=30)
    b = TokenUsage(input=5, output=15, total=20)
    a.add(b)
    assert a.input == 15
    assert a.output == 35
    assert a.total == 50


def test_cost_info_add():
    a = CostInfo(total_cost=1.0, input_cost_per_million=2.0, output_cost_per_million=3.0, projected_remaining_cost=0.5)
    b = CostInfo(total_cost=0.5, input_cost_per_million=1.0, output_cost_per_million=1.5, projected_remaining_cost=0.2)
    a.add(b)
    assert a.total_cost == 1.5
    assert a.input_cost_per_million == 3.0
    assert a.output_cost_per_million == 4.5
    assert a.projected_remaining_cost == 0.7


def test_cost_info_add_none_costs():
    a = CostInfo(total_cost=1.0, input_cost_per_million=None, output_cost_per_million=None)
    b = CostInfo(total_cost=0.5, input_cost_per_million=1.0, output_cost_per_million=1.5)
    a.add(b)
    assert a.total_cost == 1.5
    assert a.input_cost_per_million is None
    assert a.output_cost_per_million is None


def test_request_stats_add():
    a = RequestStats(total=10, succeeded=8, failed=1, in_progress=1, cached=2)
    b = RequestStats(total=5, succeeded=4, failed=0, in_progress=1, cached=1)
    a.add(b)
    assert a.total == 15
    assert a.succeeded == 12
    assert a.failed == 1
    assert a.in_progress == 2
    assert a.cached == 3


def test_performance_stats_add():
    a = PerformanceStats(total_time=10.0, requests_per_minute=60.0, input_tokens_per_minute=1000.0, output_tokens_per_minute=500.0, max_concurrent_requests=5)
    b = PerformanceStats(total_time=5.0, requests_per_minute=30.0, input_tokens_per_minute=500.0, output_tokens_per_minute=250.0, max_concurrent_requests=3)
    a.add(b)
    assert a.total_time == 15.0
    assert a.requests_per_minute == 90.0
    assert a.max_concurrent_requests == 8


def test_curator_response_post_init_none_metadata():
    ds = Dataset.from_dict({"a": [1]})
    resp = CuratorResponse(dataset=ds, metadata=None)
    assert resp.metadata == {}


def test_curator_response_getattr_raises():
    ds = Dataset.from_dict({"a": [1]})
    resp = CuratorResponse(dataset=ds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(AttributeError):
            _ = resp.nonexistent_attr


def test_curator_response_update_tracker_stats_offline():
    ds = Dataset.from_dict({"a": [1]})
    resp = CuratorResponse(dataset=ds)
    resp.update_tracker_stats(None)
    assert resp.token_usage.total == 0


def test_curator_response_get_failed_requests_no_path():
    ds = Dataset.from_dict({"a": [1]})
    resp = CuratorResponse(dataset=ds, failed_requests_path=None)
    assert list(resp.get_failed_requests()) == []


def test_curator_response_get_failed_requests_with_file():
    ds = Dataset.from_dict({"a": [1]})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"error": "timeout"}) + "\n")
        f.write(json.dumps({"error": "rate_limit"}) + "\n")
        path = Path(f.name)

    resp = CuratorResponse(dataset=ds, failed_requests_path=path)
    failed = list(resp.get_failed_requests())
    assert len(failed) == 2
    assert failed[0]["error"] == "timeout"
    path.unlink()


def test_curator_response_append():
    ds = Dataset.from_dict({"a": [1]})
    resp1 = CuratorResponse(
        dataset=ds,
        model_name="gpt-4",
        token_usage=TokenUsage(input=100, output=200, total=300),
        cost_info=CostInfo(total_cost=1.0),
        request_stats=RequestStats(total=10, succeeded=9, failed=1),
        performance_stats=PerformanceStats(total_time=5.0),
    )
    resp2 = CuratorResponse(
        dataset=ds,
        model_name="gpt-4",
        token_usage=TokenUsage(input=50, output=100, total=150),
        cost_info=CostInfo(total_cost=0.5),
        request_stats=RequestStats(total=5, succeeded=5, failed=0),
        performance_stats=PerformanceStats(total_time=3.0),
    )
    resp1.append(resp2)
    assert resp1.token_usage.total == 450
    assert resp1.cost_info.total_cost == 1.5
    assert resp1.request_stats.total == 15
    assert resp1.performance_stats.total_time == 8.0


def test_curator_response_to_dict():
    ds = Dataset.from_dict({"a": [1, 2]})
    resp = CuratorResponse(dataset=ds, model_name="gpt-4")
    d = resp.to_dict()
    assert d["dataset"]["size"] == 2
    assert d["model_name"] == "gpt-4"
    assert "token_usage" in d
    assert "cost_info" in d


def test_curator_response_save_and_load():
    ds = Dataset.from_dict({"a": [1, 2]})
    resp = CuratorResponse(
        dataset=ds,
        model_name="gpt-4",
        token_usage=TokenUsage(input=100, output=200, total=300),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        resp.save(tmpdir)
        loaded = CuratorResponse.load(tmpdir, ds)
        assert loaded.model_name == "gpt-4"
        assert loaded.token_usage.input == 100
        assert loaded.token_usage.total == 300
