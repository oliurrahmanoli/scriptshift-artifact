import pytest

from runners.endpoints import _check_base_url


def test_excluded_endpoints_rejected():
    for bad in ["https://openrouter.ai/api/v1", "https://api.openai.com/v1",
                "https://api.mistral.ai/v1"]:
        with pytest.raises(RuntimeError):
            _check_base_url(bad)


def test_local_endpoint_allowed():
    _check_base_url("http://localhost:8099/v1")  # should not raise
