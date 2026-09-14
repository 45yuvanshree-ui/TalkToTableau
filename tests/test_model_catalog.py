from __future__ import annotations

import json
from urllib.error import URLError

import pytest

from routing.model_catalog import (
    ModelCatalogResponseError,
    ModelCatalogUnavailableError,
    OpenRouterModelCatalog,
)


def test_parse_models_normalizes_openrouter_metadata() -> None:
    payload = {
        "data": [
            {
                "id": "example/free-model:free",
                "canonical_slug": "example/free-model",
                "name": "Example Free Model",
                "context_length": 32768,
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": [
                    "temperature",
                    "tools",
                    "structured_outputs",
                    "reasoning",
                ],
                "pricing": {
                    "prompt": "0",
                    "completion": "0",
                },
            }
        ]
    }

    models = OpenRouterModelCatalog._parse_models(payload)

    assert len(models) == 1

    model = models[0]

    assert model.model_id == "example/free-model:free"
    assert model.name == "Example Free Model"
    assert model.canonical_slug == "example/free-model"
    assert model.context_length == 32768
    assert model.input_modalities == ("text",)
    assert model.output_modalities == ("text",)

    assert model.supports_tools is True
    assert model.supports_structured_outputs is True
    assert model.supports_reasoning is True

    assert model.is_text_model is True
    assert model.is_free is True


def test_parse_models_uses_model_id_when_name_is_missing() -> None:
    payload = {
        "data": [
            {
                "id": "example/model",
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
                "pricing": {
                    "prompt": "0.000001",
                    "completion": "0.000002",
                },
            }
        ]
    }

    models = OpenRouterModelCatalog._parse_models(payload)

    assert len(models) == 1
    assert models[0].name == "example/model"
    assert models[0].is_free is False


def test_parse_models_skips_invalid_model_entries() -> None:
    payload = {
        "data": [
            None,
            {},
            {"id": ""},
            {"id": 123},
            {
                "id": "valid/model",
                "name": "Valid Model",
            },
        ]
    }

    models = OpenRouterModelCatalog._parse_models(payload)

    assert len(models) == 1
    assert models[0].model_id == "valid/model"


def test_parse_models_rejects_invalid_root() -> None:
    with pytest.raises(KeyError):
        OpenRouterModelCatalog._parse_models({})


def test_free_model_detection_requires_zero_prompt_and_completion_price() -> None:
    payload = {
        "data": [
            {
                "id": "free/model:free",
                "pricing": {
                    "prompt": "0",
                    "completion": "0",
                },
            },
            {
                "id": "paid/model",
                "pricing": {
                    "prompt": "0",
                    "completion": "0.000001",
                },
            },
        ]
    }

    models = OpenRouterModelCatalog._parse_models(payload)

    assert models[0].is_free is True
    assert models[1].is_free is False


def test_list_models_sends_expected_query(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        status = 200

        def read(self):
            return json.dumps(
                {
                    "data": [
                        {
                            "id": "example/model",
                            "name": "Example Model",
                            "architecture": {
                                "input_modalities": ["text"],
                                "output_modalities": ["text"],
                            },
                            "pricing": {
                                "prompt": "0",
                                "completion": "0",
                            },
                        }
                    ]
                }
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.headers)
        return FakeResponse()

    monkeypatch.setattr(
        "routing.model_catalog.urlopen",
        fake_urlopen,
    )

    catalog = OpenRouterModelCatalog(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        timeout_seconds=7,
    )

    models = catalog.list_models(
        output_modalities="text",
        supported_parameters="tools",
        query="example",
    )

    assert len(models) == 1
    assert models[0].model_id == "example/model"

    assert (
        captured["url"]
        == "https://openrouter.ai/api/v1/models"
        "?output_modalities=text"
        "&supported_parameters=tools"
        "&q=example"
    )

    assert captured["timeout"] == 7
    assert captured["headers"]["Authorization"] == "Bearer test-key"


def test_list_free_text_models_filters_models() -> None:
    catalog = OpenRouterModelCatalog(api_key="test-key")

    payload = {
        "data": [
            {
                "id": "free/text-model",
                "name": "Free Text Model",
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
                "pricing": {
                    "prompt": "0",
                    "completion": "0",
                },
            },
            {
                "id": "paid/text-model",
                "name": "Paid Text Model",
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
                "pricing": {
                    "prompt": "0.000001",
                    "completion": "0.000001",
                },
            },
            {
                "id": "free/image-model",
                "name": "Free Image Model",
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["image"],
                },
                "pricing": {
                    "prompt": "0",
                    "completion": "0",
                },
            },
        ]
    }

    monkeypatch = pytest.MonkeyPatch()

    def fake_request_models(
        self,
        *,
        output_modalities,
        supported_parameters,
        query,
    ):
        return payload

    monkeypatch.setattr(
        OpenRouterModelCatalog,
        "_request_models",
        fake_request_models,
    )

    try:
        models = catalog.list_free_text_models()
    finally:
        monkeypatch.undo()

    assert [model.model_id for model in models] == [
        "free/text-model"
    ]


def test_missing_api_key_is_safe(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    catalog = OpenRouterModelCatalog(
        api_key="",
    )

    with pytest.raises(ModelCatalogUnavailableError):
        catalog.list_models()


def test_http_errors_become_catalog_unavailable(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise URLError("network unavailable")

    monkeypatch.setattr(
        "routing.model_catalog.urlopen",
        fake_urlopen,
    )

    catalog = OpenRouterModelCatalog(
        api_key="test-key",
    )

    with pytest.raises(ModelCatalogUnavailableError):
        catalog.list_models()


def test_invalid_json_becomes_response_error(monkeypatch) -> None:
    class FakeResponse:
        status = 200

        def read(self):
            return b"not-json"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "routing.model_catalog.urlopen",
        lambda request, timeout: FakeResponse(),
    )

    catalog = OpenRouterModelCatalog(
        api_key="test-key",
    )

    with pytest.raises(ModelCatalogResponseError):
        catalog.list_models()


def test_missing_data_list_becomes_response_error(monkeypatch) -> None:
    class FakeResponse:
        status = 200

        def read(self):
            return b'{"unexpected": []}'

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        "routing.model_catalog.urlopen",
        lambda request, timeout: FakeResponse(),
    )

    catalog = OpenRouterModelCatalog(
        api_key="test-key",
    )

    with pytest.raises(ModelCatalogResponseError):
        catalog.list_models()