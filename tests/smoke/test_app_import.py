from __future__ import annotations

import importlib


def test_app_imports() -> None:
    module = importlib.import_module("app")
    assert module is not None
