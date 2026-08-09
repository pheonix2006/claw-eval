from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from claw_eval.cli import _require_provider_transport


def test_provider_transport_cli_contract_accepts_exact_match() -> None:
    cfg = SimpleNamespace(
        model=SimpleNamespace(provider_transport="anthropic-messages")
    )
    args = argparse.Namespace(provider_transport="anthropic-messages")

    _require_provider_transport(cfg, args)


def test_provider_transport_cli_contract_rejects_mismatch() -> None:
    cfg = SimpleNamespace(
        model=SimpleNamespace(provider_transport="openai-completions")
    )
    args = argparse.Namespace(provider_transport="anthropic-messages")

    with pytest.raises(RuntimeError, match="provider transport mismatch"):
        _require_provider_transport(cfg, args)
