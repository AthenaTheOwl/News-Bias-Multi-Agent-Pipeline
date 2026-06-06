from __future__ import annotations

from scripts.post_deploy_canary import check_story_pack_variance


def test_story_pack_canary_catches_generic_collapse() -> None:
    assert check_story_pack_variance() == []
