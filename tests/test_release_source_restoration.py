import hashlib
import json

import pytest

from scripts.restore_research_sources import exact_json_bytes


def test_line_endings_can_restore_identity_without_changing_json():
    original = b'{\n  "value": 1\n}\n'
    expected = original.replace(b'\n', b'\r\n')
    restored = exact_json_bytes(original, hashlib.sha256(expected).hexdigest())
    assert restored == expected
    assert json.loads(restored) == json.loads(original)


def test_different_content_cannot_be_given_an_existing_identity():
    with pytest.raises(ValueError):
        exact_json_bytes(b'{"value": 2}', hashlib.sha256(b'{"value": 1}').hexdigest())
