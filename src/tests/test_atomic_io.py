from __future__ import annotations

import hashlib
import json

import pytest

from radar_ln3.data.atomic import write_bytes, write_json


def test_write_bytes_returns_file_hash_and_reuses_identical_content(tmp_path):
    target = tmp_path / "nested" / "payload.bin"
    payload = b"canonical\x00bytes"

    digest = write_bytes(target, payload, overwrite=False, label="数据产物")

    assert target.read_bytes() == payload
    assert digest == hashlib.sha256(payload).hexdigest()
    assert write_bytes(target, payload, overwrite=False, label="数据产物") == digest


def test_write_bytes_refuses_different_content_without_overwrite(tmp_path):
    target = tmp_path / "payload.bin"
    write_bytes(target, b"first", overwrite=False, label="数据产物")

    with pytest.raises(FileExistsError) as error:
        write_bytes(target, b"second", overwrite=False, label="数据产物")

    message = str(error.value)
    assert "拒绝覆盖内容不同的数据产物" in message
    assert str(target) in message
    assert target.read_bytes() == b"first"


def test_write_bytes_overwrite_replaces_and_returns_new_hash(tmp_path):
    target = tmp_path / "payload.bin"
    write_bytes(target, b"first", overwrite=True, label="数据产物")

    digest = write_bytes(target, b"second", overwrite=True, label="数据产物")

    assert target.read_bytes() == b"second"
    assert digest == hashlib.sha256(b"second").hexdigest()


def test_write_json_serialises_utf8_with_trailing_newline(tmp_path):
    target = tmp_path / "manifest.json"
    payload = {"名称": "雷达", "values": [1, 2.5, None, True]}

    digest = write_json(target, payload, overwrite=False, label="总 manifest")

    expected = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    assert target.read_text(encoding="utf-8") == expected
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert digest == hashlib.sha256(expected.encode("utf-8")).hexdigest()


def test_write_json_sorts_keys_when_requested(tmp_path):
    target = tmp_path / "manifest.json"

    write_json(
        target,
        {"b": 2, "a": {"z": 1, "y": 2}},
        overwrite=False,
        sort_keys=True,
        label="feature cache manifest",
    )

    text = target.read_text(encoding="utf-8")
    assert text == json.dumps(
        {"a": {"y": 2, "z": 1}, "b": 2}, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"


def test_write_json_refuses_different_content_without_overwrite(tmp_path):
    target = tmp_path / "manifest.json"
    write_json(target, {"version": 1}, overwrite=False, label="总 manifest")

    with pytest.raises(FileExistsError) as error:
        write_json(target, {"version": 2}, overwrite=False, label="总 manifest")

    assert "拒绝覆盖内容不同的总 manifest" in str(error.value)
