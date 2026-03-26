from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any


def try_load_json(text: str) -> Any | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
            return obj
        except JSONDecodeError:
            continue
    return None


def dump_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
