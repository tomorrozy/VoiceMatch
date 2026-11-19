"""服务端适配层的 R29 配对状态测试。

结果页必须能回答"这个分数是校准后的相似度，还是只能当排序"。CLI 只给状态
token，适配层再补一句中文文案；两者缺一不可——只靠 ``calibrated`` 会漏掉
"``calibrated=true`` 但库由旧版本构建、无法验证配对"这种危险组合。

这些测试只构造索引，不解码音频：``pairing_status`` 是纯函数，适配层把它
包成 payload + 文案，正是最该被钉住的部分。
"""

from pathlib import Path

import numpy as np

from server.engine import LibraryBundle, _pairing_payload
from voicematch.config import EMBED_DIM
from voicematch.features.extract import FeatureVector
from voicematch.library import (
    CROSS_RECORDING,
    EMPTY_LIBRARY,
    SINGLE_RECORDING,
    UNVERIFIABLE_LEGACY,
    LibraryIndex,
    Segment,
    Singer,
)


def _index(materials: dict[str, list[str]]) -> LibraryIndex:
    """Build a minimal index whose segments carry the given material ids."""
    segments: list[Segment] = []
    for singer_id, material_ids in materials.items():
        for offset, material_id in enumerate(material_ids):
            segments.append(
                Segment(
                    f"{singer_id}-{offset:02d}",
                    singer_id,
                    0.0,
                    1.0,
                    material_id=material_id,
                )
            )
    singers = {
        singer_id: Singer(singer_id, f"Singer {singer_id}", "female", "pop")
        for singer_id in materials
    }
    return LibraryIndex(
        version="0" * 16,
        model_id="test",
        singers=singers,
        segments=tuple(segments),
        prototypes={
            singer_id: np.zeros(EMBED_DIM, dtype=np.float32)
            for singer_id in materials
        },
        features={
            segment.segment_id: FeatureVector(
                pitch={}, timbre={}, resonance={}, style={}
            )
            for segment in segments
        },
        embeddings=np.zeros((len(segments), EMBED_DIM), dtype=np.float32),
    )


def test_cross_recording_library_needs_no_notice():
    payload = _pairing_payload(_index({"s1": ["a", "b"], "s2": ["c", "d"]}))

    assert payload == {
        "status": CROSS_RECORDING,
        "single_material_singers": [],
        "notice": "",
    }


def test_single_recording_library_names_the_singers_and_explains_the_score():
    payload = _pairing_payload(_index({"s1": ["a"], "s2": ["c", "d"]}))

    assert payload["status"] == SINGLE_RECORDING
    assert payload["single_material_singers"] == ["s1"]
    # 文案要同时说清"几个人有问题"和"分数还能怎么用"，否则页面只能自己编。
    assert "1 位歌手" in payload["notice"]
    assert "不代表校准后的相似度" in payload["notice"]


def test_legacy_library_is_reported_as_unverifiable():
    payload = _pairing_payload(_index({"s1": [""]}))

    assert payload["status"] == UNVERIFIABLE_LEGACY
    assert "旧版本" in payload["notice"]


def test_empty_library_says_so():
    payload = _pairing_payload(_index({}))

    assert payload["status"] == EMPTY_LIBRARY
    assert "歌手库为空" in payload["notice"]


def test_health_of_a_real_bundle_carries_the_pairing_block():
    root = Path(__file__).resolve().parents[2] / "server" / "demo_library"
    health = LibraryBundle(root).health()

    # 演示库每人两份 take，属于合规状态；这里同时验证字段真的进了 /health。
    assert health["pairing"]["status"] == CROSS_RECORDING
    assert health["pairing"]["single_material_singers"] == []
    assert health["pairing"]["notice"] == ""
