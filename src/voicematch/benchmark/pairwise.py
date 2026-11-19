"""Pairwise 重标路径：kappa 低于门槛时改用比较式提问（设计 6.2 / 风险"标注不稳"）。

设计 6.2 明确：Cohen's kappa 低于门槛说明"问法不稳"——人耳对"从 8 位里选最像的
1–2 位"没有稳定共识——此时应改用 **pairwise 比较式提问** 重标，而不是降低门槛
把不一致藏起来。本模块把这条退路做成可执行的两步：

1. :func:`build_pairwise_sheet` —— 对争议 query，把候选歌手拆成两两比较对
   （"A 和 B 哪个更像这段清唱？"），每对独立展示，避免 8 选 1 的认知负担；
2. :func:`adjudicate_pairwise` —— 对每对做多数裁决，再把胜出关系聚合成该
   query 的 ``picked``，输出与 :func:`voicematch.benchmark.agreement.adjudicate`
   相同的 :class:`Adjudication`，因此可直接喂给 ``freeze``，无需改下游。

为什么要单独建这条链
--------------------
- **问题不同**：多选标注问"像谁"，pairwise 问"两者谁更像"。这是两个判据，
  不能混在一份 sheet 里，否则标注者会困惑到底回答哪一个。
- **口径仍然一致**：pairwise 裁决产出的是同样的 ``picked``（多数胜出的歌手），
  ``annotator_count`` 仍记实际人数，所以冻结后的 eval set 与 G1 口径不变——
  换的是"问法"，不是"指标"。
- **不篡改原标注**：pairwise 是重标结果，单独成文件；原始多选标注保留，便于
  审计"为何重标"（MANIFEST 的 disagreement 记录 + kappa）。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

from voicematch.benchmark.agreement import Adjudication


@dataclass(frozen=True)
class PairTask:
    """一个两两比较任务：给 query 听 A/B，选更像的一位。

    ``pair_id`` 在查询内唯一；``winner`` 由标注者填 ``a``/``b``，或 ``None``
    表示"分不出/都不像"。
    """

    query_id: str
    pair_id: str
    singer_a: str
    singer_b: str


@dataclass(frozen=True)
class PairAnnotation:
    """一位标注者对一对比较的回答。"""

    query_id: str
    pair_id: str
    annotator_id: str
    winner: str | None  # 'a' | 'b' | None（分不出）


def build_pairwise_sheet(
    contested: Mapping[str, Sequence[str]],
) -> list[PairTask]:
    """为争议 query 生成全部无序歌手对的比较任务。

    ``contested`` 把 query_id 映射到要比较的候选歌手（通常取多数标注里得票
    接近的头部几位，或整份候选集）。每对只出现一次，``pair_id`` 用两位歌手
    id 的字典序拼接，保证同一批输入可复现。

    Raises ``ValueError``：某 query 的歌手少于 2 位（无法构成比较）或含重复。
    """
    tasks: list[PairTask] = []
    for query_id, singers in contested.items():
        unique = list(dict.fromkeys(singers))
        if len(unique) < 2:
            raise ValueError(
                f"query {query_id!r}: need at least 2 singers for a pairwise "
                f"comparison, got {len(unique)}"
            )
        for a, b in combinations(sorted(unique), 2):
            tasks.append(
                PairTask(
                    query_id=query_id,
                    pair_id=f"{a}|{b}",
                    singer_a=a,
                    singer_b=b,
                )
            )
    return tasks


def adjudicate_pairwise(
    tasks: Sequence[PairTask],
    annotations: Sequence[PairAnnotation],
) -> dict[str, Adjudication]:
    """对每对做多数裁决，再按胜场聚合成每条 query 的 ``picked``。

    裁决流程：

    1. 每一对统计 ``a``/``b`` 的票数，严格多数（> 有效票数/2）胜出；无人达
       多数则该对比作"分不出"，不计入聚合；
    2. 每条 query 对胜出者累计胜场，取胜场**最多**的歌手进入 ``picked``（允许
       并列，仍遵守设计 6.2"1–2 位"的上限；超过 2 位并列时按胜场、再按 singer
       id 字典序截断到 2 位，保证确定性）；
    3. 没有任何胜出对时 ``picked`` 为空（"都不像"）。

    ``disputed`` 标记：某对未形成多数（平票/分不出）即为 True，供审计。

    Raises ``ValueError``：标注引用了不存在的 pair_id、winner 取值非法、或
    一对里 winner 不是该对的歌手。
    """
    tasks_by_id: dict[tuple[str, str], PairTask] = {}
    for task in tasks:
        key = (task.query_id, task.pair_id)
        if key in tasks_by_id:
            raise ValueError(f"duplicate pair task {key}")
        tasks_by_id[key] = task

    votes: dict[tuple[str, str], list[tuple[str, str | None]]] = {}
    annotators: dict[str, set[str]] = {}
    for annotation in annotations:
        key = (annotation.query_id, annotation.pair_id)
        task = tasks_by_id.get(key)
        if task is None:
            raise ValueError(f"annotation references unknown pair {key}")
        if annotation.winner not in ("a", "b", None):
            raise ValueError(
                f"pair {key}: winner must be 'a', 'b' or null, got {annotation.winner!r}"
            )
        votes.setdefault(key, []).append((annotation.annotator_id, annotation.winner))
        annotators.setdefault(annotation.query_id, set()).add(annotation.annotator_id)

    wins: dict[str, Counter] = {}
    any_unresolved: dict[str, bool] = {}
    for (query_id, pair_id), task_votes in votes.items():
        task = tasks_by_id[(query_id, pair_id)]
        pair_votes = [winner for _, winner in task_votes if winner is not None]
        counts = Counter(pair_votes)
        n = len(task_votes)
        winners_by_side = {
            "a": task.singer_a,
            "b": task.singer_b,
        }
        resolved = False
        for side, singer in winners_by_side.items():
            if counts[side] > n / 2:  # 严格多数，分母含"分不出"
                wins.setdefault(query_id, Counter())[singer] += 1
                resolved = True
        unresolved = any_unresolved.setdefault(query_id, False)
        any_unresolved[query_id] = unresolved or not resolved

    result: dict[str, Adjudication] = {}
    for query_id, annotator_ids in annotators.items():
        score = wins.get(query_id, Counter())
        # 胜场降序、singer id 升序，取前 2 位——确定性且遵守 1–2 位上限。
        ranked = sorted(score.items(), key=lambda item: (-item[1], item[0]))
        picked = frozenset(singer for singer, _ in ranked[:2]) if ranked else frozenset()
        result[query_id] = Adjudication(
            query_id=query_id,
            picked=picked,
            annotator_count=len(annotator_ids),
            disputed=any_unresolved.get(query_id, False),
        )
    return result
