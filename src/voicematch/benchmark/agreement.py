"""标注一致性与三人多数裁决（设计 6.2）。

基准集的可信度不只在"标了什么"，还在"标注本身稳不稳"。设计 6.2 要求
3 人独立标注取多数，并用 Cohen's kappa 检验一致性：低于门槛说明**问法**
不稳（人耳对"像"没有共同判据），应改用 pairwise 比较式提问，而不是调
阈值把不一致藏起来。

本模块刻意保持纯逻辑——没有 I/O、没有音频依赖——因为它是"标注质量"
这条判据的落点，和 :mod:`voicematch.eval.metrics` 一样，必须能在没有真实
数据时用合成标注完整测试。

三个必须守住的口径
------------------
1. **测不到就是 ``nan``，不是 0.0。** 只有一位标注者、或两位标注者的标签
   都恒定（边际乘积和为 1，分母为 0）时，kappa 无从算起，返回 ``nan``。
   ``0.0`` 是"测到了、刚好随机一致"的结论，二者不能混同（沿用 eval 的
   nan 约定）。
2. **多数是严格多数。** N 位标注者里，一个歌手要被选中必须得票 ``> N/2``。
   3 人时 2 票算多数，1 票不算。没有任何歌手达多数 => ``picked`` 为空，
   这是"都不像"这一真实答案（R9），不是"没标"。
3. **标注不能凭空造歌手。** 每条标注的 ``picked`` 必须来自该 query 展示
   过的候选集（sheet 的 candidates），且不超过 2 位（设计 6.2 的"1–2 位"）。
   违反即以命名 query_id 与 annotator_id 的 ``ValueError`` 拒绝，否则裁决
   结果会引用标注者根本没听过的歌手，污染 Top-K 分母。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

#: 一致性无法测量时的返回值。语义与 :mod:`voicematch.eval.metrics` 的 nan 一致。
UNMEASURABLE: float = float("nan")

#: 设计 6.2 的标注任务上限："选最像的 1–2 位"。
MAX_PICKS_PER_ANNOTATOR: int = 2

#: kappa 通过门槛（默认 0.6）。依据 Landis & Koch (1977)：0.61–0.80 为
#: "substantial agreement"，0.41–0.60 为 "moderate"。MVP 取 moderate 上界
#: 0.6 作为务实门槛——低于它说明标注者没形成稳定共识，问法本身不稳，应改用
#: pairwise 比较式提问重标（设计 6.2），而不是调低门槛把不一致藏起来。
#: CLI 的 --kappa-threshold 默认取此值；可在调用时覆盖，但覆盖需有理由记录。
KAPPA_THRESHOLD_DEFAULT: float = 0.6


@dataclass(frozen=True)
class Annotation:
    """一位标注者对一条 query 的独立标注。

    ``picked`` 是被选中的歌手 id，顺序无关；空元组表示"都不像"（设计 6.2
    允许的真实答案）。它必须是该 query 在标注清单里展示过的候选子集。
    """

    query_id: str
    annotator_id: str
    picked: tuple[str, ...]

    @property
    def top1(self) -> str | None:
        """首选歌手；"都不像"返回 ``None``。kappa 以首选为单一分类标签。"""
        return self.picked[0] if self.picked else None


@dataclass(frozen=True)
class Adjudication:
    """一条 query 的多数裁决结果。

    ``picked`` 是得票超过严格多数的歌手集合（frozen 便于 hashing 与比较）；
    ``annotator_count`` 是实际标注人数，写入 EvalQuery 的 ``annotator_count``；
    ``disputed`` 标记首选不一致——这是二次校验的输入，不是要删掉的数据。
    """

    query_id: str
    picked: frozenset[str]
    annotator_count: int
    disputed: bool


@dataclass(frozen=True)
class AgreementReport:
    """成对 Cohen's kappa 的汇总。

    ``pairwise`` 以 ``"annotator_a|annotator_b"`` 为键（按 id 排序），
    ``kappa_mean`` 是所有可测 pair 的均值；没有任何可测 pair 时为 ``nan``。
    """

    kappa_mean: float
    pairwise: dict[str, float]
    n_pairs: int
    n_measurable_pairs: int

    def is_below(self, threshold: float) -> bool:
        """均值是否低于门槛；测不到（nan）时返回 ``False``——不能据此判失败。"""
        return _finite(self.kappa_mean) and self.kappa_mean < threshold


def cohen_kappa(
    labels_a: Sequence[str | None], labels_b: Sequence[str | None]
) -> float:
    """两位标注者对同一批 item 的单一分类标签的一致度。

    ``labels_a[i]``/``labels_b[i]`` 是第 i 条 query 的首选（``None`` 表示"
    都不像"）。两名标注者的标签都恒定且相同 → 期望一致度为 1、分母为 0，
    此时一致性无从测量，返回 ``nan``（而非 1.0，后者会声称测到了完美一致）。

    Raises ``ValueError``：两序列长度不等，或任一序列为空（空集测不到东西，
    交给调用方按 nan 处理更明确）。
    """
    a = list(labels_a)
    b = list(labels_b)
    if len(a) != len(b):
        raise ValueError(
            f"label sequences must be equal length, got {len(a)} and {len(b)}"
        )
    n = len(a)
    if n == 0:
        raise ValueError("cohen_kappa needs at least one item; empty input measured nothing")

    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    labels = set(a) | set(b)
    p_a = {label: a.count(label) / n for label in labels}
    p_b = {label: b.count(label) / n for label in labels}
    expected = sum(p_a[label] * p_b[label] for label in labels)
    if expected >= 1.0:
        return UNMEASURABLE
    return (observed - expected) / (1.0 - expected)


def adjudicate(
    annotations: Sequence[Annotation],
    allowed_by_query: Mapping[str, Sequence[str]],
) -> dict[str, Adjudication]:
    """逐 query 做严格多数裁决。

    ``allowed_by_query`` 把 query_id 映射到该 query 展示过的候选歌手（标注
    清单的 ``candidates``），用于拒绝"标注了没听过的歌手"。每条 query 的
    标注在校验通过后按得票裁决；返回的 dict 键是出现过的 query_id。

    Raises ``ValueError``：标注引用了不在候选里的歌手、单条标注选中超过
    ``MAX_PICKS_PER_ANNOTATOR`` 位、或 query_id 没有对应的候选集。
    """
    grouped: dict[str, list[Annotation]] = {}
    for annotation in annotations:
        allowed = allowed_by_query.get(annotation.query_id)
        if allowed is None:
            raise ValueError(
                f"query {annotation.query_id!r}: annotation from "
                f"annotator {annotation.annotator_id!r} has no candidate set"
            )
        if len(annotation.picked) > MAX_PICKS_PER_ANNOTATOR:
            raise ValueError(
                f"query {annotation.query_id!r}, annotator "
                f"{annotation.annotator_id!r}: picked {len(annotation.picked)} "
                f"singers, at most {MAX_PICKS_PER_ANNOTATOR} allowed"
            )
        unknown = sorted(set(annotation.picked) - set(allowed))
        if unknown:
            raise ValueError(
                f"query {annotation.query_id!r}, annotator "
                f"{annotation.annotator_id!r}: picked singers {unknown} "
                "were not in the shown candidate set"
            )
        grouped.setdefault(annotation.query_id, []).append(annotation)

    result: dict[str, Adjudication] = {}
    for query_id, votes in grouped.items():
        counts = Counter(singer for vote in votes for singer in vote.picked)
        # 严格多数：得票必须超过半数，3 人时 2 票。
        picked = frozenset(
            singer for singer, count in counts.items() if count > len(votes) / 2
        )
        top1s = {vote.top1 for vote in votes}
        result[query_id] = Adjudication(
            query_id=query_id,
            picked=picked,
            annotator_count=len(votes),
            disputed=len(top1s) > 1,
        )
    return result


def agreement_report(annotations: Sequence[Annotation]) -> AgreementReport:
    """对所有标注者两两计算 Cohen's kappa，并汇总均值。

    只用两位标注者**都标注过**的 query（各自 top1 的公共 item），避免把"
    其中一人漏标"当成不一致。标注者不足两位、或所有 pair 都测不到时，
    ``kappa_mean`` 为 ``nan``。
    """
    by_annotator: dict[str, dict[str, str | None]] = {}
    for annotation in annotations:
        by_annotator.setdefault(annotation.annotator_id, {})[
            annotation.query_id
        ] = annotation.top1

    annotators = sorted(by_annotator)
    pairwise: dict[str, float] = {}
    for i, first in enumerate(annotators):
        for second in annotators[i + 1 :]:
            shared = sorted(set(by_annotator[first]) & set(by_annotator[second]))
            labels_a = [by_annotator[first][q] for q in shared]
            labels_b = [by_annotator[second][q] for q in shared]
            kappa = (
                cohen_kappa(labels_a, labels_b)
                if len(shared) >= 1
                else UNMEASURABLE
            )
            pairwise[f"{first}|{second}"] = kappa

    measurable = [value for value in pairwise.values() if _finite(value)]
    mean = sum(measurable) / len(measurable) if measurable else UNMEASURABLE
    return AgreementReport(
        kappa_mean=mean,
        pairwise=pairwise,
        n_pairs=len(pairwise),
        n_measurable_pairs=len(measurable),
    )


def _finite(value: float) -> bool:
    """Whether ``value`` is a real number (not nan/inf)."""
    import math

    return math.isfinite(value)
