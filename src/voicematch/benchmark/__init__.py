"""真人盲听基准集的采集工具链（设计 6.1/6.2）。

这是"真人采集 → 盲听标注 → 多数裁决 → 冻结基准集"这条链的可执行化。
链的前三环（招募、录音、听标）必须由真人完成，本包把它们变成填表式操作：

- ``sheet``：把真人录音 + 歌手库变成盲听候选清单（系统答案与清单物理分离）；
- ``agreement``：Cohen's kappa 一致性检验 + 严格多数裁决；低于门槛时用
  ``pairwise`` 比较式提问重标（设计 6.2 的"标注不稳"退路）；
- ``freeze``：把裁决结果锁成可被 ``run_eval`` 直接消费的版本化 eval set。

与 :mod:`voicematch.eval` 的边界：eval 负责"给定标注，指标是多少"；benchmark
负责"标注本身从哪里来、稳不稳、怎么锁版"。冻结产出的 eval_set.json 是两者
的唯一接口——它满足 ``load_eval_set`` 的契约，所以 G1 数字无法区分它来自
本工具链还是手写，但 MANIFEST 的 ``label_source`` 会说明真值来源。
"""

from voicematch.benchmark.agreement import (
    Adjudication,
    AgreementReport,
    Annotation,
    KAPPA_THRESHOLD_DEFAULT,
    adjudicate,
    agreement_report,
    cohen_kappa,
)
from voicematch.benchmark.freeze import freeze
from voicematch.benchmark.samples import (
    diagnose_catalog,
    diagnose_material,
    export_samples,
)
from voicematch.benchmark.pairwise import (
    PairAnnotation,
    PairTask,
    adjudicate_pairwise,
    build_pairwise_sheet,
)

__all__ = [
    "Adjudication",
    "AgreementReport",
    "Annotation",
    "KAPPA_THRESHOLD_DEFAULT",
    "PairAnnotation",
    "PairTask",
    "adjudicate",
    "adjudicate_pairwise",
    "agreement_report",
    "build_pairwise_sheet",
    "cohen_kappa",
    "diagnose_catalog",
    "diagnose_material",
    "export_samples",
    "freeze",
]
