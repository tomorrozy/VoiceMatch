"""生成一个可跑通的演示歌手库：合成音频 + HashEncoder。

⚠️ 这不是真实音色匹配结果。
``HashEncoder`` 是确定性的测试替身（见 ``voicematch/embed/encoder.py``），它的
向量没有听觉含义，只用于在没有 torch/真实素材时把整条链路跑起来。演示库的
用途只有一个：验证「小程序 → HTTP 服务 → 引擎 → JSON」这条链路是通的。

真实库必须用冻结的 HuBERT 主干 + 有版权/授权的真实清唱素材，按
``python -m voicematch.cli build --catalog <catalog.json> --out <dir>``
构建（catalog 格式见 docs/catalog.example.json）。

用法
----
    python tools/make_demo_library.py --out server/demo_library

每个歌手生成两段**不同**的合成录音，因为校准层要求"同一歌手、不同录音"的
正对（R29）；只有一段录音的库会退化成纯余弦排序。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_ROOT = Path(__file__).resolve().parent.parent
if (_ROOT / "src").is_dir() and str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from voicematch.config import SAMPLE_RATE  # noqa: E402
from voicematch.embed.encoder import HashEncoder  # noqa: E402
from voicematch.library.build import build_library  # noqa: E402
from voicematch.library.schema import Singer  # noqa: E402
from voicematch.library.store import save_library  # noqa: E402

#: 每位歌手的两个"录音"：同一副嗓子，不同音高/颤音/气声。
#: (singer_id, 姓名, 性别, 风格, 基频 Hz, 颤音率 Hz, 气声比例, 录音时长 s)
_SINGERS: tuple[tuple[str, str, str, str, float, float, float, float], ...] = (
    ("F001", "林晚晴", "female", "流行 · 中低音", 220.0, 5.4, 0.012, 24.0),
    ("F002", "苏见月", "female", "民谣 · 气声", 196.0, 4.6, 0.030, 24.0),
    ("M001", "沈屿", "male", "流行 · 中音", 130.8, 5.0, 0.014, 24.0),
    ("M002", "江照野", "male", "摇滚 · 颗粒感", 110.0, 6.0, 0.022, 24.0),
)

#: 同一歌手的第二段录音相对第一段的偏移（半音、颤音率倍率、气声倍率）。
_VARIATIONS: tuple[tuple[float, float, float], ...] = (
    (0.0, 1.0, 1.0),
    (2.0, 1.25, 0.7),
)


def synth_voice(
    f0_hz: float,
    seconds: float,
    *,
    vibrato_hz: float,
    breath: float,
    seed: int,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """合成一段"像人声"的稳定元音：谐波堆叠 + 颤音 + 气声 + 慢包络。

    这不是语音合成，只为让特征层有稳定的 f0、谐波结构和能量可供测量：
    纯静音或白噪声会让质量门限拒答，测不出任何分位。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sample_rate), dtype=np.float64) / sample_rate

    # 颤音：频率按 cents 正弦摆动（约 ±25 cents，接近人声自然颤音量级）
    cents = 25.0 * np.sin(2.0 * np.pi * vibrato_hz * t)
    instantaneous = f0_hz * np.power(2.0, cents / 1200.0)
    phase = 2.0 * np.pi * np.cumsum(instantaneous) / sample_rate

    # 谐波堆叠：基频 + 2~5 次谐波，幅度递减，制造"音色"差异
    signal = np.zeros_like(phase)
    for harmonic, amplitude in ((1, 1.0), (2, 0.45), (3, 0.22), (4, 0.12), (5, 0.06)):
        signal += amplitude * np.sin(harmonic * phase)

    # 慢速起伏，避免完全恒定的能量（窗口选取得分需要 rms 有差异）
    envelope = 0.85 + 0.1 * np.sin(2.0 * np.pi * 0.25 * t)
    signal *= envelope

    # 气声：轻微白噪声，让 HNR / 气声类 descriptor 不是同一个常数
    signal += breath * rng.standard_normal(t.size)

    # 换气停顿：真实清唱一定有静音间隙，质量门限的 SNR 也正来自"发声段能量 /
    # 静音段能量"。整段连续发声会让 5th percentile 能量等于人声能量，SNR 只有
    # 1 dB 左右，直接被 low_snr 拒答。
    signal *= _breath_mask(t.size, sample_rate)

    # 归一化到 -1..1 的安全范围，留 20% 余量
    peak = float(np.max(np.abs(signal))) or 1.0
    return (signal / (peak * 1.2)).astype(np.float32)


def _breath_mask(
    n_samples: int,
    sample_rate: int,
    *,
    head_sec: float = 0.6,
    tail_sec: float = 0.6,
    gap_every_sec: float = 4.0,
    gap_sec: float = 0.4,
    ramp_sec: float = 0.01,
) -> np.ndarray:
    """返回首尾静音 + 周期换气的软掩码（0=静音，1=发声）。

    边缘用 10 ms 汉宁核平滑，避免硬切的阶跃在频域里变成宽带噪声，把
    ``voiced_ratio`` 和谱类 descriptor 弄脏。
    """
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    duration = n_samples / sample_rate
    mask = np.ones(n_samples, dtype=np.float64)

    mask[t < head_sec] = 0.0
    mask[t > duration - tail_sec] = 0.0

    phase = (t - head_sec) % gap_every_sec
    in_gap = (
        (phase >= gap_every_sec - gap_sec)
        & (t >= head_sec)
        & (t <= duration - tail_sec)
    )
    mask[in_gap] = 0.0

    kernel_size = max(3, int(ramp_sec * sample_rate) | 1)
    kernel = np.hanning(kernel_size)
    total = kernel.sum() or 1.0
    return np.convolve(mask, kernel / total, mode="same")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成演示歌手库（合成音频 + HashEncoder）")
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "server" / "demo_library",
        help="输出目录（默认 server/demo_library）",
    )
    parser.add_argument(
        "--material-dir",
        type=Path,
        default=None,
        help="合成音频的存放目录（默认与 --out 同级的 _materials）",
    )
    args = parser.parse_args(argv)

    out = args.out.resolve()
    material_dir = (args.material_dir or out.parent / "_materials").resolve()
    material_dir.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[Singer, Path]] = []
    for index, (singer_id, name, gender, style, f0, vibrato, breath, seconds) in enumerate(_SINGERS):
        singer = Singer(
            singer_id=singer_id, name=name, gender=gender, style_tag=style
        )
        for variation, (semitones, vibrato_scale, breath_scale) in enumerate(_VARIATIONS):
            audio = synth_voice(
                f0 * (2.0 ** (semitones / 12.0)),
                seconds,
                vibrato_hz=vibrato * vibrato_scale,
                breath=breath * breath_scale,
                seed=index * 10 + variation,
            )
            path = material_dir / f"{singer_id}-take{variation + 1}.wav"
            sf.write(path, audio, SAMPLE_RATE)
            entries.append((singer, path))
            print(f"  合成 {path.name}  ({seconds:.0f}s, f0≈{f0:.0f}Hz)")

    print("构建库（HashEncoder，仅演示）…")
    library = build_library(entries, HashEncoder())
    save_library(library, out)

    print(
        f"完成：{out}\n"
        f"  歌手 {len(library.singers)} 位 / 片段 {len(library.segments)} 段\n"
        f"  model_id={library.model_id}  version={library.version}\n"
        "  ⚠️ 这是演示库：HashEncoder 的向量没有听觉含义，结果不代表真实音色匹配。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
