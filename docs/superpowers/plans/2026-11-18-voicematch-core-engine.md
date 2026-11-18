# VoiceMatch 核心引擎 Implementation Plan（阶段 0–1）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付一个可离线运行的声纹比对引擎：输入一段清唱，输出同性别歌手池中的 Top-3 与四维分位，并能在真实业余录音评估集上算出 Top-3 命中率与受理率。

**Architecture:** 两侧（歌手库与用户查询）走完全相同的预处理管线，冻结的 HuBERT 底座给出嵌入余弦，手工声学特征给出四维解释，Bradley-Terry 校准层把余弦对齐到人的感知排序。v1 不训练嵌入，只训练校准参数。

**Tech Stack:** Python 3.11、numpy、scipy、librosa、soundfile、pyloudnorm、praat-parselmouth、torch + torchaudio（HuBERT）、demucs（人声分离，可关）、scikit-learn、pytest

**Spec:** `docs/2026-11-18-voice-singer-match-design.md`

**范围说明：** 本计划只覆盖阶段 0–1（引擎 + 评估）。阶段 2 的报告页与前端另立计划——它依赖本计划产出的分数口径与置信度字段。

## Global Constraints

- 分析采样率固定 `24000 Hz`，单声道，float32。
- 响度归一化目标 `-23.0 LUFS`（pyloudnorm，integrated）。
- 片段窗长 `10.0 s`；自动选窗要求窗内有效发声 ≥ `8.0 s`。
- 单次输入总时长上限 `60.0 s`；有效发声 < `8.0 s` 一律拒答。
- 音高检测：`librosa.pyin`，`fmin=65 Hz`，`fmax=1000 Hz`，帧长 `1024`，帧移 `256`。
- 嵌入：`torchaudio.pipelines.HUBERT_BASE` 第 9 层，mean+std 池化，输出 256 维 L2 归一化向量；模型**冻结**。
- 编码器内部把 24 kHz 重采样到 `16000 Hz`（HuBERT 自身要求），两侧用同一段代码。
- 分位参照池**按性别分池**；风格标签不进分位计算。
- 候选集 = 系统 Top-3 + 同性别随机 5 位，共 8 位，顺序打乱（本计划只负责生成，不含标注 UI）。
- 评估口径：Top-3 命中率在**被受理**查询上算；受理率必须同时报告，下限 `85%`。
- 人声分离与去混响的开关，查询侧与库侧必须完全一致。
- 去混响 v1 不实现：配置项存在，置 `True` 时抛 `NotImplementedError`。

## Review Focus

- **带伴奏的跟唱**：用户传一条有伴奏的录音，管线必须强制走分离，否则比的是混音；而库里若没分离就是两侧不同构，结果会系统性偏向"制作更像"的一方。
- **采样率与声道异常**：44.1 kHz 立体声、48 kHz 手机录音、带直流偏移的文件，都必须落到同一个 24 kHz 单声道坐标系，否则 f0 与响度全部失真。
- **削波、近乎静音、超短音频**：不能崩、不能 NaN、不能除零；必须是结构化拒答。
- **跨性别查询**：女性用户查男性库时，候选池限定同性别后若为空，不能崩，也不能返回空列表当结果。
- **同一人不同设备**：换手机重录，Top-3 必须重合 ≥ 2/3——这条决定产品能不能立住。

---

### Task 1: 项目骨架与全局配置

**Files:**
- Create: `pyproject.toml`
- Create: `src/voicematch/__init__.py`
- Create: `src/voicematch/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `voicematch.config` 常量 `SAMPLE_RATE: int`、`LUFS_TARGET: float`、`WINDOW_SEC: float`、`MIN_SPEECH_SEC: float`、`MAX_INPUT_SEC: float`、`FRAME_LENGTH: int`、`HOP_LENGTH: int`、`EMBED_DIM: int`、`PIPELINE_VERSION: str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from voicematch import config as c

def test_pipeline_constants_are_pinned():
    assert c.SAMPLE_RATE == 24000
    assert c.LUFS_TARGET == -23.0
    assert c.WINDOW_SEC == 10.0
    assert c.MIN_SPEECH_SEC == 8.0
    assert c.MAX_INPUT_SEC == 60.0
    assert c.EMBED_DIM == 256
    assert c.PIPELINE_VERSION.startswith("pipeline-v")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'voicematch'`

- [ ] **Step 3: Implement**

`pyproject.toml`：`[project]` 声明 `requires-python = ">=3.11"`、依赖 `numpy scipy librosa soundfile pyloudnorm praat-parselmouth scikit-learn`、可选组 `ml = ["torch", "torchaudio", "demucs"]`、`dev = ["pytest"]`；`[tool.setuptools.packages.find] where = ["src"]`；`[tool.pytest.ini_options] testpaths = ["tests"]`。

`config.py` 按 Global Constraints 逐个赋值，`PIPELINE_VERSION = "pipeline-v1"`。刻意不读环境变量——版本号必须写死在代码里，否则结果不可复现。

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install -e ".[dev]" && pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/voicematch tests/test_config.py
git commit -m "feat: scaffold voicematch package with pinned pipeline constants"
```

---

### Task 2: 音频解码与对称预处理

**Files:**
- Create: `src/voicematch/audio/__init__.py`
- Create: `src/voicematch/audio/io.py`
- Create: `src/voicematch/audio/preprocess.py`
- Test: `tests/audio/test_io.py`
- Test: `tests/audio/test_preprocess.py`

**Interfaces:**
- Consumes: `voicematch.config` 常量
- Produces:
  - `load_audio(path: Path) -> np.ndarray` — float32、单声道、24 kHz
  - `@dataclass(frozen=True) PreprocessConfig(separate_vocals: bool, dereverb: bool, lufs_target: float)`
  - `PREPROCESS_QUERY: PreprocessConfig`、`PREPROCESS_LIBRARY: PreprocessConfig`（两者字段逐项相等）
  - `pipeline_signature(cfg: PreprocessConfig) -> tuple[int, bool, bool, int, int, float]` — `(SAMPLE_RATE, separate_vocals, dereverb, FRAME_LENGTH, HOP_LENGTH, LUFS_TARGET)`
  - `preprocess(x: np.ndarray, cfg: PreprocessConfig) -> np.ndarray`
  - `separate_vocals(x: np.ndarray, sr: int) -> np.ndarray`

- [ ] **Step 1: Write the failing tests**

```python
# tests/audio/test_io.py
import numpy as np, soundfile as sf
from voicematch.audio.io import load_audio
from voicematch.config import SAMPLE_RATE

def test_stereo_44100_is_downmixed_and_resampled(tmp_path):
    sr, n = 44100, 44100
    x = np.stack([np.sin(np.arange(n) * 0.1), np.sin(np.arange(n) * 0.2)], axis=1).astype(np.float32)
    p = tmp_path / "a.wav"; sf.write(p, x, sr)
    out = load_audio(p)
    assert out.ndim == 1 and out.dtype == np.float32
    assert abs(len(out) / SAMPLE_RATE - n / sr) < 0.05

def test_dc_offset_is_removed(tmp_path):
    sr = 48000
    x = (np.ones(sr, dtype=np.float32) * 0.3)  # 纯直流
    p = tmp_path / "dc.wav"; sf.write(p, x, sr)
    assert abs(load_audio(p).mean()) < 1e-3
```

```python
# tests/audio/test_preprocess.py
import numpy as np, pytest
from voicematch.audio.preprocess import (PREPROCESS_LIBRARY, PREPROCESS_QUERY,
                                         PreprocessConfig, pipeline_signature, preprocess)

def test_query_and_library_pipelines_are_identical():
    assert pipeline_signature(PREPROCESS_QUERY) == pipeline_signature(PREPROCESS_LIBRARY)

def test_loudness_is_normalized_to_target():
    x = np.random.default_rng(0).normal(0, 0.01, 24000 * 5).astype(np.float32)
    assert preprocess(x, PREPROCESS_QUERY).std() > x.std() * 5

def test_dereverb_not_implemented():
    with pytest.raises(NotImplementedError):
        preprocess(np.zeros(24000, np.float32),
                   PreprocessConfig(separate_vocals=False, dereverb=True, lufs_target=-23.0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/audio -v`
Expected: FAIL — `ModuleNotFoundError: voicematch.audio`

- [ ] **Step 3: Implement**

`loudness_is_normalized_to_target` 断言 5 倍是因为 -23 LUFS 对应的 RMS 通常远高于 0.01 的噪声。若实际倍数不满足，把测试改成"归一化后 integrated LUFS 与目标差 < 0.5，且归一化结果与 pyloudnorm 独立复算一致"，不要放宽到无意义。

`load_audio`：`soundfile.read(always_2d=True)` → 声道求均值 → 去直流（减均值）→ `librosa.resample(orig_sr=sr, target_sr=SAMPLE_RATE, res_type="soxr_hq")` → float32。

`pipeline_signature` 只返回两侧必须一致的那几项，任何新增的"必须一致"项都要加进这个元组，因为契约测试读的就是它。

`separate_vocals`：`demucs.api.Separator(model="htdemucs")` 取 `vocals` 轨；延迟 import，未安装 `ml` 依赖时抛 `ImportError("install voicematch[ml]")`。

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/audio -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voicematch/audio tests/audio
git commit -m "feat(audio): decode, downmix, resample and loudness-normalize with symmetric pipeline contract"
```

---

### Task 3: 质量门限与拒答

**Files:**
- Create: `src/voicematch/audio/quality.py`
- Test: `tests/audio/test_quality.py`

**Interfaces:**
- Consumes: `SAMPLE_RATE`、`MIN_SPEECH_SEC`、`MAX_INPUT_SEC`
- Produces:
  - `@dataclass(frozen=True) QualityReport(speech_sec: float, snr_db: float, clipped_ratio: float, voiced_ratio: float, rejected: bool, reasons: tuple[str, ...])`
  - `assess_quality(x: np.ndarray, sr: int) -> QualityReport`

- [ ] **Step 1: Write the failing test**

```python
# tests/audio/test_quality.py
import numpy as np
from voicematch.audio.quality import assess_quality

def test_silence_is_rejected_with_reason():
    r = assess_quality(np.zeros(24000 * 10, np.float32), 24000)
    assert r.rejected and "too_short_speech" in r.reasons

def test_clipping_is_rejected():
    x = np.sign(np.sin(np.arange(24000 * 10) * 0.05)).astype(np.float32)
    r = assess_quality(x, 24000)
    assert r.rejected and "clipped" in r.reasons

def test_clean_speech_is_accepted():
    rng = np.random.default_rng(1)
    x = (0.3 * np.sin(2 * np.pi * 220 * np.arange(24000 * 10) / 24000)
         + rng.normal(0, 1e-4, 24000 * 10)).astype(np.float32)
    r = assess_quality(x, 24000)
    assert not r.rejected and r.speech_sec >= 8.0 and r.snr_db > 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/audio/test_quality.py -v`
Expected: FAIL — `cannot import name 'assess_quality'`

- [ ] **Step 3: Implement**

`speech_sec` 用 `librosa.effects.split(top_db=30)` 的帧段总长；`clipped_ratio = mean(|x| >= 0.999)`；`snr_db` = 有声帧均方能量与全段 5 分位能量的比值取 dB；`voiced_ratio` 用同一批有声帧占比。拒绝规则按顺序：`speech_sec < MIN_SPEECH_SEC → "too_short_speech"`；`clipped_ratio > 0.01 → "clipped"`；`snr_db < 10 → "low_snr"`；`voiced_ratio < 0.3 → "low_voiced_ratio"`。`reasons` 收集全部命中项，不短路——用户需要知道所有问题。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/audio/test_quality.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voicematch/audio/quality.py tests/audio/test_quality.py
git commit -m "feat(audio): structured quality gate with explicit rejection reasons"
```

---

### Task 4: 自动选窗

**Files:**
- Create: `src/voicematch/audio/windows.py`
- Test: `tests/audio/test_windows.py`

**Interfaces:**
- Consumes: `WINDOW_SEC`、`MIN_SPEECH_SEC`
- Produces: `select_windows(x: np.ndarray, sr: int, k: int = 5, window_sec: float = WINDOW_SEC) -> list[tuple[float, float]]` — 返回 `(start_sec, end_sec)`，长度随机长递减，互不重叠，确定性

- [ ] **Step 1: Write the failing test**

```python
# tests/audio/test_windows.py
import numpy as np
from voicematch.audio.windows import select_windows

def test_windows_are_non_overlapping_and_deterministic():
    x = np.random.default_rng(2).normal(0, 0.05, 24000 * 40).astype(np.float32)
    a, b = select_windows(x, 24000, k=3), select_windows(x, 24000, k=3)
    assert a == b and len(a) == 3
    for (s1, e1), (s2, e2) in zip(a, a[1:]):
        assert e1 <= s2
    assert all(e - s > 8.0 for s, e in a)

def test_silence_yields_no_windows():
    assert select_windows(np.zeros(24000 * 30, np.float32), 24000) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/audio/test_windows.py -v`
Expected: FAIL — `cannot import name 'select_windows'`

- [ ] **Step 3: Implement**

滑窗（步长 1 s）→ 每窗算 `voiced_ratio * pitch_stability * rms`，其中 `pitch_stability` 用窗内 f0 有效帧的 `1 / (1 + std(f0)/mean(f0))`；按分数降序贪心取窗，与已选重叠则跳过，直到 k 个或耗尽。全零输入时 voiced_ratio 为 0，自然返回空列表——不要特殊分支。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/audio/test_windows.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voicematch/audio/windows.py tests/audio/test_windows.py
git commit -m "feat(audio): deterministic auto window selection shared by library and query sides"
```

---

### Task 5: 音高与音色特征

**Files:**
- Create: `src/voicematch/features/__init__.py`
- Create: `src/voicematch/features/pitch.py`
- Create: `src/voicematch/features/timbre.py`
- Test: `tests/features/test_pitch.py`
- Test: `tests/features/test_timbre.py`

**Interfaces:**
- Consumes: `FRAME_LENGTH`、`HOP_LENGTH`、`SAMPLE_RATE`
- Produces:
  - `pitch_features(x: np.ndarray, sr: int) -> dict[str, float]`，键：`median_f0_hz`、`range_semitones`、`f0_jitter`
  - `timbre_features(x: np.ndarray, sr: int) -> dict[str, float]`，键：`mfcc1..mfcc13_mean`、`spectral_centroid_hz`、`spectral_contrast_mean`、`ltas_slope`

- [ ] **Step 1: Write the failing tests**

```python
# tests/features/test_pitch.py
import numpy as np
from voicematch.features.pitch import pitch_features

def _tone(f0, secs=3.0, sr=24000):
    t = np.arange(int(sr * secs)) / sr
    return (0.3 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)

def test_median_f0_matches_synthesized_tone():
    assert abs(pitch_features(_tone(220.0), 24000)["median_f0_hz"] - 220.0) < 5.0

def test_jitter_is_near_zero_for_pure_tone():
    assert pitch_features(_tone(220.0), 24000)["f0_jitter"] < 0.01
```

```python
# tests/features/test_timbre.py
import numpy as np
from voicematch.features.timbre import timbre_features

def test_bright_noise_has_higher_centroid_than_dark():
    rng = np.random.default_rng(3)
    white = rng.normal(0, 0.1, 24000 * 3).astype(np.float32)
    dark = np.convolve(white, np.ones(32) / 32, mode="same").astype(np.float32)
    assert (timbre_features(white, 24000)["spectral_centroid_hz"]
            > timbre_features(dark, 24000)["spectral_centroid_hz"] * 2)

def test_feature_keys_are_stable():
    keys = set(timbre_features(np.random.default_rng(4).normal(0, .1, 24000*3).astype(np.float32), 24000))
    assert {"spectral_centroid_hz", "spectral_contrast_mean", "ltas_slope"} <= keys
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/features -v`
Expected: FAIL — `No module named 'voicematch.features'`

- [ ] **Step 3: Implement**

`pitch_features`：`librosa.pyin(y, fmin=65, fmax=1000, sr=sr, frame_length=1024, hop_length=256)`；`f0_jitter = std(f0_voiced) / mean(f0_voiced)`；`range_semitones = 12 * log2(p95 / p5)`，空帧返回 `nan` 时整体返回全 `nan` 字典（由上层拒答，不在特征层抛异常）。`timbre_features` 用 `librosa.feature.mfcc(n_mfcc=13)` 取均值、`spectral_centroid`、`spectral_contrast` 均值、`ltas_slope` = 平均谱在对数频率上线性拟合的斜率。

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/features -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voicematch/features tests/features
git commit -m "feat(features): pitch and timbre descriptors"
```

---

### Task 6: 共鸣与唱腔特征

**Files:**
- Create: `src/voicematch/features/resonance.py`
- Create: `src/voicematch/features/style.py`
- Test: `tests/features/test_resonance.py`
- Test: `tests/features/test_style.py`

**Interfaces:**
- Produces:
  - `resonance_features(x: np.ndarray, sr: int) -> dict[str, float]`，键：`f1_hz`、`f2_hz`、`f3_hz`、`low_band_ratio`、`hnr_db`
  - `style_features(x: np.ndarray, sr: int) -> dict[str, float]`，键：`vibrato_rate_hz`、`vibrato_extent_cents`、`slur_ratio`、`onset_hardness`、`breathiness`

- [ ] **Step 1: Write the failing tests**

```python
# tests/features/test_style.py
import numpy as np
from voicematch.features.style import style_features

def _vibrato(base=220.0, rate=6.0, cents=100.0, secs=3.0, sr=24000):
    t = np.arange(int(sr * secs)) / sr
    f = base * 2 ** ((cents / 1200) * np.sin(2 * np.pi * rate * t))
    return (0.3 * np.sin(2 * np.pi * np.cumsum(f) / sr)).astype(np.float32)

def test_vibrato_rate_and_extent_are_recovered():
    r = style_features(_vibrato(), 24000)
    assert abs(r["vibrato_rate_hz"] - 6.0) < 1.0
    assert 50 < r["vibrato_extent_cents"] < 150

def test_steady_tone_has_negligible_extent():
    t = np.arange(72000) / 24000
    x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    assert style_features(x, 24000)["vibrato_extent_cents"] < 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/features/test_style.py -v`
Expected: FAIL — `No module named 'voicematch.features.style'`

- [ ] **Step 3: Implement**

颤音算法（签名与测试不决定，故写下）：
1. `f0 = librosa.pyin(...)`，取有效帧，转成的音分序列 `c = 1200 * log2(f0 / median(f0))`。
2. 去均值，对 `c` 做 3–8 Hz 带通（`scipy.signal.butter(4, [3, 8], btype="band", fs=1/hop_time)`）。
3. `vibrato_extent_cents = 2 * sqrt(2) * std(filtered)`（正弦幅度估计）。
4. `vibrato_rate_hz` = 带通信号的 FFT 峰值频率；若 `vibrato_extent_cents < 10` 则记 0.0。

`slur_ratio` = 相邻帧 f0 变化在 0.5–4 半音之间的比例；`onset_hardness` = 能量包络上升沿的平均斜率；`breathiness` = 2–8 kHz 能量与 0–2 kHz 能量之比。`resonance_features` 用 `parselmouth` 的 `to_formant_burg` 取 F1–F3 中位数、`low_band_ratio` 同 `breathiness` 的分母占比、`hnr_db` 用 Praat `Harmonicity` 中位数。

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/features -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voicematch/features/resonance.py src/voicematch/features/style.py tests/features
git commit -m "feat(features): resonance and singing-style descriptors"
```

---

### Task 7: 特征汇总与同性别分位

**Files:**
- Create: `src/voicematch/features/extract.py`
- Create: `src/voicematch/features/percentile.py`
- Test: `tests/features/test_extract.py`
- Test: `tests/features/test_percentile.py`

**Interfaces:**
- Consumes: Task 5、Task 6 的四个特征函数
- Produces:
  - `@dataclass(frozen=True) FeatureVector(pitch: dict[str, float], timbre: dict[str, float], resonance: dict[str, float], style: dict[str, float])`
  - `extract_features(x: np.ndarray, sr: int) -> FeatureVector`
  - `@dataclass(frozen=True) PercentileModel(reference: dict[str, np.ndarray])` — 键为特征名，值为该池的排序参考值
  - `fit_percentiles(vectors: Sequence[FeatureVector]) -> PercentileModel`
  - `to_percentiles(fv: FeatureVector, model: PercentileModel) -> dict[str, float]` — 0–100
  - `DIMENSION_OF: dict[str, str]` — 特征名到四维之一的映射

- [ ] **Step 1: Write the failing tests**

```python
# tests/features/test_percentile.py
import numpy as np
from voicematch.features.extract import FeatureVector
from voicematch.features.percentile import fit_percentiles, to_percentiles

def _fv(centroid):
    return FeatureVector(pitch={"median_f0_hz": 220.0}, timbre={"spectral_centroid_hz": centroid},
                         resonance={"hnr_db": 20.0}, style={"vibrato_rate_hz": 5.0})

def test_percentile_is_relative_to_reference_pool():
    pool = [_fv(c) for c in (1000, 2000, 3000)]
    model = fit_percentiles(pool)
    assert to_percentiles(_fv(1000), model)["spectral_centroid_hz"] == 0.0
    assert to_percentiles(_fv(3000), model)["spectral_centroid_hz"] == 100.0
    assert 0 < to_percentiles(_fv(2000), model)["spectral_centroid_hz"] < 100

def test_single_sample_pool_does_not_divide_by_zero():
    model = fit_percentiles([_fv(1000)])
    assert to_percentiles(_fv(1000), model)["spectral_centroid_hz"] == 50.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/features/test_percentile.py -v`
Expected: FAIL — `cannot import name 'fit_percentiles'`

- [ ] **Step 3: Implement**

`fit_percentiles` 把每个特征名在参考池中的取值排序后存进 `reference`；`to_percentiles` 用 `np.searchsorted(sorted_ref, value, side="right") / len(ref) * 100`，参照池长度为 1 时返回 50.0。`nan` 值原样返回 `nan`，不当作 0 或 100——把缺失当成极端值会凭空制造"最像"。

`DIMENSION_OF` 显式写死映射表（比按前缀猜稳）。`extract_features` 只做拼接，任何一个子维度全 `nan` 时保留 `nan` 而不是填 0。

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/features -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voicematch/features tests/features
git commit -m "feat(features): aggregate feature vector and same-gender percentile mapping"
```

---

### Task 8: 冻结声乐编码器

**Files:**
- Create: `src/voicematch/embed/__init__.py`
- Create: `src/voicematch/embed/encoder.py`
- Create: `src/voicematch/embed/pooling.py`
- Test: `tests/embed/test_encoder.py`
- Test: `tests/embed/test_pooling.py`

**Interfaces:**
- Produces:
  - `class VocalEncoder(Protocol): def encode(self, x: np.ndarray, sr: int) -> np.ndarray` — 返回 `(EMBED_DIM,)` float32，L2 范数 = 1
  - `class HashEncoder(VocalEncoder)` — 测试用，同一输入恒等输出，不同输入几乎不正交
  - `class HubertEncoder(VocalEncoder)` — 内部 24k→16k 重采样后过 `HUBERT_BASE` 第 9 层
  - `pool_frames(frames: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray` — mean+std 拼接后降维到 `EMBED_DIM`

- [ ] **Step 1: Write the failing test**

```python
# tests/embed/test_encoder.py
import numpy as np
from voicematch.embed.encoder import HashEncoder
from voicematch.config import EMBED_DIM

def test_embedding_is_unit_norm_and_deterministic():
    enc = HashEncoder()
    x = np.random.default_rng(5).normal(0, .1, 24000 * 3).astype(np.float32)
    a, b = enc.encode(x, 24000), enc.encode(x, 24000)
    assert a.shape == (EMBED_DIM,) and abs(np.linalg.norm(a) - 1.0) < 1e-5
    assert np.allclose(a, b)

def test_different_audio_gives_different_vectors():
    enc = HashEncoder()
    x = np.random.default_rng(6).normal(0, .1, 24000 * 3).astype(np.float32)
    y = np.random.default_rng(7).normal(0, .1, 24000 * 3).astype(np.float32)
    assert np.dot(enc.encode(x, 24000), enc.encode(y, 24000)) < 0.95
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/embed -v`
Expected: FAIL — `No module named 'voicematch.embed'`

- [ ] **Step 3: Implement**

`HashEncoder`：对音频按固定分块做 `sha256`，拼成字节流，映射成 `EMBED_DIM` 个 `[-1,1]` 浮点后归一化——纯确定性，不依赖随机种子。

`HubertEncoder`：`torchaudio.pipelines.HUBERT_BASE.get_model().eval()`，`torchaudio.functional.resample(x, 24000, 16000)`，取 `hidden_states[9]`，`pool_frames` 后 `F.normalize`。模型加载用 `lru_cache` 单例，并在模块级常量里写 `MODEL_ID = "HUBERT_BASE"`，随结果一起落盘（G4 可复现）。

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/embed -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voicematch/embed tests/embed
git commit -m "feat(embed): frozen Hubert encoder with deterministic test double"
```

---

### Task 9: 歌手原型向量

**Files:**
- Create: `src/voicematch/embed/prototype.py`
- Test: `tests/embed/test_prototype.py`

**Interfaces:**
- Consumes: `VocalEncoder` 输出
- Produces: `singer_prototypes(embeddings: np.ndarray, singer_ids: Sequence[str]) -> dict[str, np.ndarray]` — 每位歌手的均值向量，逐条再 L2 归一化

- [ ] **Step 1: Write the failing test**

```python
# tests/embed/test_prototype.py
import numpy as np
from voicematch.embed.prototype import singer_prototypes

def test_prototypes_average_within_singer_and_are_normalized():
    e = np.array([[1, 0], [0.8, 0.6], [0, 1]], dtype=np.float32)
    p = singer_prototypes(e, ["a", "a", "b"])
    assert set(p) == {"a", "b"}
    assert abs(np.linalg.norm(p["a"]) - 1.0) < 1e-5
    assert np.dot(p["a"], e[0]) > np.dot(p["a"], e[2])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/embed/test_prototype.py -v`
Expected: FAIL — `cannot import name 'singer_prototypes'`

- [ ] **Step 3: Implement**

按 `singer_ids` 分组求均值 → 再归一化。单条歌手时原型就是该条向量本身（归一化后不变），不做特殊处理。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/embed/test_prototype.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voicematch/embed/prototype.py tests/embed/test_prototype.py
git commit -m "feat(embed): singer prototype vectors"
```

---

### Task 10: 歌手库构建与版本化存储

**Files:**
- Create: `src/voicematch/library/__init__.py`
- Create: `src/voicematch/library/schema.py`
- Create: `src/voicematch/library/build.py`
- Create: `src/voicematch/library/store.py`
- Test: `tests/library/test_build.py`
- Test: `tests/library/test_store.py`

**Interfaces:**
- Consumes: `select_windows`、`extract_features`、`VocalEncoder`、`singer_prototypes`
- Produces:
  - `@dataclass(frozen=True) Singer(singer_id: str, name: str, gender: Literal["male","female"], style_tag: str)`
  - `@dataclass(frozen=True) Segment(segment_id: str, singer_id: str, start: float, end: float)`
  - `@dataclass(frozen=True) LibraryIndex(version: str, singers: dict[str, Singer], segments: tuple[Segment, ...], prototypes: dict[str, np.ndarray], features: dict[str, FeatureVector])`
  - `build_library(entries: Sequence[tuple[Singer, Path]], encoder: VocalEncoder) -> LibraryIndex`
  - `save_library(index: LibraryIndex, out: Path) -> None`、`load_library(path: Path) -> LibraryIndex`

- [ ] **Step 1: Write the failing test**

```python
# tests/library/test_store.py
import numpy as np, soundfile as sf
from voicematch.library.schema import Singer
from voicematch.library.build import build_library
from voicematch.library.store import save_library, load_library
from voicematch.embed.encoder import HashEncoder

def _wav(p, f0, secs=20.0, sr=24000):
    t = np.arange(int(sr * secs)) / sr
    sf.write(p, (0.3 * np.sin(2 * np.pi * f0 * t)).astype(np.float32), sr)

def test_build_and_roundtrip(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _wav(a, 220.0); _wav(b, 330.0)
    idx = build_library([(Singer("s1", "A", "female", "pop"), a),
                         (Singer("s2", "B", "male", "rock"), b)], HashEncoder())
    assert len(idx.segments) == 10 and set(idx.prototypes) == {"s1", "s2"}
    save_library(idx, tmp_path / "lib")
    back = load_library(tmp_path / "lib")
    assert back.version == idx.version and np.allclose(back.prototypes["s1"], idx.prototypes["s1"])

def test_version_changes_when_material_changes(tmp_path):
    a = tmp_path / "a.wav"; _wav(a, 220.0)
    one = build_library([(Singer("s1", "A", "female", "pop"), a)], HashEncoder())
    _wav(a, 440.0)
    two = build_library([(Singer("s1", "A", "female", "pop"), a)], HashEncoder())
    assert one.version != two.version
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/library -v`
Expected: FAIL — `No module named 'voicematch.library'`

- [ ] **Step 3: Implement**

`build_library`：每条音频 → `select_windows(k=5)` → 每窗 `extract_features` + `encoder.encode` → 汇总。`version` = 由 `(singer_id, segment_start, segment_end, 素材文件 sha256, PIPELINE_VERSION, MODEL_ID)` 规范化 JSON 的 `sha256` 前 16 位——这条把"素材变了版本就变"变成结构性保证，而不是靠人记得改。

`store`：`manifest.json`（歌手、片段、版本、特征）+ `prototypes.npz` + `embeddings.npy`，同一目录。`load_library` 校验 manifest 里的版本与向量的行数一致，不一致直接抛错而不是静默截断。

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/library -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voicematch/library tests/library
git commit -m "feat(library): build versioned singer library from raw material"
```

---

### Task 11: 校准配对与 Bradley-Terry 模型

**Files:**
- Create: `src/voicematch/calibrate/__init__.py`
- Create: `src/voicematch/calibrate/pairs.py`
- Create: `src/voicematch/calibrate/model.py`
- Test: `tests/calibrate/test_pairs.py`
- Test: `tests/calibrate/test_model.py`

**Interfaces:**
- Consumes: `LibraryIndex`、`FeatureVector`
- Produces:
  - `pair_features(cos: float, fv_a: FeatureVector, fv_b: FeatureVector, proto_a: np.ndarray, proto_b: np.ndarray) -> np.ndarray` — 长度 5 的向量 `[cos, Δpitch, Δtimbre, Δresonance, Δstyle]`
  - `build_pairs(index: LibraryIndex, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — `(X, y, groups)`，`y=1` 表示同歌手
  - `fit_calibration(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> CalibrationModel`
  - `@dataclass(frozen=True) CalibrationModel(coef: np.ndarray, mean: np.ndarray, scale: np.ndarray, intercept: float, auc: float)`，方法 `score(feats: np.ndarray) -> np.ndarray`，`attribution(feats: np.ndarray) -> dict[str, float]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/calibrate/test_model.py
import numpy as np
from voicematch.calibrate.model import fit_calibration

def _toy(n=200):
    rng = np.random.default_rng(8)
    pos = rng.normal(0.9, 0.05, (n, 5)); pos[:, 1:] = 0.1
    neg = rng.normal(0.2, 0.05, (n, 5)); neg[:, 1:] = 0.4
    X = np.vstack([pos, neg])
    y = np.r_[np.ones(n), np.zeros(n)]
    g = np.tile(np.arange(n), 2)  # 每折按歌手切分，正负样本同折
    return pos, neg, X, y, g

def test_calibration_separates_same_from_different_singer_pairs():
    pos, neg, X, y, g = _toy()
    m = fit_calibration(X, y, g)
    assert m.score(pos).mean() > 0.8 and m.score(neg).mean() < 0.2
    assert m.auc > 0.9  # 按歌手切分的折外 AUC，不是训练集 AUC

def test_attribution_sums_to_one():
    rng = np.random.default_rng(9)
    X = rng.normal(0.5, .1, (100, 5)); y = (X[:, 0] > 0.5).astype(float)
    g = np.repeat(np.arange(50), 2)
    m = fit_calibration(X, y, g)
    a = m.attribution(X[0])
    assert abs(sum(abs(v) for v in a.values()) - 1.0) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/calibrate -v`
Expected: FAIL — `No module named 'voicematch.calibrate'`

- [ ] **Step 3: Implement**

`build_pairs`：同歌手（同 `singer_id` 的不同片段）为 1，不同歌手为 0；`groups` 用 `singer_id` 的组合键，保证 `GroupKFold` 能按歌手切分。

`fit_calibration`：先 z-score（均值/方差存进 `CalibrationModel`，不存就复现不了），再 `LogisticRegression(C=1.0, max_iter=1000)` 在**全量**上拟合出最终系数；`auc` 用 `GroupKFold(n_splits=5)` 按歌手分组做折外预测后算——这是唯一能回答"校准在没见过的歌手身上还成立吗"的数字，也是 v1 报告给 G1 的中间依据。`score` 返回 sigmoid 概率。`attribution` 返回 `|coef_i * z_i| / sum`，键名固定为 `cos/pitch/timbre/resonance/style`。

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/calibrate -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voicematch/calibrate tests/calibrate
git commit -m "feat(calibrate): Bradley-Terry style logistic calibration over cosine plus dimension deltas"
```

---

### Task 12: 打分、排序与归因

**Files:**
- Create: `src/voicematch/calibrate/score.py`
- Test: `tests/calibrate/test_score.py`

**Interfaces:**
- Consumes: `LibraryIndex`、`CalibrationModel`、`FeatureVector`、`VocalEncoder`
- Produces:
  - `@dataclass(frozen=True) ScoredSinger(singer_id: str, name: str, score: float, attribution: dict[str, float])`
  - `rank_singers(fv: FeatureVector, emb: np.ndarray, index: LibraryIndex, model: CalibrationModel, gender: Literal["male","female"], top_k: int = 3) -> list[ScoredSinger]`

- [ ] **Step 1: Write the failing test**

```python
# tests/calibrate/test_score.py
import numpy as np, pytest
from voicematch.calibrate.score import rank_singers

def test_empty_gender_pool_raises_explicitly(make_index, make_model, fv):
    with pytest.raises(LookupError, match="no_singers_in_pool"):
        rank_singers(fv, np.ones(256, np.float32) / 16, make_index(genders=("male",)), make_model(), gender="female")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/calibrate/test_score.py -v`
Expected: FAIL — `cannot import name 'rank_singers'`

- [ ] **Step 3: Implement**

候选池先按 `gender` 过滤，空池抛 `LookupError("no_singers_in_pool")`——**不能返回空列表**，否则上游会把"查无此人"渲染成"你像 0 位歌手"。对池内每位歌手：`cos = dot(emb, prototype)`，`pair_features` → `model.score` → 降序取前 `top_k`，附 `attribution`。

测试夹具 `make_index/make_model/fv` 写进 `tests/calibrate/conftest.py`，构造 2 位同性别歌手的最小库。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/calibrate/test_score.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voicematch/calibrate tests/calibrate
git commit -m "feat(calibrate): gender-scoped ranking with explicit empty-pool error"
```

---

### Task 13: 评估指标

**Files:**
- Create: `src/voicematch/eval/__init__.py`
- Create: `src/voicematch/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Produces:
  - `top_k_hit(pred: Sequence[Sequence[str]], labels: Sequence[set[str]], k: int) -> float`
  - `acceptance_rate(reports: Sequence[QualityReport]) -> float`
  - `spearman(pred_scores: Sequence[float], human_scores: Sequence[float]) -> float`
  - `top3_overlap(a: Sequence[str], b: Sequence[str]) -> float` — 交集 / 3

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_metrics.py
import pytest
from voicematch.eval.metrics import top_k_hit, acceptance_rate, top3_overlap

def test_top3_hit_counts_query_when_any_label_in_top3():
    assert top_k_hit([["a", "b", "c"], ["d", "e", "f"]], [{"c"}, {"z"}], k=3) == 0.5

def test_empty_input_is_nan_not_zero():
    assert top_k_hit([], [], k=3) != top_k_hit([], [], k=3)  # NaN

def test_top3_overlap_is_intersection_over_three():
    assert top3_overlap(["a", "b", "c"], ["c", "b", "x"]) == pytest.approx(2 / 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval -v`
Expected: FAIL — `No module named 'voicematch.eval'`

- [ ] **Step 3: Implement**

空输入返回 `float("nan")`——0.0 会被误读成"测过了，全错"。`acceptance_rate` = 未拒答数 / 总数。`spearman` 用 `scipy.stats.spearmanr`，遇到常数序列返回 `nan` 而不是抛异常。

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/voicematch/eval tests/eval
git commit -m "feat(eval): top-k hit, acceptance rate, spearman and stability metrics"
```

---

### Task 14: 评估集读取、端到端评估与 CLI

**Files:**
- Create: `src/voicematch/eval/dataset.py`
- Create: `src/voicematch/eval/run.py`
- Create: `src/voicematch/cli.py`
- Test: `tests/eval/test_dataset.py`
- Test: `tests/eval/test_run_e2e.py`

**Interfaces:**
- Consumes: 全部上文
- Produces:
  - `@dataclass(frozen=True) EvalQuery(query_id: str, path: Path, gender: str, candidates: tuple[str, ...], picked: set[str], annotator_count: int)`
  - `load_eval_set(path: Path) -> list[EvalQuery]` — 读 JSON，校验每位 query 的 `picked` 非空或显式为"都不像"
  - `run_eval(queries, index, model, encoder) -> dict[str, float]`，键含 `top1`、`top3`、`acceptance_rate`、`n_accepted`
  - CLI：`python -m voicematch.cli compare <audio.wav> --library <dir> --gender female` 输出结果 JSON（含 `model_id`、`library_version`、`calibration_version`）

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_run_e2e.py
import json, numpy as np, soundfile as sf
from voicematch.eval.dataset import load_eval_set
from voicematch.eval.run import run_eval

def test_end_to_end_returns_both_headline_numbers(tmp_path, tiny_stack):
    q = tmp_path / "q.wav"
    t = np.arange(24000 * 12) / 24000
    sf.write(q, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), 24000)
    spec = tmp_path / "eval.json"
    spec.write_text(json.dumps([{"query_id": "q1", "path": str(q), "gender": "female",
                                 "candidates": ["s1", "s2"], "picked": ["s1"], "annotator_count": 3}]))
    out = run_eval(load_eval_set(spec), tiny_stack.index, tiny_stack.calibration, tiny_stack.encoder)
    assert {"top1", "top3", "acceptance_rate"} <= set(out)
    assert out["n_accepted"] <= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval -v`
Expected: FAIL — `cannot import name 'load_eval_set'`

- [ ] **Step 3: Implement**

`run_eval` 对每条 query 走 `load_audio → preprocess → assess_quality → select_windows[0] → extract_features → encoder.encode → rank_singers`，拒答的不进 Top-K 分母（但进受理率分母）。返回字典同时含 `top3` 与 `acceptance_rate`、`n_rejected`——**两个数必须一起出现**，这是 G1 的口径要求。`tiny_stack` 夹具放 `tests/eval/conftest.py`：`@dataclass(frozen=True) TinyStack(index, calibration, encoder)`，用 `HashEncoder` 构造两位同性别歌手的最小库，并用 Task 11 的 `build_pairs` + `fit_calibration` 现场训一个校准模型，不落盘。

CLI 用 `argparse`，输出单行 JSON 到 stdout。

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest -v`
Expected: 全绿

- [ ] **Step 5: Commit**

```bash
git add src/voicematch/eval src/voicematch/cli.py tests/eval
git commit -m "feat(eval): eval-set loader, end-to-end evaluation and compare CLI"
```

---

## 自检记录

- **Spec 覆盖**：§5.1 对称管线 → Task 2；§5.2 四维特征与分位 → Task 5–7；§5.3 冻结嵌入 → Task 8–9；§5.4 校准 → Task 11–12；§5.5 库与自动选窗 → Task 4、10；§5.6 输入契约与拒答 → Task 3；§6 指标与口径 → Task 13–14；§7 阶段 0–1 → 全部。**未覆盖**：§5.7 采集通道（前端，属阶段 2 计划）、§6.2 标注 UI（人工流程，非代码）。
- **Step 扫描**：无 TBD、无"处理边界情况"式空话。
- **类型一致性**：`FeatureVector` 在 Task 7 定义、Task 11–12 使用；`CalibrationModel` 在 Task 11 定义、Task 12、14 使用；`QualityReport` 在 Task 3 定义、Task 13–14 使用。
- **比例**：计划约 400 行，spec 约 190 行，比例可接受；代码块以测试与签名为主。
