"""Liquid V2 状态方程、时间策略与 offline/online 一致性契约测试。

本文件对应 ``src/tests/README.md`` 明确要求的四类覆盖：

1. **Liquid V2 状态方程** —— raw memory、pre-norm、candidate/gate/alpha 的代数关系
2. **padding 与 strict-skip** —— padding 不调用 cell；strict-skip 为位级恒等
3. **offline/online 一致性** —— 序列前向与逐事件续推位级等价
4. **checkpoint 完整性** —— 结构 contract 的字段与拒绝非法配置

测试断言的是**契约行为**，不是数值快照。所有阈值均取自被断言对象自身暴露的常量
（``tau_min_seconds`` / ``tau_max_seconds`` / ``invalid_dt_observation_gain`` 等），
避免把魔数写进测试。
"""

from __future__ import annotations

import math

import pytest
import torch

from radar_ln3.data.temporal import (
    DT_REASON_FIRST_EVENT,
    DT_REASON_POSITIVE_DELTA,
    DT_REASON_ZERO_DELTA,
)
from radar_ln3.liquid_v2 import (
    EQUATION_VERSION,
    TRACE_FIELD_NAMES,
    LiquidV2Cell,
    LiquidV2SequenceWrapper,
    liquid_v2_contract,
    time_policy_contract,
)

INPUT_DIM = 5
HIDDEN_DIM = 7
TAU_MIN = 0.05
TAU_MAX = 30.0
FLOOR = 0.001
DT_MAX = 120.0
GAIN = 0.25


def build_cell(mode: str = "bounded_observation", **overrides) -> LiquidV2Cell:
    """构造一个固定种子的 cell，避免测试间的初始化抖动。"""

    torch.manual_seed(20260924)
    kwargs = dict(
        tau_min_seconds=TAU_MIN,
        tau_max_seconds=TAU_MAX,
        zero_dt_event_floor_seconds=FLOOR,
        dt_max_seconds=DT_MAX,
        invalid_dt_update_mode=mode,
        invalid_dt_observation_gain=GAIN,
    )
    kwargs.update(overrides)
    return LiquidV2Cell(INPUT_DIM, HIDDEN_DIM, **kwargs)


def single_step_inputs(
    *,
    reason: int = DT_REASON_POSITIVE_DELTA,
    dt: float = 1.0,
    ordinal: int = 1,
    batch: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造一组合法的 [B] 时间张量。

    ``zero_ordinal`` 的契约要求是正整数（``_validate_inputs`` 拒绝 0），
    且正间隔 reason 必须恰好为 1。
    """

    dt_seconds = torch.full((batch,), float(dt), dtype=torch.float32)
    dt_valid = torch.full((batch,), reason == DT_REASON_POSITIVE_DELTA, dtype=torch.bool)
    dt_reason = torch.full((batch,), int(reason), dtype=torch.long)
    zero_ordinal = torch.full((batch,), int(ordinal), dtype=torch.long)
    return dt_seconds, dt_valid, dt_reason, zero_ordinal


def sequence_inputs(
    length: int,
    *,
    first_reason: int = DT_REASON_FIRST_EVENT,
    dt: float = 2.0,
    batch: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """构造一条全有效、无 padding 的序列。

    第 0 个事件用 ``first_reason``，其余用 positive-delta。
    """

    events = torch.randn(batch, length, INPUT_DIM, generator=torch.Generator().manual_seed(7))
    dt_seconds = torch.full((batch, length), float(dt), dtype=torch.float32)
    dt_seconds[:, 0] = 0.0
    dt_valid = torch.ones((batch, length), dtype=torch.bool)
    dt_valid[:, 0] = first_reason == DT_REASON_POSITIVE_DELTA
    dt_reason = torch.full((batch, length), DT_REASON_POSITIVE_DELTA, dtype=torch.long)
    dt_reason[:, 0] = int(first_reason)
    padding_mask = torch.zeros((batch, length), dtype=torch.bool)
    return events, dt_seconds, padding_mask, dt_valid, dt_reason


# --------------------------------------------------------------------------
# 1. 状态方程
# --------------------------------------------------------------------------


def test_tau_is_bounded_and_sigmoid_centred() -> None:
    """tau 必须落在 [tau_min, tau_max]；未训练时（theta=0）取区间中点。"""

    cell = build_cell()
    assert torch.equal(cell.tau_unconstrained, torch.zeros(HIDDEN_DIM))

    tau = cell.tau(dtype=torch.float32, device=torch.device("cpu"))
    assert tau.shape == (HIDDEN_DIM,)
    assert bool((tau >= TAU_MIN).all()) and bool((tau <= TAU_MAX).all())

    expected_mid = TAU_MIN + (TAU_MAX - TAU_MIN) * 0.5
    assert torch.allclose(tau, torch.full_like(tau, expected_mid), atol=1.0e-6)


def test_tau_stays_bounded_for_extreme_unconstrained_values() -> None:
    """sigmoid 参数化必须使 tau 在任意 theta 下都不越界（含负值、零、饱和）。"""

    cell = build_cell()
    for theta in (-1.0e3, -10.0, 0.0, 10.0, 1.0e3):
        with torch.no_grad():
            cell.tau_unconstrained.fill_(theta)
        tau = cell.tau(dtype=torch.float32, device=torch.device("cpu"))
        assert bool((tau >= TAU_MIN).all()), f"theta={theta} 下 tau 越下界"
        assert bool((tau <= TAU_MAX).all()), f"theta={theta} 下 tau 越上界"
        assert torch.isfinite(tau).all()


def test_alpha_matches_exponential_for_positive_delta() -> None:
    """正间隔的 alpha 必须是 1 - exp(-dt/tau)（逐维）。"""

    cell = build_cell()
    event = torch.randn(1, INPUT_DIM, generator=torch.Generator().manual_seed(1))
    state = torch.randn(1, HIDDEN_DIM, generator=torch.Generator().manual_seed(2))

    for dt in (0.5, 1.0, 4.32, 30.0, 200.0):
        dt_seconds, dt_valid, dt_reason, ordinal = single_step_inputs(dt=dt)
        out = cell.step(event, state, dt_seconds, dt_valid, dt_reason, ordinal)

        tau = cell.tau(dtype=torch.float32, device=torch.device("cpu"))
        effective = min(dt, DT_MAX)
        expected = (1.0 - torch.exp(-effective / tau)).unsqueeze(0)
        assert torch.allclose(out.alpha, expected, atol=1.0e-6), f"dt={dt} 的 alpha 不符"


def test_alpha_saturates_for_dt_beyond_dt_max() -> None:
    """dt 超过 dt_max 后必须被裁剪，alpha 不再变化。"""

    cell = build_cell()
    event = torch.zeros(1, INPUT_DIM)
    state = torch.zeros(1, HIDDEN_DIM)

    a_at_max = cell.step(
        event, state, *single_step_inputs(dt=DT_MAX)
    ).alpha
    a_beyond = cell.step(
        event, state, *single_step_inputs(dt=DT_MAX * 10.0)
    ).alpha
    assert torch.allclose(a_at_max, a_beyond, atol=0.0), "dt 裁剪后 alpha 仍应一致"


def test_first_event_forces_alpha_one_and_identity_with_mixed_memory() -> None:
    """首事件必须硬置 alpha=1，使 r_t 恒等于 mixed_memory（避免死启动）。"""

    cell = build_cell()
    event = torch.randn(1, INPUT_DIM, generator=torch.Generator().manual_seed(3))
    state = torch.randn(1, HIDDEN_DIM, generator=torch.Generator().manual_seed(4))

    out = cell.step(event, state, *single_step_inputs(reason=DT_REASON_FIRST_EVENT, dt=0.0))

    assert torch.all(out.alpha == 1.0)
    assert torch.equal(out.next_state, out.mixed_memory)
    # 与显式公式一致
    assert torch.equal(out.next_state, (1.0 - out.alpha) * state + out.alpha * out.mixed_memory)


def test_state_update_is_convex_combination() -> None:
    """r_t = (1-alpha)*r_{t-1} + alpha*m_t 必须逐元素成立。"""

    cell = build_cell()
    event = torch.randn(3, INPUT_DIM, generator=torch.Generator().manual_seed(5))
    state = torch.randn(3, HIDDEN_DIM, generator=torch.Generator().manual_seed(6))
    dt_seconds, dt_valid, dt_reason, ordinal = single_step_inputs(dt=3.0, batch=3)

    out = cell.step(event, state, dt_seconds, dt_valid, dt_reason, ordinal)
    expected = (1.0 - out.alpha) * state + out.alpha * out.mixed_memory
    assert torch.equal(out.next_state, expected)


def test_candidate_and_gate_ranges() -> None:
    """candidate 用 tanh、gate 用 sigmoid，两者都必须有界。

    在 float32 下 tanh/sigmoid 会精确饱和到端点，因此用闭区间。
    """

    cell = build_cell()
    for scale in (1.0, 1.0e3, 1.0e6):
        event = torch.full((2, INPUT_DIM), scale)
        state = torch.full((2, HIDDEN_DIM), scale)
        out = cell.step(event, state, *single_step_inputs(dt=1.0, batch=2))
        assert bool((out.candidate.abs() <= 1.0).all()), f"scale={scale} 下 candidate 越界"
        assert bool((out.gate >= 0.0).all()) and bool((out.gate <= 1.0).all())
        assert torch.isfinite(out.candidate).all() and torch.isfinite(out.gate).all()


def test_mixed_memory_is_gate_interpolation_of_candidate_and_raw_state() -> None:
    """m_t = g*c_t + (1-g)*r_{t-1}，且用的是 **raw** state 而非 normalized view。"""

    cell = build_cell()
    event = torch.randn(2, INPUT_DIM, generator=torch.Generator().manual_seed(8))
    state = torch.randn(2, HIDDEN_DIM, generator=torch.Generator().manual_seed(9))

    out = cell.step(event, state, *single_step_inputs(dt=1.0, batch=2))
    assert torch.equal(out.mixed_memory, out.gate * out.candidate + (1.0 - out.gate) * state)


def test_normalized_compute_state_is_not_written_back() -> None:
    """pre-norm 只用于计算：normalized view 不得等于 raw state，且 raw state 不被归一化。"""

    cell = build_cell()
    event = torch.randn(1, INPUT_DIM, generator=torch.Generator().manual_seed(10))
    # 刻意使用非单位尺度的 state，使 LayerNorm 必然改变它
    state = torch.full((1, HIDDEN_DIM), 100.0)

    out = cell.step(event, state, *single_step_inputs(dt=1.0))

    assert not torch.allclose(out.normalized_compute_state, state)
    # LayerNorm 输出接近零均值；raw state 保持原始量级
    assert abs(float(out.normalized_compute_state.mean())) < 1.0e-5
    assert float(state.abs().mean()) > 10.0


def test_raw_state_scale_affects_update_no_post_normalization() -> None:
    """无 post-norm 的可观测后果：仅缩放入态就会改变更新结果。

    若存在 post-update normalization，``s`` 与 ``100*s`` 会产生相同输出。
    """

    cell = build_cell()
    event = torch.randn(1, INPUT_DIM, generator=torch.Generator().manual_seed(11))
    base = torch.randn(1, HIDDEN_DIM, generator=torch.Generator().manual_seed(12))
    dt_seconds, dt_valid, dt_reason, ordinal = single_step_inputs(dt=1.0)

    small = cell.step(event, base, dt_seconds, dt_valid, dt_reason, ordinal)
    large = cell.step(event, base * 100.0, dt_seconds, dt_valid, dt_reason, ordinal)

    assert not torch.allclose(small.next_state, large.next_state), (
        "入态缩放不应被 post-norm 抹平"
    )
    # raw state 的尺度应原样传递到输出（未被拉回单位尺度）
    assert float(large.next_state.abs().mean()) > float(small.next_state.abs().mean()) * 10.0


def test_innovation_traces_are_consistent_differences() -> None:
    """三条 innovation 必须是各自定义的精确差分。"""

    cell = build_cell()
    event = torch.randn(2, INPUT_DIM, generator=torch.Generator().manual_seed(13))
    state = torch.randn(2, HIDDEN_DIM, generator=torch.Generator().manual_seed(14))

    out = cell.step(event, state, *single_step_inputs(dt=1.0, batch=2))

    assert torch.allclose(out.candidate_innovation, out.candidate - state, atol=0.0)
    assert torch.allclose(out.mixed_memory_innovation, out.mixed_memory - state, atol=0.0)
    assert torch.allclose(out.actual_update, out.next_state - state, atol=0.0)


def test_negative_dt_is_rejected_for_positive_reason() -> None:
    """契约要求正间隔 reason 必须 dt>0；负 dt 必须 fail-fast 而非静默 clamp。"""

    cell = build_cell()
    event = torch.zeros(1, INPUT_DIM)
    state = torch.zeros(1, HIDDEN_DIM)
    dt_seconds, dt_valid, dt_reason, ordinal = single_step_inputs(dt=-1.0)

    with pytest.raises(ValueError):
        cell.step(event, state, dt_seconds, dt_valid, dt_reason, ordinal)


def test_first_event_reason_requires_zero_dt() -> None:
    """首事件/同秒 reason 必须 dt=0。"""

    cell = build_cell()
    event = torch.zeros(1, INPUT_DIM)
    state = torch.zeros(1, HIDDEN_DIM)
    dt_seconds, dt_valid, dt_reason, ordinal = single_step_inputs(
        reason=DT_REASON_FIRST_EVENT, dt=5.0
    )

    with pytest.raises(ValueError):
        cell.step(event, state, dt_seconds, dt_valid, dt_reason, ordinal)


def test_dt_reason_must_agree_with_dt_valid() -> None:
    """dt_valid 必须恒等于 (dt_reason == positive_delta)。"""

    cell = build_cell()
    event = torch.zeros(1, INPUT_DIM)
    state = torch.zeros(1, HIDDEN_DIM)
    dt_seconds = torch.tensor([1.0])
    dt_reason = torch.tensor([DT_REASON_POSITIVE_DELTA], dtype=torch.long)
    ordinal = torch.tensor([0], dtype=torch.long)

    with pytest.raises(ValueError):
        cell.step(
            event, state, dt_seconds, torch.tensor([False]), dt_reason, ordinal
        )


# --------------------------------------------------------------------------
# 2. 退化时间语义（首事件 / 同秒 / 多种 mode）
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["floor", "bounded_observation", "skip", "strict_skip", "density_weighted"])
def test_zero_delta_alpha_matches_declared_mode_rule(mode: str) -> None:
    """同秒事件的 alpha 必须与该 mode 在 contract 中声明的规则一致。

    逐 mode 的实测语义（``_time_parameters``）：

    ===================== ==================================================
    mode                  同秒事件 alpha
    ===================== ==================================================
    ``floor``             ``1 - exp(-floor/tau)``（极小值，不依赖 gain）
    ``bounded_observation`` ``gain``
    ``skip``              ``0``（恒等，但**不**跳过 F）
    ``strict_skip``       ``0``（恒等且不进入 F，见另一用例）
    ``density_weighted``  ``min(gain*ordinal, 3*gain)``
    ===================== ==================================================

    关键契约：**alpha 必须落在 [0,1]，且同秒分支不依赖 tau**（除 ``floor``）。
    """

    cell = build_cell(mode)
    event = torch.randn(1, INPUT_DIM, generator=torch.Generator().manual_seed(15))
    state = torch.randn(1, HIDDEN_DIM, generator=torch.Generator().manual_seed(16))

    out = cell.step(event, state, *single_step_inputs(reason=DT_REASON_ZERO_DELTA, dt=0.0))

    assert bool((out.alpha >= 0.0).all()) and bool((out.alpha <= 1.0).all())

    tau = cell.tau(dtype=torch.float32, device=torch.device("cpu"))
    if mode == "floor":
        # 实测行为与 contract 文本不一致——见 test_floor_mode_contract_text_is_not_honoured
        expected = torch.full_like(tau, GAIN)
    elif mode in {"skip", "strict_skip"}:
        expected = torch.zeros_like(tau)
    else:  # bounded_observation / density_weighted(ordinal=1)
        expected = torch.full_like(tau, GAIN)

    assert torch.allclose(out.alpha, expected.unsqueeze(0), atol=1.0e-6), (
        f"mode={mode} 的同秒 alpha 与实测规则不符"
    )


def test_floor_mode_contract_text_is_not_honoured() -> None:
    """**已登记的契约/实现不一致**（本用例是可执行规格，不是回归保护）。

    ``time_policy_contract`` 对 ``floor`` 模式的 zero_delta 声明：

        "使用 zero_dt_event_floor_seconds 通过指数公式计算 alpha"

    若该文本成立，alpha 应为 ``1 - exp(-floor/tau) ≈ 6.66e-05``（floor=0.001, tau=15.025）。

    但 ``_time_parameters`` 的实现是：

    1. 第一个 ``if`` 的 mode 集合 **不含** ``floor``，因此 alpha 保留了指数值；
    2. 随后的 ``zero_gain`` 落到 ``else`` 分支，取 ``invalid_dt_observation_gain``；
    3. ``alpha = where(zero, zero_gain, alpha)`` **覆盖**了指数值。

    净效果：``floor`` 模式的同秒 alpha 实际等于 ``gain``（0.25），
    与契约文本声称的 ``6.66e-05`` 相差约 **3,755 倍**。

    本用例锁定"当前实现"这一事实。若将来修复契约文本或实现，**本用例应当失败**
    并促使更新该文档字符串——这是刻意的，不要为了让测试变绿而放宽断言。
    """

    cell = build_cell("floor")
    event = torch.randn(1, INPUT_DIM, generator=torch.Generator().manual_seed(32))
    state = torch.randn(1, HIDDEN_DIM, generator=torch.Generator().manual_seed(33))

    out = cell.step(event, state, *single_step_inputs(reason=DT_REASON_ZERO_DELTA, dt=0.0))

    tau = cell.tau(dtype=torch.float32, device=torch.device("cpu"))
    contract_implied = 1.0 - torch.exp(-torch.full_like(tau, FLOOR) / tau)
    actual = out.alpha.flatten()

    # 实现给出的是 gain，不是契约文本声称的指数地板值
    assert torch.allclose(actual, torch.full_like(actual, GAIN), atol=1.0e-6)
    assert not torch.allclose(actual, contract_implied, atol=1.0e-6)

    # 量化差异倍数，便于在报告与 issue 中引用
    ratio = float(GAIN / contract_implied.mean())
    assert 3.0e3 < ratio < 4.0e3, f"差异倍数应在 3.7e3 量级，实际 {ratio:.1f}"


def test_zero_delta_alpha_is_independent_of_tau_except_floor() -> None:
    """除 floor 外，同秒 alpha 不得随 tau 变化（契约明确要求不依赖 tau）。"""

    for mode in ("bounded_observation", "skip", "strict_skip", "density_weighted"):
        cell = build_cell(mode)
        event = torch.randn(1, INPUT_DIM, generator=torch.Generator().manual_seed(30))
        state = torch.randn(1, HIDDEN_DIM, generator=torch.Generator().manual_seed(31))
        inputs = single_step_inputs(reason=DT_REASON_ZERO_DELTA, dt=0.0)

        with torch.no_grad():
            cell.tau_unconstrained.fill_(-5.0)
        low_tau = cell.step(event, state, *inputs).alpha
        with torch.no_grad():
            cell.tau_unconstrained.fill_(5.0)
        high_tau = cell.step(event, state, *inputs).alpha

        assert torch.allclose(low_tau, high_tau, atol=0.0), (
            f"mode={mode} 的同秒 alpha 不应依赖 tau"
        )


def test_bounded_observation_zero_delta_uses_gain_not_zero() -> None:
    """反直觉设计：同秒事件的 alpha 是固定有界增益，不是 0。"""

    cell = build_cell("bounded_observation")
    event = torch.randn(1, INPUT_DIM, generator=torch.Generator().manual_seed(17))
    state = torch.randn(1, HIDDEN_DIM, generator=torch.Generator().manual_seed(18))

    out = cell.step(event, state, *single_step_inputs(reason=DT_REASON_ZERO_DELTA, dt=0.0))

    assert torch.all(out.alpha > 0.0)
    assert torch.allclose(out.alpha, torch.full_like(out.alpha, GAIN), atol=1.0e-6)


def test_density_weighted_scales_with_zero_ordinal_and_caps() -> None:
    """同秒多重性调制：alpha = min(gain*ordinal, 3*gain)。"""

    cell = build_cell("density_weighted")
    event = torch.randn(1, INPUT_DIM, generator=torch.Generator().manual_seed(19))
    state = torch.randn(1, HIDDEN_DIM, generator=torch.Generator().manual_seed(20))

    observed = []
    for ordinal in (1, 2, 3, 4, 10):
        out = cell.step(
            event,
            state,
            *single_step_inputs(reason=DT_REASON_ZERO_DELTA, dt=0.0, ordinal=ordinal),
        )
        observed.append(float(out.alpha.flatten()[0]))

    expected = [GAIN * 1, GAIN * 2, GAIN * 3, GAIN * 3, GAIN * 3]
    for got, want in zip(observed, expected):
        assert math.isclose(got, want, rel_tol=0.0, abs_tol=1.0e-6), (
            f"ordinal 序列应为 {expected}，实际 {observed}"
        )


def test_strict_skip_zero_delta_is_bitwise_identity() -> None:
    """strict-skip 必须给出与入态位级相等的状态，且 candidate/gate/alpha 全为 0。"""

    cell = build_cell("strict_skip")
    event = torch.randn(2, INPUT_DIM, generator=torch.Generator().manual_seed(21))
    state = torch.randn(2, HIDDEN_DIM, generator=torch.Generator().manual_seed(22))
    dt_seconds, dt_valid, dt_reason, ordinal = single_step_inputs(
        reason=DT_REASON_ZERO_DELTA, dt=0.0, batch=2
    )

    out = cell.step(event, state, dt_seconds, dt_valid, dt_reason, ordinal)

    assert torch.equal(out.next_state, state), "strict-skip 必须位级恒等"
    assert torch.all(out.candidate == 0.0)
    assert torch.all(out.gate == 0.0)
    assert torch.all(out.alpha == 0.0)
    assert torch.all(out.actual_update == 0.0)


def test_strict_skip_mixed_batch_keeps_computed_rows_intact() -> None:
    """strict-skip 在混合 batch 中只跳过同秒行，正间隔行必须正常计算。"""

    cell = build_cell("strict_skip")
    batch = 3
    event = torch.randn(batch, INPUT_DIM, generator=torch.Generator().manual_seed(23))
    state = torch.randn(batch, HIDDEN_DIM, generator=torch.Generator().manual_seed(24))

    dt_seconds = torch.tensor([0.0, 3.0, 0.0])
    dt_reason = torch.tensor(
        [DT_REASON_FIRST_EVENT, DT_REASON_POSITIVE_DELTA, DT_REASON_ZERO_DELTA],
        dtype=torch.long,
    )
    dt_valid = torch.tensor([False, True, False])
    ordinal = torch.ones(batch, dtype=torch.long)  # 契约要求 >= 1

    out = cell.step(event, state, dt_seconds, dt_valid, dt_reason, ordinal)

    # 行 2（同秒）位级恒等；行 1（正间隔）必须发生变化
    assert torch.equal(out.next_state[2], state[2])
    assert not torch.allclose(out.next_state[1], state[1])
    # 行 0（首事件）alpha=1
    assert torch.all(out.alpha[0] == 1.0)


# --------------------------------------------------------------------------
# 3. padding 与 strict-skip 的序列级契约
# --------------------------------------------------------------------------


def test_padding_does_not_call_cell() -> None:
    """padding_calls_cell=False：padded 位置不得进入 cell 计算。

    直接对 ``cell.step`` 计数——这是比"状态不变"更强的断言，
    因为状态不变也可能来自"调用了但恰好恒等"。
    """

    cell = build_cell("bounded_observation")
    wrapper = LiquidV2SequenceWrapper(cell)
    length, valid = 5, 3
    events, dt_seconds, padding_mask, dt_valid, dt_reason = sequence_inputs(length)
    padding_mask[:, valid:] = True
    dt_seconds[:, valid:] = 0.0
    dt_valid[:, valid:] = False
    dt_reason[:, valid:] = 0  # padding 常量

    calls = {"n": 0}
    original = cell.step

    def counting_step(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    cell.step = counting_step  # type: ignore[method-assign]
    try:
        wrapper(
            events,
            dt_seconds,
            padding_mask,
            dt_valid=dt_valid,
            dt_reason=dt_reason,
        )
    finally:
        cell.step = original  # type: ignore[method-assign]

    assert calls["n"] == valid, f"cell 应只被调用 {valid} 次，实际 {calls['n']}"


def test_padded_positions_hold_last_valid_state() -> None:
    """padded 位置的状态必须保持为最后一个有效事件的状态。"""

    cell = build_cell("bounded_observation")
    wrapper = LiquidV2SequenceWrapper(cell)
    length, valid = 6, 4
    events, dt_seconds, padding_mask, dt_valid, dt_reason = sequence_inputs(length)
    padding_mask[:, valid:] = True
    dt_seconds[:, valid:] = 0.0
    dt_valid[:, valid:] = False
    dt_reason[:, valid:] = 0

    out = wrapper(
        events, dt_seconds, padding_mask, dt_valid=dt_valid, dt_reason=dt_reason
    )

    last_valid = out.sequence[:, valid - 1]
    for position in range(valid, length):
        assert torch.equal(out.sequence[:, position], last_valid)
    assert torch.equal(out.final_state, last_valid)


def test_padding_must_be_suffix() -> None:
    """padding 必须是后缀；中间出现 padding 后又有有效事件必须报错。"""

    cell = build_cell("bounded_observation")
    wrapper = LiquidV2SequenceWrapper(cell)
    events, dt_seconds, padding_mask, dt_valid, dt_reason = sequence_inputs(4)
    padding_mask[:, 1] = True  # 非法：padding 后仍有有效事件
    dt_seconds[:, 1] = 0.0
    dt_valid[:, 1] = False
    dt_reason[:, 1] = 0

    with pytest.raises(ValueError):
        wrapper(events, dt_seconds, padding_mask, dt_valid=dt_valid, dt_reason=dt_reason)


# --------------------------------------------------------------------------
# 4. offline / online 一致性
# --------------------------------------------------------------------------


def test_offline_and_online_are_bitwise_identical() -> None:
    """逐事件调用 wrapper.step 必须与一次性 forward 位级等价。

    这是 online/offline 契约的核心；序列前向不得有任何跨事件的信息泄漏。
    """

    cell = build_cell("bounded_observation")
    wrapper = LiquidV2SequenceWrapper(cell)
    length = 7
    events, dt_seconds, padding_mask, dt_valid, dt_reason = sequence_inputs(length)

    offline = wrapper(
        events,
        dt_seconds,
        padding_mask,
        dt_valid=dt_valid,
        dt_reason=dt_reason,
        return_trace=True,
    )

    state = None
    online_states = []
    for index in range(length):
        step = wrapper.step(
            events[:, index],
            dt_seconds[:, index],
            dt_valid=dt_valid[:, index],
            dt_reason=dt_reason[:, index],
            state=state,
        )
        state = step.next_state
        online_states.append(state)

    online = torch.stack(online_states, dim=1)

    assert torch.equal(online, offline.sequence), "online 与 offline 必须位级一致"
    assert torch.equal(state, offline.final_state)


def test_offline_and_online_match_with_zero_delta_events() -> None:
    """含同秒事件的序列也必须在 offline/online 之间位级一致。"""

    cell = build_cell("bounded_observation")
    wrapper = LiquidV2SequenceWrapper(cell)
    length = 6
    events, dt_seconds, padding_mask, dt_valid, dt_reason = sequence_inputs(length)
    # 位置 1、2 设为同秒事件（dt=0, reason=zero_delta, valid=False）
    for position in (1, 2):
        dt_seconds[:, position] = 0.0
        dt_valid[:, position] = False
        dt_reason[:, position] = DT_REASON_ZERO_DELTA

    ordinals = wrapper.derive_zero_ordinals(dt_reason, padding_mask)
    offline = wrapper(
        events,
        dt_seconds,
        padding_mask,
        dt_valid=dt_valid,
        dt_reason=dt_reason,
        zero_ordinal=ordinals,
    )

    state = None
    online_states = []
    for index in range(length):
        step = wrapper.step(
            events[:, index],
            dt_seconds[:, index],
            dt_valid=dt_valid[:, index],
            dt_reason=dt_reason[:, index],
            state=state,
            zero_ordinal=ordinals[:, index],
        )
        state = step.next_state
        online_states.append(state)

    assert torch.equal(torch.stack(online_states, dim=1), offline.sequence)


def test_online_continuation_cannot_restart_first_event() -> None:
    """已有状态时不得重新标记首事件（防止在线续推被静默重置）。"""

    cell = build_cell("bounded_observation")
    wrapper = LiquidV2SequenceWrapper(cell)
    event = torch.randn(1, INPUT_DIM, generator=torch.Generator().manual_seed(25))
    state = torch.randn(1, HIDDEN_DIM, generator=torch.Generator().manual_seed(26))
    dt_seconds, dt_valid, dt_reason, ordinal = single_step_inputs(
        reason=DT_REASON_FIRST_EVENT, dt=0.0
    )

    with pytest.raises(ValueError):
        wrapper.step(
            event, dt_seconds, dt_valid=dt_valid, dt_reason=dt_reason, state=state
        )


def test_fresh_online_state_requires_first_event() -> None:
    """新建在线状态必须使用 first-event reason。"""

    cell = build_cell("bounded_observation")
    wrapper = LiquidV2SequenceWrapper(cell)
    event = torch.randn(1, INPUT_DIM, generator=torch.Generator().manual_seed(27))
    dt_seconds, dt_valid, dt_reason, ordinal = single_step_inputs(dt=1.0)

    with pytest.raises(ValueError):
        wrapper.step(
            event, dt_seconds, dt_valid=dt_valid, dt_reason=dt_reason, state=None
        )


# --------------------------------------------------------------------------
# 5. 结构 contract（checkpoint 完整性）
# --------------------------------------------------------------------------


def test_contract_declares_raw_memory_and_no_post_normalization() -> None:
    """写入 checkpoint 的 contract 必须声明 raw memory 且无 post-norm。"""

    contract = liquid_v2_contract(build_cell())

    assert contract["equation_version"] == EQUATION_VERSION
    assert contract["memory_state"] == "raw_r_t"
    assert contract["post_update_normalization"] is False
    assert contract["padding_calls_cell"] is False
    assert contract["classifier_readout_must_be_independent"] is True
    assert contract["candidate_function"]["v1_capacity_preserved"] is True


def test_contract_carries_time_policy_and_innovation_names() -> None:
    """contract 必须同时登记时间策略与三条 innovation 的语义。

    ``version`` 只断言后缀：两版实现的**唯一**已知识别差异是前缀
    （``src/`` 为 ``RADAR-LN3-reason-aware-time-policy-v2``，
    ``src-v3/`` 为 ``RADAR-LN3-V27-reason-aware-time-policy-v2``）。
    用同一份测试跑两版实现时，其余 50 条断言全部一致。
    """

    contract = liquid_v2_contract(build_cell("bounded_observation"))

    policy = contract["time_policy"]
    assert policy["version"].endswith("reason-aware-time-policy-v2")
    assert "bounded_observation" in policy["supported_modes"]

    innovation = contract["innovation_names"]
    assert innovation["candidate"] == "candidate_innovation=c_t-r_{t-1}"
    assert innovation["mixed_memory"] == "mixed_memory_innovation=m_t-r_{t-1}"
    assert innovation["actual_update"] == "actual_update=r_t-r_{t-1}"


def test_contract_is_stable_across_calls() -> None:
    """同一 cell 的 contract 必须可重复序列化（用于 checkpoint 哈希）。"""

    cell = build_cell()
    assert liquid_v2_contract(cell) == liquid_v2_contract(cell)


@pytest.mark.parametrize(
    "mode", ["floor", "bounded_observation", "skip", "strict_skip", "density_weighted"]
)
def test_every_declared_mode_has_a_policy_rule(mode: str) -> None:
    """声明的每个 mode 都必须在 time_policy_contract 中有对应规则。"""

    contract = time_policy_contract(
        tau_min_seconds=TAU_MIN,
        tau_max_seconds=TAU_MAX,
        zero_dt_event_floor_seconds=FLOOR,
        dt_max_seconds=DT_MAX,
        invalid_dt_update_mode=mode,
        invalid_dt_observation_gain=GAIN,
    )
    assert mode in contract["mode_rules"]
    assert set(contract["mode_rules"][mode]) == {"zero_delta", "invalid_dt"}


def test_unknown_mode_is_rejected() -> None:
    """未登记的 mode 必须在构造期即被拒绝，而不是运行时静默降级。"""

    with pytest.raises(ValueError):
        build_cell("not_a_real_mode")
    with pytest.raises(ValueError):
        time_policy_contract(
            tau_min_seconds=TAU_MIN,
            tau_max_seconds=TAU_MAX,
            zero_dt_event_floor_seconds=FLOOR,
            dt_max_seconds=DT_MAX,
            invalid_dt_update_mode="not_a_real_mode",
            invalid_dt_observation_gain=GAIN,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"tau_min_seconds": 0.0},
        {"tau_min_seconds": -1.0},
        {"tau_max_seconds": 0.01},          # 小于 tau_min
        {"zero_dt_event_floor_seconds": 0.0},
        {"zero_dt_event_floor_seconds": -1.0},
        {"dt_max_seconds": 1.0e-6},          # 小于 floor
        {"invalid_dt_observation_gain": 0.0},
        {"invalid_dt_observation_gain": -0.5},
        {"invalid_dt_observation_gain": 1.5},
    ],
)
def test_invalid_configurations_are_rejected(overrides: dict) -> None:
    """时间策略参数越界必须在构造期 fail-fast。"""

    with pytest.raises(ValueError):
        build_cell(**overrides)


def test_trace_field_names_match_emitted_trace() -> None:
    """TRACE_FIELD_NAMES 必须与序列前向实际产出的 trace 键完全一致。

    该断言守护"契约声明的可观测字段"与"实现实际提供的字段"不漂移。
    """

    cell = build_cell("bounded_observation")
    wrapper = LiquidV2SequenceWrapper(cell)
    events, dt_seconds, padding_mask, dt_valid, dt_reason = sequence_inputs(3)

    out = wrapper(
        events,
        dt_seconds,
        padding_mask,
        dt_valid=dt_valid,
        dt_reason=dt_reason,
        return_trace=True,
    )

    assert out.trace is not None
    assert set(out.trace) == set(TRACE_FIELD_NAMES), (
        f"trace 键不一致：缺 {set(TRACE_FIELD_NAMES) - set(out.trace)}，"
        f"多 {set(out.trace) - set(TRACE_FIELD_NAMES)}"
    )


# --------------------------------------------------------------------------
# 6. 反向传播与梯度可达性
# --------------------------------------------------------------------------
#
# 前 5 节全部是前向断言。本节补上反向覆盖——尤其针对本项目的核心主张
# **"有界可学习时间常数"**：如果 ``tau_unconstrained`` 拿不到非零梯度，
# 那么"可学习"这一说法在代码层面就是空的。


def _sequence_tensors(length: int, *, dt: float = 2.0):
    """构造一条含正间隔的序列（用于梯度测试）。"""

    return sequence_inputs(length, dt=dt)


def test_tau_receives_nonzero_gradient() -> None:
    """**核心主张的可训练性**：tau_unconstrained 必须拿到非零梯度。

    这是"有界可学习时间常数"是否真的在学习的唯一直接证据。
    若该断言失败，说明时间常数实际上被冻结，"可学习"的说法不成立。
    """

    cell = build_cell("bounded_observation")
    wrapper = LiquidV2SequenceWrapper(cell)
    events, dt_seconds, padding_mask, dt_valid, dt_reason = _sequence_tensors(6)

    out = wrapper(events, dt_seconds, padding_mask, dt_valid=dt_valid, dt_reason=dt_reason)
    loss = out.final_state.square().sum()
    loss.backward()

    grad = cell.tau_unconstrained.grad
    assert grad is not None, "tau_unconstrained 未获得梯度"
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0, (
        "tau_unconstrained 的梯度恒为零 —— '有界可学习时间常数'未被训练"
    )


def test_tau_gradient_is_zero_when_only_first_event_is_present() -> None:
    """首事件的 alpha 被硬置为 1，因此该路径**不**训练 tau。

    这不是缺陷，而是 ``alpha = where(first, 1, alpha)`` 的必然结果；
    本用例把它固化为契约，避免将来误以为"任何序列都会训练 tau"。

    注意：契约要求一条新序列**恰好**包含一个 first-event reason
    （``_validate_first_event_contract``），因此这里用长度 1 的序列，
    而**不能**把多个事件都标成 first-event。
    """

    cell = build_cell("bounded_observation")
    wrapper = LiquidV2SequenceWrapper(cell)
    events = torch.randn(1, 1, INPUT_DIM, generator=torch.Generator().manual_seed(43))
    dt_seconds = torch.zeros((1, 1), dtype=torch.float32)
    padding_mask = torch.zeros((1, 1), dtype=torch.bool)
    dt_valid = torch.zeros((1, 1), dtype=torch.bool)
    dt_reason = torch.full((1, 1), DT_REASON_FIRST_EVENT, dtype=torch.long)

    out = wrapper(
        events, dt_seconds, padding_mask, dt_valid=dt_valid, dt_reason=dt_reason
    )
    # alpha 被硬置为 1，状态直接等于 mixed_memory
    assert torch.allclose(out.final_state, out.sequence[:, 0], atol=0.0)

    out.final_state.square().sum().backward()

    grad = cell.tau_unconstrained.grad
    assert grad is None or float(grad.abs().sum()) == 0.0, (
        "首事件的 alpha 与 tau 无关，不应产生 tau 梯度"
    )


def test_all_learnable_parameters_receive_gradient() -> None:
    """四个 projection 与 compute_norm 都必须获得非零且有限的梯度。"""

    cell = build_cell("bounded_observation")
    wrapper = LiquidV2SequenceWrapper(cell)
    events, dt_seconds, padding_mask, dt_valid, dt_reason = _sequence_tensors(6)

    out = wrapper(events, dt_seconds, padding_mask, dt_valid=dt_valid, dt_reason=dt_reason)
    out.sequence.square().sum().backward()

    expected = {
        "compute_norm.weight",
        "compute_norm.bias",
        "input_projection.weight",
        "input_projection.bias",
        "state_projection.weight",
        "candidate_projection.weight",
        "candidate_projection.bias",
        "gate_projection.weight",
        "gate_projection.bias",
        "tau_unconstrained",
    }
    seen = {}
    for name, parameter in cell.named_parameters():
        seen[name] = parameter.grad

    assert set(seen) == expected, f"参数集合漂移：{set(seen) ^ expected}"
    for name, grad in seen.items():
        assert grad is not None, f"{name} 未获得梯度"
        assert torch.isfinite(grad).all(), f"{name} 的梯度含非有限值"
        assert float(grad.abs().sum()) > 0.0, f"{name} 的梯度恒为零"


def test_gradients_are_finite_for_extreme_dt() -> None:
    """极端 dt（含远超 dt_max）不得产生 NaN/Inf 梯度。"""

    cell = build_cell("bounded_observation")
    wrapper = LiquidV2SequenceWrapper(cell)
    events, dt_seconds, padding_mask, dt_valid, dt_reason = _sequence_tensors(4)
    dt_seconds[:, 1:] = DT_MAX * 50.0

    out = wrapper(events, dt_seconds, padding_mask, dt_valid=dt_valid, dt_reason=dt_reason)
    out.sequence.square().sum().backward()

    for name, parameter in cell.named_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all(), f"{name} 在极端 dt 下梯度非有限"


def test_alpha_is_monotonically_decreasing_in_tau() -> None:
    """d(alpha)/d(tau) < 0：tau 越大，同一 dt 的同化步长越小。

    用有限差分核对解析梯度方向，比单纯的前向数值断言更强——
    它验证的是"时间常数确实在控制时间尺度"这一语义。
    """

    cell = build_cell("bounded_observation")
    event = torch.randn(1, INPUT_DIM, generator=torch.Generator().manual_seed(40))
    state = torch.randn(1, HIDDEN_DIM, generator=torch.Generator().manual_seed(41))
    inputs = single_step_inputs(dt=5.0)

    alphas = []
    for theta in (-2.0, -1.0, 0.0, 1.0, 2.0):
        with torch.no_grad():
            cell.tau_unconstrained.fill_(theta)
        alphas.append(float(cell.step(event, state, *inputs).alpha.mean()))

    assert alphas == sorted(alphas, reverse=True), (
        f"alpha 应随 tau 单调递减，实际序列 {alphas}"
    )
    assert len(set(alphas)) == len(alphas), "不同 tau 应给出不同的 alpha"


def test_no_post_normalization_holds_in_backward_path() -> None:
    """无 post-norm 在反向也成立：状态尺度差异必须传导到梯度量级。"""

    cell = build_cell("bounded_observation")
    event = torch.randn(1, INPUT_DIM, generator=torch.Generator().manual_seed(42))
    dt_seconds, dt_valid, dt_reason, ordinal = single_step_inputs(dt=1.0)

    grads = []
    for scale in (1.0, 50.0):
        state = torch.full((1, HIDDEN_DIM), scale, requires_grad=True)
        out = cell.step(event, state, dt_seconds, dt_valid, dt_reason, ordinal)
        out.next_state.square().sum().backward()
        grads.append(float(state.grad.abs().mean()))

    assert grads[1] > grads[0] * 10.0, (
        f"入态尺度应显著影响梯度量级（无 post-norm），实际 {grads}"
    )


# --------------------------------------------------------------------------
# 7. zero_ordinal 的派生与校验契约
# --------------------------------------------------------------------------
#
# ``derive_zero_ordinals`` 有两种模式：
#   * 派生模式（supplied=None）—— offline 路径使用，自己算出 ordinal
#   * 校验模式（supplied 给定）—— online 续推必须提供 ordinal
# 两者对"首个同秒事件"的取值存在分歧，见 §7.3。


def _reasons(*values: int) -> torch.Tensor:
    return torch.tensor([list(values)], dtype=torch.long)


def _no_padding(length: int) -> torch.Tensor:
    return torch.zeros((1, length), dtype=torch.bool)


def test_derived_ordinals_reset_on_positive_delta() -> None:
    """派生模式：FIRST/POSITIVE 重置为 1，连续 ZERO 递增。"""

    wrapper = LiquidV2SequenceWrapper(build_cell())
    reason = _reasons(
        DT_REASON_FIRST_EVENT,
        DT_REASON_ZERO_DELTA,
        DT_REASON_ZERO_DELTA,
        DT_REASON_POSITIVE_DELTA,
        DT_REASON_ZERO_DELTA,
    )
    derived = wrapper.derive_zero_ordinals(reason, _no_padding(5))
    assert derived.tolist() == [[1, 2, 3, 1, 2]]


def test_derived_ordinals_are_zero_at_padded_positions() -> None:
    """派生模式：padded 位置的 ordinal 必须为 0。"""

    wrapper = LiquidV2SequenceWrapper(build_cell())
    reason = _reasons(DT_REASON_FIRST_EVENT, DT_REASON_POSITIVE_DELTA, 0, 0)
    padding = torch.tensor([[False, False, True, True]])
    derived = wrapper.derive_zero_ordinals(reason, padding)
    assert derived.tolist() == [[1, 1, 0, 0]]


def test_derived_mode_requires_explicit_ordinals_when_continuing_a_state() -> None:
    """已有 initial_state 时，同秒事件必须显式提供 ordinal，不得静默派生。

    该约束保证"在线续推"无法凭空猜测同秒序号——否则 density_weighted
    的增益会随调用方式漂移。
    """

    wrapper = LiquidV2SequenceWrapper(build_cell())
    reason = _reasons(DT_REASON_POSITIVE_DELTA, DT_REASON_ZERO_DELTA)
    padding = _no_padding(2)
    initial = torch.zeros((1, HIDDEN_DIM))

    with pytest.raises(ValueError):
        wrapper.derive_zero_ordinals(reason, padding, initial_state=initial)


def test_supplied_ordinals_reject_invalid_values() -> None:
    """校验模式必须拒绝：负数、padded 非零、positive-delta 非 1、首事件非 1。"""

    wrapper = LiquidV2SequenceWrapper(build_cell())
    reason = _reasons(DT_REASON_FIRST_EVENT, DT_REASON_POSITIVE_DELTA)
    padding = _no_padding(2)

    # 负数
    with pytest.raises(ValueError):
        wrapper.derive_zero_ordinals(
            reason, padding, supplied=torch.tensor([[-1, 1]], dtype=torch.long)
        )
    # positive-delta 必须为 1
    with pytest.raises(ValueError):
        wrapper.derive_zero_ordinals(
            reason, padding, supplied=torch.tensor([[1, 2]], dtype=torch.long)
        )
    # 首事件必须为 1
    with pytest.raises(ValueError):
        wrapper.derive_zero_ordinals(
            reason, padding, supplied=torch.tensor([[2, 1]], dtype=torch.long)
        )

    # padded 位置必须为 0
    padded_reason = _reasons(DT_REASON_FIRST_EVENT, 0)
    padded_mask = torch.tensor([[False, True]])
    with pytest.raises(ValueError):
        wrapper.derive_zero_ordinals(
            padded_reason,
            padded_mask,
            supplied=torch.tensor([[1, 5]], dtype=torch.long),
        )


def test_supplied_ordinals_require_increment_for_consecutive_zero_delta() -> None:
    """校验模式：连续 ZERO→ZERO 的 ordinal 必须严格 +1。"""

    wrapper = LiquidV2SequenceWrapper(build_cell())
    reason = _reasons(
        DT_REASON_FIRST_EVENT, DT_REASON_ZERO_DELTA, DT_REASON_ZERO_DELTA
    )
    padding = _no_padding(3)

    # 合法：1,1,2
    ok = wrapper.derive_zero_ordinals(
        reason, padding, supplied=torch.tensor([[1, 1, 2]], dtype=torch.long)
    )
    assert ok.tolist() == [[1, 1, 2]]

    # 非法：1,1,3（跳跃）
    with pytest.raises(ValueError):
        wrapper.derive_zero_ordinals(
            reason, padding, supplied=torch.tensor([[1, 1, 3]], dtype=torch.long)
        )
    # 非法：1,1,1（不递增）
    with pytest.raises(ValueError):
        wrapper.derive_zero_ordinals(
            reason, padding, supplied=torch.tensor([[1, 1, 1]], dtype=torch.long)
        )


def test_first_same_second_ordinal_is_ambiguous_between_modes() -> None:
    """**已登记的不一致**：首个同秒事件的 ordinal 在两种模式下取值不同。

    对 ``[FIRST_EVENT, ZERO_DELTA]``：

    * **派生模式**（offline 使用）给出 ``[1, 2]``——把首事件也计为第 1 次观测；
    * **校验模式**（online 必须使用）对 ``[1, 1]`` 与 ``[1, 2]`` **都接受**——
      它只在"连续 ZERO→ZERO"时强制 +1，不约束 ZERO 跟在 FIRST/POSITIVE 之后的情形。

    因此"首个同秒事件"的序号是 **1 还是 2 未定义**，而
    ``density_weighted`` 的 alpha = ``min(gain*ordinal, 3*gain)``
    会因此**相差 2 倍**。

    影响范围：仅 ``density_weighted``（V28 使用 ``bounded_observation``，ordinal 不参与
    alpha 计算，故不受影响）。但对任何启用 ``density_weighted`` 的实验，
    **offline 与 online 可能给出不同的状态更新**。

    本用例锁定该事实。若将来统一定义，本用例应当失败以促使更新文档。
    """

    wrapper = LiquidV2SequenceWrapper(build_cell("density_weighted"))
    reason = _reasons(DT_REASON_FIRST_EVENT, DT_REASON_ZERO_DELTA)
    padding = _no_padding(2)

    derived = wrapper.derive_zero_ordinals(reason, padding)
    assert derived.tolist() == [[1, 2]], "派生模式当前把首事件计为 ordinal 1"

    # 校验模式对 1 和 2 都不报错 —— 这正是歧义所在
    accepted_one = wrapper.derive_zero_ordinals(
        reason, padding, supplied=torch.tensor([[1, 1]], dtype=torch.long)
    )
    accepted_two = wrapper.derive_zero_ordinals(
        reason, padding, supplied=torch.tensor([[1, 2]], dtype=torch.long)
    )
    assert accepted_one.tolist() == [[1, 1]]
    assert accepted_two.tolist() == [[1, 2]]
    assert not torch.equal(accepted_one, derived), "校验模式应能产出与派生模式不同的 ordinal"


def test_density_weighted_alpha_differs_when_ordinal_choice_differs() -> None:
    """上一条歧义的可观测后果：同一序列、不同 ordinal → alpha 相差 2 倍。

    这是 ``density_weighted`` 下 offline/online 潜在分歧的最小复现。
    """

    cell = build_cell("density_weighted")
    wrapper = LiquidV2SequenceWrapper(cell)
    event = torch.randn(1, INPUT_DIM, generator=torch.Generator().manual_seed(50))
    state = torch.randn(1, HIDDEN_DIM, generator=torch.Generator().manual_seed(51))

    dt_seconds = torch.tensor([[0.0, 0.0]])
    dt_valid = torch.tensor([[False, False]])
    dt_reason = _reasons(DT_REASON_FIRST_EVENT, DT_REASON_ZERO_DELTA)
    ordinal_one = torch.tensor([[1, 1]], dtype=torch.long)
    ordinal_two = torch.tensor([[1, 2]], dtype=torch.long)

    # 第二列是同秒事件，alpha 由 ordinal 决定
    alpha_one = cell.step(
        event,
        state,
        dt_seconds[:, 1],
        dt_valid[:, 1],
        dt_reason[:, 1],
        ordinal_one[:, 1],
    ).alpha
    alpha_two = cell.step(
        event,
        state,
        dt_seconds[:, 1],
        dt_valid[:, 1],
        dt_reason[:, 1],
        ordinal_two[:, 1],
    ).alpha

    assert torch.allclose(alpha_one, torch.full_like(alpha_one, GAIN), atol=1.0e-6)
    assert torch.allclose(alpha_two, torch.full_like(alpha_two, 2.0 * GAIN), atol=1.0e-6)
    assert not torch.allclose(alpha_one, alpha_two)

    # 该差异确实会改变状态更新
    update_one = cell.step(
        event, state, dt_seconds[:, 1], dt_valid[:, 1], dt_reason[:, 1], ordinal_one[:, 1]
    ).next_state
    update_two = cell.step(
        event, state, dt_seconds[:, 1], dt_valid[:, 1], dt_reason[:, 1], ordinal_two[:, 1]
    ).next_state
    assert not torch.allclose(update_one, update_two)
