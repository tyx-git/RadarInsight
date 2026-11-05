"""跨树 `liquid_v2` 等价性测试：`src/` vs `src-v3/`。

## 为什么需要这个测试

项目里存在**两份** `liquid_v2` 实现：

* ``src/radar_ln3/liquid_v2/liquid_v2.py``（969 行，包形式）
* ``src-v3/radar_ln3/liquid_v2.py``（952 行，模块形式）

`src-v3/v28/protocol.py:348-358` 的 ``public_dependency_hashes`` 绑定的是 **src-v3 那一份**，
而 v28 是产品路径。两份实现必须**手工同步**——在没有本测试之前，它们之间的漂移
不会有任何自动化机制发现。

补充篇曾用 **AST 归一化指纹**（9 个函数逐位相同）+ **40 组 8 行窗口**证明二者是同一份拷贝；
本文件提供**行为证据**：同权重、同输入下逐位比较输出。

## 实现方式

两份模块同名（都是 ``radar_ln3.liquid_v2``），无法用普通 import 同时加载。
这里把 ``src-v3/radar_ln3`` 注册为一个**合成包** ``radar_ln3_v3``，使其相对导入
（``from .data import ...``）解析到 ``src-v3`` 内部，从而与 ``src/radar_ln3`` 互不干扰。

**已知局限**：``src-v3`` 中若存在 ``from radar_ln3.x import ...`` 形式的**绝对**导入，
会解析到 ``src/`` 版本。``liquid_v2.py`` 本身只用相对导入，因此不影响本测试的有效性；
但本测试**不**覆盖 ``src-v3`` 其它模块的独立性。
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V3_PACKAGE_ROOT = PROJECT_ROOT / "src-v3" / "radar_ln3"
SYNTHETIC_PACKAGE = "radar_ln3_v3"

INPUT_DIM = 5
HIDDEN_DIM = 7
CELL_KWARGS = dict(
    tau_min_seconds=0.05,
    tau_max_seconds=30.0,
    zero_dt_event_floor_seconds=0.001,
    dt_max_seconds=120.0,
    invalid_dt_observation_gain=0.25,
)
ALL_MODES = ("floor", "bounded_observation", "skip", "strict_skip", "density_weighted")


def _load_v3_module():
    """把 src-v3/radar_ln3 注册为合成包并加载其 liquid_v2。"""

    if not V3_PACKAGE_ROOT.is_dir():
        pytest.skip(f"src-v3 实现不存在：{V3_PACKAGE_ROOT}")
    if SYNTHETIC_PACKAGE not in sys.modules:
        package = types.ModuleType(SYNTHETIC_PACKAGE)
        package.__path__ = [str(V3_PACKAGE_ROOT)]
        sys.modules[SYNTHETIC_PACKAGE] = package
    return importlib.import_module(f"{SYNTHETIC_PACKAGE}.liquid_v2")


@pytest.fixture(scope="module")
def implementations():
    """返回 (src 实现, src-v3 实现)。"""

    from radar_ln3 import liquid_v2 as src_impl

    v3_impl = _load_v3_module()
    assert "src-v3" in str(v3_impl.__file__).replace("\\", "/"), (
        f"未加载到 src-v3 实现：{v3_impl.__file__}"
    )
    assert "src-v3" not in str(src_impl.__file__).replace("\\", "/"), (
        f"未加载到 src 实现：{src_impl.__file__}"
    )
    return src_impl, v3_impl


def _build_pair(impl_a, impl_b, mode: str, seed: int = 20260924):
    """以同一随机种子构造两版 cell，返回 (cell_a, cell_b)。

    断言同种子下初始权重逐位一致——这是"同一份实现"的最强静态证据。
    """

    torch.manual_seed(seed)
    cell_a = impl_a.LiquidV2Cell(INPUT_DIM, HIDDEN_DIM, invalid_dt_update_mode=mode, **CELL_KWARGS)
    torch.manual_seed(seed)
    cell_b = impl_b.LiquidV2Cell(INPUT_DIM, HIDDEN_DIM, invalid_dt_update_mode=mode, **CELL_KWARGS)

    params_a = dict(cell_a.named_parameters())
    params_b = dict(cell_b.named_parameters())
    assert set(params_a) == set(params_b), (
        f"参数名集合不一致：{set(params_a) ^ set(params_b)}"
    )
    for name in params_a:
        assert torch.equal(params_a[name], params_b[name]), (
            f"同种子下参数 {name} 初始值不同"
        )
    return cell_a, cell_b


def _sequence(impl, length: int = 6, *, zero_at=()):
    """构造一条两版共用的序列输入（张量本身与实现无关）。"""

    events = torch.randn(
        1, length, INPUT_DIM, generator=torch.Generator().manual_seed(7)
    )
    dt_seconds = torch.full((1, length), 2.0)
    dt_seconds[:, 0] = 0.0
    dt_valid = torch.ones((1, length), dtype=torch.bool)
    dt_valid[:, 0] = False
    dt_reason = torch.full((1, length), 3, dtype=torch.long)  # positive_delta
    dt_reason[:, 0] = 1  # first_event
    for position in zero_at:
        dt_seconds[:, position] = 0.0
        dt_valid[:, position] = False
        dt_reason[:, position] = 2  # zero_delta
    padding_mask = torch.zeros((1, length), dtype=torch.bool)
    return events, dt_seconds, padding_mask, dt_valid, dt_reason


# --------------------------------------------------------------------------
# 静态等价
# --------------------------------------------------------------------------


def test_parameter_sets_are_identical(implementations) -> None:
    """两版 cell 的可学习参数名集合必须完全一致。"""

    src_impl, v3_impl = implementations
    cell_a, cell_b = _build_pair(src_impl, v3_impl, "bounded_observation")
    assert set(dict(cell_a.named_parameters())) == set(dict(cell_b.named_parameters()))
    assert len(dict(cell_a.named_parameters())) == 10


def test_trace_field_names_are_identical(implementations) -> None:
    """两版导出的 trace 字段名必须一致（否则 trace 消费方会漂移）。"""

    src_impl, v3_impl = implementations
    assert set(src_impl.TRACE_FIELD_NAMES) == set(v3_impl.TRACE_FIELD_NAMES)
    assert src_impl.TRACE_FIELD_NAMES == v3_impl.TRACE_FIELD_NAMES, "顺序也应一致"


def test_supported_modes_are_identical(implementations) -> None:
    """两版声明支持的 update mode 集合必须一致。"""

    src_impl, v3_impl = implementations
    assert (
        src_impl.LiquidV2Cell.SUPPORTED_UPDATE_MODES
        == v3_impl.LiquidV2Cell.SUPPORTED_UPDATE_MODES
    )


# --------------------------------------------------------------------------
# 行为等价（逐位）
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ALL_MODES)
def test_single_step_outputs_are_bitwise_identical(implementations, mode: str) -> None:
    """同权重、同输入下，单步输出的每一个字段都必须逐位相等。"""

    src_impl, v3_impl = implementations
    cell_a, cell_b = _build_pair(src_impl, v3_impl, mode)

    event = torch.randn(2, INPUT_DIM, generator=torch.Generator().manual_seed(11))
    state = torch.randn(2, HIDDEN_DIM, generator=torch.Generator().manual_seed(12))
    dt_seconds = torch.tensor([3.0, 0.0])
    dt_valid = torch.tensor([True, False])
    dt_reason = torch.tensor([3, 2], dtype=torch.long)
    ordinal = torch.tensor([1, 1], dtype=torch.long)

    out_a = cell_a.step(event, state, dt_seconds, dt_valid, dt_reason, ordinal)
    out_b = cell_b.step(event, state, dt_seconds, dt_valid, dt_reason, ordinal)

    for field in (
        "next_state",
        "normalized_compute_state",
        "candidate",
        "gate",
        "mixed_memory",
        "candidate_innovation",
        "mixed_memory_innovation",
        "actual_update",
        "alpha",
        "tau",
    ):
        assert torch.equal(getattr(out_a, field), getattr(out_b, field)), (
            f"mode={mode} 的 {field} 不逐位相等"
        )
    for field in ("dt_clamped_mask", "dt_zero_mask", "dt_invalid_mask", "dt_reason"):
        assert torch.equal(getattr(out_a, field), getattr(out_b, field)), (
            f"mode={mode} 的 {field} 不一致"
        )


@pytest.mark.parametrize("mode", ALL_MODES)
def test_sequence_outputs_are_bitwise_identical(implementations, mode: str) -> None:
    """整条序列（含 trace）在两版实现之间必须逐位相等。"""

    src_impl, v3_impl = implementations
    wrapper_a = src_impl.LiquidV2SequenceWrapper(_build_pair(src_impl, v3_impl, mode)[0])
    wrapper_b = v3_impl.LiquidV2SequenceWrapper(_build_pair(src_impl, v3_impl, mode)[1])

    events, dt_seconds, padding_mask, dt_valid, dt_reason = _sequence(src_impl, zero_at=(3,))

    out_a = wrapper_a(
        events,
        dt_seconds,
        padding_mask,
        dt_valid=dt_valid,
        dt_reason=dt_reason,
        return_trace=True,
    )
    out_b = wrapper_b(
        events,
        dt_seconds,
        padding_mask,
        dt_valid=dt_valid,
        dt_reason=dt_reason,
        return_trace=True,
    )

    assert torch.equal(out_a.sequence, out_b.sequence), f"mode={mode} 的 sequence 不逐位相等"
    assert torch.equal(out_a.final_state, out_b.final_state), f"mode={mode} 的 final_state 不逐位相等"

    assert out_a.trace is not None and out_b.trace is not None
    assert set(out_a.trace) == set(out_b.trace)
    for name in out_a.trace:
        assert torch.equal(out_a.trace[name], out_b.trace[name]), (
            f"mode={mode} 的 trace[{name}] 不逐位相等"
        )


def test_derived_ordinals_are_identical(implementations) -> None:
    """`derive_zero_ordinals` 的派生结果必须一致（含首事件后的同秒事件）。"""

    src_impl, v3_impl = implementations
    _, dt_seconds, padding_mask, dt_valid, dt_reason = _sequence(src_impl, zero_at=(1, 2, 5))

    derived_a = src_impl.LiquidV2SequenceWrapper.derive_zero_ordinals(dt_reason, padding_mask)
    derived_b = v3_impl.LiquidV2SequenceWrapper.derive_zero_ordinals(dt_reason, padding_mask)

    assert torch.equal(derived_a, derived_b)
    # 同时锁定当前语义：首事件计为 1，其后连续同秒递增
    assert derived_a.tolist() == [[1, 2, 3, 1, 1, 2]]


def test_offline_online_equivalence_holds_in_both_implementations(implementations) -> None:
    """offline/online 位级等价这一契约，两版实现都必须满足。"""

    src_impl, v3_impl = implementations
    length = 5

    for impl in (src_impl, v3_impl):
        wrapper = impl.LiquidV2SequenceWrapper(
            _build_pair(src_impl, v3_impl, "bounded_observation")[0 if impl is src_impl else 1]
        )
        events, dt_seconds, padding_mask, dt_valid, dt_reason = _sequence(impl, length)
        offline = wrapper(
            events, dt_seconds, padding_mask, dt_valid=dt_valid, dt_reason=dt_reason
        )

        state = None
        collected = []
        for index in range(length):
            step = wrapper.step(
                events[:, index],
                dt_seconds[:, index],
                dt_valid=dt_valid[:, index],
                dt_reason=dt_reason[:, index],
                state=state,
            )
            state = step.next_state
            collected.append(state)

        assert torch.equal(torch.stack(collected, dim=1), offline.sequence), (
            f"{impl.__file__} 的 offline/online 不等价"
        )


#: contract 中**允许**在两版之间不同的键（相对路径，扁平化后）。
#: 全部属于两类：版本标识、以及人类可读文案（src/ 中文、src-v3/ 英文）。
#: 若将来出现**不在**此集合中的差异键，说明出现了实质分歧，测试应当失败。
_ALLOWED_CONTRACT_DIFFERENCES = frozenset(
    {
        # 版本标识
        "equation_version",
        "time_policy.version",
        # 人类可读文案
        "compute_state",
        "time_policy.active_invalid_dt_rule",
        "time_policy.active_zero_delta_rule",
        "time_policy.first_event_rule",
        "time_policy.padding_rule",
        "time_policy.strict_skip_rule",
        "time_policy.zero_delta_rule",
        "time_policy.mode_rules.bounded_observation.invalid_dt",
        "time_policy.mode_rules.bounded_observation.zero_delta",
        "time_policy.mode_rules.density_weighted.invalid_dt",
        "time_policy.mode_rules.density_weighted.zero_delta",
        "time_policy.mode_rules.floor.invalid_dt",
        "time_policy.mode_rules.floor.zero_delta",
        "time_policy.mode_rules.skip.invalid_dt",
        "time_policy.mode_rules.skip.zero_delta",
        "time_policy.mode_rules.strict_skip.invalid_dt",
        "time_policy.mode_rules.strict_skip.zero_delta",
        "trace_semantics.raw_state",
        "trace_semantics.raw_state_before_update",
    }
)


def _flatten(payload, prefix: str = "") -> dict:
    """把嵌套 contract 扁平化为 ``点分键 -> 标量``。"""

    flat: dict = {}
    for key, value in payload.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, path + "."))
        else:
            flat[path] = value
    return flat


def test_contract_differences_are_confined_to_labels_and_text(implementations) -> None:
    """两版 contract 的差异**必须**只限于版本标识与人类可读文案。

    实测（2026-09-24）：49 个扁平键中 21 个不同，全部落在两类——

    * **版本标识（2）**：``equation_version``、``time_policy.version``
      （``src-v3`` 带 ``V27`` 前缀）；
    * **文案（19）**：``src/`` 为中文、``src-v3/`` 为英文的规则说明与
      ``trace_semantics`` 描述。

    **零算法差异**——所有承载语义的键（``memory_state``、
    ``post_update_normalization``、``padding_calls_cell``、
    ``candidate_function``、``innovation_names``、各 ``mode_rules`` 的**结构**等）
    完全一致。

    ## 这条断言的实际意义

    它把"算法等价"从**结构证据**（AST 指纹）与**行为证据**（逐位输出比较）
    扩展到**契约层面**：只要没有新的差异键出现，就可以确认两版没有实质分歧。

    ## 需要注意的后果

    契约**序列化结果不等价**——同一份权重在两棵树下产生的
    ``liquid_v2_contract()`` payload 哈希不同。由于 contract 会写入 checkpoint，
    **跨树迁移 checkpoint 时契约哈希必然不匹配**。这不是缺陷，但需要知道：
    v28 绑定的是 ``src-v3`` 那一份，其 contract 带 ``V27`` 前缀与英文文案。
    """

    src_impl, v3_impl = implementations
    cell_a, cell_b = _build_pair(src_impl, v3_impl, "bounded_observation")

    flat_a = _flatten(src_impl.liquid_v2_contract(cell_a))
    flat_b = _flatten(v3_impl.liquid_v2_contract(cell_b))

    assert set(flat_a) == set(flat_b), (
        f"扁平键集合不同：{sorted(set(flat_a) ^ set(flat_b))}"
    )

    differing = sorted(k for k in flat_a if flat_a[k] != flat_b[k])
    unexpected = sorted(set(differing) - _ALLOWED_CONTRACT_DIFFERENCES)

    assert not unexpected, (
        "contract 出现**未登记**的差异键（可能存在实质分歧）：\n"
        + "\n".join(f"  {k}: {flat_a[k]!r} != {flat_b[k]!r}" for k in unexpected)
    )

    # 反向检查：登记为"应当不同"的键，确实仍在不同（避免登记表腐化）
    stale = sorted(k for k in _ALLOWED_CONTRACT_DIFFERENCES if k in flat_a and flat_a[k] == flat_b[k])
    assert not stale, (
        f"以下键已不再有差异，应从登记表中移除：{stale}"
    )

    # 所有差异值都必须是字符串（版本或文案），不应有结构/数值差异
    for key in differing:
        assert isinstance(flat_a[key], str) and isinstance(flat_b[key], str), (
            f"{key} 的差异不是字符串差异：{type(flat_a[key])} vs {type(flat_b[key])}"
        )


def test_version_labels_carry_tree_specific_prefix(implementations) -> None:
    """版本标识的差异必须是可预期的前缀差，而不是版本号本体不同。"""

    src_impl, v3_impl = implementations
    cell_a, cell_b = _build_pair(src_impl, v3_impl, "bounded_observation")

    contract_a = src_impl.liquid_v2_contract(cell_a)
    contract_b = v3_impl.liquid_v2_contract(cell_b)

    for key in ("equation_version",):
        value_a, value_b = contract_a[key], contract_b[key]
        assert value_a.replace("RADAR-LN3-", "") == value_b.replace("RADAR-LN3-V27-", ""), (
            f"{key} 的差异超出前缀范围：{value_a} vs {value_b}"
        )

    policy_a = contract_a["time_policy"]["version"]
    policy_b = contract_b["time_policy"]["version"]
    assert policy_a.replace("RADAR-LN3-", "") == policy_b.replace("RADAR-LN3-V27-", "")
    assert policy_b.startswith("RADAR-LN3-V27-"), (
        f"src-v3 的版本前缀已变化：{policy_b}"
    )
