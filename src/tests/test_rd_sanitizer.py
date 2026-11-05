from __future__ import annotations

import numpy as np
import pytest

from radar_ln3.data.contracts import DataFormatError, DataIntegrityError
from radar_ln3.data.rd_sanitizer import (
    RD_AMPLITUDE_REPAIR_POLICY,
    RD_AMPLITUDE_REPAIR_SCHEMA,
    sanitize_rd_amplitude_db,
)


def test_zero_complex_power_negative_inf_is_repaired_and_audited() -> None:
    result = sanitize_rd_amplitude_db(
        np.asarray([[-np.inf, 12.0], [3.0, -np.inf]], dtype=np.float64),
        np.asarray([[0.0 + 0.0j, 2.0 + 0.0j], [1.0j, 0.0 + 0.0j]]),
        name="fixture",
    )

    assert result.values.dtype == np.float32
    assert np.array_equal(
        result.values,
        np.asarray([[0.0, 12.0], [3.0, 0.0]], dtype=np.float32),
    )
    assert result.audit == {
        "schema_version": RD_AMPLITUDE_REPAIR_SCHEMA,
        "repair_policy": RD_AMPLITUDE_REPAIR_POLICY,
        "input_nonfinite_count": 2,
        "negative_inf_count": 2,
        "positive_inf_count": 0,
        "nan_count": 0,
        "zero_power_replaced_count": 2,
        "repaired_rows": [0, 1],
        "output_finite": True,
    }


@pytest.mark.parametrize(
    "amplitude",
    [
        np.asarray([[np.nan]], dtype=np.float32),
        np.asarray([[np.inf]], dtype=np.float32),
    ],
)
def test_unsupported_amplitude_nonfinite_values_fail_closed(amplitude: np.ndarray) -> None:
    with pytest.raises(DataIntegrityError):
        sanitize_rd_amplitude_db(
            amplitude,
            np.asarray([[1.0 + 0.0j]]),
            name="fixture",
        )


def test_negative_inf_with_nonzero_complex_power_fails() -> None:
    with pytest.raises(DataIntegrityError):
        sanitize_rd_amplitude_db(
            np.asarray([[-np.inf]], dtype=np.float32),
            np.asarray([[1.0 + 0.0j]]),
            name="fixture",
        )


def test_nonfinite_complex_input_fails() -> None:
    with pytest.raises(DataIntegrityError):
        sanitize_rd_amplitude_db(
            np.asarray([[-np.inf]], dtype=np.float32),
            np.asarray([[np.nan + 0.0j]]),
            name="fixture",
        )


def test_nonfinite_amplitude_without_complex_input_cannot_be_repaired() -> None:
    with pytest.raises(DataIntegrityError):
        sanitize_rd_amplitude_db(
            np.asarray([[-np.inf]], dtype=np.float32),
            None,
            name="fixture",
        )


def test_amplitude_and_complex_shapes_must_match() -> None:
    with pytest.raises(DataFormatError):
        sanitize_rd_amplitude_db(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 1), dtype=np.complex64),
            name="fixture",
        )
