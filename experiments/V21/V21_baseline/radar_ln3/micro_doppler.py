from __future__ import annotations

import struct
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .runtime import preflight_npz_archive
from scipy.signal.windows import taylor


MICRO_DOPPLER_NAMES = (
    "peak_band_energy_time_cv_clipped_0_2_over_2",
    "minus_10db_velocity_spread_ratio",
    "modulation_period_seconds_clipped_0_1s_over_0_1s",
)
MICRO_DOPPLER_VALUE_DIM = len(MICRO_DOPPLER_NAMES)

FRAME_HEAD = 0xFA55FA55
FRAME_END = 0x55FA55FA
RANGE_BIN_COUNT = 31
CQ08_MAT_FRAME_PATTERN = re.compile(
    r"^(?P<capture_id>\d+)_DAUR_RD_(?P<label_text>[^_]+)_"
    r"(?P<frame_number>\d+)_(?P<track_id>\d+)\.mat$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Cq08MatFrameIdentity:
    capture_id: str
    label_text: str
    track_id: int
    frame_number_one_based: int
    raw_dat_frame_index_zero_based: int


def parse_cq08_mat_frame_identity(path: str | Path) -> Cq08MatFrameIdentity:
    """Recover the acquisition frame identity encoded in a CQ08 MAT filename.

    ``rd_raw_frame_indices_json`` in the V16/V3 point index is a legacy MAT-list
    ordinal.  It is not a raw ``.dat`` frame number when one or more leading MAT
    files are absent.  The acquisition frame number embedded in the MAT filename
    is one-based and maps to the zero-based frame order returned by
    :func:`parse_cq08_raw_echo`.
    """

    name = Path(path).name
    matched = CQ08_MAT_FRAME_PATTERN.fullmatch(name)
    if matched is None:
        raise ValueError(f"CQ08 MAT filename does not encode a frame identity: {name}")
    frame_number = int(matched.group("frame_number"))
    if frame_number <= 0:
        raise ValueError(f"CQ08 MAT acquisition frame number must be one-based: {name}")
    return Cq08MatFrameIdentity(
        capture_id=matched.group("capture_id"),
        label_text=str(matched.group("label_text")),
        track_id=int(matched.group("track_id")),
        frame_number_one_based=frame_number,
        raw_dat_frame_index_zero_based=frame_number - 1,
    )


@dataclass(frozen=True)
class RawTargetRecord:
    raw_target_identifier: int
    track_point_index: int
    range_bin_raw: int
    doppler_bin_raw: int


@dataclass(frozen=True)
class RawEchoFrame:
    raw_frame_index: int
    frequency_hz: float
    cpi_count: int
    prt_num: int
    prt_seconds: float
    target_records: tuple[RawTargetRecord, ...]
    iq: np.ndarray






def _unpack_u32(payload: bytes, offset: int, count: int) -> tuple[int, ...]:
    return struct.unpack_from(f"<{count}I", payload, offset)


def parse_cq08_raw_echo(path: str | Path) -> list[RawEchoFrame]:
    """Strictly parse one CQ08 raw echo file from byte zero to EOF.

    The parser retains frame order, current-frame target records and complex
    slow-time I/Q. It does not use a label, split, model output or any event
    outside the caller-selected current event.
    """

    raw_path = Path(path)
    payload = raw_path.read_bytes()
    frames: list[RawEchoFrame] = []
    offset = 0
    while offset < len(payload):
        if len(payload) - offset < 16:
            raise ValueError(f"truncated CQ08 raw frame header at byte offset {offset}")
        head, length_words, _scan_angle, point_count = _unpack_u32(payload, offset, 4)
        if head != FRAME_HEAD:
            raise ValueError(f"non-contiguous CQ08 raw frame at byte offset {offset}")
        frame_length = int(length_words) * 4
        if frame_length < 36 or offset + frame_length > len(payload):
            raise ValueError(f"invalid CQ08 raw frame length at byte offset {offset}")
        footer_offset = offset + frame_length - 4
        if _unpack_u32(payload, footer_offset, 1)[0] != FRAME_END:
            raise ValueError(f"invalid CQ08 raw frame footer at byte offset {footer_offset}")
        if point_count > 100_000:
            raise ValueError(f"implausible CQ08 target-record count at byte offset {offset}")
        target_offset = offset + 16
        target_words = _unpack_u32(payload, target_offset, int(point_count) * 4)
        metadata_offset = target_offset + int(point_count) * 16
        frequency_mhz, cpi_count, prt_num, prt_raw, _data_length = _unpack_u32(
            payload, metadata_offset, 5
        )
        if prt_num <= 0 or prt_num > 1_000_000 or frequency_mhz <= 0 or prt_raw <= 0:
            raise ValueError(f"invalid CQ08 radar metadata at byte offset {offset}")
        sample_count = int(prt_num) * RANGE_BIN_COUNT * 2
        sample_offset = metadata_offset + 20
        sample_bytes = sample_count * 4
        if sample_offset + sample_bytes > footer_offset:
            raise ValueError(f"CQ08 raw I/Q payload crosses frame boundary at byte offset {offset}")
        samples = np.frombuffer(payload, dtype="<f4", count=sample_count, offset=sample_offset)
        iq = (samples[0::2] + 1j * samples[1::2]).reshape(
            (RANGE_BIN_COUNT, int(prt_num)), order="F"
        )
        if not np.isfinite(iq).all():
            raise ValueError(f"CQ08 raw I/Q contains a non-finite value at byte offset {offset}")
        records = tuple(
            RawTargetRecord(
                raw_target_identifier=int(target_words[index * 4]),
                track_point_index=int(target_words[index * 4 + 1]),
                range_bin_raw=int(target_words[index * 4 + 2]),
                doppler_bin_raw=int(target_words[index * 4 + 3]),
            )
            for index in range(int(point_count))
        )
        frames.append(
            RawEchoFrame(
                raw_frame_index=len(frames),
                frequency_hz=float(frequency_mhz) * 1.0e6,
                cpi_count=int(cpi_count),
                prt_num=int(prt_num),
                prt_seconds=float(prt_raw) * 0.0125e-6,
                target_records=records,
                iq=np.asarray(iq, dtype=np.complex64),
            )
        )
        offset += frame_length
    if offset != len(payload) or not frames:
        raise ValueError("CQ08 raw echo did not parse contiguously to a non-empty frame sequence")
    return frames


def _target_record(frame: RawEchoFrame, track_point_index: int) -> RawTargetRecord:
    matches = [value for value in frame.target_records if value.track_point_index == int(track_point_index)]
    if len(matches) != 1:
        raise ValueError(
            "current raw frame must contain exactly one target record for the selected track point: "
            f"raw_frame_index={frame.raw_frame_index}, track_point_index={track_point_index}, matches={len(matches)}"
        )
    return matches[0]


def _target_location(frame: RawEchoFrame, record: RawTargetRecord) -> tuple[int, int]:
    """Reproduce the existing CQ08 local MTD target search without a label."""

    pulse_count = int(frame.prt_num)
    window = taylor(pulse_count, nbar=4, sll=30.0, norm=True, sym=True).astype(np.float64)
    rd = np.fft.fftshift(
        np.fft.fft(np.asarray(frame.iq, dtype=np.complex128) * window[None, :], axis=1),
        axes=1,
    )
    # The raw Doppler record and MATLAB processing are one-based. Preserve the
    # frozen mapping used by batch_process_all.m before the +/-5 local search.
    raw_bin = int(record.doppler_bin_raw)
    mapped_one_based = raw_bin - pulse_count // 2 if raw_bin > pulse_count / 2 else raw_bin + pulse_count // 2
    mapped_zero_based = int(np.clip(mapped_one_based - 1, 0, pulse_count - 1))
    range_start = max(0, 15 - 5)
    range_stop = min(RANGE_BIN_COUNT, 15 + 5 + 1)
    doppler_start = max(0, mapped_zero_based - 5)
    doppler_stop = min(pulse_count, mapped_zero_based + 5 + 1)
    local = np.abs(rd[range_start:range_stop, doppler_start:doppler_stop])
    if local.size == 0 or not np.isfinite(local).all():
        raise ValueError("CQ08 target-local MTD search produced an invalid plane")
    local_row, local_column = np.unravel_index(int(np.argmax(local)), local.shape)
    return range_start + int(local_row), doppler_start + int(local_column)














def load_micro_doppler_cache(
    path: str | Path,
    *,
    expected_sample_ids: Iterable[str] | None = None,
    expected_value_dim: int = MICRO_DOPPLER_VALUE_DIM,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_path = Path(path)
    preflight_npz_archive(cache_path)
    with np.load(cache_path, allow_pickle=False) as payload:
        required = {"sample_ids", "values", "valid"}
        if set(payload.files) != required:
            raise ValueError(f"micro-Doppler cache fields must be exactly {sorted(required)}")
        sample_ids = np.asarray(payload["sample_ids"])
        values = np.asarray(payload["values"])
        valid = np.asarray(payload["valid"])
    if sample_ids.ndim != 1 or sample_ids.dtype.kind not in {"U", "S"}:
        raise ValueError("micro-Doppler cache sample_ids must be a one-dimensional string array")
    identifiers = [str(value) for value in sample_ids.tolist()]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("micro-Doppler cache contains duplicate sample_ids")
    expected_dim = int(expected_value_dim)
    if expected_dim <= 0:
        raise ValueError("micro-Doppler cache expected value dimension must be positive")
    if values.shape != (len(identifiers), expected_dim) or values.dtype != np.float32:
        raise ValueError(
            "micro-Doppler cache values must be float32 with the configured shape: "
            f"expected={[len(identifiers), expected_dim]}, actual={list(values.shape)}"
        )
    if valid.shape != (len(identifiers),) or valid.dtype != np.bool_:
        raise ValueError("micro-Doppler cache valid must be bool with shape [N]")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("micro-Doppler cache values must be finite and bounded to [0,1]")
    if np.any(values[~valid] != 0.0):
        raise ValueError("invalid micro-Doppler cache rows must be exactly zero")
    if expected_sample_ids is not None:
        expected = [str(value) for value in expected_sample_ids]
        if len(expected) != len(set(expected)):
            raise ValueError("expected micro-Doppler sample_ids contain duplicates")
        missing = sorted(set(expected) - set(identifiers))
        extra = sorted(set(identifiers) - set(expected))
        if missing or extra or len(expected) != len(identifiers):
            raise ValueError(
                "micro-Doppler cache sample set does not exactly match the Dataset split: "
                f"missing={missing[:5]}, extra={extra[:5]}, expected={len(expected)}, actual={len(identifiers)}"
            )
    mapping = {sample_id: values[index] for index, sample_id in enumerate(identifiers)}
    valid_mapping = {sample_id: bool(valid[index]) for index, sample_id in enumerate(identifiers)}
    return mapping, {
        "path": str(cache_path),
        "event_count": len(identifiers),
        "valid_event_count": int(valid.sum()),
        "invalid_event_count": int((~valid).sum()),
        "valid_by_sample_id": valid_mapping,
    }




__all__ = ["CQ08_MAT_FRAME_PATTERN", "Cq08MatFrameIdentity", "MICRO_DOPPLER_NAMES", "MICRO_DOPPLER_VALUE_DIM", "RawEchoFrame", "RawTargetRecord", "load_micro_doppler_cache", "parse_cq08_raw_echo", "parse_cq08_mat_frame_identity"]