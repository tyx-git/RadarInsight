# Unified Session Schema

## 1. Purpose

This document defines the **canonical session data format** used by RadarInsight.

The goal is to ensure that all radar data sources, including heterogeneous sources with different modality availability, are normalized into a single on-disk schema before training.

This schema is designed to support:

- the current `tr + rd` mainline
- session-level training and evaluation
- optional modality handling
- future expansion from 4-class to 7-class and beyond

The schema standardizes **data format**, not necessarily the physical origin of each field.

For example:

- some sessions contain native radar tracks
- some sessions contain pseudo-tracks derived from RD-only sequences

Both are allowed, as long as they are expressed through the same canonical file and field structure.

---

## 2. Directory Layout

Each session must be stored under:

`data/sessions/<session_id>/`

A canonical session directory contains:

- `meta_session.json`
- `labels/labels_matched.jsonl`
- `labels/labels_trackclip.jsonl`
- `qc/qc_report.json`
- `radar/track/track_data.csv`
- `radar/rd/rd_data.h5`
- `radar/rd/rd_clips.h5`
- `radar/rd/rd_alignment_diagnostics.jsonl` (optional but recommended)

Additional modality-specific files are allowed, but they must not replace the canonical files above.

---

## 3. Core Semantics

### 3.1 Session

A session is the top-level training / evaluation unit.

A session may originate from:

- one native track + aligned RD sequence
- one sequence + one instance + one sub-session window
- one RD-only source plus a pseudo-track generated in preprocessing

### 3.2 Track

The mainline assumes the presence of a `track-like` sequence under:

`radar/track/track_data.csv`

This may be:

- native track
- pseudo-track derived from RD

The format must remain the same in either case.

### 3.3 RD

The mainline assumes the presence of RD sequence data under:

- `radar/rd/rd_data.h5`
- `radar/rd/rd_clips.h5`

---

## 4. Required Files

### 4.1 `meta_session.json`

Stores session-level metadata.

Required fields:

- `session_id`
- `source_name`
- `label_name`
- `has_track`
- `has_rd`
- `has_points`
- `track_source`
- `has_native_track`

Recommended fields:

- `sequence_id`
- `instance_id`
- `parent_instance_id`
- `sub_session_index`
- `split_name`
- `num_track_rows`
- `num_rd_frames`
- `rd_shape`
- `frame_window`
- source-specific provenance fields

### 4.2 `labels/labels_matched.jsonl`

Stores the session-level target label.

Minimum structure per line:

- `track_id`
- `label_main`
- `confidence`
- `evidence`

The current practice is one line per session.

### 4.3 `labels/labels_trackclip.jsonl`

Stores clip-level sub-window definitions used by the training pipeline.

Minimum structure per line:

- `sample_id`
- `type`
- `t0`
- `t1`
- `track_id`
- `num_frames`
- `mean_range`
- `mean_speed`

### 4.4 `qc/qc_report.json`

Stores session QC summary.

Minimum fields:

- `status`
- `track_found`
- `rd_found`
- `duration_s`
- `frame_loss_count`
- `rd_frame_count`
- `rd_shape`
- `num_tracks`
- `errors`

### 4.5 `radar/track/track_data.csv`

Stores the canonical track-like sequence.

Required columns:

- `t`
- `frame_id`
- `track_id`
- `range_m`
- `azimuth_deg`
- `radial_speed_mps`
- `snr_db`
- `elevation_deg`

Recommended extra columns:

- `track_quality`
- `track_source`
- `point_count`
- `amplitude_like`

Notes:

- `t` must be **relative time in seconds**, monotonically increasing for a valid track window.
- `track_source` should explicitly indicate whether the track is native or pseudo.
- pseudo-track fields must use the same column names even when they are only track-like approximations.

### 4.6 `radar/rd/rd_data.h5`

Stores the full RD sequence for the session.

Required datasets:

- `rd_frames`
- `range_axis_m`
- `doppler_axis_mps`

Expected shape:

- `rd_frames`: `(num_frames, range_bins, doppler_bins)`

The current pipeline accepts float or complex RD representations depending on source.

### 4.7 `radar/rd/rd_clips.h5`

Stores RD clips extracted from `labels_trackclip.jsonl`.

Expected behavior:

- one dataset per `sample_id`
- each dataset stores the cropped local RD clip sequence corresponding to one track clip

### 4.8 `radar/rd/rd_alignment_diagnostics.jsonl`

Optional but recommended.

Stores clip-level alignment diagnostics used for debugging and later error analysis.

---

## 5. Unified Field Policy

### 5.1 Format is unified, source semantics may differ

Example:

- native source: `range_m` is true radar-track range
- pseudo source: `range_m` is range-like coordinate derived from RD-side tracking logic

This is allowed.

The schema standardizes:

- file layout
- field names
- expected tensor organization

It does **not** require all sources to have identical physical meaning at every field level.

### 5.2 Source provenance must remain explicit

Every source converted into this schema must preserve provenance through metadata such as:

- `source_name`
- `track_source`
- `has_native_track`
- `track_quality`

This prevents pseudo-track data from being silently treated as native track without traceability.

---

## 6. Source Mapping Policy

### 6.1 Legacy paired sessions

- `has_track = true`
- `has_rd = true`
- `track_source = native`

### 6.2 Carrada

- `has_track = true`
- `has_rd = true`
- `track_source = pseudo_from_rd`
- `has_native_track = false`

Current label mapping:

- `1 -> PeopleTarget`
- `2 -> CyclistTarget`
- `3 -> CarTarget`

### 6.3 CQ-08

Planned mapping once preprocessing is complete:

- `1 / 2 -> DroneTarget`
- `3 -> BirdTarget`
- `4 -> BalloonTarget`
- `5 -> ClutterTarget`
- `6 -> OtherTarget`

CQ-08 should be converted into the same schema after the `.mat` preprocessing stage is fully ready and explicitly approved for training integration.

---

## 7. Sessionization Policy

### 7.1 Native paired sources

A session may directly correspond to a native labeled track / RD pair.

### 7.2 Carrada

Current policy:

- one `sequence + instance + sub-session window` = one session

Sub-session splitting uses mixed windows by default.

Current default mixed windows:

- `128:64`
- `64:32`
- `192:96`

All sub-sessions from the same parent instance must remain grouped for split assignment to avoid leakage.

---

## 8. Split Integrity Rules

To avoid train/validation/test leakage:

- all sub-sessions from the same parent instance must stay in the same split
- source-specific split metadata should be preserved when available
- split logic must be applied at the parent entity level, not only at the individual clip level

---

## 9. Training Compatibility Rules

A session is considered train-compatible when at least the following are valid:

- `meta_session.json` exists
- `labels_matched.jsonl` exists
- `labels_trackclip.jsonl` exists and is not empty
- `track_data.csv` exists and has required columns
- `rd_data.h5` exists
- `rd_clips.h5` exists
- `qc_report.json` reports a valid session state

Sessions with zero generated clips should be dropped before synchronization into the training root.

---

## 10. Recommended Future Extensions

The following are recommended but not required for the current mainline:

- explicit `source_name`-aware training sampler
- explicit `track_quality` feature usage in the model
- optional `points` branch for sources that provide point-track data
- adaptive track-like completion layer in `pro/processing`
- hierarchical label support for larger taxonomies

---

## 11. Summary

The long-term rule is:

- keep the **session data format unified**
- allow **label space expansion**
- allow **native and pseudo track sources**
- keep the mainline structured as `tr + rd`

This schema is the contract that all future preprocessing pipelines should satisfy before data enters training.
