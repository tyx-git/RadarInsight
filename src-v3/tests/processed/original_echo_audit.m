function outputs = original_echo_audit(projectRoot)
%ORIGINAL_ECHO_AUDIT Read-only audit of twelve CQ08 raw echo files.
%
% This audit intentionally does not call the legacy batch writer because it
% writes generated MAT products.  It reuses the verified frame layout in that
% file, parses raw frames in memory, and compares only selected frames with
% the existing MAT/RD products listed by cq08_unified_manifest.json.
%
% Usage:
%   cd('F:\Project\RadarInsight');
%   addpath('F:\Project\RadarInsight\src-v3\tests\processed');
%   original_echo_audit;
%
% Optional:
%   original_echo_audit('F:\Project\RadarInsight');

if nargin < 1 || isempty(projectRoot)
    thisFile = mfilename('fullpath');
    projectRoot = fileparts(fileparts(fileparts(fileparts(thisFile))));
end
projectRoot = char(projectRoot);

reportDir = fullfile(projectRoot, 'src-v3', 'processed', 'report');
if ~exist(reportDir, 'dir')
    mkdir(reportDir);
end

manifestPath = fullfile(projectRoot, 'data', 'manifests', 'cq08_unified_manifest.json');
if ~exist(manifestPath, 'file')
    error('CQ08 manifest does not exist: %s', manifestPath);
end

fprintf('[CQ08 audit] project=%s\n', projectRoot);
fprintf('[CQ08 audit] manifest=%s\n', manifestPath);
fprintf('[CQ08 audit] report_dir=%s\n', reportDir);

manifest = jsondecode(fileread(manifestPath));
records = normalizeManifestRecords(manifest);
if isempty(records)
    error('No usable CQ08 records were found in the manifest.');
end

[records, originalOrder] = sortRecordsByTrack(records); %#ok<ASGLU>
[selectedIndices, selectedRoles] = selectAuditRecords(records, 12);
if numel(selectedIndices) ~= min(12, numel(records))
    error('The audit selector did not produce the requested number of samples.');
end

sampleRows = repmat(emptySampleRow(), 0, 1);
frameRows = repmat(emptyFrameRow(), 0, 1);
rdRows = repmat(emptyRdRow(), 0, 1);
selectionRows = repmat(emptySelectionRow(), 0, 1);

allRawFrameCountsMatch = true;
allIndexMappingsMatch = true;
allRawMatDataComparable = true;
allRdRecomputationsComparable = true;
wallClockFieldDetected = false;
matWallClockFieldDetected = false;
rawComplexIqDetected = false;

for selectedPosition = 1:numel(selectedIndices)
    record = records(selectedIndices(selectedPosition));
    roles = selectedRoles{selectedPosition};
    trackId = scalarNumber(fieldValue(record, 'track_id', NaN));
    labelName = asChar(fieldValue(record, 'label_name', ''));
    rawPath = resolveManifestPath(fieldValue(record, 'raw_echo_file', ''), projectRoot);
    pointPath = resolveManifestPath(fieldValue(record, 'point_file', ''), projectRoot);
    trackPath = resolveManifestPath(fieldValue(record, 'track_file', ''), projectRoot);
    matPaths = resolveMatPaths(fieldValue(record, 'rd_mat_files', {}), projectRoot);

    selectionRows(end + 1, 1) = makeSelectionRow( ...
        selectedPosition, trackId, labelName, roles, rawPath, ...
        scalarNumber(fieldValue(record, 'track_length_hint', NaN)), numel(matPaths)); %#ok<AGROW>

    fprintf('[CQ08 audit] sample %02d/%02d track=%g label=%s raw=%s\n', ...
        selectedPosition, numel(selectedIndices), trackId, labelName, rawPath);

    pointInfo = inspectTextTrack(pointPath);
    trackInfo = inspectTextTrack(trackPath);
    expectedMatTargets = comparisonIndices(max(1, numel(matPaths)));

    sampleStatus = emptySampleRow();
    sampleStatus.sample_order = selectedPosition;
    sampleStatus.track_id = trackId;
    sampleStatus.label_name = labelName;
    sampleStatus.selection_roles = roles;
    sampleStatus.raw_echo_path = rawPath;
    sampleStatus.point_path = pointPath;
    sampleStatus.track_path = trackPath;
    sampleStatus.manifest_mat_count = numel(matPaths);
    sampleStatus.point_row_count = pointInfo.data_row_count;
    sampleStatus.track_row_count = trackInfo.data_row_count;
    sampleStatus.point_first_time = pointInfo.first_time;
    sampleStatus.point_last_time = pointInfo.last_time;
    sampleStatus.track_first_time = trackInfo.first_time;
    sampleStatus.track_last_time = trackInfo.last_time;
    sampleStatus.raw_wall_clock_time_field_detected = false;
    sampleStatus.parser = 'verified batch_process_all.m frame layout; read-only reimplementation';

    if ~exist(rawPath, 'file')
        sampleStatus.status = 'raw_file_missing';
        sampleRows(end + 1, 1) = sampleStatus; %#ok<AGROW>
        allRawFrameCountsMatch = false;
        continue;
    end

    rawAudit = parseRawFile(rawPath, expectedMatTargets);
    if rawAudit.frame_count ~= numel(matPaths)
        % Reparse only this selected file so the true raw middle frame is
        % available for comparison when the MAT count is different.
        actualTargets = unique([ ...
            comparisonIndices(max(1, rawAudit.frame_count)), ...
            comparisonIndices(max(1, numel(matPaths)))]);
        rawAudit = parseRawFile(rawPath, actualTargets);
    end

    rawComplexIqDetected = rawComplexIqDetected || rawAudit.complex_iq_detected;
    wallClockFieldDetected = wallClockFieldDetected || rawAudit.wall_clock_field_detected;
    sampleStatus.raw_frame_count = rawAudit.frame_count;
    sampleStatus.raw_valid_frame_count = rawAudit.valid_frame_count;
    sampleStatus.raw_invalid_candidate_count = rawAudit.invalid_candidate_count;
    sampleStatus.raw_file_bytes = rawAudit.file_bytes;
    sampleStatus.raw_frame_length_min_bytes = rawAudit.frame_length_min_bytes;
    sampleStatus.raw_frame_length_max_bytes = rawAudit.frame_length_max_bytes;
    sampleStatus.raw_iq_shape_consistent = rawAudit.iq_shape_consistent;
    sampleStatus.raw_complex_iq_detected = rawAudit.complex_iq_detected;
    sampleStatus.raw_wall_clock_time_field_detected = rawAudit.wall_clock_field_detected;
    sampleStatus.raw_cpi_count_min = rawAudit.cpi_count_min;
    sampleStatus.raw_cpi_count_max = rawAudit.cpi_count_max;
    sampleStatus.raw_prt_num_min = rawAudit.prt_num_min;
    sampleStatus.raw_prt_num_max = rawAudit.prt_num_max;
    sampleStatus.raw_prt_seconds_min = rawAudit.prt_seconds_min;
    sampleStatus.raw_prt_seconds_max = rawAudit.prt_seconds_max;
    sampleStatus.raw_observation_seconds_min = rawAudit.observation_seconds_min;
    sampleStatus.raw_observation_seconds_max = rawAudit.observation_seconds_max;
    sampleStatus.raw_unparsed_payload_bytes_max = rawAudit.unparsed_payload_bytes_max;
    sampleStatus.raw_hidden_payload_detected = rawAudit.unparsed_payload_bytes_max > 0;

    for frameIndex = 1:numel(rawAudit.frames)
        frameRows(end + 1, 1) = makeFrameRow( ...
            selectedPosition, trackId, labelName, rawAudit.frames(frameIndex), ...
            pointInfo, trackInfo); %#ok<AGROW>
    end

    matAudit = inspectMatFiles(matPaths, pointInfo, trackInfo, rawAudit);
    sampleStatus.mat_existing_file_count = matAudit.existing_file_count;
    sampleStatus.mat_missing_file_count = matAudit.missing_file_count;
    sampleStatus.mat_index_in_range = matAudit.index_in_range;
    sampleStatus.mat_filename_order_consistent = matAudit.filename_order_consistent;
    sampleStatus.raw_track_point_index_matches_mat = matAudit.raw_track_point_index_matches_mat;
    sampleStatus.raw_clamped_track_point_index_matches_mat = matAudit.raw_clamped_index_matches_mat;
    sampleStatus.raw_mat_data_out_comparable = matAudit.data_out_comparable;
    sampleStatus.raw_mat_rd_comparable = matAudit.rd_comparable;
    sampleStatus.rd_recompute_max_relative_error = matAudit.rd_max_relative_error;
    sampleStatus.rd_recompute_max_absolute_error = matAudit.rd_max_absolute_error;
    sampleStatus.mat_observation_duration_min = matAudit.observation_duration_min;
    sampleStatus.mat_observation_duration_max = matAudit.observation_duration_max;
    sampleStatus.mat_explicit_wall_clock_field_detected = matAudit.explicit_wall_clock_field_detected;
    sampleStatus.point_index_time_coverage = matAudit.point_index_time_coverage;
    sampleStatus.track_index_time_coverage = matAudit.track_index_time_coverage;
    sampleStatus.status = matAudit.status;
    matWallClockFieldDetected = matWallClockFieldDetected || matAudit.explicit_wall_clock_field_detected;

    allRawFrameCountsMatch = allRawFrameCountsMatch && (rawAudit.frame_count == numel(matPaths));
    allIndexMappingsMatch = allIndexMappingsMatch && matAudit.raw_clamped_index_matches_mat;
    allRawMatDataComparable = allRawMatDataComparable && matAudit.data_out_comparable;
    allRdRecomputationsComparable = allRdRecomputationsComparable && matAudit.rd_comparable;

    for rdIndex = 1:numel(matAudit.comparisons)
        rdRows(end + 1, 1) = makeRdRow( ...
            selectedPosition, trackId, labelName, matAudit.comparisons(rdIndex)); %#ok<AGROW>
    end
    sampleRows(end + 1, 1) = sampleStatus; %#ok<AGROW>
end

report = buildReport(projectRoot, manifestPath, selectedIndices, records, ...
    selectionRows, sampleRows, frameRows, rdRows, ...
    allRawFrameCountsMatch, allIndexMappingsMatch, allRawMatDataComparable, ...
    allRdRecomputationsComparable, rawComplexIqDetected, ...
    wallClockFieldDetected, matWallClockFieldDetected);

summaryPath = fullfile(reportDir, 'RADAR-LN3-original-echo-audit-summary.json');
samplesPath = fullfile(reportDir, 'RADAR-LN3-original-echo-audit-samples.csv');
framesPath = fullfile(reportDir, 'RADAR-LN3-original-echo-audit-frames.csv');
rdPath = fullfile(reportDir, 'RADAR-LN3-original-echo-audit-rd-comparison.csv');
selectionPath = fullfile(reportDir, 'RADAR-LN3-original-echo-audit-selection.csv');

writeJson(summaryPath, report);
writeTableCsv(samplesPath, sampleRows);
writeTableCsv(framesPath, frameRows);
writeTableCsv(rdPath, rdRows);
writeTableCsv(selectionPath, selectionRows);

outputs = struct();
outputs.status = 'completed';
outputs.summary_path = summaryPath;
outputs.samples_path = samplesPath;
outputs.frames_path = framesPath;
outputs.rd_comparison_path = rdPath;
outputs.selection_path = selectionPath;
outputs.sample_count = numel(selectedIndices);
outputs.raw_frame_count_match = allRawFrameCountsMatch;
outputs.raw_mat_data_out_comparable = allRawMatDataComparable;
outputs.raw_mat_rd_comparable = allRdRecomputationsComparable;
fprintf('[CQ08 audit] completed; summary=%s\n', summaryPath);
end

function records = normalizeManifestRecords(manifest)
if iscell(manifest)
    records = [manifest{:}];
else
    records = manifest;
end
if isempty(records)
    return;
end
usable = true(1, numel(records));
for index = 1:numel(records)
    value = fieldValue(records(index), 'is_usable', true);
    if islogical(value)
        usable(index) = value;
    elseif isnumeric(value)
        usable(index) = value ~= 0;
    end
end
records = records(usable);
end

function [records, originalOrder] = sortRecordsByTrack(records)
trackIds = zeros(1, numel(records));
for index = 1:numel(records)
    trackIds(index) = scalarNumber(fieldValue(records(index), 'track_id', index));
end
[~, originalOrder] = sort(trackIds);
records = records(originalOrder);
end

function [selectedIndices, roles] = selectAuditRecords(records, requestedCount)
recordCount = numel(records);
requestedCount = min(requestedCount, recordCount);
selectedIndices = zeros(1, 0);
roles = cell(1, 0);

% Boundary records are selected first so the report always has an explicit
% first/last track-id boundary, independent of label distribution.
[selectedIndices, roles] = addSelection(selectedIndices, roles, 1, 'first_track_boundary');
[selectedIndices, roles] = addSelection(selectedIndices, roles, recordCount, 'last_track_boundary');

lengths = zeros(1, recordCount);
for index = 1:recordCount
    lengths(index) = scalarNumber(fieldValue(records(index), 'track_length_hint', NaN));
end
finiteLengths = lengths(isfinite(lengths));
if isempty(finiteLengths)
    finiteLengths = ones(1, recordCount);
    lengths(:) = 1;
end
medianLength = median(finiteLengths);

labelPriority = {'DroneTarget', 'BirdTarget', 'BalloonTarget', ...
    'ClutterTarget', 'UnknownTarget'};
labelNames = cell(1, recordCount);
for index = 1:recordCount
    labelNames{index} = asChar(fieldValue(records(index), 'label_name', ''));
end
allLabels = unique(labelNames, 'stable');
labelPriority = [labelPriority, allLabels];
for labelIndex = 1:numel(labelPriority)
    label = labelPriority{labelIndex};
    candidates = find(strcmp(labelNames, label));
    if isempty(candidates)
        continue;
    end
    [~, nearest] = min(abs(lengths(candidates) - medianLength));
    [selectedIndices, roles] = addSelection( ...
        selectedIndices, roles, candidates(nearest), ['label_' label]);
    if numel(selectedIndices) >= requestedCount
        break;
    end
end

% Global length coverage: shortest, median, longest, then two quartiles.
[~, shortest] = min(lengths);
[~, medianIndex] = min(abs(lengths - medianLength));
[~, longest] = max(lengths);
lengthTargets = [shortest, medianIndex, longest];
for targetIndex = 1:numel(lengthTargets)
    [selectedIndices, roles] = addSelection( ...
        selectedIndices, roles, lengthTargets(targetIndex), ...
        sprintf('global_length_%d', targetIndex));
end
sortedLengths = sort(finiteLengths);
quantileTargets = [sortedLengths(max(1, ceil(0.25 * numel(sortedLengths)))), ...
    sortedLengths(max(1, ceil(0.75 * numel(sortedLengths))))];
for targetIndex = 1:numel(quantileTargets)
    [~, candidate] = min(abs(lengths - quantileTargets(targetIndex)));
    [selectedIndices, roles] = addSelection( ...
        selectedIndices, roles, candidate, sprintf('length_quantile_%d', targetIndex));
end

% Deterministic rank fill guarantees exactly requestedCount when the above
% semantic selectors happen to collapse onto the same parent.
rankCandidates = unique(round(linspace(1, recordCount, max(requestedCount, 2))));
for index = 1:numel(rankCandidates)
    [selectedIndices, roles] = addSelection( ...
        selectedIndices, roles, rankCandidates(index), 'deterministic_rank_fill');
end
for index = 1:recordCount
    if numel(selectedIndices) >= requestedCount
        break;
    end
    [selectedIndices, roles] = addSelection( ...
        selectedIndices, roles, index, 'deterministic_sequential_fill');
end

selectedIndices = selectedIndices(1:requestedCount);
roles = roles(1:requestedCount);
end

function [selectedIndices, roles] = addSelection(selectedIndices, roles, index, role)
existing = find(selectedIndices == index, 1);
if isempty(existing)
    selectedIndices(end + 1) = index;
    roles{end + 1} = role;
else
    roles{existing} = [roles{existing} '|' role];
end
end

function paths = resolveMatPaths(value, projectRoot)
if isempty(value)
    paths = cell(1, 0);
elseif iscell(value)
    paths = cell(1, numel(value));
    for index = 1:numel(value)
        paths{index} = resolveManifestPath(value{index}, projectRoot);
    end
elseif ischar(value) || isstring(value)
    paths = {resolveManifestPath(value, projectRoot)};
else
    paths = cell(1, 0);
end
end

function path = resolveManifestPath(value, projectRoot)
path = asChar(value);
if isempty(path)
    return;
end
if exist(path, 'file')
    return;
end
% The manifest is portable only when its paths are relative.  Preserve an
% existing absolute path in the report even if it is unavailable here.
if ~isAbsolutePath(path)
    path = fullfile(projectRoot, path);
end
end

function result = parseRawFile(datPath, targetIndices)
fileInfo = dir(datPath);
result = emptyRawAudit();
result.file_bytes = fileInfo.bytes;
result.target_indices = unique(targetIndices(:).');
fid = fopen(datPath, 'r', 'ieee-le');
if fid == -1
    error('Unable to open raw echo file: %s', datPath);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

frameHead = uint32(hex2dec('FA55FA55'));
frameEnd = uint32(hex2dec('55FA55FA'));
frames = repmat(emptyFrameInternal(), 0, 1);
examples = repmat(emptyExample(), 0, 1);
invalidCandidates = 0;
while true
    [frame, status] = parseNextFrame(fid, result.file_bytes, frameHead, frameEnd, result.target_indices);
    if strcmp(status, 'eof')
        break;
    elseif strcmp(status, 'invalid_candidate')
        invalidCandidates = invalidCandidates + 1;
        continue;
    elseif strcmp(status, 'ok')
        frameNumber = numel(frames) + 1;
        frame.metadata.raw_frame_index = frameNumber;
        frames(end + 1, 1) = frame.metadata; %#ok<AGROW>
        if any(result.target_indices == frameNumber)
            frame.example.frame_index = frameNumber;
            examples(end + 1, 1) = frame.example; %#ok<AGROW>
        end
    else
        error('Unexpected raw parser status: %s', status);
    end
end

result.frames = frames;
result.examples = examples;
result.frame_count = numel(frames);
result.valid_frame_count = numel(frames);
result.invalid_candidate_count = invalidCandidates;
if isempty(frames)
    return;
end

frameLengths = [frames.frame_length_bytes];
cpiCounts = [frames.cpi_count];
prtNums = [frames.prt_num];
prtSeconds = [frames.prt_seconds];
durations = [frames.observation_seconds];
result.frame_length_min_bytes = min(frameLengths);
result.frame_length_max_bytes = max(frameLengths);
result.iq_shape_consistent = all([frames.iq_rows] == frames(1).iq_rows) && ...
    all([frames.iq_cols] == frames(1).iq_cols);
result.complex_iq_detected = all([frames.iq_complex]);
result.wall_clock_field_detected = false;
result.cpi_count_min = min(cpiCounts);
result.cpi_count_max = max(cpiCounts);
result.prt_num_min = min(prtNums);
result.prt_num_max = max(prtNums);
result.prt_seconds_min = min(prtSeconds);
result.prt_seconds_max = max(prtSeconds);
    result.observation_seconds_min = min(durations);
    result.observation_seconds_max = max(durations);
    unparsedPayload = [frames.unparsed_payload_bytes];
    result.unparsed_payload_bytes_max = max(unparsedPayload);
end

function [frame, status] = parseNextFrame(fid, fileBytes, frameHead, frameEnd, targetIndices)
frame = struct('metadata', emptyFrameInternal(), 'example', emptyExample());
status = 'eof';
while true
    candidateOffset = ftell(fid);
    if candidateOffset < 0 || candidateOffset >= fileBytes
        status = 'eof';
        return;
    end
    head = fread(fid, 1, 'uint32=>uint32');
    if isempty(head)
        status = 'eof';
        return;
    end
    if head ~= frameHead
        fseek(fid, candidateOffset + 1, 'bof');
        continue;
    end

    lengthWords = fread(fid, 1, 'uint32=>uint32');
    if isempty(lengthWords)
        status = 'eof';
        return;
    end
    frameLengthBytes = double(lengthWords) * 4;
    if frameLengthBytes < 32 || candidateOffset + frameLengthBytes > fileBytes
        fseek(fid, candidateOffset + 1, 'bof');
        status = 'invalid_candidate';
        return;
    end

    fseek(fid, candidateOffset + frameLengthBytes - 4, 'bof');
    endValue = fread(fid, 1, 'uint32=>uint32');
    if isempty(endValue) || endValue ~= frameEnd
        fseek(fid, candidateOffset + 1, 'bof');
        status = 'invalid_candidate';
        return;
    end

    % The existing parser reads the payload metadata beginning four bytes
    % after the frame start (the repeated length word is data_temp1(1)).
    fseek(fid, candidateOffset + 4, 'bof');
    dataTemp1 = fread(fid, 3, 'uint32=>double');
    if numel(dataTemp1) ~= 3
        fseek(fid, candidateOffset + 1, 'bof');
        status = 'invalid_candidate';
        return;
    end
    pointCount = dataTemp1(3);
    if pointCount < 0 || pointCount > 100000 || pointCount ~= floor(pointCount)
        fseek(fid, candidateOffset + 1, 'bof');
        status = 'invalid_candidate';
        return;
    end
    metadataWords = pointCount * 4 + 5;
    dataTemp = fread(fid, metadataWords, 'uint32=>double');
    if numel(dataTemp) ~= metadataWords
        fseek(fid, candidateOffset + 1, 'bof');
        status = 'invalid_candidate';
        return;
    end
    trackInfo = dataTemp(1:pointCount * 4);
    freqHz = dataTemp(pointCount * 4 + 1) * 1e6;
    cpiCount = dataTemp(pointCount * 4 + 2);
    prtNum = dataTemp(pointCount * 4 + 3);
    prtSeconds = dataTemp(pointCount * 4 + 4) * 0.0125e-6;
    dataLength = dataTemp(pointCount * 4 + 5);
    if prtNum <= 0 || prtNum ~= floor(prtNum) || prtNum > 1000000
        fseek(fid, candidateOffset + 1, 'bof');
        status = 'invalid_candidate';
        return;
    end
    sampleCount = prtNum * 31 * 2;
    rawSamples = fread(fid, sampleCount, 'float32=>single');
    if numel(rawSamples) ~= sampleCount
        fseek(fid, candidateOffset + 1, 'bof');
        status = 'invalid_candidate';
        return;
    end
    dataReal = rawSamples(1:2:end);
    dataImag = rawSamples(2:2:end);
    dataOut = reshape(dataReal + 1i * dataImag, 31, prtNum);
    fseek(fid, candidateOffset + frameLengthBytes, 'bof');

    frameIndex = 1; %#ok<NASGU>
    frame.metadata = emptyFrameInternal();
    frame.metadata.raw_frame_index = NaN; % filled by parseRawFile after append
    frame.metadata.frame_offset_bytes = candidateOffset;
    frame.metadata.frame_length_bytes = frameLengthBytes;
    frame.metadata.header_ok = true;
    frame.metadata.footer_ok = true;
    frame.metadata.parse_ok = true;
    frame.metadata.point_count_in_frame = pointCount;
    frame.metadata.track_info_count = numel(trackInfo);
    frame.metadata.track_point_index_raw = valueAt(trackInfo, 2, NaN);
    frame.metadata.track_id_raw = valueAt(trackInfo, 1, NaN);
    frame.metadata.range_bin_raw = valueAt(trackInfo, 3, NaN);
    frame.metadata.doppler_bin_raw = valueAt(trackInfo, 4, NaN);
    frame.metadata.freq_hz = freqHz;
    frame.metadata.cpi_count = cpiCount;
    frame.metadata.prt_num = prtNum;
    frame.metadata.prt_seconds = prtSeconds;
    frame.metadata.data_length = dataLength;
    frame.metadata.observation_seconds = cpiCount * prtNum * prtSeconds;
    frame.metadata.iq_rows = size(dataOut, 1);
    frame.metadata.iq_cols = size(dataOut, 2);
    frame.metadata.iq_complex = ~isreal(dataOut);
    frame.metadata.iq_finite_fraction = sum(isfinite(dataOut(:))) / max(numel(dataOut), 1);
    payloadEndOffset = ftell(fid);
    footerOffset = candidateOffset + frameLengthBytes - 4;
    % The legacy parser explicitly skips one uint32 after I/Q before the
    % footer.  Exclude that known trailer and report only additional bytes.
    frame.metadata.unparsed_payload_bytes = max(0, footerOffset - payloadEndOffset - 4);
    fseek(fid, candidateOffset + frameLengthBytes, 'bof');

    frame.example = emptyExample();
    frame.example.data_out = dataOut;
    frame.example.target_indices = targetIndices;
    status = 'ok';
    return;
end

end

function audit = inspectMatFiles(matPaths, pointInfo, trackInfo, rawAudit)
audit = emptyMatAudit();
audit.existing_file_count = 0;
audit.missing_file_count = 0;
audit.index_in_range = true;
audit.filename_order_consistent = true;
audit.raw_track_point_index_matches_mat = true;
audit.raw_clamped_index_matches_mat = true;
audit.data_out_comparable = true;
audit.rd_comparable = true;
audit.point_index_in_range = true;
audit.track_index_in_range = true;
audit.comparisons = repmat(emptyComparison(), 0, 1);
matIndices = NaN(1, numel(matPaths));
filenameSequences = NaN(1, numel(matPaths));
observationDurations = NaN(1, numel(matPaths));

for matIndex = 1:numel(matPaths)
    matPath = matPaths{matIndex};
    filenameSequences(matIndex) = parseFrameSequenceFromName(matPath);
    if ~exist(matPath, 'file')
        audit.missing_file_count = audit.missing_file_count + 1;
        audit.index_in_range = false;
        continue;
    end
    audit.existing_file_count = audit.existing_file_count + 1;
    try
        metadata = load(matPath, 'index_trackPointNo', 'debug_info');
    catch exception
        audit.missing_file_count = audit.missing_file_count + 1;
        audit.index_in_range = false;
        audit.raw_track_point_index_matches_mat = false;
        audit.raw_clamped_index_matches_mat = false;
        continue;
    end
    matIndices(matIndex) = scalarNumber(fieldValue(metadata, 'index_trackPointNo', NaN));
    debugInfo = fieldValue(metadata, 'debug_info', struct());
    observationDurations(matIndex) = debugObservationSeconds(debugInfo);
    audit.explicit_wall_clock_field_detected = audit.explicit_wall_clock_field_detected || ...
        hasWallClockField(debugInfo);
    if isfinite(matIndices(matIndex)) && pointInfo.data_row_count > 0
        audit.point_index_in_range = audit.point_index_in_range && ...
            matIndices(matIndex) >= 1 && matIndices(matIndex) <= pointInfo.data_row_count;
    else
        audit.point_index_in_range = false;
    end
    if isfinite(matIndices(matIndex)) && trackInfo.data_row_count > 0
        audit.track_index_in_range = audit.track_index_in_range && ...
            matIndices(matIndex) >= 1 && matIndices(matIndex) <= trackInfo.data_row_count;
    else
        audit.track_index_in_range = false;
    end

    rawFrame = findRawFrame(rawAudit.frames, matIndex);
    if isempty(rawFrame)
        audit.raw_track_point_index_matches_mat = false;
        audit.raw_clamped_index_matches_mat = false;
        continue;
    end
    rawIndex = rawFrame.track_point_index_raw;
    audit.raw_track_point_index_matches_mat = audit.raw_track_point_index_matches_mat && ...
        equalFinite(rawIndex, matIndices(matIndex));
    if trackInfo.data_row_count > 0 && isfinite(rawIndex)
        clampedIndex = min(max(rawIndex, 1), trackInfo.data_row_count);
        audit.raw_clamped_index_matches_mat = audit.raw_clamped_index_matches_mat && ...
            equalFinite(clampedIndex, matIndices(matIndex));
    else
        audit.raw_clamped_index_matches_mat = false;
    end
end

    audit.index_in_range = audit.point_index_in_range && audit.track_index_in_range;

if ~isempty(filenameSequences)
    validSequences = filenameSequences(isfinite(filenameSequences));
    audit.filename_order_consistent = numel(validSequences) == numel(filenameSequences) && ...
        all(validSequences == 1:numel(filenameSequences));
end
validDurations = observationDurations(isfinite(observationDurations));
if isempty(validDurations)
    audit.observation_duration_min = NaN;
    audit.observation_duration_max = NaN;
else
    audit.observation_duration_min = min(validDurations);
    audit.observation_duration_max = max(validDurations);
end
audit.point_index_time_coverage = audit.point_index_in_range && ...
    pointInfo.has_time_column && pointInfo.data_row_count > 0;
audit.track_index_time_coverage = audit.track_index_in_range && ...
    trackInfo.has_time_column && trackInfo.data_row_count > 0;

comparisonIndices = comparisonIndicesForAvailableData(numel(matPaths), rawAudit);
for targetIndex = comparisonIndices
    comparison = emptyComparison();
    comparison.frame_index = targetIndex;
    comparison.mat_path = '';
    if targetIndex >= 1 && targetIndex <= numel(matPaths)
        comparison.mat_path = matPaths{targetIndex};
    end
    rawExample = findRawExample(rawAudit.examples, targetIndex);
    if isempty(rawExample) || targetIndex < 1 || targetIndex > numel(matPaths) || ...
            ~exist(matPaths{targetIndex}, 'file')
        comparison.status = 'comparison_unavailable';
        audit.data_out_comparable = false;
        audit.rd_comparable = false;
        audit.comparisons(end + 1, 1) = comparison; %#ok<AGROW>
        continue;
    end
    try
        payload = load(matPaths{targetIndex}, ...
            'data_out', 'data_proc_MTD_result', 'data_proc_MTD_result_db');
    catch exception
        comparison.status = ['mat_load_error:' exception.identifier];
        audit.data_out_comparable = false;
        audit.rd_comparable = false;
        audit.comparisons(end + 1, 1) = comparison; %#ok<AGROW>
        continue;
    end
    matDataOut = fieldValue(payload, 'data_out', []);
    matRd = fieldValue(payload, 'data_proc_MTD_result', []);
    rawDataOut = rawExample.data_out;
    comparison.raw_iq_shape = sizeString(rawDataOut);
    comparison.mat_iq_shape = sizeString(matDataOut);
    comparison.raw_rd_shape = sizeString(rawDataOut);
    comparison.mat_rd_shape = sizeString(matRd);
    comparison.data_out_shape_match = isequal(size(rawDataOut), size(matDataOut));
    comparison.data_out_max_absolute_error = NaN;
    comparison.data_out_max_relative_error = NaN;
    comparison.rd_shape_match = false;
    comparison.rd_recomputed_max_absolute_error = NaN;
    comparison.rd_recomputed_max_relative_error = NaN;
    comparison.status = 'ok';
    if comparison.data_out_shape_match
        dataDifference = double(rawDataOut) - double(matDataOut);
        comparison.data_out_max_absolute_error = max(abs(dataDifference(:)));
        comparison.data_out_max_relative_error = relativeError(rawDataOut, matDataOut);
        comparison.data_out_numeric_match = comparison.data_out_max_relative_error <= 1e-5;
    else
        comparison.data_out_numeric_match = false;
    end
    if ~comparison.data_out_numeric_match
        audit.data_out_comparable = false;
    end
    if isempty(matRd)
        comparison.status = 'mat_rd_missing';
        audit.rd_comparable = false;
    else
        try
            recomputedRd = recomputeRd(rawDataOut);
            comparison.raw_rd_shape = sizeString(recomputedRd);
            comparison.rd_shape_match = isequal(size(recomputedRd), size(matRd));
            if comparison.rd_shape_match
                comparison.rd_recomputed_max_absolute_error = max(abs(double(recomputedRd(:)) - double(matRd(:))));
                comparison.rd_recomputed_max_relative_error = relativeError(recomputedRd, matRd);
                comparison.rd_numeric_match = comparison.rd_recomputed_max_relative_error <= 1e-4;
            else
                comparison.rd_numeric_match = false;
            end
            if ~comparison.rd_numeric_match
                audit.rd_comparable = false;
            end
        catch exception
            comparison.status = ['rd_recompute_error:' exception.identifier];
            comparison.rd_numeric_match = false;
            audit.rd_comparable = false;
        end
    end
    audit.comparisons(end + 1, 1) = comparison; %#ok<AGROW>
end

audit.rd_max_relative_error = maxFiniteField(audit.comparisons, 'rd_recomputed_max_relative_error');
audit.rd_max_absolute_error = maxFiniteField(audit.comparisons, 'rd_recomputed_max_absolute_error');
if audit.missing_file_count > 0
    audit.status = 'mat_file_missing';
elseif ~audit.index_in_range
    audit.status = 'mat_index_out_of_range';
elseif ~audit.raw_clamped_index_matches_mat
    audit.status = 'raw_mat_track_index_mismatch';
elseif ~audit.data_out_comparable
    audit.status = 'raw_mat_data_out_mismatch';
elseif ~audit.rd_comparable
    audit.status = 'raw_mat_rd_mismatch_or_unavailable';
else
    audit.status = 'ok';
end
end

function rd = recomputeRd(dataOut)
dataOut = double(dataOut);
window = taylorwin(size(dataOut, 2), [], -30);
coefficient = repmat(window, [1, size(dataOut, 1)]);
coefficient = permute(coefficient, [2, 1]);
windowed = dataOut .* coefficient;
rd = fftshift(fft(windowed, [], 2), 2);
end

function info = inspectTextTrack(path)
info = struct('path', path, 'exists', false, 'data_row_count', 0, ...
    'first_time', '', 'last_time', '', 'has_time_column', false);
if isempty(path) || ~exist(path, 'file')
    return;
end
info.exists = true;
% The audit only needs the ASCII comma-separated time column; using the
% default byte mode avoids making report generation depend on a local code
% page or a MATLAB-specific GBK codec.
fid = fopen(path, 'r');
if fid == -1
    return;
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fgetl(fid); % header
while true
    line = fgetl(fid);
    if ~ischar(line)
        break;
    end
    if isempty(strtrim(line))
        continue;
    end
    fields = strsplit(line, ',');
    if isempty(fields)
        continue;
    end
    timeValue = strtrim(fields{1});
    if isempty(timeValue)
        continue;
    end
    info.data_row_count = info.data_row_count + 1;
    if isempty(info.first_time)
        info.first_time = timeValue;
    end
    info.last_time = timeValue;
end
info.has_time_column = info.data_row_count > 0;
end

function report = buildReport(projectRoot, manifestPath, selectedIndices, records, ...
    selectionRows, sampleRows, frameRows, rdRows, rawCountMatch, indexMatch, ...
    dataOutMatch, rdMatch, complexIqDetected, rawWallClockDetected, matWallClockDetected)
report = struct();
report.report_schema = 'RADAR-LN3-original-echo-audit-v1';
report.generated_at = datestr(now, 31);
report.project_root = projectRoot;
report.manifest_path = manifestPath;
report.data_source = 'CQ08 only; raw .dat + existing CQ08 MAT + point/track text';
report.read_only = true;
report.requested_sample_count = 12;
report.actual_sample_count = numel(selectedIndices);
report.selected_track_ids = [selectionRows.track_id];
report.selected_labels = {selectionRows.label_name};
report.selected_sample_roles = {selectionRows.selection_roles};
report.raw_frame_count_match_mat_count = rawCountMatch;
report.raw_track_index_mapping_matches_mat = indexMatch;
report.raw_data_out_matches_mat = dataOutMatch;
report.recomputed_rd_matches_mat = rdMatch;
report.raw_complex_iq_detected = complexIqDetected;
report.explicit_raw_wall_clock_time_detected = rawWallClockDetected;
report.explicit_mat_wall_clock_field_detected = matWallClockDetected;
report.explicit_wall_clock_time_detected_anywhere = rawWallClockDetected || matWallClockDetected;
report.raw_frame_time_semantics = ...
    'The parser found frame order and CPI/PRT-derived duration fields; a wall-clock field is not inferred from these values.';
report.interpretation = struct();
report.interpretation.raw_echo_has_temporal_order = true;
report.interpretation.raw_echo_has_cpi_prt_timing = true;
report.interpretation.raw_echo_has_explicit_wall_clock_time = rawWallClockDetected;
report.interpretation.mat_has_explicit_wall_clock_field = matWallClockDetected;
report.interpretation.raw_echo_has_unparsed_payload = any([sampleRows.raw_hidden_payload_detected]);
report.interpretation.mat_preserves_complex_iq_when_comparable = dataOutMatch;
report.interpretation.mat_rd_reproducible_from_raw_iq_when_comparable = rdMatch;
report.interpretation.sample_limit_warning = ...
    'Results cover exactly the selected sample files and must not be generalized to all 1550 CQ08 files.';
report.selection = selectionRows;
report.samples = sampleRows;
report.frame_summary = summarizeFrames(frameRows);
report.rd_comparison_summary = summarizeRd(rdRows);
report.manifest_record_count_before_selection = numel(records);
end

function summary = summarizeFrames(rows)
summary = struct();
summary.frame_row_count = numel(rows);
if isempty(rows)
    summary.raw_frame_count = 0;
    summary.raw_complex_iq_all = false;
    summary.iq_shape_values = {};
    summary.raw_observation_seconds_min = NaN;
    summary.raw_observation_seconds_max = NaN;
    return;
end
summary.raw_frame_count = numel(rows);
summary.raw_complex_iq_all = all([rows.iq_complex]);
summary.iq_shape_values = unique({rows.iq_shape});
summary.raw_observation_seconds_min = minFiniteField(rows, 'observation_seconds');
summary.raw_observation_seconds_max = maxFiniteField(rows, 'observation_seconds');
summary.invalid_header_count = sum(~[rows.header_ok]);
summary.invalid_footer_count = sum(~[rows.footer_ok]);
summary.nonfinite_iq_frame_count = sum([rows.iq_finite_fraction] < 1);
end

function summary = summarizeRd(rows)
summary = struct();
summary.comparison_row_count = numel(rows);
if isempty(rows)
    summary.data_out_numeric_match_count = 0;
    summary.rd_numeric_match_count = 0;
    summary.max_rd_relative_error = NaN;
    summary.max_rd_absolute_error = NaN;
    return;
end
summary.data_out_numeric_match_count = sum([rows.data_out_numeric_match]);
summary.rd_numeric_match_count = sum([rows.rd_numeric_match]);
summary.max_rd_relative_error = maxFiniteField(rows, 'rd_recomputed_max_relative_error');
summary.max_rd_absolute_error = maxFiniteField(rows, 'rd_recomputed_max_absolute_error');
summary.comparison_status_values = unique({rows.status});
end

function indices = comparisonIndices(count)
if count <= 0
    indices = zeros(1, 0);
elseif count == 1
    indices = 1;
else
    indices = unique([1, ceil(count / 2), count]);
end
end

function indices = comparisonIndicesForAvailableData(matCount, rawAudit)
indices = unique([comparisonIndices(matCount), comparisonIndices(rawAudit.frame_count)]);
indices = indices(indices >= 1);
end

function frame = findRawFrame(frames, index)
frame = [];
if isempty(frames)
    return;
end
matches = find([frames.raw_frame_index] == index, 1);
if ~isempty(matches)
    frame = frames(matches);
end
end

function example = findRawExample(examples, index)
example = [];
if isempty(examples)
    return;
end
matches = find([examples.frame_index] == index, 1);
if ~isempty(matches)
    example = examples(matches);
end
end

function value = parseFrameSequenceFromName(path)
[~, name, extension] = fileparts(path);
if ~strcmpi(extension, '.mat')
    value = NaN;
    return;
end
tokens = regexp(name, '_(\d{4})_', 'tokens', 'once');
if isempty(tokens)
    value = NaN;
else
    value = str2double(tokens{1});
end
end

function value = debugObservationSeconds(debugInfo)
value = NaN;
if isempty(debugInfo)
    return;
end
cpi = scalarNumber(fieldValue(debugInfo, 'cpi_count', NaN));
prtNum = scalarNumber(fieldValue(debugInfo, 'prt_num', NaN));
prt = scalarNumber(fieldValue(debugInfo, 'prt_s', NaN));
if isfinite(cpi) && isfinite(prtNum) && isfinite(prt)
    value = cpi * prtNum * prt;
end
end

function result = hasWallClockField(value)
result = false;
if ~isstruct(value)
    return;
end
names = fieldnames(value);
for index = 1:numel(names)
    name = lower(names{index});
    if contains(name, 'time') || contains(name, 'stamp') || ...
            contains(name, 'epoch') || contains(name, 'date') || contains(name, 'wall')
        result = true;
        return;
    end
end
end

function value = fieldValue(value, fieldName, defaultValue)
if nargin < 3
    defaultValue = [];
end
if isstruct(value) && isfield(value, fieldName)
    value = value.(fieldName);
else
    value = defaultValue;
end
end

function value = scalarNumber(value)
if isempty(value)
    value = NaN;
    return;
end
if ischar(value) || isstring(value)
    value = str2double(value);
end
array = double(value(:));
if isempty(array)
    value = NaN;
else
    value = array(1);
end
end

function value = valueAt(array, index, defaultValue)
if numel(array) < index
    value = defaultValue;
else
    value = double(array(index));
end
end

function textValue = asChar(value)
if isempty(value)
    textValue = '';
elseif ischar(value)
    textValue = value;
elseif isstring(value)
    textValue = char(value(1));
elseif isnumeric(value) || islogical(value)
    textValue = num2str(value(1));
else
    textValue = char(string(value));
end
end

function result = equalFinite(a, b)
result = isfinite(a) && isfinite(b) && abs(double(a) - double(b)) < 1e-6;
end

function value = relativeError(a, b)
difference = abs(double(a(:)) - double(b(:)));
if isempty(difference)
    value = 0;
    return;
end
denominator = max(abs(double(b(:))));
if isempty(denominator) || ~isfinite(denominator) || denominator < 1e-12
    denominator = 1e-12;
end
value = max(difference) / denominator;
end

function value = maxFiniteField(rows, fieldName)
if isempty(rows)
    value = NaN;
    return;
end
values = [rows.(fieldName)];
values = values(isfinite(values));
if isempty(values)
    value = NaN;
else
    value = max(values);
end
end

function value = minFiniteField(rows, fieldName)
if isempty(rows)
    value = NaN;
    return;
end
values = [rows.(fieldName)];
values = values(isfinite(values));
if isempty(values)
    value = NaN;
else
    value = min(values);
end
end

function value = sizeString(array)
if isempty(array)
    value = '';
else
    dimensions = size(array);
    value = strjoin(arrayfun(@(item) num2str(item), dimensions, 'UniformOutput', false), 'x');
end
end

function value = isAbsolutePath(path)
value = ~isempty(regexp(path, '^[A-Za-z]:[\\/]', 'once')) || ...
    startsWith(path, '/') || startsWith(path, '\\');
end

function writeJson(path, value)
jsonText = jsonencode(value, 'PrettyPrint', true);
fid = fopen(path, 'w', 'n', 'UTF-8');
if fid == -1
    error('Unable to write JSON report: %s', path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid, jsonText, 'char');
fwrite(fid, sprintf('\n'), 'char');
end

function writeTableCsv(path, rows)
if isempty(rows)
    fid = fopen(path, 'w', 'n', 'UTF-8');
    if fid == -1
        error('Unable to write empty CSV report: %s', path);
    end
    fclose(fid);
    return;
end
tableValue = struct2table(rows);
writetable(tableValue, path, 'Encoding', 'UTF-8');
end

function row = makeSelectionRow(order, trackId, labelName, roles, rawPath, lengthHint, matCount)
row = emptySelectionRow();
row.sample_order = order;
row.track_id = trackId;
row.label_name = labelName;
row.selection_roles = roles;
row.raw_echo_path = rawPath;
row.track_length_hint = lengthHint;
row.manifest_mat_count = matCount;
end

function row = makeFrameRow(order, trackId, labelName, frame, pointInfo, trackInfo)
row = emptyFrameRow();
row.sample_order = order;
row.track_id = trackId;
row.label_name = labelName;
row.raw_frame_index = frame.raw_frame_index;
row.frame_offset_bytes = frame.frame_offset_bytes;
row.frame_length_bytes = frame.frame_length_bytes;
row.header_ok = frame.header_ok;
row.footer_ok = frame.footer_ok;
row.parse_ok = frame.parse_ok;
row.point_count_in_frame = frame.point_count_in_frame;
row.track_point_index_raw = frame.track_point_index_raw;
row.track_id_raw = frame.track_id_raw;
row.range_bin_raw = frame.range_bin_raw;
row.doppler_bin_raw = frame.doppler_bin_raw;
row.freq_hz = frame.freq_hz;
row.cpi_count = frame.cpi_count;
row.prt_num = frame.prt_num;
row.prt_seconds = frame.prt_seconds;
row.observation_seconds = frame.observation_seconds;
row.data_length = frame.data_length;
row.iq_shape = sprintf('%dx%d', frame.iq_rows, frame.iq_cols);
row.iq_complex = frame.iq_complex;
row.iq_finite_fraction = frame.iq_finite_fraction;
row.point_row_count = pointInfo.data_row_count;
row.track_row_count = trackInfo.data_row_count;
end

function row = makeRdRow(order, trackId, labelName, comparison)
row = emptyRdRow();
row.sample_order = order;
row.track_id = trackId;
row.label_name = labelName;
row.frame_index = comparison.frame_index;
row.mat_path = comparison.mat_path;
row.raw_iq_shape = comparison.raw_iq_shape;
row.mat_iq_shape = comparison.mat_iq_shape;
row.raw_rd_shape = comparison.raw_rd_shape;
row.mat_rd_shape = comparison.mat_rd_shape;
row.data_out_shape_match = comparison.data_out_shape_match;
row.data_out_numeric_match = comparison.data_out_numeric_match;
row.data_out_max_absolute_error = comparison.data_out_max_absolute_error;
row.data_out_max_relative_error = comparison.data_out_max_relative_error;
row.rd_shape_match = comparison.rd_shape_match;
row.rd_numeric_match = comparison.rd_numeric_match;
row.rd_recomputed_max_absolute_error = comparison.rd_recomputed_max_absolute_error;
row.rd_recomputed_max_relative_error = comparison.rd_recomputed_max_relative_error;
row.status = comparison.status;
end

function row = emptySelectionRow()
row = struct('sample_order', NaN, 'track_id', NaN, 'label_name', '', ...
    'selection_roles', '', 'raw_echo_path', '', 'track_length_hint', NaN, ...
    'manifest_mat_count', NaN);
end

function row = emptySampleRow()
row = struct( ...
    'sample_order', NaN, 'track_id', NaN, 'label_name', '', 'selection_roles', '', ...
    'raw_echo_path', '', 'point_path', '', 'track_path', '', ...
    'manifest_mat_count', NaN, 'point_row_count', NaN, 'track_row_count', NaN, ...
    'point_first_time', '', 'point_last_time', '', 'track_first_time', '', 'track_last_time', '', ...
    'raw_wall_clock_time_field_detected', false, 'parser', '', 'raw_frame_count', NaN, ...
    'raw_valid_frame_count', NaN, 'raw_invalid_candidate_count', NaN, 'raw_file_bytes', NaN, ...
    'raw_frame_length_min_bytes', NaN, 'raw_frame_length_max_bytes', NaN, ...
    'raw_iq_shape_consistent', false, 'raw_complex_iq_detected', false, ...
    'raw_cpi_count_min', NaN, 'raw_cpi_count_max', NaN, 'raw_prt_num_min', NaN, ...
    'raw_prt_num_max', NaN, 'raw_prt_seconds_min', NaN, 'raw_prt_seconds_max', NaN, ...
    'raw_observation_seconds_min', NaN, 'raw_observation_seconds_max', NaN, ...
    'raw_unparsed_payload_bytes_max', NaN, 'raw_hidden_payload_detected', false, ...
    'mat_existing_file_count', NaN, 'mat_missing_file_count', NaN, 'mat_index_in_range', false, ...
    'mat_filename_order_consistent', false, 'raw_track_point_index_matches_mat', false, ...
    'raw_clamped_track_point_index_matches_mat', false, 'raw_mat_data_out_comparable', false, ...
    'raw_mat_rd_comparable', false, 'rd_recompute_max_relative_error', NaN, ...
    'rd_recompute_max_absolute_error', NaN, 'mat_observation_duration_min', NaN, ...
    'mat_observation_duration_max', NaN, 'mat_explicit_wall_clock_field_detected', false, ...
    'point_index_time_coverage', false, 'track_index_time_coverage', false, 'status', '');
end

function row = emptyFrameRow()
row = struct( ...
    'sample_order', NaN, 'track_id', NaN, 'label_name', '', 'raw_frame_index', NaN, ...
    'frame_offset_bytes', NaN, 'frame_length_bytes', NaN, 'header_ok', false, ...
    'footer_ok', false, 'parse_ok', false, 'point_count_in_frame', NaN, ...
    'track_point_index_raw', NaN, 'track_id_raw', NaN, 'range_bin_raw', NaN, ...
    'doppler_bin_raw', NaN, 'freq_hz', NaN, 'cpi_count', NaN, 'prt_num', NaN, ...
    'prt_seconds', NaN, 'observation_seconds', NaN, 'data_length', NaN, ...
    'iq_shape', '', 'iq_complex', false, 'iq_finite_fraction', NaN, ...
    'point_row_count', NaN, 'track_row_count', NaN);
end

function row = emptyRdRow()
row = struct( ...
    'sample_order', NaN, 'track_id', NaN, 'label_name', '', 'frame_index', NaN, ...
    'mat_path', '', 'raw_iq_shape', '', 'mat_iq_shape', '', 'raw_rd_shape', '', ...
    'mat_rd_shape', '', 'data_out_shape_match', false, 'data_out_numeric_match', false, ...
    'data_out_max_absolute_error', NaN, 'data_out_max_relative_error', NaN, ...
    'rd_shape_match', false, 'rd_numeric_match', false, ...
    'rd_recomputed_max_absolute_error', NaN, 'rd_recomputed_max_relative_error', NaN, ...
    'status', '');
end

function row = emptyFrameInternal()
row = struct( ...
    'raw_frame_index', NaN, 'frame_offset_bytes', NaN, 'frame_length_bytes', NaN, ...
    'header_ok', false, 'footer_ok', false, 'parse_ok', false, ...
    'point_count_in_frame', NaN, 'track_info_count', NaN, 'track_point_index_raw', NaN, ...
    'track_id_raw', NaN, 'range_bin_raw', NaN, 'doppler_bin_raw', NaN, 'freq_hz', NaN, ...
    'cpi_count', NaN, 'prt_num', NaN, 'prt_seconds', NaN, 'observation_seconds', NaN, ...
    'data_length', NaN, 'iq_rows', NaN, 'iq_cols', NaN, 'iq_complex', false, ...
    'iq_finite_fraction', NaN, 'unparsed_payload_bytes', NaN);
end

function row = emptyExample()
row = struct('frame_index', NaN, 'data_out', [], 'target_indices', []);
end

function row = emptyRawAudit()
row = struct( ...
    'file_bytes', NaN, 'target_indices', [], 'frames', emptyFrameInternal(), ...
    'examples', emptyExample(), 'frame_count', 0, 'valid_frame_count', 0, ...
    'invalid_candidate_count', 0, 'frame_length_min_bytes', NaN, ...
    'frame_length_max_bytes', NaN, 'iq_shape_consistent', false, ...
    'complex_iq_detected', false, 'wall_clock_field_detected', false, ...
    'cpi_count_min', NaN, 'cpi_count_max', NaN, 'prt_num_min', NaN, ...
    'prt_num_max', NaN, 'prt_seconds_min', NaN, 'prt_seconds_max', NaN, ...
    'observation_seconds_min', NaN, 'observation_seconds_max', NaN, ...
    'unparsed_payload_bytes_max', NaN);
row.frames = repmat(emptyFrameInternal(), 0, 1);
row.examples = repmat(emptyExample(), 0, 1);
end

function row = emptyMatAudit()
row = struct( ...
    'existing_file_count', 0, 'missing_file_count', 0, 'index_in_range', false, ...
    'point_index_in_range', false, 'track_index_in_range', false, ...
    'filename_order_consistent', false, 'raw_track_point_index_matches_mat', false, ...
    'raw_clamped_index_matches_mat', false, 'data_out_comparable', false, ...
    'rd_comparable', false, 'rd_max_relative_error', NaN, 'rd_max_absolute_error', NaN, ...
    'observation_duration_min', NaN, 'observation_duration_max', NaN, ...
    'explicit_wall_clock_field_detected', false, 'point_index_time_coverage', false, ...
    'track_index_time_coverage', false, 'comparisons', emptyComparison(), 'status', '');
row.comparisons = repmat(emptyComparison(), 0, 1);
end

function row = emptyComparison()
row = struct( ...
    'frame_index', NaN, 'mat_path', '', 'raw_iq_shape', '', 'mat_iq_shape', '', ...
    'raw_rd_shape', '', 'mat_rd_shape', '', 'data_out_shape_match', false, ...
    'data_out_numeric_match', false, 'data_out_max_absolute_error', NaN, ...
    'data_out_max_relative_error', NaN, 'rd_shape_match', false, ...
    'rd_numeric_match', false, 'rd_recomputed_max_absolute_error', NaN, ...
    'rd_recomputed_max_relative_error', NaN, 'status', '');
end
