function outputs = track_0333_original_echo_audit(projectRoot)
%TRACK_0333_ORIGINAL_ECHO_AUDIT Read-only CQ08 Track 333 raw/MAT audit.
%
% This focused audit parses every frame in 333_Label_3.dat, compares the
% raw target metadata and complex I/Q against all 22 existing MAT files,
% recomputes RD with MATLAB taylorwin, and reports whether Track points
% 1-15 exist in the raw file. It never modifies MAT, data, config, labels,
% checkpoints, or training artifacts.
%
% Example from F:\Project\RadarInsight:
%   addpath(fullfile(pwd, 'src-v3', 'tests', 'processed'));
%   outputs = track_0333_original_echo_audit(pwd);
%   disp(outputs);

if nargin < 1 || isempty(projectRoot)
    scriptPath = mfilename('fullpath');
    projectRoot = fileparts(fileparts(fileparts(fileparts(scriptPath))));
end
projectRoot = char(projectRoot);

rawPath = fullfile(projectRoot, 'OriData', '23', 'CQ-08', ...
    '原始回波', '333_Label_3.dat');
matDir = fullfile(projectRoot, 'OriData', '23', 'CQ-08', 'MAT');
trackPath = fullfile(projectRoot, 'OriData', '23', 'CQ-08', ...
    '航迹', 'Tracks_333_3_22.txt');
reportDir = fullfile(projectRoot, 'src-v3', 'processed', 'report', ...
    'RADAR-LN3-track-0333-original-echo-audit', 'matlab-v2');
if ~exist(reportDir, 'dir')
    mkdir(reportDir);
end
if ~exist(rawPath, 'file')
    error('Track 333 raw echo does not exist: %s', rawPath);
end
if ~exist(trackPath, 'file')
    error('Track 333 trajectory does not exist: %s', trackPath);
end

fprintf('[Track 0333 audit] project=%s\n', projectRoot);
fprintf('[Track 0333 audit] raw=%s\n', rawPath);
fprintf('[Track 0333 audit] report_dir=%s\n', reportDir);

rawFrames = parseRawFrames(rawPath);
matFiles = dir(fullfile(matDir, '*_333.mat'));
[~, order] = sort({matFiles.name});
matFiles = matFiles(order);
trackPointCount = countTextDataRows(trackPath);

[comparisonRows, rawRd, matRd] = compareRawAndMat(rawFrames, matFiles);
targetRows = buildTargetRows(rawFrames);
coverageRows = buildCoverageRows(rawFrames, comparisonRows, trackPointCount);

comparisonPath = fullfile(reportDir, 'frame-comparison.csv');
targetPath = fullfile(reportDir, 'raw-target-records.csv');
coveragePath = fullfile(reportDir, 'point-coverage.csv');
rdPlotPath = fullfile(reportDir, 'rd-comparison.png');
coveragePlotPath = fullfile(reportDir, 'point-coverage.png');
summaryPath = fullfile(reportDir, 'summary.json');

writetable(struct2table(comparisonRows), comparisonPath, 'Encoding', 'UTF-8');
writetable(struct2table(targetRows), targetPath, 'Encoding', 'UTF-8');
writetable(struct2table(coverageRows), coveragePath, 'Encoding', 'UTF-8');
plotRdComparison(comparisonRows, rawRd, matRd, rdPlotPath);
plotPointCoverage(coverageRows, coveragePlotPath);

rawPointIndices = unique([targetRows.track_point_index_raw]);
pointsOneToFifteenPresent = any(ismember(1:15, rawPointIndices));
pointsSixteenToTwentyTwoPresent = all(ismember(16:22, rawPointIndices));
allFrameMatched = numel(comparisonRows) == 22 && ...
    all(strcmp({comparisonRows.status}, 'matched'));
allCoverageMatched = all([coverageRows.raw_matches_mat]);
fileInfo = dir(rawPath);
parsedBytes = sum([rawFrames.frame_length_bytes]);
rawComplete = parsedBytes == fileInfo.bytes && ...
    all([rawFrames.header_ok]) && all([rawFrames.footer_ok]) && ...
    sum([rawFrames.unparsed_payload_bytes]) == 0;

if rawComplete && allFrameMatched && allCoverageMatched && ...
        ~pointsOneToFifteenPresent && pointsSixteenToTwentyTwoPresent
    status = 'passed';
    conclusion = 'source_raw_has_no_frames_for_track_points_1_to_15';
    pointsRecoverable = false;
    action = ['Keep explicit RD-missing masks for Track 333 points 1-15; ' ...
        'this .dat contains no frame record from which those RD observations can be reconstructed.'];
else
    status = 'needs_review';
    conclusion = 'inconclusive_or_source_mapping_conflict';
    pointsRecoverable = [];
    action = ['Do not rebuild data. Resolve the raw/MAT discrepancy before changing ' ...
        'the data contract or index.'];
end

summary = struct();
summary.status = status;
summary.schema_version = 'RADAR-LN3-track-0333-original-echo-matlab-audit-v2';
summary.audit_logic_revision = ...
    'coverage sequences are compared as orientation-independent column vectors';
summary.generated_at = datestr(now, 31);
summary.scope = 'CQ08 Track 333 only; read-only raw/MAT audit; test excluded';
summary.project_root = projectRoot;
summary.track_id = 333;
summary.current_label = 'BirdTarget';
summary.label_policy = ['Keep the current label unless multiple independent evidence ' ...
    'sources support a review request; motion strength and model confidence are not label evidence.'];
summary.raw_echo_path = rawPath;
summary.raw_echo_bytes = fileInfo.bytes;
summary.raw_echo_sha256 = sha256File(rawPath);
summary.raw_frame_count = numel(rawFrames);
summary.raw_parsed_bytes = parsedBytes;
summary.raw_contiguous_from_zero_to_eof = rawComplete;
summary.mat_file_count = numel(matFiles);
summary.track_point_count = trackPointCount;
summary.raw_point_indices = rawPointIndices;
summary.raw_frame_counts_by_point = pointCountSummary(targetRows, rawPointIndices);
summary.raw_points_1_to_15_present = pointsOneToFifteenPresent;
summary.raw_points_16_to_22_present = pointsSixteenToTwentyTwoPresent;
summary.all_raw_target_records_match_mat_debug = all([comparisonRows.raw_target_match]);
summary.all_point_indices_match = all([comparisonRows.point_index_match]);
summary.all_metadata_match = all([comparisonRows.metadata_match]);
summary.all_iq_shapes_match = all([comparisonRows.iq_shape_match]);
summary.all_iq_numeric_match = all([comparisonRows.iq_numeric_match]);
summary.maximum_iq_absolute_error = max([comparisonRows.iq_max_absolute_error]);
summary.maximum_iq_relative_error = max([comparisonRows.iq_max_relative_error]);
summary.all_rd_shapes_match = all([comparisonRows.rd_shape_match]);
summary.all_rd_numeric_match = all([comparisonRows.rd_numeric_match]);
summary.maximum_rd_absolute_error = max([comparisonRows.rd_max_absolute_error]);
summary.maximum_rd_relative_error = max([comparisonRows.rd_max_relative_error]);
summary.all_point_coverage_raw_matches_mat = allCoverageMatched;
summary.recoverability = struct( ...
    'conclusion', conclusion, ...
    'points_1_to_15_recoverable_from_this_dat', pointsRecoverable, ...
    'action', action);
summary.outputs = struct( ...
    'frame_comparison_csv', comparisonPath, ...
    'raw_target_records_csv', targetPath, ...
    'point_coverage_csv', coveragePath, ...
    'rd_comparison_png', rdPlotPath, ...
    'point_coverage_png', coveragePlotPath, ...
    'summary_json', summaryPath);
summary.interpretation_limits = { ...
    'The raw target identifier is not assumed to be CQ08 parent Track ID 333.', ...
    'This audit proves what exists in this .dat file and cannot recover a lost acquisition record.', ...
    'No label, index, MAT, config, checkpoint or training artifact was modified.', ...
    'No training, test evaluation, threshold fitting or deployment switch was executed.'};
summary.test = '未执行';
summary.training = '未执行';
summary.data_rebuild = '未执行';
writeJson(summaryPath, summary);

outputs = struct();
outputs.status = status;
outputs.summary_path = summaryPath;
outputs.frame_comparison_path = comparisonPath;
outputs.point_coverage_path = coveragePath;
outputs.rd_comparison_plot = rdPlotPath;
outputs.point_coverage_plot = coveragePlotPath;
outputs.raw_frame_count = numel(rawFrames);
outputs.mat_file_count = numel(matFiles);
outputs.raw_point_indices = rawPointIndices;
outputs.raw_points_1_to_15_present = pointsOneToFifteenPresent;
outputs.all_iq_numeric_match = summary.all_iq_numeric_match;
outputs.all_rd_numeric_match = summary.all_rd_numeric_match;
outputs.conclusion = conclusion;

fprintf('[Track 0333 audit] raw_frames=%d mat_files=%d raw_points=%s\n', ...
    numel(rawFrames), numel(matFiles), mat2str(rawPointIndices));
fprintf('[Track 0333 audit] points_1_to_15_present=%d\n', pointsOneToFifteenPresent);
fprintf('[Track 0333 audit] IQ_match=%d RD_match=%d\n', ...
    summary.all_iq_numeric_match, summary.all_rd_numeric_match);
fprintf('[Track 0333 audit] completed; summary=%s\n', summaryPath);
end

function frames = parseRawFrames(rawPath)
fileInfo = dir(rawPath);
fid = fopen(rawPath, 'r', 'ieee-le');
if fid == -1
    error('Unable to open raw echo: %s', rawPath);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
frameHead = uint32(hex2dec('FA55FA55'));
frameEnd = uint32(hex2dec('55FA55FA'));
frames = repmat(emptyRawFrame(), 0, 1);
offset = 0;
while offset < fileInfo.bytes
    fseek(fid, offset, 'bof');
    head = fread(fid, 1, 'uint32=>uint32');
    lengthWords = fread(fid, 1, 'uint32=>double');
    scanAngle = fread(fid, 1, 'uint32=>double');
    pointCount = fread(fid, 1, 'uint32=>double');
    if isempty(head) || isempty(lengthWords) || isempty(scanAngle) || isempty(pointCount)
        error('Truncated frame header at byte offset %d.', offset);
    end
    if head ~= frameHead
        error('Invalid/non-contiguous frame head at byte offset %d.', offset);
    end
    frameLength = lengthWords * 4;
    if frameLength < 36 || offset + frameLength > fileInfo.bytes
        error('Invalid frame length %g at byte offset %d.', frameLength, offset);
    end
    if pointCount < 0 || pointCount > 100000 || pointCount ~= floor(pointCount)
        error('Invalid target record count at byte offset %d.', offset);
    end
    targetWords = fread(fid, pointCount * 4, 'uint32=>double');
    if numel(targetWords) ~= pointCount * 4
        error('Truncated target records at byte offset %d.', offset);
    end
    targetMatrix = reshape(targetWords, 4, pointCount);
    metadata = fread(fid, 5, 'uint32=>double');
    if numel(metadata) ~= 5
        error('Truncated radar metadata at byte offset %d.', offset);
    end
    frequencyMHz = metadata(1);
    cpiCount = metadata(2);
    prtNum = metadata(3);
    prtRaw = metadata(4);
    dataLength = metadata(5);
    if prtNum <= 0 || prtNum ~= floor(prtNum) || prtNum > 1000000
        error('Invalid PRT count at byte offset %d.', offset);
    end
    sampleCount = prtNum * 31 * 2;
    rawSamples = fread(fid, sampleCount, 'float32=>single');
    if numel(rawSamples) ~= sampleCount
        error('Truncated I/Q payload at byte offset %d.', offset);
    end
    dataReal = rawSamples(1:2:end);
    dataImag = rawSamples(2:2:end);
    dataOut = reshape(dataReal + 1i * dataImag, 31, prtNum);
    footerOffset = offset + frameLength - 4;
    payloadEnd = ftell(fid);
    unparsedBytes = footerOffset - payloadEnd;
    if unparsedBytes < 0
        error('I/Q payload crossed the footer at byte offset %d.', offset);
    end
    fseek(fid, footerOffset, 'bof');
    footer = fread(fid, 1, 'uint32=>uint32');
    if isempty(footer) || footer ~= frameEnd
        error('Invalid frame footer at byte offset %d.', footerOffset);
    end
    row = emptyRawFrame();
    row.frame_sequence = numel(frames) + 1;
    row.frame_offset_bytes = offset;
    row.frame_length_words = lengthWords;
    row.frame_length_bytes = frameLength;
    row.scan_angle_raw = scanAngle;
    row.point_count = pointCount;
    row.frequency_mhz_raw = frequencyMHz;
    row.frequency_hz = frequencyMHz * 1e6;
    row.cpi_count = cpiCount;
    row.prt_num = prtNum;
    row.prt_raw = prtRaw;
    row.prt_seconds = prtRaw * 0.0125e-6;
    row.data_length = dataLength;
    row.target_matrix = targetMatrix;
    row.data_out = dataOut;
    row.header_ok = true;
    row.footer_ok = true;
    row.unparsed_payload_bytes = unparsedBytes;
    frames(end + 1, 1) = row; %#ok<AGROW>
    offset = offset + frameLength;
end
if offset ~= fileInfo.bytes
    error('Raw parser ended at %d but file size is %d.', offset, fileInfo.bytes);
end
end

function [rows, recomputedRd, matRd] = compareRawAndMat(rawFrames, matFiles)
count = max(numel(rawFrames), numel(matFiles));
rows = repmat(emptyComparisonRow(), count, 1);
recomputedRd = cell(count, 1);
matRd = cell(count, 1);
for index = 1:count
    row = emptyComparisonRow();
    row.frame_sequence = index;
    if index > numel(rawFrames)
        row.status = 'raw_frame_missing';
        rows(index) = row;
        continue;
    end
    raw = rawFrames(index);
    row.frame_offset_bytes = raw.frame_offset_bytes;
    row.frame_length_bytes = raw.frame_length_bytes;
    row.point_count = raw.point_count;
    row.frequency_hz = raw.frequency_hz;
    row.cpi_count = raw.cpi_count;
    row.prt_num = raw.prt_num;
    row.prt_seconds = raw.prt_seconds;
    row.data_length = raw.data_length;
    row.iq_shape = sizeString(raw.data_out);
    if index > numel(matFiles)
        row.status = 'mat_file_missing';
        rows(index) = row;
        continue;
    end
    matPath = fullfile(matFiles(index).folder, matFiles(index).name);
    row.mat_path = matPath;
    loaded = load(matPath, 'index_trackPointNo', 'debug_info', 'data_out', ...
        'data_proc_MTD_result', 'label');
    debugHead = double(loaded.debug_info.track_no_info_head(:));
    matches = find(all(bsxfun(@eq, double(raw.target_matrix), debugHead), 1));
    row.raw_target_match = numel(matches) == 1;
    if row.raw_target_match
        selected = raw.target_matrix(:, matches(1));
        row.raw_target_identifier = selected(1);
        row.raw_track_point_index = selected(2);
        row.raw_range_bin = selected(3);
        row.raw_doppler_bin = selected(4);
    end
    row.mat_track_point_index = double(loaded.index_trackPointNo);
    row.point_index_match = row.raw_target_match && ...
        row.raw_track_point_index == row.mat_track_point_index;
    row.label_match = double(loaded.label) == 3;
    row.metadata_match = equalFinite(raw.frequency_hz, loaded.debug_info.freq_hz, 1e-6) && ...
        equalFinite(raw.cpi_count, loaded.debug_info.cpi_count, 0) && ...
        equalFinite(raw.prt_num, loaded.debug_info.prt_num, 0) && ...
        equalFinite(raw.prt_seconds, loaded.debug_info.prt_s, 1e-15) && ...
        equalFinite(raw.data_length, loaded.debug_info.data_length, 0);
    row.iq_shape_match = isequal(size(raw.data_out), size(loaded.data_out));
    if row.iq_shape_match
        row.iq_max_absolute_error = max(abs(double(raw.data_out(:)) - double(loaded.data_out(:))));
        row.iq_max_relative_error = relativeError(raw.data_out, loaded.data_out);
        row.iq_numeric_match = row.iq_max_relative_error <= 1e-7;
    end
    rdRaw = recomputeRd(raw.data_out);
    recomputedRd{index} = rdRaw;
    matRd{index} = loaded.data_proc_MTD_result;
    row.rd_shape_match = isequal(size(rdRaw), size(loaded.data_proc_MTD_result));
    if row.rd_shape_match
        row.rd_max_absolute_error = max(abs(double(rdRaw(:)) - ...
            double(loaded.data_proc_MTD_result(:))));
        row.rd_max_relative_error = relativeError(rdRaw, loaded.data_proc_MTD_result);
        row.rd_numeric_match = row.rd_max_relative_error <= 1e-4;
    end
    if row.raw_target_match && row.point_index_match && row.label_match && ...
            row.metadata_match && row.iq_numeric_match && row.rd_numeric_match
        row.status = 'matched';
    else
        row.status = 'mismatch';
    end
    rows(index) = row;
end
end

function rows = buildTargetRows(rawFrames)
rows = repmat(emptyTargetRow(), 0, 1);
for frameIndex = 1:numel(rawFrames)
    targetMatrix = rawFrames(frameIndex).target_matrix;
    for targetIndex = 1:size(targetMatrix, 2)
        row = emptyTargetRow();
        row.frame_sequence = frameIndex;
        row.target_record_index = targetIndex;
        row.raw_target_identifier = targetMatrix(1, targetIndex);
        row.track_point_index_raw = targetMatrix(2, targetIndex);
        row.range_bin_raw = targetMatrix(3, targetIndex);
        row.doppler_bin_raw = targetMatrix(4, targetIndex);
        rows(end + 1, 1) = row; %#ok<AGROW>
    end
end
end

function rows = buildCoverageRows(rawFrames, comparisons, trackPointCount)
rows = repmat(emptyCoverageRow(), trackPointCount, 1);
matPoints = [comparisons.mat_track_point_index];
for point = 1:trackPointCount
    rawSequences = [];
    for frameIndex = 1:numel(rawFrames)
        if any(rawFrames(frameIndex).target_matrix(2, :) == point)
            rawSequences(end + 1) = frameIndex; %#ok<AGROW>
        end
    end
    matSequences = find(matPoints == point);
    row = emptyCoverageRow();
    row.track_point_index = point;
    row.raw_frame_count = numel(rawSequences);
    row.raw_frame_sequences_one_based = sequenceString(rawSequences);
    row.mat_frame_count = numel(matSequences);
    row.mat_frame_sequences_one_based = sequenceString(matSequences);
    row.raw_present = ~isempty(rawSequences);
    row.mat_present = ~isempty(matSequences);
    % MATLAB represents the two empty sets above as 0x0 and 1x0.  Their
    % orientation is not part of the coverage contract; compare values in
    % a canonical column representation so empty/empty is a true match.
    row.raw_matches_mat = isequal(rawSequences(:), matSequences(:));
    rows(point) = row;
end
end

function value = sequenceString(sequence)
if isempty(sequence)
    value = '[]';
else
    value = mat2str(sequence(:).');
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

function plotRdComparison(rows, rawRd, matRd, outputPath)
valid = find(strcmp({rows.status}, 'matched'));
if isempty(valid)
    return;
end
indices = unique([valid(1), valid(ceil(numel(valid) / 2)), valid(end)]);
figureValue = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100, 100, 1500, 950]);
cleanup = onCleanup(@() close(figureValue)); %#ok<NASGU>
for plotIndex = 1:numel(indices)
    frameIndex = indices(plotIndex);
    rawDb = 20 * log10(abs(rawRd{frameIndex}) + eps);
    matDb = 20 * log10(abs(matRd{frameIndex}) + eps);
    difference = abs(double(rawRd{frameIndex}) - double(matRd{frameIndex}));
    subplot(numel(indices), 3, (plotIndex - 1) * 3 + 1);
    imagesc(rawDb); axis xy; colorbar;
    title(sprintf('Raw recomputed RD | frame %d point %d', ...
        frameIndex, rows(frameIndex).raw_track_point_index));
    xlabel('Doppler bin'); ylabel('Range bin');
    subplot(numel(indices), 3, (plotIndex - 1) * 3 + 2);
    imagesc(matDb); axis xy; colorbar;
    title('Existing MAT RD'); xlabel('Doppler bin'); ylabel('Range bin');
    subplot(numel(indices), 3, (plotIndex - 1) * 3 + 3);
    imagesc(log10(difference + eps)); axis xy; colorbar;
    title(sprintf('log10 absolute error | max %.3g', max(difference(:))));
    xlabel('Doppler bin'); ylabel('Range bin');
end
if exist('sgtitle', 'file') || exist('sgtitle', 'builtin')
    sgtitle('CQ08 Track 333 raw-to-MAT RD comparison (first/middle/last)');
end
print(figureValue, outputPath, '-dpng', '-r150');
end

function plotPointCoverage(rows, outputPath)
figureValue = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100, 100, 1200, 550]);
cleanup = onCleanup(@() close(figureValue)); %#ok<NASGU>
points = [rows.track_point_index];
counts = [[rows.raw_frame_count].', [rows.mat_frame_count].'];
bar(points, counts, 'grouped');
xlabel('Track point index'); ylabel('Frame count');
title('CQ08 Track 333 raw/MAT frame coverage');
legend({'Raw .dat', 'Existing MAT'}, 'Location', 'northwest');
grid on; xlim([0, max(points) + 1]);
print(figureValue, outputPath, '-dpng', '-r150');
end

function count = countTextDataRows(path)
fid = fopen(path, 'r');
if fid == -1
    error('Unable to open trajectory file: %s', path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fgetl(fid);
count = 0;
while true
    line = fgetl(fid);
    if ~ischar(line)
        break;
    end
    if ~isempty(strtrim(line))
        count = count + 1;
    end
end
end

function result = pointCountSummary(rows, pointIndices)
result = repmat(struct('track_point_index', NaN, 'raw_frame_count', 0), ...
    numel(pointIndices), 1);
for index = 1:numel(pointIndices)
    point = pointIndices(index);
    result(index).track_point_index = point;
    result(index).raw_frame_count = sum([rows.track_point_index_raw] == point);
end
end

function result = equalFinite(a, b, tolerance)
a = double(a); b = double(b);
result = isfinite(a) && isfinite(b) && abs(a - b) <= tolerance;
end

function value = relativeError(a, b)
difference = abs(double(a(:)) - double(b(:)));
denominator = max(abs(double(b(:))));
if isempty(denominator) || ~isfinite(denominator) || denominator < 1e-12
    denominator = 1e-12;
end
value = max(difference) / denominator;
end

function value = sizeString(array)
dimensions = size(array);
value = strjoin(arrayfun(@num2str, dimensions, 'UniformOutput', false), 'x');
end

function hash = sha256File(path)
hash = '';
try
    digest = java.security.MessageDigest.getInstance('SHA-256');
    fid = fopen(path, 'r');
    if fid == -1
        return;
    end
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    while true
        bytes = fread(fid, 1024 * 1024, '*uint8');
        if isempty(bytes)
            break;
        end
        digest.update(bytes);
    end
    bytes = typecast(digest.digest(), 'uint8');
    hash = lower(reshape(dec2hex(bytes, 2).', 1, []));
catch
    hash = '';
end
end

function writeJson(path, value)
text = jsonencode(value, 'PrettyPrint', true);
fid = fopen(path, 'w', 'n', 'UTF-8');
if fid == -1
    error('Unable to write JSON report: %s', path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fwrite(fid, text, 'char');
fwrite(fid, sprintf('\n'), 'char');
end

function row = emptyRawFrame()
row = struct( ...
    'frame_sequence', NaN, 'frame_offset_bytes', NaN, ...
    'frame_length_words', NaN, 'frame_length_bytes', NaN, ...
    'scan_angle_raw', NaN, 'point_count', NaN, ...
    'frequency_mhz_raw', NaN, 'frequency_hz', NaN, ...
    'cpi_count', NaN, 'prt_num', NaN, 'prt_raw', NaN, ...
    'prt_seconds', NaN, 'data_length', NaN, ...
    'target_matrix', zeros(4, 0), 'data_out', [], ...
    'header_ok', false, 'footer_ok', false, 'unparsed_payload_bytes', NaN);
end

function row = emptyComparisonRow()
row = struct( ...
    'frame_sequence', NaN, 'frame_offset_bytes', NaN, ...
    'frame_length_bytes', NaN, 'point_count', NaN, ...
    'raw_target_identifier', NaN, 'raw_track_point_index', NaN, ...
    'raw_range_bin', NaN, 'raw_doppler_bin', NaN, ...
    'frequency_hz', NaN, 'cpi_count', NaN, 'prt_num', NaN, ...
    'prt_seconds', NaN, 'data_length', NaN, 'iq_shape', '', ...
    'mat_path', '', 'mat_track_point_index', NaN, ...
    'raw_target_match', false, 'point_index_match', false, ...
    'label_match', false, 'metadata_match', false, ...
    'iq_shape_match', false, 'iq_max_absolute_error', NaN, ...
    'iq_max_relative_error', NaN, 'iq_numeric_match', false, ...
    'rd_shape_match', false, 'rd_max_absolute_error', NaN, ...
    'rd_max_relative_error', NaN, 'rd_numeric_match', false, ...
    'status', 'comparison_unavailable');
end

function row = emptyTargetRow()
row = struct( ...
    'frame_sequence', NaN, 'target_record_index', NaN, ...
    'raw_target_identifier', NaN, 'track_point_index_raw', NaN, ...
    'range_bin_raw', NaN, 'doppler_bin_raw', NaN);
end

function row = emptyCoverageRow()
row = struct( ...
    'track_point_index', NaN, 'raw_frame_count', 0, ...
    'raw_frame_sequences_one_based', '', 'mat_frame_count', 0, ...
    'mat_frame_sequences_one_based', '', 'raw_present', false, ...
    'mat_present', false, 'raw_matches_mat', false);
end
