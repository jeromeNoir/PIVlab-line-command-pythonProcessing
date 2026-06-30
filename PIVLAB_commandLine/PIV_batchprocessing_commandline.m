% Batch script: process PIV images for ALL sub-folders of a project root.
% This is the batch version of PIVlab_process_commandline.m.
% Instead of processing a single run_folder, it loops over every sub-folder
% found inside project_root and runs the full workflow on each:
% 1) load image pairs
% 2) preprocess the images
% 3) run the PIV analysis
% 4) postprocess the vector field
% 5) save results and a figure into a mirrored sub-folder under local_folder
%
% The original PIVlab_process_commandline.m is left untouched.


%% Tell MATLAB where the images and the results should live
project_root = '/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/Test_Batch';
local_folder = '/Users/jeromenoir/Documents/MyDocuments/LOCAL_PROJECT/TOPOGRAPHY_LIBRATION/CylinderExperimentsGMA/Test_Batch/results-batch-process';
file_pattern = '*.tif'; % for example '*.bmp', '*.tif', '*.png', '*.jpg'

% addpath(project_root)

%% Find all sub-folders of project_root
dir_listing = dir(project_root);
% Keep only directories, drop '.' and '..' and hidden folders.
is_run_folder = [dir_listing.isdir] & ~ismember({dir_listing.name}, {'.', '..'}) ...
    & ~startsWith({dir_listing.name}, '.');
run_folders = {dir_listing(is_run_folder).name}';
run_folders = sortrows(run_folders);

if isempty(run_folders)
    error('No sub-folders found in project_root: %s', project_root)
end

disp(['Found ' num2str(numel(run_folders)) ' sub-folders to process in: ' project_root])

%% Define preprocessing settings
% These settings are applied to every image before the PIV analysis.
roi_inpt = [];            % [] means: use the full image. Otherwise: [x, y, width, height]
clahe = 1;                % contrast enhancement
clahesize = 64;           % size of the local contrast tiles
highp = 1;                % high-pass filter
highpsize = 15;           % size of the high-pass filter
intenscap = 0;            % intensity capping
wienerwurst = 0;          % Wiener filter
wienerwurstsize = 3;      % Wiener filter size
minintens = 0.0;          % lower intensity limit
maxintens = 1.0;          % upper intensity limit

%% Define PIV analysis settings
% Required for piv.piv_FFTmulti are:
% image1, image2, interrogationarea
%
% The remaining arguments are shown explicitly here so it is clear how they
% can be controlled from a script.
interrogationarea = 64;   % first pass interrogation window size
step = 32;                % distance between neighboring vectors
subpixfinder = 1;         % 1 = 3-point Gauss, 2 = 2D Gauss
mask_inpt = [];           % [] means: no mask. Otherwise: logical matrix, same size as images (true = masked out)
passes = 2;               % number of passes
int2 = 32;                % interrogation area in pass 2
int3 = 16;                % interrogation area in pass 3
int4 = 16;                % interrogation area in pass 4
imdeform = '*linear';     % '*linear' or '*spline'
repeat = 0;               % repeated correlation
mask_auto = 0;            % disable autocorrelation in first pass
do_linear_correlation = 0;% 0 = circular, 1 = linear
repeat_last_pass = 0;     % repeat the last pass
delta_diff_min = 0.025;   % stop repeated last pass below this improvement
limit_peak_search_area = 1;% 1 = limit peak search to central region (recommended), 0 = search full correlation map

%% Define postprocessing settings
% These settings are applied to the raw vector field after the PIV analysis.
calu = 1;                           % calibration factor for u
calv = 1;                           % calibration factor for v
valid_vel = [-50; 50; -50; 50];     % [u_min; u_max; v_min; v_max]
do_stdev_check = 1;                 % global standard deviation check
stdthresh = 7;                      % threshold for the standard deviation check
do_local_median = 1;                % local median check
neigh_thresh = 3;                   % threshold for the local median check
paint_nan = 1;                      % fill filtered vectors by interpolation

%% Main loop over all run folders
for folder_idx = 1:numel(run_folders)
    run_folder = run_folders{folder_idx};
    image_folder = fullfile(project_root, run_folder);

    results_folder = fullfile(local_folder, run_folder);
    file_results = fullfile(results_folder, 'PIVlab_results_uncalibrated.mat');
    file_figure = fullfile(results_folder, 'PIVlab_figure_uncalibrated_firstFrame.jpg');

    disp(' ')
    disp(['==== Run folder ' num2str(folder_idx) ' of ' num2str(numel(run_folders)) ...
        ': ' run_folder ' ===='])

    %% Find the images in this run folder
    disp(['Looking for ' file_pattern ' files in: ' image_folder])
    image_files = dir(fullfile(image_folder, file_pattern));
    image_names = {image_files.name}';
    image_names = sortrows(image_names);

    if isempty(image_names)
        warning('No images found in %s. Skipping this folder.', image_folder)
        continue
    end

    if mod(numel(image_names), 2) ~= 0
        warning(['Folder %s contains an odd number of images (%d). ' ...
            'PIVlab analyzes image pairs, so this folder is skipped.'], ...
            image_folder, numel(image_names))
        continue
    end

    num_pairs = numel(image_names) / 2;
    disp(['Found ' num2str(numel(image_names)) ' images, i.e. ' num2str(num_pairs) ' image pairs.'])

    %% Create the results folder
    if ~exist(results_folder, 'dir')
        mkdir(results_folder);
        disp(['Created results folder: ' results_folder])
    end

    % Copy auxiliary files from image_folder to results_folder if they exist
    auxFiles = {'acquisition_log.txt', 'background.mat','PIVlab_Capture_Session.mat'};
    for k = 1:numel(auxFiles)
        src = fullfile(image_folder, auxFiles{k});
        dst = fullfile(results_folder, auxFiles{k});
        if exist(src, 'file')
            try
                copyfile(src, dst);
                disp(['Copied ' auxFiles{k} ' to results folder.'])
            catch ME
                warning('Failed to copy %s: %s', auxFiles{k}, ME.message)
            end
        end
    end

    %% Prepare result variables for this folder
    % All results are stored in 3D matrices:
    % first dimension = vertical position
    % second dimension = horizontal position
    % third dimension = image pair number / time step
    x = [];
    y = [];
    u = [];
    v = [];
    typevector = [];
    correlation_map = [];
    u_filt = [];
    v_filt = [];
    typevector_filt = [];

    %% Loop over all image pairs in this folder
    for pair_idx = 1:num_pairs
        % Images 1+2 are the first pair, 3+4 the second pair, and so on.
        image_name_1 = image_names{2 * pair_idx - 1};
        image_name_2 = image_names{2 * pair_idx};

        disp(['Processing pair ' num2str(pair_idx) ' of ' num2str(num_pairs) ...
            ': ' image_name_1 ' and ' image_name_2])

        %% Load the raw images
        image1_raw = imread(fullfile(image_folder, image_name_1));
        image2_raw = imread(fullfile(image_folder, image_name_2));

        %% Preprocess the raw images
        image1_preprocessed = preproc.PIVlab_preproc( ...
            in=image1_raw, ...
            roirect=roi_inpt, ...
            clahe=clahe, ...
            clahesize=clahesize, ...
            highp=highp, ...
            highpsize=highpsize, ...
            intenscap=intenscap, ...
            wienerwurst=wienerwurst, ...
            wienerwurstsize=wienerwurstsize, ...
            minintens=minintens, ...
            maxintens=maxintens);

        image2_preprocessed = preproc.PIVlab_preproc( ...
            in=image2_raw, ...
            roirect=roi_inpt, ...
            clahe=clahe, ...
            clahesize=clahesize, ...
            highp=highp, ...
            highpsize=highpsize, ...
            intenscap=intenscap, ...
            wienerwurst=wienerwurst, ...
            wienerwurstsize=wienerwurstsize, ...
            minintens=minintens, ...
            maxintens=maxintens);

        %% Run the actual PIV analysis
        [x(:,:,pair_idx), y(:,:,pair_idx), u(:,:,pair_idx), v(:,:,pair_idx), ...
            typevector(:,:,pair_idx), correlation_map(:,:,pair_idx)] = piv.piv_FFTmulti( ...
            image1=image1_preprocessed, ...
            image2=image2_preprocessed, ...
            interrogationarea=interrogationarea, ...
            step=step, ...
            subpixfinder=subpixfinder, ...
            mask_inpt=mask_inpt, ...
            roi_inpt=roi_inpt, ...
            passes=passes, ...
            int2=int2, ...
            int3=int3, ...
            int4=int4, ...
            imdeform=imdeform, ...
            repeat=repeat, ...
            mask_auto=mask_auto, ...
            do_linear_correlation=do_linear_correlation, ...
            repeat_last_pass=repeat_last_pass, ...
            delta_diff_min=delta_diff_min, ...
            limit_peak_search_area=limit_peak_search_area);

        %% Postprocess the vector field
        [u_filt(:,:,pair_idx), v_filt(:,:,pair_idx)] = postproc.PIVlab_postproc( ...
            u=u(:,:,pair_idx), ...
            v=v(:,:,pair_idx), ...
            calu=calu, ...
            calv=calv, ...
            valid_vel=valid_vel, ...
            do_stdev_check=do_stdev_check, ...
            stdthresh=stdthresh, ...
            do_local_median=do_local_median, ...
            neigh_thresh=neigh_thresh);

        % Keep track of which vectors were filtered.
        typevector_filt_slice = typevector(:,:,pair_idx);
        typevector_filt_slice(isnan(u_filt(:,:,pair_idx))) = 2;
        typevector_filt_slice(isnan(v_filt(:,:,pair_idx))) = 2;
        typevector_filt_slice(typevector(:,:,pair_idx) == 0) = 0;
        typevector_filt(:,:,pair_idx) = typevector_filt_slice;

        % Optionally fill filtered vectors by interpolation.
        if paint_nan
            u_filt(:,:,pair_idx) = misc.inpaint_nans(u_filt(:,:,pair_idx), 4);
            v_filt(:,:,pair_idx) = misc.inpaint_nans(v_filt(:,:,pair_idx), 4);
        end
    end

    % Ensure masked regions stay NaN (inpaint_nans may have filled them).
    u_filt(typevector == 0) = NaN;
    v_filt(typevector == 0) = NaN;

    disp(['Processing finished for run folder: ' run_folder])

    %% Plot 1: one single velocity field (first image pair), saved to file
    fig = figure('Visible', 'off');
    hold on
    quiver(x(:,:,1), y(:,:,1), ...
        u_filt(:,:,1), v_filt(:,:,1), 'g')
    hold off
    title(['Filtered velocity field of image pair 1 - ' run_folder], 'Interpreter', 'none')
    saveas(fig, file_figure);
    close(fig)

    %% Save all results for this run folder
    save(file_results, ...
        'x', 'y', 'u', 'v', 'typevector', 'correlation_map', ...
        'u_filt', 'v_filt', 'typevector_filt', ...
        'image_folder', 'results_folder', 'image_names', 'num_pairs', ...
        'run_folder', 'file_pattern', ...
        'roi_inpt', 'clahe', 'clahesize', 'highp', 'highpsize', 'intenscap', ...
        'wienerwurst', 'wienerwurstsize', 'minintens', 'maxintens', ...
        'interrogationarea', 'step', 'subpixfinder', 'mask_inpt', 'passes', ...
        'int2', 'int3', 'int4', 'imdeform', 'repeat', 'mask_auto', ...
        'do_linear_correlation', 'repeat_last_pass', 'delta_diff_min', ...
        'limit_peak_search_area', ...
        'calu', 'calv', 'valid_vel', 'do_stdev_check', 'stdthresh', ...
        'do_local_median', 'neigh_thresh', 'paint_nan');
    disp(['Saved results to: ' file_results])
end

disp(' ')
disp('Batch processing finished for all run folders.')
