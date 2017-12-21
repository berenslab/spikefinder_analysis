
%% Define global variable for the main data folder 

% This is the folder that contains the spikefinder.test and
% spikefinder.test sub-folders. Additional folders will be created in it by
% the function spf_folders.

global SpikeFinderDataDir
SpikeFinderDataDir = 'E:\Users\Thomas\Thomas\PROJECTS\1611_SpikeFinder\data';

%% Choose a dataset

dataset = 1;

%% Train on the specified dataset
% Note that the current procedure for training MLspike takes hours if not
% days...

% When loading dataset 2, a few manual corrections are applied to the data
% to remove big jumps. The program is notified to apply these corrections
% by passing dataflag 92 instead of 2
if dataset==2
    dataflag = 92;
else
    dataflag = dataset;
end

% While some parameters are trained, some others are fixed manually.
% Different manual parameter sets are defined in function spf_parameters in
% the form of a 3-digits code called 'methodflag'. 
% Briefly, the 1st digits indicates whether to use MAP, proba or samples
% output (at the end, proba output where systematically preferred), define
% the type of nonlinearity (sub-linear for OGB, supra-linear for GCamp) and
% whether to use heuristics for estimating noise level. The 2nd digit sets
% some internal parameters of the algorithm. The 3rd digit defines temporal
% binning and some additional optional parameters.
% Below are indicated the best methodflags found for each dataset, but
% others can be tested as well.
best_method_flags = [223 223 523 423 623 623 523 623 525 623];
methodflag = best_method_flags(dataset);

% Go!
% Results are stored regularly to the 'precomp' folder, so that
% computations can be resumed after an interruption, just by calling the
% same command (but 'fromscratch' option forces a restart from zero).
% Range for the parameters being estimated are defined in function
% spf_getrange.
spf_train(dataflag, methodflag)

% It is possible to execute the training in a separate Matlab process with
% the 'batch' flag, and to recruit N workers for parallel computing with
% 'batchN' flag.
% spf_train(dataflag, methodflag, 'batch6')

%% During the training

% The state of the current training can be displayed with functions
% spf_summary, spf_displayEvalPoints and spf_monitor (see their help for
% options)
spf_summary

%% Run on the test data and publish everything

% The following function will first publish the training results (in the
% 'submissions' folder), and then if test data exist for this dataset, run
% MLspike on it with the current best parameters and publish the results.
% More details on the test data estimations will also be saved in the
% 'testres' folder.
spf_publish(dataset)


