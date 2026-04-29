clear
clc

% Assemble RF Data From Individual Tx Beams Into Full Synthetic Aperture
addpath(genpath(pwd)); % Include Functions from all Subfolders
addpath(genpath('../k-Wave')); % Load k-Wave

% Load Info Needed for K-Wave Simulation
map_number = 1; % Choice of Abdominal Map
filename = ['sim_info/SimInfo', num2str(map_number), '.mat'];
load(filename)

% Build up the Full-Synthetic Aperture Dataset
fsa_rf_data = zeros(numel(kgrid.t_array), no_elements, numel(tx_elements));
for tx_elmt_idx = 1:numel(tx_elements)
    filename = ['scratch/rf_data_tx_elem_', num2str(tx_elmt_idx), '.mat'];
    data = load(filename); % Load RF Data From This Transmit Beam
    assert(data.map_number == map_number, "Mismatched Abdominal Map Number/ID")
    fsa_rf_data(:, :, tx_elmt_idx) = data.rf_data; % Assemble
    disp(['Assembled ' num2str(tx_elmt_idx), ' Transmit Beam']);
end
load(filename); % Load All Data
clear rf_data; % Get Rid of Single Transmit RF Data

% Calculate Variables Needed for Output File
downsamp = 2; % Downsampling Rate (in Time)
x_ctr_xdcr = x_xdcr((end+1)/2, :);
pitch = mean(diff(x_ctr_xdcr));
rxAptPos = [x_ctr_xdcr(:), zeros(no_elements, 1), zeros(no_elements, 1)];
scat = fsa_rf_data(1:downsamp:end, :, :); 
time = kgrid.t_array+t_offset; time = time(1:downsamp:end);
% Save Data for Output File
filename = ['datasets/AbdominalMap', num2str(map_number), '.mat'];
save(filename, '-v7.3', 'no_elements', 'pitch', 'rxAptPos', 'scat', 'time', 'x', 'z', 'C');