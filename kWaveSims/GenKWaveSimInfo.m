clear
clc

% Load Functions
addpath(genpath('../k-Wave'));
addpath(genpath(pwd));

% Define Transducer + Element Discretization
xdcr_len = 0.0256; % Total Length of Transducer [m]
no_elements = 128; % Total Number of Transducer Elements
no_subelem = 5; % Number of Mathematical Subelements per Element
dx = xdcr_len/(no_elements*no_subelem); % Width of Subelement (Ignoring Kerf)

% Simulation Grid
dz = dx; % Axial Grid Spacing Should Be Same For Easy Use of Eikonal Solver
lateral_flank = 0.005; % Distance to Left and Right of Transducer [m]
axial_depth = 0.050; % Axial Depth for Imaging and Simulation [m]
total_lateral = xdcr_len + 2*lateral_flank; % Total Lateral Span [m]
x = 0:dx:total_lateral; x = x-mean(x); % Centered Lateral Coordinates [m]
z = 0:dz:axial_depth; % Axial Depth Coordinates [m]
[X, Z] = meshgrid(x, z); % Mesh for Lateral and Axial Coordinates [m]
Nx = numel(x); Nz = numel(z); % Dimensions of Simulation Grid

% Create Sound Speed Map from Abdominal Map
materials2; % File holding material properties for different tissue types
map_number = 1; % Choice of Abdominal Map
[C, rho, A, N, rho0, c0] = imagemap_params('mast/', numel(x), dx, ... 
    numel(z), dz, tissue, fat, muscle, connective, map_number); 
C = C'; % Transpose Sound Speed Map

% Blur Sound Speed Map to Remove Abrupt Interfaces
reslen = 5e-4; % Resolution Length [m]
C = imgaussfilt(C, reslen/dx); % Blurred Sound Speed Map

% Assign Grid Points to Each Transducer Elements for Transmit
start_elmt = ceil(lateral_flank/dx);
start_subelmt = start_elmt + no_subelem * (1:no_elements);
tx_src = zeros(2, no_subelem, no_elements);
x_xdcr = zeros(no_elements, no_subelem);
for elmt = 1:no_elements
    curr_subelmts = start_subelmt(elmt):start_subelmt(elmt)+no_subelem-1;
    tx_src(:, :, elmt) = [ones(1, no_subelem); curr_subelmts];
    x_xdcr(elmt, :) = x(curr_subelmts);
end

% Make Transducer Element a Point Receiver on Receive
rx_src = tx_src(:, ceil(no_subelem/2), :);

% Load Focusing Functions
addpath(genpath(pwd));
addpath(genpath('../../k-Wave'));

% Load All Information Needed to Plug Into K-Wave
% SimInfoKWave.m    File to Run Before Running Intensive K-Wave Simulation 
% Sets up all parameters for medium, transducer, transmit signals, receive 
% and transmit (angular) sensitivity, and computational grid for k-Wave    

% Define Transducer
x_xdcr = x_xdcr'; % X-Coordinates For Each Grid Point on Each Transducer Elements

% Create Computational Grid
Nx = numel(x); % Number of Lateral Grid Points
Nz = numel(z); % Number of Axial Grid Points
kgrid = makeGrid(Nz, dz, Nx, dx); % K-Space Grid Object

% Put Transducer on Computational Grid   
source.u_mask = zeros(Nz, Nx);
z_idx = tx_src(1, :, :); z_idx = z_idx(:);
x_idx = tx_src(2, :, :); x_idx = x_idx(:);
idx = sub2ind([Nz, Nx], z_idx, x_idx);
source.u_mask(idx) = 1;

% Define the Properties of the Propagation Medium
rho = 1000; % Density [kg/m^3]
c = 1540; % Mean Sound Speed for Medical Ultrasound [m/s]
impedance = c*rho; % Acoustic Impedance [kg/((m^2)(s))]
medium_isoimp.sound_speed = C; % [m/s]
medium_isoimp.density = impedance./C; % [kg/m^3]
medium_isoimp.alpha_coeff = 0*ones(Nz, Nx); % [dB/(MHz^y cm)]
medium_isoimp.alpha_power = 1.5;

% Create Time Array
t_end = 2 * Nz * dz / c; cfl = 0.3;
[kgrid.t_array, dt] = makeTime(kgrid, c, cfl, t_end);
fs = 1/dt; % Sampling Frequency [Hz]

% Define Properties of the Tone Burst Used to Drive the Transducer

% Transmit Pulse and Impulse Response
fracBW = 0.7;  % Fractional Bandwidth of the Transducer
fTx = 8.0e6; % Transmit Frequency [Hz]
tc = gauspuls('cutoff', fTx, fracBW, -6, -80); % Cutoff time at -80dB, fracBW @ -6dB
t = (-ceil(tc/dt):1:ceil(tc/dt))*dt; % (s) Time Vector centered about t=0
src_amplitude = 10;
tx_signal = (src_amplitude/(rho*c)) * gauspuls(t, fTx, fracBW); % Calculate Transmit Pulse
[~, emission_samples] = size(tx_signal); 
t_offset = -(emission_samples-1)*dt/2; 

% Define a Binary Sensor Mask
sensor.mask = source.u_mask; % Make Sensor the Same As Source

% Define the Angle of Max Directivity for Each Sensor Point:
%    0             = Max Sensitivity in x Direction (Up/Down)
sensor.directivity_angle = zeros(Nz,Nx);

% Define the Directivity Size
sensor.directivity_size = kgrid.dy;

% Define the Directivity Pattern
sensor.directivity_pattern = 'gradient';

% Create a Display Mask to Display the Transducer
display_mask = source.u_mask;

% Create Isoimpedance Medium
medium = medium_isoimp;

% More Pulse Information
omega0 = fTx*2*pi; % Center Radian Frequency of Transmitted Wave [rad/s]
ncycles = 1/fracBW; % Number of Cycles as Function of Bandwidth
% Scatter Properties
num_scat = 24;
scat_size = 40e-6;
scale = 0.005; % Max Scat Variation from rho
[scat, seedUsed] = scatfield(num_scat, scat_size, scale, rho, omega0, ...
    axial_depth, xdcr_len, ncycles, dx, dz, Nx, Nz);

% Create Anechoic Lesion in Isoimpedance Medium
rhoDeviation = scat' - rho;
dB_lesion = -6;
lesion_radius = 1.5e-3; % [m]
for lesion_ctr_x = (-4.5e-3):(4.5e-3):(4.5e-3)
    for lesion_ctr_z = 27e-3+((-4.5e-3):(4.5e-3):(4.5e-3))
        lesion_pixels = ( ( (X-lesion_ctr_x).^2 + (Z-lesion_ctr_z).^2 ) <= lesion_radius^2 );
        rhoDeviation(lesion_pixels) = (10^(dB_lesion/20))*rhoDeviation(lesion_pixels);
    end
end
medium.density = medium.density + rhoDeviation;

% Create Grid of Point Targets
numPtTargsPerRow = 5; % Number of Point Targets Per Row
numRowsOfPtTargs = 5; % Number of Rows of Point Targets
xlimPtTargs = [-7.5e-3, 7.5e-3]; % X Location Bounds of Point Targets
zlimPtTargs = [18.0e-3, 36.0e-3]; % Z Location Bounds of Point Targets
ptTarg_x = linspace(xlimPtTargs(1), xlimPtTargs(2), numRowsOfPtTargs); % Lateral Locations of Point Targets [m]
ptTarg_z = linspace(zlimPtTargs(1), zlimPtTargs(2), numPtTargsPerRow); % Axial Locations of Point Targets [m]
[ptTargX, ptTargZ] = meshgrid(ptTarg_x, ptTarg_z);
ptsToRemove = (ptTargX>xlimPtTargs(1)+mean(diff(ptTarg_x))/2) & ...
    (ptTargX<xlimPtTargs(2)-mean(diff(ptTarg_x))/2) & ...
    (ptTargZ>zlimPtTargs(1)+mean(diff(ptTarg_z))/2) & ...
    (ptTargZ<zlimPtTargs(2)-mean(diff(ptTarg_z))/2); % Removing Interior Point Targets
ptTargX(ptsToRemove) = []; ptTargZ(ptsToRemove) = []; % Leaving Behind Outer Edge of Point Targets
ptTargXIdx = zeros(1, numel(ptTargX));
ptTargZIdx = zeros(1, numel(ptTargZ));
for k = 1:numel(ptTargX)
    [~, ptTargXIdx(k)] = min(abs(x-ptTargX(k)));
    [~, ptTargZIdx(k)] = min(abs(z-ptTargZ(k)));
end
ptTargIdx = sub2ind([Nz, Nx], ptTargZIdx, ptTargXIdx);
medium.density(ptTargIdx) = 1080;

% Assign the Input Options
input_args = {'DisplayMask', display_mask, 'PMLInside', false, 'PlotPML', false, 'PMLAlpha', 10, 'PlotSim', false, 'DataCast', 'gpuArray-single'};

% Generate Receive Data for Each Transmit Element (Full Synthetic Aperture)
tx_elements = 1:1:no_elements;
ux_src_elmts = zeros([no_subelem*no_elements, emission_samples, numel(tx_elements)]); 
% Assemble Each Element Source Prior to ParFor Loop
for tx_elmt_idx = 1:numel(tx_elements)
    tx_elmt = tx_elements(tx_elmt_idx);
    ux_src_elmts( (tx_elmt-1)*no_subelem + (1:no_subelem), :, tx_elmt_idx) = ...
        ones(no_subelem, 1) * tx_signal;
end

% Save All Arrival Times Calculated by Eikonal Equation
filename = ['sim_info/SimInfo', num2str(map_number), '.mat'];
save(filename); % Save with name containing bottom layer sound speed