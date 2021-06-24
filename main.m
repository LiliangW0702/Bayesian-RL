clear all 
clc
close all

num_samples = 2000;

% setup the true linear system
state_dim = 1;
Atrue = [[-3]];
Btrue = [[2]];
noise_cov = 1e-2 * np.eye(state_dim);

% setup optimal control
R=10;      % quadratic penalty on controls
Q=1;       % quadratic penalty on states


[P, K] = lqgOptNew(Atrue, Btrue, noise_cov, R, Q);