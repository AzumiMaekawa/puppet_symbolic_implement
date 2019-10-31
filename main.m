clearvars;
close all;
clc;

%% Init to prepare the values to simualte
global n n_p n_s m M g T plannningHorizon h debug;
global w_traj w_acc w_cyc s_0 k epsilon stringsPairs;
n = 1; % number of mass points. n = 1 means a pendulum case
n_p = 1; % number of attachment points
n_s = 1; % number of strings
m = 1.0;
M = eye(3*n) * m; % Mass matrix
g = [0; 0; 9.81];
T = 40; % "The number of time steps was 40 in all cases" from the paper
plannningHorizon = 1.0; % [s]
h = plannningHorizon / T; % step size
debug = false; % Enable debug mode to see what's happening

% weight for objective
w_traj = 0.15;
w_acc = 0.000001;
w_cyc = 1;

s_0 = 0.5; % lengthes of the strings
k = 1e4; % spring constant
epsilon = 0.001;
stringsPairs = reshape([1 1], [2, n_s]);
% linkPairs = reshape([1 2 2 3], [2,2]);

%% Compute gradient and hessian for forward simulation
global psy pi_sym xi_sym xim1_sym xim2_sym gradU hessU;

if(debug)
    idx = linspace(-5,5,200);
    result = eval(psy(idx));
    plot(idx, result);
    title("string mode :: psy(s) i.e. piece-wise function");
    xlim([-5 5]); ylim([-2 14]);
end
    
xi_sym = sym('x%d', [3*n 1]);
pi_sym = sym('p%d', [3*n_p 1]);
xim1_sym = sym('x%d_m1', [3*n 1]);
xim2_sym = sym('x%d_m2', [3*n 1]);

% compute the time-discretized acceleration
xdot2_i = (xi_sym - 2 * xim1_sym + xim2_sym) / (h^2);

% internal potential deformation energy (strings and trusses)
W = compute_potential_energy(xi_sym, pi_sym);

% To calculate x_i with implicit Euler time stepping scheme
U_i = (h^2 / 2) * xdot2_i.' * M * xdot2_i + W + g.' * M * xi_sym;
% compute gradient and hessian of U_i
gradU = jacobian(U_i, xi_sym);
gradU = gradU(:);
hessU = jacobian(gradU, xi_sym);

% compute G_i and its derivative
% G_i depends only on x_i, x_im1, x_im2, and p_i
global DGi_Dxi DGi_Dxim1 DGi_Dxim2 DGi_Dpi
Gi = gradU; % G_i is equal to gradient of U_i
DGi_Dxi = hessU; % partial G / partial x_i (equal to hessian)
DGi_Dxim1 = jacobian(Gi, xim1_sym);
DGi_Dxim2 = jacobian(Gi, xim2_sym);
DGi_Dpi = jacobian(Gi, pi_sym);



% Initialize
x_0 = randn([3*n 1]) * 0.0001;
x_m1 = randn([3*n 1]) * 0.0001;
t = linspace(0, plannningHorizon, T);
% p_x = t.*t;
p_x = linspace(0, 0.7, T);
%p_y = sin(t*pi/2);
p_y = zeros([1 T]);
p_z = zeros([1 T]) + s_0(1);
p = [p_x; p_y; p_z];
p = p(:);
% specify target trajecory
syms f(s);
f(s) = 519*s^7 + -1471*s^6 + 1632*s^5 - 900.6*s^4 + 265.1*s^3 - ...
        44.22*s^2 + 4.268*s + 0.00026816;
target_x = linspace(0, 0.7, T);
target_y = zeros([1 T]);
target_z = double(f(target_x));
%target_x = [linspace(0, 0.25, T/2) linspace(0.25, 0, T/2)];
%target_y = zeros([1 T]);
%target_z = zeros([1 T]);
x_target = [target_x;target_y;target_z];
x_target = x_target(:);

% for regularization
r = 1e-5;
I = eye(size(3*n_p*T)); % same size as H

%% compute 2nd derivative
% compute DO2_Dx2
DO2_Dx2 = 2 * eye(3*n*T) * w_traj;
% O does not explicitly couple x and p,
% and therefore the mixed derivative term vanishes.
DO2_DxDp = zeros([3*n*T 3*n_p*T]);
DO2_Dp2 = compute_DO2Dp2();


%% Main Processing 
counter = 0;
max_main_iteration = 10;
objective_criterion = 1;
cost_array = zeros([max_main_iteration 1]);

x = forward_sim(p, x_0, x_m1);
while(1)
    counter = counter + 1;
    S = compute_dxdp(x, p, x_0, x_m1);
    dO_dp = compute_dOdp(x, x_target, p, S, false);

    % compute H
    H = S.' * DO2_Dx2 * S + 2 * S.' * DO2_DxDp + DO2_Dp2;

    d = -inv(H + r*I) * dO_dp;
    [p, x] = line_search(p, d, 1, 0.1, x, x_target, x_0, x_m1);

    O = compute_objective(x, x_target, p, false);
    fprintf("main loop :: %d, O: %f\n", counter, O);

    cost_array(counter) = O;

    % if O < objective_criterion
    %     fprintf("optimization completed!\n")
    %     break
    % end
    if counter >= max_main_iteration
        fprintf("reach max iteration!\n")
        break
    end
end
disp(O);


%% draw results
% objective
draw_results();
