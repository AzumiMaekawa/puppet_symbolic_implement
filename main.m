clear all;
close all;
clc;

%% Init to prepare the values to simualte
global n n_p n_s m M g T plannningHorizon h s_0 k epsilon stringsPairs debug;
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

s_0 = [0.5]; % lengthes of the strings
k = 1e4; % spring constant
epsilon = 0.001;
stringsPairs = reshape([1 1], [2, n_s]);
% linkPairs = reshape([1 2 2 3], [2,2]);

%% Compute gradient and hessian for forward simulation
global psy p_i_sym xi_sym xim1_sym xim2_sym gradU hessU;

if(debug)
    idx = linspace(-5,5,200);
    result = eval(psy(idx));
    plot(idx, result);
    title("string mode :: psy(s) i.e. piece-wise function");
    xlim([-5 5]); ylim([-2 14]);
end
    
xi_sym = sym('x%d', [3*n 1]);
p_i_sym = sym('p%d', [3*n_p 1]);
xim1_sym = sym('x%d_m1', [3*n 1]);
xim2_sym = sym('x%d_m2', [3*n 1]);

% compute the time-discretized acceleration
xdot2_i = (xi_sym - 2 * xim1_sym + xim2_sym) / (h^2);

% internal potential deformation energy (strings and trusses)
W = compute_potential_energy(xi_sym, p_i_sym);

% To calculate x_i with implicit Euler time stepping scheme
U_i = (h^2 / 2) * xdot2_i.' * M * xdot2_i + W + g.' * M * xi_sym;
% compute gradient and hessian of U_i
gradU = jacobian(U_i, xi_sym);
gradU = gradU(:);
hessU = jacobian(gradU, xi_sym);

% compute G_i and its derivative
% G_i depends only on x_i, x_im1, x_im2, and p_i
global Gi_xi Gi_xim1 Gi_xim2 Gi_pi
Gi = gradU; % G_i is equal to gradient of U_i
Gi_xi = hessU; % partial G / partial x_i (equal to hessian)
Gi_xim1 = jacobian(Gi, xim1_sym);
Gi_xim2 = jacobian(Gi, xim2_sym);
Gi_pi = jacobian(Gi, p_i_sym);



% Initialize
x_0 = randn([3*n 1]) * 0.001;
x_m1 = zeros([3*n 1]);
t = linspace(0, plannningHorizon, T);
% p_x = t.*t;
p_x = linspace(0, 0.5, T);
%p_y = sin(t*pi/2);
p_y = zeros([1 T]);
p_z = zeros([1 T]) + s_0(1);
p = [p_x; p_y; p_z];
p = p(:);
% specify target trajecory
syms f(s);
f(s) = 519*s^7 + -1471*s^6 + 1632*s^5 - 900.6*s^4 + 265.1*s^3 - ...
        44.22*s^2 + 4.268*s + 0.00026816;
target_x = t;
target_y = zeros([1 T]);
target_z = double(f(t));
x_target = [target_x;target_y;target_z];
x_target = x_target(:);


%% Main Processing 
x = forward_sim(p, x_0, x_m1);
O = compute_objective(x, x_target, p, false);
disp(O);


%% show results
x = reshape(x, [3,n*T]);
figure; scatter3(x(1,:),x(2,:),x(3,:)); 
hold on;
grid on;
rotate3d;
% draw initial posotion(red), final position(black)
scatter3(x(1,1), x(2,1), x(3,1), 15,'r','filled'); % position, size, color, filled marker 
scatter3(x(1,n*T), x(2,n*T), x(3,n*T),15,'k','filled');
% draw control trajectory
plot3(p_x, p_y, p_z,'Color', 'g'); % plot control position with green line
for i = 1:T
    plot3([p_x(i) x(1,i)], [p_y(i) x(2,i)], [p_z(1) x(3,i)],'Color','k');
end

title('the simulated location');
xlabel('x'); ylabel('y'); zlabel('z');
 xlim([-0.25 0.75]); ylim([-0.5 0.5]); zlim([-0.5 0.5]);

%% forward simulation x(p)
function x = forward_sim(p, x_0, x_m1)
    global n_p T p_i_sym gradU hessU xim1_sym xim2_sym x_im1 x_im2;
    x_i = x_0;
    x_im1 = x_0;
    x_im2 = x_m1;

    x = [];
    for i = 1:T
        p_i = p((i-1)*3*n_p+1:i*3*n_p);
        grad_i = subs(gradU, p_i_sym, p_i);
        grad_i = subs(grad_i, xim1_sym, x_im1);
        grad_i = subs(grad_i, xim2_sym, x_im2);
        hess_i = subs(hessU, p_i_sym, p_i);
        hess_i = subs(hess_i, xim1_sym, x_im1);
        hess_i = subs(hess_i, xim2_sym, x_im2);

        % compute x_i with Newton's method or gradient descent
        fprintf('forward_sim :: %d/%d\n', i, T);
        % simplified one to see if it can be converged or not.
        x_i = Newtons_method(grad_i, hess_i, x_i, 1e-6, 20); %% -> succeeded to converge
        % x_i = gradient_descent(grad_i, x_i, 1e-5, 50);
        x = [x; x_i];

        x_im2 = x_im1;
        x_im1 = x_i;
    end
end

function O = compute_objective(x, x_target, p, periodic)
    global h n_p T

    % Compute O_traj
    % O_traj measures the similarity between x and the user specified trajectory: x_target.
    O_traj = norm(x - x_target)^2;

    % Compute O_acc
    % O_acc regularizes the acceleration of the string attachment points to promote smooth motion.
    % compute time-discretized acceleration of p
    % pdot_0 = 0, pdot2_0 = 0
    pdot = zeros([3*n_p*T 1]);
    pdot(3*n_p+1:3*n_p*T) = (1/h) * (p(3*n_p+1:3*n_p*T) - p(1:3*n_p*(T-1)));
    pdot2 = zeros([3*n_p*T 1]);
    pdot2(3*n_p+1:3*n_p*T) = (1/h) * (pdot(3*n_p+1:3*n_p*T) - p(1:3*n_p*(T-1)));

    O_acc = norm(pdot2)^2;

    % Compute O_CycPos
    % for a periodic motion
    O_CycPos = 0;
    if periodic == true
        % TODO 
    end

    % return objective
    O = O_traj + O_acc + O_CycPos;
end

%% compute the sensitivity term S = dx/dp
function dx_dp = compute_dOdp(x, p, x_0, x_im1)
    global n n_p Gi_xi Gi_xim1 Gi_xim2 Gi_pi xi_sym xim1_sym xim2_sym p_i_sym

    % compute partial derivative of G with respect to x
    G_x = cell(T); % T*T cell
    for row = 1:T
        for col = 1:T
            if row = col
                % Gi_xi
                G_row_x_col =subs(Gi_xi, xi_sym, x(3*n*(col-1)+1:3*n*col));
                if col =< 2 
                    G_row_x_col = subs(G_row_x_col, xim1_sym, x_0);
                else
                    G_row_x_col = subs(G_row_x_col, xim1_sym, x(3*n*(col-2)+1:3*n*(col-1)));
                end
                if col == 1
                    G_row_x_col = subs(G_row_x_col, xim2_sym, x_im1)
                else
                    G_row_x_col = subs(G_row_x_col, xim2_sym, x(3*n*(col-3)+1:3*n*(col-2)));
                end
            elseif row == col + 1
                % Gi_xim1
                G_row_x_col = subs(Gi_xim1, xi_sym, x(3*n*col+1:3*n*(col+1));
                G_row_x_col = subs(G_row_x_col, xim1_sym, x(3*n*(col-1)+1:3*n*col));
                if col == 1
                   G_row_x_col = subs(G_row_x_col, xim2_sym, x_0);
                else 
                    G_row_x_col = subs(G_row_x_col, xim2_sym, x(e*n*(col-2)+1:3*n*(col-1));
                end
            elseif row == col + 2
                % Gi_xim2
                G_row_x_col = subs(Gi_xim2, xi_sym, x(3*n*(col+1)+1:3*n*(col+2)));
                G_row_x_col = subs(G_row_x_col, xim1_sym, x(3*n*col+1:3*n*(col+1)));
                G_row_x_col = subs(G_row_x_col, xim2_sym, x(3*n*(col-1)+1:3*n*col));
            else
                % other submatrices are zeros because Gi depends xi, xim1, xim2
                G_row_x_col = zeros(3*n);
            end

            G_x{row, col} = G_row_x_col;
        end
    end

    % compute partial derivative of G with respect to p
    G_p = cell(T); % T*T cell
    for row = 1:T
        for col = 1:T
            if row == col
                G_row_p_col = subs(Gi_pi, p_i_sym, p(3*n_p*(col-1)+1:3*n_p*col));
            else
                % other submatrices are zeros because Gi depends on pi
                G_row_p_col = zeros(3*n_p);
            end

            G_p{row, col} = G_row_p_col;
        end
    end

    % convert cell to matrix
    G_x = cell2mat(G_x);
    G_p = cell2mat(G_p);

    % sensitivity term
    dx_dp = -inv(G_x) * G_p;
end