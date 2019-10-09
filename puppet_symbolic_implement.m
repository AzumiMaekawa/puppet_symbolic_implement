clear all;
close all;
clc;

global n n_p n_s m M g T plannningHorizon h s_0 k epsilon stringsPairs;
n = 1; % Pendulum casea
n_p = 1; % number of attachment points
n_s = 1; % number of strings
m = 1.0;
M = eye(3) * m; % Mass matrix
g = [0; 0; -9.81];
T = 40; % "The number of time steps was 40 in all cases"
plannningHorizon = 1.0; % [s]
h = plannningHorizon / T; % step size

s_0 = [0.5]; % lengthes of the strings
k = 1e4; % spring constant
epsilon = 0.001;
stringsPairs = reshape([1 1], [2, n_s]);
% linkPairs = reshape([1 2 2 3], [2,2]);


global psy p_i_sym x_i_sym x_im1_sym x_im2_sym;
syms psy(s);
psy(s) = piecewise(s>0, (1/2)*s^2 + (epsilon/2)*s + epsilon^2 / 6,...
            0>s>-epsilon, (1/(6*epsilon))*s^3 + (1/2)*s^2 + (epsilon/2)*s + epsilon^2/6, ...
            0);

x_i_sym = sym('x%d', [3*n 1]);
p_i_sym = sym('p%d', [3*n_p 1]);
x_im1_sym = sym('x%d_m1', [3*n 1]);
x_im2_sym = sym('x%d_m2', [3*n 1]);

% compute the time-discretized acceleration
% xdot_i = (x_i_sym - x_im1_sym) / h;
xdot2_i = (x_i_sym - 2 * x_im1_sym + x_im2_sym) / (h^2);

% internal potential deformation energy (strings and trusses)
W = compute_potential_energy(x_i_sym, p_i_sym);

% calculate x_i with implicit Euler time stepping scheme
U_i = (h^2 / 2) * xdot2_i.' * M * xdot2_i + W + g.' * M * x_i_sym;

% compute gradient and hessian
global gradU hessU;
gradU = jacobian(U_i, x_i_sym);
gradU = gradU(:);
hessU = jacobian(gradU, x_i_sym);


% Initialize
x_0 = randn([3*n 1]);
xdot_0 = zeros([3*n 1]);
p = randn([3*n_p*T 1]);

x = forward_sim(p, x_0, xdot_0);
disp(size(x));
fplot(x);


% forward simulation x(p)
function x = forward_sim(p, x_0, xdot_0)
    global n_p T p_i_sym gradU hessU x_im1_sym x_im2_sym;
    x_i = x_0;
    x_im1 = x_0;
    x_im2 = x_0;
    % xdot_i = xdot_0;
    % xdot_im1 = xdot_0;

    x = x_0;
    for i = 1:T
        p_i = p(i:i+3*n_p-1);
        grad_i = subs(gradU, p_i_sym, p_i);
        grad_i = subs(grad_i, x_im1_sym, x_im1);
        grad_i = subs(grad_i, x_im2_sym, x_im2);
        hess_i = subs(hessU, p_i_sym, p_i);
        hess_i = subs(hess_i, x_im1_sym, x_im1);
        hess_i = subs(hess_i, x_im2_sym, x_im2);

        % compute x_i with Newton's method
        x_opt = NewtonsMethod(grad_i, hess_i, x_i, 1e-8, 50);

        x_im2 = x_im1;
        x_im1 = x_i;
        x_i = x_opt;
        % xdot_im1 = xdot_i;
        % xdot_i = (x_i - x_im1_sym) / h;
        x = [x; x_i];
    end
end

function x_opt = NewtonsMethod(grad_i, hess_i, x_init, err, max_iteration)
    global x_i_sym
    % for regularization
    r = 1e-6;
    I = eye(size(x_init,1));

    x_k = x_init;

    cnt = 0;
    while(1)
        cnt = cnt + 1;
        grad_k = subs(grad_i, x_i_sym, x_k);
        hess_k = subs(hess_i, x_i_sym, x_k);
        dx = -inv(hess_k + r*I) * grad_k;
        x_k = x_k + dx;
        if norm(dx) < err
            x_opt = x_k;
            break
        elseif cnt == max_iteration
            x_opt = x_k;
            break
        end
    end
end
