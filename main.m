clear all;
close all;
clc;

%% Init to prepare the values to simualte
global n n_p n_s m M g T plannningHorizon h s_0 k epsilon stringsPairs debug;
n = 1; % Pendulum casea
n_p = 1; % number of attachment points
n_s = 1; % number of strings
m = 1.0;
M = eye(3) * m; % Mass matrix
g = [0; 0; 9.81];
T = 40; % "The number of time steps was 40 in all cases"
plannningHorizon = 1.0; % [s]
h = plannningHorizon / T; % step size
debug = false; % Enable debug mode to see what's happening

s_0 = [0.5]; % lengthes of the strings
k = 1e4; % spring constant
epsilon = 0.001;
stringsPairs = reshape([1 1], [2, n_s]);
% linkPairs = reshape([1 2 2 3], [2,2]);

%% String model creation
global psy p_i_sym x_i_sym x_im1_sym x_im2_sym;
syms psy(s);
psy(s) = piecewise(0<s, (1/2)*s^2 + (epsilon/2)*s + epsilon^2 / 6,...
            -epsilon<s<0, (1/(6*epsilon))*s^3 + (1/2)*s^2 + (epsilon/2)*s + epsilon^2/6, ...
            0);

if(debug)
    idx = linspace(-5,5,200);
    result = eval(psy(idx));
    plot(idx, result);
    title("string mode :: psy(s) i.e. piece-wise function");
    xlim([-5 5]); ylim([-2 14]);
end
    
x_i_sym = sym('x%d', [3*n 1]);
p_i_sym = sym('p%d', [3*n_p 1]);
x_im1_sym = sym('x%d_m1', [3*n 1]);
x_im2_sym = sym('x%d_m2', [3*n 1]);

%%  compute the time-discretized acceleration
xdot2_i = (x_i_sym - 2 * x_im1_sym + x_im2_sym) / (h^2);

%%  internal potential deformation energy (strings and trusses)
W = compute_potential_energy(x_i_sym, p_i_sym);

% calculate x_i with implicit Euler time stepping scheme
 U_i = (h^2 / 2) * xdot2_i.' * M * xdot2_i + W + g.' * M * x_i_sym;
% U_i = (h^2 / 2) * xdot2_i.' * M * xdot2_i + g.' * M * x_i_sym;% + W + g.' * M * x_i_sym;
%%  compute gradient and hessian
global gradU hessU;
gradU = jacobian(U_i, x_i_sym);
gradU = gradU(:);
hessU = jacobian(gradU, x_i_sym);

%% Main Processing 
% Initialize
x_0 = zeros([3*n 1]);
xdot_0 = zeros([3*n 1]);
p_x = linspace(0, 0.5, n_p*T);
%p_x = zeros([1 n_p*T]);
p_y = zeros([1 n_p*T]);
p_z = zeros([1 n_p*T]) + s_0(1);
p = [p_x; p_y; p_z];
p = p(:);

x = forward_sim(p, x_0, xdot_0);

% show results
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
% xlim([-1 1]); ylim([-1 1]); zlim([-1 1]);

%% forward simulation x(p)
function x = forward_sim(p, x_0, xdot_0)
    global h n_p T p_i_sym gradU hessU x_im1_sym x_im2_sym x_im1 x_im2;
    x_i = x_0;
    x_im1 = x_0;
    x_im2 = x_0;

    x = x_0;
    for i = 1:(T-1)
        p_i = p((i-1)*3*n_p+1:i*3*n_p);
        grad_i = subs(gradU, p_i_sym, p_i);
        grad_i = subs(grad_i, x_im1_sym, x_im1);
        grad_i = subs(grad_i, x_im2_sym, x_im2);
        hess_i = subs(hessU, p_i_sym, p_i);
        hess_i = subs(hess_i, x_im1_sym, x_im1);
        hess_i = subs(hess_i, x_im2_sym, x_im2);

        % compute x_i with Newton's method
        fprintf('forward_sim :: %d/%d\n', i, T);
        % simplified one to see if it can be converged or not.
        %x_i = Newtons_method(grad_i, hess_i, x_i, 1e-5, 20); %% -> succeeded to converge
        x_i = gradient_descent(grad_i, x_i, 1e-5, 50);
        x = [x; x_i];

        x_im2 = x_im1;
        x_im1 = x_i;
    end
end

%% Newton Method to solve the optimized value
function x_opt = Newtons_method(grad_i, hess_i, x_init, err, max_iteration)
    global x_i_sym x_im1 x_im2 p_i
    % for regularization
    r = 1e-5;
    I = eye(size(x_init,1));

    x_k = x_init;

    cnt = 0;
    while(1)
        cnt = cnt + 1;
        fprintf('Newtons method :: %d/%d\n', cnt, max_iteration)
        grad_k = subs(grad_i, x_i_sym, x_k);
        hess_k = subs(hess_i, x_i_sym, x_k);
        dx = double(-inv(hess_k + r*I) * grad_k);
        disp(dx);
        x_k = double(x_k + dx);
        calcNorm = double(norm(dx));
        if calcNorm < err
            x_opt = x_k;
            fprintf('converged :: %e at x_k[%f, %f, %f] \n', ...
                calcNorm, x_opt(1), x_opt(2), x_opt(3))
            break
        elseif cnt == max_iteration
            x_opt = x_k;
            fprintf('reached to max iteration :: %e at x_k[%f, %f, %f] \n', ...
                calcNorm, x_opt(1), x_opt(2), x_opt(3))
            break
        end
    end
end

function x_opt = gradient_descent(grad_i, x_init, err, max_iteration)
    global x_i_sym x_im1 x_im2

    x_k = x_init;
    r = 5e-4; % learning rate

    cnt = 0;
    while(1)
        cnt = cnt + 1;
        fprintf('Gradient descent :: %d/%d\n', cnt, max_iteration);
        grad_k = subs(grad_i, x_i_sym, x_k);
        disp(grad_k);
        grad_k = double(grad_k);
        x_k = x_k - r * grad_k;
        disp(x_k);
        calcNorm = double(norm(grad_k));
        if calcNorm < err
            x_opt = x_k;
            fprintf('converged :: %e at x_k[%f,%f,%f] \n', ...
                calcNorm, x_opt(1), x_opt(2), x_opt(3)); 
            break
        elseif cnt == max_iteration
            x_opt = x_k;
            fprintf('reached to max iteration :: %e at x_k[%f,%f,%f] \n', ...
                calcNorm, x_opt(1), x_opt(2), x_opt(3));
            break
        end
    end
end
