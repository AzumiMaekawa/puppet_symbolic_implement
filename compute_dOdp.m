function dO_dp = compute_dOdp(x, x_target, p, S, periodic)
global h n_p T w_traj w_acc 

% objective O = O_traj(x, x_target) + O_acc(pdot2) + O_CycPos  
%% compute the derivative with respect to x
DO_Dx = 2 * (x - x_target);
DO_Dx = DO_Dx * w_traj;

%% compute the derivative with respect to p
pdot = zeros([3*n_p*T 1]);
pdot(3*n_p+1:3*n_p*T) = (1/h) * (p(3*n_p+1:3*n_p*T) - p(1:3*n_p*(T-1)));
pdot2 = zeros([3*n_p*T 1]);
pdot2(3*n_p+1:3*n_p*T) = (1/h) * (pdot(3*n_p+1:3*n_p*T) - pdot(1:3*n_p*(T-1)));
pdot2_ip1 = [pdot2(3*n_p+1:end); zeros([3*n_p, 1])];
pdot2_ip2 = [pdot2_ip1(3*n_p+1:end); zeros([3*n_p, 1])];
% pdot2_im1 = [zeros([3*n_p 1]); pdot2(1:end-3*n_p)];
% pdot2_im2 = [zeros([3*n_p 1]); pdot2_im1(1:end-3*n_p)];
DO_Dp = (2./h^2) * (pdot2 - 2 * pdot2_ip1 + pdot2_ip2);
% DO_Dp = (2./h^2) * (pdot2 - 2 * pdot2_im1 + pdot2_im2);
DO_Dp = DO_Dp * w_acc;

% first order sensitivity term
% S = compute_dxdp(x, p, x_0, x_im1);

dO_dp = S.' * DO_Dx + DO_Dp;