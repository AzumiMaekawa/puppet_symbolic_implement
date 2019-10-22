function O = compute_objective(x, x_target, p, periodic)
global h n_p T

%% Compute O_traj
% O_traj measures the similarity between x and the user specified trajectory: x_target.
O_traj = norm(x - x_target)^2;

%% Compute O_acc
% O_acc regularizes the acceleration of the string attachment points to promote smooth motion.
% compute time-discretized acceleration of p
% pdot_0 = 0, pdot2_0 = 0
pdot = zeros([3*n_p*T 1]);
pdot(3*n_p+1:3*n_p*T) = (1/h) * (p(3*n_p+1:3*n_p*T) - p(1:3*n_p*(T-1)));
pdot2 = zeros([3*n_p*T 1]);
pdot2(3*n_p+1:3*n_p*T) = (1/h) * (pdot(3*n_p+1:3*n_p*T) - p(1:3*n_p*(T-1)));

O_acc = norm(pdot2)^2;

%% Compute O_CycPos
% for a periodic motion
O_CycPos = 0;
if periodic == true
    % TODO 
end

% return objective
O = O_traj + O_acc + O_CycPos;
