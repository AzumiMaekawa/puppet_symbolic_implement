function [p_updated, x_updated] = line_search(p, dp, step_size, scale_factor, x, x_target, x_0, x_m1)
% p: current state
% dp: search direction
% step_size: initial step length
% scale_factor: scaling factor (0 < scale_factor < 1)
% reocompute x for each test candidate p_updated to evaluate O(x(p_updated), p)
O = compute_objective(x, x_target, p, false);
fprintf("cost: %f\n", O);
while 1
    p_updated = p + step_size * dp;
    x_updated = forward_sim(p_updated, x_0, x_m1);
    O_updated = compute_objective(x_updated, x_target, p_updated, false);
    fprintf("updated cost: %f  at step_size: %f\n", O_updated, step_size);
    if O_updated < O
        break
    end
    step_size = step_size * scale_factor;

    if step_size < 1e-10
        break
    end
end