%% Newton Method to solve the optimized value
function x_opt = Newtons_method(grad_i, hess_i, x_init, err, max_iteration)
global xi_sym
% for regularization
r = 1e-5;
I = eye(size(x_init,1));

x_k = x_init;

cnt = 0;
while(1) cnt = cnt + 1;
    % fprintf('Newtons method :: %d/%d\n', cnt, max_iteration)
    grad_k = subs(grad_i, xi_sym, x_k);
    hess_k = subs(hess_i, xi_sym, x_k);
    dx = double(-inv(hess_k + r*I) * grad_k);
    x_k = double(x_k + dx);
    calcNorm = double(dx.' * dx);
    if calcNorm < err
        x_opt = x_k;
        % fprintf('converged :: %e at x_k[%f, %f, %f] \n', ...
            % calcNorm, x_opt(1), x_opt(2), x_opt(3))
        break
    elseif cnt == max_iteration
        x_opt = x_k;
        % fprintf('reached to max iteration :: %e at x_k[%f, %f, %f] \n', ...
            % calcNorm, x_opt(1), x_opt(2), x_opt(3))
        break
    end
end
