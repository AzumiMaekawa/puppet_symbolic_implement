function x_opt = gradient_descent(grad_i, x_init, err, max_iteration)
global xi_sym

x_k = x_init;
r = 5e-4; % learning rate

cnt = 0;
while(1)
    cnt = cnt + 1;
    fprintf('Gradient descent :: %d/%d\n', cnt, max_iteration);
    grad_k = subs(grad_i, xi_sym, x_k);
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
