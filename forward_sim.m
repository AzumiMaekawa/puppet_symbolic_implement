%% forward simulation x(p)
function x = forward_sim(p, x_0, x_m1)
global n_p T pi_sym gradU hessU xim1_sym xim2_sym x_im1 x_im2;
x_i = x_0;
x_im1 = x_0;
x_im2 = x_m1;

x = [];
for i = 1:T
    p_i = p((i-1)*3*n_p+1:i*3*n_p);
    grad_i = subs(gradU, pi_sym, p_i);
    grad_i = subs(grad_i, xim1_sym, x_im1);
    grad_i = subs(grad_i, xim2_sym, x_im2);
    hess_i = subs(hessU, pi_sym, p_i);
    hess_i = subs(hess_i, xim1_sym, x_im1);
    hess_i = subs(hess_i, xim2_sym, x_im2);

    % compute x_i with Newton's method or gradient descent
    % fprintf('forward_sim :: %d/%d\n', i, T);
    % simplified one to see if it can be converged or not.
    x_i = Newtons_method(grad_i, hess_i, x_i, 1e-4, 20); %% -> succeeded to converge
    % x_i = gradient_descent(grad_i, x_i, 1e-5, 50);
    x = [x; x_i];

    x_im2 = x_im1;
    x_im1 = x_i;
end
