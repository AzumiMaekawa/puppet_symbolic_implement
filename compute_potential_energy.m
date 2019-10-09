function W = compute_potential_energy(x_i, p_i)
global n n_s k s_0 psy(s);

% W_puppet: the energy stored in the stiff springs (puppet linkages)
W_puppet = 0;
if n > 1
    % TODO: compute W_puppet
elseif n == 1
    W_puppet = 0;
else
    disp("ERROR: n is invalid value")
end

% W_string: the energy stored in the strings
% psy: model of a string
W_string = 0;
for i = 1:n_s
    W_string = W_string + k * psy(norm(x_i - p_i) - s_0(i));
end

W = W_puppet + W_string;