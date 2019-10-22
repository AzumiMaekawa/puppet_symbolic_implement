function W = compute_potential_energy(xi, pi)
global n n_s k s_0 psy epsilon debug;

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
% psy: model of a string, piece-wise function
syms psy(s);
psy(s) = piecewise(0<s, (1/2)*s^2 + (epsilon/2)*s + epsilon^2 / 6,...
            -epsilon<s<0, (1/(6*epsilon))*s^3 + (1/2)*s^2 + (epsilon/2)*s + epsilon^2/6, ...
            0);

W_string = 0;
for i = 1:n_s
    % sym tmp_f; % being local to be readable
    tmp_f = norm(xi - pi) - s_0(i);
    W_string = W_string + k * psy(tmp_f);
end

W = W_puppet + W_string;