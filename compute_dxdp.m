%% compute the sensitivity term S = dx/dp
function dx_dp = compute_dxdp(x, p, x_0, x_im1)
global n n_p DGi_Dxi DGi_Dxim1 DGi_Dxim2 DGi_Dpi xi_sym xim1_sym xim2_sym pi_sym

% compute partial derivative of G with respect to x
DG_Dx = cell(T); % T*T cell
for row = 1:T
    for col = 1:T
        if row == col
            % DGi_Dxi
            DG_row_Dx_col =subs(DGi_Dxi, xi_sym, x(3*n*(col-1)+1:3*n*col));
            if col <= 2 
                DG_row_Dx_col = subs(DG_row_Dx_col, xim1_sym, x_0);
            else
                DG_row_Dx_col = subs(DG_row_Dx_col, xim1_sym, x(3*n*(col-2)+1:3*n*(col-1)));
            end
            if col == 1
                DG_row_Dx_col = subs(DG_row_Dx_col, xim2_sym, x_im1);
            else
                DG_row_Dx_col = subs(DG_row_Dx_col, xim2_sym, x(3*n*(col-3)+1:3*n*(col-2)));
            end
        elseif row == col + 1
            % DGi_Dxim1
            DG_row_Dx_col = subs(DGi_Dxim1, xi_sym, x(3*n*col+1:3*n*(col+1)));
            DG_row_Dx_col = subs(DG_row_Dx_col, xim1_sym, x(3*n*(col-1)+1:3*n*col));
            if col == 1
               DG_row_Dx_col = subs(DG_row_Dx_col, xim2_sym, x_0);
            else 
                DG_row_Dx_col = subs(DG_row_Dx_col, xim2_sym, x(e*n*(col-2)+1:3*n*(col-1)));
            end
        elseif row == col + 2
            % DGi_Dxim2
            DG_row_Dx_col = subs(DGi_Dxim2, xi_sym, x(3*n*(col+1)+1:3*n*(col+2)));
            DG_row_Dx_col = subs(DG_row_Dx_col, xim1_sym, x(3*n*col+1:3*n*(col+1)));
            DG_row_Dx_col = subs(DG_row_Dx_col, xim2_sym, x(3*n*(col-1)+1:3*n*col));
        else
            % other submatrices are zeros because Gi depends xi, xim1, xim2
            DG_row_Dx_col = zeros(3*n);
        end

        DG_Dx{row, col} = DG_row_Dx_col;
    end
end

% compute partial derivative of G with respect to p
DG_Dp = cell(T); % T*T cell
for row = 1:T
    for col = 1:T
        if row == col
            DG_row_Dp_col = subs(DGi_Dpi, pi_sym, p(3*n_p*(col-1)+1:3*n_p*col));
        else
            % other submatrices are zeros because Gi depends on pi
            DG_row_Dp_col = zeros(3*n_p);
        end

        DG_Dp{row, col} = DG_row_Dp_col;
    end
end

% convert cell to matrix
DG_Dx = cell2mat(DG_Dx);
DG_Dp = cell2mat(DG_Dp);

% return sensitivity term
dx_dp = -inv(DG_Dx) * DG_Dp;