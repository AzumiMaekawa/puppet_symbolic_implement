function DO2_Dp2 = compute_DO2Dp2()
global n_p h T w_acc
DO2_Dp2 = cell(T);
I = eye(3*n_p);
for row = 1:T
    for col = 1:T
        if (row == T-1) & (col == T-1)
            DO2_Dp2{row,col} = 5 * I;
        elseif (row == T) & (col == T)
            DO2_Dp2{row, col} = I;
        elseif ((row==T) & (col==T-1)) | ((row==T-1) & (col==T))
            DO2_Dp2{row, col} = -2*I;
        % elseif (row == 1) & (col == 1)
            % DO2_Dp2{row, col} = I;
        % elseif (row == 2) & (col == 2)
            % DO2_Dp2{row, col} = 5*I;
        % elseif ((row==1) & (col==2)) | ((row==2) & (col==1))
            % DO2_Dp2{row, col} = -2*I;
        elseif row == col
            DO2_Dp2{row, col} = 6*I;
        elseif abs(row - col) == 1
            DO2_Dp2{row, col} = -4*I;
        elseif abs(row - col) == 2
            DO2_Dp2{row, col} = I;
        else
            DO2_Dp2{row, col} = zeros(3*n_p);
        end
    end
end
DO2_Dp2 = cell2mat(DO2_Dp2);
DO2_Dp2 = DO2_Dp2 * (2./ (h^4)) * w_acc;

