%% show results
% objective
%figure; 
%plot(1:max_main_iteration, cost_array);

% trajectories
x = reshape(x, [3 n*T]);
figure; scatter3(x(1,:),x(2,:),x(3,:)); 
hold on;
grid on;
rotate3d;
% draw initial posotion(red), final position(black)
scatter3(x(1,1), x(2,1), x(3,1), 15,'r','filled'); % position, size, color, filled marker 
scatter3(x(1,n*T), x(2,n*T), x(3,n*T),15,'k','filled');
% draw target trajectory
scatter3(target_x, target_y, target_z, 15, 'c', '+');
% draw control trajectory
p = reshape(p, [3 n_p*T]);
p_x = p(1,:);
p_y = p(2,:);
p_z = p(3,:);
plot3(p_x, p_y, p_z,'Color', 'g'); % plot control position with green line
for i = 1:T
    plot3([p_x(i) x(1,i)], [p_y(i) x(2,i)], [p_z(1) x(3,i)],'Color','k');
end
title('the simulated location');
xlabel('x'); ylabel('y'); zlabel('z');
xlim([-0.15 0.85]); ylim([-0.5 0.5]); zlim([-0.2 0.8]);