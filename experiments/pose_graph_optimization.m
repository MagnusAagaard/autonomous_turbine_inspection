% Pose graph optimization experiment
point_model = readmatrix('pose_graph_optimization/point_model.txt');
true_poses = readmatrix('pose_graph_optimization/true_poses.txt');
offset_poses = readmatrix('pose_graph_optimization/without_offset_poses.txt');
wps = readmatrix('pose_graph_optimization/inspection_waypoints.txt');

figure(1)
hold on;
plot3(point_model(1:3,1), point_model(1:3,2), point_model(1:3,3))
plot3(point_model(3:4,1), point_model(3:4,2), point_model(3:4,3))
plot3(point_model(5:6,1), point_model(5:6,2), point_model(5:6,3))
plot3(point_model(7:8,1), point_model(7:8,2), point_model(7:8,3))

plot3(true_poses(:,1),true_poses(:,2),true_poses(:,3),'g-*')
plot3(offset_poses(:,1),offset_poses(:,2),offset_poses(:,3),'r-*')
plot3(wps(:,1), wps(:,2), wps(:,3),'b*')

n = 7;
% True poses is our true pose (offset by estimating offset during
% inspection)
sum_error_true_poses = sum(abs(wps(1:n,:) - true_poses(1:n,:)))
% Offset poses are the non-offset poses during inspection. Error should be
% higher for this one
sum_error_offset_poses = sum(abs(wps(1:n,:) - offset_poses(1:n,:)))