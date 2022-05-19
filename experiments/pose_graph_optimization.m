% Pose graph optimization experiment
format compact;
point_model = readmatrix('pose_graph_optimization/point_model.txt');
wps = readmatrix('pose_graph_optimization/inspection_waypoints.txt');
true_poses11 = readmatrix('pose_graph_optimization/1m_error1/true_poses.txt');
offset_poses11 = readmatrix('pose_graph_optimization/1m_error1/without_offset_poses.txt');
true_poses12 = readmatrix('pose_graph_optimization/1m_error2/true_poses.txt');
offset_poses12 = readmatrix('pose_graph_optimization/1m_error2/without_offset_poses.txt');
true_poses13 = readmatrix('pose_graph_optimization/1m_error3/true_poses.txt');
offset_poses13 = readmatrix('pose_graph_optimization/1m_error3/without_offset_poses.txt');
true_poses14 = readmatrix('pose_graph_optimization/1m_error4/true_poses.txt');
offset_poses14 = readmatrix('pose_graph_optimization/1m_error4/without_offset_poses.txt');
true_poses15 = readmatrix('pose_graph_optimization/1m_error5/true_poses.txt');
offset_poses15 = readmatrix('pose_graph_optimization/1m_error5/without_offset_poses.txt');
true_poses21 = readmatrix('pose_graph_optimization/2m_error1/true_poses.txt');
offset_poses21 = readmatrix('pose_graph_optimization/2m_error1/without_offset_poses.txt');
true_poses22 = readmatrix('pose_graph_optimization/2m_error2/true_poses.txt');
offset_poses22 = readmatrix('pose_graph_optimization/2m_error2/without_offset_poses.txt');
true_poses23 = readmatrix('pose_graph_optimization/2m_error3/true_poses.txt');
offset_poses23 = readmatrix('pose_graph_optimization/2m_error3/without_offset_poses.txt');
true_poses24 = readmatrix('pose_graph_optimization/2m_error4/true_poses.txt');
offset_poses24 = readmatrix('pose_graph_optimization/2m_error4/without_offset_poses.txt');
true_poses25 = readmatrix('pose_graph_optimization/2m_error5/true_poses.txt');
offset_poses25 = readmatrix('pose_graph_optimization/2m_error5/without_offset_poses.txt');

n = 42;
% True poses is our true pose (offset by estimating offset during
% inspection)
error_true_poses11 = norm(wps(1:n,:) - true_poses11(1:n,:))
% Offset poses are the non-offset poses during inspection. Error should be
% higher for this one
error_offset_poses11 = norm(wps(1:n,:) - offset_poses11(1:n,:))
error_true_poses12 = norm(wps(1:n,:) - true_poses12(1:n,:))
error_offset_poses12 = norm(wps(1:n,:) - offset_poses12(1:n,:))
error_true_poses13 = norm(wps(1:n,:) - true_poses13(1:n,:))
error_offset_poses13 = norm(wps(1:n,:) - offset_poses13(1:n,:))
error_true_poses14 = norm(wps(1:n,:) - true_poses14(1:n,:))
error_offset_poses14 = norm(wps(1:n,:) - offset_poses14(1:n,:))
error_true_poses15 = norm(wps(1:n,:) - true_poses15(1:n,:))
error_offset_poses15 = norm(wps(1:n,:) - offset_poses15(1:n,:))
error_true_poses21 = norm(wps(1:n,:) - true_poses21(1:n,:))
error_offset_poses21 = norm(wps(1:n,:) - offset_poses21(1:n,:))
error_true_poses22 = norm(wps(1:n,:) - true_poses22(1:n,:))
error_offset_poses22 = norm(wps(1:n,:) - offset_poses22(1:n,:))
error_true_poses23 = norm(wps(1:n,:) - true_poses23(1:n,:))
error_offset_poses23 = norm(wps(1:n,:) - offset_poses23(1:n,:))
error_true_poses24 = norm(wps(1:n,:) - true_poses24(1:n,:))
error_offset_poses24 = norm(wps(1:n,:) - offset_poses24(1:n,:))
error_true_poses25 = norm(wps(1:n,:) - true_poses25(1:n,:))
error_offset_poses25 = norm(wps(1:n,:) - offset_poses25(1:n,:))
exp1_true_mu = mean([error_true_poses11, error_true_poses12, error_true_poses13, error_true_poses14, error_true_poses15])
exp1_off_mu = mean([error_offset_poses11, error_offset_poses12, error_offset_poses13, error_offset_poses14, error_offset_poses15])
exp2_true_mu = mean([error_true_poses21, error_true_poses22, error_true_poses23, error_true_poses24, error_true_poses25])
exp2_off_mu = mean([error_offset_poses21, error_offset_poses22, error_offset_poses23, error_offset_poses24, error_offset_poses25])

%%
% Plotting
figure(1)
hold on;
plot3(point_model(1:3,1), point_model(1:3,2), point_model(1:3,3))
plot3(point_model(3:4,1), point_model(3:4,2), point_model(3:4,3))
plot3(point_model(5:6,1), point_model(5:6,2), point_model(5:6,3))
plot3(point_model(7:8,1), point_model(7:8,2), point_model(7:8,3))

plot3(true_poses(:,1),true_poses(:,2),true_poses(:,3),'g-*')
plot3(offset_poses(:,1),offset_poses(:,2),offset_poses(:,3),'r-*')
plot3(wps(:,1), wps(:,2), wps(:,3),'b*')