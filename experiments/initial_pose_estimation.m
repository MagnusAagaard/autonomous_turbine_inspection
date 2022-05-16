% Inital pose estimation test
%%
% Create test params
%clear; clc;
%test_params = zeros(100,5);
%writematrix(test_params,'recorded_values.txt')
%test_params(:,1) = 110;
%test_params(:,2) = 0.0;
%test_params(:,3) = 66;
%test_params(:,4) = deg2rad(randi([0 119],100,1));
%test_params(:,5) = deg2rad(randi([0 359],100,1));
%writematrix(test_params,'test_params.txt')

%%
% Load test params
clear;clc;
n1 = 1;
n = 100;
bad_idx = [3 6 29 48 50 53 55 59 88 90 93];
test_params = readmatrix("test_params.txt");
recorded_values = readmatrix("recorded_values.txt");
x = abs(test_params(n1:n,1) - recorded_values(n1:n,1));
y = abs(test_params(n1:n,2) - recorded_values(n1:n,2));
z = abs(test_params(n1:n,3) - recorded_values(n1:n,3));
phi = abs(mod(abs(mod(rad2deg(test_params(n1:n,4)),120) - mod(rad2deg(recorded_values(n1:n,4)),120)) + 3*60, 2*60) - 60)
omega = abs(mod(abs(mod(rad2deg(test_params(n1:n,5)),360) - mod(rad2deg(recorded_values(n1:n,5)),360)) + 3*180, 2*180) - 180)
for i=1:length(bad_idx)
    x(bad_idx(i)) = 0.0;
    y(bad_idx(i)) = 0.0;
    z(bad_idx(i)) = 0.0;
    phi(bad_idx(i)) = 0.0;
    omega(bad_idx(i)) = 0.0;
end
mu_x = mean(x)
std_x = std(x)
mu_y = mean(y)
std_y = std(y)
mu_z = mean(z)
std_z = std(z)
mu_phi = mean(phi)
std_phi = std(phi)
mu_omega = mean(omega)
std_omega = std(omega)
%figure(1)
%scatter(phi,omega)
figure(2)
hold on;
boxplot([x,y,z,phi,omega],'Labels',{'x','y','z',char(966),char(969)});
ylabel('Error')
% Save figure
clear figure_property;
figure_property.units = 'inches';
figure_property.format = 'pdf';
figure_property.Preview= 'none';
figure_property.Width= '11'; % Figure width on canvas
figure_property.Height= '8'; % Figure height on canvas
figure_property.Units= 'inches';
figure_property.Color= 'rgb';
figure_property.Background= 'w';
figure_property.FixedfontSize= '12';
figure_property.ScaledfontSize= 'auto';
figure_property.FontMode= 'scaled';
figure_property.FontSizeMin= '12';
figure_property.FixedLineWidth= '1';
figure_property.ScaledLineWidth= 'auto';
figure_property.LineMode= 'none';
figure_property.LineWidthMin= '0.1';
figure_property.FontName= 'Times New Roman';% Might want to change this to something that is available
figure_property.FontWeight= 'auto';
figure_property.FontAngle= 'auto';
figure_property.FontEncoding= 'latin1';
figure_property.PSLevel= '3';
figure_property.Renderer= 'painters';
figure_property.Resolution= '600';
figure_property.LineStyleMap= 'none';
figure_property.ApplyStyle= '0';
figure_property.Bounds= 'tight';
figure_property.LockAxes= 'off';
figure_property.LockAxesTicks= 'off';
figure_property.ShowUI= 'off';
figure_property.SeparateText= 'off';
chosen_figure=gcf;
set(chosen_figure,'PaperUnits','inches');
set(chosen_figure,'PaperPositionMode','auto');
set(chosen_figure,'PaperSize',[str2num(figure_property.Width) str2num(figure_property.Height)]); % Canvas Size
set(chosen_figure,'Units','inches');
hgexport(gcf,'initial_pose_estimation_boxplot.pdf',figure_property); %Set desired file name