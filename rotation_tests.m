% Try to calculate relative pose
P1_cam = [1 0 0 0;
      0 0 -1 65;
      0 1 0 -10;
      0 0 0 1];

P2_cam = [0 1 0 -10;
      0 0 -1 65;
      -1 0 0 10;
      0 0 0 1];
% Get in in world frame
P1world = inv(P1_cam)
P2world = inv(P2_cam)

% Tij = Ti^-1*Tj --> Tij_cam = inv(Tij_world) --> Tij_cam = inv(inv(Ti^-1)*inv(Tj))
Tij_world = inv(P1world) * P2world
Tij_cam = inv(Tij_world)
Tij_cam1 = inv(inv(inv(P1_cam))*inv(P2_cam))
% inv(inv(T)) = T
Tij_cam2 = inv(P1_cam*inv(P2_cam))
% inv(T1*T2^-1) = T2*inv(T1)
Tij_cam3 = P2_cam*inv(P1_cam)
inv(Tij_cam)