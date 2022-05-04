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
%%
% Offset rotation from cam frame to world frame
offset_cam = [1, -0.002, 0.0071, -0.1999;
             0.002, 1, -0.0021, -0.083;
             -0.0071, 0.0021, 1, 0.0097]
w2c = [0,-1,0;0,0,-1;1,0,0]
offset_world = -1*inv(w2c)*offset_cam
quat = quaternion(offset_world(1:3,1:3),'rotmat','frame')
%%
% Offset addition in world frame
Pbf = [0.9998,0.0043,-0.02,10;
        -0.0045, 0.9999, -0.013, 0;
        0.02, 0.0131, 0.9997, 65;
       0,0,0,1]
Poff = [0.9998,-0.0046,0.0199,1.4802;
        0.0042,0.9999,0.0131, 1.0708;
        -0.0201,-0.0130,0.9997, -0.3791;
        0,0,0,1]
Poff2 = [1,0,0,1.4802;
        0,1,0, 1.0708;
        0,0,1, -0.3791;
        0,0,0,1]
inv(Poff(1:3,1:3))
new = Pbf*Poff
inv(Poff2)
new2 = Pbf*inv(Poff2)
new3 = inv(Poff2)*Pbf
%%
% Test
Poff = [[ 0.99991237 -0.00742189 -0.01096169 -0.04094639]
 [ 0.00740133  0.99997078 -0.00191521 -0.14138499]
 [ 0.01097559  0.00183391  0.99993808 -0.20035225]
 [ 0.          0.          0.          1.        ]]
Pbf = [[ -0.68199838   0.73135368  -0.         123.85842431]
 [ -0.73135368  -0.68199838   0.          -1.80250023]
 [  0.           0.           1.          42.24959117]
 [  0.           0.           0.           1.        ]]
pnew = Pbf*inv(Poff)
pnew2 = Pbf*Poff
quat = quaternion(pnew(1:3,1:3),'rotmat','frame')