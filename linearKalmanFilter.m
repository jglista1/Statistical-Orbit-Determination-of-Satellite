clear all; close all; opengl software
load('orbitdeterm_finalproj_KFdata.mat');
r0 = 6678; %km
mu = 398600; %km^3/sec^2
dT = 10; %sec
om0 = sqrt(mu/r0^3);
tspan = 0:dT:1400*dT;
[ydata] = MeasFormat(tspan, ydata);
 
x0 = [r0; 0; 0; r0*sqrt(mu/r0^3)];
dx_init = [0; 0.075; 0; -0.021];
x_init = x0;
x_init_Pert = x0 + dx_init;
P0 = 0.2*diag([r0, r0*sqrt(mu/r0^3), r0, r0*sqrt(mu/r0^3)]);
 
rng(100)
Gamma = [0 0; 1 0; 0 0; 0 1];
%% Step 1: Determine a priori nominal state
xnom = nominalTrajectory(mu, tspan, x_init);
%% Step 2: Calculate Offline Jacobians (linearized about nominal state) for each time step and each tracking station
[F] = Fjacobian(mu, tspan, dT, xnom);
[H] = Hjacobian(tspan, dT, xnom);
 
%% Step 3: Execute LKF with Sample Noisy Data
[ynom] = StateToMeas(tspan, dT, xnom); Q = eye(2)*10^-10;
 
    w = randn(length(tspan),2)*chol(Qtrue);w = w';
    xSim = nominalTrajectory(mu, tspan, x_init_Pert) + Gamma*w;
    v = mvnrnd([0 0 0], Rtrue, length(tspan))';
    ySim = StateToMeas(tspan, dT, xSim);
    ySim = ySim(1:3,:,:) + v;
[dx_array, dy_array, P_array, S_array] = LKF(tspan, dT, ynom, ydata, F, H, Rtrue, Q, P0 );
xLKF = dx_array + xnom;
yLKF = StateToMeas(tspan, dT, xLKF);
plot2States(tspan, xnom, xSim, 'Typical Noisy Truth Trajectory vs Nominal', 'Nominal', 'Noisy')
plot2Meas(tspan, ynom, ySim, 'Typical Noisy Measurements vs Nominal', 'Nominal', 'Noisy')
 
 
plotMeas(tspan, yLKF, 'yLKF')
plotStates(tspan, xLKF, 'xLKF')
plotLKFError(tspan, xSim - xLKF, P_array, 'State Errors and 2\sigma Bounds for LKF')
plotLKFError(tspan, xLKF, P_array, 'Trajectory Estimation for Observed Data Log for One Orbital Period (LKF)')
 
%% Step 4: NEES and NIS Testing
% For each iteration, simulate ground truth trajectories with process noise
% along nominal trajectory (and slight initial perturbation). Then
% calculate the associated measurements and
% add measurement noise. These will be our "true" state and input data
% respectively. Run the LKF and get the state estimation. Then calculate NEES and NIS statistics. 
Q = eye(2)*10^-15; R = Rtrue;
 
N = 10; alpha = 0.05;
NEES = zeros(N,length(tspan));NIS = zeros(N,length(tspan),12);
for m = 1:N
    w = randn(length(tspan),2)*chol(Qtrue);w = w';
    xSim = nominalTrajectory(mu, tspan, x_init_Pert) + Gamma*w;
    v = mvnrnd([0 0 0], R, length(tspan))';
    ySim = StateToMeas(tspan, dT, xSim);
    ySim = ySim(1:3,:,:) + v;
    
    [dx_array, dy_array, P_array, S_array] = LKF(tspan, dT, ynom, ySim, F, H, Rtrue, Q, P0 );
    xLKF = dx_array + xnom;
    yLKF = StateToMeas(tspan, dT, xLKF);
    
    for k = 1:length(tspan)
        exk = xSim(:,k) - xLKF(:,k);
        NEES(m,k) = exk'*P_array(:,:,k)*exk;
        for i = 1:12
        eyk = ySim(1:3,k,i) - yLKF(1:3,k,i);
        NIS(m,k,i) = eyk'*S_array(:,:,k,i)*eyk;
        end
    end
 
end
plotTMT(tspan, NEES, NIS, N, alpha, 'NEES and NIS Statistics for LKF, N = 10, \alpha = 0.05')
 
function plotTMT(tspan, NEES, NIS, N, alpha, title)
r1 = chi2inv(alpha/2, N*4)./N;
    r2 = chi2inv(1-alpha/2, N*4)./N;
figure;
sgtitle(title)
 
subplot(2,1,1)
    hold on
    for m = 1:N
        if m == 1
        plot(tspan, NEES(m,:),'.r')
        else
            plot(tspan, NEES(m,:),'.r','HandleVisibility','off')
        end
    end
    plot(tspan, r1*ones(length(tspan)),'--k')
    plot(tspan, r2*ones(length(tspan)),'--k')
    hold off
            xlabel('Time [s]'); ylabel('NEES Statistic')
    legend('NEES', 'r_1 Bound','r_2 Bound','Location','NorthEast')
    ylim([-5 200]); 
% xlim([0 5500])
    
subplot(2,1,2)
    hold on;
        for m = 1:N
            for i = 1:12
                if i == 1 && m==1
                    plot(tspan, NIS(m,:,i), '.b')
                else
                    plot(tspan, NIS(m,:,i), '.b','HandleVisibility','off')
                end
            end
        end
        plot(tspan, r1*ones(length(tspan)),'--k')
        plot(tspan, r2*ones(length(tspan)),'--k')
    hold off;
    xlabel('Time [s]'); ylabel('NIS Statistic')
        legend('NIS', 'r_1 Bound','r_2 Bound','Location','best')
    ylim([-5 2000]); 
%   xlim([0 5500])
end
function plotLKFError(tspan, x, P_array, title)
    S1 =[];S2=[];S3=[];S4=[];
    for k = 1:length(tspan)
       S1 = [S1 P_array(1,1,k)]; S2 = [S2 P_array(2,2,k)]; 
       S3 = [S3 P_array(3,3,k)]; S4 = [S4 P_array(4,4,k)];
    end
 
figure;
subplot(4, 1, 1);
hold on
plot(tspan, x(1,:),'.k');
plot(tspan, x(1,:) + 2*S1,'--g')
plot(tspan, x(1,:) - 2*S1,'--g')
hold off
sgtitle(title);
legend('x_1 State Error','2\sigma Error Bounds')
% legend('x_1','2\sigma Error Bounds')
ylabel('x_1 = X (km)');
% ylim([-700 600]); 
xlim([0 5500])
grid on;
subplot(4, 1, 2);
hold on
plot(tspan, x(2,:),'.k');
plot(tspan, x(2,:) + 2*S2,'--g')
plot(tspan, x(2,:) - 2*S2,'--g')
hold off
% legend('x_2 State Error','2\sigma Error Bounds')
legend('x_2','2\sigma Error Bounds')
ylabel('x_2 = Xdot (km/s)');
% ylim([-2 2]);
xlim([0 5500])
grid on;
subplot(4, 1, 3);
hold on
plot(tspan, x(3,:),'.k');
plot(tspan, x(3,:) + 2*S3,'--g')
plot(tspan, x(3,:) - 2*S3,'--g')
hold off
% legend('x_3 State Error','2\sigma Error Bounds')
legend('x_3','2\sigma Error Bounds')
ylabel('x_3 = Y (km)');
% ylim([-700 600]);
xlim([0 5500])
grid on;
subplot(4, 1, 4);
hold on
plot(tspan, x(4,:),'.k');
plot(tspan, x(4,:) + 2*S4,'--g')
plot(tspan, x(4,:) - 2*S4,'--g')
hold off
legend('x_4 State Error','2\sigma Error Bounds')
% legend('x_4','2\sigma Error Bounds')
ylabel('x_4 = Ydot (km/s)');
% ylim([-2 2]);
xlim([0 5500])
xlabel('Time [s]');
grid on;
end
function plot2Meas(tspan, y1, y2, title, legend1, legend2)
    figure;
    sgtitle(title)
 
    subplot(3,1,1)
    hold on
    for i = 1:12
        plot(tspan,y1(1,:,i),'r')
        plot(tspan,y2(1,:,i),'b')
    end
    hold off
    legend(legend1, legend2)
    xlabel('Time [s]'); ylabel('rho [km]')
    ylim([500 2000]);
 
    subplot(3,1,2)
    hold on
    for i = 1:12
        plot(tspan,y1(2,:,i),'r')
        plot(tspan,y2(2,:,i),'b')
    end
    hold off
    legend(legend1, legend2)
    xlabel('Time [s]'); ylabel('rhodot [km/s]')
    ylim([-10 10]);
 
    subplot(3,1,3)
    hold on
    for i = 1:12
        plot(tspan,y1(3,:,i),'r')
        plot(tspan,y2(3,:,i),'b')
    end
    hold off
    legend(legend1, legend2)
    xlabel('Time [s]'); ylabel('phi [rads]');
    ylim([-5 5]);
 
end
function plotMeas(tspan, y, title)
    figure;
    sgtitle(title)
 
    subplot(4,1,1)
    hold on
    for i = 1:12
        plot(tspan,y(1,:,i),'x')
    end
    hold off
    xlabel('Time [s]'); ylabel('rho [km]')
    ylim([500 2000]);
 
    subplot(4,1,2)
    hold on
    for i = 1:12
        plot(tspan,y(2,:,i),'o')
    end
    hold off
    xlabel('Time [s]'); ylabel('rhodot [km/s]')
    ylim([-10 10]);
 
    subplot(4,1,3)
    hold on
    for i = 1:12
        plot(tspan,y(3,:,i),'s')
    end
    hold off
    xlabel('Time [s]'); ylabel('phi [rads]');
    ylim([-5 5]);
 
    subplot(4,1,4)
    hold on
    for i = 1:12
        plot(tspan,y(4,:,i),'d')
    end
    hold off
    xlabel('Time [s]'); ylabel('Tracking ID')
    ylim([0 15]);
end
function plot2States(tspan, x1, x2, title, legend1, legend2)
%% Plot LFK Total State Estimation
figure;
subplot(4, 1, 1);
hold on
plot(tspan, x1(1,:));
plot(tspan, x2(1,:));
hold off
legend(legend1, legend2)
sgtitle(title);
ylabel('x_1 = X (km)');
grid on;
subplot(4, 1, 2);
hold on
plot(tspan, x1(2,:));
plot(tspan, x2(2,:));
hold off
legend(legend1, legend2)
ylabel('x_2 = Xdot (km/s)');
grid on;
subplot(4, 1, 3);
hold on
plot(tspan, x1(3,:));
plot(tspan, x2(3,:));
hold off
legend(legend1, legend2)
ylabel('x_3 = Y (km)');
grid on;
subplot(4, 1, 4);
hold on
plot(tspan, x1(4,:));
plot(tspan, x2(4,:));
hold off
legend(legend1, legend2)
ylabel('x_4 = Ydot (km/s)');
xlabel('Time [s]');
grid on;
end
function plotStates(tspan, x, title)
figure;
subplot(4, 1, 1);
plot(tspan, x(1,:));
sgtitle(title);
ylabel('x_1 = X (km)');
grid on;
subplot(4, 1, 2);
plot(tspan, x(2,:));
ylabel('x_2 = Xdot (km/s)');
grid on;
subplot(4, 1, 3);
plot(tspan, x(3,:));
ylabel('x_3 = Y (km)');
grid on;
subplot(4, 1, 4);
plot(tspan, x(4,:));
ylabel('x_4 = Ydot (km/s)');
xlabel('Time [s]');
grid on;
end
function [dx_array, dy_array, P_array, S_array] = LKF(tspan, dT, ynom, ydata, F, H, Rf, Qf, P0 )
    %% Linearized Kalman Filter
    % Add in extra DT matrices
    G = [0 0; dT 0; 0 0; 0 dT];
    O = [0 0; 1 0; 0 0; 0 1];
    
    % Encode KF initial mean and covariance
    dx = zeros(4,1); % initial guess is that trajectory is perfectly nominal
    P = P0;
    T = length(tspan);
    dx_array = zeros(4,T);
    dy_array = zeros(4,T,12);
    P_array = zeros(4,4,T); P_array(:,:,1) = P0;
    S_array = zeros(3,3,T,12);
 
    % Linearized Kalman Filter
    for k = 2:T
 
        % Prediction step should be done once per timestep
        dx_minus = F(:,:,k)*dx;
        P_minus = F(:,:,k)*P*F(:,:,k)' + O*Qf*O';
 
        for i = 1:12
            if isnan(ynom(1,k,i))
                continue
            end
            if isnan(ydata(1,k,i))
                continue
            end
            y = ydata(:,k,i);
            
            
            % Gain
            K = P_minus*H(:,:,k,i)'*inv(H(:,:,k,i)*P_minus*H(:,:,k,i)' + Rf);
           % Update measurements once per in-range station
            S = H(:,:,k,i)*P*H(:,:,k,i)'+Rf; % Innovation
            dy = y(1:3)-ynom(1:3,k,i); % Actual sensor measurement minus simulated nominal measurement
            dx_plus = dx_minus + K*(dy - H(:,:,k,i)*dx_minus); % Update mean with gain times simulated (real) - predicted measurements
            P_plus = (eye(4) - K*H(:,:,k,i))*P_minus; % Update covariance
 
 
            dx = dx_plus; 
            P = P_plus;
            dy_array(1:3,k,i) = dy;
            dy_array(4,k,i) = i;
            S_array(:,:,k,i) = S; 
        end
        % Store state and covariance in array 
        dx_array(:,k) = dx_plus;
        P_array(:,:,k) = P_plus;
    end
end
function [ydata] = MeasFormat(tspan, ydata)
    y = NaN*ones(4,length(tspan),12);
    for k = 2:length(tspan)
        yk = cell2mat(ydata(k));
        if size(yk) == [0 0]
            continue
        end
        
        for j = 1:size(yk,2)
            y(:,k, yk(4,j)) = yk(:,j);
        end
    end
    ydata = y;
end
function [yMeas] = StateToMeas(tspan, dT, x)
    % Calculate associated measurements
    yMeas = NaN*ones(4,length(tspan),12); 
    Xs = NaN*ones(length(tspan),4,12);
    x2 = x';
    
    RE = 6378; %km
    wE = 2*pi/86400; %rad/s
    for k = 1:length(tspan)
        for i = 1:12
            ti = ((k-1)*dT);
            theta0 = (i-1)*(pi/6);
            Xi = RE*cos(wE*ti + theta0);
            Yi = RE*sin(wE*ti + theta0);
            Xidot = -RE*wE*sin(wE*ti + theta0);
            Yidot = RE*wE*cos(wE*ti + theta0);
 
            thetai = atan2(Yi,Xi);
            phii = atan2((x2(k,3)-Yi),(x2(k,1)-Xi));
 
            if abs(angdiff(phii,thetai)) < pi()/2
                
                yMeas(1,k,i) = sqrt((x2(k,1)-Xi)^2 + (x2(k,3)-Yi)^2);
                yMeas(2,k,i) = ((x2(k,1)-Xi)*(x2(k,2)-Xidot)+(x2(k,3)-Yi)*(x2(k,4)-Yidot))/yMeas(1,k,i);
                yMeas(3,k,i) = phii;
                yMeas(4,k,i) = i;
                Xs(k,1,i) = Xi;
                Xs(k,1,i) = Yi;
                Xs(k,1,i) = Xidot;
                Xs(k,1,i) = Yidot;
            end
        end
    end
 
end
function [H] = Hjacobian(tspan, dT, xnom)
 
rho = NaN*ones(length(tspan),1,12);
rhodot = NaN*ones(length(tspan),1,12);
phi = NaN*ones(length(tspan),1,12);
idCount = NaN*ones(length(tspan),1,12);
 
Xi_s = NaN*ones(length(tspan),1,12);
Yi_s = NaN*ones(length(tspan),1,12);
Xidot_s = NaN*ones(length(tspan),1,12);
Yidot_s = NaN*ones(length(tspan),1,12);
 
RE = 6378; %km
wE = 2*pi/86400; %rad/s
 
xnom = xnom';
for k = 1:1401
    for i = 1:12
        ti = ((k-1)*dT);
        theta0 = (i-1)*(pi/6);
        Xi = RE*cos(wE*ti + theta0);
        Yi = RE*sin(wE*ti + theta0);
        Xidot = -RE*wE*sin(wE*ti + theta0);
        Yidot = RE*wE*cos(wE*ti + theta0);
        
        thetai = atan2(Yi,Xi);
        phii = atan2((xnom(k,3)-Yi),(xnom(k,1)-Xi));
        
        if abs(angdiff(phii,thetai)) < pi()/2
            rho(k,1,i) = sqrt((xnom(k,1)-Xi)^2 + (xnom(k,3)-Yi)^2);
            rhodot(k,1,i) = ((xnom(k,1)-Xi)*(xnom(k,2)-Xidot)+(xnom(k,3)-Yi)*(xnom(k,4)-Yidot))/rho(k,1,i);
            phi(k,1,i) = phii;
            idCount(k,1,i) = i;
            Xi_s(k,1,i) = Xi;
            Yi_s(k,1,i) = Yi;
            Xidot_s(k,1,i) = Xidot;
            Yidot_s(k,1,i) = Yidot;
          
            
            
            H11(k,1,i) = (xnom(k,1)-Xi_s(k,1,i))/(((xnom(k,1)-Xi_s(k,1,i))^2 + (xnom(k,3)-Yi_s(k,1,i))^2))^(1/2);
            H12 = zeros(length(tspan),1,12);
            H13(k,1,i) = (xnom(k,3)-Yi_s(k,1,i))/(((xnom(k,1)-Xi_s(k,1,i))^2 + (xnom(k,3)-Yi_s(k,1,i))^2))^(1/2);
            H14 = zeros(length(tspan),1,12);
            H21(k,1,i) = ((xnom(k,3)-Yi_s(k,1,i))*((xnom(k,2)-Xidot_s(k,1,i))*(xnom(k,3)-Yi_s(k,1,i))-(xnom(k,1)-Xi_s(k,1,i))*(xnom(k,4)-Yidot_s(k,1,i))))/(((xnom(k,1)-Xi_s(k,1,i))^2 + (xnom(k,3)-Yi_s(k,1,i))^2))^(3/2);
            H22(k,1,i) = (xnom(k,1)-Xi_s(k,1,i))/(((xnom(k,1)-Xi_s(k,1,i))^2 + (xnom(k,3)-Yi_s(k,1,i))^2))^(1/2);
            H23(k,1,i) = ((xnom(k,1)-Xi_s(k,1,i))*((xnom(k,1)-Xi_s(k,1,i))*(xnom(k,4)-Yidot_s(k,1,i))-(xnom(k,2)-Xidot_s(k,1,i))*(xnom(k,3)-Yi_s(k,1,i))))/(((xnom(k,1)-Xi_s(k,1,i))^2 + (xnom(k,3)-Yi_s(k,1,i))^2))^(3/2);
            H24(k,1,i) = (xnom(k,3)-Yi_s(k,1,i))/(((xnom(k,1)-Xi_s(k,1,i))^2 + (xnom(k,3)-Yi_s(k,1,i))^2))^(1/2);
            H31(k,1,i) = (Yi_s(k,1,i)-xnom(k,3))/(((xnom(k,1)-Xi_s(k,1,i))^2 + (xnom(k,3)-Yi_s(k,1,i))^2));
            H32 = zeros(length(tspan),1,12);
            H33(k,1,i) = (xnom(k,1)-Xi_s(k,1,i))/(((xnom(k,1)-Xi_s(k,1,i))^2 + (xnom(k,3)-Yi_s(k,1,i))^2));
            H34 = zeros(length(tspan),1,12);
            
            % Build the H Matrix
            H(:,:,k,i) = [H11(k,1,i) H12(k,1,i) H13(k,1,i) H14(k,1,i);
                          H21(k,1,i) H22(k,1,i) H23(k,1,i) H24(k,1,i);
                          H31(k,1,i) H32(k,1,i) H33(k,1,i) H34(k,1,i)];
            
        end
    end
end
 
end
function [F] = Fjacobian(mu, tspan, dT, xnom)
 
% Calculating Perturbation Dynamics
A = zeros(4,4,length(tspan));
F = zeros(4,4,length(tspan));
 
    for k = 1:length(tspan)
        A21(:,k) = (mu*(2*xnom(1,k)^2-xnom(3,k)^2))/(xnom(1,k)^2 + xnom(3,k)^2)^(5/2);
        A23(:,k) = (3*mu*xnom(1,k)*xnom(3,k))/(xnom(1,k)^2 + xnom(3,k)^2)^(5/2);
        A41(:,k) = (3*mu*xnom(1,k)*xnom(3,k))/(xnom(1,k)^2 + xnom(3,k)^2)^(5/2);
        A43(:,k) = (mu*(2*xnom(3,k)^2-xnom(1,k)^2))/(xnom(1,k)^2 + xnom(3,k)^2)^(5/2);
        A(:,:,k) = [0 1 0 0; A21(:,k) 0 A23(:,k) 0; 0 0 0 1; A41(:,k) 0 A43(:,k) 0];
 
        F(:,:,k) = eye(4) + dT*A(:,:,k);
    end
end
function [xNom] = nominalTrajectory(mu, tspan, x_init)
xdot = @(x,u) [x(2);
                       -(mu/sqrt(x(1)^2+x(3)^2)^3)*x(1); % + u(1) + w(1)
                       x(4);
                       -(mu/sqrt(x(1)^2+x(3)^2)^3)*x(3)]; % + u(2) + w(2)
                   
xdotWrap = @(t,x) xdot(x, [0;0]);
 
options = odeset('RelTol', 1e-12);
[~, xNom] = ode45(xdotWrap, tspan, x_init, options);
 
xNom = xNom';
 
end
