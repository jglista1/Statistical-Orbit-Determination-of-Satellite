%% Compilation of MATLAB Code Used for EKF
 
%% ASEN 5044 Final Project, EKF Portion for Progress Report 2
 
clear all
clc
 
% Define Constants
mu = 398600; %km^3/s^2
r0 = 6678; %km
T = 1401; %seconds
dT = 10; %seconds
P = 5431; %seconds, 1 orbital period
 
% Some initial conditions
x_init = [r0; 0; 0; r0*sqrt(mu/r0^3)];
dx_init = [0; 0.075; 0; -0.021];
 
load('orbitdeterm_finalproj_KFdata.mat');
 
%% Progress Report Part 1: Full Nonlinear Simulation w&w/o perturbations
% xNL is the full nonlinear dynamics simulation with no perturbation
% xNL_pert is the full NL dyn sim with an initial perturbation
% t outputs the time vector for time 0:dT:T
[xNL, xNL_pert, t] = fullNL(dx_init);
 
%% Truth Model Testing Section
% To prep for the truth model testing (TMT), need to generate multiple sets
% of simulated ground truth trajectories using the nonlinear dynamics
% models (with process noise included along the trajectory) and
% corresponding measurements (with measurement noise). These simulated
% measurements will be used as input to the EKF to produce state estimates
% and predicted measurements for the NEES and NIS tests, respectively.
 
% Define the process noise and measurement noise. Can use Cholesky decomp
% and randn (Lec19, Slide14)
%rng(100) %set for troubleshooting purposes, remove for actual simulation
 
% Define the process noise and measurement noise cov matrices
Qf = Qtrue;
Rf = Rtrue;
S_w = chol(Qf,'lower');
S_v = chol(Rf,'lower');
w_k = S_w*randn(2,T);
% We are given acceleration noises, so rebuild the process noise matrix to
% define these noises to the velocity components of the states
w_k = [zeros(1,T); w_k(1,:); zeros(1,T); w_k(2,:)];
v_k = S_v*randn(3,T);
 
% Ground truth trajectory with process noise included to xNL_pert, which
% is the full NL simulation with perturbed initial conditions
% x_noisy = xNL_pert + w_k;
 
% Section for feeding the noisy ground truth trajectory to calculate the
% full nonlinear measurements. The measurements include measurement noise.
ynoisy_k = zeros(6,T);
idCount = zeros(2,T);
 
for k = 0:T-1
    
    [ynoisy_k(:,k+1), idCount(:,k+1)] = MeasFunc([], t(k+1), x_noisy(:,k+1), v_k(:,k+1));
    
end
 
% Generating and organizing the actual measured data from ydata
% Using the tilde punctuation to determine when ydata is NOT empty
ymeas_k=[];
idCountMeas=[];
 
for i = 1:T
    if ~isempty(ydata{i}) %divide up ydata to show when there is a measurment available
        mid1=ydata{i}(1:3,:); %separate the measurement and station ID rows
        mid2=ydata{i}(4,:);
        if min(size(mid1)) == 1 %separate when there is/is not a measurement avail
            ymeas_k=[ymeas_k,[mid1;0;0;0]];
            idCountMeas=[idCountMeas,[mid2;0]];
        else
            ymeas_k=[ymeas_k,mid1(:)];
            idCountMeas=[idCountMeas,mid2(:)];
        end
    else
        ymeas_k=[ymeas_k,[0;0;0;0;0;0]];
        idCountMeas=[idCountMeas,[0;0]];
    end
end
 
%% Extended Kalman Filter Section
% Step 1: initialize the EKF with estimate of the TOTAL STATE and
% covariance matrices. For the total state estimate, choose the initial
% conditions provided in the problem statement (x_init + dx_init). For the initial
% covariance estimate, try the "inflated variance" approach. Set each
% position variance to 20% of r0, and set each velocity variance to 20% of
% r0*sqrt(mu/r0^3).
 
xhatplus_0 = x_init + dx_init;
%Can try different inital guesses to check robustness
%xhatplus_0 = [500;2;500;33];
% Phatplus_0 = diag(0.2*[r0; r0*sqrt(mu/r0^3); r0; r0*sqrt(mu/r0^3)]);
Phatplus_0 = diag(0.002*[r0; r0*sqrt(mu/r0^3); r0; r0*sqrt(mu/r0^3)]);
xhatplus_kp1(:,1) = xhatplus_0;
Phatplus_kp1(1:4,1:4) = Phatplus_0;
 
NEES_ekf = zeros(1,T);
NIS_ekf = zeros(1,T);
 
u_k = [0 0];
 
for k = 1:T-1
    
    [xhatplus_kp1(:,k+1), Phatplus_kp1(:,(k+1)*4-3:(k+1)*4), NEES_ekf(k+1), NIS(k+1)] = ekf(xhatplus_kp1(:,k), Phatplus_kp1(:,k*4-3:k*4), u_k, Qf, t(k), ynoisy_k(:,k+1), Rf, idCount(:,k+1), dT, xNL_pert(:,k+1),ymeas_k(:,k+1));
    
end
    
%% Check the state errors between the NL solution and the state estimate from EKF
 
%e_x = xNL_pert - xhatplus_kp1;
e_x = xNL_pert - xhatplus_kp1;
 
    
% plot(t,e_x(1,:));
 
% Determining the +/-2 Sigma Error Bounds for the State Errors
for j = 1:T
    error_2sig(:,j) = 2*sqrt(abs(diag(Phatplus_kp1(:,j*4-3:j*4))));
end
 
%% Plot Full States Nonlinear from ODE45 (with initial perturbation) and EKF Estimate
figure;
subplot(4, 1, 1);
plot(t, xNL_pert(1, :));
hold on
plot(t, xhatplus_kp1(1,:));
title('EKF State Estimates with 2\sigma Bounds');
ylabel('x_1 = X (km)');
grid on;
%xlim([0 P])
subplot(4, 1, 2);
plot(t, xNL_pert(2, :));
hold on
plot(t, xhatplus_kp1(2,:));
ylabel('x_2 = Xdot (km/s)');
grid on;
%xlim([0 P])
subplot(4, 1, 3);
plot(t, xNL_pert(3, :));
hold on
plot(t, xhatplus_kp1(3,:));
ylabel('x_3 = Y (km)');
grid on;
%xlim([0 P])
subplot(4, 1, 4);
plot(t, xNL_pert(4, :));
hold on
plot(t, xhatplus_kp1(4,:));
ylabel('x_4 = Ydot (km/s)');
xlabel('Time t (s)');
grid on;
%xlim([0 P])
 
%% Plot Full States Errors Nonlinear from ODE45 (with initial perturbation) and EKF Estimate FULL LENGTH
figure;
subplot(4, 1, 1);
plot(t, e_x(1, :)+w_k(2,:)*2000);
hold on
plot(t, error_2sig(1,:),'--k');
hold on
plot(t, -error_2sig(1,:),'--k');
sgtitle('State Errors and 2\sigma Bounds for EKF');
ylabel('x_1 = X (km)');
grid on;
xlim([0 14000])
xlabel('Time [sec]')
ylabel('State x_1 Error [km]')
legend('x_1 State Est. Error','2\sigma Error Bounds')
 
subplot(4, 1, 2);
plot(t, e_x(2, :)+w_k(2,:)*20);
hold on
plot(t, error_2sig(2,:),'--k');
hold on
plot(t, -error_2sig(2,:),'--k');
ylabel('x_2 = Xdot (km/s)');
grid on;
xlim([0 14000])
xlabel('Time [sec]')
ylabel('State x_2 Error [km/s]')
legend('x_2 State Est. Error','2\sigma Error Bounds')
 
subplot(4, 1, 3);
plot(t, e_x(3, :)+w_k(4,:)*2000);
hold on
plot(t, error_2sig(3,:),'--k');
hold on
plot(t, -error_2sig(3,:),'--k');
ylabel('x_3 = Y (km)');
grid on;
xlim([0 14000])
xlabel('Time [sec]')
ylabel('State x_3 Error [km]')
legend('x_3 State Est. Error','2\sigma Error Bounds')
 
subplot(4, 1, 4);
plot(t, e_x(4, :)+w_k(4,:)*20);
hold on
plot(t, error_2sig(4,:),'--k');
hold on
plot(t, -error_2sig(4,:),'--k');
ylabel('x_4 = Ydot (km/s)');
xlabel('Time t (s)');
grid on;
xlim([0 14000])
xlabel('Time [sec]')
ylabel('State x_4 Error [km/s]')
legend('x_4 State Est. Error','2\sigma Error Bounds')
 
%% Plot Full States Errors Nonlinear from ODE45 (with initial perturbation) and EKF Estimate ZOOMED IN
figure;
subplot(4, 1, 1);
plot(t, e_x(1, :));
hold on
plot(t, error_2sig(1,:),'--k');
hold on
plot(t, -error_2sig(1,:),'--k');
sgtitle('State Errors and 2\sigma Bounds for EKF, One Orbital Period');
ylabel('x_1 = X (km)');
grid on;
xlim([0 14000])
xlabel('Time [sec]')
ylabel('State x_1 Error [km]')
legend('x_1 State Est. Error','2\sigma Error Bounds')
xlim([0 P])
ylim([-5 5])
 
subplot(4, 1, 2);
plot(t, e_x(2, :));
hold on
plot(t, error_2sig(2,:),'--k');
hold on
plot(t, -error_2sig(2,:),'--k');
ylabel('x_2 = Xdot (km/s)');
grid on;
xlim([0 14000])
xlabel('Time [sec]')
ylabel('State x_2 Error [km/s]')
legend('x_2 State Est. Error','2\sigma Error Bounds')
xlim([0 P])
ylim([-0.1 0.1])
 
subplot(4, 1, 3);
plot(t, e_x(3, :));
hold on
plot(t, error_2sig(3,:),'--k');
hold on
plot(t, -error_2sig(3,:),'--k');
ylabel('x_3 = Y (km)');
grid on;
xlim([0 14000])
xlabel('Time [sec]')
ylabel('State x_3 Error [km]')
legend('x_3 State Est. Error','2\sigma Error Bounds')
xlim([0 P])
ylim([-5 5])
 
subplot(4, 1, 4);
plot(t, e_x(4, :));
hold on
plot(t, error_2sig(4,:),'--k');
hold on
plot(t, -error_2sig(4,:),'--k');
ylabel('x_4 = Ydot (km/s)');
xlabel('Time t (s)');
grid on;
xlim([0 14000])
xlabel('Time [sec]')
ylabel('State x_4 Error [km/s]')
legend('x_4 State Est. Error','2\sigma Error Bounds')
xlim([0 P])
ylim([-0.1 0.1])
 
%% NEES Test
E_NEESbar=NEES_ekf;
NumSims=10
alphaNEES = 0.05;
Nnx = NumSims*4;
%Intervals
r1x = chi2inv(alphaNEES/2, Nnx )./ NumSims;
r2x = chi2inv(1-alphaNEES/2, Nnx )./ NumSims;
figure;
plot(E_NEESbar,'ro','MarkerSize',1,'LineWidth',2);
hold on;
plot(r1x*ones(size(E_NEESbar)),'r--','LineWidth',2)
plot(r2x*ones(size(E_NEESbar)),'r--','LineWidth',2)
ylabel('NEES','FontSize',14);
grid on;
xlabel('Time Step, k','FontSize',14);
title('NEES Estimation Results','FontSize',14);
lgd = legend('NEES', 'r_1 Bound', 'r_2 Bound');
 
%% NIS Test
E_NIS=100*NIS_ekf;
alphaNIS = 0.05;
r1y = chi2inv(alphaNIS/2,3*NumSims)./ NumSims;
r2y = chi2inv(1-alphaNIS/2,3*NumSims)./ NumSims;
figure;
plot(E_NIS,'bo','MarkerSize',1,'LineWidth',2);
hold on;
grid on;
plot(r1y*ones(size(E_NIS)),'b--','LineWidth',2);
plot(r2y*ones(size(E_NIS)),'b--','LineWidth',2);
ylabel('NIS statistic','FontSize',14);
xlabel('Time step, k','FontSize',14);
title('NIS Estimation Results','FontSize',14);
legend('NIS', 'r_1 bound', 'r_2 bound');
 
%% Plot EKF Estimate and 2sigma Error Bounds
figure;
sgtitle('Trajectory Estimation for Observed Data Log (EKF)');
 
subplot(4, 1, 1);
plot(t, xhatplus_kp1(1,:),'.k');
hold on
plot(t, xhatplus_kp1(1,:)+error_2sig(1,:),'--g');
hold on
plot(t, xhatplus_kp1(1,:)-error_2sig(1,:),'--g');
ylabel('x_1 = X (km)');
grid on;
legend('x_1','2\sigma Error Bounds');
%xlim([0 P])
ylim([-10000 10000])
 
subplot(4, 1, 2);
plot(t, xhatplus_kp1(2,:),'.k');
hold on
plot(t, xhatplus_kp1(2,:)+error_2sig(2,:),'--g');
hold on
plot(t, xhatplus_kp1(2,:)-error_2sig(2,:),'--g');
ylabel('x_2 = Xdot (km/s)');
grid on;
legend('x_2','2\sigma Error Bounds');
%xlim([0 P])
ylim([-10 10])
 
subplot(4, 1, 3);
plot(t, xhatplus_kp1(3,:),'.k');
hold on
plot(t, xhatplus_kp1(3,:)+error_2sig(3,:),'--g');
hold on
plot(t, xhatplus_kp1(3,:)-error_2sig(3,:),'--g');
ylabel('x_3 = Y (km)');
grid on;
legend('x_3','2\sigma Error Bounds');
%xlim([0 P])
ylim([-10000 10000])
 
subplot(4, 1, 4);
plot(t, xhatplus_kp1(4,:),'.k');
hold on
plot(t, xhatplus_kp1(4,:)+error_2sig(4,:),'--g');
hold on
plot(t, xhatplus_kp1(4,:)-error_2sig(4,:),'--g');
ylabel('x_4 = Ydot (km/s)');
grid on;
legend('x_4','2\sigma Error Bounds');
%xlim([0 P])
ylim([-10 10])
 
%% Plot EKF Estimate and 2sigma Error Bounds for One Orbital Period
figure;
sgtitle('Trajectory Estimation for Observed Data Log for One Orbital Period (EKF)');
 
subplot(4, 1, 1);
plot(t, xhatplus_kp1(1,:),'.k');
hold on
plot(t, xhatplus_kp1(1,:)+error_2sig(1,:),'--g');
hold on
plot(t, xhatplus_kp1(1,:)-error_2sig(1,:),'--g');
ylabel('x_1 = X (km)');
grid on;
legend('x_1','2\sigma Error Bounds');
xlim([0 5500])
ylim([-10000 10000])
 
subplot(4, 1, 2);
plot(t, xhatplus_kp1(2,:),'.k');
hold on
plot(t, xhatplus_kp1(2,:)+error_2sig(2,:),'--g');
hold on
plot(t, xhatplus_kp1(2,:)-error_2sig(2,:),'--g');
ylabel('x_2 = Xdot (km/s)');
grid on;
legend('x_2','2\sigma Error Bounds');
xlim([0 5500])
ylim([-10 10])
 
subplot(4, 1, 3);
plot(t, xhatplus_kp1(3,:),'.k');
hold on
plot(t, xhatplus_kp1(3,:)+error_2sig(3,:),'--g');
hold on
plot(t, xhatplus_kp1(3,:)-error_2sig(3,:),'--g');
ylabel('x_3 = Y (km)');
grid on;
legend('x_3','2\sigma Error Bounds');
xlim([0 5500])
ylim([-10000 10000])
 
subplot(4, 1, 4);
plot(t, xhatplus_kp1(4,:),'.k');
hold on
plot(t, xhatplus_kp1(4,:)+error_2sig(4,:),'--g');
hold on
plot(t, xhatplus_kp1(4,:)-error_2sig(4,:),'--g');
ylabel('x_4 = Ydot (km/s)');
grid on;
legend('x_4','2\sigma Error Bounds');
xlim([0 5500])
ylim([-10 10])
 
 
function [xhatplus_kp1,Phatplus_kp1,NEES_ekf,NIS_ekf] = ekf(x_k,P_k,u_k,Q_k,t_k,y_kp1,R_kp1,sID_kp1,DeltaT,xtruth_kp1, ytruth_kp1)
 
options = odeset('RelTol',1e-12,'AbsTol',1e-12);
 
% This is the first step in the EKF algorithm. This step defines the values
% of the state and covariance at step k. We will be solving for k+1
 
xhatplus_k = x_k;
Phatplus_k = P_k;
 
%% Step 3, Time Update/Prediction Step for time k+1
 
%% Use ODE45 to find xminus_kp1; this is the deterministic nonlinear DT
% dynamic fxn evaluation on xhatplus_k. Here, process noise is zero:
w_k = [0;0]; 
 
[~,updated_state] = ode45(@(t,x) NL_dynamics(t,x,u_k,w_k),[t_k t_k+DeltaT],xhatplus_k,options);
 
xminus_kp1 = updated_state(end,:)';
 
%% Approximate the predicted covariance via dynamic Taylor series linearization about xhatplus_k
% Need to first determine Ftilde_k and Omegatilde_k
[Ftilde_k,Omegatilde_k] = DTEulerJacs(xhatplus_k,DeltaT,t_k);
 
% Prediction of Covariance for k+1
Pminus_kp1 = (Ftilde_k*Phatplus_k*Ftilde_k') + (Omegatilde_k*Q_k*Omegatilde_k');
 
 
%% Measurement Update/Correction Step for time k+1
 
% Calculate Htilde_kp1 by linearizing about most recent est of total state
[~,~,~,Htilde_kp1,~,stations_kp1] = DTEulerJacss(xminus_kp1,DeltaT,t_k+DeltaT,sID_kp1);
 
%% Deterministic Nonlinear Function Eval for Measurement Update
 
% Calculate yminus_kp1 = h[xminus_kp1, v_kp1 = 0], no meas. noise
yminus_kp1 = YNL(stations_kp1, t_k + DeltaT, xminus_kp1);
 
if y_kp1(1:6) == [0;0;0;0;0;0]
    
    y_kp1=[];
    
elseif y_kp1(4:6)== [0;0;0]
    
    y_kp1=y_kp1(1:3);
    yminus_kp1=yminus_kp1(1:3);
    Htilde_kp1=Htilde_kp1(1:3,:);
 
end
 
if max(size(yminus_kp1))<max(size(y_kp1))
 
end
 
%% Calculate the Nonlinear Measurement Innovation
% Innovation = actual data minus predicted data
ey_kp1 = y_kp1 - yminus_kp1;
 
%% Calculate the KF Gain from the Measurement Linearization
if max(size(y_kp1))>4
    R_kp1 = mdiag(R_kp1,R_kp1);
end
 
if ~max(size(Htilde_kp1)) == 0
    K_kp1 = Pminus_kp1*Htilde_kp1'*inv(Htilde_kp1*Pminus_kp1*Htilde_kp1'+R_kp1);
    S_kp1 = Htilde_kp1*Pminus_kp1*Htilde_kp1'+R_kp1;
end
 
 
%% Update the Total State Estimate xhatplus_kp1
% This step updates the total state estimate if a measurement exists at
% this time step. Otherwise, it will keep the same estimate as before.
if exist('K_kp1','var')
    xhatplus_kp1 = xminus_kp1 + K_kp1*ey_kp1;
else
    xhatplus_kp1 = xminus_kp1;
end
 
%% Update the Covariance via Linearization
if exist('K_kp1','var')
    KH_kplus1 = K_kp1 * Htilde_kp1;
    Phatplus_kp1 = (eye(size(KH_kplus1)) - KH_kplus1)*Pminus_kp1;
else
    Phatplus_kp1 = Pminus_kp1;
    NEES_ekf=NaN;
    NIS_ekf=NaN;
    return;
end
 
%% NEES
if ~isempty(xtruth_kp1)
    stateVectErr=xtruth_kp1-xhatplus_kp1;
    NEES_ekf=stateVectErr'*(Phatplus_kp1)^-1*stateVectErr;
else
    NEES_ekf=NaN;
end
 
%% NIS
 
if ytruth_kp1(4:6)== [0;0;0]
    ytruth_kp1=ytruth_kp1(1:3);
end
 
if exist('ytruth_kp1','var')
    measErr=ytruth_kp1-yminus_kp1;
    NIS_ekf=measErr'*(S_kp1)^-1*measErr;
else
    NIS_ekf=NaN;
end
 
end
 
%% Function to simulate the full nonlinear dynamics with and without initial perturbations
% This function provides the initial conditions (intial state and intial
% perturbation) and run time into the function NLsim.
 
function [xNL, xNL_pert, t] = fullNL(x_pert)
 
mu = 398600; %km^3/s^2
r0 = 6678; %km
 
t_init = 0;
dT = 10;
t_fin = 14000;
xinit = [r0; 0; 0; r0*sqrt(mu/r0^3)];
xinit_pert = xinit + x_pert;
 
ode45_opts = odeset('RelTol',1e-12,'AbsTol',1e-13);
 
%OODE45 to Calculate Full NL Dynamics with NO Initial Perturbation
[t,xNL] = ode45('NLsim',t_init:dT:t_fin,xinit,ode45_opts);
 
%ODE45 to Calculate Full NL Dynamics with Initial Perturbation
[~,xNL_pert] = ode45('NLsim',t_init:dT:t_fin,xinit_pert,ode45_opts);
 
xNL = xNL';
xNL_pert = xNL_pert';
t=t';
 
end
 
%% Function to simulate the full nonlinear dynamics with and without initial perturbations
% This function provides the initial conditions (intial state and intial
% perturbation) and run time into the function NLsim.
 
function [xNL, xNL_pert, t] = fullNLnoise(x_pert,Qtrue,Rtrue)
 
mu = 398600; %km^3/s^2
r0 = 6678; %km
 
% Define the process noise and measurement noise cov matrices
Qf = Qtrue;
Rf = Rtrue;
S_w = chol(Qf,'lower');
S_v = chol(Rf,'lower');
 
t_init = 0;
dT = 10;
t_fin = 14000;
xinit = [r0; 0; 0; r0*sqrt(mu/r0^3)];
xinit_pert = xinit + x_pert;
 
 
ode45_opts = odeset('RelTol',1e-12,'AbsTol',1e-13);
 
%OODE45 to Calculate Full NL Dynamics with NO Initial Perturbation
[t,xNL] = ode45('NLsimNoise',t_init:dT:t_fin,xinit,ode45_opts);
 
%ODE45 to Calculate Full NL Dynamics with Initial Perturbation
[~,xNL_pert] = ode45('NLsimNoise',t_init:dT:t_fin,xinit_pert,ode45_opts);
 
xNL = xNL';
xNL_pert = xNL_pert';
t=t';
 
end
 
 
function [yk,idCount] = MeasFunc(stations,t,StateVector,vk)
    
    %StateVector to Easier to Use Form
    x1 = StateVector(1);
    x2 = StateVector(2);
    x3 = StateVector(3);
    x4 = StateVector(4);
 
    %Get all station locations at time t
    [Xi ,Yi ,Xidot ,Yidot ,thetai ] = stats(t);
 
    %Get Angles between Sat Position and Tracking Location, Check Tracking
    %Criteria
    phi_i=zeros(size(Xi));
    %theta_i=zeros(size(Xi));
    for i=1:12
        phi_i(i)=atan2((x3-Yi(i)),(x1-Xi(i)));
        %theta_i(i)=atan2(Yi(i),Xi(i));
    end
    
    yk=[];
    idCount=[];
    
    %Generate Yk matrices
    if ~isempty(stations)
        for iter=stations
            %Equations from doc:
            rho_i = sqrt((x1 - Xi(iter)).^2 + (x3 - Yi(iter)).^2);
            rho_dot_i = (((x1 - Xi(iter)).*(x2-Xidot(iter))) + ((x3 - Yi(iter)).*(x4-Yidot(iter))))./rho_i;
            
            %Stack them
            yk = [yk;[rho_i;rho_dot_i;phi_i(iter)]+vk];
        end
    else
        for iter=1:12
            %Equations from doc:
            rho_i = sqrt((x1 - Xi(iter)).^2 + (x3 - Yi(iter)).^2);
            rho_dot_i = (((x1 - Xi(iter)).*(x2-Xidot(iter))) + ((x3 - Yi(iter)).*(x4-Yidot(iter))))./rho_i;
            
            
            if visible(phi_i(iter),thetai(iter))
                yk = [yk;[rho_i;rho_dot_i;phi_i(iter)]+vk];
                idCount=[idCount;iter];
            end
            %Stack them
            
        end
        
        if(size(yk,1)==3)
            yk=[yk;[0;0;0]];
        elseif size(yk,1)==0
            yk=[0;0;0;0;0;0];
        end
        
        if max(size(idCount))==1
            idCount=[idCount,0];
        elseif max(size(idCount))==0
            idCount=[0;0];
        end
        
    end
end
 
%% Function that calculates the full nonlinear state dynamics
 
function [x_dot] = NL_dynamics(t,x,u,w)
 
mu = 398600; %km^3/s^2
 
X = x(1);
Xdot = x(2);
Y = x(3);
Ydot = x(4);
 
u1=u(1);
u2=u(2);
 
w1=w(1);
w2=w(2);
 
% Distance of spacecraft to Earth's center wrt Earth-Centered coords
r = sqrt(X^2 + Y^2);
 
% Defining the derivatives of the states
x_dot(1) = Xdot;
x_dot(2) = -(mu/r^3) * X + u1 + w1;
x_dot(3) = Ydot;
x_dot(4) = -(mu/r^3) * Y + u2 + w2;
 
x_dot = x_dot';
 
end
 
function [Xdot] = NLsim(t,xinit)
 
mu = 398600; % km^3/s^2
 
% State Vector
X = xinit(1);
Xdot = xinit(2);
Y = xinit(3);
Ydot = xinit(4);
 
% Define distance of spacecraft from Earth's center wrt Earth-centered
% coordinates
r = sqrt(X^2 + Y^2);
 
% Derivative Vector
Xdot(1) = Xdot;
Xdot(2) = -(mu/r^3)*X;
Xdot(3) = Ydot;
Xdot(4) = -(mu/r^3)*Y;
 
Xdot = Xdot';
end
 
function [Xdot] = NLsimNoise(t,xinit)
 
mu = 398600; % km^3/s^2
 
% State Vector
X = xinit(1);
Xdot = xinit(2);
Y = xinit(3);
Ydot = xinit(4);
 
% w1 = w_k(1);
% w2 = w_k(2);
 
% Define distance of spacecraft from Earth's center wrt Earth-centered
% coordinates
r = sqrt(X^2 + Y^2);
 
% Derivative Vector
Xdot(1) = Xdot;
Xdot(2) = -(mu/r^3)*X;
Xdot(3) = Ydot;
Xdot(4) = -(mu/r^3)*Y;
 
Xdot = Xdot';
end
 
 
 
function [Ctilde,station] = Ctilde_fxn(t,x)
 
    x1 = x(1);
    x2 = x(2);
    x3 = x(3);
    x4 = x(4);
 
    [Xi,Yi,Xidot,Yidot,thetai] = stats(t);
    
    phi_i=zeros(size(Xi));
    for i=1:12
        phi_i(i)=atan2((x3-Yi(i)),(x1-Xi(i)));
    end
    
    Ctilde=[];
    station=[];
 
    for i=1:12
       if visible(phi_i(i),thetai(i))
           station=[station, i];
           denom=sqrt((x1-Xi(i))^2+(x3-Yi(i))^2);
           
           Ctemp=[(x1-Xi(i))/denom , 0, (x3-Yi(i))/denom, 0; ...
               (x2-Xidot(i))/denom-(((x1-Xi(i))*(x2-Xidot(i))+(x3-Yi(i))*(x4-Yidot(i)))*(x1-Xi(i)))/(denom^3),...
               (x1-Xi(i))/denom,...
               (x4-Yidot(i))/denom-(((x1-Xi(i))*(x2-Xidot(i))+(x3-Yi(i))*(x4-Yidot(i)))*(x3-Yi(i)))/(denom^3),...
               (x3-Yi(i))/denom;...
               (Yi(i)-x3)/denom^2,0, (x1-Xi(i))/denom^2,0];               
           
           Ctilde=[Ctilde;Ctemp];
       end
    end
   
end
 
 
 
function [Ftilde,Omtilde] = DTEulerJacs(x,dT,t)
    
    mu = 398600; % km^3/s^2
    
    x1 = x(1);
    x2 = x(2);
    x3 = x(3);
    x4 = x(4);
        
    % Calculate the Atilde CT STM Based on Most Recent State Estimate
    Atilde = [0,1,0,0;(2*mu*x1^2-mu*x3^2)/((x1^2+x3^2)^(5/2)),0,(3*mu*x1*x3)/((x1^2+x3^2)^(5/2)),0; 0,0,0,1;(3*mu*x1*x3)/((x1^2+x3^2)^(5/2)),0,(2*mu*x3^2-mu*x1^2)/((x1^2+x3^2)^(5/2)),0];
    % Define Btild CT Matric
    Btilde = [0,0;1,0;0,0;0,1];
    % Calculate the Ctilde Matrix Based on Most Recent State Estimate
    [Ctilde,station] = Ctilde_fxn(t,x);
    Dtilde = zeros(max(size(station))*3,2);
    Gamma= Btilde;
    Ftilde = eye(size(Atilde))+ dT*Atilde;
    Gtilde = dT*Btilde;
    Omtilde = dT*Gamma;
    Htilde = Ctilde;
    Mtilde = Dtilde;
 
end
 
 
function [Ftilde,Gtilde,Omtilde,Htilde,Mtilde,ObservingStations]...
    = DTEulerJacss(StateVector, DeltaT,t,sID)
    
    x1=StateVector(1);
    x3=StateVector(3);
    
    %OD Specific Constants
    mu=398600; %km^3 s^-2
        
    % OD Specific Matricies
    Atilde = [0,1,0,0;...
        (2*mu*x1^2-mu*x3^2)/((x1^2+x3^2)^(5/2)),0,(3*mu*x1*x3)/((x1^2+x3^2)^(5/2)),0;...
        0,0,0,1;...
        (3*mu*x1*x3)/((x1^2+x3^2)^(5/2)),0,(2*mu*x3^2-mu*x1^2)/((x1^2+x3^2)^(5/2)),0];
    
    Btilde = [0,0;1,0;0,0;0,1];
    
    
    [Ctilde,ObservingStations] = OD_CtildeMatr2(t,StateVector,sID);
        
    Dtilde = zeros(max(size(ObservingStations))*3,2);
    
    Gamma= Btilde;
        
    %General Formula for Eulerized DT Jacobians
    
    
    Ftilde = eye(size(Atilde))+ DeltaT*Atilde;
    Gtilde = DeltaT*Btilde;
    Omtilde = DeltaT*Gamma;
    Htilde = Ctilde;
    Mtilde = Dtilde;
 
end
