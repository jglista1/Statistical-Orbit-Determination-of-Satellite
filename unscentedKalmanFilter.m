clear all
load('orbitdeterm_finalproj_KFdata.mat');
r0 = 6678; %km
mu = 398600; %km^3/sec^2
dT = 10; %sec
tspan = 0:dT:1400*dT;
 
n = 4; alpha = 0.75; beta = 2; kappa = 0; lambda = alpha^2*(n-kappa)-n;
 
x0 = [r0; 0; 0; r0*sqrt(mu/r0^3)];
dx_init = [0; 0.075; 0; -0.021];
x_init = x0;
x_init_Pert = x0 + dx_init;
 
 
%% Step 3: Execute LKF with Sample Noisy Data
P0 = 0.2*diag([r0, r0*sqrt(mu/r0^3), r0, r0*sqrt(mu/r0^3)]);
Q = Qtrue; R = Rtrue;
[xUKF, yUKF, PUKF, SUKF] = UKF(x_init_Pert, n, mu, lambda,alpha,beta, tspan, dT, R, Q, P0);
 
plotStates(tspan, xUKF, 'UKF')
 
% plotMeas(tspan, yLKF, 'yLKF')
% plotStates(tspan, xLKF, 'xLKF')
% plotStates(tspan, xLKF - xSim,'LKF Estimation Error')
% plotNEES(tspan, NEES, iMC, 'NEES')
% plotNIS(tspan, NIS,iMC, 'NIS')
 
function plotNIS(tspan, NIS, iMC, title)
figure;
hold on;
    for m = 1:iMC
        for i = 1:12
            plot(tspan, NIS(:,i,m), 'ob')
        end
    end
hold off;
xlabel('Time [s]'); ylabel('NIS Statistic')
end
function plotNEES(tspan, NEES,iMC, title)
    figure
    hold on
    for m = 1:iMC
        plot(tspan, NEES(:,m),'or')
    end
    hold off
    xlabel('Time [s]'); ylabel('NEES Statistic')
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
function plotStates(tspan, x, title)
%% Plot LFK Total State Estimation
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
xlabel('Time t (s)');
grid on;
end
function [chi_array chibar_array] = sigmaPoints(x, n, mu, lambda,dT, S)
        chi_array = x;
        chibar = nominalTrajectory(mu,[0 dT 2*dT], chi_array);
        chibar_array = chibar(:,1);
        for i = 1:n        
            chi = x + (sqrt(n+lambda)*S(i,:)');
            chibar = nominalTrajectory(mu, [0 dT 2*dT], chi);
            chi_array = [chi_array chi]; chibar_array = [chibar_array chibar(:,1)]; 
        end
        for i = n+1:2*n
            chi = x - (sqrt(n+lambda)*S(i-n,:)');
            chibar = nominalTrajectory(mu, [0 dT 2*dT], chi);
            chi_array = [chi_array chi]; chibar_array = [chibar_array chibar(:,1)]; 
        end
end
function [x_array, y_array, P_array, S_array] = UKF(x_init, n, mu, lambda,alpha,beta, tspan, dT, Rf, Qf, P0)
    %% Unscented Kalman Filter
 
    wm = [lambda/(n+lambda)]; wc = [lambda/(n+lambda)+1-alpha^2+beta];
    for i = 2:2*n+1
        wm = [wm; lambda/(2*(n+lambda))];
        wc = [wc; lambda/(2*(n+lambda))];
    end
    O = [0 0; 1 0; 0 0; 0 1];
    
    load('orbitdeterm_finalproj_KFdata.mat');
    % Encode KF initial mean and covariance
    x_plus = x_init;
    P_plus = P0;
    T = length(tspan);
    x_array = zeros(4,T);
    y_array = zeros(4,T,12);
    P_array = zeros(4,4,T);
    S_array = zeros(4,4,T,12);
 
    for k = 2:T
        ydata_k = cell2mat(ydata(k));
        
% Prediction step should be done once per timestep
        S = chol(P_plus);
 
        [chi_array, chibar_array] = sigmaPoints(x_plus, n, mu, lambda,dT, S); % calculate sigma points
 
            x_minus = zeros(n,1); P_minus = zeros(n,n);% clear 
        for m = 1:2*n+1
            x_minus = x_minus + wm(m)*chibar_array(:,m);
            P_minus = P_minus + wc(m)*(chibar_array(:,m)-x_minus)*(chibar_array(:,m)-x_minus)'+O*Qf*O';
        end
 
        for i = 1:12
 
            if size(ydata_k) == [0 0]
                continue
            end
            if i == ydata_k(4,1) 
                y = ydata_k(1:3,1);
            elseif size(ydata_k) == [4 2]
                if i == ydata_k(4,2)
                    y = ydata_k(1:3,2);
                else 
                    continue % skip update if tracking station cannot see satellite
                end
            else
                continue
            end
 
         % Measurement Update Step           
            Sbar = chol(P_minus); % Innovation
 
            % Generate new sigma points
            [chi_array, chibar_array] = sigmaPoints(x_minus, n, mu, lambda,dT, Sbar); % calculate sigma points
 
            gamma_array = []; y_pred = 0; 
            for m = 1:2*n+1 % Propagate chi through measurement function and calculate predicted measurements
                
                gamma = ChiToGamma(dT, k, i, chi_array(:,m));
                gamma_array = [gamma_array gamma];
                y_pred = y_pred + wm(m)*gamma;
            end
 
            Pyy = zeros(size(Rf)); Pxy = zeros(size(x_minus*y_pred'));
            for m = 1:2*n+1 
                % Calculate measurement covariance
                Pyy = Pyy + wc(m)*(gamma_array(:,m)-y_pred)*(gamma_array(:,m)-y_pred)'+Rf;
                % Calculate state-measurement cross covariance
                Pxy = Pxy + wc(m)*(chi_array(:,m)-x_minus)*(gamma_array(:,m)-y_pred)';
            end
            
            
            % Estimate Gain
            K = Pxy*inv(Pyy);
            x_plus = x_minus + K*(y - y_pred); % Update mean with gain times simulated (real) - predicted measurements
            P_plus = P_minus - K*Pyy*K'; % Update covariance
 
            y_array(1:3,k,i) = y_pred;
            y_array(4,k,i) = i;
            S_array(:,:,k,i) = Sbar; 
        end
        % Store state and covariance in array 
        x_array(:,k) = x_plus;
        P_array(:,:,k) = P_plus;
    end
end
function [gamma] = ChiToGamma(dT, k, i, chi)
    tspan = 1;
% Calculate associated measurements
    gamma = zeros(3,1); 
    x2 = chi';
    
    RE = 6378; %km
    wE = 2*pi/86400; %rad/s
    
            ti = ((k-1)*dT);
            theta0 = (i-1)*(pi/6);
            Xi = RE*cos(wE*ti + theta0);
            Yi = RE*sin(wE*ti + theta0);
            Xidot = -RE*wE*sin(wE*ti + theta0);
            Yidot = RE*wE*cos(wE*ti + theta0);
 
            thetai = atan2(Yi,Xi);
            phii = atan2((x2(3)-Yi),(x2(1)-Xi));
 
                
            gamma(1) = sqrt((x2(1)-Xi)^2 + (x2(3)-Yi)^2);
            gamma(2) = ((x2(1)-Xi)*(x2(2)-Xidot)+(x2(3)-Yi)*(x2(4)-Yidot))/gamma(1);
            gamma(3) = phii;
end
function [yMeas] = StateToMeas(tspan, dT, x)
    % Calculate associated measurements
    yMeas = NaN*ones(4,length(tspan),12); 
    Xs = NaN*ones(length(tspan),4,12);
    x2 = x';
    
    RE = 6378; %km
    wE = 2*pi/86400; %rad/s
    
    for k = 1:size(x2,1)
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
