clc; clear all; close all; warning off;
% Nonlinear System Koopman eDMD + LinearMPC Controller
% System:  x1[n+1] = 0.1 - x1^2 + x1*x2
%          x2[n+1] = -x1 + exp(-x2) + u
%          y[n]    = x1 + x2
% Written By: Rasit Evduzen
% Date: 28-Apr-2026
%% ----------- Lifting Selection -----------
% 'physics' : physics-informed basis [x1,x2,x1^2,x1*x2,exp(-x2),1]  →  n_lift=6
% 'poly'    : polynomial basis up to 3rd order                        →  n_lift=8
% 'rbf'     : Gaussian radial basis functions centered on data        →  n_lift=Nc+3
liftType = 'rbf';

%% Algorithm Parameters
Ts         = 1e-3;    % sampling time [s]
T          = 20;      % simulation time [s]
N          = round(T / Ts);
umin       = -3;      % control lower saturation limit [N]
umax       =  3;      % control upper saturation limit [N]
Ku         = 5;       % MPC control horizon  (number of free moves)
Ky         = 10;      % MPC prediction horizon
lamda      = 0.25;    % delta-u penalty weight (larger = smoother input)
alpha_edmd = 1e-6;    % Tikhonov regularization for eDMD least-squares
N_data     = 5000;    % number of PRBS samples for offline eDMD training

% RBF parameters (only used when liftType = 'rbf')
Nc      = 8;    % number of RBF centers
rbf_eps = 2.0;  % RBF width

plot_interval = 500;

%% Reference Signal
t_hist = (0:N)'      * Ts;
t_ref  = (0:N+Ky-2)' * Ts;
half   = floor(length(t_ref) / 2);
seg    = floor(half / 4);
ref_step = [0.4*ones(seg,1); 0.8*ones(seg,1); ...
            1.2*ones(seg,1); 0.8*ones(half-3*seg,1)];
ref_sin  = 0.3*sin(0.5*pi*t_ref(1:length(t_ref)-half)) + 0.8;
yref     = [ref_step; ref_sin];

%%  OFFLINE PHASE  Collect PRBS training data
rng(42);
u_prbs = zeros(1, N_data); period = 20;
for i = 1:N_data
    if mod(i,period)==0; u_prbs(i) = 2*(rand-0.5);
    else; u_prbs(i) = u_prbs(max(i-1,1)); end
end

x1d = zeros(1,N_data); x2d = zeros(1,N_data);
for n = 1:N_data-1
    x1d(n+1) = 0.1 - x1d(n)^2 + x1d(n)*x2d(n);
    x2d(n+1) = -x1d(n) + exp(-x2d(n)) + u_prbs(n);
end

% RBF centers: uniform grid over observed state space
if strcmp(liftType, 'rbf')
    x1_range = linspace(min(x1d), max(x1d), ceil(sqrt(Nc)));
    x2_range = linspace(min(x2d), max(x2d), ceil(sqrt(Nc)));
    [C1, C2] = meshgrid(x1_range, x2_range);
    centers  = [C1(:)'; C2(:)']; centers = centers(:,1:Nc);
else
    centers = [];
end


%%  OFFLINE PHASE 2  eDMD: learn Koopman operator K and B_lift
% Lift all snapshots: G = [g(x1),...], Gp = [g(x1+1),...]
n_lift = GetNlift(liftType, Nc);
G  = zeros(n_lift, N_data-1);
Gp = zeros(n_lift, N_data-1);
for n = 1:N_data-1
    G(:,n)  = LiftState([x1d(n);   x2d(n)],   liftType, centers, rbf_eps);
    Gp(:,n) = LiftState([x1d(n+1); x2d(n+1)], liftType, centers, rbf_eps);
end
U_data = u_prbs(1:N_data-1);

% Tikhonov-regularized least squares: Gp ≈ K*G + B*U
Omega_edmd = [G; U_data];
KB         = Gp * Omega_edmd' / (Omega_edmd*Omega_edmd' + alpha_edmd*eye(n_lift+1));
K_koop     = KB(:, 1:n_lift);  % [n_lift x n_lift] Koopman matrix
B_lift     = KB(:, end);       % [n_lift x 1]       lifted input matrix

% Output map: y = x1 + x2 = g(1) + g(2) for all lifting types
c_lift = zeros(1, n_lift); c_lift(1) = 1; c_lift(2) = 1;

%%  OFFLINE PHASE  MPC matrices (computed once from Koopman model)
[L, M, Z] = LDTS_MPC_matrices(K_koop, B_lift, c_lift', Ku, Ky);
Hes   = M'*M + lamda*L;
Hters = Hes \ eye(size(Hes));

%% Closed-loop simulation
x = zeros(2, N+1);   x(:,1) = [0; 0];
u = zeros(Ku+1, N+1);
y = zeros(1, N+1);
y(1) = c_lift * LiftState(x(:,1), liftType, centers, rbf_eps);

figure('units','normalized','outerposition',[0 0 1 1],'color','w');
for i = 1:N

    g_cur  = LiftState(x(:,i), liftType, centers, rbf_eps);

    gv     = Hes*u(:,i)-(M'*(yref(i:i+Ky-1)-Z*g_cur)+[lamda*u(1,i);zeros(Ku,1)]);
    deltau = -Hters*gv;

    mu     = min(0.1, 0.1/max([max(abs(deltau)),max(abs(diff(u(:,i)+deltau))), 1e-6]));
    deltau = mu*deltau;

    u(:,i+1) = u(:,i)+deltau;
    u(1,i+1) = max(umin,min(umax,u(1,i+1)));

    x(:,i+1) = NLSystem(x(:,i), u(1,i+1));
    y(i+1)   = c_lift*LiftState(x(:,i+1), liftType, centers, rbf_eps);

    if mod(i, plot_interval) == 0 || i == N
        clf

        subplot(221); hold on; grid on;
        plot(t_hist(1:N),   yref(1:N),  'r', 'LineWidth', 2);
        plot(t_hist(1:i+1), y(1:i+1),   'k', 'LineWidth', 2);
        xlabel('t [s]'); ylabel('y');
        if i == N
            RMSE = sqrt(mean((yref(1:N)' - y(1:N)).^2));
            title(sprintf('Koopman MPC [%s]  |  RMSE=%.5f', liftType, RMSE));
        else
            title(sprintf('Koopman MPC [%s]  |  n_{lift}=%d', liftType, n_lift));
        end
        legend('y_{ref}','y[n]');
        axis([0 T min(yref(1:N))-0.3 max(yref(1:N))+0.3]);

        subplot(222); hold on; grid on;
        plot(t_hist(1:i+1), yref(1:i+1)' - y(1:i+1), 'b', 'LineWidth', 1);
        yline(0,'k--');
        xlabel('t [s]'); title('Tracking Error');

        subplot(223); hold on; grid on;
        plot(t_hist(1:i+1), u(1,1:i+1), 'b', 'LineWidth', 1);
        yline( umax,'r--','u_{max}');
        yline( umin,'r--','u_{min}');
        xlabel('t [s]'); ylabel('u'); title('Control Input');

        subplot(224); hold on; grid on;
        plot(t_hist(1:i+1), x(1,1:i+1), 'k--', 'LineWidth', 1);
        plot(t_hist(1:i+1), x(2,1:i+1), 'r--', 'LineWidth', 1);
        xlabel('t [s]'); ylabel('x_i'); title('States');
        legend('x_1','x_2');

        drawnow;
    end
end



%% LOCAL FUNCTIONS


function n = GetNlift(liftType, Nc)
    switch liftType
        case 'physics'; n = 6;
        case 'poly';    n = 8;
        case 'rbf';     n = 2 + Nc + 1;
    end
end

% -------------------------------------------------------------------------
function g = LiftState(x, liftType, centers, rbf_eps)
    switch liftType
        case 'physics'
            g = [x(1); x(2); x(1)^2; x(1)*x(2); exp(-x(2)); 1];

        case 'poly'
            g = [x(1); x(2); x(1)^2; x(2)^2; x(1)*x(2); x(1)^3; x(2)^3; 1];

        case 'rbf'
            Nc = size(centers,2);
            rbfs = zeros(Nc,1);
            for k = 1:Nc
                d = x - centers(:,k);
                rbfs(k) = exp(-rbf_eps^2 * (d'*d));
            end
            g = [x(1); x(2); rbfs; 1];
    end
end

% -------------------------------------------------------------------------
function x_next = NLSystem(x, u)
    x_next    = zeros(2,1);
    x_next(1) = 0.1 - x(1)^2 + x(1)*x(2);
    x_next(2) = -x(1) + exp(-x(2)) + u;
end

% -------------------------------------------------------------------------
function [L, M, Z] = LDTS_MPC_matrices(A, b, c, Ku, Ky)
    L = 2*eye(Ku+1); L(end,end) = 1;
    for i = 1:Ku; L(i+1,i)=-1; L(i,i+1)=-1; end
    if Ku==0; L=1; end

    Z = zeros(Ky, size(A,1));
    for k = 1:Ky; Z(k,:) = c'*A^k; end

    M = zeros(Ky, Ku+1);
    M(1,:) = [c'*b, zeros(1,Ku)];
    for i = 2:Ky; M(i,:) = [c'*A^(i-1)*b, M(i-1,1:end-1)]; end
    for k = Ku+2:Ky
        aa = 0;
        for l = 1:k-Ku-1; aa = aa + c'*A^(l-1)*b; end
        M(k,Ku+1) = M(k,Ku+1) + aa;
    end
end