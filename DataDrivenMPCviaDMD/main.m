clc; clear all; close all; warning off;
% Nonlinear System  DMDc LTV + MPC Controller
% Approach:
%   1. Sliding window of W steps → DMD regression → local A(n), B(n)
%   2. LinearMPC on local model → u(n)
%   3. True nonlinear system update
%   4. Slide window → repeat
%
% System:  x1[n+1] = 0.1 - x1^2 + x1*x2
%          x2[n+1] = -x1 + exp(-x2) + u
%          y[n]    = x1 + x2

%% Algorithm Parameters
Ts    = 1e-3;          % sampling time [s]
T     = 20;            % simulation time [s]
N     = round(T / Ts); % simulation steps
umin  = -3;            % input lower bound
umax  =  3;            % input upper bound
Ku    = 5;             % control horizon
Ky    = 10;            % prediction horizon
W     = 10;            % DMD sliding window size 
lamda = 0.25;           % delta-u penalty
alpha = 1e-9;               % regularization parameter
c     = [1; 1];        % output matrix: y = x1 + x2

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

%% Initial Conditions
x  = zeros(2, N+1);   x(:,1) = [0; 0];
u  = zeros(Ku+1, N+1);
y  = zeros(1, N+1);   y(1) = c' * x(:,1);

% Pre-fill window with open-loop PRBS data
rng(42);
X_buf = zeros(2, W+1);   % state buffer   [2 x W+1]
U_buf = zeros(1, W);     % input buffer   [1 x W]
x_buf = x(:,1);
for k = 1:W
    u_prbs      = (0.1*rand)-0.1;
    X_buf(:,k)  = x_buf;
    U_buf(k)    = u_prbs;
    x_buf       = NLSystem(x_buf, u_prbs);
end
X_buf(:,W+1) = x_buf;

%% Default linear model for DMD
A_cur = eye(2);
B_cur = [0; 1];

%% Main Loop
figure('units','normalized','outerposition',[0 0 1 1],'color','w');

for i = 1:N

    % --- DMD Regression: fit local A(n), B(n) from window ---
    Xw  = X_buf(:, 1:W);       % [2 x W]  current states
    Xwp = X_buf(:, 2:W+1);     % [2 x W]  next states
    Uw  = U_buf;                % [1 x W]  inputs

    % Least-squares: [A B] = Xw' * pinv([Xw; Uw])
    Omega = [Xw; Uw];           % [3 x W]
    AB    = Xwp * Omega' / (Omega*Omega' + alpha*eye(3));  % [2 x 3]
    A_cur = AB(:, 1:2);         % [2 x 2]
    B_cur = AB(:, 3);           % [2 x 1]


    % --- MPC on local model ---
    [L, M, Z] = LDTS_MPC_matrices(A_cur, B_cur, c, Ku, Ky);
    Hes   = M'*M + lamda*L;
    Hters = Hes \ eye(size(Hes));

    g      = Hes*u(:,i) - (M'*(yref(i:i+Ky-1) - Z*x(:,i)) + [lamda*u(1,i); zeros(Ku,1)]);
    deltau = -Hters * g;

    % Rate constraint
    mu     = min(0.1, 0.1 / max([max(abs(deltau)),max(abs(diff(u(:,i)+deltau))), 1e-6]));
    deltau = mu * deltau;

    % Update and saturate
    u(:,i+1)   = u(:,i) + deltau;
    u(1,i+1)   = max(umin, min(umax, u(1,i+1)));

    %--- True nonlinear system update ---
    x(:,i+1) = NLSystem(x(:,i), u(1,i+1));
    y(i+1)   = c' * x(:,i+1);

    % --- Slide window ---
    X_buf = [X_buf(:,2:end), x(:,i+1)];   % drop oldest, append new
    U_buf = [U_buf(2:end),   u(1,i+1)];

    % --- Live Plot ---
    if mod(i, plot_interval) == 0 || i == N
        clf

        subplot(221); hold on; grid on;
        plot(t_hist(1:N),   yref(1:N),  'r', 'LineWidth', 2);
        plot(t_hist(1:i+1), y(1:i+1),   'k', 'LineWidth', 2);
        xlabel('t [s]'); ylabel('y');
        if i == N
            RMSE = sqrt(mean((yref(1:N)' - y(1:N)).^2));
            title(sprintf('DMDc-LTV MPC  |  RMSE=%.5f', RMSE));
        else
            title(sprintf('DMDc-LTV MPC  |  W=%d  Ky=%d  Ku=%d', W, Ky, Ku));
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
function x_next = NLSystem(x, u)
% True nonlinear system
x_next = zeros(2,1);
x_next(1) = 0.1 - x(1)^2 + x(1)*x(2);
x_next(2) = -x(1) + exp(-x(2)) + u;
end

% -------------------------------------------------------------------------
function [L, M, Z] = LDTS_MPC_matrices(A, b, c, Ku, Ky)
% MPC prediction matrices (same as LinearMPC.m)

% L tridiagonal delta-u penalty (Ku+1 x Ku+1)
L = 2*eye(Ku+1); L(end,end) = 1;
for i = 1:Ku
    L(i+1,i) = -1; L(i,i+1) = -1;
end
if Ku == 0; L = 1; end

% Z free response (Ky x nx)
Z = zeros(Ky, length(b));
for k = 1:Ky
    Z(k,:) = c'*A^k;
end

% M  forced response (Ky x Ku+1)
M = zeros(Ky, Ku+1);
M(1,:) = [c'*b, zeros(1,Ku)];
for i = 2:Ky
    M(i,:) = [c'*A^(i-1)*b, M(i-1,1:end-1)];
end
for k = Ku+2:Ky
    aa = 0;
    for l = 1:k-Ku-1
        aa = aa + c'*A^(l-1)*b;
    end
    M(k,Ku+1) = M(k,Ku+1) + aa;
end
end