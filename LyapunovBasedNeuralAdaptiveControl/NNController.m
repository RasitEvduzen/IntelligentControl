%% LYAPUNOV-BASED NEURAL ADAPTIVE CONTROL
%  TRUE SYSTEM — OPEN-LOOP DIFFERENTIAL EQUATION (2nd order nonlinear):
%  d^2(y)/dt^2 = -2*sin(y) - 0.8*(dy/dt)*|dy/dt| + w(t) + b*u
%  STATE-SPACE FORM:
%  dx1/dt = x2                      (x1 = y     : position)
%  dx2/dt = f(x1, x2) + w(t) + b*u  (x2 = dy/dt : velocity)
%  NEURAL NETWORK LEARNING LAW (Lyapunov-derived):
%  dW_i/dt = gamma * r(t) * Phi_i(X)   (adaptation law for each neuron weight)

clear; clc; close all;

%%  Simulation Parameters
dt    = 0.01;        % sampling time [s]
T_max = 20;            % total simulation time [s]
t     = 0:dt:T_max;
N     = length(t);
plot_interval = 50;   % live plot refresh rate

%%  System Parameters
b = 2.5;               % true input gain (assumed known by the controller)

%% RBF Neural Network Architecture
%  Network input is the instantaneous state vector: X = [x1; x2] = [y; dy/dt]
C = [-2.0, -1.0, 0.0, 1.0, 2.0;    % Gaussian centers for input 1 (y)
     -2.0, -1.0, 0.0, 1.0, 2.0];   % Gaussian centers for input 2 (dy/dt)
B_width     = 3.0;                  % Gaussian width / spread factor (variance)
num_neurons = size(C, 2);           % total number of hidden neurons (5)

% --- NETWORK WEIGHT MATRIX (W) ---
W = zeros(num_neurons, N);
W(:,1) = 0.1 * ones(num_neurons, 1);  % initial weights at t=0

%% Controller Design & Learning Rate (Tuning Parameters)
lambda = 2.5;           % error-surface filter coefficient (r = de/dt + lambda*e)
kv     = 12.0;          % PD-type gain governing overall system stiffness
gamma  = 80.0;          % NEURAL NETWORK LEARNING RATE (multiplier in dW/dt)

%% Reference Trajectory (NARMA-L2 style: multi-step + sinusoidal)
%  First half: multi-step reference, second half: sinusoidal reference
%  Analytical derivatives (dyd, ddyd) are required since this is a 2nd-order system.
half = floor(N / 2);
seg  = floor(half / 4);

yd_step = [0.4*ones(seg,1); 1.0*ones(seg,1); ...
           1.6*ones(seg,1); 1.0*ones(half-3*seg,1)];
dyd_step  = zeros(half, 1);    % piecewise-constant → zero derivative
ddyd_step = zeros(half, 1);

t_sin = t(half+1 : N)' - t(half+1);     % local time for sinusoidal segment
A_sin = 0.6;  w_sin = 0.5*pi;  off_sin = 1.0;

yd_sin   =  A_sin*sin(w_sin*t_sin) + off_sin;
dyd_sin  =  A_sin*w_sin*cos(w_sin*t_sin);
ddyd_sin = -A_sin*w_sin^2*sin(w_sin*t_sin);

yd   = [yd_step;   yd_sin]';
dyd  = [dyd_step;  dyd_sin]';
ddyd = [ddyd_step; ddyd_sin]';

%% Memory Allocation
y  = zeros(1, N);   dy  = zeros(1, N);    % true system states (y and dy/dt)
u  = zeros(1, N);   r_surface = zeros(1, N);
f_actual = zeros(1, N);   f_estimated = zeros(1, N);

%% Initial Conditions
y(1) = 0; dy(1) = 0;    % x1(0) = 0, x2(0) = 0

%% Main Simulation Loop (RK4 Integration)
figure('units','normalized','outerposition',[0 0 1 1],'color','w');
for k = 1:N-1

    %% 8.1. Filtered Error Surface
    %  DIFFERENTIAL EQUATION: r(t) = de/dt + lambda * e
    %  e = y - yd  and  de/dt = dy/dt - dyd/dt
    e  = y(k) - yd(k);          % position error
    de = dy(k) - dyd(k);        % velocity error (de/dt)
    r  = de + lambda * e;       % Lyapunov error surface (r)
    r_surface(k) = r;

    %% Neural Network Forward Pass
    X_input = [y(k); dy(k)];    % input vector X = [x1; x2]

    %  Phi_i = exp( - ||X - C_i||^2 / B_width^2 )
    Phi = zeros(num_neurons, 1);
    for i = 1:num_neurons
        dist_sq = sum((X_input - C(:, i)).^2);
        Phi(i)  = exp(-dist_sq / (B_width^2));
    end

    %  INSTANTANEOUS FUNCTION ESTIMATE: f_hat(t) = W(t)^T * Phi(X)
    f_hat = W(:, k)' * Phi;
    f_estimated(k) = f_hat;

    %% Lyapunov-Based Online Learning Law (Weight Adaptation)
    %  DIFFERENTIAL EQUATION: dW/dt = gamma * r(t) * Phi(X)
    W_dot = gamma * r * Phi;

    % NUMERICAL INTEGRATION (Euler): W(t+dt) = W(t) + (dW/dt) * dt
    W(:, k+1) = W(:, k) + dt * W_dot;

    %% Neural Control Law
    %  u(t) = (1/b) * ( -f_hat(t) + d^2(yd)/dt^2 - lambda*(de/dt) - kv*r(t) )
    u(k) = (1/b) * (-f_hat + ddyd(k) - lambda*de - kv*r);
    u(k) = min(max(u(k), -30), 30);   % actuator saturation

    %% True System Dynamics (RK4 Solver)
    %  OPEN-LOOP DIFFERENTIAL EQUATION SYSTEM:
    %  dy/dt      = state(2)
    %  d^2(y)/dt^2 = -2*sin(y) - 0.8*(dy/dt)*|dy/dt| + w(t) + b*u
    w = 0;
    if t(k) >= T_max/2
        w = 15.0;     % external disturbance shock injected 
    end
    f_internal   = -2 * sin(y(k)) - 0.8 * dy(k) * abs(dy(k));
    f_actual(k)  = f_internal + w;   % true total physical f(t)

    sys_dynamics = @(t_val, state, ctrl) [state(2); f_internal + w + b * ctrl];

    current_state = [y(k); dy(k)];
    k1 = sys_dynamics(t(k),        current_state,           u(k));
    k2 = sys_dynamics(t(k)+dt/2,   current_state + dt/2*k1,  u(k));
    k3 = sys_dynamics(t(k)+dt/2,   current_state + dt/2*k2,  u(k));
    k4 = sys_dynamics(t(k)+dt,     current_state + dt*k3,    u(k));

    next_state = current_state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);
    y(k+1)  = next_state(1);   % new position: y(t+dt)
    dy(k+1) = next_state(2);   % new velocity: dy/dt(t+dt)

    %% Live Plot
    if mod(k, plot_interval) == 0 || k == N-1
        clf

        subplot(221); hold on; grid on;
        plot(t(1:k), yd(1:k), 'r--', 'LineWidth', 2);
        plot(t(1:k), y(1:k),  'b-',  'LineWidth', 1.5);
        xlabel('t [s]'); ylabel('y');
        if k == N-1
            RMSE = sqrt(mean((yd(1:k) - y(1:k)).^2));
            title(sprintf('Tracking Performance  |  RMSE=%.5f', RMSE));
        else
            title(sprintf('Tracking Performance  |  \\gamma=%.1f  k_v=%.1f', gamma, kv));
        end
        legend('y_{ref}', 'y[n]', 'Location','best');
        axis([0 T_max min(yd)-1 max(yd)+1]);

        subplot(222); hold on; grid on;
        plot(t(1:k), f_actual(1:k),    'r-',  'LineWidth', 2);
        plot(t(1:k), f_estimated(1:k), 'b--', 'LineWidth', 1.5);
        xlabel('t [s]'); ylabel('f(x)');
        legend('f true','f\_hat','Location','best');
        title('Online Function Approximation');

        subplot(223); hold on; grid on;
        plot(t(1:k), u(1:k), 'm-', 'LineWidth', 1.5);
        yline( 30,'r--','u_{max}');
        yline(-30,'r--','u_{min}');
        xlabel('t [s]'); ylabel('u'); title('Control Input');

        subplot(224); hold on; grid on;
        plot(t(1:k), W(:,1:k)', 'LineWidth', 1.2);
        xlabel('t [s]'); ylabel('W_i');
        title('Neural Network Weight Evolution (Lyapunov)');

        sgtitle('Lyapunov-Based Neural Adaptive Control', 'FontSize', 14);
        drawnow;
    end
end
