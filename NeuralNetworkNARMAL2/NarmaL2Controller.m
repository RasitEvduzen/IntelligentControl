clc; clear all; close all;
% Nonlinear System — NARMA-L2 Controller
% Written By: Rasit Evduzen
% Date: 28-Mar-2026
%   u(n) = (yref(n+1) - NN_f(x1,x2)) / NN_g(x1,x2)

%% Load Plant Models
load('NL_PlantModel.mat', ...
     'Wg_f','bh_f','Wc_f','bc_f', ...
     'Wg_g','bh_g','Wc_g','bc_g', ...
     'S','R_fg');
fprintf('Plant models loaded.  S=%d  R_fg=%d\n', S, R_fg);

%% Algorithm Parameters
Ts    = 1e-1;          % Sampling time [s]
T     = 40;            % Total simulation time [s]
N     = round(T / Ts); % Number of simulation steps
u_max = 3.0;           % Control saturation

plot_interval = 10;

%% Time and Reference Signal
Ky       = 1;
t_hist   = (0:N)'   * Ts;
t_ref    = (0:N)'   * Ts;          
half     = floor((N+1) / 2);
seg      = floor(half / 4);
ref_step = [0.4*ones(seg,1); 0.8*ones(seg,1); 1.2*ones(seg,1); 0.8*ones(half-3*seg,1)];
ref_sin  = 0.3*sin(0.5*pi*t_ref(1:(N+1)-half)) + 0.8;
yref     = [ref_step; ref_sin];    % length = N+1

%% Initial Conditions
x1 = zeros(1, N+1);   x1(1) = 0;
x2 = zeros(1, N+1);   x2(1) = 0;
y  = zeros(1, N+1);   y(1)  = x1(1) + x2(1);
u  = zeros(1, N+1);

%% Main Loop
figure('units','normalized','outerposition',[0 0 1 1],'color','w');
for i = 1:N

    % --- NARMA-L2 Controller ---
    state = [x1(i); x2(i)];
    f_hat = NNforward(state, Wg_f, bh_f, Wc_f, bc_f);
    g_hat = NNforward(state, Wg_g, bh_g, Wc_g, bc_g);
    if abs(g_hat) < 1e-6; g_hat = sign(g_hat+eps)*1e-6; end

    u_raw = (yref(i+1) - f_hat) / g_hat;
    u(i)  = max(-u_max, min(u_max, u_raw));

    % --- System Update ---
    x1(i+1) = 0.1 - x1(i)^2 + x1(i)*x2(i);
    x2(i+1) = -x1(i) + exp(-x2(i)) + u(i);
    y(i+1)  = x1(i+1) + x2(i+1);

    % --- Plot ---
    if mod(i, plot_interval) == 0 || i == N
        clf

        % compute running metrics
        RMSE   = @(a,b) sqrt(mean((a-b).^2));
        rmse_i = RMSE(yref(1:i)', y(1:i));
        ss_i   = abs(yref(i) - y(i));

        subplot(221); hold on; grid on;
        plot(t_hist(1:N),   yref(1:N),  'r', 'LineWidth', 2);
        plot(t_hist(1:i+1), y(1:i+1),   'k', 'LineWidth', 2);
        xlabel('t [s]'); ylabel('y');
        if i == N
            title(sprintf('NARMA-L2  |  RMSE=%.5f  |  SS Error=%.5f', rmse_i, ss_i));
        else
            title(sprintf('NARMA-L2  |  t=%.1f s', t_hist(i)));
        end
        legend('y_{ref}','y[n]');
        axis([0 T min(yref(1:N))-0.3 max(yref(1:N))+0.3]);

        subplot(222); hold on; grid on;
        plot(t_hist(1:i+1), yref(1:i+1)' - y(1:i+1), 'b', 'LineWidth', 1);
        yline(0,'k--');
        xlabel('t [s]');
        title(sprintf('Tracking Error  |  Current=%.5f', ss_i));

        subplot(223); hold on; grid on;
        plot(t_hist(1:i+1), u(1:i+1), 'b', 'LineWidth', 1);
        yline( u_max,'r--','u_{max}');
        yline(-u_max,'r--','u_{min}');
        xlabel('t [s]'); ylabel('u'); title('Control Input');

        subplot(224); hold on; grid on;
        plot(t_hist(1:i+1), x1(1:i+1), 'k--', 'LineWidth', 1);
        plot(t_hist(1:i+1), x2(1:i+1), 'r--', 'LineWidth', 1);
        xlabel('t [s]'); ylabel('x_i'); title('States');
        legend('x_1','x_2');

        drawnow;
    end
end


%% -- Local Function

function yhat = NNforward(x_in, Wg, bh, Wc, bc)
    yhat = Wc * tanh(Wg * x_in(:) + bh) + bc;
end