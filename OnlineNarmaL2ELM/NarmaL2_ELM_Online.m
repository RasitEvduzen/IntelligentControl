clc; clear; close all;
% Online NARMA-L2 Control with RLS-ELM System Identification
% Written By: Rasit Evduzen
% Date: 17-Sep-2026
%
% Plant (time-varying, states not measured):
%   x1[n+1] = a(n) - x1^2 + x1*x2
%   x2[n+1] = -x1 + exp(-x2) + b(n)*(1 + 0.4*cos(x2))*u[n] + d[n]
%   y[n]    = x1 + x2,   y_m[n] = y[n] + v[n]      (d: load disturbance, v: noise)
%
% NARMA-L2 model:   y[n+1] = f(z) + g(z)*u[n],   z = [y[n]; y[n-1]; u[n-1]]
% ELM:              h(z) = [1; tanh(W*z + bh)],  f = beta_f'*h,  g = beta_g'*h
%                   y[n+1] = theta'*phi,  phi = [h; h*u[n]],  theta = [beta_f; beta_g]
% RLS:              single recursive LS on theta (linear), forgetting factor lambda
% Controller:       u[n] = (yref[n+1] - f_hat) / g_hat      (uses y_m only)
% Phases:           t < T_warm open-loop PRBS, then closed loop (RLS runs throughout)

%% Algorithm Parameters
Ts    = 1e-1;          % [s]
T     = 50;            % [s]
N     = round(T / Ts);
u_max = 3.0;
plot_interval = 20;

%% ELM / RLS Parameters
Nh       = 20;     % hidden neurons
ny       = 2;      % past outputs in z
nu       = 1;      % past inputs in z
w_scale  = 0.3;    % std of random input weights
lambda   = 0.95;   % forgetting factor
P0       = 1e4;    % initial covariance
Peig_max = 1e2;    % eigenvalue bound on P (anti-windup)
g_min    = 0.4;    % bounds on |g_hat| in control law
g_max    = 3.0;
g0       = 1.0;    % initial g_hat

%% Warm-up Parameters
T_warm    = 5;     % [s] open-loop PRBS duration (0 = closed loop from start)
prbs_hold = 5;     % [samples]
u_prbs    = 0.5;

%% Time-Varying Plant Parameters
t_a_step = 15;             % [s] a: 0.1 -> 0.3 step
t_b_ramp = [30 35];        % [s] b: 1.0 -> 0.6 ramp
a_fun = @(t) 0.1 + 0.2*(t >= t_a_step);
b_fun = @(t) 1.0 - 0.4*min(max((t - t_b_ramp(1))/diff(t_b_ramp), 0), 1);
g_fun = @(x2, t) b_fun(t) * (1 + 0.4*cos(x2));

%% Disturbance and Noise
sigma_n   = 0.01;          % measurement noise std
t_pulse   = 20;            % [s] impulse disturbance start
pulse_len = 3;             % [samples]
d_pulse   = 0.3;
t_dstep   = 40;            % [s] persistent load disturbance start
d_step    = 0.15;
d_fun = @(t) d_pulse*(t >= t_pulse & t < t_pulse + pulse_len*Ts) + d_step*(t >= t_dstep);

%% Time and Reference Signal
t_hist   = (0:N)' * Ts;
half     = floor((N+1) / 2);
seg      = floor(half / 4);
ref_step = [0.4*ones(seg,1); 0.8*ones(seg,1); 1.2*ones(seg,1); 0.8*ones(half-3*seg,1)];
ref_sin  = 0.3*sin(0.5*pi*t_hist(1:(N+1)-half)) + 0.8;
yref     = [ref_step; ref_sin];

%% ELM Initialization
rng(7);
W  = w_scale * randn(Nh, ny+nu);   % fixed random input weights
bh = w_scale * randn(Nh, 1);
nH = Nh + 1;                       % hidden size incl. bias node

theta = zeros(2*nH, 1);            % [beta_f; beta_g]
theta(nH+1) = g0;
P = P0 * eye(2*nH);

%% Initial Conditions and Histories
x1 = zeros(1, N+1);   x2 = zeros(1, N+1);
y  = zeros(1, N+1);   y(1) = x1(1) + x2(1);
ym = zeros(1, N+1);   ym(1) = y(1) + sigma_n*randn;
u  = zeros(1, N+1);

f_hist  = zeros(1, N);   g_hist = zeros(1, N);
e_pred  = zeros(1, N);   trP    = zeros(1, N);
g_true  = zeros(1, N);   f_true = zeros(1, N);   d_hist = zeros(1, N);
u_hold  = 0;

%% Plot Setup
plt = struct('T',T, 'N',N, 'T_warm',T_warm, 'u_max',u_max, 'g_min',g_min, ...
             'Nh',Nh, 'lambda',lambda, 'Peig_max',Peig_max, 'Ts',Ts, 'sigma_n',sigma_n, 'g_max',g_max, ...
             't_a_step',t_a_step, 't_b_ramp',t_b_ramp, 't_pulse',t_pulse, 't_dstep',t_dstep);
fig = figure('units','normalized','outerposition',[0 0 1 1],'color','w', ...
             'Name','Online NARMA-L2 with RLS-ELM','NumberTitle','off');

%% Main Loop
for i = 1:N
    t_now = t_hist(i);

    % --- ELM forward ---
    z = Regressor(ym, u, i, ny, nu);
    h = HiddenLayer(z, W, bh);
    f_hat = theta(1:nH)'     * h;
    g_hat = theta(nH+1:end)' * h;
    f_hist(i) = f_hat;   g_hist(i) = g_hat;
    g_hat = min(max(g_hat, g_min), g_max);        % gain sign known positive

    % --- Control law ---
    if t_now < T_warm
        if mod(i-1, prbs_hold) == 0
            u_hold = u_prbs * (2*rand - 1);
        end
        u_raw = u_hold;
    else
        u_raw = (yref(i+1) - f_hat) / g_hat;
    end
    u(i) = max(-u_max, min(u_max, u_raw));

    % --- Plant ---
    a_n = a_fun(t_now);   g_n = g_fun(x2(i), t_now);   d_n = d_fun(t_now);
    g_true(i) = g_n;      d_hist(i) = d_n;
    x1(i+1) = a_n - x1(i)^2 + x1(i)*x2(i);
    x2(i+1) = -x1(i) + exp(-x2(i)) + g_n*u(i) + d_n;
    y(i+1)  = x1(i+1) + x2(i+1);
    ym(i+1) = y(i+1) + sigma_n*randn;
    f_true(i) = y(i+1) - g_n*u(i);

    % --- RLS update ---
    phi = [h; h*u(i)];
    [theta, P, e_pred(i)] = RlsUpdate(phi, ym(i+1), theta, P, lambda, Peig_max);
    trP(i) = trace(P);

    % --- Live plot ---
    if mod(i, plot_interval) == 0 || i == N
        PlotLive(fig, i, t_hist, yref, y, u, x1, x2, d_hist, f_hist, g_hist, f_true, g_true, e_pred, trP, plt);
    end
end


%% -- Local Functions

function z = Regressor(y, u, i, ny, nu)
% z = [y[i]; ...; y[i-ny+1]; u[i-1]; ...; u[i-nu]]
    z = zeros(ny+nu, 1);
    for k = 1:ny
        z(k) = y(max(i-k+1, 1));
    end
    for k = 1:nu
        idx = i - k;
        if idx >= 1
            z(ny+k) = u(idx);
        end
    end
end

function h = HiddenLayer(z, W, bh)
    h = [1; tanh(W*z(:) + bh)];
end

function [theta, P, e] = RlsUpdate(phi, yk, theta, P, lambda, Peig_max)
% RLS with forgetting factor and eigenvalue-bounded covariance
    e     = yk - phi'*theta;                    % a priori error
    K     = (P*phi) / (lambda + phi'*P*phi);
    theta = theta + K*e;
    P     = (P - K*(phi'*P)) / lambda;
    P     = (P + P')/2;
    [V, D] = eig(P);
    if max(diag(D)) > Peig_max                  % anti-windup
        D = min(D, Peig_max);
        P = V*D*V';  P = (P + P')/2;
    end
end

function PlotLive(fig, i, t, yref, y, u, x1, x2, d_hist, f_hist, g_hist, f_true, g_true, e_pred, trP, p)
    cRef  = [0.85 0.10 0.10];   cOut  = [0.10 0.10 0.10];
    cCtrl = [0.00 0.45 0.74];   cErr  = [0.00 0.45 0.74];
    cF    = [0.10 0.10 0.10];   cG    = [0.85 0.33 0.10];
    cX1   = [0.30 0.30 0.30];   cX2   = [0.85 0.33 0.10];

    N = p.N;   ti = t(1:i);   ti1 = t(1:i+1);
    RMSE   = @(a,b) sqrt(mean((a-b).^2));
    ctrl_i = find(t(1:i) >= p.T_warm);
    if isempty(ctrl_i), rmse_i = NaN; else, rmse_i = RMSE(yref(ctrl_i)', y(ctrl_i)); end
    err_i  = yref(1:i+1)' - y(1:i+1);
    sc = ctrl_i;  if isempty(sc), sc = 1:i; end     % axis scaling window

    clf(fig);
    tl = tiledlayout(fig, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    if i == N
        hdr = sprintf('Online NARMA-L2 with RLS-ELM,  RMSE_{ctrl} = %.4f,  final error = %.4f', ...
                      rmse_i, abs(err_i(end)));
    else
        hdr = sprintf('Online NARMA-L2 with RLS-ELM,  t = %.1f s', t(i));
    end
    title(tl, hdr, 'FontSize', 15, 'FontWeight', 'bold');
    subtitle(tl, sprintf('N_h = %d,  \\lambda = %.3f,  eig(P) \\leq %.0e,  T_s = %.2f s,  \\sigma_n = %.3f', ...
                         p.Nh, p.lambda, p.Peig_max, p.Ts, p.sigma_n), 'FontSize', 11);

    % (1) tracking
    ax = nexttile(1); StyleAxes(ax);
    yl = [min(yref)-0.5, max(yref)+0.5];
    Bands(ax, p, yl);
    plot(ax, t(1:N), yref(1:N), '-', 'Color', cRef, 'LineWidth', 2);
    plot(ax, ti1, y(1:i+1), '-', 'Color', cOut, 'LineWidth', 1.6);
    ylim(ax, yl); xlabel(ax, 't [s]'); ylabel(ax, 'y');
    title(ax, 'Output Tracking');
    legend(ax, {'$y_{ref}$', '$y[n]$'}, 'Interpreter', 'latex', 'Location', 'northeast');

    % (2) tracking error
    ax = nexttile(2); StyleAxes(ax);
    m = max(0.1, 1.1*max(abs(err_i(sc))));   yl = [-m m];   % scaled on closed-loop data only
    Bands(ax, p, yl);
    plot(ax, ti1, err_i, '-', 'Color', cErr, 'LineWidth', 1.2);
    yline(ax, 0, '-', 'Color', [0.4 0.4 0.4]);
    ylim(ax, yl); xlabel(ax, 't [s]'); ylabel(ax, 'y_{ref} - y');
    title(ax, sprintf('Tracking Error,  current = %.4f', abs(err_i(end))));

    % (3) control input
    ax = nexttile(3); StyleAxes(ax);
    yl = [-p.u_max-0.4, p.u_max+0.4];
    Bands(ax, p, yl);
    plot(ax, ti, u(1:i), '-', 'Color', cCtrl, 'LineWidth', 1.2);
    yline(ax,  p.u_max, '--', '$u_{max}$', 'Color', cRef, 'Interpreter', 'latex', 'LabelHorizontalAlignment', 'left');
    yline(ax, -p.u_max, '--', '$u_{min}$', 'Color', cRef, 'Interpreter', 'latex', 'LabelHorizontalAlignment', 'left');
    ylim(ax, yl); xlabel(ax, 't [s]'); ylabel(ax, 'u');
    title(ax, 'Control Input');

    % (4) ELM estimates
    ax = nexttile(4); StyleAxes(ax);
    lo = min([0, f_hist(1:i), g_hist(1:i), f_true(1:i)]) - 0.15;
    hi = max([g_hist(1:i), g_true(1:i), f_hist(1:i), f_true(1:i)]) + 0.15;
    yl = [lo hi];
    Bands(ax, p, yl);
    plot(ax, ti, f_hist(1:i), '-',  'Color', cF, 'LineWidth', 1.5);
    plot(ax, ti, f_true(1:i), '--', 'Color', cF, 'LineWidth', 1.2);
    plot(ax, ti, g_hist(1:i), '-',  'Color', cG, 'LineWidth', 1.5);
    plot(ax, ti, g_true(1:i), '--', 'Color', cG, 'LineWidth', 1.2);
    yline(ax, p.g_min, ':', '$g_{min}$', 'Color', [0.6 0.2 0.6], 'Interpreter', 'latex', 'LabelHorizontalAlignment', 'left');
    ylim(ax, yl); xlabel(ax, 't [s]');
    title(ax, 'ELM Estimates');
    legend(ax, {'$\hat f(z)$', '$f_{true}$', '$\hat g(z)$', '$g_{true}$'}, 'Interpreter', 'latex', 'Location', 'northeast');

    % (5) one-step prediction error
    ax = nexttile(5); StyleAxes(ax);
    m = max(0.05, 1.1*max(abs(e_pred(sc))));   yl = [-m m];   % scaled on closed-loop data only
    Bands(ax, p, yl);
    plot(ax, ti, e_pred(1:i), '-', 'Color', cErr, 'LineWidth', 1.2);
    yline(ax, 0, '-', 'Color', [0.4 0.4 0.4]);
    ylim(ax, yl); xlabel(ax, 't [s]'); ylabel(ax, 'e_{pred}');
    title(ax, sprintf('One-Step Prediction Error,  trace(P) = %.2e', trP(i)));

    % (6) states
    ax = nexttile(6); StyleAxes(ax);
    lo = min([x1(1:i+1), x2(1:i+1), d_hist(1:i)]) - 0.1;   hi = max([x1(1:i+1), x2(1:i+1), d_hist(1:i)]) + 0.1;
    yl = [lo hi];
    Bands(ax, p, yl);
    plot(ax, ti1, x1(1:i+1), '-', 'Color', cX1, 'LineWidth', 1.2);
    plot(ax, ti1, x2(1:i+1), '-', 'Color', cX2, 'LineWidth', 1.2);
    plot(ax, ti,  d_hist(1:i), '--', 'Color', cCtrl, 'LineWidth', 1.2);
    ylim(ax, yl); xlabel(ax, 't [s]'); ylabel(ax, 'x_i,  d');
    title(ax, 'Plant States and Disturbance');
    legend(ax, {'$x_1$', '$x_2$', '$d[n]$'}, 'Interpreter', 'latex', 'Location', 'northeast');

    drawnow;
end

function StyleAxes(ax)
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    ax.GridAlpha = 0.15;  ax.FontSize = 10.5;  ax.LineWidth = 0.8;
    ax.TitleFontSizeMultiplier = 1.1;
end

function Bands(ax, p, yl)
% warm-up region and plant-change markers
    xlim(ax, [0 p.T]);
    cWarm = [0.30 0.70 0.35];   cEvt = [0.50 0.50 0.50];
    if p.T_warm > 0
        patch(ax, [0 p.T_warm p.T_warm 0], [yl(1) yl(1) yl(2) yl(2)], cWarm, ...
              'FaceAlpha', 0.10, 'EdgeColor', 'none', 'HandleVisibility', 'off');
        xline(ax, p.T_warm, '--', 'closed loop', 'Color', cWarm, 'LineWidth', 1.2, ...
              'LabelVerticalAlignment', 'top', 'LabelOrientation', 'horizontal', ...
              'FontSize', 8.5, 'HandleVisibility', 'off');
    end
    patch(ax, [p.t_b_ramp(1) p.t_b_ramp(2) p.t_b_ramp(2) p.t_b_ramp(1)], [yl(1) yl(1) yl(2) yl(2)], cEvt, ...
          'FaceAlpha', 0.08, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    xline(ax, p.t_a_step, ':', 'a step', 'Color', cEvt, 'LineWidth', 1.2, ...
          'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal', ...
          'FontSize', 8.5, 'HandleVisibility', 'off');
    xline(ax, p.t_b_ramp(1), ':', 'b ramp', 'Color', cEvt, 'LineWidth', 1.2, ...
          'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal', ...
          'FontSize', 8.5, 'HandleVisibility', 'off');
    xline(ax, p.t_pulse, ':', 'impulse', 'Color', cEvt, 'LineWidth', 1.2, ...
          'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal', ...
          'FontSize', 8.5, 'HandleVisibility', 'off');
    xline(ax, p.t_dstep, ':', 'load step', 'Color', cEvt, 'LineWidth', 1.2, ...
          'LabelVerticalAlignment', 'bottom', 'LabelOrientation', 'horizontal', ...
          'FontSize', 8.5, 'HandleVisibility', 'off');
end
