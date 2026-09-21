clc; clear; close all;
% Paper Figures for Online NARMA-L2 Control with RLS-ELM
% Written By: Rasit Evduzen
% Date: 18-Sep-2026
%
% Reruns the loop of NarmaL2_ELM_Online.m without animation and saves the four
% paper figures separately in ./figs (vector PDF + PNG):
%   sim_tracking   reference and output
%   sim_error      tracking error
%   sim_control    control input against its saturation limits
%   sim_elm        ELM estimates of f and g against their true values

out_dir = fullfile(pwd, 'figs');
if ~isfolder(out_dir), mkdir(out_dir); end
fig_w = 3.4;   % [in] figure width (one column of the Wiley two-column layout)
fig_h = 2.3;   % [in] figure height
fs    = 8;     % [pt] axis font size at that width

%% Algorithm Parameters
Ts    = 1e-1;          % [s]
T     = 50;            % [s]
N     = round(T / Ts);
u_max = 3.0;

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
T_warm    = 5;     % [s] open-loop PRBS duration
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
g_true  = zeros(1, N);   f_true = zeros(1, N);
u_hold  = 0;

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
    g_true(i) = g_n;
    x1(i+1) = a_n - x1(i)^2 + x1(i)*x2(i);
    x2(i+1) = -x1(i) + exp(-x2(i)) + g_n*u(i) + d_n;
    y(i+1)  = x1(i+1) + x2(i+1);
    ym(i+1) = y(i+1) + sigma_n*randn;
    f_true(i) = y(i+1) - g_n*u(i);

    % --- RLS update ---
    phi = [h; h*u(i)];
    [theta, P, e_pred(i)] = RlsUpdate(phi, ym(i+1), theta, P, lambda, Peig_max);
    trP(i) = trace(P);
end

%% Performance Summary
plt = struct('T',T, 'T_warm',T_warm, 'u_max',u_max, 'g_min',g_min, ...
             't_a_step',t_a_step, 't_b_ramp',t_b_ramp, 't_pulse',t_pulse, 't_dstep',t_dstep);
ctrl  = find(t_hist >= T_warm);                    % closed-loop window (into t_hist)
ctrlN = ctrl(ctrl <= N);                           % same window into the length-N histories
err   = yref' - y;                                 % tracking error
rmse  = sqrt(mean((yref(ctrl)' - y(ctrl)).^2));    % closed-loop tracking RMSE
g_err = sqrt(mean((g_hist(ctrlN) - g_true(ctrlN)).^2));
f_err = sqrt(mean((f_hist(ctrlN) - f_true(ctrlN)).^2));
e_rms = sqrt(mean(e_pred(ctrlN).^2));               % RLS one-step residual, quoted in the text
fprintf('Closed-loop tracking RMSE    = %.4f\n', rmse);
fprintf('Closed-loop f estimate RMSE  = %.4f\n', f_err);
fprintf('Closed-loop g estimate RMSE  = %.4f\n', g_err);
fprintf('One-step prediction residual = %.4f\n', e_rms);
fprintf('Final trace(P)               = %.2e\n', trP(N));
fprintf('Final tracking error         = %.4f\n', abs(err(end)));
fprintf('Peak |u| after warm-up       = %.3f\n', max(abs(u(ctrlN))));

%% Colors
cRef  = [0.85 0.10 0.10];   cOut  = [0.10 0.10 0.10];
cCtrl = [0.00 0.45 0.74];   cErr  = [0.00 0.45 0.74];
cF    = [0.10 0.10 0.10];   cG    = [0.85 0.33 0.10];

%% Figure - Output Tracking
[fig, ax] = NewFig(fig_w, fig_h, fs);
yl = [min(yref)-0.5, max(yref)+0.5];
Bands(ax, plt, yl);
plot(ax, t_hist(1:N), yref(1:N), '-', 'Color', cRef, 'LineWidth', 1.2);
plot(ax, t_hist, y, '-', 'Color', cOut, 'LineWidth', 1.0);
ylim(ax, yl); xlabel(ax, 't [s]'); ylabel(ax, 'y');
title(ax, 'Output Tracking');
legend(ax, {'$y_{ref}$', '$y[n]$'}, 'Interpreter', 'latex', 'Location', 'northeast');
SaveFig(fig, 'sim_tracking', out_dir);

%% Figure - Tracking Error
[fig, ax] = NewFig(fig_w, fig_h, fs);
m = max(0.1, 1.1*max(abs(err(ctrl))));   yl = [-m m];   % scaled on closed-loop data only
Bands(ax, plt, yl);
plot(ax, t_hist, err, '-', 'Color', cErr, 'LineWidth', 0.8);
yline(ax, 0, '-', 'Color', [0.4 0.4 0.4]);
ylim(ax, yl); xlabel(ax, 't [s]'); ylabel(ax, 'y_{ref} - y');
title(ax, sprintf('Tracking Error,  RMSE_{ctrl} = %.4f', rmse));
SaveFig(fig, 'sim_error', out_dir);

%% Figure - Control Input
[fig, ax] = NewFig(fig_w, fig_h, fs);
yl = [-u_max-0.4, u_max+0.4];
Bands(ax, plt, yl);
plot(ax, t_hist(1:N), u(1:N), '-', 'Color', cCtrl, 'LineWidth', 0.8);
yline(ax,  u_max, '--', '$u_{max}$', 'Color', cRef, 'Interpreter', 'latex', 'LabelHorizontalAlignment', 'left');
yline(ax, -u_max, '--', '$u_{min}$', 'Color', cRef, 'Interpreter', 'latex', 'LabelHorizontalAlignment', 'left');
ylim(ax, yl); xlabel(ax, 't [s]'); ylabel(ax, 'u');
title(ax, 'Control Input');
SaveFig(fig, 'sim_control', out_dir);

%% Figure - ELM Estimates
[fig, ax] = NewFig(fig_w, fig_h, fs);
lo = min([0, f_hist, g_hist, f_true]) - 0.15;
hi = max([g_hist, g_true, f_hist, f_true]) + 0.15;
yl = [lo hi];
Bands(ax, plt, yl);
plot(ax, t_hist(1:N), f_hist, '-',  'Color', cF, 'LineWidth', 1.0);
plot(ax, t_hist(1:N), f_true, '--', 'Color', cF, 'LineWidth', 0.8);
plot(ax, t_hist(1:N), g_hist, '-',  'Color', cG, 'LineWidth', 1.0);
plot(ax, t_hist(1:N), g_true, '--', 'Color', cG, 'LineWidth', 0.8);
yline(ax, g_min, ':', '$g_{min}$', 'Color', [0.6 0.2 0.6], 'Interpreter', 'latex', 'LabelHorizontalAlignment', 'left');
ylim(ax, yl); xlabel(ax, 't [s]');
title(ax, 'ELM Estimates');
legend(ax, {'$\hat f(z)$', '$f_{true}$', '$\hat g(z)$', '$g_{true}$'}, 'Interpreter', 'latex', 'Location', 'northeast');
SaveFig(fig, 'sim_elm', out_dir);

fprintf('Figures written to %s\n', out_dir);


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

function [fig, ax] = NewFig(w, h, fs)
    fig = figure('Units','inches', 'Position',[1 1 w h], 'Color','w');
    ax  = axes(fig);
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    ax.GridAlpha = 0.15;  ax.FontSize = fs;    ax.LineWidth = 0.7;
    ax.TitleFontSizeMultiplier = 1.1;
end

function SaveFig(fig, name, out_dir)
    exportgraphics(fig, fullfile(out_dir, [name '.pdf']), 'ContentType', 'vector');   % for LaTeX
    exportgraphics(fig, fullfile(out_dir, [name '.eps']), 'ContentType', 'vector');   % for submission
    exportgraphics(fig, fullfile(out_dir, [name '.png']), 'Resolution', 300);         % for quick viewing
end

function Bands(ax, p, yl)
% Warm-up band, gain-ramp band and the plant events. The event labels are set
% along the lines rather than across them, so that five of them fit at one
% column width.
    xlim(ax, [0 p.T]);
    cWarm = [0.30 0.70 0.35];   cEvt = [0.50 0.50 0.50];
    Shade = @(x0, x1, col, al) patch(ax, [x0 x1 x1 x0], [yl(1) yl(1) yl(2) yl(2)], col, ...
                                     'FaceAlpha', al, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    Mark  = @(x, sty, txt, col) xline(ax, x, sty, txt, 'Color', col, 'LineWidth', 1.0, ...
                                      'FontSize', 6, 'LabelVerticalAlignment', 'bottom', ...
                                      'HandleVisibility', 'off');
    if p.T_warm > 0
        Shade(0, p.T_warm, cWarm, 0.10);
        Mark(p.T_warm, '--', 'closed loop', cWarm);
    end
    Shade(p.t_b_ramp(1), p.t_b_ramp(2), cEvt, 0.08);
    Mark(p.t_a_step,     ':', 'a step',    cEvt);
    Mark(p.t_pulse,      ':', 'impulse',   cEvt);
    Mark(p.t_b_ramp(1),  ':', 'b ramp',    cEvt);
    Mark(p.t_dstep,      ':', 'load step', cEvt);
end
