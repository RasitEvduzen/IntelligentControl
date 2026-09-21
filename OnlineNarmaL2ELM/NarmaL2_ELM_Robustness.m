clc; clear; close all;
% Robustness Analysis - Online NARMA-L2 Control with RLS-ELM
% Written By: Rasit Evduzen
% Date: 17-Sep-2026
%
% (1) Empirical frequency response of the adaptive loop:
%       |S| = e_amp / d_amp   for a sinusoidal load disturbance, constant reference
%       |T| = y_amp / r_amp   for a sinusoidal reference, no disturbance
% (2) Plant parameter map (a, b): constant plant, nominal controller
% (3) Delay margin: k samples of input dead time not known to the controller
%
% Produces two paper figures, saved separately in ./figs:
%   rob_frequency    empirical |S| and |T| of the adaptive loop
%   rob_plant_map    median RMSE over the plant parameters a and b
% The delay-margin result is printed to the console; it is a single number, not a figure.

%% Nominal Controller and Scenario (as in NarmaL2_ELM_Online.m)
nom = struct( ...
    'Ts',0.1, 'T',50, 'u_max',3.0, ...
    'Nh',20, 'ny',2, 'nu',1, 'w_scale',0.3, 'lambda',0.95, 'P0',1e4, ...
    'Peig_max',1e2, 'g_min',0.4, 'g_max',3.0, 'g0',1.0, ...
    'T_warm',5, 'prbs_hold',5, 'u_prbs',0.5, ...
    'tv',true, 'a',0.1, 'b',1.0, ...                 % tv = nominal time variation of a and b
    'ref','scenario', 'y0',0.8, 'r_amp',0.3, 'r_freq',0.25, ...
    'dist','scenario', 'd_scale',1.0, 'd_amp',0.1, 'd_freq',0.25, ...
    'sigma_n',0.01, 'delay',0, 'seed',1);

%% (1) Empirical Frequency Response
freqs = logspace(log10(0.05), log10(4), 10);      % [Hz], Nyquist = 5 Hz
S_mag = zeros(size(freqs));  T_mag = zeros(size(freqs));
for k = 1:numel(freqs)
    o = nom;  o.T = 60;  o.tv = false;  o.sigma_n = 0;
    o.ref = 'const';  o.dist = 'sine';  o.d_freq = freqs(k);
    r = RunSim(o);
    S_mag(k) = SineAmp(r.t, r.yref - r.y, freqs(k), 20) / o.d_amp;
    o.ref = 'sine';  o.r_freq = freqs(k);  o.dist = 'none';
    r = RunSim(o);
    T_mag(k) = SineAmp(r.t, r.y - o.y0, freqs(k), 20) / o.r_amp;
end

%% (2) Plant Parameter Map
a_grid = [0 0.1 0.2 0.3 0.4 0.5];
b_grid = [0.3 0.5 0.75 1 1.5 2];
seeds_ab = 1:3;
AB_rmse = zeros(numel(a_grid), numel(b_grid));  AB_div = AB_rmse;
for i = 1:numel(a_grid)
    for j = 1:numel(b_grid)
        o = nom;  o.tv = false;  o.a = a_grid(i);  o.b = b_grid(j);  o.dist = 'none';
        o.u_prbs = 0.5/o.b;  % excitation scaled to the plant gain so the warm-up stays inside the basin
        M = RunSeeds(o, seeds_ab);
        AB_div(i,j) = 100*mean(M.div);  AB_rmse(i,j) = median(M.rmse(~M.div));
    end
end
a_zd = (4 - (1 - 0.4)^2)/8;      % zero-dynamics limit for the lowest reference level 0.4

%% (3) Delay Margin
delays = 0:3;
seeds_d = 1:10;
D_rmse = zeros(size(delays));  D_p90 = D_rmse;  D_div = D_rmse;
for k = 1:numel(delays)
    o = nom;  o.delay = delays(k);
    M = RunSeeds(o, seeds_d);
    D_div(k) = 100*mean(M.div);  D_rmse(k) = median(M.rmse(~M.div));  D_p90(k) = Pct(M.rmse(~M.div), 90);
end

%% Paper Figures
out_dir = fullfile(pwd, 'figs');
if ~isfolder(out_dir), mkdir(out_dir); end

cS = [0.00 0.45 0.74];  cT = [0.85 0.33 0.10];  cDiv = [0.85 0.10 0.10];  cLine = [0.10 0.10 0.10];

% (1) frequency response
[fig, ax] = NewFig(3.4, 2.4, 8);  set(ax, 'XScale', 'log');
plot(ax, freqs, 20*log10(S_mag), '-o', 'Color', cS, 'LineWidth', 1.2, 'MarkerFaceColor', cS, 'MarkerSize', 3.5);
plot(ax, freqs, 20*log10(T_mag), '-s', 'Color', cT, 'LineWidth', 1.2, 'MarkerFaceColor', cT, 'MarkerSize', 3.5);
yline(ax, 0, '-', 'Color', [0.4 0.4 0.4]);
yline(ax, -3, ':', '-3 dB', 'Color', [0.4 0.4 0.4], 'LabelHorizontalAlignment', 'left');
xlabel(ax, 'frequency [Hz]');  ylabel(ax, 'magnitude [dB]');
title(ax, 'Empirical Frequency Response');
legend(ax, {'disturbance to error,  |S|', 'reference to output,  |T|'}, 'Location', 'southeast');
xlim(ax, [freqs(1)*0.8, 5]);
SaveFig(fig, 'rob_frequency', out_dir);

% (2) plant parameter map, drawn one column wide so that it floats with the text
% instead of having to wait for the top of a page
[fig, ax] = NewFig(3.4, 2.6, 7);
HeatMap(ax, b_grid, a_grid, AB_rmse, AB_div, 'input gain b', 'plant parameter a');
yline(ax, interp1(a_grid, 1:numel(a_grid), a_zd, 'linear', 'extrap'), '--', ...
      'Color', cLine, 'LineWidth', 1.0);       % the caption says what the line marks
title(ax, 'Plant Parameter Map');
SaveFig(fig, 'rob_plant_map', out_dir);

% (3) delay margin is a one-line result, reported in the text rather than as a figure
fprintf('\ndelay [samples]   div [%%]   RMSE med   RMSE p90\n');
for k = 1:numel(delays)
    fprintf('%10d %11.0f %10.4f %10.4f\n', delays(k), D_div(k), D_rmse(k), D_p90(k));
end

fprintf('\nFigures written to %s\n', out_dir);


%% -- Local Functions

function M = RunSeeds(o, seeds)
    n = numel(seeds);
    M = struct('div',false(1,n), 'rmse',nan(1,n));
    for s = 1:n
        o.seed = seeds(s);
        r = RunSim(o);
        M.div(s) = r.div;
        if ~r.div
            ctrl = r.t >= o.T_warm;
            M.rmse(s) = sqrt(mean((r.yref(ctrl) - r.y(ctrl)).^2));
        end
    end
end

function r = RunSim(o)
% Closed-loop simulation of NarmaL2_ELM_Online.m with configurable plant, reference,
% disturbance and input dead time. Returns time, output, reference, input, flag.
    N = round(o.T / o.Ts);
    t_hist = (0:N)' * o.Ts;
    if o.tv
        a_fun = @(t) 0.1 + 0.2*(t >= 15);
        b_fun = @(t) 1.0 - 0.4*min(max((t - 30)/5, 0), 1);
    else
        a_fun = @(t) o.a;  b_fun = @(t) o.b;
    end
    g_fun = @(x2, t) b_fun(t) * (1 + 0.4*cos(x2));
    switch o.dist
        case 'scenario', d_fun = @(t) o.d_scale * (0.3*(t >= 20 & t < 20 + 3*o.Ts) + 0.15*(t >= 40));
        case 'sine',     d_fun = @(t) o.d_amp * sin(2*pi*o.d_freq*t) .* (t >= o.T_warm);
        otherwise,       d_fun = @(t) 0;
    end
    switch o.ref
        case 'scenario'
            half = floor((N+1)/2);  seg = floor(half/4);
            yref = [0.4*ones(seg,1); 0.8*ones(seg,1); 1.2*ones(seg,1); 0.8*ones(half-3*seg,1); ...
                    0.3*sin(0.5*pi*t_hist(1:(N+1)-half)) + 0.8];
        case 'const', yref = o.y0 * ones(N+1, 1);
        case 'sine',  yref = o.y0 + o.r_amp * sin(2*pi*o.r_freq*t_hist) .* (t_hist >= o.T_warm);
    end

    rng(7);
    W = o.w_scale*randn(o.Nh, o.ny+o.nu);  bh = o.w_scale*randn(o.Nh, 1);  nH = o.Nh + 1;
    theta = zeros(2*nH, 1);  theta(nH+1) = o.g0;  P = o.P0*eye(2*nH);
    rng(o.seed);

    x1 = zeros(1,N+1);  x2 = zeros(1,N+1);  y = zeros(1,N+1);  ym = zeros(1,N+1);  u = zeros(1,N+1);
    ym(1) = o.sigma_n*randn;  u_hold = 0;  div = false;

    for i = 1:N
        t_now = t_hist(i);
        z = zeros(o.ny+o.nu, 1);
        for k = 1:o.ny, z(k) = ym(max(i-k+1, 1)); end
        for k = 1:o.nu, if i-k >= 1, z(o.ny+k) = u(i-k); end, end
        h = [1; tanh(W*z + bh)];
        f_hat = theta(1:nH)'*h;  g_hat = min(max(theta(nH+1:end)'*h, o.g_min), o.g_max);
        if t_now < o.T_warm
            if mod(i-1, o.prbs_hold) == 0, u_hold = o.u_prbs*(2*rand-1); end
            u_raw = u_hold;
        else
            u_raw = (yref(i+1) - f_hat)/g_hat;
        end
        u(i) = max(-o.u_max, min(o.u_max, u_raw));
        u_app = 0;  if i - o.delay >= 1, u_app = u(i - o.delay); end     % delayed input reaches the plant

        g_n = g_fun(x2(i), t_now);
        x1(i+1) = a_fun(t_now) - x1(i)^2 + x1(i)*x2(i);
        x2(i+1) = -x1(i) + exp(-x2(i)) + g_n*u_app + d_fun(t_now);
        y(i+1)  = x1(i+1) + x2(i+1);
        ym(i+1) = y(i+1) + o.sigma_n*randn;
        if ~isfinite(y(i+1)) || abs(y(i+1)) > 1e3, div = true; y(i+1:end) = NaN; break; end

        phi = [h; h*u(i)];
        e = ym(i+1) - phi'*theta;
        K = (P*phi)/(o.lambda + phi'*P*phi);
        theta = theta + K*e;
        P = (P - K*(phi'*P))/o.lambda;  P = (P + P')/2;
        [V, D] = eig(P);
        if max(diag(D)) > o.Peig_max, D = min(D, o.Peig_max); P = V*D*V'; P = (P + P')/2; end
    end
    r.t = t_hist(1:N);  r.y = y(1:N)';  r.yref = yref(1:N);  r.u = u(1:N)';  r.div = div;
end

function A = SineAmp(t, x, f, T_win)
% Amplitude of the component at frequency f, least-squares fit over the last T_win seconds
    w = t >= t(end) - T_win;
    B = [sin(2*pi*f*t(w)), cos(2*pi*f*t(w)), ones(nnz(w),1)];
    c = B \ x(w);
    A = hypot(c(1), c(2));
end

function HeatMap(ax, xv, yv, Z, Zdiv, xlab, ylab)
    im = imagesc(ax, 1:numel(xv), 1:numel(yv), Z);  im.AlphaData = ~isnan(Z);
    set(ax, 'YDir', 'normal', 'XTick', 1:numel(xv), 'XTickLabel', compose('%.3g', xv), ...
            'YTick', 1:numel(yv), 'YTickLabel', compose('%.3g', yv));
    colormap(ax, flipud(parula));  cb = colorbar(ax);  cb.Label.String = 'median RMSE';
    for a = 1:numel(yv)
        for b = 1:numel(xv)
            if isnan(Z(a,b))
                txt = 'all diverged';  col = [0.85 0.10 0.10];
            elseif Zdiv(a,b) > 0
                txt = sprintf('%.3f\n%.0f%%', Z(a,b), Zdiv(a,b));  col = [0.85 0.10 0.10];
            else
                txt = sprintf('%.3f', Z(a,b));  col = [0 0 0];
            end
            text(ax, b, a, txt, 'HorizontalAlignment', 'center', 'FontSize', 5.5, 'Color', col, 'FontWeight', 'bold');
        end
    end
    xlim(ax, [0.5, numel(xv)+0.5]);  ylim(ax, [0.5, numel(yv)+0.5]);   % no white margin round the cells
    xlabel(ax, xlab);  ylabel(ax, ylab);  grid(ax, 'off');
end

function q = Pct(x, p)
    x = sort(x(:));  n = numel(x);
    if n == 0, q = NaN; return; end
    pos = 1 + (n-1)*p/100;  lo = floor(pos);  hi = ceil(pos);
    q = x(lo) + (pos-lo)*(x(hi)-x(lo));
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
