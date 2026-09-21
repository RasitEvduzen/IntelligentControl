clc; clear; close all;
% Sensitivity Analysis - Online NARMA-L2 Control with RLS-ELM
% Written By: Rasit Evduzen
% Date: 17-Sep-2026
%
% Re-runs the closed-loop simulation of NarmaL2_ELM_Online.m (plot-free copy in
% RunSim) over the controller design parameters, 10 noise seeds per point.
% Produces two paper figures, saved separately in ./figs:
%   sens_tornado     change of the median closed-loop RMSE over each parameter range
%   sens_lambda_noise  median RMSE over lambda and measurement noise
% Noise and disturbance robustness are covered in NarmaL2_ELM_Robustness.m.

%% Nominal Parameters (must match NarmaL2_ELM_Online.m)
nom = struct( ...
    'Ts',0.1, 'T',50, 'u_max',3.0, ...
    'Nh',20, 'ny',2, 'nu',1, 'w_scale',0.3, 'lambda',0.95, 'P0',1e4, ...
    'Peig_max',1e2, 'g_min',0.4, 'g_max',3.0, 'g0',1.0, ...
    'T_warm',5, 'prbs_hold',5, 'u_prbs',0.5, ...
    't_a_step',15, 't_b_ramp',[30 35], ...
    'sigma_n',0.01, 't_pulse',20, 'pulse_len',3, 'd_pulse',0.3, 't_dstep',40, 'd_step',0.15, ...
    'd_scale',1.0, 'seed',1);

seeds = 1:10;

%% One-Dimensional Sweeps
sw = struct('name',{}, 'label',{}, 'values',{}, 'logx',{});
sw(end+1) = struct('name','lambda',   'label','\lambda',       'values',[0.85 0.90 0.93 0.95 0.97 0.98 0.99], 'logx',false);
sw(end+1) = struct('name','Nh',       'label','N_h',           'values',[5 10 15 20 30 40],                   'logx',false);
sw(end+1) = struct('name','w_scale',  'label','w_{scale}',     'values',[0.1 0.2 0.3 0.5 0.8],                'logx',false);
sw(end+1) = struct('name','Peig_max', 'label','eig(P)_{max}',  'values',[1e1 1e2 1e3 1e4],                    'logx',true);

M0 = RunSeeds(nom, seeds);  rmse_nom = median(M0.rmse(~M0.div));
fprintf('nominal median RMSE %.4f\n', rmse_nom);
fprintf('%-18s %8s %8s %8s %8s %8s %8s\n', 'sweep', 'value', 'div [%]', 'RMSE med', 'RMSE p90', 'pkLoad', 'tRec [s]');
for k = 1:numel(sw)
    v = sw(k).values;  n = numel(v);
    R = struct('div',zeros(1,n), 'rmse_med',zeros(1,n), 'rmse_p10',zeros(1,n), 'rmse_p90',zeros(1,n), ...
               'pk_load',zeros(1,n), 't_rec',zeros(1,n), 'g_err',zeros(1,n));
    for j = 1:n
        o = nom;  o.(sw(k).name) = v(j);
        M = RunSeeds(o, seeds);
        R.div(j)      = 100*mean(M.div);
        ok            = ~M.div;
        R.rmse_med(j) = median(M.rmse(ok));
        R.rmse_p10(j) = Pct(M.rmse(ok), 10);
        R.rmse_p90(j) = Pct(M.rmse(ok), 90);
        R.pk_load(j)  = median(M.pk_load(ok));
        R.t_rec(j)    = median(M.t_rec(ok));
        R.g_err(j)    = median(M.g_err(ok));
        fprintf('%-18s %8.3g %8.0f %8.4f %8.4f %8.3f %8.1f\n', sw(k).name, v(j), R.div(j), R.rmse_med(j), R.rmse_p90(j), R.pk_load(j), R.t_rec(j));
    end
    sw(k).R = R;
end

%% Two-Dimensional Grid
lam_grid = [0.85 0.90 0.93 0.95 0.97 0.99];
sig_grid = [0 0.01 0.02 0.05 0.1];

G2_div = zeros(numel(sig_grid), numel(lam_grid));   G2_rmse = G2_div;
for a = 1:numel(sig_grid)
    for b = 1:numel(lam_grid)
        o = nom;  o.sigma_n = sig_grid(a);  o.lambda = lam_grid(b);
        M = RunSeeds(o, seeds);
        G2_div(a,b)  = 100*mean(M.div);
        G2_rmse(a,b) = median(M.rmse(~M.div));
    end
end

%% Paper Figures
out_dir = fullfile(pwd, 'figs');
if ~isfolder(out_dir), mkdir(out_dir); end

cLo = [0.00 0.45 0.74];  cHi = [0.85 0.33 0.10];  cNom = [0.30 0.30 0.30];

% (1) tornado chart: relative change of the median RMSE over each parameter range
nP = numel(sw);  lo = zeros(nP,1);  hi = zeros(nP,1);  v_lo = lo;  v_hi = hi;  span = lo;
for k = 1:nP
    rel = 100*(sw(k).R.rmse_med - rmse_nom)/rmse_nom;
    [lo(k), il] = min(rel);  [hi(k), ih] = max(rel);
    v_lo(k) = sw(k).values(il);  v_hi(k) = sw(k).values(ih);  span(k) = hi(k) - lo(k);
end
[~, ord] = sort(span, 'ascend');
[fig, ax] = NewFig(3.4, 2.4, 8);
for r = 1:nP
    k = ord(r);
    barh(ax, r, lo(k), 0.55, 'FaceColor', cLo, 'EdgeColor', 'none');
    barh(ax, r, hi(k), 0.55, 'FaceColor', cHi, 'EdgeColor', 'none');
    text(ax, lo(k) - 0.5, r, sprintf('%.3g', v_lo(k)), 'HorizontalAlignment', 'right', 'FontSize', 7, 'Color', cLo);
    text(ax, hi(k) + 0.5, r, sprintf('%.3g', v_hi(k)), 'HorizontalAlignment', 'left',  'FontSize', 7, 'Color', cHi);
end
xline(ax, 0, '-', 'Color', cNom, 'LineWidth', 1.2);
set(ax, 'YTick', 1:nP, 'YTickLabel', {sw(ord).label}, 'TickLabelInterpreter', 'tex');
xl = max(abs([lo; hi]))*1.25 + 2;  xlim(ax, [-xl xl]);  ylim(ax, [0.4 nP+0.6]);
ax.Position = [0.18 0.20 0.78 0.68];            % room for the eig(P) tick label
xlabel(ax, 'change of median RMSE relative to nominal [%]');
title(ax, sprintf('Parameter Sensitivity,  nominal RMSE = %.4f', rmse_nom));
% colour meaning is given in the caption; a legend does not fit at one column width
SaveFig(fig, 'sens_tornado', out_dir);

% (2) lambda x measurement noise
[fig, ax] = NewFig(3.4, 2.6, 7);
HeatMap(ax, lam_grid, sig_grid, G2_rmse, G2_div, '\lambda', '\sigma_n', 'Median RMSE over \lambda and \sigma_n');
SaveFig(fig, 'sens_lambda_noise', out_dir);

fprintf('Figures written to %s\n', out_dir);


%% -- Local Functions

function M = RunSeeds(o, seeds)
    n = numel(seeds);
    M = struct('div',false(1,n), 'rmse',nan(1,n), 'pk_load',nan(1,n), 't_rec',nan(1,n), 'g_err',nan(1,n));
    for s = 1:n
        o.seed = seeds(s);
        m = RunSim(o);
        M.div(s) = m.div;  M.rmse(s) = m.rmse;  M.pk_load(s) = m.pk_load;  M.t_rec(s) = m.t_rec;  M.g_err(s) = m.g_err;
    end
end

function m = RunSim(o)
% Plot-free copy of the closed-loop simulation in NarmaL2_ELM_Online.m
    N  = round(o.T / o.Ts);
    a_fun = @(t) 0.1 + 0.2*(t >= o.t_a_step);
    b_fun = @(t) 1.0 - 0.4*min(max((t - o.t_b_ramp(1))/diff(o.t_b_ramp), 0), 1);
    g_fun = @(x2, t) b_fun(t) * (1 + 0.4*cos(x2));
    d_fun = @(t) o.d_scale * (o.d_pulse*(t >= o.t_pulse & t < o.t_pulse + o.pulse_len*o.Ts) + o.d_step*(t >= o.t_dstep));

    t_hist = (0:N)' * o.Ts;
    half = floor((N+1)/2);  seg = floor(half/4);
    ref_step = [0.4*ones(seg,1); 0.8*ones(seg,1); 1.2*ones(seg,1); 0.8*ones(half-3*seg,1)];
    ref_sin  = 0.3*sin(0.5*pi*t_hist(1:(N+1)-half)) + 0.8;
    yref = [ref_step; ref_sin];

    rng(7);
    W  = o.w_scale * randn(o.Nh, o.ny+o.nu);
    bh = o.w_scale * randn(o.Nh, 1);
    nH = o.Nh + 1;
    theta = zeros(2*nH, 1);  theta(nH+1) = o.g0;
    P = o.P0 * eye(2*nH);
    rng(o.seed);

    x1 = zeros(1,N+1);  x2 = zeros(1,N+1);  y = zeros(1,N+1);  ym = zeros(1,N+1);  u = zeros(1,N+1);
    ym(1) = o.sigma_n*randn;
    g_hist = zeros(1,N);  g_true = zeros(1,N);
    u_hold = 0;  diverged = false;

    for i = 1:N
        t_now = t_hist(i);
        z = zeros(o.ny+o.nu,1);
        for k = 1:o.ny, z(k) = ym(max(i-k+1,1)); end
        for k = 1:o.nu, if i-k >= 1, z(o.ny+k) = u(i-k); end, end
        h = [1; tanh(W*z + bh)];
        f_hat = theta(1:nH)'*h;  g_hat = theta(nH+1:end)'*h;  g_hist(i) = g_hat;
        g_hat = min(max(g_hat, o.g_min), o.g_max);

        if t_now < o.T_warm
            if mod(i-1, o.prbs_hold) == 0, u_hold = o.u_prbs*(2*rand-1); end
            u_raw = u_hold;
        else
            u_raw = (yref(i+1) - f_hat) / g_hat;
        end
        u(i) = max(-o.u_max, min(o.u_max, u_raw));

        g_n = g_fun(x2(i), t_now);  g_true(i) = g_n;
        x1(i+1) = a_fun(t_now) - x1(i)^2 + x1(i)*x2(i);
        x2(i+1) = -x1(i) + exp(-x2(i)) + g_n*u(i) + d_fun(t_now);
        y(i+1)  = x1(i+1) + x2(i+1);
        ym(i+1) = y(i+1) + o.sigma_n*randn;
        if ~isfinite(y(i+1)) || abs(y(i+1)) > 1e3
            diverged = true;  y(i+1:end) = NaN;  break;
        end

        phi = [h; h*u(i)];
        e = ym(i+1) - phi'*theta;
        K = (P*phi) / (o.lambda + phi'*P*phi);
        theta = theta + K*e;
        P = (P - K*(phi'*P)) / o.lambda;  P = (P + P')/2;
        [V, D] = eig(P);
        if max(diag(D)) > o.Peig_max, D = min(D, o.Peig_max); P = V*D*V'; P = (P + P')/2; end
    end

    m.div = diverged;
    if diverged
        m.rmse = NaN;  m.pk_load = NaN;  m.t_rec = NaN;  m.g_err = NaN;
        return
    end
    err  = yref(1:N)' - y(1:N);
    ctrl = t_hist(1:N) >= o.T_warm;
    m.rmse = sqrt(mean(err(ctrl).^2));
    win  = t_hist(1:N) >= o.t_dstep & t_hist(1:N) < o.t_dstep + 4;
    m.pk_load = max(abs(err(win)));
    post = t_hist(1:N) >= o.t_dstep;
    kk = find(abs(err(post)) > 0.03, 1, 'last');
    if isempty(kk), m.t_rec = 0; else, m.t_rec = kk*o.Ts; end
    late = t_hist(1:N) >= o.t_b_ramp(2) + 1 & t_hist(1:N) < o.t_dstep;
    m.g_err = mean(abs(g_hist(late) - g_true(late)));
end

function HeatMap(ax, xv, yv, Z, Zdiv, xlab, ylab, ttl)
    im = imagesc(ax, 1:numel(xv), 1:numel(yv), Z);  im.AlphaData = ~isnan(Z);
    set(ax, 'YDir', 'normal', 'XTick', 1:numel(xv), 'XTickLabel', compose('%.3g', xv), ...
            'YTick', 1:numel(yv), 'YTickLabel', compose('%.3g', yv), 'YScale', 'linear');
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
    xlabel(ax, xlab);  ylabel(ax, ylab);  title(ax, ttl);  grid(ax, 'off');
end

function q = Pct(x, p)
% percentile by linear interpolation of the sorted sample
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
