clc; clear; close all;
% Stability Analysis - Online NARMA-L2 Control with RLS-ELM
% Written By: Rasit Evduzen
% Date: 17-Sep-2026
%
% (1) Zero dynamics: under perfect tracking y = y*, the hidden state obeys
%       x1[n+1] = a - 2 x1^2 + x1 y*
%     stable fixed point  x1* = ( -(1-y*) + sqrt((1-y*)^2 + 8a) ) / 4
%     local multiplier    dx1[n+1]/dx1 = 1 - sqrt((1-y*)^2 + 8a)
%     fixed point stable  <=>  (1-y*)^2 + 8a < 4
%     beyond that boundary the fixed point flips into a bounded periodic orbit (period doubling),
%     the internal state stays bounded and the output can still be tracked; escape only for larger a
% (2) Basin of attraction of the plant under the ideal NARMA-L2 law with |u| <= u_max
%
% Produces two paper figures, saved separately in ./figs:
%   stab_zero_dynamics   zero-dynamics stability region in the (y*, a) plane
%   stab_basin           basin of attraction in the (x1, x2) plane

%% Parameters (nominal, as in NarmaL2_ELM_Online.m)
u_max = 3.0;   b = 1.0;
a_vals = [0.1 0.3];          % plant parameter before / after the step
y_ref_range = [0.4 1.2];     % reference levels used in the simulation

%% (1) Zero-Dynamics Stability Region
ys = linspace(-1.5, 3.5, 1001);
as = linspace(0, 1.3, 521);
[Y, A] = meshgrid(ys, as);
mult   = 1 - sqrt((1 - Y).^2 + 8*A);          % local multiplier of the zero dynamics
stable = abs(mult) < 1;
y_bnd  = @(a) 1 + [-1; 1]*sqrt(max(4 - 8*a, 0));   % analytic boundary  (1-y*)^2 + 8a = 4

x = (-(1 - Y) + sqrt((1 - Y).^2 + 8*A))/4 + 1e-3;   % start next to the fixed point
bounded = true(size(Y));
for n = 1:1000
    x = A - 2*x.^2 + x.*Y;
    bounded = bounded & isfinite(x) & abs(x) < 1e3;
    x(~bounded) = 0;
end
zd_region = double(bounded) + double(stable);        % 0 unbounded, 1 bounded oscillation, 2 stable fixed point

%% (2) Basin of Attraction, ideal NARMA-L2 with saturation
y_star = 0.8;   a0 = a_vals(1);
[X1, X2] = meshgrid(linspace(-3, 3, 601), linspace(-3, 3, 601));
x1 = X1;  x2 = X2;  alive = true(size(X1));
for n = 1:200
    f_true = a0 - x1.^2 + x1.*x2 - x1 + exp(-x2);
    g_true = b*(1 + 0.4*cos(x2));
    u = max(-u_max, min(u_max, (y_star - f_true)./g_true));
    x1n = a0 - x1.^2 + x1.*x2;
    x2n = -x1 + exp(-x2) + g_true.*u;
    x1 = x1n;  x2 = x2n;
    alive = alive & isfinite(x1) & isfinite(x2) & (abs(x1) + abs(x2) < 1e3);
    x1(~alive) = 0;  x2(~alive) = 0;
end
basin = alive;
x1_fp = (-(1 - y_star) + sqrt((1 - y_star)^2 + 8*a0))/4;   % operating point
x2_fp = y_star - x1_fp;

%% Paper Figures
out_dir = fullfile(pwd, 'figs');
if ~isfolder(out_dir), mkdir(out_dir); end

cStable = [0.80 0.92 0.80];  cOsc = [0.98 0.93 0.75];  cUnst = [0.96 0.82 0.80];  cOp = [0.10 0.10 0.10];  cA = [0.85 0.33 0.10];

% (1) zero dynamics
[fig, ax] = NewFig(3.4, 2.4, 7);
image(ax, ys, as, RegionRGB(zd_region + 1, [cUnst; cOsc; cStable]));  set(ax, 'YDir', 'normal');
aa = linspace(0, 0.5, 200);  yb = y_bnd(aa);
plot(ax, yb(1,:), aa, '-', 'Color', cOp, 'LineWidth', 1.2);
plot(ax, yb(2,:), aa, '-', 'Color', cOp, 'LineWidth', 1.2);
for a = a_vals
    plot(ax, y_ref_range, [a a], '-', 'Color', cA, 'LineWidth', 3);
end
text(ax, 1.0, 0.05, 'stable fixed point', 'HorizontalAlignment', 'center', 'FontSize', 6.5);
text(ax, 1.0, 0.75, 'bounded oscillation', 'HorizontalAlignment', 'center', 'FontSize', 6.5);
text(ax, 1.0, 1.22, 'unbounded', 'HorizontalAlignment', 'center', 'FontSize', 6.5);
text(ax, y_ref_range(2) + 0.1, a_vals(1), 'a = 0.1', 'Color', cA, 'FontSize', 6, 'VerticalAlignment', 'middle');
text(ax, y_ref_range(2) + 0.1, a_vals(2), 'a = 0.3', 'Color', cA, 'FontSize', 6, 'VerticalAlignment', 'middle');
xlim(ax, ys([1 end]));  ylim(ax, as([1 end]));
xlabel(ax, 'reference level y^*');  ylabel(ax, 'plant parameter a');
title(ax, 'Zero Dynamics');
SaveFig(fig, 'stab_zero_dynamics', out_dir);

% (2) basin of attraction
[fig, ax] = NewFig(3.4, 3.2, 8);
image(ax, X1(1,:), X2(:,1), RegionRGB(double(basin) + 1, [cUnst; cStable]));  set(ax, 'YDir', 'normal');
plot(ax, x1_fp, x2_fp, 'o', 'MarkerFaceColor', 'w', 'MarkerEdgeColor', cOp, 'MarkerSize', 5, 'LineWidth', 1.2);
text(ax, x1_fp + 0.15, x2_fp, 'operating point', 'FontSize', 6, 'VerticalAlignment', 'middle');
text(ax, 0, 2.5, 'converges to y^*', 'HorizontalAlignment', 'center', 'FontSize', 6.5);
text(ax, -2.2, -2.5, 'diverges', 'HorizontalAlignment', 'center', 'FontSize', 6.5);
xlim(ax, X1(1,[1 end]));  ylim(ax, X2([1 end],1));
xlabel(ax, 'x_1(0)');  ylabel(ax, 'x_2(0)');
title(ax, sprintf('Basin of Attraction,  |u| \\leq %.0f,  y^* = %.1f', u_max, y_star));
axis(ax, 'square');
SaveFig(fig, 'stab_basin', out_dir);

fprintf('Figures written to %s\n', out_dir);


%% -- Local Functions

function rgb = RegionRGB(idx, pal)
% Turn an integer region map into a truecolor image. Writing the colours into the
% image rather than into a colormap keeps the regions correct through the vector
% PDF export, which does not always carry an indexed image and its colormap.
    [m, n] = size(idx);
    rgb = zeros(m, n, 3);
    for c = 1:3
        v = pal(:, c);
        rgb(:,:,c) = reshape(v(idx), m, n);
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
% These two figures are region maps. A vector export embeds the region image at
% the on-screen pixel size, about 75 dpi, so they go out as a 600 dpi raster
% instead; the grids above are dense enough that the boundaries stay sharp.
    exportgraphics(fig, fullfile(out_dir, [name '.pdf']), 'ContentType', 'image', 'Resolution', 600);  % for LaTeX
    exportgraphics(fig, fullfile(out_dir, [name '.tif']), 'Resolution', 600);                        % for submission
    exportgraphics(fig, fullfile(out_dir, [name '.png']), 'Resolution', 300);
end
