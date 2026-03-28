clc; clear all; close all;
% Nonlinear System — NARMA-L2 System Identification 
%
% System:  x1[n+1] = 0.1 - x1^2 + x1*x2
%          x2[n+1] = -x1 + exp(-x2) + u
%          y[n]    = x1 + x2
%
% NARMA-L2 companion form:
%          y[n+1] = f(x1,x2) + g(x1,x2) * u[n]
%
% where:
%   f(x1,x2) = 0.1 - x1^2 + x1*x2 + (-x1 + exp(-x2))   [nonlinear, no u]
%   g(x1,x2) = 1                                          [input gain]
%
% Two networks trained separately:
%   NN_f : [x1(n), x2(n)]        -> f(x1,x2)
%   NN_g : [x1(n), x2(n)]        -> g(x1,x2)
%
% Controller (analytic rearrangement — no separate NN needed):
%   u(n) = (yref(n+1) - NN_f(x1,x2)) / NN_g(x1,x2)

%% Training Input
trainInput = 'prbs';   % 'prbs' | 'step' | 'dirac' | 'sweep'
N_sim = 2000; % Simulation Setup

%% Generate Training Data
rng(42);
u_train = GenerateInput(trainInput, N_sim);
[x1_tr, x2_tr, y_tr] = SimulateSystem(u_train, N_sim);

%% Generate Test Data (Sine Sweep)
n_vec  = (0:N_sim-1);
u_test = 1.5 * sin(2*pi*(0.01 + 9.99*n_vec/(2*N_sim)) .* n_vec/N_sim);
[x1_te, x2_te, y_te] = SimulateSystem(u_test, N_sim);

%% Prepare Targets for NN_f and NN_g
% y(n+1) = f(x1,x2) + g(x1,x2)*u(n)
%
% NN_f target: f(x1,x2) = y(n+1) - g*u(n) = y(n+1) - u(n)  [since g=1]
% NN_g target: g(x1,x2) = 1  [constant for this system — but NN must learn it]
%
% Input for both NN_f and NN_g: [x1(n), x2(n)]  — no u!

T_fg     = [x1_tr(1:end-1)', x2_tr(1:end-1)'];   % [N-1 x 2]
Yf_all   = y_tr(2:end)' - u_train(1:end-1)';      % f target = y(n+1) - u(n)
Yg_all   = ones(N_sim-1, 1);                       % g target = 1 (constant)

T_fg_te  = [x1_te(1:end-1)', x2_te(1:end-1)'];
Yf_test  = y_te(2:end)' - u_test(1:end-1)';
Yg_test  = ones(N_sim-1, 1);

%% NN Setup
[N, R_fg] = size(T_fg);   % R_fg = 2 (x1, x2 only)
S = 10;

%% Train / Validation Split
TRidx = 1:2:N;   VAidx = 2:2:N;
Xtr_f = T_fg(TRidx,:);   Ytr_f = Yf_all(TRidx,:);
Xva_f = T_fg(VAidx,:);   Yva_f = Yf_all(VAidx,:);
Xtr_g = T_fg(TRidx,:);   Ytr_g = Yg_all(TRidx,:);
Xva_g = T_fg(VAidx,:);   Yva_g = Yg_all(VAidx,:);

%% Train NN_f
fprintf('======= Training NN_f =======\n');
rng(1);
Wg_f = rand(S,R_fg)-0.5; bh_f = rand(S,1)-0.5;
Wc_f = rand(1,S)-0.5;    bc_f = rand(1,1)-0.5;
[Wg_f,bh_f,Wc_f,bc_f,Ftr_f,Fva_f,FvalMIN_f] = ...
    TrainLM(Xtr_f,Ytr_f,Xva_f,Yva_f,Wg_f,bh_f,Wc_f,bc_f,S,R_fg,100);

%% Train NN_g
fprintf('\n======= Training NN_g =======\n');
rng(2);
Wg_g = rand(S,R_fg)-0.5; bh_g = rand(S,1)-0.5;
Wc_g = rand(1,S)-0.5;    bc_g = rand(1,1)-0.5;
[Wg_g,bh_g,Wc_g,bc_g,Ftr_g,Fva_g,FvalMIN_g] = ...
    TrainLM(Xtr_g,Ytr_g,Xva_g,Yva_g,Wg_g,bh_g,Wc_g,bc_g,S,R_fg,100);

%% Evaluation
RMSE     = @(a,b) sqrt(mean((a-b).^2));
yhatf_TR = NNforward(Xtr_f, Wg_f,bh_f,Wc_f,bc_f);
yhatf_VA = NNforward(Xva_f, Wg_f,bh_f,Wc_f,bc_f);
yhatf_TE = NNforward(T_fg_te, Wg_f,bh_f,Wc_f,bc_f);
yhatg_TE = NNforward(T_fg_te, Wg_g,bh_g,Wc_g,bc_g);

% Composite output: y_hat(n+1) = NN_f + NN_g * u
y_composite_te = yhatf_TE + yhatg_TE .* u_test(1:end-1)';

fprintf('\n--- NN_f RMSE ---\n');
fprintf('Train: %.6f  Val: %.6f  Test: %.6f\n', ...
        RMSE(Ytr_f,yhatf_TR), RMSE(Yva_f,yhatf_VA), RMSE(Yf_test,yhatf_TE));
fprintf('\n--- Composite y(n+1) RMSE (test) ---\n');
fprintf('RMSE: %.6f\n', RMSE(y_te(2:end)', y_composite_te));

%% Plots
PlotResults(TRidx, VAidx, ...
            Ytr_f, yhatf_TR, Yva_f, yhatf_VA, ...
            Yf_test, yhatf_TE, y_te(2:end)', y_composite_te, ...
            Ftr_f, Fva_f, Ftr_g, Fva_g, S, FvalMIN_f, FvalMIN_g, ...
            RMSE, trainInput);

%% Save Model
save('NL_PlantModel.mat', ...
     'Wg_f','bh_f','Wc_f','bc_f', ...
     'Wg_g','bh_g','Wc_g','bc_g', ...
     'S','R_fg');
fprintf('Saved: NL_PlantModel.mat  [trainInput: %s]\n', trainInput);


%%                        LOCAL FUNCTIONS

function u = GenerateInput(type, N)
    switch type
        case 'prbs'
            u = zeros(1,N); period = 20;
            for i = 1:N
                if mod(i,period)==0; u(i) = 2*(rand-0.5);
                else; u(i) = u(max(i-1,1)); end
            end
        case 'step'
            u = ones(1,N);
        case 'dirac'
            u = zeros(1,N); u(1) = 1;
        case 'sweep'
            n = 0:N-1;
            u = sin(2*pi*(0.01 + 9.99*n/(2*N)) .* n/N);
        otherwise
            error('Unknown type: %s', type);
    end
end

% -------------------------------------------------------------------------
function [x1, x2, y] = SimulateSystem(u, N)
% Discrete-time nonlinear system
    x1 = zeros(1,N); x2 = zeros(1,N); y = zeros(1,N);
    for n = 1:N-1
        x1(n+1) = 0.1 - x1(n)^2 + x1(n)*x2(n);
        x2(n+1) = -x1(n) + exp(-x2(n)) + u(n);
        y(n+1)  = x1(n+1) + x2(n+1);
    end
    y(1) = x1(1) + x2(1);
end

% -------------------------------------------------------------------------
function yhat = NNforward(T, Wg, bh, Wc, bc)
    N = size(T,1); yhat = zeros(N,1);
    for n = 1:N
        yhat(n) = Wc * tanh(Wg * T(n,:)' + bh) + bc;
    end
end

% -------------------------------------------------------------------------
function J = ComputeJacobian(T, Wg, bh, Wc, S, R)
    N_params = S*(R+2)+1;
    N        = size(T,1);
    J        = zeros(N, N_params);
    idx_bc   = N_params;
    idx_Wc   = S*(R+1)+1 : S*(R+2);
    idx_bh   = S*R+1     : S*R+S;
    for i = 1:N
        a  = tanh(Wg * T(i,:)' + bh);
        da = 1 - a.^2;
        J(i,idx_bc) = -1;
        J(i,idx_Wc) = -a';
        J(i,idx_bh) = -(Wc .* da');
        for j = 1:S*R
            nr = mod(j-1,S)+1; ni = fix((j-1)/S)+1;
            J(i,j) = -Wc(nr)*T(i,ni)*da(nr);
        end
    end
end

% -------------------------------------------------------------------------
function [Wg,bh,Wc,bc,Ftr,Fva,FvalMIN] = ...
         TrainLM(Xtr,Ytr,Xva,Yva,Wg,bh,Wc,bc,S,R,Nmax)
    I_mat = eye(S*(R+2)+1); mu = 1;
    FvalMIN = inf; xbest = mat2vec(Wg,bh,Wc,bc);
    Ftr = zeros(1,Nmax); Fva = zeros(1,Nmax);
    condition = true; iter = 0;
    while condition
        iter = iter+1;
        yhat = NNforward(Xtr,Wg,bh,Wc,bc);
        e    = Ytr-yhat; f = e'*e;
        J    = ComputeJacobian(Xtr,Wg,bh,Wc,S,R);
        inner = true;
        while inner
            p  = -(J'*J+mu*I_mat)\(J'*e);
            xv = mat2vec(Wg,bh,Wc,bc);
            [Wgz,bhz,Wcz,bcz] = vec2mat(xv+p,S,R);
            fz = norm(Ytr-NNforward(Xtr,Wgz,bhz,Wcz,bcz))^2;
            if fz < f
                xv=xv+p; [Wg,bh,Wc,bc]=vec2mat(xv,S,R);
                mu=0.1*mu; inner=false;
            else
                mu=10*mu;
                if mu>1e20; inner=false; condition=false; end
            end
        end
        yhat=NNforward(Xtr,Wg,bh,Wc,bc); e=Ytr-yhat;
        Ftr(iter) = e'*e;
        Fva(iter) = norm(Yva-NNforward(Xva,Wg,bh,Wc,bc))^2;
        if Fva(iter)<FvalMIN
            xbest=mat2vec(Wg,bh,Wc,bc); FvalMIN=Fva(iter);
        end
        fprintf('Iter: %4d  ||g||: %.2e  f_tr: %.2e  f_val: %.2e\n', ...
                iter,norm(2*J'*e),Ftr(iter),Fva(iter));
        if iter>=Nmax; condition=false; end
    end
    Ftr=Ftr(1:iter); Fva=Fva(1:iter);
    [Wg,bh,Wc,bc]=vec2mat(xbest,S,R);
end

% -------------------------------------------------------------------------
function x = mat2vec(Wg,bh,Wc,bc)
    R=size(Wg,2); x=[];
    for r=1:R; x=[x;Wg(:,r)]; end
    x=[x;bh;Wc';bc];
end

function [Wg,bh,Wc,bc] = vec2mat(x,S,R)
    Wg=[];
    for r=1:R; Wg=[Wg,x((r-1)*S+1:r*S)]; end
    bh=x(S*R+1:S*R+S); Wc=x(S*(R+1)+1:S*(R+2))'; bc=x(S*(R+2)+1);
end

% -------------------------------------------------------------------------
function PlotResults(TRidx,VAidx, ...
                     Ytr_f,yhatf_TR,Yva_f,yhatf_VA, ...
                     Yf_test,yhatf_TE,y_true_te,y_comp_te, ...
                     Ftr_f,Fva_f,Ftr_g,Fva_g,S,FvalMIN_f,FvalMIN_g, ...
                     RMSE,trainInput)
    figure('units','normalized','outerposition',[0 0 1 1],'color','w');

    subplot(2,3,1)
    plot(TRidx,Ytr_f,'r','LineWidth',2); hold on; grid on;
    plot(TRidx,yhatf_TR,'k--','LineWidth',1.8);
    legend('True $f$','NN\_f','Interpreter','latex');
    xlabel('Step'); ylabel('$f(x_1,x_2)$','Interpreter','latex');
    title(sprintf('NN\\_f Train  RMSE=%.5f', RMSE(Ytr_f,yhatf_TR)));

    subplot(2,3,2)
    plot(VAidx,Yva_f,'r','LineWidth',2); hold on; grid on;
    plot(VAidx,yhatf_VA,'k--','LineWidth',1.8);
    legend('True $f$','NN\_f','Interpreter','latex');
    xlabel('Step'); ylabel('$f(x_1,x_2)$','Interpreter','latex');
    title(sprintf('NN\\_f Validation  RMSE=%.5f', RMSE(Yva_f,yhatf_VA)));

    subplot(2,3,3)
    plot(y_true_te,'r','LineWidth',2); hold on; grid on;
    plot(y_comp_te,'k--','LineWidth',1.8);
    legend('True $y$','$\hat{f}+\hat{g}\cdot u$','Interpreter','latex');
    xlabel('Step'); ylabel('$y(n+1)$','Interpreter','latex');
    title(sprintf('Composite Test  RMSE=%.5f', RMSE(y_true_te,y_comp_te)));

    subplot(2,3,4)
    semilogy(Ftr_f,'b','LineWidth',2); hold on; grid on;
    semilogy(Fva_f,'r--','LineWidth',2);
    legend('Train','Validation','Interpreter','latex');
    xlabel('Iteration'); ylabel('SSE');
    title(sprintf('NN\\_f  S=%d  BestVal=%.5f', S, FvalMIN_f));

    subplot(2,3,5)
    semilogy(Ftr_g,'b','LineWidth',2); hold on; grid on;
    semilogy(Fva_g,'r--','LineWidth',2);
    legend('Train','Validation','Interpreter','latex');
    xlabel('Iteration'); ylabel('SSE');
    title(sprintf('NN\\_g  S=%d  BestVal=%.5f', S, FvalMIN_g));

    subplot(2,3,6)
    plot(Yf_test,'r','LineWidth',2); hold on; grid on;
    plot(yhatf_TE,'k--','LineWidth',1.8);
    legend('True $f$','NN\_f','Interpreter','latex');
    xlabel('Step'); ylabel('$f(x_1,x_2)$','Interpreter','latex');
    title(sprintf('NN\\_f Test (Sweep)  RMSE=%.5f', RMSE(Yf_test,yhatf_TE)));

    sgtitle(sprintf('NARMA-L2 System ID train Input: %s', trainInput),'FontSize',14);
end