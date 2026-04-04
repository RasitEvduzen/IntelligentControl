clc; clear all; close all;
% Water Wheel Chaos Control  RL Agent
% Written By: Rasit
% Date: 04-Apr-2026
%% Load trained agent
load('waterwheel_agent.mat', 'agent')
fprintf('RL agent loaded.\n')

%% Parameters
K  = 1.0;    % Leaking rate       how fast water drains from buckets [1/s]
q1 = 10.0;   % Pump strength      water input rate, first Fourier harmonic [m/s]
v  = 5.0;    % Angular damping    rotational friction coefficient [kg·m²/s]
I  = 1.0;    % Moment of inertia  resistance of wheel to angular acceleration [kg·m²]
g  = 9.81;   % Gravity            gravitational acceleration [m/s²]
r  = 1.0;    % Wheel radius       distance from center to buckets [m]


Ts         = 0.02;
T_total    = 80;       % longer simulation
t_on1      = 20;       % RL first ON
t_off      = 40;       % RL OFF
t_on2      = 60;       % RL second ON
SimStep    = 20;

omega_star = 7.787;    % Referance Angular Velocity
u_max      = 25;

NoD      = round(T_total / Ts);
tspan    = (0:NoD-1) * Ts;
k_on1    = round(t_on1 / Ts);
k_off    = round(t_off  / Ts);
k_on2    = round(t_on2 / Ts);

%% Simulate
X = zeros(3, NoD);
U = zeros(1, NoD);

% Start from chaotic attractor
s = [0.1; 0.5; 0.0];
for i = 1:NoD
    s = rk4step(s, Ts, K, q1, v, I, g, r, 0);
end
X(:,1) = s;

for k = 1:NoD
    s = X(:,k);
    rl_active = (k >= k_on1 && k < k_off) || (k >= k_on2);
    if rl_active
        e_omega = (s(1) - omega_star);
        obs     = max(-1, min(1, [s(1)/15; s(2)/5; s(3)/10; e_omega]));
        actionCell = getAction(agent, {obs});
        u          = actionCell{1}(1) * u_max;
    else
        u = 0;
    end
    U(k)     = u;
    X(:,k+1) = rk4step(s, Ts, K, q1, v, I, g, r, u);
end

%% Visualization
NBUCKETS = 12;
R_W      = 1.0;
bAngles  = linspace(0, 2*pi*(NBUCKETS-1)/NBUCKETS, NBUCKETS);
theta    = linspace(0, 2*pi, 300);
accumAng = 0;

figure('units','normalized','outerposition',[0 0 1 1],'color','w')
% wheel animation
ax1 = subplot(2,3,[1 4]);
axis(ax1,[-2 2 -2 2]); axis(ax1,'equal','off'); hold(ax1,'on')
plot(ax1,R_W*cos(theta),R_W*sin(theta),'Color',[0.6 0.6 0.6],'LineWidth',2)
plot(ax1,0.12*cos(theta),0.12*sin(theta),'Color',[0.3 0.3 0.3],'LineWidth',2)
plot(ax1,0,0,'ko','MarkerFaceColor',[0.2 0.2 0.2],'MarkerSize',7)

hS = gobjects(NBUCKETS,1); hB = gobjects(NBUCKETS,1);
for k = 1:NBUCKETS
    hS(k) = plot(ax1,[0,R_W*cos(bAngles(k))],[0,R_W*sin(bAngles(k))],...
        'Color',[0.82 0.82 0.82],'LineWidth',0.9);
    hB(k) = plot(ax1,R_W*cos(bAngles(k)),R_W*sin(bAngles(k)),...
        's','MarkerSize',15,'MarkerFaceColor',[0.2 0.5 1],'MarkerEdgeColor',[0.1 0.3 0.7]);
end
text(ax1,0,1.72,'Water In','FontSize',10,'Color',[0.1 0.4 0.9],'HorizontalAlignment','center')
text(ax1,0,-1.72,'Drain','FontSize',10,'Color',[0.4 0.6 0.9],'HorizontalAlignment','center')

hArrow = quiver(ax1,0,0,0,0.4,'r','LineWidth',3,'MaxHeadSize',0.6,'AutoScale','off');
hDir   = text(ax1,0,1.45,'','FontSize',12,'FontWeight','bold','HorizontalAlignment','center');
hTime  = text(ax1,-1.90,-1.60,'','FontSize',10,'Color',[0 0 0]);
hMode  = text(ax1,-1.90,-2,'','FontSize',11,'FontWeight','bold');
title(ax1,'Malkus Water Wheel','FontSize',14)

% omega time series
ax2 = subplot(2,3,3);
hold(ax2,'on'); grid(ax2,'on'); box(ax2,'on')
xline(ax2,t_on1,'--','Color',[0.1 0.6 0.1],'LineWidth',2,'Label','RL ON')
xline(ax2,t_off,'--','Color',[0.8 0.1 0.1],'LineWidth',2,'Label','RL OFF')
xline(ax2,t_on2,'--','Color',[0.1 0.6 0.1],'LineWidth',2,'Label','RL ON')
yline(ax2,omega_star,'--','Color',[0 0 0],'LineWidth',1.5,'Label','\omega*')
hLw = plot(ax2,NaN,NaN,'b-','LineWidth',1.5);
xlabel(ax2,'Time [s]'); ylabel(ax2,'\omega  [rad/s]')
title(ax2,'Angular velocity'); xlim(ax2,[0 T_total]); ylim(ax2,[-20 20])

% control signal
ax3 = subplot(2,3,6);
hold(ax3,'on'); grid(ax3,'on'); box(ax3,'on')
xline(ax3,t_on1,'--','Color',[0.1 0.6 0.1],'LineWidth',2)
xline(ax3,t_off,'--','Color',[0.8 0.1 0.1],'LineWidth',2)
xline(ax3,t_on2,'--','Color',[0.1 0.6 0.1],'LineWidth',2)
yline(ax3,0,'k--')
hLu = plot(ax3,NaN,NaN,'r-','LineWidth',1.5);
xlabel(ax3,'Time [s]'); ylabel(ax3,'Control torque u  [N·m]')
title(ax3,'RL Control Input'); xlim(ax3,[0 T_total]); ylim(ax3,[-u_max*1.2 u_max*1.2])

ax4 = subplot(2,3,[2 5]);
hSt = plot3(ax4,NaN,NaN,NaN,'k','LineWidth',1);
hold(ax4,'on'),title("Phase Space")
hStc = scatter3(ax4,NaN,NaN,NaN,100,'r','filled');
view(10,20)
%% Loop
for k = 1:NoD
    if mod(k,SimStep) == 0 || k == NoD
        w  = X(1,k); tn = tspan(k);

        accumAng = accumAng + w*Ts*SimStep*0.1;
        for bk = 1:NBUCKETS
            ang = bAngles(bk)+accumAng;
            fill = max(0,cos(ang));
            bx = R_W*cos(ang); by = R_W*sin(ang);
            set(hB(bk),'XData',bx,'YData',by,'MarkerFaceColor',[0.05,0.15+0.3*fill,0.25+0.65*fill])
            set(hS(bk),'XData',[0,bx],'YData',[0,by])
        end
        arrowLen = min(max(w*0.10,-0.8),0.8);
        set(hArrow,'XData',-0.3,'YData',0,'UData',0,'VData',arrowLen)
        set(hTime,'String',sprintf('t = %.2f s',tn))

        rl_on = (tn >= t_on1 && tn < t_off) || (tn >= t_on2);
        if rl_on
            set(hMode,'String','RL ACTIVE  stable','Color',[0.1 0.6 0.1])
        else
            set(hMode,'String','CHAOTIC Regime! no control','Color',[0.8 0.1 0.1])
        end

        if w > 0.1
            set(hDir,'String','Clockwise \rightarrow','Color',[0.1 0.55 0.1])

        elseif w < -0.1
            set(hDir,'String','\leftarrow Counter-CW','Color',[0.75 0.15 0.15])

        else
            set(hDir,'String','Near stall','Color',[0.5 0.5 0.5])
        end
        set(hSt,'XData',X(1,1:k),'YData',X(2,1:k),'ZData',X(3,1:k))
        set(hStc,'XData',X(1,k),'YData',X(2,k),'ZData',X(3,k))
        set(hLw,'XData',tspan(1:k),'YData',X(1,1:k))
        set(hLu,'XData',tspan(1:k),'YData',U(1:k))
        drawnow;
    end
end


%% Functions
function s_next = rk4step(s, dt, K, q1, v, I, g, r, u)
k1=wODE(s,K,q1,v,I,g,r,u);
k2=wODE(s+dt/2*k1,K,q1,v,I,g,r,u);
k3=wODE(s+dt/2*k2,K,q1,v,I,g,r,u);
k4=wODE(s+dt*k3,K,q1,v,I,g,r,u);
s_next = s+dt/6*(k1+2*k2+2*k3+k4);
end

function ds = wODE(s, K, q1, v, I, g, r, u)
ds = [(pi*g*r*s(2)-v*s(1))/I+u;
    s(1)*s(3)-K*s(2);
    -s(1)*s(2)+q1-K*s(3)];
end