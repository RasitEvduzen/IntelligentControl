classdef WaterWheelRLEnv < rl.env.MATLABEnvironment
    % Water Wheel Chaos Control  RL Environment
    % Written By: Rasit
    % Date: 04-Apr-2026
    properties
        K  = 1.0;    % Leaking rate       how fast water drains from buckets [1/s]
        q1 = 10.0;   % Pump strength      water input rate, first Fourier harmonic [m/s]
        v  = 5.0;    % Angular damping    rotational friction coefficient [kg·m²/s]
        I  = 1.0;    % Moment of inertia  resistance of wheel to angular acceleration [kg·m²]
        g  = 9.81;   % Gravity            gravitational acceleration [m/s²]
        r  = 1.0;    % Wheel radius       distance from center to buckets [m]

        Ts       = 0.02;
        MaxSteps = 1500;        % long enough to see stabilization 

        omega_star = 7.787;
        a1_star    = 1.263;
        b1_star    = 0.162;
        u_max      = 25.0;

        State      = zeros(3,1);
        StepCount  = 0;
        EpisodeNum = 0;
        TotalEp    = 800;       % same as trainOpts.MaxEpisodes

        ChaosTraj  = [];
    end

    methods
        function this = WaterWheelRLEnv()
            % Obs: [omega/15, a1/5, b1/10, e_omega/15] 
            ObsInfo = rlNumericSpec([4 1],'LowerLimit',-ones(4,1),'UpperLimit',ones(4,1));
            ObsInfo.Name = 'WaterWheelStates';
            ActInfo = rlNumericSpec([1 1],'LowerLimit',-1,'UpperLimit',1);
            ActInfo.Name = 'ControlTorque';

            this = this@rl.env.MATLABEnvironment(ObsInfo, ActInfo);

            fprintf('Pre-computing chaotic attractor...\n')
            s = [0.1;0.5;0.0];
            traj = zeros(3,10000);
            for i=1:10000
                s=wRK4(s,this.Ts,this.K,this.q1,this.v,this.I,this.g,this.r,0);
                traj(:,i)=s;
            end
            this.ChaosTraj = traj(:,2000:end);
            this.State = this.sampleInitialState();
            fprintf('Done.\n')
        end

        function [obs,reward,isDone,loggedSignals] = step(this, action)
            loggedSignals = [];
            u = double(action) * this.u_max;

            this.State    = wRK4(this.State,this.Ts,this.K,this.q1,...
                this.v,this.I,this.g,this.r,u);
            this.StepCount = this.StepCount+1;

            omega = this.State(1);
            a1    = this.State(2);

            % Dense quadratic reward  omega error is primary signal
            e_w    = (omega - this.omega_star) / this.omega_star;
            e_a    = (a1    - this.a1_star)    / 5;
            reward = -(e_w^2 + 0.1*e_a^2) - 0.001*action^2;

            obs    = this.makeObs(this.State);
            isDone = abs(omega)>50 || this.StepCount>=this.MaxSteps;
        end

        function obs = reset(this)
            this.EpisodeNum = this.EpisodeNum + 1;
            this.State      = this.sampleInitialState();
            this.StepCount  = 0;
            obs             = this.makeObs(this.State);
        end
    end

    methods (Access = private)
        function s0 = sampleInitialState(this)
            % Curriculum: early episodes start near fixed point
            progress = min(this.EpisodeNum / this.TotalEp, 1.0);
            if rand < (1 - progress)^2        % 100% easy at start, 0% at end
                % Near fixed point: easy
                noise = 0.5 * (1 + progress);  % noise grows with training
                s0 = [this.omega_star; this.a1_star; this.b1_star] + noise*randn(3,1);
            else
                % From chaotic attractor: hard
                idx = randi(size(this.ChaosTraj,2));
                s0  = this.ChaosTraj(:,idx);
            end
        end

        function obs = makeObs(this, s)
            e_omega = (s(1) - this.omega_star) / 15;   % explicit error
            obs = max(-1,min(1, [s(1)/15; s(2)/5; s(3)/10; e_omega]));
        end
    end
end

function s_next = wRK4(s,dt,K,q1,v,I,g,r,u)
k1=wODE(s,K,q1,v,I,g,r,u); 
k2=wODE(s+dt/2*k1,K,q1,v,I,g,r,u);
k3=wODE(s+dt/2*k2,K,q1,v,I,g,r,u); 
k4=wODE(s+dt*k3,K,q1,v,I,g,r,u);
s_next=s+dt/6*(k1+2*k2+2*k3+k4);
end

function ds = wODE(s,K,q1,v,I,g,r,u)
ds=[(pi*g*r*s(2)-v*s(1))/I+u; 
    s(1)*s(3)-K*s(2); 
    -s(1)*s(2)+q1-K*s(3)];
end
