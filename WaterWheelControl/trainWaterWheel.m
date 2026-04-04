clc; clear; close all;
% SAC Training  Water Wheel Chaos Control
% Written By: Rasit
% Date: 04-Apr-2026

%% Environment
env     = WaterWheelRLEnv();
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);
obsSize = obsInfo.Dimension(1);   % 4
actSize = actInfo.Dimension(1);   % 1
fprintf('obs:%d  act:%d  omega*:%.3f\n', obsSize, actSize, env.omega_star)

%% Actor
lgA = layerGraph([featureInputLayer(obsSize,'Name','obs')
                  fullyConnectedLayer(128,  'Name','fc1')
                  reluLayer(               'Name','r1')
                  fullyConnectedLayer(128,  'Name','fc2')
                  reluLayer(               'Name','r2')]);
lgA = addLayers(lgA,[fullyConnectedLayer(actSize,'Name','fc_m'); tanhLayer('Name','mean')]);
lgA = addLayers(lgA,[fullyConnectedLayer(actSize,'Name','fc_s'); softplusLayer('Name','std')]);
lgA = connectLayers(lgA,'r2','fc_m');
lgA = connectLayers(lgA,'r2','fc_s');
actor = rlContinuousGaussianActor(dlnetwork(lgA), obsInfo, actInfo, ...
    'ActionMeanOutputNames','mean','ActionStandardDeviationOutputNames','std');

%% Critic 1
obsP1 = [featureInputLayer(obsSize,'Name','obs_in')
         fullyConnectedLayer(128,  'Name','obs_fc')
         reluLayer(               'Name','obs_rl')];
actP1 = [featureInputLayer(actSize,'Name','act_in')
         fullyConnectedLayer(128,  'Name','act_fc')
         reluLayer(               'Name','act_rl')];
jnP1  = [concatenationLayer(1,2,  'Name','cat')
         fullyConnectedLayer(128, 'Name','jn_fc')
         reluLayer(              'Name','jn_rl')
         fullyConnectedLayer(1,  'Name','qval')];
lg1 = addLayers(layerGraph(obsP1),actP1);
lg1 = addLayers(lg1,jnP1);
lg1 = connectLayers(lg1,'obs_rl','cat/in1');
lg1 = connectLayers(lg1,'act_rl','cat/in2');
critic1 = rlQValueFunction(dlnetwork(lg1), obsInfo, actInfo, ...
    'ObservationInputNames','obs_in','ActionInputNames','act_in');

%% Critic 2
obsP2 = [featureInputLayer(obsSize,'Name','obs_in')
         fullyConnectedLayer(128,  'Name','obs_fc')
         reluLayer(               'Name','obs_rl')];
actP2 = [featureInputLayer(actSize,'Name','act_in')
         fullyConnectedLayer(128,  'Name','act_fc')
         reluLayer(               'Name','act_rl')];
jnP2  = [concatenationLayer(1,2,  'Name','cat')
         fullyConnectedLayer(128, 'Name','jn_fc')
         reluLayer(              'Name','jn_rl')
         fullyConnectedLayer(1,  'Name','qval')];
lg2 = addLayers(layerGraph(obsP2),actP2);
lg2 = addLayers(lg2,jnP2);
lg2 = connectLayers(lg2,'obs_rl','cat/in1');
lg2 = connectLayers(lg2,'act_rl','cat/in2');
critic2 = rlQValueFunction(dlnetwork(lg2), obsInfo, actInfo, ...
    'ObservationInputNames','obs_in','ActionInputNames','act_in');

fprintf('Networks built. Critic Q-test: %.4f\n', ...
        getValue(critic1,{rand(obsSize,1)*2-1},{rand(actSize,1)*2-1}))

%% Agent
opt = rlSACAgentOptions( ...
    'SampleTime',             env.Ts, ...
    'ExperienceBufferLength', 2e5, ...
    'MiniBatchSize',          256, ...
    'DiscountFactor',         0.99, ...
    'TargetSmoothFactor',     5e-3);
opt.ActorOptimizerOptions.LearnRate     = 3e-4;
opt.CriticOptimizerOptions(1).LearnRate = 3e-4;
opt.CriticOptimizerOptions(2).LearnRate = 3e-4;
opt.EntropyWeightOptions.LearnRate      = 3e-4;
agent = rlSACAgent(actor,[critic1 critic2],opt);

%% Training
trainOpts = rlTrainingOptions( ...
    'MaxEpisodes',                800, ...
    'MaxStepsPerEpisode',         env.MaxSteps, ...
    'ScoreAveragingWindowLength', 30, ...
    'StopTrainingCriteria',       'AverageReward', ...
    'StopTrainingValue',          -50, ...
    'SaveAgentCriteria',          'EpisodeReward', ...
    'SaveAgentValue',             -80, ...
    'SaveAgentDirectory',         'saved_agents', ...
    'Verbose',                    true, ...
    'Plots',                      'training-progress');

fprintf('Training (curriculum: easy start -> full chaos over 800 episodes)...\n')
result = train(agent, env, trainOpts);
save('waterwheel_agent.mat','agent','result')
fprintf('Done.\n')
