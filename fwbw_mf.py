import numpy as np
#import matplotlib.pyplot as plt
#import math
npr = np.random
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
import tensorflow as tf
#from six.moves import cPickle
#from collect_samples import CollectSamples
#from get_true_action import GetTrueAction
import os
#import copy
from helper_funcs import create_env
#from helper_funcs import perform_rollouts
#from helper_funcs import add_noise
#from feedforward_network import feedforward_network
#from helper_funcs import visualize_rendering
import argparse

#TRPO things
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from rllab.misc.instrument import run_experiment_lite
from data_manipulation import from_observation_to_usablestate, get_indices, generate_training_data_inputs, generate_training_data_outputs
#from dynamics_model import Dyn_Model
import yaml
from bw_transition_op import Bw_Trans_Model
import theano.tensor as TT
import lasagne
import theano
#import numpy
from logger import Logger as hist_logging

def zero_mean_unit_std(dataX):
    mean_x = np.mean(dataX, axis = 0)
    dataX = dataX - mean_x
    std_x = np.std(dataX, axis = 0)
    dataX = np.nan_to_num(dataX/std_x)
    return dataX, mean_x, std_x


def run_task(v):
        env, _ = create_env(v["which_agent"])
        fw_learning_rate = v['fw_learning_rate'] # 0.0005!

        yaml_path = os.path.abspath('yaml_files/'+v['yaml_file']+'.yaml')
        assert(os.path.exists(yaml_path))
        with open(yaml_path, 'r') as f:
            params = yaml.load(f)
        num_fc_layers = params['dyn_model']['num_fc_layers']
        depth_fc_layers = params['dyn_model']['depth_fc_layers']
        batchsize = params['dyn_model']['batchsize']
        lr = params['dyn_model']['lr']
        print_minimal= v['print_minimal']
        nEpoch = params['dyn_model']['nEpoch']
        save_dir = '/data/milatmp1/goyalani/fwbw_icml_2018/' + v['exp_name']
        inputSize = env.spec.action_space.flat_dim + env.spec.observation_space.flat_dim
        outputSize = env.spec.observation_space.flat_dim

        #Initialize the forward policy
        policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(64, 64))
                 #learn_std=False, #v['learn_std'],
                 #adaptive_std=False, #v['adaptive_std'],
                 #output_gain=1, #v['output_gain'],
                 #init_std=1) #v['polic)
        baseline = LinearFeatureBaseline(env_spec=env.spec)


        #Update function for the forward policy (immitation learning loss!)
        fwd_obs = TT.matrix('fwd_obs')
        fwd_act_out = TT.matrix('act_out')
        policy_dist = policy.dist_info_sym(fwd_obs)
        fw_loss = -TT.sum(policy.distribution.log_likelihood_sym(fwd_act_out, policy_dist))
        fw_params = policy.get_params_internal()
        fw_update = lasagne.updates.adam(fw_loss, fw_params, learning_rate=fw_learning_rate)
        fw_func = theano.function([fwd_obs, fwd_act_out], fw_loss,
                                   updates=fw_update, allow_input_downcast=True)
        hist_logger = hist_logging(v['yaml_file'])

        optimizer_params = dict(base_eps=1e-5)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(save_dir+'/losses')
            os.makedirs(save_dir+'/models')
            os.makedirs(save_dir+'/saved_forwardsim')
            os.makedirs(save_dir+'/saved_trajfollow')
            os.makedirs(save_dir+'/training_data')

        x_index, y_index, z_index, yaw_index,\
        joint1_index, joint2_index, frontleg_index,\
        frontshin_index, frontfoot_index, xvel_index, orientation_index = get_indices(v['which_agent'])
        dyn_model = Bw_Trans_Model(inputSize, outputSize, env, v, lr, batchsize,
                                   v['which_agent'], x_index, y_index, num_fc_layers,
                                   depth_fc_layers, print_minimal)


        for outer_iter in range(1, v['outer_iters']):

            algo  = TRPO(
                    env=env,
                    policy=policy,
                    baseline=baseline,
                    batch_size=v["batch_size"],
                    max_path_length=v["steps_per_rollout"],
                    n_itr=v["num_trpo_iters"],
                    discount=0.995,
                    optimizer=v["ConjugateGradientOptimizer"](hvp_approach=v["FiniteDifferenceHvp"](**optimizer_params)),
                    step_size=0.05,
                    plot_true=True
                    )
            all_paths = algo.train()

            #Collect the trajectories, using these trajectories which leads to high value states
            # learn a backwards model!
            observations_list = []
            actions_list = []
            rewards_list = []
            returns_list = []
            for indexing in all_paths:
                for paths in indexing:
                    observations = []
                    actions = []
                    returns = []
                    reward_for_rollout = 0
                    for i_ in range(len(paths['observations'])):
                        #since, we are building backwards model using trajectories,
                        #so, reversing the trajectories.
                        index_ = len(paths['observations']) - i_ - 1
                        observations.append(paths['observations'][index_])
                        actions.append(paths['actions'][index_])
                        returns.append(paths['returns'][index_])
                        reward_for_rollout += paths['rewards'][index_]
                        #if something_ == 1:
                        #    actions_bw.append(path['actions'][::-1])
                        #    observations_bw.append(path['observations'][::-1])
                    observations_list.append(observations)
                    actions_list.append(actions)
                    rewards_list.append(reward_for_rollout)
                    returns_list.append(returns)

            hist_logger.log_scalar(save_dir, np.sum(rewards_list)/len(rewards_list), outer_iter * v["num_trpo_iters"] )
            selected_observations_list = []
            selected_observations_list_for_state_seletection = []
            selected_actions_list = []
            selected_returns_list = []


            #Figure out how to build the backwards model.
            #Conjecture_1
            #------- Take quantile sample of trajectories which recieves highest cumulative rewards!

            number_of_trajectories = int(np.floor(v['top_k_trajectories'] * len(rewards_list)/100))
            rewards_list_np = np.asarray(rewards_list)
            trajectory_indices = rewards_list_np.argsort()[-number_of_trajectories:][::-1]
            for index_ in range(len(trajectory_indices)):
                selected_observations_list.append(observations_list[trajectory_indices[index_]])
                selected_actions_list.append(actions_list[trajectory_indices[index_]])

            selected_observations_list_for_state_selection = []
            number_of_trajectories = int(np.floor(v['top_k_trajectories_state_selection'] * len(rewards_list)/100))
            rewards_list_np = np.asarray(rewards_list)
            trajectory_indices = rewards_list_np.argsort()[-number_of_trajectories:][::-1]
            for index_ in range(len(trajectory_indices)):
                selected_observations_list_for_state_seletection.append(observations_list[trajectory_indices[index_]])
                selected_returns_list.append(returns_list[trajectory_indices[index_]])

            #Figure out from where to start the backwards model.
            #Conjecture_1
            #------ Take quantile sample of high value states, and start the backwards model from them!
            #which amounts to just taking a non parametric buffer of high values states, which should be
            #fine!

            if v['use_good_trajectories'] == 1:
                returns_list = selected_returns_list
                observations_list = selected_observations_list_for_state_selection

            flatten_ret_list = np.asarray(returns_list).flatten()
            flatten_obs_list = np.vstack(np.asarray(observations_list))
            number_of_bw_samples = int(np.floor(v['top_k_bw_samples'] * len(flatten_ret_list)/100))
            samples_indices = flatten_ret_list.argsort()[-number_of_bw_samples:][::-1]
            bw_samples = []
            for bw_index in range(len(samples_indices)):
                bw_samples.append(flatten_obs_list[samples_indices[bw_index]])



            #Not all parts of the state are actually used.
            states = from_observation_to_usablestate(selected_observations_list, v["which_agent"], False)
            controls = selected_actions_list
            dataX , dataY = generate_training_data_inputs(states, controls)
            states = np.asarray(states)
            dataZ = generate_training_data_outputs(states, v['which_agent'])

            #every component (i.e. x position) should become mean 0, std 1
            dataX, mean_x, std_x = zero_mean_unit_std(dataX)
            dataY, mean_y, std_y = zero_mean_unit_std(dataY)
            dataZ, mean_z, std_z = zero_mean_unit_std(dataZ)

            ## concatenate state and action, to be used for training dynamics
            inputs = np.concatenate((dataX, dataY), axis=1)
            outputs = np.copy(dataZ)
            assert inputs.shape[0] == outputs.shape[0]

            if v['num_imagination_steps'] == 10:
                nEpoch = 20
            elif v['num_imagination_steps'] == 50:
                nEpoch = 20
            elif v['num_imagination_steps'] == 100:
                nEpoch = 30
            else:
                nEpoch = 20

            nEpoch = v['nEpoch']

            training_loss = dyn_model.train(inputs, outputs, inputs, outputs, nEpoch, save_dir, 1)
            print("Training Loss for Backwards model", training_loss)

            if v['running_baseline'] == False:
                for goal_ind in range(min(v['fw_iter'], len(bw_samples))):
                    #train the backwards model
                    #Give inital state, perform rollouts from backwards model.Right now, state is random, but it should
                    #be selected from some particular list
                    forwardsim_x_true=bw_samples[goal_ind]
                    state_list, action_list = dyn_model.do_forward_sim(forwardsim_x_true, v['num_imagination_steps'], False, env, v['which_agent'],
                                                                       mean_x, mean_y, mean_z, std_x, std_y, std_z)

                    #Incorporate the backwards trace into model based system.
                    fw_func(np.vstack(state_list), np.vstack(action_list))
                    #print("Immitation Learning loss", loss)
            else:
                print('running TRPO baseline')



##########################################
##########################################

#ARGUMENTS TO SPECIFY
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default='0')
parser.add_argument('--nEpoch', type=int, default='20')
parser.add_argument('--use_good_trajectories', type=int, default='1')
parser.add_argument('--top_k_trajectories_state_selection', type=int, default='10')
parser.add_argument('--bw_model_hidden_size', type=int, default='64')
parser.add_argument('--policy_variance', type=int, default='0')
parser.add_argument('--num_trpo_iters', type=int, default='5')
parser.add_argument('--running_baseline', type=bool, default=False)
parser.add_argument('--outer_iters', type=int, default=2500)
parser.add_argument('--fw_iter', type=int, default=1)
parser.add_argument('--top_k_trajectories', type=int, default=10)
parser.add_argument('--top_k_bw_samples', type=int, default=1)
parser.add_argument('--num_imagination_steps', type=int, default=20)
parser.add_argument('--fw_learning_rate', type=float, default='0.0005')
parser.add_argument('--bw_learning_rate', type=float, default='0.0001')
parser.add_argument('--steps_per_rollout', type=int, default='1000')
parser.add_argument('--save_trpo_run_num', type=int, default='1')
parser.add_argument('--which_agent', type=int, default= 2)
parser.add_argument('--num_workers_trpo', type=int, default=2)
parser.add_argument('--yaml_file', type=str, default='ant_forward')
parser.add_argument('--print_minimal', action="store_true", dest='print_minimal', default=False)
args = parser.parse_args()

batch_size = 50000

steps_per_rollout = args.steps_per_rollout
if(args.which_agent==1):
	num_trpo_iters = 2500
if(args.which_agent==2):
	steps_per_rollout=333
	num_trpo_iters = 500
if(args.which_agent==4):
	num_trpo_iters= 2500
if(args.which_agent==6):
	num_trpo_iters= 2000

num_trpo_iters = args.num_trpo_iters

if args.policy_variance == 0:
    bw_variance_learn = False
else:
    bw_variance_learn = True

##########################################
##########################################

# set tf seed
npr.seed(args.seed)
tf.set_random_seed(args.seed)
run_experiment_lite(run_task, plot=True, snapshot_mode="all", use_cloudpickle=True,
                    n_parallel=str(args.num_workers_trpo),
                    exp_name='agent_'+ str(args.which_agent)+'_seed_'+str(args.seed)+'_mf'+ '_run'+ str(args.save_trpo_run_num) + '_trpo_inner_iters_'+ str(args.num_trpo_iters) + '_fw_lr_' + str(args.fw_learning_rate) + '_bw_lr_' + str(args.bw_learning_rate) + '_num_immi_updates_' + str(args.fw_iter) + '_bw_rolls_' + str(args.num_imagination_steps) + '_top_k_trajectories_' + str(args.top_k_trajectories) + '_top_k_bw_samples_' + str(args.top_k_bw_samples) + '_running_baseline_' +
                    str(args.running_baseline) + '_bw_variance_'+ str(bw_variance_learn) + '_bw_hidden_size_' + str(args.bw_model_hidden_size) + '_Epoch_' + str(args.nEpoch) + '_use_good_trajectories_' + str(args.use_good_trajectories),
                    variant=dict(batch_size=batch_size,
                    which_agent=args.which_agent,
                    yaml_file = args.yaml_file,
                    fw_learning_rate = args.fw_learning_rate,
                    outer_iters = args.outer_iters,
                    fw_iter = args.fw_iter,
                    top_k_trajectories = args.top_k_trajectories,
                    top_k_bw_samples = args.top_k_bw_samples,
                    num_imagination_steps = args.num_imagination_steps,
                    bw_learning_rate = args.bw_learning_rate,
                    print_minimal = args.print_minimal,
                    steps_per_rollout=steps_per_rollout,
                    num_trpo_iters=num_trpo_iters,
                    bw_model_hidden_size = args.bw_model_hidden_size,
                    running_baseline = args.running_baseline,
                    FiniteDifferenceHvp=FiniteDifferenceHvp,
                    bw_variance_learn = bw_variance_learn,
                    nEpoch = args.nEpoch,
                    use_good_trajectories = args.use_good_trajectories,
                    top_k_trajectories_state_selection = args.top_k_trajectories_state_selection,
                    ConjugateGradientOptimizer=ConjugateGradientOptimizer))
