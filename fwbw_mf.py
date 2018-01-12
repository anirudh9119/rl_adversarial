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
import numpy
def run_task(v):
        env, _ = create_env(v["which_agent"])
        fw_learning_rate = 0.005
        #Initialize the forward policy
        policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(64, 64))
        fwd_obs = TT.matrix('fwd_obs')
        fwd_act_out = TT.matrix('act_out')
        policy_dist = policy.dist_info_sym(fwd_obs)
        fw_loss = -TT.sum(policy.distribution.log_likelihood_sym(fwd_act_out, policy_dist))
        fw_params = policy.get_params_internal()
        fw_update = lasagne.updates.adam(fw_loss, fw_params, learning_rate=fw_learning_rate)
        fw_func = theano.function([fwd_obs, fwd_act_out], fw_loss,
                                   updates=fw_update, allow_input_downcast=True)

        baseline = LinearFeatureBaseline(env_spec=env.spec)
        optimizer_params = dict(base_eps=1e-5)
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
        save_dir = 'run_temp'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(save_dir+'/losses')
            os.makedirs(save_dir+'/models')
            os.makedirs(save_dir+'/saved_forwardsim')
            os.makedirs(save_dir+'/saved_trajfollow')
            os.makedirs(save_dir+'/training_data')


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

        #Collect the trajectories, using these trajectories which leads to high value states
        # learn a backwards model!
        all_paths = algo.train()
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


        #Not all parts of the state are actually used.
        states = from_observation_to_usablestate(observations_list, v["which_agent"], False)
        controls = actions_list
        dataX , dataY = generate_training_data_inputs(states, controls)
        states = np.asarray(states)
        dataZ = generate_training_data_outputs(states, v['which_agent'])

        #every component (i.e. x position) should become mean 0, std 1
        mean_x = np.mean(dataX, axis = 0)
        dataX = dataX - mean_x
        std_x = np.std(dataX, axis = 0)
        dataX = np.nan_to_num(dataX/std_x)

        mean_y = np.mean(dataY, axis = 0)
        dataY = dataY - mean_y
        std_y = np.std(dataY, axis = 0)
        dataY = np.nan_to_num(dataY/std_y)

        mean_z = np.mean(dataZ, axis = 0)
        dataZ = dataZ - mean_z
        std_z = np.std(dataZ, axis = 0)
        dataZ = np.nan_to_num(dataZ/std_z)

        ## concatenate state and action, to be used for training dynamics
        inputs = np.concatenate((dataX, dataY), axis=1)
        outputs = np.copy(dataZ)

        assert inputs.shape[0] == outputs.shape[0]
        inputSize = inputs.shape[1]
        outputSize = outputs.shape[1]
        x_index, y_index, z_index, yaw_index, joint1_index, joint2_index, frontleg_index, frontshin_index, frontfoot_index, xvel_index, orientation_index = get_indices(v['which_agent'])
        tf_datatype = tf.float64
        dyn_model = Bw_Trans_Model(inputSize, outputSize, env, v, lr, batchsize, v['which_agent'], x_index, y_index, num_fc_layers,
                              depth_fc_layers, mean_x, mean_y, mean_z, std_x, std_y, std_z, tf_datatype, print_minimal)

        #train the backwards model
        training_loss = dyn_model.train(inputs, outputs, inputs, outputs, nEpoch, save_dir, 1)
        print("Training Loss for Backwards model", training_loss)
        #Give inital state, perform rollouts from backwards model.
        forwardsim_x_true=numpy.random.rand(20)
        state_list, action_list = dyn_model.do_forward_sim(forwardsim_x_true, 20, False, env, v['which_agent'])

        #Incorporate the backwards trace into model based system.
        loss = fw_func(np.vstack(state_list), np.vstack(action_list))
        print("Immitation Learning loss", loss)



##########################################
##########################################

#ARGUMENTS TO SPECIFY
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default='0')
parser.add_argument('--steps_per_rollout', type=int, default='1000')
parser.add_argument('--save_trpo_run_num', type=int, default='1')
parser.add_argument('--which_agent', type=int, default= 2)
parser.add_argument('--num_workers_trpo', type=int, default=2)
parser.add_argument('--yaml_file', type=str, default='ant_forward')
parser.add_argument('--print_minimal', action="store_true", dest='print_minimal', default=False)
args = parser.parse_args()

batch_size = 50000

steps_per_rollout = args.steps_per_rollout
num_trpo_iters = 2500
if(args.which_agent==1):
	num_trpo_iters = 2500
if(args.which_agent==2):
	steps_per_rollout=333
	num_trpo_iters = 500
if(args.which_agent==4):
	num_trpo_iters= 2500
if(args.which_agent==6):
	num_trpo_iters= 2000

num_trpo_iters= 1

##########################################
##########################################

# set tf seed
npr.seed(args.seed)
tf.set_random_seed(args.seed)

run_experiment_lite(run_task, plot=True, snapshot_mode="all", use_cloudpickle=True,
                    n_parallel=str(args.num_workers_trpo),
                    exp_name='agent_'+ str(args.which_agent)+'_seed_'+str(args.seed)+'_mf'+ '_run'+ str(args.save_trpo_run_num),
                    variant=dict(batch_size=batch_size,
                    which_agent=args.which_agent,
                    yaml_file = args.yaml_file,
                    print_minimal = args.print_minimal,
                    steps_per_rollout=steps_per_rollout,
                    num_trpo_iters=num_trpo_iters,
                    FiniteDifferenceHvp=FiniteDifferenceHvp,
                    ConjugateGradientOptimizer=ConjugateGradientOptimizer))
