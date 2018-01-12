
import numpy as np
import numpy.random as npr
#import tensorflow as tf
import time
import math
import theano
import lasagne
#from feedforward_network import feedforward_network
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from rllab.envs.env_spec import EnvSpec
from rllab.spaces import Box
import theano.tensor as TT

class Bw_Trans_Model:

    def __init__(self, inputSize, outputSize, env, v, learning_rate, batchsize, which_agent, x_index, y_index,
                num_fc_layers, depth_fc_layers, mean_x, mean_y, mean_z, std_x, std_y, std_z, tf_datatype, print_minimal):

        #init vars
        #self.sess = sess
        self.batchsize = batchsize
        self.which_agent = which_agent
        self.x_index = x_index
        self.y_index = y_index
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.mean_z = mean_z
        self.std_x = std_x
        self.std_y = std_y
        self.std_z = std_z
        self.print_minimal = print_minimal

        LOW = -1000000
        HIGH = 1000000
        self.act_dim = env.spec.action_space.flat_dim
        self.obs_dim = env.spec.observation_space.flat_dim
        obs_to_act_spec = env.spec
        obsact_to_obs_spec = EnvSpec(observation_space=Box(LOW, HIGH, shape=(self.obs_dim + self.act_dim,)),
                                            action_space=Box(LOW, HIGH, shape=(self.obs_dim,)))

        self.bw_act_pol = GaussianMLPPolicy(
         env_spec=obs_to_act_spec,
         hidden_sizes=(32, 32),
         learn_std=False, #v['learn_std'],
         adaptive_std=False, #v['adaptive_std'],
         std_hidden_sizes=(16, 16),
         output_gain=1, #v['output_gain'],
         init_std=1, #v['policy_init_std'],
         )

        self.bw_obs_pol = GaussianMLPPolicy(
         env_spec=obsact_to_obs_spec,
         hidden_sizes=(32, 32),
         learn_std=False, #['learn_std'],
         adaptive_std=False, #v['adaptive_std'],
         std_hidden_sizes=(16, 16),
         output_gain=1, #Falsev['output_gain'],
         init_std=1, #Falsev['policy_init_std'],
         )

        self.obs_in = TT.matrix('obs_in')
        self.obsact_in = TT.matrix('obsact_in')
        self.act_out = TT.matrix('act_out')
        self.diff_out = TT.matrix('diff_out')

        bw_learning_rate = 0.005
        self.bw_act_dist = self.bw_act_pol.dist_info_sym(self.obs_in)
        self.bw_obs_dist = self.bw_obs_pol.dist_info_sym(self.obsact_in)
        self.bw_act_loss = -TT.sum(self.bw_act_pol.distribution.log_likelihood_sym(self.act_out, self.bw_act_dist))
        bw_obs_loss = -TT.sum(self.bw_obs_pol.distribution.log_likelihood_sym(self.diff_out, self.bw_obs_dist))

        bw_act_params = self.bw_act_pol.get_params_internal()
        bw_obs_params = self.bw_obs_pol.get_params_internal()
        #bw_params = bw_act_params + bw_obs_params
        bw_s_to_a_update = lasagne.updates.adam(self.bw_act_loss, bw_act_params,
                        learning_rate=bw_learning_rate)
        bw_sa_to_s_update = lasagne.updates.adam(bw_obs_loss, bw_obs_params,
                        learning_rate=bw_learning_rate)

        self.bw_act_train = theano.function([self.obs_in, self.act_out],self.bw_act_loss,
                       updates=bw_s_to_a_update, allow_input_downcast=True)
        self.bw_obs_train = theano.function([self.obsact_in, self.diff_out], bw_obs_loss,
                        updates=bw_sa_to_s_update, allow_input_downcast=True)


    def train(self, dataX, dataZ, dataX_new, dataZ_new, nEpoch, save_dir, fraction_use_new):

        #init vars
        start = time.time()
        training_loss_list = []
        nData_old = dataX.shape[0]
        num_new_pts = dataX_new.shape[0]

        #how much of new data to use per batch
        if(num_new_pts<(self.batchsize*fraction_use_new)):
            batchsize_new_pts = num_new_pts #use all of the new ones
        else:
            batchsize_new_pts = int(self.batchsize*fraction_use_new)

        #how much of old data to use per batch
        batchsize_old_pts = int(self.batchsize- batchsize_new_pts)

        #training loop
        for i in range(nEpoch):

            #reset to 0
            avg_loss=0
            num_batches=0

            if(batchsize_old_pts>0):
                print("nothing is going on")

            #train completely from new set
            else:
                for batch in range(int(math.floor(num_new_pts / batchsize_new_pts))):

                    #walk through the shuffled new data
                    dataX_batch = dataX_new[batch*batchsize_new_pts:(batch+1)*batchsize_new_pts, :]
                    dataZ_batch = dataZ_new[batch*batchsize_new_pts:(batch+1)*batchsize_new_pts, :]

                    data_x = dataX_batch[:,0:self.obs_dim]
                    data_y = dataX_batch[:, self.obs_dim:]

                    loss = self.bw_act_train(data_x, data_y)
                    bw_obs_losses = self.bw_obs_train(dataX_batch, dataZ_batch)

                    training_loss_list.append(loss)
                    avg_loss+= bw_obs_losses#[0]
                    num_batches+=1

                #shuffle new dataset after an epoch (if training only on it)
                p = npr.permutation(dataX_new.shape[0])
                dataX_new = dataX_new[p]
                dataZ_new = dataZ_new[p]

            #save losses after an epoch
            np.save(save_dir + '/training_losses.npy', training_loss_list)
            if(not(self.print_minimal)):
                if((i%10)==0):
                    print("\n=== Epoch {} ===".format(i))
                    print ("loss: ", avg_loss/num_batches)

        if(not(self.print_minimal)):
            print ("Training set size: ", (nData_old + dataX_new.shape[0]))
            print("Training duration: {:0.2f} s".format(time.time()-start))


        #done
        return (avg_loss/num_batches)#, old_loss, new_loss


    #multistep prediction using the learned dynamics model at each step
    def do_forward_sim(self, forwardsim_x_true, num_step, many_in_parallel, env_inp, which_agent):

        #init vars
        state_list = []
        action_list = []
        if(many_in_parallel):
            #init vars
            print("Future work..")
        else:
            curr_state = np.copy(forwardsim_x_true) #curr state is of dim NN input
            for i in range(num_step):
                curr_state_preprocessed = curr_state - self.mean_x
                curr_state_preprocessed = np.nan_to_num(curr_state_preprocessed/self.std_x)
                action = self.bw_act_pol.get_action(curr_state_preprocessed)[0]
                action_ = action * self.std_y + self.mean_y
                state_difference = self.bw_obs_pol.get_action(np.concatenate((curr_state_preprocessed, action)))[0]
                state_differences= (state_difference*self.std_z)+self.mean_z
                next_state = curr_state + state_differences
                #copy the state info
                curr_state= np.copy(next_state)
                state_list.append(np.copy(curr_state))
                action_list.append(np.copy(action_))

        return state_list, action_list
