from stable_baselines.common.tf_layers import linear
from stable_baselines.common.policies import ActorCriticPolicy
import numpy as np
import tensorflow as tf


def build_impala_cnn(scaled_images, depths=[16,32,32], dense=tf.layers.dense, use_bn=False, randcnn=False, **conv_kwargs):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """

    layer_num = 0

    def get_layer_num_str():
        nonlocal layer_num
        num_str = str(layer_num)
        layer_num += 1
        return num_str

    def conv_layer(out, depth):
        out = tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + get_layer_num_str())
        if use_bn:
            # always use train mode BN
            out = tf.layers.batch_normalization(out, center=True, scale=True, training=True)
        return out

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out
    
    #scaling (don't need since stable-baselines scales for us)
    #out = tf.cast(unscaled_images, tf.float32) / 255.
    out = scaled_images

    if randcnn:
        out = tf.layers.conv2d(out, 3, 3, padding='same', 
                               kernel_initializer=tf.initializers.glorot_normal(), 
                               trainable=False, 
                               name='randcnn')

    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = dense(out, 256, activation=tf.nn.relu, name='layer_' + get_layer_num_str())

    return out



# ImpalaCNN Policy network with MIXREG
class ImpalaCnn(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, mix_alpha=0.2, **kwargs):
        super(ImpalaCnn, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)
        ###----MIXREG IMPLEMENTATION---###
        extra_tensors = {}
        COEFF = tf.placeholder(tf.float32, [None])
        INDICES = tf.placeholder(tf.int32, [None])
        OTHER_INDICES = tf.placeholder(tf.int32, [None])
        coeff = tf.reshape(COEFF, (-1, 1, 1, 1))
        extra_tensors['coeff'] = COEFF
        extra_tensors['indices'] = INDICES
        extra_tensors['other_indices'] = OTHER_INDICES
        self.__dict__.update(extra_tensors)
        ###-----------------------------###
        with tf.variable_scope("model", reuse=reuse):
            
            extracted_features = build_impala_cnn(self.processed_obs, **kwargs)

            pi_latent = vf_latent = extracted_features

            self._value_fn = linear(vf_latent, 'vf', 1)
            
            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()
    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
