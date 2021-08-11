# MixReg
Stable-Baselines Implementation of MixReg regularization technique for PPO2
https://arxiv.org/abs/2010.10814
# Use:
  from MIXREG_ImpalaCnn import ImpalaCnn
  from mixreg import MIXREG
  
  # Use exactly the same as PPO2
  model = MIXREG(ImpalaCnn, env, verbose=0, n_steps = 2048, nminibatches=8)
