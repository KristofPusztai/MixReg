# MixReg
Stable-Baselines Implementation of MixReg regularization technique for PPO2 (uses impala CNN as feature extractor as specified in paper)
https://arxiv.org/abs/2010.10814
# Use:
    from MIXREG_ImpalaCnn import ImpalaCnn
    from mixreg import MIXREG
  
    # Use exactly the same as PPO2
    model = MIXREG(ImpalaCnn, env, verbose=0, n_steps = 2048, nminibatches=8)
# Results from use:
![alt text](https://github.com/KristofPusztai/CS-W182-Final-Project/blob/master/Test_Reward.png?raw=true)

Source: https://github.com/KristofPusztai/CS-W182-Final-Project
