# MixReg
Stable-Baselines Implementation of MixReg regularization technique for PPO2 (uses impala CNN as feature extractor as specified in paper)
https://arxiv.org/abs/2010.10814

Note: Dependency on stable-baselines(2.10.1 at time of writing) python library, https://pypi.org/project/stable-baselines/
# Use:
    from MIXREG_ImpalaCnn import ImpalaCnn
    from mixreg import MIXREG
  
    # Use exactly the same as PPO2
    model = MIXREG(ImpalaCnn, env, verbose=0, n_steps = 2048, nminibatches=8)
# Results from use:
Performance of Impala CNN compared to Nature CNN feature extraction in base PPO2 model using FruitBot Environment:

![alt text](https://github.com/KristofPusztai/CS-W182-Final-Project/blob/master/fruitbot-impala_vs_nature.png?raw=true)

implementation of MixReg outperforms base PPO2 in terms of generalization ability on limited training levels for FruitBot Environment:

![alt text](https://github.com/KristofPusztai/CS-W182-Final-Project/blob/master/Test_Reward.png?raw=true)

![alt text](https://github.com/KristofPusztai/CS-W182-Final-Project/blob/master/mixregVppo.png)

Source: https://github.com/KristofPusztai/CS-W182-Final-Project
