# DDPG MUJOCO
##### Solving MuJoCo physics based environments with Deep Deterministic Policy Gradients.

Architecture is standard with hidden layers of size 400 and 300 for both actor and critic. L2 regularization is applied and hyperparameters are the same as DDPG paper. Temporarily correlated Ornstein–Uhlenbeck noise is applied during training. Environment finds a good policy for Hopper-v2 and HalfCheetah-v2. I will test with other environments and update this page. 

Hopper Weights: https://www.dropbox.com/sh/3hwpv1ggghbzu4l/AAAduhkKqD53rbuRuWHh7Io3a?dl=0