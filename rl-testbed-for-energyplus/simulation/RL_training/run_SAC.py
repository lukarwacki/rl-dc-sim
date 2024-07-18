# Imports
import sys
from stable_baselines3 import SAC 

# Custom imports
from simulation.helpers.gym_make_environment import make_env
from simulation.gym_energyplus.envs import EnergyPlusEnv # needed to register env.
from simulation.helpers.energyplus_util import energyplus_arg_parser
from simulation.helpers.run_common import train_agent, evaluate_agent

def create_SAC(env, args):
    """
    Create a SAC agent with the provided environment and hyperparameters.

    Parameters:
        env (gym.Env): The environment to train the agent on.
        args (argparse.Namespace): The command-line arguments containing hyperparameters.

    Returns:
        SAC: The created SAC agent.
    """

    # Check if schedules are used 
    if args.learning_rate_schedule == 'constant':
        learning_rate = args.learning_rate
    elif args.learning_rate_schedule == 'linear':
        learning_rate = lambda f: f * args.learning_rate
    else:
        raise ValueError(f"Invalid learning rate schedule: {args.learning_rate_schedule}, should be either 'constant' or 'linear'")

    agent = SAC(
        # General
        env=env, 
        verbose=1,
        tensorboard_log=env.tb_log_dir,        
        seed=args.seed,                 
        # Policy-related         
        policy=args.policy,        
        learning_rate=learning_rate,
        policy_kwargs=args.policy_kwargs,
        # RL-related
        gamma=args.gamma,
        batch_size=args.mini_batch_size,         
        ent_coef=args.ent_coef,                
        # SAC-related 
        tau=args.tau, 
        train_freq=1, 
        gradient_steps=1, 
        action_noise=None,
        buffer_size=args.buffer_size, 
        replay_buffer_class=None,
        replay_buffer_kwargs=None, 
        target_update_interval=1, 
        target_entropy='auto', 
        sde_sample_freq=-1, 
        use_sde_at_warmup=False, 
    )
    return agent

def main():
    """
    This function handles the training of the SAC agent on the EnergyPlus environment. 
    First arguments are parsed, and can be modified to the desired settings. Then, an 
    enviroment is created, the agent is trained, and finally the agent is evaluated.
    """
    # Parse arguments
    args = energyplus_arg_parser().parse_args()

    # Modify settings
    args.num_episodes = 5                           # Number of episodes to train the agent
    args.log_actions_states = False                 # Log the normalized actions and states during training (only for debugging purposes)
    args.evaluate_agent = True                      # Do an evaluation run on a new weather & IT load file after training
    args.seed = 1                                   # Seed for reproducibility
    args.mini_batch_size = 48                       # Size of the mini-batch for each training iteration of the NNs
    args.n_epochs = 25
    args.buffer_size = 128

    # Modify hyperparams (now set to tuned hyperparameters) taken from arXiv:2005.05719v2
    args.learning_rate = 7.3e-4                     # Learning rate of the NNs in the SAC
    args.learning_rate_schedule = 'constant'        # Learning rate schedule over the training iterations
    args.n_steps = 512                              # Number of steps to collect samples for each training iteration
    args.gamma = 0.98                               # Discount factor
    args.policy_kwargs = dict(net_arch=dict(pi=[28],qf=[18]))   # Architecture of the NNs 
    args.ent_coef = 'auto'                          # Entropy coefficient
    args.vf_coef = 0.7478320995825463               # Value function coefficient 
    args.tau = 0.02
    args.use_rms_prop = True
    args.use_sde = False

    args.GAE_lambda = 0.9946797892895495            # Factor for the Generalized Advantage Estimation
    args.clip_range = 0.4487117411242193            # Clipping range for the PPO
    args.clip_range_schedule = 'linear'             # Clipping range schedule over the training iterations
    args.clip_range_vf = None                       # Clipping range for the value function (StableBaselines specific PPO factor)

    # Reward settings
    args.use_reward_file = True                     # True if reward settings are defined here, False if reward settings are hardcoded in the environment (old version)
    args.reward_P_type = 'P_HVAC'                   # Type of reward for the power consumption, can be: 'P_HVAC', 'PUE', None
    args.reward_T_type = 'Gaussian'                 # Type of reward for the leaving CRAH temperature, can be: 'Gaussian', None
    args.penalty_T_type = 'ReLU'                    # Type of penalty for the leaving server temperature, can be: 'ReLU', 'ReLU2', 'Softplus', None
    args.penalty_a_fluctuation_type = 'Trapezoidal' # Type of penalty for the action fluctuation, can be: 'Trapezoidal', 'Linear', 'Quadratic', None
    args.lambda_T_r = .1                            # Weight of the reward for the leaving CRAH temperature
    args.lambda_T_p = .8                            # Weight of the penalty for the leaving server temperature
    args.fluct_T_sp = 0.1                           # Weight of the penalty for the chilled water setpoint fluctuation
    args.fluct_m_chill = 0.0                        # Weight of the penalty for the chilled water mass flow fluctuation
    args.fluct_m_air = 0.0                          # Weight of the penalty for the air mass flow fluctuation
    args.T_SP = 24.0                                # Setpoint for the leaving CRAH temperature     
    args.T_SP_bandwidth = 3.                        # Allowable temperature deviation for the leaving CRAH temperature setpoint
    args.T_constr = 30.                             # Maximum allowable leaving server air temperature
    args.action_bandwidth = 1.                      # Maximum allowable action fluctuation (for the trapezoidal penalty only)

    # Create a wrapped environment
    env = make_env(args)
    
    # Train the agent
    trained_agent = train_agent(env, args, create_SAC)
    
    # Evaluate the agent
    if args.evaluate_agent:
        rew_eval, _ = evaluate_agent(trained_agent, env.log_dir)

    # Close environment
    env.close()

    return trained_agent

if __name__ == '__main__':
    main()
    sys.exit()  # Exit the program