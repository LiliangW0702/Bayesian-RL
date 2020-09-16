from gym import Env, spaces
import numpy as np
from shapely.geometry import LineString, Point
from Simulation import Simulator

class Linear_Env(Env):
    def __init__(self, type='continuous', action_dim = 1):
        self.type = type
        self.action_dim = action_dim
        self.observation_space = spaces.Box(low=np.array([-5.0, -10.0,-np.pi, -5.0*np.pi]), high=np.array([5.0, 10.0,np.pi, 5.0*np.pi]))
        self.init_space = spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([0, 0,np.pi , 0]))
        self.Linear_data = None
        self.last_pos = np.zeros(2) #only position, no velocities
        self.last_action = np.zeros(self.action_dim)
        self.simulator = Simulator()
        self.point_a = (0.0, 0.0)
        self.point_b = (2000, 0.0)
        self.max_x_episode = (5000, 0) #may need to change
        self.guideline = LineString([self.point_a, self.max_x_episode])
        self.start_pos = np.zeros(2)
        self.number_loop = 0  # loops in the screen -> used to plot
        #self.borders = [[0, 150], [2000, 150], [2000, -150], [0, -150]] #?



    def step(self, action): #here action is a scalar
        u=action
        state_prime = self.simulator.step(u) #here u is a scalar
        # convert simulator states into obervable states
        obs = self.convert_state(state_prime)
        # print('Observed state: ', obs)
        dn = self.end(state_prime=state_prime, obs=obs)
        reward = self.calculate_reward(obs=obs)
        self.last_pos = [state_prime[0],  state_prime[2]]
        self.last_action = np.array([u])
        info = dict() #dict() returns None
        return obs, reward, dn, info

    def convert_state(self, state):
        """
        This function is to transform the simulator space-state to the environment space-state
        """
        observations = self.simulator.H*np.array([state[0], state[1], state[2], state[3]])
        return observations

    def calculate_reward(self, current_obs):
        Q=0.01*np.identity(4)
        R=0.01*0.025
        u=self.simulator.current_action #u=self.last_action
        u=np.array([u])
        x=np.array([current_obs[0],current_obs[1],current_obs[2],current_obs[3]])
        if not self.observation_space.contains(current_obs):
            print("state beyond bounds")
            return -1000
        else:
            reward=-(x.T*Q*x+u.T*R*u)
            return reward


    def end(self, state_prime, obs): #need to recheck
        if not self.observation_space.contains(obs):
            if not self.observation_space.contains(obs):
                print("\n Smashed")
            #if self.viewer is not None:
                #self.viewer.end_episode()
            if self.Linear_data is not None:
                if self.Linear_data.iterations > 0:
                    self.Linear_data.save_experiment(self.name_experiment) #？
            return True
        else:
            return False

    def set_init_space(self, low, high):
        self.init_space = spaces.Box(low=np.array(low), high=np.array(high))

    def reset(self):
        init = list(map(float, self.init_space.sample()))
        init_states = np.array([init[0], init[1], init[2], init[3]])
        self.simulator.reset_start_pos(init_states)
        self.last_pos = np.array([ init[0],  init[2]])
        print('Reseting position')
        state = self.simulator.get_state()
        #if self.viewer is not None:
            #self.viewer.end_episode()
        return self.convert_state(state)