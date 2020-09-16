import numpy as np
import scipy
from scipy.integrate import RK45




class Simulator:
    def __init__(self):
        self.laststate = None
        self.current_action = None
        self.steps = 0
        self.time_span = 10           # 20 seconds for each iteration
        self.number_iterations = 100  # 100 iterations for each step
        self.integrator = None

        #system constants
        self.M=1.0
        self.m=0.1
        self.g=9.8
        self.l=0.2

        #system parameters
        self.J=1/3*self.m*pow(self.l,2)
        self.A=np.array([[0,1,0,0],[0,0,self.m*self.m*self.g*self.l*self.l/(self.J*(self.M+self.m)+self.M*self.m*self.l*self.l),0],[0,0,0,1],[0,0,self.m*self.g*self.l*(self.M+self.m)/(self.J*(self.M+self.m)+self.M*self.m*self.l*self.l),0]])
        self.B=np.array([[0],[(self.J+self.m*self.l*self.l)/(self.J*(self.M+self.m)+self.M*self.m*self.l*self.l)],[0],[self.m*self.l/(self.J*(self.M+self.m)+self.M*self.m*self.l*self.l)]])
        self.H=np.identity(4)
        self.mean=[0,0,0,0]
        self.cov=[[0.1,0,0,0],[0,0.1,0,0],[0,0,0.1,0],[0,0,0,0.1]]

    def reset_start_pos(self, state_vector):
        x1, x2,x3,x4 = state_vector[0], state_vector[1],state_vector[2],state_vector[3]
        self.last_state = np.array([x1, x2,x3,x4])
        self.current_action = np.zeros(1)
        self.integrator = self.scipy_integration(self.simulate, self.get_state(), t_bound=self.time_span)

    def step(self, u):
        self.current_action = u #np.array([u])
        while not (self.integrator.status == 'finished'):
            self.integrator.step()
        print("hi")
        self.last_state = self.integrator.y #?
        self.integrator = self.scipy_integration(self.simulate, self.get_state(), t0=self.integrator.t, t_bound=self.integrator.t+self.time_span)
        return self.last_state

    def simulate(self, t,states):
        """
        :param local_states: Space state
        :return df_local_states
        """
        x1 = states[0]
        x2 = states[1]
        x3 = states[2]
        x4 = states[3]
        x=np.array([[x1],[x2],[x3],[x4]])


        # Derivative function

        dx_dt=self.A*x+self.B*self.current_action+np.random.multivariate_normal(self.mean,self.cov,1)

        return dx_dt

    def scipy_integration(self, fun, x0, t0=0, t_bound=10):
        return RK45(fun, t0, x0,t_bound, rtol=self.time_span/self.number_iterations, atol=1e-4)


    def get_state(self):
        return self.last_state




