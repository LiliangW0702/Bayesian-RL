import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


#==================================================================
#System modelling
#==================================================================
class inverted_pendulum(object):

    def __init__(self):
        self.g = 9.8
        self.M = 1
        self.m = 0.1
        self.total_mass = self.m+ self.M
        self.l = 0.2  # actually half the pendulum's length
        self.J=self.m*np.square(self.l)/3.0
        self.den=self.J * (self.total_mass) + self.M * self.m * self.l * self.l
        self.A = np.array([[0, 1, 0, 0], [0, 0, -1.0*self.m * self.m * self.g * self.l * self.l / self.den, 0], [0, 0, 0, 1], [0, 0,self.m * self.g * self.l * self.total_mass/ self.den,0]])
        self.B = np.array([[0],[(self.J + self.m * self.l * self.l) / self.den],[0], [-1.0*self.m * self.l / self.den]])
        self.mean = [0, 0, 0, 0]
        self.cov = [[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1]]
        #self.state = np.array([[0],[0],[0],[0]])

    def dynamics(self, t, current_state):  # dx/dt=f(t,x)
        x, x_dot, theta, theta_dot = current_state
        force = policy(current_state)

        # ========================================================
        # dynamics
        s = np.array([[x], [x_dot], [theta], [theta_dot]])
        ds_dt = np.matmul(self.A, s) + self.B * force #+ np.random.multivariate_normal(self.mean, self.cov, 1)

        x_dot, xacc, theta_dot, thetaacc = ds_dt[0, 0], ds_dt[1, 0], ds_dt[2, 0], ds_dt[3, 0]
        return x_dot, xacc, theta_dot, thetaacc


def policy(current_state):
    K = [-10.0000, -8.9036, -45.7046, -7.6749]
    k1, k2, k3, k4 = K
    x, x_dot, theta, theta_dot = current_state
    u = -(k1 * x + k2 * x_dot + k3 * theta + k4 * theta_dot)
    return u

Inverted_Pendulum=inverted_pendulum()
print(Inverted_Pendulum.A)
print(Inverted_Pendulum.B)
t = np.linspace(0, 10, 2000)
#sol = solve_ivp(time_step(t,y,1,Inverted_Pendulum), [0, 10], [0, 0, 0])
#K=np.array([-6.3246,-6.8626,41.0496,6.9387]).reshape((-1,1)) #row vector
K=[-10.0000,  -8.9036,  -45.7046,   -7.6749]
integrator=solve_ivp(Inverted_Pendulum.dynamics,[0,10],[0,0,math.pi/3.0,0],t_eval=t)
#sol = odeint(time_step,[0,0,math.pi/3.0,0],t,args=(Inverted_Pendulum,))
sol=integrator.y
x=sol[0,:]
x_dot=sol[1,:]
theta=sol[2,:]
theta_dot=sol[3,:]

#plot ===========================================
plt.subplot(2,2,1)
plt.plot(t,x)
plt.title('cart position')
plt.xlabel('time/t')
plt.ylabel('meter')
#plt.show()
plt.subplot(2,2,2)
plt.plot(t,x_dot)
plt.title('cart velocity')
plt.xlabel('time/t')
plt.ylabel('m/s')
plt.subplot(2,2,3)
plt.plot(t,theta)
plt.title('pendulum angle')
plt.xlabel('time/t')
plt.ylabel('radian')
plt.subplot(2,2,4)
plt.plot(t,theta_dot)
plt.title('angular velocity')
plt.xlabel('time/t')
plt.ylabel('radian/s')
plt.show()
