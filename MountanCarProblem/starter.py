import sys

import pylab as plb
import numpy as np
import mountaincar


class DummyAgent:
    """A not so good agent for the mountain-car task.
    """

    def __init__(self, mountain_car=None, parameter1=3.0):

        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        self.parameter1 = parameter1

        self.learning_rate = 0.1
        self.eta = 0.95

        self.x_space = np.linspace(-150, 30, 20)

        self.speed_space = np.linspace(-15, 15, 20)

        self.eligibility_traces = np.zeros((3, 400))

        self.sigma_x = np.abs(self.x_space[0] - self.x_space[1])
        self.sigma_speed = np.abs(self.speed_space[0] - self.speed_space[1])

        self.input_layer = np.array(np.meshgrid(self.x_space, self.speed_space)).reshape(2, 400)
        self.input_weight = np.random.rand(3, 400)

    def visualize_trial(self, n_steps=5000):
        """Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """

        # prepare for the visualization
        plb.ion()
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()

        # make sure the mountain-car is reset
        self.mountain_car.reset()

        for n in range(n_steps):
            print('\rt =', self.mountain_car.t)
            sys.stdout.flush()

            # choose an action
            r, a = self.get_action_from_policy()
            self.mountain_car.apply_force(a)
            q_value = self.get_q_value(a)

            # simulate the time step
            self.mountain_car.simulate_timesteps(100, 0.01)

            _, a2 = self.get_action_from_policy()
            q_value2 = self.get_q_value(a2)

            delta = self.mountain_car.R - (q_value - self.eta*q_value2)

            self.eligibility_traces *= self.eta * 1
            for i in range(400):
                self.eligibility_traces[a+1, i] += r[i]

            self.input_weight += delta * self.learning_rate * self.eligibility_traces

            # update the visualization
            mv.update_figure()
            plb.show()
            plb.pause(0.00000001)

            # check for rewards
            if self.mountain_car.R > 0.0:
                print("\rreward obtained at t = ", self.mountain_car.t)
                break

    def get_q_value(self, action):
        r = np.zeros(400)
        x = self.mountain_car.x
        x_d = self.mountain_car.x_d

        for i in range(400):
            center = self.input_layer[:, i]
            r[i] = np.exp(-((center[0] - x) / self.sigma_x)**2 - ((center[1] - x_d) / self.sigma_speed)**2)

        return np.sum(np.multiply(self.input_weight[action + 1, :], r))

    def get_action_from_policy(self):
        x = self.mountain_car.x
        x_d = self.mountain_car.x_d

        r = np.zeros(400)
        for i in range(400):
            center = self.input_layer[:, i]
            r[i] = np.exp(-((center[0] - x) / self.sigma_x)**2 - ((center[1] - x_d) / self.sigma_speed)**2)

        q = np.sum(np.multiply(self.input_weight, np.array([r, r, r])), axis=1)
        denominator = np.sum(np.exp(q))

        p = np.exp(q) / denominator

        random = np.random.rand()

        #return np.argmin(np.abs(p - random)) - 1
        return r, np.argmax(p) - 1


if __name__ == "__main__":
    d = DummyAgent()
    d.visualize_trial()
    plb.show()
