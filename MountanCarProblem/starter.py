import sys
import pylab as plb
import numpy as np
import matplotlib.pyplot as mplb
import math
import mountaincar
from multiprocessing import Pool, cpu_count


class SARSALambdaAgent:
    """
    Agent learning from the SARSA algorithm in combination with a neural network
    """

    def __init__(self, mountain_car=None, parameter1=3.0):

        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        self.parameter1 = parameter1

        # Hyper parameters
        self.learning_rate = 0.001
        self.reward_factor = 0.95
        self.trace_decay = 0.5
        self.exploration_temp = 1
        self.size = 20

        # Initialize the different spaces
        self.x_space = np.linspace(-150, 30, self.size)
        self.speed_space = np.linspace(-15, 15, self.size)

        self.eligibility_traces = np.zeros((3, self.size ** 2))

        self.sigma_x = np.abs(self.x_space[0] - self.x_space[1])
        self.sigma_speed = np.abs(self.speed_space[0] - self.speed_space[1])

        self.input_layer = np.array(np.meshgrid(self.x_space, self.speed_space)).reshape(2, self.size ** 2)
        self.input_weight = np.random.rand(3, self.size ** 2)
        # self.input_weight = np.ones((3, self.size**2))
        # self.input_weight = np.zeros((3, self.size**2))

    def trial(self, max_steps=20000, visualize=False, logs=False, expl_temp_decay=False):
        """
        Execute the algorithm until the agent reaches the top of the hill or the max
        """

        if expl_temp_decay:
            self.exploration_temp = 10 ** 10

        # prepare for the visualization if required
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        if visualize:
            mv.create_figure(max_steps, max_steps)
            plb.ion()
            plb.draw()

        # make sure the mountain-car is reset
        self.mountain_car.reset()

        r, a = self.get_action_from_policy()
        while self.mountain_car.R <= 0.0 and self.mountain_car.t < max_steps:

            # choose an action
            self.mountain_car.apply_force(a)
            q_value = self.get_q_value(a)

            # simulate the time step
            self.mountain_car.simulate_timesteps(100, 0.01)

            r2, a2 = self.get_action_from_policy()
            q_value2 = self.get_q_value(a2)

            delta = self.mountain_car.R - (q_value - self.reward_factor * q_value2)

            self.eligibility_traces *= self.reward_factor * self.trace_decay
            for k in range(self.size ** 2):
                self.eligibility_traces[a + 1, k] += r[k]

            self.input_weight += delta * self.learning_rate * self.eligibility_traces

            a = a2
            r = r2

            # update the visualization
            if visualize:
                mv.update_figure()
                plb.show()
                plb.pause(0.00000001)

            if expl_temp_decay:
                # Decrease the value but set the min to 1 as the property is symmetric around 1
                self.exploration_temp = max(1, self.exploration_temp * 0.8)

        if logs:
            if self.mountain_car.t >= max_steps:
                print("Maximum step reached")
            else:
                print("\rreward obtained at t = ", self.mountain_car.t)

        return self.mountain_car.t

    def get_nn_activity(self, x=None, x_d=None):
        """
        Return the values of the neural activity for the current state or the specified in parameters
        :param x: Position
        :param x_d: Velocity
        :return:
        """
        r = np.zeros(self.size ** 2)
        x = self.mountain_car.x if x is None else x
        x_d = self.mountain_car.x_d if x_d is None else x_d

        for k in range(self.size ** 2):
            center = self.input_layer[:, k]
            r[k] = np.exp(-((center[0] - x) / self.sigma_x) ** 2 - ((center[1] - x_d) / self.sigma_speed) ** 2)

        return r

    def get_q_value(self, action, r=None):
        """
        Return the Q-Value of the action and the specified neural activity or the current state
        :param action: Action of the agent
        :param r: NNetwork activity
        :return:
        """
        r = self.get_nn_activity() if r is None else r
        return np.sum(np.multiply(self.input_weight[action + 1, :], r))

    def get_action_from_policy(self):
        """
        Choose an action according to the policy and the current state
        :return:
        """
        r = self.get_nn_activity()

        q = np.sum(np.multiply(self.input_weight, np.array([r, r, r])), axis=1)
        denominator = np.sum(np.exp(q / self.exploration_temp))

        prob = np.exp(q / self.exploration_temp) / denominator

        random = np.random.rand()
        action = 0
        offset = 0
        for index, prob in enumerate(prob):
            if offset <= random < offset + prob:
                action = index - 1
                break
            else:
                offset += prob

        return r, action

    def plot_vector_field(self):
        """
        Display the vector field of the current neural network state
        :return:
        """
        X, Y = np.meshgrid(self.x_space, self.speed_space)
        U = np.zeros((20, 20))
        V = np.zeros((20, 20))

        for k in range(self.size ** 2):
            state = self.input_layer[:, k]
            x_coord = k % self.size
            speed_coord = math.floor(k / self.size)

            r = self.get_nn_activity(x=state[0], x_d=state[1])
            q_values = [self.get_q_value(a - 1, r=r) for a in range(3)]
            a = np.argmax(q_values) - 1

            U[x_coord, speed_coord] = a

        mplb.quiver(X, Y, U, V, U, scale=40)
        mplb.ylabel("Speed [m/s]")
        mplb.xlabel("Position [m]")
        mplb.show()


def launch_agent(trials):
    """
    Launch a learning procedure in a separate process for a new agent
    :param trials: Number of trials
    :return:
    """

    print('Start agent')

    a = SARSALambdaAgent()
    results = np.zeros(trials)

    for i in range(trials):
        t = a.trial(visualize=False)
        print('Missing agent trials', trials - i - 1)

        results[i] = t

    return results


def moving_average(data, n=3):
    ma = []
    for i, x in enumerate(data):
        ma.append(np.mean(data[max(0, i - n):i + n + 1]))

    return ma


if __name__ == "__main__":

    cmd = 0
    if len(sys.argv) > 1:
        cmd = int(sys.argv[1])

    # np.random.seed(42)

    if cmd == 0:
        num_agents = 20
        num_trials = 50

        num_procs = max(1, (cpu_count() - 1))  # Prevent overload of the computer
        pool = Pool(processes=num_procs)

        args = np.ones(num_agents, dtype=int) * num_trials

        times = pool.map(launch_agent, args)

        # Mean over the agents
        times = np.mean(np.array(times), axis=0)
        ma = moving_average(times)

        mplb.plot(range(num_trials), times, '', range(num_trials), ma)
        mplb.show()

    if cmd == 1:
        agent = SARSALambdaAgent()
        for l in range(0):
            print('Trial', l + 1)
            agent.trial(logs=True)

        agent.plot_vector_field()
