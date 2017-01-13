import sys
import pylab as plb
import numpy as np
import matplotlib.pyplot as mplb
import math
import mountaincar
from multiprocessing import Process, Queue, Value


class SARSALambdaAgent:
    """A not so good agent for the mountain-car task.
    """

    def __init__(self, mountain_car=None, parameter1=3.0):

        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        self.parameter1 = parameter1

        self.learning_rate = 0.01
        self.eta = 0.95
        self.nu = 0.5
        self.size = 20

        self.x_space = np.linspace(-150, 30, self.size)

        self.speed_space = np.linspace(-15, 15, self.size)

        self.eligibility_traces = np.zeros((3, self.size ** 2))

        self.sigma_x = np.abs(self.x_space[0] - self.x_space[1])
        self.sigma_speed = np.abs(self.speed_space[0] - self.speed_space[1])

        self.input_layer = np.array(np.meshgrid(self.x_space, self.speed_space)).reshape(2, self.size ** 2)
        self.input_weight = np.random.rand(3, self.size ** 2)

    def trial(self, max_steps=20000, visualize=False, logs=False):
        """
        Execute the algorithm until the agent reaches the top of the hill
        """

        # prepare for the visualization
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

            delta = self.mountain_car.R - (q_value - self.eta * q_value2)

            self.eligibility_traces *= self.eta * self.nu
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

        if logs:
            if self.mountain_car.t >= max_steps:
                print("Maximum step reached")
            else:
                print("\rreward obtained at t = ", self.mountain_car.t)

        return self.mountain_car.t

    def get_nn_activity(self, x=None, x_d=None):
        r = np.zeros(self.size ** 2)
        x = self.mountain_car.x if x is None else x
        x_d = self.mountain_car.x_d if x_d is None else x_d

        for k in range(self.size ** 2):
            center = self.input_layer[:, k]
            r[k] = np.exp(-((center[0] - x) / self.sigma_x) ** 2 - ((center[1] - x_d) / self.sigma_speed) ** 2)

        return r

    def get_q_value(self, action, r=None):
        r = self.get_nn_activity() if r is None else r
        return np.sum(np.multiply(self.input_weight[action + 1, :], r))

    def get_action_from_policy(self):
        r = self.get_nn_activity()

        q = np.sum(np.multiply(self.input_weight, np.array([r, r, r])), axis=1)
        denominator = np.sum(np.exp(q))

        prob = np.exp(q) / denominator

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


def launch_agent(q, c, trials):
    """
    Launch a learning procedure in a separate process for a new agent
    :param c: share counter
    :param q: shared queue
    :param trials: Number of trials
    :return:
    """

    a = SARSALambdaAgent()
    results = np.zeros(trials)

    for i in range(trials):
        t = a.trial(visualize=False)
        with c.get_lock():
            c.value -= 1
            print(c.value, 'Trials missing')

        results[i] = t

    q.put(results)


def moving_average(data, n=1):
    ma = []
    for i, x in enumerate(data):
        ma.append(np.mean(data[max(0, i-n):i+n+1]))

    return ma


if __name__ == "__main__":

    cmd = 0
    if len(sys.argv) > 1:
        cmd = sys.argv[1]

    # np.random.seed(42)

    if cmd == 0:
        num_agents = 1
        num_trials = 5

        processes = []
        queue = Queue()
        counter = Value('i', num_agents * num_trials)
        for l in range(num_agents):
            p = Process(target=launch_agent, args=(queue, counter, num_trials))
            p.start()

            processes.append(p)

        # Get the results
        times = np.mean(np.array([queue.get() for p in processes]), axis=0)
        ma = moving_average(times)

        mplb.plot(range(num_trials), times, '', range(num_trials), ma)
        mplb.show()

    if cmd == 1:
        agent = SARSALambdaAgent()
        for l in range(100):
            print('Trial', l + 1)
            agent.trial(logs=True)

        agent.plot_vector_field()
