import sys

import pylab as plb
import numpy as np
import mountaincar

class DummyAgent():
    """A not so good agent for the mountain-car task.
    """

    def __init__(self, mountain_car = None, parameter1 = 3.0):
        
        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        self.parameter1 = parameter1

    def visualize_trial(self, n_steps = 200):
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
            print '\rt =', self.mountain_car.t,
            sys.stdout.flush()
            
            # choose a random action
            self.mountain_car.apply_force(np.random.randint(3) - 1)
            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

            # update the visualization
            mv.update_figure()
            plb.draw()            
            
            # check for rewards
            if self.mountain_car.R > 0.0:
                print "\rreward obtained at t = ", self.mountain_car.t
                break
                
    def boltzman_activity_selection(self, s, t):
        #selects an action using Epsilon-greedy strategy
        # change the sum to for loop
        # s: the current state        
                
        a = np.exp(Q(s,a)/t)/(np.sum(Q(s,a)/t));
        return a
    
    def Q(s,a):
        # change sum to a for loop
        #q = np.sum(w * r);
        #no implementation yet
        
        return q
    
    def inputActivity (x1,x2,psi,sigma):
        
        r = np.exp(   -np.square(x1-x2)/np.square(sigma)  -    np.square(psi-x2)/np.square(sigma) )
        
        return  r   
        
    def learn(self):
        # This is your job!
        """Do a trial without learning, with display.

          Weight Update comes here and it replaces dummy agent function
        """
        
        # weight update here check pdf report for update formular
       

    
    
if __name__ == "__main__":
    d = DummyAgent()
    d.visualize_trial()
    plb.show()
