import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import unittest

# my Enumber is E/13/095 so my differential function is -20(x^3)
def differtial_function(currentValue):
    return 20*np.power(currentValue,3)
class GradientDescent:
    def __init__(self,initial_x,precision,learning_rate):
        self.initial_x=initial_x
        self.precision=precision
        self.learning_rate=learning_rate
        self.previous_stepsize=initial_x
    def plot_graph(self):
        matplotlib.rc('xtick', labelsize=30)
        matplotlib.rc('ytick', labelsize=30)
        matplotlib.rc('axes', titlesize=30)
        matplotlib.rc('legend', fontsize=30)
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(hspace=1)
        axes = plt.subplot(1, 1, 1)
        # fucntion with my enumber f(x)=14-(5(x^4))
        #since it is not possible to get a minima when the coefficent is negative i make the coeficent positive
        X = np.linspace(-10,10)
        Y = 14+5*np.power(X,4)
        axes.plot(X, Y, linewidth=1, ls='--', color='r', marker="*")
        plt.show()

    def local_minima(self):
        current_x = self.initial_x
        step_size = self.previous_stepsize
        while(step_size>self.precision):
            previous_x = current_x
            current_x = current_x - (self.learning_rate * differtial_function(previous_x))
            step_size = abs(current_x-previous_x)
        return current_x
class Test(unittest.TestCase):
    def setUp(self):
        self.graph1= GradientDescent(0.1,0,0.01)     #initial x = 0
        self.graph2= GradientDescent(0.1,100,0.01)   #intial x = 100
        self.graph3 =GradientDescent(0.1,-100,0.01) #initial x = -100

    def test_one(self):
        self.assertAlmostEqual(self.graph1.local_minima(),0.1)
        print "Since there are  no other maximums or minmums whatever value given should retrun the minimum 0.1.Here since the initial value is zero it should" \
              "give the result without any doubt."

    def test_two(self):
        self.assertAlmostEqual(self.graph2.local_minima(),0.1)
        print "Even if 100 or 1000 is given it still gives the local minima as 0.1 because there are no other local minma or maxima in my fucntion"

    def test_three(self):
        self.assertAlmostEqual(self.graph3.local_minima(),0.1)
        print "Even if -100 or -1000 is given it still gives the local minima as 0.1 because there are no other local minma or maxima in my fucntion"

    def tearDown(self):
        print "Finished the test\n"

graph = GradientDescent(0.1,5,0.01)
graph.plot_graph()
print graph.local_minima()
unittest.main()

#Q2.It is better to select inital x always closer to the local minma or else if it finds a maximum it stoppes the loop there the local minma cannot be found.
#Q3.If we increases the learning rate it increases the speed we get the answer, but after specific range it might skip the answer so we wont be able to identify the local minima.
#Q4. The result i get is always 0.1 for any inital x since my graph only contains one turning point. This wont be the case is someones function has another turning point.