import autograd.numpy as np 
from autograd import grad
from autograd.misc.optimizers import adam
from matplotlib import pyplot as plt
from matplotlib import interactive
interactive(True)
#import pylab
import matplotlib.patches as patches
from .utils import np_printoptions       

class Box(object):
    """a fairly simple class, with some non-oop goofiness to play nicely with autograd grad and their adam"""
    def __init__(self,n_balls,n_dims=None,n_iters=None,box=None,x=None):
        # number of balls in box
        self.n_balls = n_balls
        self.n_dims = n_dims if n_dims is not None else 2
        # defines the bounding box; just a vector of the outer corner; the inner corner is assumed to be the origin
        self.box = box if box is not None else np.array([1.0]*self.n_dims)
        # defines the ball positions
        self.x = x if x is not None else self.box*np.random.randn(self.n_balls,self.n_dims)
        # some optimization parameters
        self.n_iters = n_iters if n_iters is not None else 100000
        self.step_size = 0.001
     
    def ball_radius(self,x=None,i=None):
        """calculates the maximum size sphere that can be 
        packed in a given constellation of x points.  
        Note: assumes no boundary, so that's really for the warper to determine"""
        x = x if x is not None else self.sig_box()  # still oop but allowing autograd to do its thing
        # Note that i is in the argument here because I was too lazy to rewrite autograd.misc.optimizers adam
        sum_squares = np.sum((x[np.newaxis,:,:] - x[:,np.newaxis,:])**2, axis=2) + np.diag([np.inf]*np.shape(x)[0])
        return 0.5*np.sqrt(np.min(sum_squares))
    
    def sig_box(self,x=None):
        """warps real x into the specified box.  Note here self.box is used in usual oop fashion,
        though the fp approach of grad is such that we need grad to pass x around, hence not self.x"""
        x = x if x is not None else self.x  # still oop but allowing autograd to do its thing
        return (self.box / (1.0 + np.exp(-x)))
    
    
    def print_status(self,x=None,i=None,g=None):
        """just a print callback"""
        x = x if x is not None else self.x  # still oop but allowing autograd to do its thing
        if i % 5000 == 0:
            print("{:9}|{:23}|{:20}".format(i, self.ball_radius(self.sig_box(x),i), self.density(x) ))
    
    def pack(self):
        print("   Iter  |    Ball radius        |     Density  ")
        self.x = adam(grad(lambda x,i: -1*self.ball_radius(self.sig_box(x),i)), self.x, num_iters=self.n_iters,callback=self.print_status)
        # one more print at final iteration
        self.print_status(i=self.n_iters)
        
    def density(self,x=None):
        x = x if x is not None else self.x  # still oop but allowing autograd to do its thing
        rad = self.ball_radius(self.sig_box(x))
        return 100*rad**2*np.pi*self.n_balls/(np.prod(self.box+(2*rad)))
    
    def plot(self):
        """visualize the balls packed in"""
        xx = self.sig_box(self.x)
        rad = self.ball_radius(xx)
        #rad_norm = rad/(1+2*rad) # not quite right, box is normalizing oddly when nonsquare
        print("Optimized Ball radii: {:04.2f}".format(rad))
        with np_printoptions(precision=3, suppress=True):
                print('Ball centers:\n {}'.format(xx))
        # print("Normalized (true) Ball radii: {:06.4f}".format(rad_norm))
        print("Density %: {:04.2f}%".format(self.density()))
        print("Waste %: {:04.2f}%".format(100-self.density()))
        if self.n_dims==2:
            fig,ax = plt.subplots()
            # plot balls
            for i in range(self.n_balls):
                ax.add_artist(plt.Circle((xx[i,0],xx[i,1]),rad,fill=False,color='C0',linewidth=2))
            # plot centers
            plt.scatter(xx[:,0],xx[:,1])
            # plot bounding box
            rect = patches.Rectangle((0,0)-rad,self.box[0]+2*rad,self.box[1]+2*rad,linewidth=2,edgecolor='k',facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            ax.axis('equal')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            plt.show() 

