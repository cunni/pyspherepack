import autograd.numpy as np 
from autograd import grad
from autograd.misc.optimizers import adam
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import interactive
interactive(True)
#import pylab
import matplotlib.patches as patches
from .utils import np_printoptions       
import pickle
import pylatex

class Box(object):
    """a fairly simple class, with some non-oop goofiness to play nicely with autograd grad and their adam"""
    def __init__(self,n_balls,n_dims=None,n_iters=None,box=None,logits=None):
        # number of balls in box
        self.n_balls = n_balls
        self.n_dims = n_dims if n_dims is not None else 2
        # defines the bounding box; just a vector of the outer corner; the inner corner is assumed to be the origin
        self.box = box if box is not None else np.array([1.0]*self.n_dims)
        # defines the ball positions in logit space; always 
        self.logits = logits if logits is not None else np.random.randn(self.n_balls,self.n_dims)
        # some optimization parameters
        self.n_iters = n_iters if n_iters is not None else 100000
        self.step_size = 0.001
     
    def ball_radius(self,x=None,i=None):
        """calculates the maximum size sphere that can be 
        packed in a given constellation of x points.  
        Note: assumes no boundary, so that's really for the warper to determine"""
        x = x if x is not None else self.box_warp()  # still oop but allowing autograd to do its thing
        # Note that i is in the argument here because I was too lazy to rewrite autograd.misc.optimizers adam
        sum_squares = np.sum((x[np.newaxis,:,:] - x[:,np.newaxis,:])**2, axis=2) + np.diag([np.inf]*np.shape(x)[0])
        return 0.5*np.sqrt(np.min(sum_squares))
    
    def box_warp(self,logits=None):
        """warps real logits into the specified box.  Note here self.box is used in usual oop fashion,
        though the fp approach of grad is such that we need grad to pass logits around, hence not self.logits"""
        logits = logits if logits is not None else self.logits  # still oop but allowing autograd to do its thing
        return (self.box / (1.0 + np.exp(-logits)))
    
    
    def print_status(self,logits=None,i=None,g=None):
        """just a print callback"""
        logits = logits if logits is not None else self.logits  # still oop but allowing autograd to do its thing
        if i % 5000 == 0:
            print("{:9}|{:23}|{:20}".format(i, self.ball_radius(self.box_warp(logits),i), self.density(logits) ))
    
    def pack(self):
        print("   Iter  |    Ball radius        |     Density  ")
        self.logits = adam(grad(lambda logits,i: -1*self.ball_radius(self.box_warp(logits),i)), self.logits, num_iters=self.n_iters,callback=self.print_status)
        # one more print at final iteration
        self.print_status(i=self.n_iters)
        
    def density(self,logits=None):
        logits = logits if logits is not None else self.logits  # still oop but allowing autograd to do its thing
        rad = self.ball_radius(self.box_warp(logits))
        return 100*rad**2*np.pi*self.n_balls/(np.prod(self.box+(2*rad)))
    
    def plot(self,scaled_rad=None,clamp_edge=0.0):
        """visualize the balls packed in"""
        x = self.box_warp(self.logits)
        rad = self.ball_radius(x)
        scaled_rad = scaled_rad if scaled_rad is not None else rad
        scaled_box = scaled_rad/rad*(self.box+2*rad)
        scaled_x = scaled_rad/rad*(x + rad)
        #rad_norm = rad/(1+2*rad) # not quite right, box is normalizing oddly when nonsquare
        print("Optimized Ball radii: {:04.2f}, scaled {:04.2f}".format(rad,scaled_rad))
        with np_printoptions(precision=6, suppress=True):
                print('Ball centers (scaled):\n {}'.format(scaled_x))
        # print("Normalized (true) Ball radii: {:06.4f}".format(rad_norm))
        print("Density %: {:04.2f}%".format(self.density()))
        print("Waste %: {:04.2f}%".format(100-self.density()))
        print("Density with clamp edge %: {:04.2f}%".format((self.density()*np.prod(scaled_box)/(scaled_box[1]*(scaled_box[0]+2*clamp_edge)))))
        print("Waste with clamp edge %: {:04.2f}%".format(100-(self.density()*np.prod(scaled_box)/(scaled_box[1]*(scaled_box[0]+2*clamp_edge)))))
        if self.n_dims==2:
            fig,ax = plt.subplots()
            # plot bounding box
            #rect = patches.Rectangle((0,0)-rad,self.box[0]+2*rad,self.box[1]+2*rad,linewidth=2,edgecolor='k',facecolor='none')
            rect = patches.Rectangle((-clamp_edge,0),scaled_box[0]+2*clamp_edge,scaled_box[1],hatch='x',linewidth=2,edgecolor='k',facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            rect2 = patches.Rectangle((0,0),scaled_box[0],scaled_box[1],linewidth=2,edgecolor='k',facecolor='w')
            ax.add_patch(rect2)
            # plot balls
            for i in range(self.n_balls):
                ax.add_artist(plt.Circle((scaled_x[i,0],scaled_x[i,1]),scaled_rad,fill=False,color='C0',linewidth=2))
            # plot centers
            ax.add_artist(plt.scatter(scaled_x[:,0],scaled_x[:,1]))
            ax.axis('equal')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            # add annotations
            ax.add_artist(plt.scatter(0.0,0.0,color='k'))
            ax.text(0.0, 0.0, '(0.00,0.00)',horizontalalignment='left',verticalalignment='top')
            ax.add_artist(plt.scatter(scaled_box[0],0.0,color='k'))
            ax.text(scaled_box[0], 0.0, '({:04.2f},0.00)'.format(scaled_box[0]),horizontalalignment='right',verticalalignment='top')
            ax.add_artist(plt.scatter(0.0,scaled_box[1],color='k'))
            ax.text(0.0, scaled_box[1], '(0.00,{:04.2f})'.format(scaled_box[1]),horizontalalignment='left',verticalalignment='bottom')
            ax.add_artist(plt.scatter(scaled_box[0],scaled_box[1],color='k'))
            ax.text(scaled_box[0], scaled_box[1], '({:04.2f},{:04.2f})'.format(scaled_box[0],scaled_box[1]),horizontalalignment='right',verticalalignment='bottom')
            if clamp_edge > 0:
                ax.add_artist(plt.scatter(-clamp_edge,0.0,color='k'))
                ax.text(-clamp_edge, 0.0, '-{:03.1f}'.format(clamp_edge),horizontalalignment='right',verticalalignment='top')
                ax.add_artist(plt.scatter(scaled_box[0]+clamp_edge,0.0,color='k'))
                ax.text(scaled_box[0]+clamp_edge, 0.0, '+{:03.1f}'.format(clamp_edge),horizontalalignment='left',verticalalignment='top')
            plt.show() 

class ManyBox(object):
    """instantiates many boxes of a size, packs through them, for local optima silliness"""
    def __init__(self,n_balls,n_boxes=None,locked=None,filename=None,**kwargs):
        self.n_balls = n_balls
        self.n_boxes = n_boxes if n_boxes is not None else 10
        self.boxes = [Box(n_balls,**kwargs) for i in range(self.n_boxes)]
        self.best_box = {'i':None, 'density':0, 'box':None}
        self.locked = locked if locked is not None else True
        self.filename = filename if filename is not None else 'data/manybox.pkl'

    def pack(self):
        for i in range(self.n_boxes):
            print('=========packing box {}==========='.format(i))
            self.boxes[i].pack()
            self.best_box = self.best_box if self.best_box['density'] > self.boxes[i].density() else {'i':i,'density':self.boxes[i].density(),'box':self.boxes[i]}
            self.save()
            print('=========done box {}, with density {}========'.format(i,self.boxes[i].density()))

    def density_distrib(self):
        return [b.density() for b in self.boxes]

    def save(self):
        with open(self.filename,'wb') as f:
            pickle.dump(self,f)

    @classmethod
    def load(cls,filename=None):
        # https://stackoverflow.com/questions/2709800/how-to-pickle-yourself
        filename = filename if filename is not None else 'data/manybox.pkl'
        try: 
            with open(filename,'rb') as f:
                return pickle.load(f)
        except:
            raise IOError('unable to load file: {}'.format(filename))

    @classmethod
    def tex_best(cls,filenames=None,texname=None,scaled_rad=None,clamp_edge=None):
        filenames = filenames if filenames is not None else ['data/mb_50_2x1.pkl','data/mb_50_3x1.pkl']
        texname = texname if texname is not None else 'data/aggregated_results' 
        # set up pylatex doc
        geometry_options = {"margin": "1in"}
        doc = pylatex.Document(texname, geometry_options=geometry_options)
        dapne = lambda s: doc.append(pylatex.NoEscape(s))
        with doc.create(pylatex.Section('Introduction')):
            doc.append('Each section that follows shows an optimized layout for a given number of circles and an approximate aspect ratio of the sheet. Throughout, the following parameters are assumed: clamp edge of 10.0mm, circle diameter of 20mm, spacing between circles of 0.50mm.')
        for fn in filenames:
            mb = cls.load(filename=fn)
            b = mb.best_box['box']
            b.plot(clamp_edge=clamp_edge,scaled_rad=scaled_rad)
            # pylatex to put this in tex
            #matplotlib.use('Agg')
            with doc.create(pylatex.Section(pylatex.NoEscape(r'{} circles, box aspect ratio of roughly ${}\times{}$'.format(b.n_balls,b.box[0],b.box[1])),label=fn)):
                with doc.create(pylatex.Figure(position='htbp')) as plot:
                    plot.add_plot(width=pylatex.NoEscape(r'0.8\textwidth'))
                    #plot.add_caption('Optimized circle packing for this sheet size.')

            x = b.box_warp(b.logits)
            rad = b.ball_radius(x)
            clamp_edge = clamp_edge if clamp_edge is not None else 0.0
            scaled_rad = scaled_rad if scaled_rad is not None else rad
            scaled_box = scaled_rad/rad*(b.box+2*rad)
            scaled_x = scaled_rad/rad*(x + rad)
            #doc.append(pylatex.NoEscape('\noindent Density %:'))
            dapne(r'\noindent Density \%: {:04.2f}\% \\'.format(b.density()))
            dapne(r'Waste \%: {:04.2f}\% \\'.format(100-b.density()))
            dapne(r'Density with clamp edge \%: {:04.2f}\% \\'.format((b.density()*np.prod(scaled_box)/(scaled_box[1]*(scaled_box[0]+2*clamp_edge)))))
            dapne(r'Waste with clamp edge \%: {:04.2f}\% \\'.format(100-(b.density()*np.prod(scaled_box)/(scaled_box[1]*(scaled_box[0]+2*clamp_edge)))))
            
            dapne(r'Circle center coordinates: \\')
            for i in range(b.n_balls):
                #dapne(r'$c_{{{}}}$: {}\\'.format(i+1,scaled_x[i,:]))
                dapne(r'$[{}~~{}]$ \\'.format(scaled_x[i,0],scaled_x[i,1]))
            dapne(r'\clearpage') 

        doc.generate_tex()




