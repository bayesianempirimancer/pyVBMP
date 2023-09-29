from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm
import torch 
import numpy as np
class animate_results():
    def __init__(self,assignment_type='sbz', f=r'./movie_temp.', xlim = (-2.5,2.5), ylim = (-2.5,2.5), fps=20):
        self.assignment_type = assignment_type
        self.f=f
        self.xlim = xlim
        self.ylim = ylim
        self.fps = fps

    def animation_function(self,frame_number, fig_data, fig_assignments, fig_confidence):
        fn = frame_number
        T=fig_data.shape[0]
        self.scatter.set_offsets(fig_data[fn%T, fn//T,:,:].numpy())
        self.scatter.set_array(fig_assignments[fn%T, fn//T,:].numpy())
        self.scatter.set_alpha(fig_confidence[fn%T, fn//T,:].numpy())
        return self.scatter,
        
    def make_movie(self,model,data, batch_numbers):
        print('Generating Animation using',self.assignment_type, 'assignments')


        if(self.assignment_type == 'role'):
            rn = model.role_dims[0] + model.number_of_objects*(model.role_dims[1]+model.role_dims[2])
            assignments = model.obs_model.assignment()/(rn-1)
            confidence = model.obs_model.assignment_pr().max(-1)[0]
        elif(self.assignment_type == 'sbz'):
            assignments = model.assignment()/2.0/model.number_of_objects
            confidence = model.assignment_pr().max(-1)[0]
        elif(self.assignment_type == 'particular'):
            assignments = model.particular_assignment()/model.number_of_objects
            confidence = model.assignment_pr().max(-1)[0]

        fig_data = data[:,batch_numbers,:,0:2]
        fig_assignments = assignments[:,batch_numbers,:]
        fig_confidence = confidence[:,batch_numbers,:]
        fig_confidence[fig_confidence>1.0]=1.0

        self.fig = plt.figure(figsize=(7,7))
        self.ax = plt.axes(xlim=self.xlim,ylim=self.ylim)
        self.scatter=self.ax.scatter([], [], cmap = cm.rainbow_r, c=[], vmin=0.0, vmax=1.0)
        ani = FuncAnimation(self.fig, self.animation_function, frames=range(fig_data.shape[0]*fig_data.shape[1]), fargs=(fig_data,fig_assignments,fig_confidence,), interval=5).save(self.f,writer= PillowWriter(fps=self.fps) )
        plt.show()
