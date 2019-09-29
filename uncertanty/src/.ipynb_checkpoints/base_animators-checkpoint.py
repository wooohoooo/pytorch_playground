import torch
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt



from matplotlib import animation
#Writer = animation.writers['pillow']
#writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from ipywidgets import interact, fixed
from IPython.display import HTML, display

import seaborn as sns
ffmpeg_path = 'C://Users//thoma//Documents//ffmpeg//FFmpeg//bin'#//bin'
#C:\Users\thoma\Documents\ffmpeg\FFmpeg\bin
#plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path


N = 100
min_x, max_x = -3, 1

class AnimatepEnsemble(object):
    def __init__(self,X_obs,y_obs,X_true,y_true,p=0.00, decay=0.001, non_linearity=torch.nn.LeakyReLU, n_models=10, model_list=None,u_iters=100, l2=1, n_std=4, title="",dataset_lenght=None):
        #super(AnimateBootstrapEnsemble, self).__init__(p, decay, non_linearity, n_models, model_list,dataset_lenght)

        self.losses = []
        self.n_std = n_std
        self.u_iters = u_iters
        self.l2 = l2
        self.title = title
        
        #
        self.X_obs = X_obs
        self.y_obs = y_obs
        self.X_true = X_true
        self.y_true = y_true
        
        ## plot items
        self.fig, self.ax0 = plt.subplots(1,1)
        self.ax0.set_ylim([15, -15])
        
        self.ax0.plot(self.X_obs, self.y_obs, ls="none", marker="o", color="0.1", alpha=0.5, label="observed")
        self.ax0.plot(self.X_true, self.y_true, ls="-", color="r", label="true")
        self.ln_mean, = self.ax0.plot([], [], ls="-", color="b", label="mean")
        
        self.loss_text = self.ax0.set_title('', fontsize=15)
        
        self.fill_stds = []
        for i in range(self.n_std):
            fill_t = self.ax0.fill_between(
                [], [], [],
                color="b",
                alpha=0.5**(i+1)
            )
            self.fill_stds.append(fill_t)
            
        self.ax0.legend(loc="upper left")
        


    def init_plot(self):
        self.ln_mean.set_data([], [])
        self.loss_text.set_text('')
        return self.ln_mean, self.loss_text
    
    def animate_plot(self, i,iters=50):
        for j in range(iters):
            loss = self.fit_ensemble(self.X_obs,self.y_obs)
            self.losses.append(loss)
        
        #self.loss_text.set_text('{}, loss[{}]={:.3f}'.format(self.title, (i+1)*100, loss))
        
        y_mean, y_std = self.ensemble_uncertainity_estimate(
            self.X_true, self.u_iters, l2=self.l2,
            range_fn=range
        )
        
        self.ln_mean.set_data(self.X_true, y_mean)
        for i in range(self.n_std):
            self.fill_stds[i].remove()
            self.fill_stds[i] = self.ax0.fill_between(
                self.X_true,
                y_mean - y_std * ((i+1)/2),
                y_mean + y_std * ((i+1)/2),
                color="b",
                alpha=0.5**(i+1)
            )
        return [self.ln_mean, self.loss_text] + self.fill_stds
        
        
    def train(self, iters, interval=100):
        anim = animation.FuncAnimation(
            self.fig, self.animate_plot, init_func=self.init_plot,
            frames=range(iters), interval=interval, blit=True)
        return HTML(anim.to_html5_video())