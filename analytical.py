import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import os

# Other variables
L = 10
n = L
T = 1

# speed
C = [0.20, 0.55, 0.85]
c = C[0]

# snapshot time
tfrac = np.array([0.25, 0.50, 0.75])
tilde_t = tfrac*T

# number of scan points
num_points = 5000

# RMS function
def RMS(S):
    rms = np.sqrt(np.mean(S**2))
    return rms

def u_analytic(t,x,c,L,n):
    # x and t in mesh form
    usol = -np.sin(math.pi*x)*np.cos(n*math.pi*c*t/L)
    return usol

# Full analytical solution of the domain [0,L] and [0,T]
t = np.linspace(0,T,num_points)
x = np.linspace(0,L,num_points)
x_mesh, t_mesh = np.meshgrid(x, t)
usol_full = u_analytic(t_mesh, x_mesh, c, L, n)

# index of snapshot time in the array
idx = [int(p) for p in num_points*tfrac]

# plot u(t,x) distribution as a color-map
fig = plt.figure(figsize=(7,4))
gs = GridSpec(2, 3) # A grid layout to place subplots within a figure.
plt.subplot(gs[0, :])
vmin, vmax = -1.0, +1.0
plt.pcolormesh(t_mesh, x_mesh, usol_full, cmap='rainbow', shading = 'auto', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t')
plt.ylabel('x')
cbar = plt.colorbar(pad=0.05, aspect=10)
cbar.set_label('u(t,x)')
cbar.mappable.set_clim(vmin, vmax)

# plot u(t=const, x) cross-sections
for i, t_cs in enumerate(tilde_t):
    plt.subplot(gs[1, i])
    plt.plot(x, usol_full[idx[i],:], 'b', linewidth = 2)
    plt.title('t={}'.format(t_cs))
    plt.xlabel('x')
    plt.ylabel('u(t,x)')
    plt.ylim(-1,1)
plt.tight_layout()
plt.show()

#%% usol at particular snapshot time
from scipy.io import savemat

# data storing
path = "C:\MSR\data\PINNBO\data"

# amplitude of noise
beta_range = [0.0075] #[0.0075, 0.01, 0.025, 0.05]

# these parameters need to be set
usol_app = []
usol_n_app = []
for ii in range(len(C)):
    for jj in range(len(tilde_t)):
        for kk in range(len(beta_range)):
            c = C[ii]
            t_obs = tilde_t[jj]
            beta = beta_range[kk]
            
            # without noise
            tt = np.full(t.shape, t_obs)
            xx = np.linspace(0,L,num_points)
            xx_mesh, tt_mesh = np.meshgrid(xx, tt)
            usol = u_analytic(tt_mesh, xx_mesh, c, L, n)
            usol = usol[0,:] # all rows (time) are same
            usol_app.append(usol)
            
            # Add white noise to the data
            mu = 0
            sigma = 1
            noise = beta*(sigma*np.random.randn(num_points,1) + mu)
            
            # noisy data
            usol_n = usol[:,np.newaxis] + noise
            usol_n_app.append(usol_n)
            
            # signal to noise ratio
            snr = 20*np.log10(RMS(usol)/RMS(noise))
            snr_percent = RMS(noise)/RMS(usol)*100
            
            f1 = "c="+str(c)
            f2 = "t="+str(t_obs)
            f3 = "snr="+str(np.round(snr,2))
            
            plt.figure(figsize=(22,4))
            fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,4))
            ax1.plot(xx,usol)
            ax1.set_title('Without noise', fontsize = 15)
            ax1.set(xlabel='x', ylabel='Normalized u(x,t)')
            ax1.set_ylim(-1.2,1.2)
            
            ax2.plot(xx,usol_n)
            ax2.set_title(f1+f2+f3, fontsize = 15)  
            ax2.set(xlabel='x')
            ax2.set_ylim(-1.2,1.2)
            
            plt.show()

            # save data
            filename = "u_analytic_"+f1+f2+f3
            filepath = os.path.join(path, filename)
            mdic = {"a1": usol_n, "label": "experiment"}
            savemat(filepath+".mat", mdic)

#%% analysis of experimental data
usol_app = np.array(usol_app)
usol_n_app = np.array(usol_n_app)
color = ['k','tab:blue','tab:orange',
         'tab:green','tab:red','tab:purple',
         'tab:brown','tab:pink','tab:gray','tab:cyan']

plt.figure(figsize=(10,4))
for i in range(usol_app.shape[0]):
    plt.plot(xx,usol_app[i,:],color[i])
plt.legend(['1','2','3','4','5','6','7','8','9'])
#plt.savefig('analytical_9_withoutnoise', dpi = 300)

plt.figure(figsize=(10,4))
for i in range(usol_app.shape[0]):
    plt.plot(xx,usol_n_app[i,:],color[i])
plt.legend(['1','2','3','4','5','6','7','8','9'])
#plt.savefig('analytical_9_withnoise', dpi = 300)
