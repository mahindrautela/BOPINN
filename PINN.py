#import lib.tf_silent
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from lib.pinn_wave import PINN
from lib.network import Network
from lib.optimizer import L_BFGS_B
import math
import time

# number of training samples
num_train_samples = 25000
    
# number of test samples
num_test_samples = 5000
    
c = 0.2 #scaled speed
L = 10
n = L
T = 1

# Initial conditions
def u0(t):
    z = -np.sin(1*math.pi*t)
    return z

def du0_dt(tx):
    with tf.GradientTape() as g:
        g.watch(tx)
        u = u0(tx)
    du_dt = g.batch_jacobian(u, tx)[..., 0]
    return du_dt

# Analytical solution
xx = np.linspace(0,L,num_test_samples)
tt = np.linspace(0,T,num_test_samples)
usol = np.zeros((num_test_samples,num_test_samples))
for i,xi in enumerate(xx):
    for j,tj in enumerate(tt):
        usol[i,j] = -np.sin(math.pi*xi)*np.cos(n*math.pi*c*tj/L)


########################################################################
######################## collocation points ############################
########################################################################

# create training input
tx_eqn = np.random.rand(num_train_samples, 2)
tx_eqn[..., 0] = T*tx_eqn[..., 0]                      # t =  0 ~ +1
tx_eqn[..., 1] = L*tx_eqn[..., 1]                      # x = 0 ~ +10
#print('\nShape of t_eqn ==>',tx_eqn.shape)

tx_ini = np.random.rand(num_train_samples, 2)
tx_ini[..., 0] = 0                                     # t = 0
tx_ini[..., 1] = L*tx_ini[..., 1]                      # x = 0 ~ +10
#print('\nShape of tx_ini ==>',tx_ini.shape)

tx_bnd = np.random.rand(num_train_samples, 2)
tx_bnd[..., 0] = T*tx_bnd[..., 0]                      # t =  0 ~ +1
tx_bnd[..., 1] = L*np.round(tx_bnd[..., 1])            # x =  0 or +10
#print('\nShape of tx_bnd ==>',tx_bnd.shape)

u_zero = np.zeros((num_train_samples, 1))
u_ini = u0(tx_ini[:,1,None])
du_dt_ini = np.zeros((num_train_samples, 1))

#########################################################################
########################### TRAINING PINNs ##############################
#########################################################################

# build a core network model
network = Network.build()
#network.summary()

# build a PINN model
pinn = PINN(network,c).build()

# train the model using L-BFGS-B algorithm
begin = time.time()
x_train = [tx_eqn, tx_ini, tx_bnd]
y_train = [u_zero, u_ini, du_dt_ini, u_zero]
lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
lbfgs.fit()
end = time.time()
totaltime = end-begin
print("\n Total runtime of the program is (min.)",totaltime/60)

#########################################################################
######################## PREDICTION #####################################
#########################################################################

# predict u(t,x) distribution
t_flat = np.linspace(0, T, num_test_samples)
x_flat = np.linspace(0, L, num_test_samples)
t, x = np.meshgrid(t_flat, x_flat)
tx = np.stack([t.flatten(), x.flatten()], axis=-1)
u = network.predict(tx, batch_size=num_test_samples)
u = u.reshape(t.shape)

# plot u(t,x) distribution as a color-map
fig = plt.figure(figsize=(12,8))
gs = GridSpec(2, 3) # A grid layout to place subplots within a figure.
plt.subplot(gs[0, :])
vmin, vmax = -1.0, +1.0
plt.pcolormesh(t, x, u, cmap='rainbow', shading = 'auto', norm=Normalize(vmin=vmin, vmax=vmax))
plt.xlabel('t',fontsize=20)
plt.ylabel('x',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
cbar = plt.colorbar(pad=0.05, aspect=10)
cbar.set_label('u(t,x)', fontsize=20) 
cbar.ax.tick_params(labelsize=20)
cbar.mappable.set_clim(vmin, vmax)

# plot u(t=const, x) cross-sections
tfrac = np.array([0.25,0.5,0.75])
t_cross_sections = (T*tfrac).tolist()
idx = [int(x) for x in (num_test_samples*tfrac)]

for i, t_cs in enumerate(t_cross_sections):
    plt.subplot(gs[1, i])
    full = np.full(t_flat.shape, t_cs)
    tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
    u = network.predict(tx, batch_size=num_test_samples)
    #print(u.shape)
    plt.plot(x_flat, u, '.b')
    plt.plot(x_flat, usol[:,idx[i]], 'r--', linewidth = 2)
    plt.title('t = {}'.format(t_cs),fontsize=20)
    plt.xlabel('x',fontsize=20)
    plt.ylabel('u(t,x)',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-1,1)
    plt.legend(['Prediction','Exact'], loc = 'upper right',fontsize=8)
plt.tight_layout()
plt.savefig('PINNs_at_'+str(c)+'.png', transparent=True, dpi = 900)
plt.show()

