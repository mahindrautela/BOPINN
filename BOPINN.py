import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from lib.pinn_wave import PINN
from lib.network import Network
from lib.optimizer import L_BFGS_B
import math
from bayes_opt import BayesianOptimization, UtilityFunction
import scipy.io
import os
from os.path import join   
import time

# number of training samples
num_train_samples = 25000
    
# number of test samples
num_test_samples = 5000
    
# Other variables
L = 10
n = L
T = 1

# define x,t for PINNs prediction
x_test = np.linspace(0,L,num_test_samples)
t_test = np.linspace(0,T,num_test_samples)

# upload the snapshot observation
path = "E:\MSR\data\BOPINN\data"
dir_list = os.listdir(path)
print("Files in directory",dir_list)

idx_data = 0 # 0,3,6
data = dir_list[idx_data] 
print("Imported file", data)

file = join(path, data)
u_analy = scipy.io.loadmat(file)
u_analy = u_analy['a1']

# plot the snapshot observation
fig = plt.figure(figsize=(7,4))
plt.plot(x_test,u_analy, '-', linewidth = 2)   
plt.xlabel('$x$', fontsize = 15)
plt.ylabel('Normalized u(x,t)', fontsize = 15)
plt.xticks(fontsize = 12) 
plt.yticks(fontsize = 12)

# time of observation
tilde_t = 0.25

#%% Initial conditions
def u0(t):
    z = -np.sin(1*math.pi*t)
    return z

def du0_dt(tx):
    with tf.GradientTape() as g:
        g.watch(tx)
        u = u0(tx)
    du_dt = g.batch_jacobian(u, tx)[..., 0]
    return du_dt

def RMS(S):
    rms = np.sqrt(np.mean(S**2))
    return rms
    
#%% collocation points 
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

# initial and boundary conditions
u_zero = np.zeros((num_train_samples, 1))
u_ini = u0(tx_ini[:,1,None])
du_dt_ini = np.zeros((num_train_samples, 1))

#%% g(c) = (u_pred - u_true)^2; u_pred via PINNs

def model_builder(ic):
    #ic = hp.Float('ic', min_value=0.1, max_value=1, step=10)
    print('\n ## ->>>> PINNs simulation at speed = ' + str(ic))
            
    # build a PINN model
    network = Network.build()
    pinn = PINN(network,ic).build()
    
    # train the model using L-BFGS-B algorithm
    begin = time.time()
    x_train = [tx_eqn, tx_ini, tx_bnd]
    y_train = [u_zero, u_ini, du_dt_ini, u_zero]
    lbfgs = L_BFGS_B(model=pinn, x_train=x_train, y_train=y_train)
    lbfgs.fit()
    end = time.time()
    totaltime = end-begin
    print("\n Total runtime is (min.)",totaltime/60)
    
    # test the model
    tx = np.stack([np.full(t_test.shape, tilde_t), x_test], axis=-1)
    u_pred = network.predict(tx, batch_size=num_test_samples)
    
    # mse between u_pred via PINN and snapshot observation
    mse = -np.mean(np.square(u_analy - u_pred))
    
    del network, pinn, lbfgs, u_pred
    
    return mse

#%% Bayesian Optimization
# Attributes of BO
itt_explore = 5
itt = 45
itt_all = itt_explore + itt
n_runs = 10

# bounds of BO
pbounds = {'ic': (0.1, 1)}

# Start BO
mse_star_all = []
cstar_all = []
mse_all_all = []
ic_all_all = []

for r in range(n_runs):
    print('\n ## ->>>> Run =  ' + str(r))
    
    # define the model
    optimizer = BayesianOptimization(
        f=model_builder,
        pbounds=pbounds,
        allow_duplicate_points=True)
    
    # utility function
    util = UtilityFunction(kind='ucb',
                           kappa=2.576,
                           xi=0.0,
                           kappa_decay=1,
                           kappa_decay_delay=0)
    
    # run the model
    optimizer.maximize(init_points=itt_explore, 
                       n_iter=itt, 
                       acquisition_function=util)
    
    soln = optimizer.max
    resi = optimizer.res
    
    # optimum values
    mse_star = list(soln.values())[0]
    cstar = list(soln.values())[1]
    cstar2 = list(cstar.values())[0]
    
    # append all optimum values
    mse_star_all.append(mse_star)
    cstar_all.append(cstar2)
    
    # all run values
    mse_all = []
    ic_all = []
    for i,res in enumerate(resi):
        mse = list(res.values())[0]
        ic = list(res.values())[1]
        ic2 = list(ic.values())[0]
        
        # append all run values
        mse_all.append(mse)
        ic_all.append(ic2)
    
    mse_all_all.append(np.array(mse_all))
    ic_all_all.append(np.array(ic_all))
    
    del optimizer

mse_all_all = np.array(mse_all_all)
mse_star_all = np.array(mse_star_all)
ic_all_all = np.array(ic_all_all)
cstar_all = np.array(cstar_all)

#%% Process the BO results
# max, min, mean and sd target function/objective function value across different runs
max_mse_star_allruns, min_mse_star_allruns = np.max(mse_star_all), np.min(mse_star_all)
mean_mse_star_allruns, std_mse_star_allruns = np.mean(mse_star_all), np.std(mse_star_all)

# optima corresponding to abovementioned optimal points
idx_max_mse_star_allruns = np.where(max_mse_star_allruns == mse_star_all)
idx_min_mse_star_allruns = np.where(min_mse_star_allruns == mse_star_all)

max_cstar_allruns = cstar_all[idx_max_mse_star_allruns]
min_cstar_allruns = cstar_all[idx_min_mse_star_allruns]
mean_cstar_allruns = np.mean(cstar_all)
std_cstar_allruns = np.std(cstar_all)

print("Max (best optimal) tf across runs = ",max_mse_star_allruns)
print("Min (worst optimal) tf across runs = ",min_mse_star_allruns)
print("Mean tf across runs = ",mean_mse_star_allruns)
print("Std tf across runs = ",std_mse_star_allruns)

print("Max (best optimal) c* across runs = ",max_cstar_allruns)
print("Min (worst optimal) c* across runs = ",min_cstar_allruns)
print("Mean c* across runs = ",mean_cstar_allruns)
print("Std c* across runs = ",std_cstar_allruns)

# plot best optimal run with the optima
idx_max_all = []
for i in range(mse_all_all.shape[0]):
    idx_max = np.where(mse_all_all[i,:] == mse_star_all[i])
    idx_max = idx_max[0][0]
    idx_max_all.append(idx_max)

mean_mse_all = np.mean(mse_all_all, axis=0)
std_mse_all = np.std(mse_all_all, axis=0)
mean_ic_all = np.mean(ic_all_all, axis=0)
std_ic_all = np.std(ic_all_all, axis=0)

opt_mse_run = mse_all_all[idx_max_mse_star_allruns[0][0]]
opt_c_run = ic_all_all[idx_max_mse_star_allruns[0][0]]
opt_mse = mse_star_all[idx_max_mse_star_allruns[0][0]]
opt_c = cstar_all[idx_max_mse_star_allruns[0][0]]

txt = 'c* = '+ str(round(opt_c,4))
plt.figure(figsize = (8, 6))
plt.plot(opt_c_run,opt_mse_run,'ob',markersize=6)
plt.plot(opt_c,opt_mse,'*r',markersize=8)
plt.text(0.75, -0.02, txt, fontsize=15, c = 'r')
plt.xlabel("velocity, c",fontsize=20)
plt.ylabel("target function, g(c)",fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(['TF vs c','Optimal (Max)'], fontsize = 16)
plt.savefig('tfvsc_'+str(idx_data+1)+'.png', bbox_inches='tight', dpi=600)
plt.show()