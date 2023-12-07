# BOPINN (Bayesian optimized physics-informed neural network)
BOPINN presents new paradigm to solve inverse problems by bringing an amalgamation of PINNs and BO. It uses BO (a gradient-free and global optimization scheme) and PINNs (a fast neural surrogate solver for PDEs). In BOPINN, a PINN utilizes a neural surrogate to solve the partial differential equation (wave propagation here). Bayesian optimization runs over the PINN model and estimates the optimum parameters (wave velocity in the medium here) using a single snapshot observation of the field. BOPINN queries the black-box PINN model at different wave velocities until it converges to the true wave velocity. The proposed method is simpler (uses single neural network), robust (capturs uncertainty) and flexible (useful in real-time and online settings) as compared to it's counterparts.  
![BOPINN Algorithm](https://github.com/mahindrautela/BOPINN/tree/main/images/BOPINNalgo.PNG)  
For more information:  
1. 

## About the repository:
1. Written in tensorflow 2.10 with cuda 11.8 and cudnn 8.x
2. "BOPINN.py" is the python file (run in spyder) and "BOPINN.ipynb" is a notebook (use colab or jupyter)
3. "PINN.py" is a PINN based solver for forward wave propagation problem. It's an auxillary code to understand the forward problem
4. "analytical.py" gives the exact solution of the wave equation with dirichlet BC and it is used to collect data (added white noise)  
5. data folder contains the snapshot observation collected from "analytical.py"  
6. lib folder has .py files required to run "PINN.py" and "BOPINN.py"
