# BOPINN (Bayesian optimized physics-informed neural network)
PINN is the forward model whereas BOPINN is the inversion scheme which runs BO over PINN to estimate the parameter of the PDE. In BOPINN, a PINN utilizes a neural surrogate to solve the partial differential equation (wave propagation here). Bayesian optimization runs over the PINN model and estimates the optimum parameters (wave velocity in the medium here) using a single snapshot observation of the field.  
For more information:  
1. 

## About the repository:
1. Written in tensorflow 2.10 with cuda 11.8 and cudnn 8.x
2. "BOPINN.py" is the python file (run in spyder) and "BOPINN.ipynb" is a notebook (use colab or jupyter)
3. "PINN.py" is a PINN based solver for forward wave propagation problem. It's an auxillary code to understand the forward problem
4. "analytical.py" gives the exact solution of the wave equation with dirichlet BC and it is used to collect data (added white noise)  
5. data folder contains the snapshot observation collected from "analytical.py"  
6. lib folder has .py files required to run "PINN.py" and "BOPINN.py"
