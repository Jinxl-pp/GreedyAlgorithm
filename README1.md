# GreedyAlgorithm
python/Matlab codes of solving PDEs with greedy algorithms written by Xianlin Jin

# 2022.01.06
# Added python codes for Orthogonal Greedy Algorithm
# So far I only tested the 1d second order elliptic equations and 2d function fitting.

# You can run the code with

python oga_train.py --dim 1 --k 2 --num_neuron 256 --eqn_order 2 --h1 '1/1000' --h2 '1/1000' --h3 '1/4000' --pde 'cos1d' --trainer 'gd'

# and 

python oga_train.py --dim 2 --k 2 --num_neuron 256 --eqn_order 0 --h1 '1/100' --h2 '1/100' --h3 '1/200' --pde 'cos2d' --trainer 'gd'

# Here is the parameter description:

--dim, the dimension of the equation, use with 1 2 or 3.

--k, the power of the ReLU activation function, use correctly with integers and by the order of PDEs.

--num_neuron, the final number of neurons in the shallow network.

--eqn_order, the order of PDEs, use with 0,2,4,6,... where 0 stands for the function fitting.

--h1, the mesh size of the grid for generating the quadrature points for computing integrals(training process).

--h2, the mesh size of the grid of parameter domain for the initial guess.

--h3, the mesh size of the grid for generating the quadrature points for computing the numerical error(testing process).

--pde, the name of exact solutions, use with
                          1. 1D functions 
                             1.1. u = cos(pi*x) 
                                  pde = 'cos1d'
                             1.2. u = ...
                          2. 2D functions
                             2.1. u = cos(2*pi*x)*cos(2*pi*y)
                                  pde = 'cos2d'
                             2.2. u = ...
                          3. 3D functions
                             3.1. u = cos(2*pi*x)*cos(2*pi*y)*cos(2*pi*z)
                                  pde = 'cos3d'
                             3.2. u = ...

--trainer, the optimization algorithm for training the parameters from initial guesses, use with 'gd'
or 'newton'

