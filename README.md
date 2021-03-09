# Solving high-dimensional eigenvalue problems using deep neural networks: A diffusion Monte Carlo like approach
Accompanying code for [*Solving high-dimensional eigenvalue problems using deep neural networks: A diffusion Monte Carlo like approach*](https://doi.org/10.1016/j.jcp.2020.109792), Jiequn Han, Jianfeng Lu, Mo Zhou, Journal of Computational Physics, 423, 109792 (2020). Deep learning based algorithms for solving high-dimensional eigenvalue problems.

Run main.py for the code. Before running, choose the proper config you want to run by modifying line 15: the name of the config you want to run.
"FPUni" is the Fokker Planck euqation with a uniform distribution as invariant measure (if you want a drift, change the "eqn_name" into "FPEigen" in the config); "Sdg" is the linear Schrodinger; "CubicSdg" is the cubic Schrodinger; "DWSep" is for the double well problem with well separated eigenvalues;  "DWClose" is for the double well problem with close eigenvalues.
If there is "ma" in the name of the json file, it means we use moving average for the normalization factor.
If there is "second" in the name of the json file, it means we are solving for the second eigenpair.

Also, you need to choose the model by having only one of the lines in 43-48 not commented.
When you choose the model, "linear" means we are solving linear eigenvalue problems (and similar for "nonlinear").
"Consist" means we use two neural networks to represent the eigenfunction and its scaled gradient, while "grad" means we use one neural network for the eigenfunction and use auto-differentiation for its gradient.
"build_true" is used confirm the code is written correctly.
"Double_well" means we are solving the eigenvalue problem with a double-well potential and we have the choice to choose either to solve for the first or second eigenpair.

The code can be run on either Tensorflow 2.0 directly or Tensorflow 1.0 with small modification. For instance, to use Tensorflow 1.12, change the two lines in three .py files
```
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```
into
```
import tensorflow as tf
```
