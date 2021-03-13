# Solving high-dimensional eigenvalue problems using deep neural networks: A diffusion Monte Carlo like approach
Accompanying code for [*Solving high-dimensional eigenvalue problems using deep neural networks: A diffusion Monte Carlo like approach*](https://doi.org/10.1016/j.jcp.2020.109792), Jiequn Han, Jianfeng Lu, Mo Zhou, Journal of Computational Physics, 423, 109792 (2020).

Run the following command to solve the eigenvalue problem directly:
```
python main.py --config_path=configs/fp_d2.json
```

**Names of config files**:
"FP" denotes the Fokker-Planck opeartor;
"Sdg" denotes the linear Schrodinger opeartor with a well-behaved potential;
"CubicSdg" denotes the nonlinear Schrodinger operator with a cubic term;
"DWSep" denotes the linear Schrodinger opeartor with a double well potential leading to well separated eigenvalues;
"DWClose" denotes the linear Schrodinger opeartor with a double well potential leading to close eigenvalues.

If there is "ma" in the name of the config file, it means to use moving average for the normalization factor.
If there is "second" in the name of the config file, it means to solve the second eigenpair.

| Experiments in the paper                                     | Config names                                                 |
|--------------------------------------------------------------|--------------------------------------------------------------|
| Fokker-Planck (Figure 2)                                     | fp_d5_ma.json, fp_d10_ma.json                                |
| Linear Schrodinger (Figure 3)                                | sdg_d5_ma.json, sdg_d10_ma.json                              |
| Nonlinear Schrodinger (Figure 4)                             | cubicsdg_d5_ma.json, cubicsdg_d10_ma.json                    |
| Linear Schrodinger with double-well potential (Figure 5)     | dwsep_d10_ma.json, dwsep_d10_ma_second.json                  |


**Fileds in config files**
"problem_type": "linear" means to solve linear eigenvalue problems (and similar for "nonlinear").
"eigenpair": "first" means to solve the first eigenpair (and similar for "second").
"model_type": "consistent" means to use two neural networks to represent the eigenfunction and its scaled gradient, while "gradient" means to use one neural network for the eigenfunction and call auto-differentiation to get its gradient.

The code can be run on either Tensorflow 2.0 directly or Tensorflow 1.0 with small modification. For instance, to use Tensorflow 1.12, change the two lines in three .py files
```
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```
into
```
import tensorflow as tf
```
