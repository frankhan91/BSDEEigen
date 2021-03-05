# BSDEEigen
This is the code for https://www.sciencedirect.com/science/article/pii/S0021999120305660?casa_token=5-WZ8YH-1zAAAAAA:GLihySDuAHymfccXU_XIqbceARbjM7qUOgXOZmW37e6Zldc7v5GLXiu0YMX2v8TgVbFyKqyluQY.
We solve high-dimensional eigenvalue problems with deep neural networks.

The code can be run on either Tensorflow 1.0 or Tensorflow 2.0. For Tensorflow 1.0, please change the two lines
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
into:
import tensorflow as tf
For Tensorflow 2.0, run the code directly.

Run main.py for the code. Before running, choose the proper config you want to run by modifying line 15: the name of the config you want to run.
"FPUni" is the Fokker Planck; "Sdg" is the linear Schrodinger; "CubicNew" is the cubic Schrodinger; "DoubleWell is for the double well problem".
If there is "ma" in the json file, it means we use moving average for the normalization factor.
If there is "second" in the json file, it means we are solving for the second eigenpair.
You can also modify the parameters in teh configs.

Also, you need to choose the model by having only one ofthe lines in 43-48 not commented.
When you choose the model, "linear" means we are solving linear eigenvalue problems (and similar for "nonlinear").
"Consist" means we use two neural networks to represent the eigenfunction and its scaled gradient, while "grad" means we use one neural network for the eigenfunction and use auto-differentiation for its gradient.
"build_true" is used confirm the code is written correctly.
"Double_well" means we are solving the eigenvalue problem with a double-well potential and we have the choice to choose either to solve for the first or second eigenpair.
