Discrete optimal EXperiment design
============================================
An open-source Python package for optimal experiment design, essential to a modeller's toolbelt. If you are someone who develops a model of any kind, you will relate to the difficulty of estimating its model parameters. This tool will help design maximally informative experiments for collecting data to calibrate your model. This package is a simple and powerful toolkit you must have. With an intuitive syntax and helpful documentation, it should take you little time to start designing optimal experiments for estimating the parameters of your model.

## Installation
  * PyPI

        $ pip install pydex

## Quick Start
  * [IPython demo](
  https://colab.research.google.com/drive/1mGE8-WBhT5rAEQvQOaT1IBorGeHemyVG
  ) of basic features.
  

## Features:
1. Strive to be as simple as possible to use, but not simpler.
2. Designs continuous experimental designs.
3. Interfaces to optimization solvers through scipy, and cvxpy.
4. Convenient built-in visualization capabilities.
5. Supports virtually any model, as long as it can be written as a Python function.

## Dependencies:
1. matplotlib: used for visualization.
2. numdifftools: used for numerical estimation of parameter sensitivities.
3. scipy: used for numerical optimization.
4. numpy: a core package.
5. pickle: for saving objects (results, data, etc.).
6. dill: for saving objects with weak-references.
7. python 3.x: a core package.

## Examples
In this repository, you will find examples of designing optimal experiments for a handful response surface models, and models made of a system of ordinary differential equations. Currently, there are examples for using scipy and pyomo for integrating the ODE models. Syntax for solution, visualization, and saving & loading progress are shown in the examples.

Do you feel something is confusing? Something can be improved? Interested to contribute? Have a feature to request? Feel free to drop a line at: kennedy.putra.kusumo@gmail.com.
