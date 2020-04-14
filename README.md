**D**iscrete Optimal **Ex**periment Design
An open-source Python package for optimal experiment design. The optimal experimental design technique is essential to a modeller's toolbelt. If you are someone who develops a model of any kind, you will relate to the difficulty of estimating its model parameters. This tool will help design experiments for collecting essential data to develop your model. This package is a simple and powerful toolkit you must have. With an intuitive syntax and helpful documentation, it should take you little time to start designing optimal experiments for estimating the parameters of your model.

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
In this repository, you will find examples of designing an optimal experiment for a model made of ordinary differential equations. Currently, there are examples for using scipy and pyomo for solving the model. Syntax for solution, visualization, and saving & loading progress is shown in the examples.

Do you feel something is confusing? Something can be improved? Interested to contribute? Have a feature to request? Feel free to drop a line at: kennedy.putra.kusumo@gmail.com.
