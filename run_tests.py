import os

os.chdir("examples/non_linear/")
from examples.non_linear import test_non_linear
os.chdir("..")
os.chdir("linear/")
from examples.linear import test_linear_examples
os.chdir("..")
os.chdir("ode/")
from examples.ode import test_ode
os.chdir("..")
os.chdir("non_cubic_spaces/")
from examples.non_cubic_spaces import test_non_cubic_spaces
os.chdir("..")
os.chdir("time_varying_controls/")
from examples.time_varying_controls import test_tvc

print("All tests are sucessful!")
