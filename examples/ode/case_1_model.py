from pyomo import environ as po
from pyomo import dae as pod
# from scipy.integrate import odeint
# import numpy as np
from autograd import jacobian
from autograd.scipy.integrate import odeint
from matplotlib import pyplot as plt
import autograd.numpy as np
from autograd.builtins import tuple

""" define ODE model, and simulation using pyomo """
def create_pyomo_model(sampling_times):
    """
    Creates a pyomo model.

    Return:
        model       : the ConcreteModel instance.
        simulator   : Pyomo.DAE's simulator object attached to the model.
    """
    model = po.ConcreteModel()
    norm_spt = sampling_times / max(sampling_times)
    model.t = pod.ContinuousSet(bounds=(0, 1), initialize=norm_spt)
    model.tau = po.Var()

    model.ca = po.Var(model.t, bounds=(0, 50))  # set of concentrations in reaction mixture
    model.dca_dt = pod.DerivativeVar(model.ca, wrt=model.t)

    model.beta = po.Var()  # model parameters

    def _material_balance(m, t):
        return m.dca_dt[t] / m.tau == -m.beta * m.ca[t]
    model.material_balance = po.Constraint(model.t, rule=_material_balance)

    return model

def pyomo_simulate(ti_controls, sampling_times, model_parameters):
    """
    The simulation function to be passed on to the pydex.designer. The function takes in
    the pyomo model and simulator, nominal model parameter values, and experimental
    candidates. Returns the predictions at specified sampling times.

    Parameters:
        model               : a Pyomo model instance
        simulator           : a Pyomo simulator instance
        ti_controls         : time-invariant controls of candidate (1D np.array)
        sampling_times      : sampling time choices of candidate (1D np.array)
        model_parameters    : nominal model parameter values (1D np.array)
    Return:
        responses           : a 2D np.array with shapes N_spt by n_r, corresponding to
                              the model's prediction on value of the n_r number of
                              responses at all N_spt number of sampling times.
    """
    """ fixing the control variables """
    model = create_pyomo_model(sampling_times)

    # time-invariant
    model.beta.fix(model_parameters[0])
    model.tau.fix(max(sampling_times))
    model.ca[0].fix(ti_controls[0])
    # no time-varying control for this example

    """ simulating """
    simulator = pod.Simulator(model, "casadi")
    simulator.simulate(numpoints=100, integrator='idas')
    simulator.initialize_model()

    """" extracting results and returning it in appropriate format """
    normalized_sampling_times = sampling_times / model.tau.value
    ca = np.array([model.ca[t].value for t in normalized_sampling_times])

    return ca

""" define ODE model, and simulation using scipy """
def scipy_simulate(ti_controls, sampling_times, model_parameters):
    """
    The simulation function to be passed on to the pydex.designer. The function takes in
    the nominal model parameter values, and experimental candidates. Returns the
    predictions of the model at specified sampling times.

    Parameters:
        model               : a Pyomo model instance
        simulator           : a Pyomo simulator instance
        ti_controls         : time-invariant controls of candidate (1D np.array)
        sampling_times      : sampling time choices of candidate (1D np.array)
        model_parameters    : nominal model parameter values (1D np.array)
    Return
        responses           : a 2D np.array with shapes N_spt by n_r, corresponding to
                              the model's prediction on value of the n_r number of
                              responses at all N_spt number of sampling times.
    """

    def scipy_model(t, ca, theta):
        dca_dt = - theta[0] * ca
        return dca_dt

    sol = odeint(
        scipy_model,
        args=tuple((model_parameters,)),
        y0=ti_controls[0],
        t=sampling_times,
        tfirst=True
    )
    return sol

if __name__ == '__main__':
    spt = np.linspace(0, 10, 11)

    pyomo_model = create_pyomo_model(spt)
    y = pyomo_simulate(
        [1],
        spt,
        [0.25],
    )

    fig = plt.figure()
    axes = fig.add_subplot(121)
    axes.plot(
        spt,
        y
    )

    # dydtheta = jacobian(scipy_simulate, 2)
    # sens = dydtheta(
    #     np.array([1]),
    #     spt,
    #     np.array([0.25]),
    # )
    # axes2 = fig.add_subplot(122)
    # axes2.plot(
    #     spt,
    #     sens[:, 0, 0]
    # )
    plt.show()
