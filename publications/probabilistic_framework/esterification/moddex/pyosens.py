from pyomo.dae import Simulator
from pyomo.dae.simulator import convert_pyomo2casadi
from pyomo.core.base import Constraint, Param, value, Suffix, Block

from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error

from pyomo.core.expr import current as EXPR
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer

from six import iterkeys

import logging
import sys
import inspect

__all__ = ('Simulator', )
logger = logging.getLogger('pyomo.core')

from pyomo.common.dependencies import (
    numpy as np, numpy_available, attempt_import,
)

# Check integrator availability
# scipy_available = True
# try:
#     import platform
#     if platform.python_implementation() == "PyPy":  # pragma:nocover
#         # scipy is importable into PyPy, but ODE integrators don't work. (2/18)
#         raise ImportError
#     import scipy.integrate as scipy
# except ImportError:
#     scipy_available = False
import platform
is_pypy = platform.python_implementation() == "PyPy"

scipy, scipy_available = attempt_import('scipy.integrate', alt_names=['scipy'])

casadi_intrinsic = {}
def _finalize_casadi(casadi, available):
    if available:
        casadi_intrinsic.update({
            'log': casadi.log,
            'log10': casadi.log10,
            'sin': casadi.sin,
            'cos': casadi.cos,
            'tan': casadi.tan,
            'cosh': casadi.cosh,
            'sinh': casadi.sinh,
            'tanh': casadi.tanh,
            'asin': casadi.asin,
            'acos': casadi.acos,
            'atan': casadi.atan,
            'exp': casadi.exp,
            'sqrt': casadi.sqrt,
            'asinh': casadi.asinh,
            'acosh': casadi.acosh,
            'atanh': casadi.atanh,
            'ceil': casadi.ceil,
            'floor': casadi.floor,
        })
casadi, casadi_available = attempt_import('casadi', callback=_finalize_casadi)



class PyosensSimulator(Simulator):
    def __init__(self, m, package='scipy'):
        super().__init__(m, package)
        # Inputs for sensitivity analysis
        self._simsensvars = None
        self._varying_inputs = {}
        self.do_sensitivity_analysis = False
        self.detect_sensitivity_analyis = False
        self.sensitivity_analysis_function = "eval_sensitivities"
        self.finite_difference_module = "numdifftools"

    def get_variable_order(self, vartype=None):
        """
        This function returns the ordered list of differential variable
        names. The order corresponds to the order being sent to the
        integrator function. Knowing the order allows users to provide
        initial conditions for the differential equations using a
        list or map the profiles returned by the simulate function to
        the Pyomo variables.
        Parameters
        ----------
        vartype : `string` or None
            Optional argument for specifying the type of variables to return
            the order for. The default behavior is to return the order of
            the differential variables. 'time-varying' will return the order
            of all the time-dependent algebraic variables identified in the
            model. 'algebraic' will return the order of algebraic variables
            used in the most recent call to the simulate function. 'input'
            will return the order of the time-dependent algebraic variables
            that were treated as inputs in the most recent call to the
            simulate function.
        Returns
        -------
        `list`
        """
        if vartype == 'sensitivity':
            return [self._simsensvars[h]
                    for h in self._varying_inputs.keys() if self._simsensvars[h]]
        else:
            return super().get_variable_order(vartype)

    def simulate(self, numpoints=None, tstep=None, integrator=None,
                 varying_inputs=None, initcon=None, integrator_options=None):
        """
        Simulate the model. Integrator-specific options may be specified as
        keyword arguments and will be passed on to the integrator.
        Parameters
        ----------
        numpoints : int
            The number of points for the profiles returned by the simulator.
            Default is 100
        tstep : int or float
            The time step to use in the profiles returned by the simulator.
            This is not the time step used internally by the integrators.
            This is an optional parameter that may be specified in place of
            'numpoints'.
        integrator : string
            The string name of the integrator to use for simulation. The
            default is 'lsoda' when using Scipy and 'idas' when using CasADi
        varying_inputs : ``pyomo.environ.Suffix``
            A :py:class:`Suffix<pyomo.environ.Suffix>` object containing the
            piecewise constant profiles to be used for certain time-varying
            algebraic variables.
        initcon : list of floats
            The initial conditions for the the differential variables. This
            is an optional argument. If not specified then the simulator
            will use the current value of the differential variables at the
            lower bound of the ContinuousSet for the initial condition.
        integrator_options : dict
            Dictionary containing options that should be passed to the
            integrator. See the documentation for a specific integrator for a
            list of valid options.
        Returns
        -------
        numpy array, numpy array
            The first return value is a 1D array of time points corresponding
            to the second return value which is a 2D array of the profiles for
            the simulated differential and algebraic variables.
        """

        if not numpy_available:
            raise ValueError("The numpy module is not available. "
                              "Cannot simulate the model.")

        if integrator_options is None:
            integrator_options = {}

        if self._intpackage == 'scipy':
            # Specify the scipy integrator to use for simulation
            valid_integrators = ['vode', 'zvode', 'lsoda', 'dopri5', 'dop853']
            if integrator is None:
                integrator = 'lsoda'
            elif integrator == 'odeint':
                integrator = 'lsoda'
        else:
            # Specify the casadi integrator to use for simulation.
            # Only a subset of these integrators may be used for
            # DAE simulation. We defer this check to CasADi.
            valid_integrators = ['cvodes', 'idas', 'collocation', 'rk']
            if integrator is None:
                integrator = 'idas'

        if integrator not in valid_integrators:
            raise DAE_Error("Unrecognized %s integrator \'%s\'. Please select"
                            " an integrator from %s" % (self._intpackage,
                                                        integrator,
                                                        valid_integrators))

        # Set the time step or the number of points for the lists
        # returned by the integrator
        if tstep is not None and \
           tstep > (self._contset.last() - self._contset.first()):
            raise ValueError(
                "The step size %6.2f is larger than the span of the "
                "ContinuousSet %s" % (tstep, self._contset.name()))

        if tstep is not None and numpoints is not None:
            raise ValueError(
                "Cannot specify both the step size and the number of "
                "points for the simulator")
        if tstep is None and numpoints is None:
            # Use 100 points by default
            numpoints = 100

        if tstep is None:
            tsim = np.linspace(
                self._contset.first(), self._contset.last(), num=numpoints)

            # Consider adding an option for log spaced time points. Can be
            # important for simulating stiff systems.
            # tsim = np.logspace(-4,6, num=100)
            # np.log10(self._contset.first()),np.log10(
            # self._contset.last()),num=1000, endpoint=True)

        else:
            tsim = np.arange(
                self._contset.first(), self._contset.last(), tstep)

        switchpts = []
        self._siminputvars = {}
        self._simalgvars = []
        self._simsensvars = {}
        if varying_inputs is not None:
            if type(varying_inputs) is not Suffix:
                raise TypeError(
                    "Varying input values must be specified using a "
                    "Suffix. Please refer to the simulator documentation.")

            self._hash_indexed_inputs(varying_inputs)
            for alg in self._algvars:
                if alg._hash in self._varying_inputs:
                    # Find all the switching points
                    newpts = self._varying_inputs[alg._hash].keys()
                    switchpts += newpts
                    # Add to dictionary of siminputvars
                    self._siminputvars[alg] = alg
                    if len(newpts) == 1:
                        self._simsensvars[alg._hash] = alg
                    else:
                        self._simsensvars[alg._hash] = None
                else:
                    self._simalgvars.append(alg)

            if self._intpackage == 'scipy' and len(self._simalgvars) != 0:
                raise DAE_Error("When simulating with Scipy you must "
                                "provide values for all parameters "
                                "and algebraic variables that are indexed "
                                "by the ContinuoutSet using the "
                                "'varying_inputs' keyword argument. "
                                "Please refer to the simulator documentation "
                                "for more information.")

            # Get the set of unique points
            switchpts = list(set(switchpts))
            switchpts.sort()

            # Make sure all the switchpts are within the bounds of
            # the ContinuousSet
            if switchpts[0] < self._contset.first() or \
                            switchpts[-1] > self._contset.last():
                raise ValueError("Found a switching point for one or more of "
                                 "the time-varying inputs that is not within "
                                 "the bounds of the ContinuousSet.")

            # Update tsim to include input switching points
            # This numpy function returns the unique, sorted points
            tsim = np.union1d(tsim, switchpts)
        else:
            self._simalgvars = self._algvars

        # Check if initial conditions were provided, otherwise obtain
        # them from the current variable values
        if initcon is not None:
            if len(initcon) > len(self._diffvars):
                raise ValueError(
                    "Too many initial conditions were specified. The "
                    "simulator was expecting a list with %i values."
                    % len(self._diffvars))
            if len(initcon) < len(self._diffvars):
                raise ValueError(
                    "Too few initial conditions were specified. The "
                    "simulator was expecting a list with %i values."
                    % len(self._diffvars))
        else:
            initcon = []
            for v in self._diffvars:
                for idx, i in enumerate(v._args):
                    if type(i) is IndexTemplate:
                        break
                initpoint = self._contset.first()
                vidx = tuple(v._args[0:idx]) + (initpoint,) + \
                       tuple(v._args[idx + 1:])
                # This line will raise an error if no value was set
                initcon.append(value(v._base[vidx]))

        # Call the integrator
        if self._intpackage == 'scipy':
            if not scipy_available:
                raise ValueError("The scipy module is not available. "
                                 "Cannot simulate the model.")
            if is_pypy:
                raise ValueError("The scipy ODE integrators do not work "
                                 "under pypy. Cannot simulate the model.")
            tsim, profile = self._simulate_with_scipy(initcon, tsim, switchpts,
                                                      varying_inputs,
                                                      integrator,
                                                      integrator_options)
        else:

            if len(switchpts) != 0:
                tsim, profile = \
                    self._simulate_with_casadi_with_inputs(initcon, tsim,
                                                           varying_inputs,
                                                           integrator,
                                                           integrator_options)
            else:
                tsim, profile = \
                    self._simulate_with_casadi_no_inputs(initcon, tsim,
                                                         integrator,
                                                         integrator_options)

        self._tsim = tsim
        if self.do_sensitivity_analysis:
            self._simsolution = profile[0]
        else:
            self._simsolution = profile

        return [tsim, profile]

    def _simulate_with_casadi_with_inputs(self, initcon, tsim, varying_inputs,
                                          integrator, integrator_options):

        xalltemp = [self._templatemap[i] for i in self._diffvars]
        xall = casadi.vertcat(*xalltemp)

        time = casadi.SX.sym('time')

        odealltemp = [time * convert_pyomo2casadi(self._rhsdict[i])
                      for i in self._derivlist]
        odeall = casadi.vertcat(*odealltemp)

        # Time-varying inputs
        ptemp = [self._templatemap[i]
                 for i in self._siminputvars.values()]
        pall = casadi.vertcat(time, *ptemp)

        dae = {'x': xall, 'p': pall, 'ode': odeall}

        if len(self._algvars) != 0:
            zalltemp = [self._templatemap[i] for i in self._simalgvars]
            zall = casadi.vertcat(*zalltemp)
            # Need to do anything special with time scaling??
            algalltemp = [convert_pyomo2casadi(i) for i in self._alglist]
            algall = casadi.vertcat(*algalltemp)
            dae['z'] = zall
            dae['alg'] = algall

        # This approach removes the time scaling from tsim so must
        # create an array with the time step between consecutive
        # time points
        tsimtemp = np.hstack([0, tsim[1:] - tsim[0:-1]])
        tsimtemp.shape = (1, len(tsimtemp))

        palltemp = [casadi.DM(tsimtemp)]

        # Need a similar np array for each time-varying input
        for p in self._siminputvars.keys():
            profile = self._varying_inputs[p._hash]
            tswitch = list(profile.keys())
            tswitch.sort()
            tidx = [tsim.searchsorted(i) for i in tswitch] + \
                   [len(tsim) - 1]
            ptemp = [profile[0]] + \
                    [casadi.repmat(profile[tswitch[i]], 1,
                                   tidx[i + 1] - tidx[i])
                     for i in range(len(tswitch))]
            temp = casadi.horzcat(*ptemp)
            palltemp.append(temp)

        profile = self._simulate_with_casadi_with_sens(dae, tsim, initcon, casadi.vertcat(*palltemp),
                                                       varying_inputs, integrator, integrator_options)

        return [tsim, profile]

    def _simulate_with_casadi_with_sens(self, dae, tsim, initcon, pall, varying_inputs,
                                        integrator, integrator_options):
        nvar = dae['x'].shape[0]
        npar = dae['p'].shape[0] - 1
        psens = self._simsensvars
        nsens = sum(bool(v) for v in psens.values())

        # integrator function definition
        integrator_options['tf'] = 1
        F = casadi.integrator('F', integrator, dae, integrator_options)
        N = len(tsim)
        I = F.mapaccum('simulator', N)
        sol = I(x0=initcon, p=pall)
        profile = sol['xf'].full().T

        if len(self._algvars) != 0:
            algprofile = sol['zf'].full().T
            profile = np.concatenate((profile, algprofile), axis=1)

        if self.detect_sensitivity_analyis:
            self._detect_sensitivity_analysis_function()
        if not self.do_sensitivity_analysis:
            return profile

        sens = []
        # Evaluate vector jacobian product for each parameter
        xfwd = I.factory('xfwd', ['x0', 'p', 'fwd:p'], ['fwd:xf'])
        lh = list(psens.keys())
        xfp = [xfwd(x0=initcon, p=pall,
                    fwd_p=[1 if j == lh.index(h)+1 else 0 for j in range(npar + 1)])['fwd_xf']
               for h in self._varying_inputs.keys() if psens[h]]

        # shape of vjp is (nvar, N) for each parameter
        for i in range(N):
            # split first column of vjp from remaining columns for each parameter
            psplit = [casadi.horzsplit(xfp[ipar], [0, 1, N - i]) for ipar in range(nsens)]
            # concatenate columns for each parameter
            sens.append(casadi.horzcat(*[psplit[ipar][0] for ipar in range(nsens)]))
            # remaining columns for next iteration
            xfp = [psplit[ipar][1] for ipar in range(nsens)]

        # convert casadi matrix of sensitivities at each time point to numpy array
        sensprofile = np.array([x.full() for x in sens])

        return profile, sensprofile

    def _detect_sensitivity_analysis_function(self):
        self.do_sensitivity_analysis = False
        frame = sys._getframe(1)
        while frame:
            if self.finite_difference_module in frame.f_code.co_filename:
                self.do_sensitivity_analysis = False
                break
            if frame.f_code.co_name is self.sensitivity_analysis_function:
                self.do_sensitivity_analysis = True
                break
            frame = frame.f_back
        return

    def _hash_indexed_inputs(self, varying_inputs):
        self._varying_inputs = {}
        if varying_inputs is None:
            return

        for k, v in varying_inputs.items():
            if isinstance(k, EXPR.GetItemExpression):
                expr = k
            else:
                expr = EXPR.GetItemExpression([k, IndexTemplate(k._index)])
            indexer = _GetItemIndexer(expr)
            self._varying_inputs[indexer._hash] = v

        return