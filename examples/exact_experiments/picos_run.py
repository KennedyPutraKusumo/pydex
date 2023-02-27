import cvxopt as cvx
import picos
#---------------------------------#
# First generate some data :      #
#       _ a list of 8 matrices A  #
#       _ a vector c              #
#---------------------------------#
A = [cvx.matrix([[1,0,0,0,0],
                 [0,3,0,0,0],
                 [0,0,1,0,0]]),
     cvx.matrix([[0,0,2,0,0],
                 [0,1,0,0,0],
                 [0,0,0,1,0]]),
     cvx.matrix([[0,0,0,2,0],
                 [4,0,0,0,0],
                 [0,0,1,0,0]]),
     cvx.matrix([[1,0,0,0,0],
                 [0,0,2,0,0],
                 [0,0,0,0,4]]),
     cvx.matrix([[1,0,2,0,0],
                 [0,3,0,1,2],
                 [0,0,1,2,0]]),
     cvx.matrix([[0,1,1,1,0],
                 [0,3,0,1,0],
                 [0,0,2,2,0]]),
     cvx.matrix([[1,2,0,0,0],
                 [0,3,3,0,5],
                 [1,0,0,2,0]]),
     cvx.matrix([[1,0,3,0,1],
                 [0,3,2,0,0],
                 [1,0,0,2,0]])]

c = cvx.matrix([1,2,3,4,5])
c_primal_SOCP = picos.Problem()
AA = [picos.Constant('A[{0}]'.format(i), Ai)
      for i, Ai in enumerate(A)] # each AA[i].T is a 3 x 5 observation matrix
s  = len(AA)
cc = picos.Constant('c', c)
z  = [picos.RealVariable('z[{0}]'.format(i), AA[i].size[1])
       for i in range(s)]
mu = picos.RealVariable('mu', s)

cones = c_primal_SOCP.add_list_of_constraints([abs(z[i]) <= mu[i] for i in range(s)])
lin   = c_primal_SOCP.add_constraint(picos.sum([AA[i] * z[i] for i in range(s)]) == cc)
c_primal_SOCP.set_objective('min', (1|mu) )
print(c_primal_SOCP)
D_SOCP = picos.Problem()
m  = AA[0].size[0]
mm = picos.Constant('m', m)
L = picos.RealVariable('L', (m,m))
V = [picos.RealVariable('V['+str(i)+']', AA[i].T.size) for i in range(s)]
w = picos.RealVariable('w',s)
# additional variable to handle the geometric mean in the objective function
t = picos.RealVariable('t',1)
# define the constraints and objective function
lin_cons = D_SOCP.add_constraint(picos.sum([AA[i]*V[i] for i in range(s)]) == L)
# L is lower triangular
lowtri_cons = D_SOCP.add_list_of_constraints( [L[i,j] == 0
               for i in range(m)
               for j in range(i+1,m)])
cone_cons = D_SOCP.add_list_of_constraints([abs(V[i]) <= (mm**0.5)*w[i]
                                                for i in range(s)])
wgt_cons = D_SOCP.add_constraint(1|w <= 1)
geomean_cons = D_SOCP.add_constraint(t <= picos.geomean(picos.maindiag(L)))
D_SOCP.set_objective('max',t)
print(D_SOCP)
solution = D_SOCP.solve(solver='cvxopt')
print(w)
# create the problem, variables and params
D_exact = picos.Problem()
L = picos.RealVariable('L',(m,m))
V = [picos.RealVariable('V['+str(i)+']',AA[i].T.size) for i in range(s)]
T = picos.RealVariable('T', (s,m))
n = picos.IntegerVariable('n', s)
N = picos.Constant('N', 20)
# additional variable to handle the geomean inequality
t = picos.RealVariable('t',1)
# define the constraints and objective function
lin_cons = D_exact.add_constraint(
        picos.sum([AA[i]*V[i] for i in range(s)]) == L)
# L is lower triangular
lowtri_cons = D_exact.add_list_of_constraints( [L[i,j] == 0
                                 for i in range(m)
                                 for j in range(i+1,m)])
cone_cons = D_exact.add_list_of_constraints([ abs(V[i][:,k])**2 <= n[i]/N*T[i,k]
                for i in range(s) for k in range(m)])
lin_cons2 = D_exact.add_list_of_constraints([(1|T[:,k]) <= 1
                      for k in range(m)])
wgt_cons = D_exact.add_constraint(1|n <= N)
geomean_cons = D_exact.add_constraint(t <= picos.geomean( picos.maindiag(L)))
D_exact.set_objective('max',t)
print(D_exact)
from time import time
start = time()
solution = D_exact.solve()
print(f"Exact design solution took {time()- start:.2f} s")
print(n)
