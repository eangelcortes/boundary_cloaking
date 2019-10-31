import sympy
from sympy import cos, sin, exp

class Boundary:
    '''
    Defines the class Boundary, that contains geometric features of a boundary
    
    Attributes
    ==========
    param: parametrization of a boundary
    veloc: derivative of the parametrization
    accel: second derivative of the parametrization
    normal: normal vector to the boundary
    tangent: tangent vectors to the boundary
    curvature: signed curvature of the boundary
    jacobian: Jacobian of the boundary
    '''
    
    def __init__(self, b):
        self.y = b
    
    @property
    def yp(self):
        return [sympy.diff(p, t) for p in self.y]
    @property
    def ypp(self):
        return [sympy.diff(p, t) for p in self.yp]
    @property
    def J(self):
        v2 = [i**2 for i in self.yp]
        sv2 = sum(v2)
        return sympy.sqrt(sv2).simplify()
    @property
    def τ(self):
        return [p/self.J for p in self.yp]
    @property
    def ν(self):
        if len(self.y) == 2:
            tmp = (self.yp[1]/self.J, -self.yp[0]/self.J )
            return tmp
        else:
            raise ValueError('Need to define the normal vector for higher dimensions')
    @property
    def κ(self): #curvature kappa
        if len(self.y) == 2:
            tmp = (self.yp[0]*self.ypp[1] - self.yp[1]*self.ypp[0])/self.J**3  
            return tmp.simplify()
        else:
            raise ValueError('Need to define the mean curvature for higher dimensions')
            
t = sympy.Symbol('t')

def sym_to_num(t, B):
    bdy = sympy.lambdify(t, B.y)
    ν = sympy.lambdify(t, B.ν)
    J = sympy.lambdify(t, B.J)
    κ = sympy.lambdify(t, B.κ)
    return bdy, ν, J, κ