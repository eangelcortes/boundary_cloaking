import sympy as sp
#from sympy import cos, sin, exp
#CC: is this above line useful here ? 
t = sp.Symbol('t')

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
        return [sp.diff(p, t) for p in self.y]
    @property
    def ypp(self):
        return [sp.diff(p, t) for p in self.yp]
    @property
    def J(self):
        v2 = [i**2 for i in self.yp]
        sv2 = sum(v2)
        return sp.sqrt(sv2).simplify()
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
            
    def items(self): # all elements of the class
        list_elem = ['y', 'yp', 'ypp', 'J', 'τ', 'ν', 'κ']
        return [getattr(self,p) for p in list_elem]
    
    def items_lambdified(self): # lambdified all elements
        return [sp.lambdify(t, p) for p in self.items()]
            
def lclass(cls, attributes):
    '''
    Lambdify a class with specific attributes
    
    Parameters
    ==========
    cls: class
    attributes: attributes of the class cls
    
    Returns
    ==========
    cls: class lambdified   
    '''
    for a in attributes:
        symbolic = getattr(cls, a)
        lam = sp.lambdify(t, symbolic)
        setattr(cls, a+'_l', lam)
    return cls

def lclassB(B):
    '''
    Lambdify the Boundary class with specific attributes
    
    Parameters
    ==========
    cls: class
    
    Returns
    ==========
    cls: class lambdified   
    '''
    attributes = ['y', 'yp', 'ypp', 'J', 'τ', 'ν', 'κ']
    for a in attributes:
        symbolic = getattr(B, a)
        lam = sp.lambdify(t, symbolic)
        setattr(B, a+'_l', lam)
    return B

# def sym_to_num(t, B):
#     '''
#     Convert the Boundary class elements (sympy) via lamdify (numpy)
    
#     Inputs
#     ==========
#     t: symbol used in the parameterization B
#     B: Boundary, define via the class Boundary
#     Outputs
#     ==========
#     bdy: lambdified B.y
#     bdy_p: lambdified B.yp
#     bdy_pp: lambdified B.ypp
#     J: lambdified B.J
#     τ: lambdified B.τ
#     ν: lambdified B.ν 
#     κ: lambdified B.κ
#     '''
#     bdy = sp.lambdify(t, B.y)
#     bdy_p = sp.lambdify(t, B.yp)
#     bdy_pp = sp.lambdify(t, B.ypp)
#     τ = sp.lambdify(t, B.τ)
#     ν = sp.lambdify(t, B.ν)
#     J = sp.lambdify(t, B.J)
#     κ = sp.lambdify(t, B.κ)
#     return bdy, bdy_p, bdy_pp, J, τ, ν, κ
# CC: I suggest to convert all elements of the class

#Note: dir(E) allows to looks at all attributes of E
#      getattr(E, a) get the attribute a in E 

