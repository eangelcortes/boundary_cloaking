import numpy as np
from lib.Boundary import xy_bdy

'''
grid.py 
===============

- x, y = make_grid_between(ngrid, Bdy1, Bdy2, M)

- x, y = make_grid(ngrid, Boundary, ν, M, D, direction)

- xbdys, ybdys, xs, ys = fullmake_grid(Mvals, nvals, Boundaries, Ds)

'''

def make_outer_grid(ngrid, Bdy, ν, M, D):
    xbdy, ybdy = xy_bdy(Bdy)
    dn = D/ngrid
    
    rgrid = np.linspace(D, 0+dn, num=ngrid,endpoint=True, retstep=False, dtype=None, axis=0)
    x  = np.zeros((M + 1, ngrid ))
    y  = np.zeros((M + 1, ngrid ))
    for m in range(0, M):
        x[m, :] = xbdy[m] + rgrid*ν[0][m]
        y[m, :] = ybdy[m] + rgrid*ν[1][m]
    
    #Impose Periodicity 
    x[M][:] = x[0][:]
    y[M][:] = y[0][:]
    
    return x, y


def make_grid_between(ngrid, Bdy1, Bdy2, M):
    '''
    Computes the grid points between two boundaries
    
    Parameters
    ==========
    ngrid : number of pts along direction vector
    Bdy1 : Exterior boundary
    Bdy2 : Interior boundary
    M : number of pts on boundary

    
    Returns
    =======
    x : x pts in layer
    y : y pts in layer
    
    '''
    xbdy1, ybdy1 = xy_bdy(Bdy1)
    xbdy2, ybdy2 = xy_bdy(Bdy2)
    
    x  = np.zeros((M + 1, ngrid ))
    y  = np.zeros((M + 1, ngrid ))

    for m in range(0, M):
        x_diff = xbdy2[m] - xbdy1[m]
        y_diff = ybdy2[m] - ybdy1[m]
        dist = np.sqrt(x_diff**2 + y_diff**2)
    
        dn = dist/ngrid
        rgrid = np.linspace(0+dn, dist, num=ngrid,endpoint=False, retstep=False, dtype=None, axis=0)
        
        x[m, :] = xbdy1[m] + rgrid*(x_diff / dist)
        y[m, :] = ybdy1[m] + rgrid*(y_diff / dist)
    
    #Impose Periodicity 
    x[M][:] = x[0][:]
    y[M][:] = y[0][:]
    
    return x, y

def make_grid(ngrid, Boundary, ν, M, D, direction):
    '''
    Computes the grid points along normal from boundary layer
    
    Parameters
    ==========
    ngrid : number of pts along normal
    xbdy : x pts of boundary
    ybdy : y pts of boundary
    ν : normal vector of boundary
    M : number of pts on boundary
    D : depth of  layer
    direction : +1 for outward normal, -1 for inward normal
    
    Returns
    =======
    x : x pts in layer
    y : y pts in layer
    
    '''
    xbdy, ybdy = xy_bdy(Boundary)
    dn = D/ngrid
    
    rgrid = np.linspace(0+dn, D, num=ngrid,endpoint=True, retstep=False, dtype=None, axis=0)
    x  = np.zeros((M + 1, ngrid ))
    y  = np.zeros((M + 1, ngrid ))
    for m in range(0, M):
        x[m, :] = xbdy[m] + direction*rgrid*ν[0][m]
        y[m, :] = ybdy[m] + direction*rgrid*ν[1][m]
    
    #Impose Periodicity 
    x[M][:] = x[0][:]
    y[M][:] = y[0][:]
    
    return x, y

def fullmake_grid(M, ngrid, Boundaries, Ds):
    '''
    Finds grid points on each boundary and in the layers between them according to number of points chosen in Mvals and nvals
    
    Parameters
    ==========
    M : number of pts on boundaries
    ngrid: number of pts along normal
    Boundaries : list of boundaries 
    
    Returns
    =======
    xbdys : x pts of each boundary
    ybdys : y pts of each boundary
    xs : x pts of each layer
    ys : y pts of each layer
    
    '''
    xbdys = []
    ybdys = []
    
    xs = []
    ys = []
    
    θ, dt = np.linspace(0, 2*np.pi, num=M, endpoint=False, retstep=True)
    for m in range(0, len(Boundaries)):
        
        
        xbdy, ybdy = xy_bdy(Boundaries[m].y_l(θ))
        xbdys.append(xbdy)
        ybdys.append(ybdy)
        if m == 0:
            x,y = make_outer_grid(ngrid, Boundaries[m].y_l(θ), Boundaries[m].ν_l(θ), M, Ds[0])
            xs.append(x)
            ys.append(y)
            
        if m < (len(Boundaries) - 1):
            x,y = make_grid_between(ngrid, Boundaries[m].y_l(θ), Boundaries[m+1].y_l(θ), M)
            xs.append(x)
            ys.append(y)
            
        if m == (len(Boundaries) - 1):
            x,y = make_grid(ngrid, Boundaries[m].y_l(θ), Boundaries[m].ν_l(θ), M, Ds[1], -1)
            xs.append(x)
            ys.append(y)
    
    return xbdys, ybdys, xs, ys
