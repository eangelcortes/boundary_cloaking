import numpy as np
from numpy import sqrt, log


def xy_bdy(bdy):
    xbdy = bdy[0]
    ybdy = bdy[1]
    return xbdy, ybdy

def find_exact_planewave(x, y, k, alpha):
    exact = np.exp(1j * k * (np.cos(alpha) * x + np.sin(alpha) * y))
    return exact

def make_grid_between(ngrid, Bdy1, Bdy2, M):
    '''
    Computes the grid points along normal from boundary layer
    
    Parameters
    ==========
    ngrid : number of pts along direction vector
    xbdy : x pts of boundary
    ybdy : y pts of boundary
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
    
    #ν_x = np.zeros(M)
    #ν_y = np.zeros(M)
    
    for m in range(0, M):
        x_diff = xbdy2[m] - xbdy1[m]
        y_diff = ybdy2[m] - ybdy1[m]
        dist = sqrt(x_diff**2 + y_diff**2)
    
        dn = dist/ngrid
        rgrid = np.linspace(0+dn, dist, num=ngrid,endpoint=False, retstep=False, dtype=None, axis=0)
        
        #ν_x[m] = x_diff / dist
        #ν_y[m] = y_diff / dist
        
        x[m, :] = xbdy1[m] + rgrid*(x_diff / dist)
        y[m, :] = ybdy1[m] + rgrid*(y_diff / dist)
    
    #Impose Periodicity 
    x[M][:] = x[0][:]
    y[M][:] = y[0][:]
    
    #ν = [ν_x, ν_y]
    
    return x, y #ν

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
    
    # Includes endpoint. Does this introduce problem?  Overlapping points?
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

def fullmake_grid(Mvals, nvals, Boundaries, Ds):
    '''
    Finds grid points on each boundary and in the layers between them according to number of points chosen in Mvals and nvals
    
    Parameters
    ==========
    Mvals : pts per boundary
    nvals: pts between each layer
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
    
    
    for m in range(0, len(Mvals)):
        dt = 2.0 * np.pi/Mvals[m]
        θ = np.arange(0, 2*np.pi, dt)
        
        xbdy, ybdy = xy_bdy(Boundaries[m].y_l(θ))
        xbdys.append(xbdy)
        ybdys.append(ybdy)
        if m == 0:
            x,y = make_grid(nvals[m], Boundaries[m].y_l(θ), Boundaries[m].ν_l(θ), Mvals[m], Ds[0], 1)
            xs.append(x)
            ys.append(y)
            
        if m < (len(Mvals) - 1):
            x,y = make_grid_between(nvals[m + 1], Boundaries[m].y_l(θ), Boundaries[m+1].y_l(θ), Mvals[m])
            xs.append(x)
            ys.append(y)
            
        if m == (len(Mvals) - 1):
            x,y = make_grid(nvals[m + 1], Boundaries[m].y_l(θ), Boundaries[m].ν_l(θ), Mvals[m], Ds[1], -1)
            xs.append(x)
            ys.append(y)
    
    return xbdys, ybdys, xs, ys