import numpy as np

'''
essentials.py 
===============

- uin = find_exact_planewave(x, y, k, alpha)

- xbdys2, ybdys2, us2 = add_periodicity(xbdys, ybdys, us, num)

- full = fullify(layers, bdys, num)

'''

def find_exact_planewave(x, y, k, alpha):
    '''
    Computes the incident solution (planewave) for the given parameters
    
    Parameters
    ==========
    x : x pts
    y : y pts
    k : wave number
    alpha : incident angle

    
    Returns
    =======
    uin : the incident solution
    
    '''
    uin = np.exp(1j * k * (np.cos(alpha) * x + np.sin(alpha) * y))
    return uin

def add_periodicity(xbdys, ybdys, us, num):
    '''
    Imposes missing periodicity condition to boundary arrays
    
    Parameters
    ==========
    xbdy : x bdy pts
    ybdy : y bdy pts
    us : bdy data
    num : number of bdys

    
    Returns
    =======
    xbdys2 : x bdy pts + periodicity
    ybdys2 : y bdy pts + periodicity
    us2 : u bdy pts + periodicity
    
    '''
    xbdys2 = []; ybdys2 = []; us2 = []
    
    for b in range(0, num):
        xbdys2.append(np.append(xbdys[b], xbdys[b][0]))
        ybdys2.append(np.append(ybdys[b], ybdys[b][0]))
        us2.append(np.append(us[b], us[b][0]))
        
    return xbdys2, ybdys2, us2


def fullify(layers, bdys, num):
    '''
    Combines stacks of layer and boundary data into one matrix
    
    Parameters
    ==========
    layers : stack of layer data
    bdys : stack of bdy data
    num : number of bdys
    
    Returns
    =======
    full : combined data matrix
    
    '''
    full = []
    for i in range(0, num):
        temp = np.append(np.transpose(layers[i]), bdys[i])
        temp = np.reshape(temp, (np.size(layers[i], 1) + 1, np.size(layers[i], 0)))
        full.append(temp)
        if i == (num - 1):
            temp = np.append(bdys[i], np.transpose(layers[i+1]))
            temp = np.reshape(temp, (np.size(layers[i + 1], 1) + 1, np.size(layers[i + 1], 0)))
            full.append(temp)            
    full = np.reshape(full, (np.size(full, 0) * np.size(full, 1), np.size(full, 2)))
    return full