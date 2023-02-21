import numpy as np
from numpy import sqrt, log
from scipy.special import hankel1
from scipy.special import jv as besselj
from scipy.fft import irfft
from scipy.linalg import toeplitz, solve
from scipy.sparse import identity

def find_diag_block(M, Bdy, xbdy, ybdy, k_e, k_i): # All checked
    θ, dt = np.linspace(0, 2*np.pi, num=M, endpoint=False, retstep=True) #CC: changed bounds
    T, S = np.meshgrid(θ, θ)
    
    xbdy_s, ybdy_s = Bdy.y_l(S)
    xbdy_t, ybdy_t = Bdy.y_l(T)
    
    xdiff, ydiff = xbdy_s - xbdy_t, ybdy_s - ybdy_t
    dist = np.hypot(xdiff, ydiff)
    
    jac = Bdy.J_l(T)
    ν_x, ν_y = Bdy.ν_l(T)
    
    κ = Bdy.κ_l(θ)
    E_C =  np.euler_gamma
    
    is_zero = np.diag_indices(M)
    not_zero = np.where(np.logical_not(np.eye(M, dtype=bool)))
    
    cos_term = np.empty((M, M))
    cos_term[not_zero] = (
        ν_x[not_zero] * xdiff[not_zero] + ν_y[not_zero] * ydiff[not_zero]
    ) / dist[not_zero]
    
    coeffs = np.zeros(M // 2 + 1)
    coeffs[1:] = -2 * np.pi / np.arange(1, M // 2 + 1)
    R = irfft(coeffs, n=M)
    R = toeplitz(R)  
    
    L1_i = np.empty((M, M))
    L1_i[not_zero] = (
        (-k_i / (4 * np.pi))
        * cos_term[not_zero]
        * besselj(1, k_i * dist[not_zero])
        * jac[not_zero]
    )
    L1_i[is_zero] = 0

    L2_i = np.empty((M, M), dtype=complex)
    L2_i[not_zero] = (
        (0.25 * 1j * k_i)
        * hankel1(1, k_i * dist[not_zero])
        * cos_term[not_zero]
        * jac[not_zero]
    )
    L2_i[not_zero] -= L1_i[not_zero] * np.log(
        4 * np.sin(0.5 * (S[not_zero] - T[not_zero])) ** 2
    )
    L2_i[is_zero] = (-1 / (4 * np.pi)) * jac[is_zero] * Bdy.κ_l(θ)
    L2_i *= dt

    M1_i = (-0.25 / np.pi) * besselj(0, k_i * dist)
 
    M2_i = np.empty((M, M), dtype=complex)
    M2_i[not_zero] = (0.25 * 1j) * hankel1(0, k_i * dist[not_zero])
    M2_i[not_zero] -= M1_i[not_zero] * np.log(
        4 * np.sin(0.5 * (S[not_zero] - T[not_zero]))** 2
    )
    M2_i[is_zero] = (0.25 * 1j - 0.5 * E_C / np.pi ) - (0.5 / np.pi) * np.log(
            (0.5 * k_i) * Bdy.J_l(θ)
    )

    M2_i *= dt    
    
    L1_e = np.empty((M, M))
    L1_e[not_zero] = (
        (-k_e / (4 * np.pi))
        * cos_term[not_zero]
        * besselj(1, k_e * dist[not_zero])
        * jac[not_zero]
    )
    L1_e[is_zero] = 0

    L2_e = np.empty((M, M), dtype=complex)
    L2_e[not_zero] = (
        (0.25 * 1j * k_e)
        * hankel1(1, k_e * dist[not_zero])
        * cos_term[not_zero]
        * jac[not_zero]
    )

    L2_e[not_zero] -= L1_e[not_zero] * np.log(
        4 * np.sin(0.5 * (S[not_zero] - T[not_zero])) ** 2
    )

    L2_e[is_zero] = (-1 / (4 * np.pi)) * jac[is_zero] * Bdy.κ_l(θ)

    L2_e *= dt

    M1_e = (-0.25 / np.pi) * besselj(0, k_e * dist)
    
    M2_e = np.empty((M, M), dtype=complex)
    M2_e[not_zero] = (0.25 * 1j) * hankel1(0, k_e * dist[not_zero])
    M2_e[not_zero] -= M1_e[not_zero] * np.log(
        4 * np.sin(0.5 * (S[not_zero] - T[not_zero]))** 2
    )
    M2_e[is_zero] = (0.25 * 1j - 0.5 * E_C / np.pi ) - (0.5 / np.pi) * np.log(
            (0.5 * k_e) * Bdy.J_l(θ)
    ) 
    M2_e *= dt
    
    
    L_i = R * L1_i + L2_i
    M_i = R * M1_i + M2_i
    L_e = R * L1_e + L2_e
    M_e = R * M1_e + M2_e
    
    return L_e, jac * M_e, L_i, jac * M_i
    #return L_e, M_e, L_i, M_i
    
def build_A(M, Bdys, xbdys, ybdys, ks):  #bdys is the number of boundaries
    bdys = np.size(Bdys)
    A = np.zeros((2*bdys*M, 2*bdys*M), dtype=complex) # 4 for 3 layers, 6 for 4 layers
    # We need to build it one section at a time
    for b in range(0, bdys):
        D_e, S_e, D_i, S_i = find_diag_block(M, Bdys[b], xbdys[b], ybdys[b], ks[b], ks[b+1])
        
        A[2*b*M:(2*b+1)*M, 2*b*M:(2*b+1)*M] = 0.5 * np.identity(M) - D_e
        A[2*b*M:(2*b+1)*M, (2*b+1)*M:(2*b+2)*M] = S_e
        A[(2*b+1)*M:(2*b+2)*M, 2*b*M:(2*b+1)*M] = 0.5 * np.identity(M) + D_i
        A[(2*b+1)*M:(2*b+2)*M, (2*b+1)*M:(2*b+2)*M] = -(ks[b+1]**2/ks[b]**2) * S_i
        
        # We build off-diagonals, but have exceptions in first and last blocks of the matrix
        if (b != 0):
            D_0, S_0 = find_offdiag(M, Bdys[b-1], Bdys[b], ks[b])
            A[2*b*M:(2*b+1)*M, (2*b-2)*M:(2*b-1)*M] = D_0
            A[2*b*M:(2*b+1)*M, (2*b-1)*M:2*b*M] = -(ks[b]**2/ks[b-1]**2) * S_0
            
        if (b != bdys - 1):
            D_1, S_1 = find_offdiag(M, Bdys[b+1], Bdys[b], ks[b+1])
            A[(2*b+1)*M:(2*b+2)*M, (2*b+2)*M:(2*b+3)*M] = -D_1
            A[(2*b+1)*M:(2*b+2)*M, (2*b+3)*M:(2*b+4)*M] = S_1
            
    return A


def solve_BIE_sys(M, Bdys, xbdys, ybdys, ks, uin):
    bdys = np.size(Bdys)
    A = build_A(M, Bdys, xbdys, ybdys, ks)
    F = np.zeros(2*bdys*M, dtype=complex)
    #F = np.block([uin, np.zeros(M)])
    
    F[0:M] = uin
    U = solve(A, F)
    us = []
    dvus = []
    for b in range(0, bdys):
        us.append(U[2*b*M:(2*b+1)*M])
        dvus.append(U[(2*b+1)*M:(2*b+2)*M])
    
    return us, dvus    
    


def find_offdiag(M, Bdy1, Bdy2, k):
    θ, dt = np.linspace(0, 2*np.pi, num=M, endpoint=False, retstep=True) #CC: changed bounds
    T, S = np.meshgrid(θ, θ)
                       
    xbdy_s, ybdy_s = Bdy2.y_l(S)
    xbdy_t, ybdy_t = Bdy1.y_l(T)
    
    xdiff, ydiff = xbdy_s - xbdy_t, ybdy_s - ybdy_t
    dist = np.hypot(xdiff, ydiff)
    
    ν_x, ν_y = Bdy1.ν_l(T)
    jac = Bdy1.J_l(T)
    cos_term = ( ν_x * xdiff + ν_y * ydiff ) / dist
    
    D = (np.pi / M) * 0.5 * 1j * k * cos_term * jac * hankel1(1, k * dist) #CC is this correct ?
    S = (np.pi / M) * 0.5 * 1j * hankel1(0, k * dist) * jac 
    #return D, jac * S
    return D, S



def stack_xys(M, ngrid, xs, ys, Bdys):
    bdys = np.size(Bdys)
    stack_x = np.stack((xs[b], xs[b+1]) for b in range(0, bdys))
    stack_y = np.stack((ys[b], ys[b+1]) for b in range(0, bdys))
    stack_x = stack_x.reshape(2*bdys, M + 1, ngrid)
    stack_y = stack_y.reshape(2*bdys, M + 1, ngrid)
    return stack_x, stack_y

def stack_xybdys(M, xbdys, ybdys, Bdys):
    bdys = np.size(Bdys)
    stack_xbdy = np.stack((xbdys[b], xbdys[b]) for b in range(0, bdys))
    stack_ybdy = np.stack((ybdys[b], ybdys[b]) for b in range(0, bdys))
    stack_xbdy = stack_xbdy.reshape(2*bdys, M)
    stack_ybdy = stack_ybdy.reshape(2*bdys, M)
    return stack_xbdy, stack_ybdy

def stack_νdiff(M, ngrid, Bdys, superx, supery):
    θ, dt = np.linspace(0, 2*np.pi, num=M, endpoint=False, retstep=True)
    bdys = np.size(Bdys)
    stack_x = np.stack((Bdys[b].ν_l(θ)[0][:]*superx[2*b], Bdys[b].ν_l(θ)[0][:]*superx[2*b + 1]) for b in range(0, bdys))
    stack_y = np.stack((Bdys[b].ν_l(θ)[1][:]*supery[2*b], Bdys[b].ν_l(θ)[1][:]*supery[2*b + 1]) for b in range(0, bdys))
    stack_x = stack_x.reshape(2*bdys, M + 1, ngrid, M)
    stack_y = stack_y.reshape(2*bdys, M + 1, ngrid, M)        
    return stack_x, stack_y

def stack_products(M, ngrid, Bdys, factors, stack):
    bdys = np.size(Bdys)
    newstack = np.stack((factors[b] * stack[2*b], factors[b + 1]*stack[2*b + 1]) for b in range(0,bdys))
    newstack = newstack.reshape(2*bdys, M + 1, ngrid, M)
    return newstack

def append_products(Bdys, factors, arrays):
    bdys = np.size(Bdys)
    newarr = []
    for b in range(0, bdys):
        newarr.append(factors[b] * arrays[2*b])
        newarr.append(factors[b + 1] * arrays[2*b + 1])
    return newarr

def find_layers(M, ngrid, Bdys, xbdys, ybdys, xs, ys, ks, uin, us, dvus):
    D_e = np.zeros((M + 1, ngrid), dtype=complex)
    S_e = np.zeros((M + 1, ngrid), dtype=complex)
    D_i = np.zeros((M + 1, ngrid), dtype=complex)
    S_i = np.zeros((M + 1, ngrid), dtype=complex)
    
    superx, supery = stack_xys(M, ngrid, xs, ys, Bdys)
    superxbdy, superybdy = stack_xybdys(M, xbdys, ybdys, Bdys)
    
    superxdiff = superx.reshape(*superx.shape, 1) - np.expand_dims(superxbdy, (1, 2))
    superydiff = supery.reshape(*supery.shape, 1) - np.expand_dims(superybdy, (1, 2))
    superdistance = sqrt(superxdiff**2 + superydiff**2)
    
    νx_diff, νy_diff = stack_νdiff(M, ngrid, Bdys, superxdiff, superydiff)
    supercosθ = (νx_diff + νy_diff) / superdistance
    
    kdistance = append_products(Bdys, ks, superdistance)
    kcosθ = stack_products(M, ngrid, Bdys, ks, supercosθ)
    superhankel1_1 = hankel1(1, kdistance)
    superhankel1_0 = hankel1(0, kdistance)
    
    kernelsD_mn = 0.25 * 1j * kcosθ * superhankel1_1
    kernelsS_mn = 0.25 * 1j * superhankel1_0
    
    # All kernels needed for creation of layer potentials ... still needs to be done for each index at the moment
    θ, dt = np.linspace(0, 2*np.pi, num=M, endpoint=True, retstep=True)
    bdys = np.size(Bdys)
    vs = []
    for b in range(0, bdys + 1):
        for m in range(0, M + 1):
            for n in range(0, ngrid):
                if (b != 0):
                    kernelD_i = kernelsD_mn[2*b - 1, m, n]
                    kernelS_i = kernelsS_mn[2*b - 1, m, n]
                    D_i[m, n] = np.sum(kernelD_i * Bdys[b - 1].J_l(θ) * us[b - 1]) * dt
                    S_i[m, n] = np.sum(kernelS_i * Bdys[b - 1].J_l(θ) * dvus[b - 1]) * dt
                if (b != bdys):
                    kernelD_e = kernelsD_mn[2*b, m, n]
                    kernelS_e = kernelsS_mn[2*b, m, n]
                    D_e[m, n] = np.sum(kernelD_e * Bdys[b].J_l(θ) * us[b]) * dt
                    S_e[m, n] = np.sum(kernelS_e * Bdys[b].J_l(θ) * dvus[b]) * dt
        if (b == 0):
            vs.append(uin + D_e - S_e)
        elif (b == bdys):
            vs.append( (ks[b]**2/ks[b-1]**2) * S_i - D_i)
        else:
            vs.append( (ks[b]**2/ks[b-1]**2) * S_i - D_i + D_e - S_e)
            
            
    return vs #kernelsD_mn, kernelsS_mn