import numpy as np
from numpy import sqrt, log
from scipy.special import hankel1
from scipy.special import jv as besselj

g = lambda k, dotpr_0: np.cos(k * dotpr_0)
dvg = lambda k, dotpr_p, dotpr_0: -k * dotpr_p * np.sin(k * dotpr_0)

f = lambda k, dotpr_0: 1/k * np.sin(k * dotpr_0)
dvf = lambda k, dotpr_p, dotpr_0: dotpr_p * np.cos(k * dotpr_0)

# Generalize entire BRIEF method and then break up into nicer functions ...



def brief(M, ngrid, ncut, Bdys, xbdys, ybdys, xs, ys, ks, vs, us, dvus):
    θ, dt = np.linspace(0, 2*np.pi, num=M, endpoint=False, retstep=True)
    for l in range(0, np.size(ks)): # Iterate over the layers
        x, y, k = xs[l], ys[l], ks[l]
        for m in range(0, M):
            for n in range(0, ncut):
                if l == 0:
                    # Is outermost layer
                    xbdy, ybdy = xbdys[l], ybdys[l]
                    ν, J = Bdys[l].ν_l(θ), Bdys[l].J_l(θ)
                    xdiff, ydiff = x[m, ngrid - n - 1] - xbdy[m], y[m, ngrid - n - 1] - ybdy[m]
                    dotpr_0 = (ν[0][m] * xdiff) + (ν[1][m] * ydiff)
                    dotpr_p = 1 # This is always the case for our chosen formula
                    ψp = us[l][m] * g(k, dotpr_0) + dvus[l][m] * f(k, dotpr_0)
                    
                    xdiff, ydiff = x[m, ngrid - n - 1] - xbdy, y[m, ngrid - n - 1] - ybdy
                    distance = sqrt(xdiff**2 + ydiff**2)
                    cosθ = (ν[0][:] * xdiff + ν[1][:] * ydiff) / distance
                    
                    kernelD_i = 0.25 * 1j * k * cosθ * hankel1(1, k * distance)
                    kernelS_i = 0.25 * 1j * hankel1(0, k * distance)
                    
                    xbdiff, ybdiff = xbdy - xbdy[m], ybdy - ybdy[m]
                    dotpr_0 = (ν[0][m] * xbdiff) + (ν[1][m] * ybdiff)
                    dotpr_p = (ν[0][:] * ν[0][m]) + (ν[1][:] * ν[1][m])
                    ψ = us[l] * g(k, dotpr_0) + dvus[l] * f(k, dotpr_0)
                    dψ = us[l] * dvg(k, dotpr_p, dotpr_0) + dvus[l] * dvf(k, dotpr_p, dotpr_0)
                    
                    kernelD_i0 = np.nan_to_num(0.25 * 1j * k * cosθ * hankel1(1, k * distance))
                    kernelS_i0 = np.nan_to_num(0.25 * 1j * hankel1(0, k * distance))

                    D_i = sum((kernelD_i - kernelD_i0) * J * (us[l] - ψ)) * dt
                    S_i = sum((kernelS_i - kernelS_i0) * J * (dvus[l] - dψ)) * dt
                                
                    vs[l][m, ngrid - n - 1] = ψp + D_i - S_i

                elif l == (np.size(Bdys)):
                    # Is innermost layer
                    xbdy, ybdy = xbdys[l - 1], ybdys[l - 1]
                    ν, J = Bdys[l - 1].ν_l(θ), Bdys[l - 1].J_l(θ)
                    
                    xdiff, ydiff = x[m, n] - xbdy[m], y[m, n] - ybdy[m]
                    dotpr_0 = (ν[0][m] * xdiff) + (ν[1][m] * ydiff)
                    dotpr_p = 1 # This is always the case for our chosen formula
                    ψp = us[l - 1][m] * g(k, dotpr_0) + (k/ks[l-1])**2 * dvus[l - 1][m] * f(k, dotpr_0)
                    
                    xdiff, ydiff = x[m, n] - xbdy, y[m, n] - ybdy
                    distance = sqrt(xdiff**2 + ydiff**2)
                    cosθ = (ν[0][:] * xdiff + ν[1][:] * ydiff) / distance
                    
                    kernelD_e = 0.25 * 1j * k * cosθ * hankel1(1, k * distance)
                    kernelS_e = 0.25 * 1j * hankel1(0, k * distance)
                    
                    xbdiff, ybdiff = xbdy - xbdy[m], ybdy - ybdy[m]
                    dotpr_0 = (ν[0][m] *  xbdiff) + (ν[1][m] * ybdiff)
                    dotpr_p = (ν[0][:] * ν[0][m]) + (ν[1][:] * ν[1][m])
                    ψ = us[l - 1] * g(k, dotpr_0) + (k/ks[l-1])**2 * dvus[l - 1] * f(k, dotpr_0)
                    dψ = us[l - 1] * dvg(k, dotpr_p, dotpr_0) \
                    + (k/ks[l-1])**2 * dvus[l - 1] * dvf(k, dotpr_p, dotpr_0)
                    
                    kernelD_e0 = np.nan_to_num(0.25 * 1j * k * cosθ * hankel1(1, k * distance))
                    kernelS_e0 = np.nan_to_num(0.25 * 1j * hankel1(0, k * distance))
                    
                    D_e = sum((kernelD_e - kernelD_e0) * J * (us[l - 1] - ψ)) * dt
                    S_e = sum((kernelS_e - kernelS_e0) * J * ((k/ks[l-1])**2 * dvus[l - 1] - dψ)) * dt
                                
                    vs[l][m, n] = ψp + S_e - D_e
                    
                else:
                    # Is in-between layer
                        # Outer boundary
                    xbdy_e, ybdy_e, xbdy_i, ybdy_i = xbdys[l - 1], ybdys[l - 1], xbdys[l], ybdys[l]
                    ν_e, J_e, ν_i, J_i = Bdys[l - 1].ν_l(θ), Bdys[l - 1].J_l(θ), \
                    Bdys[l].ν_l(θ), Bdys[l].J_l(θ)
                    
                    xdiff, ydiff = x[m, n] - xbdy_e[m], y[m, n] - ybdy_e[m]
                    dotpr_0 = (ν_e[0][m] * xdiff) + (ν_e[1][m] * ydiff)
                    dotpr_p = 1 # This is always the case for our chosen formula
                    ψp = us[l - 1][m] * g(k, dotpr_0) + (k/ks[l-1])**2 * dvus[l - 1][m] * f(k, dotpr_0)
                    
                    xdiff, ydiff = x[m, n] - xbdy_i, y[m, n] - ybdy_i
                    distance = sqrt(xdiff**2 + ydiff**2)
                    cosθ = (ν_i[0][:] * xdiff + ν_i[1][:] * ydiff) / distance
                    
                    kernelD_i = 0.25 * 1j * k * cosθ * hankel1(1, k * distance)
                    kernelS_i = 0.25 * 1j * hankel1(0, k * distance)
                    
                    xbdiff, ybdiff = xbdy_i - xbdy_e[m], ybdy_i - ybdy_e[m]
                    dotpr_0 = (ν_e[0][m] *  xbdiff) + (ν_e[1][m] * ybdiff)
                    dotpr_p = (ν_e[0][:] * ν_e[0][m]) + (ν_e[1][:] * ν_e[1][m])
                    ψ = us[l] * g(k, dotpr_0) + dvus[l] * f(k, dotpr_0)
                    dψ = us[l] * dvg(k, dotpr_p, dotpr_0) + dvus[l] * dvf(k, dotpr_p, dotpr_0)
                    
                    kernelD_i0 = np.nan_to_num(0.25 * 1j * k * cosθ * hankel1(1, k * distance))
                    kernelS_i0 = np.nan_to_num(0.25 * 1j * hankel1(0, k * distance))

                    D_i = sum((kernelD_i - kernelD_i0) * J * (us[l] - ψ)) * dt
                    S_i = sum((kernelS_i - kernelS_i0) * J * (dvus[l] - dψ)) * dt

                    xdiff, ydiff = x[m, n] - xbdy_e, y[m, n] - ybdy_e
                    distance = sqrt(xdiff**2 + ydiff**2)
                    cosθ = (ν_e[0][:] * xdiff + ν_e[1][:] * ydiff) / distance
                    
                    kernelD_e = 0.25 * 1j * k * cosθ * hankel1(1, k * distance)
                    kernelS_e = 0.25 * 1j * hankel1(0, k * distance)
                    
                    xbdiff, ybdiff = xbdy_e - xbdy_e[m], ybdy_e - ybdy_e[m]
                    dotpr_0 = (ν_e[0][m] *  xbdiff) + (ν_e[1][m] * ybdiff)
                    dotpr_p = (ν_e[0][:] * ν_e[0][m]) + (ν_e[1][:] * ν_e[1][m])
                    ψ = us[l - 1] * g(k, dotpr_0) + (k/ks[l-1])**2 * dvus[l - 1] * f(k, dotpr_0)
                    dψ = us[l - 1] * dvg(k, dotpr_p, dotpr_0) \
                    + (k/ks[l-1])**2 * dvus[l - 1] * dvf(k, dotpr_p, dotpr_0)
                    
                    kernelD_e0 = np.nan_to_num(0.25 * 1j * k * cosθ * hankel1(1, k * distance))
                    kernelS_e0 = np.nan_to_num(0.25 * 1j * hankel1(0, k * distance))
                    
                    D_e = sum((kernelD_e - kernelD_e0) * J * (us[l - 1] - ψ)) * dt
                    S_e = sum((kernelS_e - kernelS_e0) * J * ((k/ks[l-1])**2 * dvus[l - 1] - dψ)) * dt
                    
                    vs[l][m, n] = ψp + S_e - D_e - S_i + D_i
                    
                        # Inner boundary
                    
                    xdiff, ydiff = x[m, ngrid - n - 1] - xbdy_i[m], y[m, ngrid - n - 1] - ybdy_i[m]
                    dotpr_0 = (ν_i[0][m] * xdiff) + (ν_i[1][m] * ydiff)
                    dotpr_p = 1 # This is always the case for our chosen formula
                    ψp = us[l][m] * g(k, dotpr_0) + dvus[l][m] * f(k, dotpr_0)
                    
                    xdiff, ydiff = x[m, n] - xbdy_i, y[m, n] - ybdy_i
                    distance = sqrt(xdiff**2 + ydiff**2)
                    cosθ = (ν_i[0][:] * xdiff + ν_i[1][:] * ydiff) / distance
                    
                    kernelD_i = 0.25 * 1j * k * cosθ * hankel1(1, k * distance)
                    kernelS_i = 0.25 * 1j * hankel1(0, k * distance)
                    
                    xbdiff, ybdiff = xbdy_i - xbdy_i[m], ybdy_i - ybdy_i[m]
                    dotpr_0 = (ν_i[0][m] *  xbdiff) + (ν_i[1][m] * ybdiff)
                    dotpr_p = (ν_i[0][:] * ν_i[0][m]) + (ν_i[1][:] * ν_i[1][m])
                    ψ = us[l] * g(k, dotpr_0) + dvus[l] * f(k, dotpr_0)
                    dψ = us[l] * dvg(k, dotpr_p, dotpr_0) + dvus[l] * dvf(k, dotpr_p, dotpr_0)
                    
                    kernelD_i0 = np.nan_to_num(0.25 * 1j * k * cosθ * hankel1(1, k * distance))
                    kernelS_i0 = np.nan_to_num(0.25 * 1j * hankel1(0, k * distance))

                    D_i = sum((kernelD_i - kernelD_i0) * J * (us[l] - ψ)) * dt
                    S_i = sum((kernelS_i - kernelS_i0) * J * (dvus[l] - dψ)) * dt

                    xdiff, ydiff = x[m, ngrid - n - 1] - xbdy_i, y[m, ngrid - n - 1] - ybdy_i
                    distance = sqrt(xdiff**2 + ydiff**2)
                    cosθ = (ν_e[0][:] * xdiff + ν_e[1][:] * ydiff) / distance
                    
                    kernelD_e = 0.25 * 1j * k * cosθ * hankel1(1, k * distance)
                    kernelS_e = 0.25 * 1j * hankel1(0, k * distance)
                    
                    xbdiff, ybdiff = xbdy_e - xbdy_i[m], ybdy_e - ybdy_i[m]
                    dotpr_0 = (ν_e[0][m] *  xbdiff) + (ν_e[1][m] * ybdiff)
                    dotpr_p = (ν_e[0][:] * ν_i[0][m]) + (ν_e[1][:] * ν_i[1][m])
                    ψ = us[l] * g(k, dotpr_0) + dvus[l] * f(k, dotpr_0)
                    dψ = us[l] * dvg(k, dotpr_p, dotpr_0) + dvus[l] * dvf(k, dotpr_p, dotpr_0)
                    
                    kernelD_e0 = np.nan_to_num(0.25 * 1j * k * cosθ * hankel1(1, k * distance))
                    kernelS_e0 = np.nan_to_num(0.25 * 1j * hankel1(0, k * distance))
                    
                    D_e = sum((kernelD_e - kernelD_e0) * J * (us[l] - ψ)) * dt
                    S_e = sum((kernelS_e - kernelS_e0) * J * (dvus[l] - dψ)) * dt
                    
                    vs[l][m, ngrid - n - 1] = ψp + S_e - D_e - S_i + D_i
                    
        vs[l][M, :] = vs[l][0, :]
    return vs