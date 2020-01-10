## Definition of functions outside of Jupyter notebook.
import numpy as np
from sympy import sin, cos, exp
from numpy import sqrt, log
from scipy.linalg import toeplitz
from scipy.special import hankel1
from scipy.special import jv as besselj

planewave_f = lambda xbdy, ybdy, k, alpha: np.exp(1j * k * (np.cos(alpha) * xbdy + np.sin(alpha) * ybdy))

def find_boundary_data(f, dt, xbdy, ybdy, ν, J, κ, M, k_i, k_e):
    '''
    Computes u and du/dv on the boundary by solving the BIE system
    
    Attributes
    ======
    kernel of the double layer, L
    kernel of the single layer, M
    solution on the boundary, u
    normal derivative on the boundary, dvu
    '''
    
    E_C = 0.5772156649015328606065120900824;  # Euler's constant
    N = M / 2;
    m = np.arange(1, N);
    k_ratio = k_e/k_i
    
    # Create array for ifft...
    a = [0]
    a.extend(1/m)
    a.append(1/N)
    a.extend((1/m)[::-1])

    Rj = -2 * np.pi * np.fft.ifft(a);
    R = np.real(toeplitz(Rj, Rj)); #check coefficients Matlab Toeplitz, order correct ?

    # Prepare arrays for solving integral equation of the boundary (Kress pt. 2)
    L_e = np.zeros((M, M), dtype=complex);
    M_e = np.zeros((M, M), dtype=complex);
    L_i = np.zeros((M, M), dtype=complex);
    M_i = np.zeros((M, M), dtype=complex);
    
    L1_e = np.zeros((M, M), dtype=complex);
    L2_e = np.zeros((M, M), dtype=complex);
    M1_e = np.zeros((M, M), dtype=complex);
    M2_e = np.zeros((M, M), dtype=complex);
    
    L1_i = np.zeros((M, M), dtype=complex);
    L2_i = np.zeros((M, M), dtype=complex);
    M1_i = np.zeros((M, M), dtype=complex);
    M2_i = np.zeros((M, M), dtype=complex);

    A = np.zeros((2*M, 2*M), dtype=complex);
    F = np.zeros(2*M, dtype=complex);
    for m in range(0, M):
        for n in range(0, M):
            rdiff = np.asarray([xbdy[m] - xbdy[n], ybdy[m] - ybdy[n]]);
            distance = sqrt(np.power(rdiff[:][0], 2) + np.power(rdiff[:][1], 2));
            M1_e[m][n] = -0.5 / np.pi * J[n] * besselj(0, k_e * distance);
            M1_i[m][n] = -0.5 / np.pi * J[n] * besselj(0, k_i * distance);
            if m == n:
                #Implementration of the Kress Quadrature
                L1_e[m][n] = 0;
                L2_e[m][n] = 0.5 / np.pi * κ[n] * J[n];
                M2_e[m][n] = J[n] * ( 0.5 * 1j - E_C / np.pi - 0.5 / np.pi * np.log( 0.25 * np.power(k_e, 2) * np.power(J[n], 2)));
                
                L1_i[m][n] = 0;
                L2_i[m][n] = 0.5 / np.pi * κ[n] * J[n];
                M2_i[m][n] = J[n] * ( 0.5 * 1j - E_C / np.pi - 0.5 / np.pi * np.log( 0.25 * np.power(k_i, 2) * np.power(J[n], 2)));
            
            else:
                #Compute cosθ and logterm for kernel calculation
                a = np.asmatrix([ν[0][n], ν[1][n]]);
                b = np.asmatrix(rdiff/distance);
                cosθ = np.asscalar(np.matmul(a, b.transpose()));
                
                logterm = log(float(4 * np.power(sin(0.5 * (m - n) * dt), 2))); #why float ?
                
                #Compute kernels for the BIE system
                L1_e[m][n] = 0.5 * k_e / np.pi * J[n] * cosθ * besselj(1, k_e * distance);
                L2_e[m][n] = 0.5 * 1j * k_e * J[n] * cosθ * hankel1(1, k_e * distance) - L1_e[m][n] * logterm;
                M2_e[m][n] = 0.5 * 1j * J[n] * hankel1(0, k_e * distance) - M1_e[m][n] * logterm;
                
                L1_i[m][n] = 0.5 * k_i / np.pi * J[n] * cosθ * besselj(1, k_i * distance);
                L2_i[m][n] = 0.5 * 1j * k_i * J[n] * cosθ * hankel1(1, k_i * distance) - L1_i[m][n] * logterm;
                M2_i[m][n] = 0.5 * 1j * J[n] * hankel1(0, k_i * distance) - M1_i[m][n] * logterm;
    
            L_e[m][n] = 0.5 * R[m][n] * (L1_e[m][n]) + 0.5 * (np.pi / N) * (L2_e[m][n]);
            M_e[m][n] = (0.5 * R[m][n] * (M1_e[m][n]) + 0.5 * (np.pi / N) * (M2_e[m][n]));
            
            L_i[m][n] = 0.5 * R[m][n] * (L1_i[m][n]) + 0.5 * (np.pi / N) * (L2_i[m][n]);
            M_i[m][n] = 0.5 * R[m][n] * (M1_i[m][n]) + 0.5 * (np.pi / N) * (M2_i[m][n]);

    #Matrix of combined representation formulas       
    A11 = 0.5 * np.identity(M) - L_e;
    A12 = k_ratio * M_e;
    A21 = 0.5 * np.identity(M) + L_i;
    A22 = -M_i;

    #Combine A's into one matrix, A
    A[0:M, 0:M] = A11; A[0:M, M: 2*M] = A12;
    A[M: 2*M, 0:M] = A21; A[M: 2*M, M: 2*M] = A22;
    
    #Make matrix for f and 0's, call it F
    F[0:M] = f;
    
    #Solve for u and dvu from AU = F
    U = np.linalg.solve(A,F);
    u = U[0:M];
    dvu = U[M: 2*M];
    return u, dvu

def opt_find_boundary_data(f, dt, xbdy, ybdy, ν, J, κ, M, k_i, k_e):
    '''
    Computes u and du/dv on the boundary by solving the BIE system
    
    Attributes
    ======
    kernel of the double layer, L
    kernel of the single layer, M
    solution on the boundary, u
    normal derivative on the boundary, dvu
    '''
    k_ratio = k_e/k_i
    E_C =  np.euler_gamma # Euler's constant
    N = M / 2
    m = np.arange(1, N)

    # Create array for ifft...
    a = [0]
    a.extend(1/m)
    a.append(1/N)
    a.extend((1/m)[::-1])

    Rj = -2 * np.pi * np.fft.ifft(a)
    R = np.real(toeplitz(Rj, Rj))

    # Prepare arrays for solving integral equation of the boundary (Kress pt. 2)
    A = np.zeros((2*M, 2*M), dtype=complex)
    F = np.zeros(2*M, dtype=complex)

    cosθ_mn = np.zeros((M,M))
    #distance_mn = np.zeros((M, M))
    logterm_mn = np.zeros((M, M))
    
    besselj_0e_mn = np.zeros((M, M), dtype=complex)
    besselj_1e_mn = np.zeros((M, M), dtype=complex)
    hankel1_0e_mn = np.zeros((M, M), dtype=complex)
    hankel1_1e_mn = np.zeros((M, M), dtype=complex)
    
    besselj_0i_mn = np.zeros((M, M), dtype=complex)
    besselj_1i_mn = np.zeros((M, M), dtype=complex)
    hankel1_0i_mn = np.zeros((M, M), dtype=complex)
    hankel1_1i_mn = np.zeros((M, M), dtype=complex)
    
    logterm_o = {}
    for i in range(-M,M):
        if i != 0: 
            logterm_o[i] = log(float(4 * np.power(np.sin(0.5 * i * dt), 2)))
        
    
    for m in range(0, M):
        for n in range(0, M):
            rdiff = np.asarray([xbdy[m] - xbdy[n], ybdy[m] - ybdy[n]]);
            r_distance = sqrt(rdiff[:][0]**2 + rdiff[:][1]**2)
            #distance_mn[m,n] = distance
            besselj_0e_mn[m,n] = besselj(0, k_e * r_distance)
            besselj_0i_mn[m,n] = besselj(0, k_i * r_distance)
            if m != n:
                cosθ_mn[m, n] = ν[0][n]*(rdiff[0]/r_distance) + ν[1][n]*(rdiff[1]/r_distance)
                logterm_mn[m,n] = logterm_o[m-n]

                besselj_1e_mn[m,n] = besselj(1, k_e * r_distance)
                hankel1_1e_mn[m,n] = hankel1(1, k_e * r_distance)
                hankel1_0e_mn[m,n] = hankel1(0, k_e * r_distance)

                besselj_1i_mn[m,n] = besselj(1, k_i * r_distance)
                hankel1_1i_mn[m,n] = hankel1(1, k_i * r_distance)
                hankel1_0i_mn[m,n] = hankel1(0, k_i * r_distance)
            
    #Compute kernels of exterior of boundary
    L1_e = 0.5 * k_e / np.pi * J * cosθ_mn * besselj_1e_mn
    L2_e = 0.5 * 1j * k_e * J * cosθ_mn * hankel1_1e_mn - L1_e * logterm_mn
    M1_e = -0.5 / np.pi * J * besselj_0e_mn
    M2_e = 0.5 * 1j * J * hankel1_0e_mn - M1_e * logterm_mn
     
    # Usage of Kress Quadrature
    diag = np.arange(0,M)
    M2_e[diag, diag] = J * ( 0.5 * 1j - E_C / np.pi - 0.5 / np.pi * np.log( 0.25 * (k_e**2) * (J**2) ))
    L1_e[diag, diag] = 0
    L2_e[diag, diag] = 0.5 / np.pi * κ * J
            
    L_e = 0.5 * R * L1_e + 0.5 * (np.pi / N) * L2_e
    M_e = 0.5 * R * M1_e + 0.5 * (np.pi / N) * M2_e
                
    #Calculate kernels of interior of boundary
    L1_i = 0.5 * k_i / np.pi * J * cosθ_mn * besselj_1i_mn
    L2_i = 0.5 * 1j * k_i * J * cosθ_mn * hankel1_1i_mn - L1_i * logterm_mn
    M1_i = -0.5 / np.pi * J * besselj_0i_mn
    M2_i = 0.5 * 1j * J * hankel1_0i_mn - M1_i * logterm_mn
    
    #Replace diagonals where m=n with exception cases
    L1_i[diag, diag] = 0
    L2_i[diag, diag] = 0.5 / np.pi * κ * J;
    M2_i[diag, diag] = J * ( 0.5j - E_C / np.pi - 0.5 / np.pi * np.log( 0.25 * (k_i**2) * (J**2)))
    
    L_i =0.5 * R * L1_i + 0.5 * (np.pi / N) * L2_i
    M_i = 0.5 * R * M1_i + 0.5 *(np.pi / N) * M2_i

    #Matrix of combined representation formulas       
    A11 = 0.5 * np.identity(M) - L_e
    A12 = k_ratio * M_e
    A21 = 0.5 * np.identity(M) + L_i
    A22 = -M_i

    #Combine A's into one matrix, A
    A[0:M, 0:M] = A11; A[0:M, M: 2*M] = A12
    A[M: 2*M, 0:M] = A21; A[M: 2*M, M: 2*M] = A22
    
    #Make matrix for f and 0's, call it F
    F[0:M] = f
    
    #Solve for u and dvu from AU = F
    U = np.linalg.solve(A,F)
    u = U[0:M]
    dvu = U[M: 2*M]
    return u, dvu

def make_solution_grid(ngrid, f, u, dvu, xbdy, ybdy, ν, J, κ, M, k_i, k_e):
    '''
    Calculate the solution for interior and exterior domains using u and dvu
    
    Attributes
    ==========
    domain and range values for plot of the solution grid, x and y
    approximation of the solution using PTR, v
    exact solution for known case of ke = ki, exact
    
    '''
    # Find the absolute maximum curvature |κ|max
    κ_max = np.amax(np.abs(κ));

    k_ratio = k_e/k_i
    dn = (1/κ_max)/ngrid;
    rgrid = np.arange(0+dn, 1/ κ_max, dn);

    # Allocate memory for the solution
    x_e  = np.zeros((M + 1, ngrid - 1));
    y_e  = np.zeros((M + 1, ngrid - 1));
    v_e  = np.zeros((M + 1, ngrid - 1), dtype=complex); # Kernel and sum will include complex numbers
    
    x_i  = np.zeros((M + 1, ngrid - 1));
    y_i  = np.zeros((M + 1, ngrid - 1));
    v_i  = np.zeros((M + 1, ngrid - 1), dtype=complex); # Kernel and sum will include complex numbers
    
    for m in range(0, M):
        for n in range(0, ngrid-1):
            x_e[m][n] = xbdy[m] + rgrid[n]*ν[0][m];
            y_e[m][n] = ybdy[m] + rgrid[n]*ν[1][m];

            xe_diff = x_e[m][n] - xbdy;
            ye_diff = y_e[m][n] - ybdy;
        
            distance_e = sqrt(np.power(xe_diff, 2) + np.power(ye_diff, 2)); #(r in kress)
            cosθ_e = (ν[0][:]*xe_diff + ν[1][:]*ye_diff ) / distance_e;
            
            kernelD_e = 0.25 * 1j * (k_e * cosθ_e * hankel1(1, k_e*distance_e));
            kernelS_e = 0.25 * 1j * hankel1(0, k_e*distance_e); # kress (2.2+)
            
            fx = np.exp(1j * k_e * (np.cos(alpha) * x_e[m][n] + np.sin(alpha) * y_e[m][n]));
            v_e[m][n] = fx + sum(kernelD_e * J * u)* dt - k_ratio * sum(kernelS_e * J * dvu) * dt; #From Representation Formula (1)
             
            # Do the same as above but for the interior...
            x_i[m][n] = xbdy[m] - rgrid[n]*ν[0][m];
            y_i[m][n] = ybdy[m] - rgrid[n]*ν[1][m];
            
            xi_diff = x_i[m][n] - xbdy;
            yi_diff = y_i[m][n] - ybdy;
            
            distance_i = sqrt(np.power(xi_diff, 2) + np.power(yi_diff, 2)); #(r in kress)
            cosθ_i = (ν[0][:]*xi_diff + ν[1][:]*yi_diff) / distance_i;
            
            kernelD_i = 0.25 * 1j * (k_i * cosθ_i * hankel1(1, k_i*distance_i));
            kernelS_i = 0.25 * 1j * hankel1(0, k_i*distance_i);
            v_i[m][n] = -sum(kernelD_i * J * u) * dt + sum(kernelS_i * J * dvu) * dt; #From Representation Formula (2)
    
    #Impose Periodicity
    x_e[M][:] = x_e[0][:];
    y_e[M][:] = y_e[0][:];
    v_e[M][:] = v_e[0][:];
    
    x_i[M][:] = x_i[0][:];
    y_i[M][:] = y_i[0][:];
    v_i[M][:] = v_i[0][:];
    
    return x_i, x_e, y_i, y_e, v_i, v_e

def opt_make_solution_grid(ngrid, alpha, dt, f, u, dvu, xbdy, ybdy, ν, J, κ, M, k_i, k_e):
    '''
    Calculate the solution for interior and exterior domains using u and dvu
    
    Attributes
    ==========
    domain and range values for plot of the solution grid, x and y
    approximation of the solution using PTR, v
    exact solution for known case of ke = ki, exact
    
    '''
    k_ratio = k_e/k_i
    # Find the absolute maximum curvature |κ|max
    κ_max = np.amax(np.abs(κ))

    dn = (1/κ_max)/ngrid
    rgrid = np.arange(0+dn, 1/ κ_max, dn)

    # Allocate memory for the solution
    kernelD_e_mn = np.zeros((M + 1, ngrid - 1), dtype=complex)
    kernelS_e_mn = np.zeros((M + 1, ngrid - 1), dtype=complex)
    kernelD_i_mn = np.zeros((M + 1, ngrid - 1), dtype=complex)
    kernelS_i_mn = np.zeros((M + 1, ngrid - 1), dtype=complex) 
    
    x_e  = np.zeros((M + 1, ngrid - 1))
    y_e  = np.zeros((M + 1, ngrid - 1))

    x_i  = np.zeros((M + 1, ngrid - 1))
    y_i  = np.zeros((M + 1, ngrid - 1))

    #Can move grid calculation to its own function...
    for m in range(0, M):
        x_e[m] = xbdy[m] + rgrid*ν[0][m]
        y_e[m] = ybdy[m] + rgrid*ν[1][m]
        x_i[m] = xbdy[m] - rgrid*ν[0][m]
        y_i[m] = ybdy[m] - rgrid*ν[1][m]
    
    fx = planewave_f(x_e, y_e, k_e, alpha)
    #Ask about simplifying loops, because Kernels are each size MxN and each used to sum single index of v_i and v_e...
    for m in range(0, M):
        for n in range(0, ngrid-1):
            xe_diff = x_e[m][n] - xbdy
            ye_diff = y_e[m][n] - ybdy
        
            distance_e = sqrt(xe_diff**2 + ye_diff**2)
            cosθ_e = (ν[0][:]*xe_diff + ν[1][:]*ye_diff ) / distance_e;
    
            kernelD_e = 0.25 * 1j * k_e * cosθ_e * hankel1(1, k_e*distance_e)
            kernelS_e = 0.25 * 1j * hankel1(0, k_e*distance_e)
            
            kernelD_e_mn[m,n] = sum(kernelD_e * J * u) * dt
            kernelS_e_mn[m,n] = sum(kernelS_e * J * dvu) * dt
            
            xi_diff = x_i[m][n] - xbdy
            yi_diff = y_i[m][n] - ybdy
            
            distance_i = sqrt(xi_diff**2 + yi_diff**2)
            cosθ_i = (ν[0][:]*xi_diff + ν[1][:]*yi_diff) / distance_i
            
            kernelD_i = 0.25 * 1j * (k_i * cosθ_i * hankel1(1, k_i*distance_i))
            kernelS_i = 0.25 * 1j * hankel1(0, k_i*distance_i)

            kernelD_i_mn[m,n] = sum(kernelD_i * J * u) * dt
            kernelS_i_mn[m,n] = sum(kernelS_i * J * dvu) * dt
    
    v_e = fx + kernelD_e_mn - k_ratio * kernelS_e_mn
    v_i = -kernelD_i_mn + kernelS_i_mn
    
    #Impose Periodicity
    x_e[M][:] = x_e[0][:]
    y_e[M][:] = y_e[0][:]
    v_e[M][:] = v_e[0][:]
    
    x_i[M][:] = x_i[0][:]
    y_i[M][:] = y_i[0][:]
    v_i[M][:] = v_i[0][:]

    return x_i, x_e, y_i, y_e, v_i, v_e

