# For matplotlib plots
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# For plotly plots
from plotly.offline import plot, iplot, init_notebook_mode
from plotly.graph_objs import *
init_notebook_mode()
init_notebook_mode(connected=True)
from IPython.display import HTML

# Inputs:  xbdy, ybdy, x_i, y_i, v_i, x_e, y_e, v_e, exact_i, exact_e
# Outputs: Plots of PTR solution and Exact solution, real and imaginary

def comparison_plots(xbdy, ybdy, x_i, x_e, y_i, y_e, v_i, v_e, exact_i, exact_e):
    fig = plt.figure(figsize=(12,10))
    fig.suptitle("Ellipse Boundary for k_e = k_i")
    ax = fig.add_subplot(2 ,2 ,1)
    real1 = plt.contourf(x_i, y_i, np.real(v_i), 100, cmap=plt.get_cmap('viridis'), vmin=-1, vmax=1)
    real2 = plt.contourf(x_e, y_e, np.real(v_e), 100, cmap=plt.get_cmap('viridis'), vmin=-1, vmax=1)
    ax.plot(xbdy, ybdy, 'r--')
    fig.colorbar(real1, ticks = [-1, -0.5, 0, 0.5, 1])
    plt.axis('off')

    ax = fig.add_subplot(2, 2, 2)
    imag1 = plt.contourf(x_i, y_i, np.imag(v_i), 100, cmap=plt.get_cmap('viridis'), vmin=-1, vmax=1)
    imag2 = plt.contourf(x_e, y_e, np.imag(v_e), 100, cmap=plt.get_cmap('viridis'), vmin=-1, vmax=1)
    ax.plot(xbdy, ybdy, 'r--')
    plt.title("Imaginary Part of the PTR Solution")
    fig.colorbar(imag1, ticks = [-1, -0.5, 0, 0.5, 1])
    plt.axis('off')


    ax = fig.add_subplot(2 ,2 ,3)
    real1 = plt.contourf(x_i, y_i, np.real(exact_i), 100, cmap=plt.get_cmap('viridis'), vmin=-1, vmax=1)
    real2 = plt.contourf(x_e, y_e, np.real(exact_e), 100, cmap=plt.get_cmap('viridis'), vmin=-1, vmax=1)
    ax.plot(xbdy, ybdy, 'r--')
    plt.title("Real Part of the Exact Solution")
    fig.colorbar(real1, ticks = [-1, -0.5, 0, 0.5, 1])
    plt.axis('off')


    ax = fig.add_subplot(2, 2, 4)
    imag1 = plt.contourf(x_i, y_i, np.imag(exact_i), 100, cmap=plt.get_cmap('viridis'), vmin=-1, vmax=1)
    imag2 = plt.contourf(x_e, y_e, np.imag(exact_e), 100, cmap=plt.get_cmap('viridis'), vmin=-1, vmax=1)
    ax.plot(xbdy, ybdy, 'r--')
    plt.title("Imaginary Part of the Exact Solution")
    fig.colorbar(imag1, ticks = [-1, -0.5, 0, 0.5, 1])
    plt.axis('off');

    return None