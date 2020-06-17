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

def PTR_contourplots(xbdy, ybdy, x_i, x_e, y_i, y_e, v_i, v_e):
    min_val_r = np.r_[np.real(v_i), np.real(v_e)].min()
    max_val_r = np.r_[np.real(v_i), np.real(v_e)].max()
    min_val_i = np.r_[np.imag(v_i), np.imag(v_e)].min()
    max_val_i = np.r_[np.imag(v_i), np.imag(v_e)].max();
    
    fig = plt.figure(figsize=(14,5))
    ax = fig.add_subplot(1 ,2 ,1)
    real1 = plt.contourf(x_i, y_i, np.real(v_i), 100, cmap=plt.get_cmap('coolwarm'),
                     vmin = min_val_r, vmax = max_val_r)
    real2 = plt.contourf(x_e, y_e, np.real(v_e), 100, cmap=plt.get_cmap('coolwarm'), 
                     vmin = min_val_r, vmax = max_val_r)
    ax.plot(xbdy, ybdy, 'r--')
    plt.title("Real Part of the PTR Solution")
    fig.colorbar(real1)
    plt.axis('off')

    ax = fig.add_subplot(1, 2, 2)
    imag1 = plt.contourf(x_i, y_i, np.imag(v_i), 100, cmap=plt.get_cmap('coolwarm'),
                     vmin = min_val_i, vmax = max_val_i)
    imag2 = plt.contourf(x_e, y_e, np.imag(v_e), 100, cmap=plt.get_cmap('coolwarm'), 
                     vmin = min_val_i, vmax = max_val_i)
    ax.plot(xbdy, ybdy, 'r--')
    plt.title("Imaginary Part of the PTR Solution")
    fig.colorbar(imag1)
    plt.axis('off');
    
    return None

def PTR_3dplots(x_i, x_e, y_i, y_e, v_i, v_e):
    min_val = np.r_[np.real(v_i), np.real(v_e)].min()
    max_val = np.r_[np.real(v_i), np.real(v_e)].max()

    data= [Surface(x=x_i, y=y_i, z=np.real(v_i), cmin=min_val, cmax=max_val),
       Surface(x=x_e, y=y_e, z=np.real(v_e), cmin=min_val, cmax=max_val)]

    layout=dict(xaxis=dict(zeroline=False),
            yaxis=dict(zeroline=False),
            width=500,
            height=500,
            margin=dict(l=50, r=50, b=100, t=100),
           title='Real Part of the Solution', hovermode='closest'
           )
          
    figure1=dict(data=data, layout=layout)          
    iplot(figure1)

    min_val = np.r_[np.imag(v_i), np.imag(v_e)].min()
    max_val = np.r_[np.imag(v_i), np.imag(v_e)].max();

    data= [Surface(x=x_i, y=y_i, z=np.imag(v_i), cmin=min_val, cmax=max_val), 
       Surface(x=x_e, y=y_e, z=np.imag(v_e), cmin=min_val, cmax=max_val)]

    layout=dict(xaxis=dict(zeroline=False),
            yaxis=dict(zeroline=False),
            width=500,
            height=500,
            margin=dict(l=50, r=50, b=100, t=100),
           title='Imaginary Part of the Solution', hovermode='closest'
           )
          
    figure2=dict(data=data, layout=layout)          
    iplot(figure2)
    
    return None