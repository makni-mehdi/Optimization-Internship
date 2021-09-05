import matplotlib
from matplotlib.widgets import Slider, Button
from IPython.display import clear_output, display

import random
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.interpolate import interp2d 

import argparse
import time


def plotpart(X, N, n, Xs, Ys, ax=None, FIGS=None, last=False):
    """
    This is the plotting function. 
    For each cell, we find the points in the grid that belong to it with the condition being the density is greter than a certain threshold given by 0.99
    We color the cells differently to have clear visualizations of the cells.
    If we are using ax i.e. it is not None then we are using the command line version of the algorithm in which case we have a window that shows each iteration and saves 5 plots in the same directory where the program was run.
    Otherwise, we are running the algorithm from the jupyter notebook version in which case the figures of each iteration are saved in a variable FIGS. We can move around between all the figures thanks to a slider inline the notebook.
    """
    xvec = np.zeros(n**2)
    for i in range(0,N):
        xvec[np.where((X[:,i] > 0.99))] = 2*i+5
    xvec = xvec.reshape((n,n))
    xvec = np.around(xvec)
    if ax is not None:
        ax.contourf(Xs, Ys, xvec)
        ax.axis("scaled")
        plt.show()
        if last:
            plt.savefig(f'figure_{n}.png')
        plt.pause(0.001)
    else:
        plt.contourf(Xs, Ys, xvec)
        plt.axis("scaled")
        FIGS.append(plt.figure())
    
def variableStepOptpart(n, N, c, iters, outer_shape, ax=None, FIGS=None, threshold=0.01, k=5, init_data=None):
    """
    This is the optimization function that runs the algorithm for fixed variables as follows:
        - n: Number of subdivisions of the side of the squared grid we work with. The higher n the better the resolution.
        - N: Number of cells inside the grid.
        - c: Penalization term for points outside a cell.
        - iters: The number of iterations to run the gradient descent with variable step.
        - outer_shape: The shape we start by penalizing. The values that it can take are 
            'S': square.
            'D': disk.
            'T': (equilateral) triangle.
        - ax: The axis on which we are going to display the figures if we run the command line version.
        - FIGS: The list that stores all figures and through which we can iterate thanks to a slider in the notebook version.
        - threshold: The variable representing the threshold after which a point is considered to belong to a cell. Note that for non initializaed data, it is set to be very low and consequently a point can belong to different cells.
        - k: The neighbours of a given point are the points that are at most k steps away from that point.
        - init_data: The dictionary that contains info about previous initialization if it exists.
    """
    # outer_shape is the shape we start by penalizing. The values it takes are S: Square and D: Disk
    
    new_data = {
        'n' : n,
        'N' : N,
    }
    
    print("The given paramteres are:")
    for key, value in new_data.items():
        print(f"{key} = {value}")
                        
    xs = np.linspace(0,1,n)
    ys = np.linspace(0,1,n)

    if init_data:
        # If previous data is available we interpolate.
        X = np.zeros((n**2,N))
        init_X = init_data['X']
        init_n = init_data['n']
        init_N = init_data['N']
        init_xs = np.linspace(0,1,init_n)
        init_ys = np.linspace(0,1,init_n)
        alpha = max(init_data['alpha'],1)
        threshold = 0.99
        if init_data['Nk'].shape[0] == n:
            Nk = init_data['Nk']
            L = init_data['L']
        else:
            N_matrix = diags([1,1,1,1],[-1,1,n,-n],shape=(n**2,n**2),dtype=int)
            Nk = np.linalg.matrix_power(np.matrix(N_matrix),k)[0,0]
            h = 1/n
            L = 1/h**2*diags([4,-1,-1,-1,-1],[0,-1,1,n,-n],shape=(n**2,n**2)) 
        for i in range(N):
            fi = interp2d(init_xs, init_ys, init_X[:,i].reshape(init_n, init_n), kind='linear')
            X[:,i] = fi(xs,ys).reshape(n**2)
            
    else:
        h = 1/n
        L = 1/h**2*diags([4,-1,-1,-1,-1],[0,-1,1,n,-n],shape=(n**2,n**2)) 
        X = np.random.rand(n**2,N)
        alpha = 1
        N_matrix = diags([1,1,1,1],[-1,1,n,-n],shape=(n**2,n**2), dtype=int)
        Nk = np.linalg.matrix_power(np.matrix(N_matrix), k)[0,0]
        
        
    X = normalize(X, axis=1, norm='l1')
    
    Xs, Ys = np.meshgrid(xs, ys)
    
    xf = Xs.flatten()
    yf = Ys.flatten()
    
    if outer_shape == 'S':
        indx1 = np.where(xf <= 1e-4)
        indx2 = np.where(xf >= 0.999)
        indy1 = np.where(yf <= 1e-4)
        indy2 = np.where(yf >= 0.999)
        
        ind = np.union1d(indx1,indx2)
        ind = np.union1d(ind,indy1)
        ind = np.union1d(ind,indy2)

    elif outer_shape == 'D':
        ind = np.where((yf-0.5)**2+(xf-0.5)**2 >= 0.2) #(0.25=r**2 where r is the radius equal to 0.5) 
        
    elif outer_shape == 'T':
        sr3 = np.sqrt(3)
        ind1 = np.where(yf-sr3*xf >= 0.05)
        ind2 = np.where(yf <= 0.1)
        ind3 = np.where(yf+sr3*xf >= sr3+0.05)
        
        ind = np.union1d(ind1, ind2)
        ind = np.union1d(ind, ind3)
        
    vec = np.zeros(n**2)
    vec[ind] = c    

    D   = diags(vec,0,shape=(n**2,n**2))
    L   = L+D

    val_prev = np.inf
    
    for tt in range(iters):
        val = 0
        X_bis = np.zeros((n**2,N))
        for i in range(N):
            phi_i = X[:,i].reshape(n**2,1) 
            # The shape of a density is (n^2,1)
            bool_i = phi_i >= threshold 
            # Array containing booleans whether points belong to density i (it belongs if density is greater than a certain threshold)
            v = Nk.dot(bool_i)
            coords_i = np.where(v>=1) 
            # N is symmetric so is N^k so no need for transpose
            coords_i = np.union1d(coords_i,np.where(bool_i == True))
            # Coords_i contains coordinates of points belonging to density i as well as their neighbors
            bool_i = phi_i[coords_i]
            n_i = len(bool_i)
            D_i = diags([c*(1-bool_i).ravel()],[0],shape=(n_i,n_i))
            L_i = L[coords_i,:][:,coords_i]
            A_i = L_i+D_i
            eigenvalue, u_i = eigsh(A_i,k=1,which='SM',tol=0.01) 
            # since A is real symmetric square matrix we use eigsh for speed
            u_i = u_i.real
            eigenvalue = eigenvalue.real
            val = val + eigenvalue
            u = np.zeros(n**2)
            u[coords_i] = u_i.reshape(n_i,)
            X_bis[:, i] = phi_i.reshape(n**2) + alpha*c*np.square(u).reshape(n**2)
        if val < val_prev:
            X = X_bis
            alpha = min(100*alpha, 1e32)
            val_prev = val
        else:
            alpha /= 1000
            if alpha < 1e-1:
                break
        
        X = normalize(X, axis=1, norm='l1')
        plotpart(X, N, n, Xs, Ys, ax, FIGS)
        print("Iter: ",tt," Val=",val, " Step=", alpha)

    new_data['val'] = val
    new_data['X'] = X
    new_data['alpha'] = alpha
    new_data['Nk'] = Nk
    new_data['L'] = L
    
    return new_data

    
def opt_algo(N,outer_shape, ax):
    """
    This function is called in the command line version.
    It runs the gradient descent with different resolutions and saves the plots in between for the output of different resolutions.
    """
    init_data = variableStepOptpart(n=50, N=N, c=1e3, iters=150, outer_shape=outer_shape, ax=ax, init_data=None)
    Xs, Ys = np.meshgrid(np.linspace(0, 1, init_data['n']), np.linspace(0, 1, init_data['n']))
    plotpart(init_data['X'], init_data['N'], init_data['n'], Xs, Ys, ax=ax, last=True)
    init_data = variableStepOptpart(n=100, N=N, c=1e4, iters=50, outer_shape=outer_shape, ax=ax, init_data=init_data)
    Xs, Ys = np.meshgrid(np.linspace(0, 1, init_data['n']), np.linspace(0, 1, init_data['n']))
    plotpart(init_data['X'], init_data['N'], init_data['n'], Xs, Ys, ax=ax, last=True)
    init_data = variableStepOptpart(n=150, N=N, c=1e5, iters=30, outer_shape=outer_shape, ax=ax, init_data=init_data)
    Xs, Ys = np.meshgrid(np.linspace(0,1,init_data['n']),np.linspace(0,1,init_data['n']))
    plotpart(init_data['X'], init_data['N'], init_data['n'], Xs, Ys, ax=ax, last=True)
    init_data = variableStepOptpart(n=200, N=N, c=1e5, iters=10, outer_shape=outer_shape, ax=ax, init_data=init_data)
    Xs, Ys = np.meshgrid(np.linspace(0,1,init_data['n']),np.linspace(0,1,init_data['n']))
    plotpart(init_data['X'], init_data['N'], init_data['n'], Xs, Ys, ax=ax, last=True)
    init_data = variableStepOptpart(n=250, N=N, c=1e5, iters=10, outer_shape=outer_shape, ax=ax, init_data=init_data)
    Xs, Ys = np.meshgrid(np.linspace(0, 1, init_data['n']), np.linspace(0, 1, init_data['n']))
    plotpart(init_data['X'], init_data['N'], init_data['n'], Xs, Ys, ax=ax, last=True)

def main(args):
    """
    This is the main function of the command line version.
    It specifies the visualization should be in a seperate window and calls opt_algo.
    """
    plt.ion()
    fig, ax = plt.subplots()
    N = args.number_partitions
    outer_shape = args.outer_shape
    opt_algo(N, outer_shape, ax)

    
if __name__ == "__main__":
    """
    Gets the parsing information in the command line version.
    The arguments are:
        -s / --outer_shape : Specifies the outer shape inside which we run the Optimal Partitioning.
        -N / --number_partitions : Specifies the number of cells for which we run the Optimal Partitioning.
    """
    
    parser = argparse.ArgumentParser(description='Optimiazation Internship code.')

    parser.add_argument("-s", "--outer_shape", help="Gives the outer_shape in which we run the Optimal Partitioning. \nBy default, it is 'S' corresponding to square.", default="S")

    parser.add_argument("-N", "--number_partitions", help="Gives the number of cells for which we run the Optimal Partitioning. \nBy default, it is 3.", default=3, type=int)

    args = parser.parse_args()

    main(args)    
    

    