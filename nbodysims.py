from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Iterable
import ode_solvers as ods
import numerical as pds
import matplotlib.animation as mpl_anim
import sys
import matplotlib.figure as mpfig
import os
from scipy.optimize import curve_fit
from timeit import default_timer
import multiprocessing as mp
from math import pi
from typing import Dict
import numba as nb

'''
Create N-body simulations using the Barnes-Hut algorithm
or direct summation between particles.
Both algorithms have been significantly sped-up from pure python
by implementing a vectorized version using numpy.
'''



@nb.njit
def Uint(d, epsilon=0.0):
    return -1/(d**2 + epsilon**2)**0.5

@nb.njit
def _fint(s: np.ndarray, d, epsilon):
    return s/(d**2 + epsilon**2)**1.5

@nb.njit
def fint(s: np.ndarray, epsilon=0.0):
    d = norm(s)
    return _fint(s, d, epsilon)


def _vdot_direct_vectorized(x:np.ndarray[np.ndarray], m: np.ndarray, G=1.0, epsilon=0.0):
    N = len(x)
    a = np.zeros((N, 2))
    for i in nb.prange(N):
        s = x - x[i]
        a[i] = np.sum(G*m[:,np.newaxis] * _fint(s, norm(s)[:, np.newaxis], epsilon), axis=0)
    return a


@nb.njit(parallel=True)
def _vdot_direct_vectorized_parallel(x:np.ndarray[np.ndarray], m: np.ndarray, G=1.0, epsilon=0.0):
    N = len(x)
    a = np.zeros((N, 2))
    for i in nb.prange(N):
        for j in range(N):
            if i != j:
                s = x[j]-x[i]
                a[i] += G*m[j]*fint(s, epsilon)
    return a


@nb.njit
def Energy(x: np.ndarray, v: np.ndarray, m: np.ndarray, G, epsilon):
    '''
    x: shape = (N, 2)
    v: shape = (N, 2)
    m: shape = (N,)

    The energy is calculated analytically for a given epsilon.
    If epsilon == 0, the well known formula U ~ 1/r is used for the potential
    Otherwise, the formula used is \int dr/(r^2 + e^2) ~ arctan(r/e)
    The second formula approaches 1/r when e->0
    '''
    K = 1/2*np.sum(m*np.sum(v**2, axis=1))
    U = 0
    for i in range(len(x)-1):
        s = x[i, np.newaxis]-x[i+1:]
        dx = norm(s)
        U += np.sum(G*m[i]*m[i+1:]*Uint(dx, epsilon))
    return K + U


def draw_samples(pdf, a, b, N, n=int(1e6), args=()):
    '''
    Draw n samples out of a probability distribution function

    Parameters
    ----------
    pdf (func: x (float) -> float): The probability distribution function (not necessarily normalized)
    a (float): The minimum value of the pdf parameter "x"
    b (float): The maximum value of the pdf parameter "x"
    N (int): The number of samples
    n (int): Number of points to discretize the pdf
    args: Additional arguments to pass into pdf: pdf(x, *args)

    Returns
    -------
    x_sampled: np.ndarray

    Note: It would be best if the pdf is dimensionless,
    because physical constants can get very large or very small in arbitraty units
    '''
    x    = np.linspace(a, b, n)
    npdf = pdf(x, *args)/np.sum(pdf(x, *args)) #normalized pdf: npdf(x) = pdf(x)/(\int_a^b pdf(x)*dx)
    Ipdf = np.cumsum(npdf) #Integrated probability: P(x) = \int_0^x npdf(t)*dt
    P    = np.random.uniform(0, 1, size = N)
    x_sampled = np.interp(P, Ipdf, x)
    
    return x_sampled


class DensityModel:

    '''
    Base class for defining a spherically symmetric density model
    '''
    _rho: Callable

    def __init__(self, rho: Callable, rmax):
        self._rho = rho
        self.rmax = rmax

    def plot(self):
        pass

    def rho(self, r):
        return self._rho(r)
    
    def draw(self, N):
        '''
        \int_0^rmax rho(r) 4πr^2 dr ~ N for constant mass stars
        So the probability density that a star is located at some radius r
        is dP/dr ~ rho(r) * r**2

        The pdf will be normalized at rmax (makes sense only when rmax -> oo)
        '''
        return draw_samples(self._pdf, 0, self.rmax, N)
    
    def _pdf(self, r):
        #not normalized
        return self.rho(r) * r**2

    def draw_keplerian(self, N, m: np.ndarray, G=1.0):
        '''
        Draw N random particles

        Returns
        ------------
        x: shape=(N, 2), the projection of the particles' coordinates on the x-y plane
        v: shape=(N, 2), keplerian velocity of each particle
        '''
        r = self.draw(N)
        p1, p2 = np.random.uniform(0, 1, N), np.random.uniform(0, 1, N)
        theta, phi = np.arccos(2*p1-1), 2*pi*p2
        v = []
        for i in range(N):
            M_in = np.sum(m[r<r[i]])
            v.append(np.sqrt(G*M_in/r[i]))
        v = np.array([np.sin(phi)*v, -np.cos(phi)*v]).transpose()
        x = np.array([r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi)]).transpose()
        return x, v

    def keplerian_velocities(self, x: np.ndarray, m: np.ndarray, G=1.0):
        v = np.zeros((len(x), 2))
        r = norm(x)
        for i in range(len(x)):
            if r[i] == 0.0:
                v[i,:] = 0.0
            else:
                M_in = np.sum(m[r<r[i]])
                vi = np.sqrt(G*M_in/r[i])
                v[i] = np.array([x[i,1]/r[i]*vi, -x[i,0]/r[i]*vi]).transpose()
        return v

class AnalyticalDensityModel(DensityModel):
    '''
    Class representing a density model whose reverse integrated probability density is given analytically.
    The Plummer density model makes use of this class.
    '''
    def __init__(self, rho: Callable, rho_inv: Callable[..., np.ndarray], rmax):
        super().__init__(rho, rmax)
        self.rho_inv = rho_inv

    def draw(self, N):
        r = []
        while len(r) < N:
            p = np.random.uniform(0, 1, N)
            r_arr = self.rho_inv(p)
            r_arr = r_arr[r_arr<=self.rmax]
            r += r_arr.tolist()
        return np.array(r[:N])


def Plummer(N, a=1, M=1, rv=1, rmax=10, G=1, central_mass=0.0)->tuple[np.ndarray]:
    '''
    Draw N particle coordinates and velocities, assuming all of them have the same mass M/N,
    are distributed around the center following the Plummer density model, 
    and have keplerian veolocities


    Returns
    ------------
    x: shape = (N, 2)
    v: shape = (N, 2)
    m: shape = (N,)
    '''
    def rho(r):
        return 3*M/(4*pi*a**3) * (1 + r**2/a**2)**-2.5
    
    def rho_inv(p):
        return rv/np.sqrt(p**(-2/3) - 1)
    
    m = M/N*np.ones(N)
    if central_mass > 0.0:
        m[0] = central_mass
        pl = AnalyticalDensityModel(rho, rho_inv, rmax)
        x = pl.draw_keplerian(N, m, G)[0]
        x[0, :] = 0.0
        v = pl.keplerian_velocities(x, m, G)
        return x, v, m
    else:
        return *AnalyticalDensityModel(rho, rho_inv, rmax).draw_keplerian(N, m, G), m


class QuadTreeNode:
    '''
    Tree node in 2 dimensions, with properties relevant to the Barnes-Hut algorithm
    '''

    pos: np.ndarray
    L: float
    nmax: int
    mass: float
    x_cm: np.ndarray
    m: list[float]
    x: list[np.ndarray]
    children: list[QuadTreeNode]
    N: int

    def __init__(self, x, L, nmax = 1):
        self.pos = x # xm = x + L/2
        self.L = L
        self.nmax = nmax
        self.empty()

    def __repr__(self):
        return f'Node({self.pos[0]}, {self.pos[1]}, L={self.L})'

    @property
    def is_external(self):
        return not self.children #empty list returns False

    def split(self, dmin: float):
        '''
        Split the node in 4 children nodes (can only call this method once)

        The node preserves its current attributes (mass, number of particles, ...)
        but its particles are also distributes in its new 4 children nodes.
        '''



        '''
        |-----------|
        |(0,1)|(1,1)|       2   3
        |-----------| --->  
        |(0,0)|(1,0)|       0   1
        |-----------|
        '''
        if self.is_external:
            x, L = self.pos, self.L
            dx, dy = [L/2, 0], [0, L/2]
            args = self.nmax,
            self.children = [QuadTreeNode(x, L/2, *args), QuadTreeNode(x+dx, L/2, *args), QuadTreeNode(x+dy, L/2, *args), QuadTreeNode(x+L/2, L/2, *args)]
            for i in range(self.N):
                self._insert(self.x[i], self.m[i], dmin)
        else:
            raise NotImplementedError('.split() only applies for external nodes')

    def _insert(self, x: np.ndarray, m: float, dmin: float):
        #map the particle to the right child node
        #without affecting self parameters
        #internal use only
        i, j = int(x[0] > self.pos[0]+self.L/2), int(x[1] > self.pos[1]+self.L/2)
        self.children[2*j+i].add_particle(x, m, dmin)
    
    def add_particle(self, x: np.ndarray, m: float, dmin=0.0):
        '''
        Add a particle in this node with cartesian coordinates x = [x1,x2]
        '''
        s = x-self.pos
        if not ((0 <= s[0] <= self.L) and (0 <= s[1] <= self.L)):
            raise ValueError('x parameter does not belong in this node')
        self.m.append(m)
        self.x.append(x)
        self.x_cm = (self.mass*self.x_cm + m*x)/(self.mass + m)
        self.mass += m
        if self.L/2 > dmin:
            if self.N == self.nmax:
                self.split(dmin)
            if self.N >= self.nmax:
                self._insert(x, m, dmin)
        self.N += 1

    def plot(self, ax: plt.Axes, linewidth=None):
        if not self.is_external:
            ax.plot([self.pos[0], self.pos[0]+self.L], 2*[self.pos[1]+self.L/2], color='k', linewidth=linewidth, zorder=0)
            ax.plot(2*[self.pos[0]+self.L/2], [self.pos[1], self.pos[1]+self.L], color='k', linewidth=linewidth, zorder=0)
            for child in self.children:
                child.plot(ax, linewidth=linewidth)

    def empty(self):
        self.mass = 0.0
        self.x_cm = self.pos+self.L/2 #not important initially
        self.m = []
        self.x: list[np.ndarray] = []
        self.children: list[QuadTreeNode] = []
        self.N = 0


class QuadTree:

    def __init__(self, L, nmax=1, theta=0.0, epsilon=0.0, dmin=0.0, G=1.0):
        self.L = L
        self.nmax = nmax
        self.dmin = dmin
        self.theta = theta
        self.epsilon = epsilon
        self.G = G
        self.node = QuadTreeNode(vec(-L, -L), 2*L, nmax)

    def update(self, x: np.ndarray):
        '''
        Update the tree with new particle coordinates. The number of particles
        should be the same as the number of particles in the previous update.

        x: shape = (number of particles, 2)
        '''
        self.node.empty()
        for i in range(len(x)):
            if abs(x[i,0]) < self.L and abs(x[i,1]) < self.L:
                self.node.add_particle(x[i], self.m[i], self.dmin)

    def Energy(self, x, v):
        return Energy(x, v, self.m, self.G, self.epsilon)

    def vdot_direct_vectorized_parallel(self, x: np.ndarray):
        '''
        Calculate the exact forces between using a vectorized version of direct summation algorithm, that scales as n^2.
        For most cases, this method is the fastest in python, because it approaches numpy speed, while the vectorized form
        of the Barnes-Hut algorithm, although faster than than the non-vectorized version, cannot squeeze so much
        performance out of numpy. In a faster (compiled) programming language, the Barned-Hut algorithm wins for lower
        number of particles.
        '''
        return _vdot_direct_vectorized_parallel(x, self.m, self.G, self.epsilon)
    
    def vdot_direct_vectorized(self, x: np.ndarray):
        return _vdot_direct_vectorized(x, self.m, self.G, self.epsilon)

    def vdot_direct(self, x: np.ndarray):
        '''
        The naive approach to calculating all forces between particles.
        This is the slowest implementation.
        '''
        N = len(x)
        a = np.zeros((N, 2))
        for i in range(N):
            for j in range(N):
                if i != j:
                    a[i] += self.G * self.m[j] * fint(x[j]-x[i], self.epsilon)
        return a

    def vdot_bh_vectorized(self, x: np.ndarray):
        '''
        This is the barnes-hut algorithm, but istead of calculating the forces from all nodes
        on a specific particle (iterating all particles), the nodes are iterated instead.
        On every iteration, the force of a node on all particles (wherever applied) is calculated,
        before moving on to the next node (or child node). The result is numerically identical to
        the vdot_bh function, but is a lot faster depending on the theta parameter, and the number
        of particles. This is because the algorithm is partly vectorized, taking advantage of
        the numpy speed-up.
        
        '''
        self.update(x)
        N = len(x)
        a = np.zeros((N, 2))
        gen = [self.node]
        p_node = [np.ones(N, dtype=bool)]
        while gen:
            node = gen.pop()
            p = p_node.pop()
            s = node.x_cm[np.newaxis, :] - x
            d = norm(s)
            far = np.logical_and(self.theta*d>node.L, p)
            cl = np.logical_and(~far, p)
            a[far] += self.G*node.mass*_fint(s[far], d[far, np.newaxis], self.epsilon)
            if node.is_external:
                g = 0.0 #the total force applied on every closeby particle by the entire node
                xcl = x[cl]
                for j in range(node.N):
                    s = node.x[j][np.newaxis, :] - xcl #TODO
                    d = norm(s)[:, np.newaxis]
                    g += self.G*self.m[j]*_fint(s, d, self.epsilon)
                a[cl] += g
            else:
                p_node += len(node.children)*[cl]
                gen += node.children
        
        return a

    def vdot_bh_iterative(self, x: np.ndarray):
        '''
        The standard Barnes-Hut algorithm. This is a non-recursive implementation,
        and a "breadth first search"-like algorithm is used.
        '''
        self.update(x)
        N = len(x)
        a = np.zeros((N, 2))
        for i in range(N):
            gen = [self.node]
            while gen:
                node = gen.pop()
                s = node.x_cm - x[i]
                d = norm(s)
                if self.theta*d > node.L:
                    a[i] += self.G*node.mass*fint(s, self.epsilon)
                elif node.is_external:
                    for j in range(node.N):
                        s = node.x[j] - x[i]
                        d = norm(s)
                        if d != 0:
                            a[i] += self.G*self.m[j]*fint(s, self.epsilon)
                else:
                    gen += node.children
        return a
    
    def vdot_bh_recursive(self, x: np.ndarray):
        '''
        The standard Barnes-Hut algorithm. This is a non-recursive implementation,
        and a "breadth first search"-like algorithm is used.
        '''
        self.update(x)
        N = len(x)
        a = np.zeros((N, 2))
        for i in range(N):
            a[i] = self.vdot_from_node(x[i], self.node)
        return a
    
    def vdot_from_node(self, x: np.ndarray, node: QuadTreeNode):
        s = node.x_cm - x
        d = norm(s)
        if self.theta * d > node.L:
            a = self.G*node.mass*fint(s, self.epsilon)
        elif node.is_external:
            a = 0
            for j in range(node.N):
                s = node.x[j] - x
                a += self.G*self.m[j]*fint(s, self.epsilon)
        else:
            a = 0
            for child in node.children:
                a += self.vdot_from_node(x, child)
        return a
    
    def _vdot_direct_vectorized_parallel(self, t, x):
        '''
        Function compatible with the ode solver
        '''
        return self.vdot_direct_vectorized_parallel(x)

    def _vdot_direct_vectorized(self, t, x):
        '''
        Function compatible with the ode solver
        '''
        return self.vdot_direct_vectorized(x)
    
    def _vdot_direct(self, t, x):
        '''
        Function compatible with the ode solver
        '''
        return self.vdot_direct(x)

    def _vdot_bh_vectorized(self, t, x):
        '''
        Function compatible with the ode solver
        '''
        return self.vdot_bh_vectorized(x)

    def _vdot_bh(self, t, x):
        '''
        Function compatible with the ode solver
        '''
        return self.vdot_bh_iterative(x)

    def set_particles(self, x: np.ndarray, m: np.ndarray):
        '''
        Update the tree with new particles. The number of particles can be arbitrary, not
        related to the number of particles in the previous update.

        x: shape = (number of particles, 2)
        '''
        self.m = m
        self.update(x)

    def plot(self, gridlines = True, cm=False):
        fig, ax = plt.subplots()
        ax.set_xlim(-self.L, self.L)
        ax.set_ylim(-self.L, self.L)
        L = self.L
        ax.plot([-L, L, L, -L, -L], [-L, -L, L, L, -L], linewidth=0.5)
        if gridlines:
            self.node.plot(ax, linewidth=0.5)
        for v in self.node.x:
            ax.scatter(*v, color='r', s=0.5, zorder=1)
        if cm:
            ax.scatter(*self.node.x_cm, color='blue')
        return fig, ax


class NBodySimulation:
    t: np.ndarray
    x: np.ndarray[np.ndarray]
    v: np.ndarray[np.ndarray]
    energy: np.ndarray

    def __init__(self, x0: np.ndarray, v0: np.ndarray, m: np.ndarray, L: float, nmax=1, theta=0.0, epsilon=0.0, dmin=0.0, G=1.0, parallel=True):
        '''
        x0 = [(x1, y1), (x2, y2), ...]
        v0 = [(vx1, vy1), (vx2, vy2), ...]
        m = [m1, m2, ...]
        '''
        self.N = len(x0)
        self.x0 = x0
        self.v0 = v0
        self.m = m
        self.tree = QuadTree(L, nmax, theta, epsilon, dmin, G)
        self.tree.set_particles(x0, m)
        
        ics = (0, x0, v0)

        self.ode: Dict[str, ods.HamiltonianSystem] = {}
        if self.N > os.cpu_count() and parallel is True:
            self.ode['direct'] = ods.ForceField(self.tree._vdot_direct_vectorized_parallel, ics=ics)
        else:
            self.ode['direct'] = ods.ForceField(self.tree._vdot_direct_vectorized, ics=ics)

        if theta < 5:
            self.ode['bh'] = ods.ForceField(self.tree._vdot_bh_vectorized, ics=ics)
        else:
            self.ode['bh'] = ods.ForceField(self.tree._vdot_bh, ics=ics)

    def get(self, t, dt, algorithm, method: ods.HamiltonianSystem.literal = 'RK4', **kwargs)->tuple[np.ndarray]:
        res: np.ndarray
        self.runtime, (t, res) = timeit(self.ode[algorithm].solve, t, dt, method, **kwargs) #res_ijkl: i=timestep, j=x or v, k=k_th body, l=x/y-component
        x, v = res.swapaxes(0, 3).swapaxes(0, 2).swapaxes(0, 1) #x_ijk: i=body, j=component, k=step
        self.t, self.x, self.v, self.energy = t, x, v, np.array([self.tree.Energy(x[:,:,i], v[:,:,i]) for i in range(len(t))])

        method_map = {'bh': 'Barnes-Hut', 'direct': 'Direct sum'}
        self.method_log = method, method_map[algorithm], f'dt = {dt}'
        return t, x, v, self.energy

    def get_direct(self, t, dt, method='RK4', **kwargs):
        return self.get(t, dt, algorithm='direct', method=method, **kwargs)

    def get_barnes_hut(self, t, dt, method='RK4', **kwargs):
        return self.get(t, dt, algorithm='bh', method=method, **kwargs)

    def _plot(self, i, t, x, energy, fig: mpfig.Figure, show_trajectories, xlims=None, ylims=None, color_particles=False):
        if i is None:
            i = len(t)-1
        ax1, ax2, ax3, ax4 = fig.axes
        if xlims is None:
            xlims = -self.tree.L, self.tree.L
        if ylims is None:
            ylims = -self.tree.L, self.tree.L
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        ax1, ax2, ax3, ax4 = fig.axes
        ax1.clear(), ax2.clear(), ax3.clear(), ax4.clear()
        

        ax1.set_xlim(*xlims), ax1.set_ylim(*ylims)
        ax2.set_xlim(*xlims), ax2.set_ylim(*ylims)
        if show_trajectories:
            for j in range(self.N):
                c = colors[color_particles*j]
                ax1.plot(x[j,0,:i+1], x[j,1,:i+1], c=c, linewidth=0.5)
                ax2.plot(x[j,0,:i+1], x[j,1,:i+1], c=c, linewidth=0.5)

        if color_particles:
            ax1.scatter(x[:,0,i], x[:,1,i], s=0.5, c = colors[:self.N], zorder=1)
            ax2.scatter(x[:,0,i], x[:,1,i], s=0.5, c = colors[:self.N], zorder=1)
        else:
            ax1.scatter(x[:,0,i], x[:,1,i], s=0.5)
            ax2.scatter(x[:,0,i], x[:,1,i], s=0.5)

        self.tree.update(x[:,:,i])
        self.tree.node.plot(ax2, linewidth=0.5)

        ax3.plot(t[:i+1], energy[:i+1])
        ax3.axhline(energy[0], label=r'$E_0$', c='r', lw=0.5)
        ax3.axhline(0, c='k', lw=0.5)
        ax3.legend(loc='upper right', fontsize=5)
        ax4.plot(t[:i+1], np.abs((energy[:i+1]-energy[0])/energy[0]), label=r'$|\Delta E/E_0|$')
        ax4.legend(loc='upper right', fontsize=5)

        fig.subplots_adjust(hspace=0.3, wspace=0.5)
        fig.suptitle(r'$N = %d, \theta$ = %g, $\epsilon$ = %g, $n_{max}$ = %d, $d_{min}$ = %s'% (self.N, self.tree.theta, self.tree.epsilon, self.tree.nmax, pds.format_number(self.tree.dmin))+'\n'+', '.join(self.method_log)+'\n'+'t = %g' % t[i])
        fig.subplots_adjust(top=0.8)

    def plot(self, i=None, xlims=None, ylims=None, show_trajectories=False, color_particles=False):
        fig, axes = plt.subplots(2, 2)
        self._plot(i, self.t, self.x, self.energy, fig, show_trajectories, xlims, ylims, color_particles)
        return fig, axes

    def animate(self, duration, fps, save: str, focus_on_timestep=False, show_trajectories=False, xlims=None, ylims=None, color_particles=False):
        '''
        Focus_on_timestep: if True, then the idea is that the frames shown correspond to the timestep of the solution. So the interval between two frames
        in the animation must be proportional to the timestep used in the solution. So the animations "slows down" where the ode is very stiff if the method used
        was one with an adaptive time step.
        
        The idea is to create a function/map, t_prime(t):
            This function maps the real-time variable "t" to the t_prime variable of the solution time-grid
        '''
        if focus_on_timestep:
            real_time_grid = pds.Uniform1D(0, duration, n=fps*duration)
            t_prime = pds.LambdaField(self.t, np.linspace(0, duration, num=len(self.t))).array(real_time_grid) #t_prime(t) discretized
        else:
            t_prime = np.linspace(0, self.t[-1], num=fps*duration)
        
        t_prime_grid = pds.Unstructured1D(t_prime)
        x = np.array([[pds.LambdaField(self.x[i, j], self.t).array(t_prime_grid) for j in range(2)] for i in range(self.N)])
        energy = pds.LambdaField(self.energy, self.t).array(t_prime_grid)

        fig, _ = plt.subplots(2, 2)
        def update(i):
            pds.fprint('Animating: %.1f' % (i/x.shape[2]*100) + ' %')
            self._plot(i, t_prime, x, energy, fig, show_trajectories, xlims, ylims, color_particles)

        anim_dir = os.path.join(sys.path[0], 'animations')

        if not os.path.exists(anim_dir):
            os.makedirs(anim_dir)
        ani = mpl_anim.FuncAnimation(fig, update, frames=x.shape[2], interval=0)

        ani.save(os.path.join(anim_dir, save), fps=x.shape[2]/duration, dpi=500, writer='ffmpeg')
        return ani
    
    def fast_animation(self, duration, fps, save: str, plot_energy=False, focus_on_timestep=False, xlims=None, ylims=None):
        if focus_on_timestep:
            real_time_grid = pds.Uniform1D(0, duration, n=fps*duration)
            t_prime = pds.LambdaField(self.t, np.linspace(0, duration, num=len(self.t))).array(real_time_grid) #t_prime(t) discretized
        else:
            t_prime = np.linspace(0, self.t[-1], num=fps*duration)
        
        t_prime_grid = pds.Unstructured1D(t_prime)
        x = np.array([[pds.LambdaField(self.x[i, j], self.t).array(t_prime_grid) for j in range(2)] for i in range(self.N)])
        if xlims is None:
            xlims = -self.tree.L, self.tree.L
        if ylims is None:
            ylims = -self.tree.L, self.tree.L
        energy = pds.LambdaField(self.energy, self.t).array(t_prime_grid)
        elims = [min([0, energy[0]]), max([0, energy[0]])]

        if plot_energy:
            fig, _ = plt.subplots(1, 2, figsize=(12, 5))
            axes = fig.axes
            scatter = axes[0].scatter([], [], s=1)
            l1 = axes[1].axhline(0, c='k')
            l2 = axes[1].axhline(energy[0], c='r')
            l3 = axes[1].plot([], [])[0]
            axes[1].set_xlabel('time')
            axes[1].set_ylabel('energy')
        else:
            fig, ax = plt.subplots()
            scatter = ax.scatter([], [], s=1)
        
        def init_func():
            fig.subplots_adjust(top=0.8)
            if plot_energy:
                axes[0].set_xlim(*xlims)
                axes[0].set_ylim(*ylims)
                return scatter, l1, l2
            else:
                ax.set_xlim(*xlims)
                ax.set_ylim(*ylims)
                return scatter,
        
        def update(i, elims=None):
            pds.fprint('Animating: %.1f' % (i/x.shape[2]*100) + ' %')
            fig.suptitle(r'$N = %d, \theta$ = %g, $\epsilon$ = %g, $n_{max}$ = %d, $d_{min}$ = %s'% (self.N, self.tree.theta, self.tree.epsilon, self.tree.nmax, pds.format_number(self.tree.dmin))+'\n'+', '.join(self.method_log)+'\n'+'t = %g' % t_prime[i])
            scatter.set_offsets(x[:,:,i])
            if plot_energy:
                l3.set_data(t_prime[:i+1], energy[:i+1])
                if energy[i] < elims[0]:
                    elims[0] = min([energy[i], 0, energy[0]])
                elif energy[i] > elims[1]:
                    elims[1] = max([energy[i], 0, energy[0]])
                de = elims[1] - elims[0]
                if i > 0:
                    bottom = elims[0] - de*0.1
                    top = elims[1] + de*0.1
                    axes[1].set_xlim(0, t_prime[i])
                    axes[1].set_ylim(bottom, top)
                return scatter, l3
            else:
                return scatter,
            
        
        anim_dir = os.path.join(sys.path[0], 'animations')
        if plot_energy:
            ani = mpl_anim.FuncAnimation(fig, update, x.shape[2], init_func, interval=0, fargs=(elims,), blit=False)
        else:
            ani = mpl_anim.FuncAnimation(fig, update, x.shape[2], init_func, interval=0, blit=True)
        ani.save(os.path.join(anim_dir, save), fps=x.shape[2]/duration, dpi=500, writer='ffmpeg')
        return ani


def _n2(n, a):
    return a * n**2

def _nlogn(n, a):
    return a * n*np.log(n)

class ComplexityEstimator:

    methods = 'vdot_direct', 'vdot_bh_recursive', 'vdot_bh_iterative', 'vdot_bh_vectorized', 'vdot_direct_vectorized','vdot_direct_vectorized_parallel'
    names = 'Direct', 'Barnes-Hut (recursive)', 'Barnes-Hut (iterative)', 'Barnes-Hut (vectorized)', 'Direct (vectorized)', 'Direct (vectorized, parallel)'

    def __init__(self, L, nmax=1, theta=0.0, epsilon=0.0, dmin=0.0, i: Iterable = None):
        if i is None:
            i = range(self.methods)
        self.tree = QuadTree(L, nmax, theta, epsilon, dmin)
        self.vdot = [getattr(self.tree, self.methods[j]) for j in i]
        complexity = [_n2, _nlogn, _nlogn, _nlogn, _n2, _n2]
        self.complexity = [complexity[j] for j in i]
        self.methods = [self.methods[j] for j in i]
        self.names = [self.names[j] for j in i]


    def timeit(self, n1, n2, Ntry=1):
        n = np.linspace(n1, n2+1, min([n2-n1+1, 50]), dtype=int)
        t = np.zeros((len(self.methods), len(n)))
        p = 0
        for j in range(Ntry):
            for k, ni in enumerate(n):
                x, v, m = Plummer(ni)
                self.tree.set_particles(x, m)
                for i, f in enumerate(self.vdot):
                    if i >= len(self.methods)-2:
                        f(x)
                    ti = timeit(f, x)[0]
                    while (ti*n[k-1]**2 > 2*t[i, k-1]*n[k]**2 and k > 0):
                        ti = timeit(f, x)[0]
                    t[i, k] += ti
                    pds.tools.fprint(('%.5g' % ((j*len(n) + k)/(len(n)*Ntry)*100))+' %')
                p += 1
        t = t/Ntry

        tn = [curve_fit(self.complexity[i], n, t[i])[0][0] for i in range(len(self.methods))]
        return tn, t
        
    def plot(self, n1, n2, Ntry=1, ax=None):
        n = np.linspace(n1, n2+1, min([n2-n1+1, 50]), dtype=int)
        tn, t = self.timeit(n1, n2, Ntry)
        c = ['b', 'r', 'green', 'orange', 'brown', 'k']
        if ax is None:
            fig, ax = plt.subplots()
            figax = fig, ax
        else:
            figax = ax

        for i, f in enumerate(self.methods):
            ax.plot(n, t[i], label=self.names[i], c=c[i])
            # ax.plot(n, self.complexity[i](n, tn[i]), linestyle='--', c=c[i])
        ax.legend()
        ax.set_title('Total acceleration calculation time\n'+r'$\theta$ = %.2f' % self.tree.theta)
        ax.set_xlabel('number of particles')
        ax.set_ylabel('time [s]')
        return figax


class MultiSimulation:
    def __init__(self, t, dt, N: list, theta_vals: list, e_vals:list, methods: list, algorithms=['bh']):

        sims:list[NBodySimulation] = []
        combs = []
        for n in N:
            x, v, m = Plummer(n)
            for theta in theta_vals:
                for epsilon in e_vals:
                    for algo in algorithms:
                        for method in methods:
                            sims.append(NBodySimulation(x, v, m, L=10, theta=theta, epsilon=epsilon, dmin=1e-8, parallel=False))
                            combs.append([sims[-1], algo, method])

        self.N, self.theta_vals, self.e_vals = N, theta_vals, e_vals
        self.methods = methods
        self.algorithms = algorithms
        self.t = t
        self.dt = dt
        self.sims = sims
        self.run_combs = combs
        

    def run(self, cores=os.cpu_count()):
        with mp.Pool(cores) as pool:
            res = pool.starmap(self._runtime, self.run_combs)
            n_tasks = len(res)
            print('Estimated computation time: %.2f sec' % (((n_tasks-1)//cores + 1)*sum(res)/len(res)*self.t/self.dt))
            time_tot, self.sims = timeit(pool.starmap, self._run, self.run_combs)
            print('Computation complete: %.2f sec' % time_tot)

    def _runtime(self, sim: NBodySimulation, algorithm, method):
        if algorithm == 'direct':
            sim.get(self.dt, self.dt, algorithm, method)
        sim.get(self.dt, self.dt, algorithm, method)
        return sim.runtime
    
    def _run(self, sim: NBodySimulation, algorithm, method):
        sim.get(self.t, self.dt, algorithm, method)
        return sim

    def plot_results(self):
        for sim in self.sims:
            sim.plot()

    def plot_best_results(self):
        sims = self.get_best_sims()
        for sim in sims:
            sim.plot()
    
    def get_best_sims(self):
        self.run()
        sims: list[NBodySimulation] = []
        i = 0
        for n in self.N:
            for theta in self.theta_vals:
                for epsilon in self.e_vals:
                    chi2 = []
                    j = 0
                    for algo in self.algorithms:
                        for method in self.methods:
                            dE = self.sims[i].energy-self.sims[i].energy[0]
                            E = self.sims[i].energy[0]
                            chi2.append(np.sum((dE/E)**2)**0.5)
                            i += 1
                            j += 1
                    k = chi2.index(min(chi2))
                    sims.append(self.sims[i-j+k])
        return sims

    
def timeit(func, *args, **kwargs):
    t1 = default_timer()
    res = func(*args, **kwargs)
    t2 = default_timer()
    return  t2 - t1, res

def vec(x, y):
    return np.array([x, y])

@nb.njit
def norm(vec: np.ndarray):
    if len(vec.shape) == 1:
        return (vec[0]**2 + vec[1]**2)**0.5
    else:
        return np.sum(vec**2, axis=1)**0.5
