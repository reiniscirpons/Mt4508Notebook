from itertools import product
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from tabulate import tabulate


plt.rcParams['image.aspect'] = 'equal'
plt.rcParams['axes.labelsize'] = 'large'
#plt.rcParams['axes.grid'] = True


# Calculate a fragment of an orbit of x0 under the map f
#  with parameter c, N is the number of iterations
# Inputs:
#  x0 - initial point
#  f  - function to iterate
#  c  - parameter of function or None if no parameter present
#  N  - number of iterations
# Outputs:
#  gamma - a list of iterates
def orbit_of_map(x0, f, c, N):
    gamma = [x0]
    if type(c) == type(None):
        for _ in range(N):
            gamma.append(f(gamma[-1]))
    else:
        for _ in range(N):
            gamma.append(f(gamma[-1], c))
    return gamma

# Calculate the mean shifted distance of points
# The idea is that we define the mean k-shifted distance as
#  C(k) = 1/(n-k) sum || gamma[i] - gamma[i+k] ||
# The mean shifted distance is then the list of all C(k)
#  for all values of k.
# I think that the value of C(k) would be minimized at 
#  a value close to the real period
# Inputs:
#  X - a list of points
#  M - maximum k to consider
# Outputs:
#  MSD - the mean-shifted distance list, starting at k=1 
def mean_shifted_distance(X, M):
    MSD = []
    for k in range(1, min(M+1,len(X))):
        r = 0
        for i in range(len(X)-k):
            r += np.linalg.norm(X[i]-X[i+k])
        r /= len(X)-k
        MSD.append(r)
    return MSD

# Given an orbit converging to an attractor, figure
#  out a likely value for its period
# Inputs:
#  X, M - as in mean_shifted_distance
#  eps - sensitivity parameter, larger biases smaller periods
# Outputs:
#  p - guess for period
def period_of_attractor(gamma, M, eps = 1e-3):
    p = 1
    MSD = mean_shifted_distance(gamma, M)
    for i in range(len(MSD)):
        if MSD[i] + eps < MSD[p-1]:
            p = i+1
    return p

# Plot a cobweb plot
# Inputs:
#  x0, f, c, N - as in orbit_of_map
#  l  - left endpoint of plot
#  r  - right endpoint of plot
#  ax - axis to plot to
# Outputs:
#  None - plots to the axis
def cobweb_plot(x0, f, c, l, r, N, ax):
    # Plot f and the line y=x
    x = np.linspace(l, r)
    if type(c) != type(None):
        ax.plot(x, f(x, c), 'k', lw=2)
    else:
        ax.plot(x, f(x), 'k', lw=2)
    ax.plot(x, x, 'b', lw=2)

    # Recursively apply y=f(x) and plot two lines:
    # (x, x) -> (x, y)
    # (x, y) -> (y, y)
    gamma = orbit_of_map(x0, f, c, N)
    for i in range(N-1):
        x = gamma[i]
        y = gamma[i+1]
        # Plot the two lines.
        ax.plot([x, x], [x, y], 'r', lw=1)
        ax.plot([x, y], [y, y], 'r', lw=1)
        # Plot the positions with increasing opacity.
        ax.plot([x], [y], 'ok', ms=10, alpha=(i + 1) / N)
    if c != None:
        ax.set_title(f"$a={c:.3f}, \, x_0={x0:.3f}$")
    else:
        ax.set_title(f"$x_0={x0:.3f}$")

# Compute the directional derivative of f at x0
#  along the direction u with precision eps
# Inputs:
#  x0   - point about which to compute derivative
#  f, c - function and its parameter, optionally empty
#  u    - direction vector, must be a unit vector 
#  pointing in the direction were taking the derivative in
#  eps  - precision parameter
# Outputs:
#  Float - the derivative    
def directional_derivative(x0, f, c, u, eps = 1e-4):
    if type(c) == type(None):
        return (f(x0+eps*u)-f(x0-eps*u))/(2*eps)
    return (f(x0+eps*u, c)-f(x0-eps*u, c))/(2*eps)

# Compute the derivative of f at x0 with precision eps
# Inputs:
#  x0, f, c, eps - as in directional_derivative
# Outputs:
#  Float - the derivative    
def derivative(x0, f, c, eps = 1e-4):
    return directional_derivative(x0, f, c, 1, eps)

# Compute the Jacobian of f at x0 with precision eps
# Inputs:
#  x0, f, c, eps - as in directional_derivative
#  n - dimension of the space
# Outputs:
#  n x n array of floats - The Jacobian  
def jacobian(x0, f, c, n, eps = 1e-4):
    J = []
    for i in range(n):
        u = np.zeros(n)
        u[i] = 1
        J.append(directional_derivative(x0, f, c, u, eps))
    return np.array(J).T

# Compute the Lyapunov number of f at x_0  
# Inputs:
#  x0, f, c, eps - as in derivative
#  N - number of points in the orbit to consider
# Outputs:
#  Float - the Lyapunov number
def lyapunov_number(x0, f, c, N, eps = 1e-4):
    gamma = orbit_of_map(x0, f, c, N)
    L = 1
    for x in gamma:
        L *= np.power(np.abs(derivative(x, f, c, eps)), 1/N)
    return L

# Compute the Lyapunov exponent of f at x_0  
# Inputs:
#  x0, f, c, N, eps - as in lyapunov_number
# Outputs:
#  Float - the Lyapunov exponent
def lyapunov_exponent(x0, f, c, N, eps = 1e-4):
    gamma = orbit_of_map(x0, f, c, N)
    h = 0
    for x in gamma:
        d = np.abs(derivative(x, f, c, eps))
        h += np.log(d, where = d!=0)
        h = np.where(d==0, -np.inf, h)
    return h/N

# Plays the "Chaos game" on an IFS with given probability vector p
# Inputs:
#  x0  - initial value
#  IFS - list of functions representing and IFS
#  p   - list of probaiblities, the i-th of which
#  is the probability that IFS[i] is picked.
#  If p = None then a uniform prob is assumed.
#  N   - number of points in the orbit to consider
#  M   - number of initial points to discard 
# Outputs:
#  gamma - orbit of initial point under the chaos game of the IFS
def chaos_game(x0, IFS, p, N, M):
    def f(x):
        return IFS[np.random.choice(len(IFS), p = p)](x)
    gamma = orbit_of_map(x0, f, None, N)
    return gamma[M:]

# Compute the basin of attraction of the orbit given by the list X, 
#  or find the basin of infinity if X is empty. 
# To do this consider the n-hypercube whole bottom left and 
#  top right corners are l and r respectively.
# Subdivide each side of the hypercube into M parts creating
#  a grid of M**n smaller hypercubes, thne pick one 
#  representative point from each smaller hypercube.
# Iterate the point N times and then check if the value
#  is less than one grid-width away from some point of the 
#  given orbit X, or if its outside the cube if X is empty.
# Inputs:
#  X - a possibly empty list of points representing the attractor
#  f - function
#  c - function parameter or None if no parameters needed
#  l - left corner of the bounding hypercube
#  r - right corner of the bounding hypercube
#  n - dimension of the space
#  N - number of iterates to consider
#  M - number of grid cells along one edge, for M**n cells total
# Outputs:
#  is_attracted - a list of zeroes or ones depending on whether 
#   the point was attracted to X in which case get a 1 or not
#   in which case its a 0.
#  If X is empty, then this is flipped - 1 for all points that 
#   do not escape the hypercube and 0 otherwise.
def basin_of_attraction(X, f, c, l, r, n, N, M):
    # Compute grid cell representatives
    points = gen_cell_reps(l, r, n, M)
    
    is_attracted = []
    for x0 in points:
        y = x0
        for i in range(N):
            # If we are outside the bounding hypercube, run risk of overflows,
            #  so we end early (not likely we would return into the hypercube
            #  after having jumped out).
            if not(all(np.less_equal(l, y)) and all(np.less_equal(y, r))):
                break
            if c == None:
                y = f(y)
            else:
                y = f(y, c)
        if len(X) == 0:
            # Basin of infinity case, just check if outside bounding
            #  hypercube
            if all(np.less_equal(l, y)) and all(np.less_equal(y, r)):
                is_attracted.append(1)
            else:
                is_attracted.append(0)
        else:
            # Basin of attraction case, need to see if is close to any
            #  of the points in the orbit
            for x in X:
                if N*np.linalg.norm(x-y) < np.linalg.norm(l-r):
                    is_attracted.append(1)
                    break
            else:
                is_attracted.append(0)
    # Finally reshape the array so that we can readily plot it
    return np.array(is_attracted).reshape((M,)*n)

# Compute all the attractors and basins of attraction of a map. 
# To do this consider the n-hypercube whole bottom left and 
#  top right corners are l and r respectively.
# Subdivide each side of the hypercube into M parts creating
#  a grid of M**n smaller hypercubes, then pick one 
#  representative point from each smaller hypercube.
# Now consider which of the M**n smaller hypercubes
#  each representative is mapped to and record this
#  (record -1 if the representative is mapped outside the
#  bounding hypercube).
# We now have a directed graph that represents the discretized dynamics 
#  of the original map. As M becomes large this discretization represents
#  what is actually happening more and more closely.
# Then find the attractors in the discretized graph -
#  these are simply the strongly connected components (SSC),
#  and the structure of the digraph make them easy to find
# Afterwards just see which SSC does each cell eventually reach.
# Inputs:
#  f, c, l, r, n, M - as in basin_of_attraction
# Outputs:
#  is_attracted - 
#  attractors - 
#  frequency - 
def basins_of_attraction(f, c, l, r, n, M):
    # Compute grid cell representatives
    points = gen_cell_reps(l, r, n, M)
    
    # Find out which cell each representative goes to
    #  after a single iteration
    maps_to = []
    for x in points:
        if c == None:
            y = f(x)
        else:
            y = f(x, c)
        maps_to.append(point_to_cell(y, l, r, n, M))
    
    attractors = []
    is_attracted = [-1 for i in range(len(points))]
    count = 1
    seen = set()
    seen.add(-1)
    for i in range(len(points)):
        point = i
        if point not in seen:
            # Go until we hit a familiar point
            while point not in seen:
                seen.add(point)
                point = maps_to[point]
            
            start = point
            # If have not assigned a color then we have a new cycle
            if point != -1 and is_attracted[point] == -1:
                # attractors[i][0] = how many cells are attracted to 
                #  this attractor
                # attractors[i][1] = what is its color
                # attractors[i][2] = what are its points
                attractors.append([0, count, [points[point]]])
                is_attracted[point] = count
                point = maps_to[point]
                count += 1
                # Iterate around the cycle coloring and recording the points
                while point != start:
                    is_attracted[point] = is_attracted[start]
                    attractors[-1][2].append(points[point])
                    point = maps_to[point]
            
            # Now mark all the points mapping into the cycle
            point = i
            while point != start:
                if start == -1:
                    is_attracted[point] = 0
                else:
                    is_attracted[point] = is_attracted[start]
                point = maps_to[point]
    # Reorder the attractors by color
    attractors.sort()
    # Count how many points hit each attractor
    for i in range(len(points)):
        if is_attracted[i] != 0:
            attractors[is_attracted[i]-1][0] += 1
    # Sort by popularity
    attractors.sort()
    # Change colors according to popularity, so that 
    #  less popular = smaller color
    remap = {0:0}
    for i in range(len(attractors)):
        remap[attractors[i][1]] = i+1
        attractors[i] = [attractors[i][0], attractors[i][2]]
    for i in range(len(points)):
        is_attracted[i] = remap[is_attracted[i]]
    # Separate frequency from the attractors themselves
    frequency, attractors = zip(*attractors)
    return np.array(is_attracted).reshape((M,)*n), attractors, frequency




def basins_of_attraction2(f, c, l, r, n, N, M):
    # Compute grid cell representatives
    points = gen_cell_reps(l, r, n, M)
    
    # Find out which cell each representative goes to
    #  after N iterations
    maps_to = []
    maps_to_val = []
    for x in points:
        y = x
        for i in range(N):
             # If we are outside the bounding hypercube, run risk of overflows,
            #  so we end early (not likely we would return into the hypercube
            #  after having jumped out).
            if not(all(np.less_equal(l, y)) and all(np.less_equal(y, r))):
                y = 2*r
                break
            if c == None:
                y = f(y)
            else:
                y = f(y, c)
        maps_to.append(point_to_cell(y, l, r, n, M))
        maps_to_val.append(y)
    
    apoints = []
    cell_to_ap = [-1 for i in range(len(points))]
    for i in range(len(points)):
        if maps_to[i] != -1:
            if  cell_to_ap[maps_to[i]] == -1:
                cell_to_ap[maps_to[i]] = len(apoints)
                apoints.append([])
            apoints[cell_to_ap[maps_to[i]]].append(maps_to_val[i])
    
    for i in range(len(apoints)):
        apoints[i] = sum(apoints[i])/len(apoints[i])
        
    ap_maps_to = []
    for x in apoints:
        if c == None:
            y = f(x)
        else:
            y = f(x, c)
        ap_maps_to.append(cell_to_ap[point_to_cell(y, l, r, n, M)])
        
    
    attractors = []
    is_attracted = [-1 for i in range(len(apoints))]
    count = 1
    seen = set()
    seen.add(-1)
    for i in range(len(apoints)):
        point = i
        if point not in seen:
            # Go until we hit a familiar point
            while point not in seen:
                seen.add(point)
                point = ap_maps_to[point]
            
            start = point
            # If have not assigned a color then we have a new cycle
            if point != -1 and is_attracted[point] == -1:
                # attractors[i][0] = how many cells are attracted to 
                #  this attractor
                # attractors[i][1] = what is its color
                # attractors[i][2] = what are its points
                attractors.append([0, count, [apoints[point]]])
                is_attracted[point] = count
                point = ap_maps_to[point]
                count += 1
                # Iterate around the cycle coloring and recording the points
                while point != start:
                    is_attracted[point] = is_attracted[start]
                    attractors[-1][2].append(apoints[point])
                    point = ap_maps_to[point]
            
            # Now mark all the points mapping into the cycle
            point = i
            while point != start:
                if start == -1:
                    is_attracted[point] = 0
                else:
                    is_attracted[point] = is_attracted[start]
                point = ap_maps_to[point]

    # Reorder the attractors by color
    attractors.sort()
    
    # Count how many points hit each attractor
    for i in range(len(points)):
        if cell_to_ap[maps_to[i]] != -1 and is_attracted[cell_to_ap[maps_to[i]]] != 0:
            attractors[is_attracted[cell_to_ap[maps_to[i]]]-1][0] += 1
    # Sort by popularity
    attractors.sort()
    # Change colors according to popularity, so that 
    #  less popular = smaller color
    remap = {0:0}
    for i in range(len(attractors)):
        remap[attractors[i][1]] = i+1
        attractors[i] = [attractors[i][0], attractors[i][2]]
    for i in range(len(apoints)):
        is_attracted[i] = remap[is_attracted[i]]
    point_is_attracted = []
    for i in range(len(points)):
        if cell_to_ap[maps_to[i]] != -1:
            point_is_attracted.append(is_attracted[cell_to_ap[maps_to[i]]])
        else:
            point_is_attracted.append(0)
    # Separate frequency from the attractors themselves
    frequency, attractors = zip(*attractors)
    return np.array(point_is_attracted).reshape((M,)*n), attractors, frequency

def gen_cell_reps(l, r, n, M):
    points = []
    for p in product(range(M), repeat = n):
        points.append(l + (np.array(tuple(reversed(p)))+0.5)*(r-l)/(M-1))
    return points

def point_to_cell(x, l, r, n, M):
    x = ((((x-l)/(r-l))*(M-1))-0.5).astype(int)
    j = 0
    for k in reversed(x):
        if k < 0 or k > M-1:
            j = -1
            break
        j *= M
        j += k
    return j

def bifurcation_diagram(f, l, r, n, N, M):
    pass


def ode23(t0, t1, x0, f, c, tol = 1e-3):
    T = np.array([t0])
    X = np.array([x0])
    eps = tol
    t = t0
    x = x0
    while t < t1:
        t = t + eps
        T = np.append(T, t)
        s1 = f(t, x, c)
        s2 = f(t+eps, x+eps*s1, c)
        s3 = f(t+eps/2, x+eps*(s1+s2)/4, c)
        x =  x + eps*(s1+s2+4*s3)/6
        X = np.append(X, [x], axis = 0)
        tt = eps*np.linalg.norm((2*s3-s1-s2)/3)
        if tt!= 0:
            eps *= 0.9 * (tol/tt)**(1/3)
    return T, X