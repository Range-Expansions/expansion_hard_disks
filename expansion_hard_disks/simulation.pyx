#cython: profile=False
#cython: boundscheck=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import pylab as pl
from pylab import *
import numpy as np
cimport numpy as np
cimport cython
import random
from libc.math cimport atan2, sin, cos, sqrt, abs, floor
import matplotlib.pyplot as plt
from matplotlib import colors
# from interval import interval, inf, imath
import sys

cdef float pi=np.pi

# initial condition
cdef grates=np.array([1,1.2]);  # growth rates vector --- can be changed in time
cdef float time=0;  # real time in units of 1/r1 (r1 is initially set to 1)
cdef float t_switch=4;  # time at which the environment switches

cdef float R=100   # initial radius of the homeland (in units of cell radius)
cdef float fraction=0.25   # initial fraction of occupied space in the initial disk or ring
cdef float ratio=0.95    # initial ratio of species 1 vs species 2 abundances

cdef int switch=0   # equal to 1 after the environment has switched
cdef int n1=0;  # tot no. cells of species 1
cdef int n2=0;  # tot no. cells of species 1

cdef int origin=np.random.choice([0,1],1,p=[ratio,1-ratio]) # assigns the species to the cell at the center
cdef positions=np.array([[0,0,origin]]) # vector of coordinates and species identity. Each item is [x_coordinate,y_coordinate,species_indentity]
cdef frontier=np.array([])  # vector of coordinates, species identity and growth rate at the frontier. Only cells that can bud are in this vector. Each item is [x_coordinate,y_coordinate,species_indentity,species_growth_rate]

# sets the initial condition to be a disk filled uniformly. Initializes the positions and frontier vectors
def initialize_disk(float fraction):
    global positions
    global frontier
    positions=np.array([[0,0,origin]])  # re-initializes positions vector
    frontier=np.array([])   # re-initializes frontier vector

    # This section initializes a circular homeland
    # pi N = pi R^2/fraction
    cdef float r, theta, x, y
    cdef int i, q, t
    for i in range(int((R**2)*fraction)):
        t=0
        while t==0:
            # r is the radius at which the cell is placed and theta is the angle
            r=R*sqrt(random.uniform(0,1))   # in this way points are distributed uniformly in the homeland
            theta=random.uniform(0,2*pi)
            x=r*cos(theta)
            y=r*sin(theta)
            t=1
            # checks that the new cell at x,y does not overlap with other ones
            for q in range(len(positions)):
                if (positions[q][0]-x)**2+(positions[q][1]-y)**2<=4:
                    t=0
        species=np.random.choice([0,1],1,p=[ratio,1-ratio]) # assigns species identity
        positions=np.append(positions,[[x,y,species]],axis=0)   # updates positions vector

    # initializes frontier
    for pos in positions:
        if is_at_frontier(pos)[1]:
            if frontier.size==0:
                frontier=np.append(pos,grates[int(pos[2])])
            else:
                frontier=np.vstack((frontier,np.append(pos,grates[int(pos[2])])))

# sets the initial condition to be a ring filled uniformly, with a few cells in the interior. Initializes the positions and frontier vectors
def initialize_ring(float fraction):
    global positions
    global frontier
    positions=np.array([[0,0,origin]])
    frontier=np.array([])

    # This section initializes a ring homeland with a few scattered cells in the interior
    # pi N = pi (R^2-(0.8*R)^2)/fraction
    cdef float r, theta, x, y
    cdef int i, q, t
    for i in range(int((R**2-(0.9*R)**2)*fraction)):
        t=0
        while t==0:
            # r is the radius at which the cell is placed and theta is the angle
            r=0.9*R+0.1*R*sqrt(random.uniform(0,1))
            theta=random.uniform(0,2*pi)
            x=r*cos(theta)
            y=r*sin(theta)
            t=1
            # checks that the new cell at x,y does not overlap with other ones
            for q in range(len(positions)):
                if (positions[q][0]-x)**2+(positions[q][1]-y)**2<=4:
                    t=0
        species=np.random.choice([0,1],1,p=[ratio,1-ratio]) # assigns species identity
        positions=np.append(positions,[[x,y,species]],axis=0)   # updates positions vector
    # removes cell at the origin
    positions=np.delete(positions,0,0)
    # populates interior of the ring with a sixth of the ring density
    for i in range(int((0.9*R)**2*fraction/6)):
        t=0
        while t==0:
            # r is the radius at which the cell is placed and theta is the angle
            r=0.9*R*sqrt(random.uniform(0,1))
            theta=random.uniform(0,2*pi)
            x=r*cos(theta)
            y=r*sin(theta)
            t=1
            # checks that the new cell at x,y does not overlap with other ones
            for q in range(len(positions)):
                if (positions[q][0]-x)**2+(positions[q][1]-y)**2<=4:
                    t=0
        species=np.random.choice([0,1],1,p=[ratio,1-ratio]) # assigns species identity
        positions=np.append(positions,[[x,y,species]],axis=0)   # updates positions vector

    # initializes frontier
    for pos in positions:
        if is_at_frontier(pos)[1]:
            if frontier.size==0:
                frontier=np.append(pos,grates[int(pos[2])])
            else:
                frontier=np.vstack((frontier,np.append(pos,grates[int(pos[2])])))

# sign function
cdef int sign(float x) nogil:
    if x > 0:
        return 1
    else:
        return -1

# returns the intersection of the two real intervals set1 and set2, which may be each composed by more than one segment, e.g. set1=[[0,1],[2,3]] has two segments
def intersection(list set1, list set2):
    cdef int n1=len(set1)
    cdef int n2=len(set2)
    cdef list setI=[]

    cdef int k
    for k in range(n2):
        for i in range(n1):
            if single_intersection(set1[i],set2[k]):
                setI.append([max(set1[i][0],set2[k][0]),min(set1[i][1],set2[k][1])])

    return setI

# this function returns the intersection of two 1-component intervals
# recipe from https://fgiesen.wordpress.com/2011/10/16/checking-for-interval-overlap/
def single_intersection(list set1,list set2):
    return set1[0]<set2[1] and set2[0]<set1[1]

# returns the number of non-overlapping components of the interval
def components(list interval):
    return len(interval)

# returns the union of the two real intervals set1 and set2, which may be each composed by more than one segment, e.g. set1=[[0,1],[2,3]] has two segments
# algorithm in http://www.geeksforgeeks.org/merging-intervals/
def union(list set1, list set2):
    cdef int n1=len(set1)
    cdef int n2=len(set2)
    cdef list setUL=set1[:]
    cdef int i
    for i in range(len(set2)):
        setUL.append(set2[i])
    setUL=sorted(setUL) # order intervals according to initial point
    setU=[setUL[0]]
    cdef int l=0;
    cdef int k
    for k in range(n1+n2-1):
        if not single_intersection(setUL[k],setUL[k+1]):
            l += 1
            setU.append(setUL[k+1])
        else:
            if setUL[k+1][1]>setUL[k][1]:
                setU[l][1]=setUL[k+1][1]
    return setUL

# return the total length of the interval
cpdef length(list interval):
    cdef float length=0
    cdef int i
    for i in range(components(interval)):
        length += interval[i][1]-interval[i][0]
    return length

# checks if the cell in cur_loc has space to divide
# input: coordinates
# output: available angles where the cell can bud and a bool True - False that says if the cell is at the frontier
cpdef is_at_frontier(cur_loc):
    # Find cells that can impede growth (those within distance 4)
    closeby_cells=(positions[:,0]-cur_loc[0])**2+(positions[:,1]-cur_loc[1])**2<4**2
    closeby_positions=positions[closeby_cells]  # x and y coordinates of the cells within distance 4
    # Remove cur_loc from these arrays
    closeby_cells=(closeby_positions[:,0]-cur_loc[0])**2+(closeby_positions[:,1]-cur_loc[1])**2>0
    closeby_positions=closeby_positions[closeby_cells]

    cdef list sets=[[0, 2*pi]]  # set of available angles at which to place the bud

    cdef float x0=cur_loc[0]
    cdef float y0=cur_loc[1]
    cdef float x1
    cdef float y1
    cdef float theta1
    cdef float theta2
    cdef int i
    # the centre of daughter cells can only be placed along the circle of radius 2 centered in pos
    # checks if there are locations along the circle of radius 2 centered in pos that are at least at distance 2 from all other cells

    for i,pos in enumerate(closeby_positions):
        # for each cell within distance 4, find which angles it blocks
        x1=pos[0]
        y1=pos[1]
        if y0!=y1:
            if x0==x1:
                theta1=atan2((-y0+y1),(-sqrt(-(-4+y0-y1)*(4+y0-y1))))
                theta2=atan2((-y0+y1),(sqrt(-(-4+y0-y1)*(4+y0-y1))))
            else:
                theta1=atan2((-(y0-y1)*((x0-x1)**2+(y0-y1)**2)+(x0-x1)*sqrt(-(-16+(x0-x1)**2+(y0-y1)**2)*((x0-x1)**2+(y0-y1)**2))*((y0-y1)/abs(y0-y1))),(-(x0-x1)*((x0-x1)**2+(y0-y1)**2)-sqrt(-(-16+(x0-x1)**2+(y0-y1)**2)*((x0-x1)**2+(y0-y1)**2))*abs(y0-y1)))
                theta2=atan2((-(y0-y1)*((x0-x1)**2+(y0-y1)**2)-(x0-x1)*sqrt(-(-16+(x0-x1)**2+(y0-y1)**2)*((x0-x1)**2+(y0-y1)**2))*((y0-y1)/abs(y0-y1))),(-(x0-x1)*((x0-x1)**2+(y0-y1)**2)+sqrt(-(-16+(x0-x1)**2+(y0-y1)**2)*((x0-x1)**2+(y0-y1)**2))*abs(y0-y1)))
        else:
            theta1=atan2((-sqrt(-(-4+x0-x1)*(4+x0-x1))),(-x0+x1))
            theta2=atan2((sqrt(-(-4+x0-x1)*(4+x0-x1))),(-x0+x1))
        if theta1<0:
            theta1 += 2*pi
        if theta2<0:
            theta2 += 2*pi

        # the above lines return the two solutions of:
        # [x0+2 cos(theta)-x1]^2+[y0+2 sin(theta)]^2=2^2
        # which identifies the two angles within which the cell cannot bud
        # the following line are used to find whether the blocked angles are those between theta1 and theta,
        # or the intervals [0,min(theta1,theta2)] and [max(theta1,theta2),2pi]
        if max(theta1,theta2)-min(theta1,theta2)>=2*pi-(max(theta1,theta2)-min(theta1,theta2)):
            sets=intersection([[min(theta1,theta2),max(theta1,theta2)]],sets)
        else:
            sets=intersection(sets,union([[0,min(theta1,theta2)]],[[max(theta1,theta2),2*pi]]))

    return sets, components(sets)>0

# checks if there is empty space in position new_loc where a new cell was placed
# a new cell can only bother other cells within a distance of 4
# this function updates the vectors positions and frontier
cpdef update_frontier(new_loc):
    global frontier
    global positions
    positions=np.append(positions,[new_loc],axis=0)

    # if the new cell is at the frontier, add it to the frontier vector
    if is_at_frontier(new_loc)[1]:
        frontier=np.vstack((frontier,np.append(new_loc,grates[int(new_loc[2])])))

    # the following lines are used to update the frontier vector, which might have changed due to the new cell
    # Find cells that can impede growth (those within distance 4)
    closeby_cells=(positions[:,0]-new_loc[0])**2+(positions[:,1]-new_loc[1])**2<=4**2
    closeby_positions=positions[closeby_cells]
    # Remove new_loc from the array
    closeby_cells=(closeby_positions[:,0]-new_loc[0])**2+(closeby_positions[:,1]-new_loc[1])**2>0
    closeby_positions=closeby_positions[closeby_cells]

    in_frontier=[]
    for pos in closeby_positions:
        # check if there are locations along the circle of radius 2 centered in pos that are at least at distance 2 from all other cells
        in_frontier.append(is_at_frontier(pos)[1])
    # these are the cells that are not in the frontier anymore
    remove=where(np.logical_not(in_frontier))
    if size(remove)>0:
        for i in range(size(remove)):
            frontier=np.delete(frontier,(np.where((frontier==np.append(closeby_positions[remove[0][i]],grates[int(closeby_positions[remove[0][i]][2])])).all(axis=1))),axis=0)

# these functions can be useful for debugging
cpdef print_variables():
        print time
        print grates[0]
        print grates[1]
        print('-----')

cpdef print_frontier():
        print frontier

cpdef print_positions():
        print positions

# this function switches the growth rates and updates the frontier array accordingly
cpdef switch_grates():
    global switch, grates, frontier, positions
    cdef int q
    cdef float temp
    temp=grates[0]
    grates[0]=grates[1]
    grates[1]=temp
    switch=1
    for q in range(shape(frontier)[0]):
        frontier[q][3]=grates[int(frontier[q][2])]
    print 'Switched growth rates'

# Performs one growth step, i.e. one cell division
cpdef growth():
    global time
    # if time>t_switch and switch==0:
    #     switch_grates()

    n1=len(positions[:,2]==0)   # updates tot number of cells of species 1
    n2=len(positions)-n1   # updates tot number of cells of species 2

    # updates the real time
    time += log(1/random.uniform(0,1))/(n1*grates[0]+n2*grates[1])

    # selects the parent cell at the frontier according to the different growth rates
    parent=frontier[np.random.choice(range(shape(frontier)[0]),1,p=frontier[:,3]/sum(frontier[:,3]))][0]
    availableTheta=is_at_frontier(parent)[0]    # this list includes the angles at which the cell can bud

    cdef float thetaL=random.uniform(0,length(availableTheta))  # random number between 0 and the total length of available intervals
    cdef float lower_bound=0;   # minimum angle at which the daughter cell can be placed
    cdef float theta=0; # angle at which the daughter cell is placed

    # the following lines find the angle at which the daughter cell is placed
    cdef int i
    for i in range(components(availableTheta)):
        lower_bound = availableTheta[i][0]
        if thetaL+lower_bound<=availableTheta[i][1]:
            theta=thetaL+lower_bound
            break
        else:
            thetaL=thetaL-length([availableTheta[i]])

    # this list gives the coordinates and the species identity of the new cell
    new_loc=[parent[0]+2*cos(theta),parent[1]+2*sin(theta),parent[2]]

    # updates the frontier
    update_frontier(new_loc)

# this function is called to perform T cell divisions
cpdef prolonged_growth(T):
    cdef int i
    for i in range(1,T):
        growth()
        if mod(i+1,floor(T/10))==0:
            print 'Completion: ',100.*(i+1)/T,'% - Time: ',time # displays percentage of completion and real time

# displays the colony. If the colony is large, it uses a coarse_grained representation of the colony
cpdef display_colony():
    global time
    # print time
    if shape(positions)[0]<50000:
        display_colony_circles()
    else:
        display_coarse_grained_colony(4)

# Coarse grained representation of the colony, each pixel consists of 16 cells. Note: it does not average across the pixel, but only returns
# the color of the last cell found in that pixel
cpdef display_coarse_grained_colony(int resolution):
    coordinates=concatenate((positions[:,1],positions[:,0]))
    px=int((max([abs(min(coordinates)),abs(max(coordinates))])+1)/resolution)
    lx=min(positions[:,0])
    ly=min(positions[:,1])
    img=np.zeros((2*px+1,2*px+1))-1

    cdef int i
    for i in range(shape(positions)[0]):
        px1=(2+positions[i][0]-lx)/resolution
        px2=(2+positions[i][1]-ly)/resolution
        img[int(floor(px1)),int(floor(px2))] = positions[i][2]

    cmap = colors.ListedColormap(['white','blue','red'])
    bounds=[-1.5,-0.5,0.5,1.5]

    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(img, interpolation='none', origin='lower', cmap=cmap, norm=norm)

# Displays colony by representing cells as circles
cpdef display_colony_circles():
    figure(figsize=(8,8))
    ax=subplot(aspect='equal')
    cdef int i
    for i in range(shape(positions)[0]):
        if positions[i][2]==0:
            ax.add_artist(plt.Circle([positions[i][1],positions[i][0]], 1, alpha=0.5, color='blue'))
        else:
            # ax.add_artist(plt.Circle(positions[i][0:2], 1, alpha=0.5, color='red'))
            ax.add_artist(plt.Circle([positions[i][1],positions[i][0]], 1, alpha=0.5, color='red'))
    coordinates=concatenate((positions[:,1],positions[:,0]))
    xlim(min(coordinates)-2,max(coordinates)+2)
    ylim(min(coordinates)-2,max(coordinates)+2)