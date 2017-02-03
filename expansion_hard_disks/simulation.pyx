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
cdef float r1=1;
cdef float r2=1.25;
cdef float time=0;
cdef int switch=0
cdef int n1=0;
cdef int n2=0;

cdef float R=100   # initial radius of the homeland
cdef float fraction=0.25   # initial fraction of occupied space

cdef float origin=choice([r1,r2])
cdef positions=np.array([[0,0,origin]])
cdef frontier=np.array([])

def initialize_disk(float fraction):
    global positions
    global frontier

    # This section initializes a circular homeland
    # pi N = pi R^2/fraction
    cdef float r, theta, x, y
    cdef int i, q, t
    for i in range(int((R**2)*fraction)):
        t=0
        while t==0:
            r=R*sqrt(random.uniform(0,1))   # in this way points are distributed uniformly in the homeland
            theta=random.uniform(0,2*pi)
            x=r*cos(theta)
            y=r*sin(theta)
            t=1
            for q in range(len(positions)):
                if (positions[q][0]-x)**2+(positions[q][1]-y)**2<=4:
                    t=0
        positions=np.append(positions,[[x,y,choice([r1,r2])]],axis=0)

    for pos in positions:
        if is_at_frontier(pos)[1]:
            if frontier.size==0:
                frontier=pos
            else:
                frontier=np.vstack((frontier,pos))

def initialize_ring(float fraction):
    global positions
    global frontier

    # This section initializes a ring homeland with a few scattered cells in the interior
    # pi N = pi (R^2-(0.8*R)^2)/fraction
    cdef float r, theta, x, y
    cdef int i, q, t
    for i in range(int((R**2-(0.9*R)**2)*fraction)):
        t=0
        while t==0:
            r=0.9*R+0.1*R*sqrt(random.uniform(0,1))
            theta=random.uniform(0,2*pi)
            x=r*cos(theta)
            y=r*sin(theta)
            t=1
            for q in range(len(positions)):
                if (positions[q][0]-x)**2+(positions[q][1]-y)**2<=4:
                    t=0
        positions=np.append(positions,[[x,y,choice([r1,r2])]],axis=0)
    # remove cell in the origin
    positions=np.delete(positions,0,0)
    # populates interior of the ring with a sixth of the ring density
    for i in range(int((0.9*R)**2*fraction/6)):
        t=0
        while t==0:
            r=0.9*R*sqrt(random.uniform(0,1))
            theta=random.uniform(0,2*pi)
            x=r*cos(theta)
            y=r*sin(theta)
            t=1
            for q in range(len(positions)):
                if (positions[q][0]-x)**2+(positions[q][1]-y)**2<=4:
                    t=0
        positions=np.append(positions,[[x,y,choice([r1,r2])]],axis=0)

    for pos in positions:
        if is_at_frontier(pos)[1]:
            if frontier.size==0:
                frontier=pos
            else:
                frontier=np.vstack((frontier,pos))

cdef int sign(float x) nogil:
    if x > 0:
        return 1
    else:
        return -1

# note: set2 in our implementation is at most composed of two segments
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

cpdef length(list interval):
    cdef float length=0
    cdef int i
    for i in range(components(interval)):
        length += interval[i][1]-interval[i][0]
    return length

# checks if cell in cur_loc has space to divide
# input: coordinates
# output: bool True - False
cpdef is_at_frontier(cur_loc):
    # Find cells that can impede growth (those within distance 4)
    closeby_cells=(positions[:,0]-cur_loc[0])**2+(positions[:,1]-cur_loc[1])**2<4**2
    closeby_positions=positions[closeby_cells]
    # Remove cur_loc from the array
    closeby_cells=(closeby_positions[:,0]-cur_loc[0])**2+(closeby_positions[:,1]-cur_loc[1])**2>0
    closeby_positions=closeby_positions[closeby_cells]

    cdef list sets=[[0, 2*pi]]

    cdef float x0=cur_loc[0]
    cdef float y0=cur_loc[1]
    cdef float x1
    cdef float y1
    cdef float theta1
    cdef float theta2
    cdef int i
    for i,pos in enumerate(closeby_positions):
        # check if there are locations along the circle or radius 2 centered in pos that are at least at distance 2 from all other cells
        #sets.append(solve_univariate_inequality((cur_loc[0]+2*cos(x)-pos[0])**2+(cur_loc[1]+2*sin(x)-pos[1])**2>4,x,False))
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

        # decides internal or external interval
        if max(theta1,theta2)-min(theta1,theta2)>=2*pi-(max(theta1,theta2)-min(theta1,theta2)):
            sets=intersection([[min(theta1,theta2),max(theta1,theta2)]],sets)
        else:
            sets=intersection(sets,union([[0,min(theta1,theta2)]],[[max(theta1,theta2),2*pi]]))

    return sets, components(sets)>0

# checks if there is empty space in position new_loc where a new cell was placed
# a new cell can only bother other cells within a distance of 4
cpdef update_frontier(new_loc):
    global frontier
    global positions
    positions=np.append(positions,[new_loc],axis=0)

    if is_at_frontier(new_loc)[1]:
        frontier=np.vstack((frontier,new_loc))
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
            frontier=np.delete(frontier,(np.where((frontier==closeby_positions[remove[0][i]]).all(axis=1))),axis=0)

cpdef disp_variables():
        print time
        print r1
        print r2
        print('-----')

# growth step
cpdef growth():
    global time, switch, r1, r2
    if time>3 and switch==0:
        temp=r1
        r1=1.5
        r2=1
        switch=1
        print 'switched'
    # pick a random cell at the frontier
#     parent=random.sample(frontier,1)[0]
    n1=len(positions[:,2]==r1)
    n2=len(positions)-n1
    time += log(1/random.uniform(0,1))/(r1*n1+r2*n2)
    parent=frontier[np.random.choice(range(shape(frontier)[0]),1,p=frontier[:,2]/sum(frontier[:,2]))][0]
    availableTheta=is_at_frontier(parent)[0]
    cdef float thetaL=random.uniform(0,length(availableTheta))

    cdef float lower_bound=0;
    cdef float theta=0;

    cdef int i
    for i in range(components(availableTheta)):
        lower_bound = availableTheta[i][0]
        if thetaL+lower_bound<=availableTheta[i][1]:
            theta=thetaL+lower_bound
            break
        else:
            thetaL=thetaL-length([availableTheta[i]])

    new_loc=[parent[0]+2*cos(theta),parent[1]+2*sin(theta),parent[2]]

    update_frontier(new_loc)

cpdef prolonged_growth(T):
    cdef int i
    for i in range(1,T):
        growth()
        if mod(i+1,floor(T/10))==0:
            print 'Completion: ',100.*(i+1)/T,'% - Time: ',time
            # disp_variables()

cpdef display_colony():
    global time
    # print time
    if shape(positions)[0]<20000:
        display_colony_circles()
    else:
        display_coarse_grained_colony(4)

cpdef display_coarse_grained_colony(int resolution):
    coordinates=concatenate((positions[:,1],positions[:,0]))
    px=int((max([abs(min(coordinates)),abs(max(coordinates))])+1)/resolution)
    lx=min(positions[:,0])
    ly=min(positions[:,1])
    img=np.zeros((2*px+1,2*px+1))

    cdef int i
    for i in range(shape(positions)[0]):
        px1=(2+positions[i][0]-lx)/resolution
        px2=(2+positions[i][1]-ly)/resolution
        img[int(floor(px1)),int(floor(px2))] = positions[i][2]

    if r1<r2:
        cmap = colors.ListedColormap(['white','blue','red'])
        bounds=[0,r1,r2,r2+1]
    else:
        cmap = colors.ListedColormap(['white','blue','red'])
        bounds=[0,r2,r1,r1+1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.imshow(img, interpolation='none', origin='lower', cmap=cmap, norm=norm)

cpdef display_colony_circles():
    figure(figsize=(8,8))
    ax=subplot(aspect='equal')
    cdef int i
    if r1<r2:
        for i in range(shape(positions)[0]):
            if positions[i][2]==r1:
                ax.add_artist(plt.Circle(positions[i][0:2], 1, alpha=0.5, color='blue'))
            else:
                ax.add_artist(plt.Circle(positions[i][0:2], 1, alpha=0.5, color='red'))
    else:
        for i in range(shape(positions)[0]):
            if positions[i][2]==r1:
                ax.add_artist(plt.Circle(positions[i][0:2], 1, alpha=0.5, color='red'))
            else:
                ax.add_artist(plt.Circle(positions[i][0:2], 1, alpha=0.5, color='blue'))
    coordinates=concatenate((positions[:,1],positions[:,0]))
    xlim(min(coordinates)-2,max(coordinates)+2)
    ylim(min(coordinates)-2,max(coordinates)+2)