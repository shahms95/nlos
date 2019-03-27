import math

import numpy as np

from nlos.conf import *


class Point:
    def __init__(self, x=0, y=0):
        self.coords = np.array([x, y])

    def set(self, a, b):
        self.coords = np.array([a, b])

    def __sub__(self, otherPoint):
        x = self.coords[0] - otherPoint.coords[0]
        y = self.coords[1] - otherPoint.coords[1]
        p = Point()
        p.set(x, y)
        return p

    def __add__(self, otherPoint):
        x = self.coords[0] + otherPoint.coords[0]
        y = self.coords[1] + otherPoint.coords[1]
        p = Point()
        p.set(x, y)
        return p

    def getCoords(self):
        return self.coords

    def setCoords(self, coords):
        self.coords = coords

    def printPoint(self):
        print(self.coords[0], self.coords[1])


class Segment:
    def __init__(self):
        self.start = Point()
        self.end = Point()

    def setStart(self, a, b):
        self.start.set(a, b)

    def setEnd(self, a, b):
        self.end.set(a, b)

    def setStartPoint(self, a):
        self.start = a

    def setEndPoint(self, a):
        self.end = a

    def getStart(self):
        return self.start

    def getEnd(self):
        return self.end

    def printSegment(self):
        print("Start : ")
        self.getStart().printPoint()
        print("End : ")
        self.getEnd().printPoint()


class Surface:
    def __init__(self, n=1):
        self.components = [Segment() for _ in range(n)]

    def addComponent(self, a):
        self.components.append(a)

    def getComponents(self, n=None):
        if n == None:
            return self.components
        else:
            return self.components[n]

    def createSurface(self, endpoints):
        l = len(endpoints) - 1
        self.components = [Segment() for _ in range(l)]
        for i in range(l):
            start = endpoints[i]
            end = endpoints[i + 1]
            # print(start, end)
            self.getComponents(i).setStart(start[0], start[1])
            self.getComponents(i).setEnd(end[0], end[1])
            self.getComponents(i).printSegment()
            objct = self.getComponents()
            # print("iteration : ", i)
            # print("Length of object ", len(objct))
            # for oo in objct:
            #     print("Address : ", hex(id(objct[i])))
            #     oo.printSegment()

    def printSurface(self):
        print("Start printing surface")
        for c in self.components:
            c.printSegment()
        print("End printing surface")


def distance(a, b):
    return np.sqrt(np.sum(np.square((a - b).getCoords())))


def movePoint(point, delta, vec):
    coords = vec.getCoords()
    coords = delta * coords
    p = Point()
    p.setCoords(coords)
    return point + p


def moveSegment(obj, speed):
    obj.setStartPoint(obj.getStart() + speed)
    obj.setEndPoint(obj.getEnd() + speed)


def moveSurface(obj, speed):
    # print("Before moving")
    # obj.printSurface()
    components = obj.getComponents()
    for o in components:
        moveSegment(o, speed)
    # print("After moving")
    # obj.printSurface()


def cosine(p1, p2, p3):
    x1, y1 = p1.getCoords()
    x2, y2 = p2.getCoords()
    x3, y3 = p3.getCoords()
    u = (x1 - x2, y1 - y2)
    v = (x3 - x2, y3 - y2)
    norm_u = math.sqrt(u[0] * u[0] + u[1] * u[1])
    norm_v = math.sqrt(v[0] * v[0] + v[1] * v[1])

    # this conditional is to check there has been movement between the points
    if norm_u < EPS or norm_v < EPS:
        return None
    prod_n = norm_u * norm_v
    dot_uv = u[0] * v[0] + u[1] * v[1]
    cos_uv = dot_uv / prod_n

    # fixes floating point rounding
    if cos_uv > 1.0 or cos_uv < -1.0:
        cos_uv = round(cos_uv)
    return cos_uv


def coords_from_indices(x_ind, y_ind):
    x_c = bb[0][0] + x_ind * (1 / BOX_PER_UNIT) + (0.5) / BOX_PER_UNIT
    y_c = bb[0][1] + y_ind * (1 / BOX_PER_UNIT) + (0.5) / BOX_PER_UNIT
    return x_c, y_c


# init point coordinates of wallpoints being observed
wallPoints = []

while x <= END_X:
    p = Point()
    p.set(x, FIXED_Y)
    wallPoints.append(p)
    x = x + DELTA_X

bb = [np.zeros((1, 2)) for _ in range(4)]
bb[0] = np.asarray([bb0x, bb0y])
bb[1] = np.asarray([bb1x, bb1y])
bb[2] = np.asarray([bb2x, bb2y])
bb[3] = np.asarray([bb3x, bb3y])

bb_xnum = (bb[1][0] - bb[0][0]) * BOX_PER_UNIT
bb_ynum = (bb[3][1] - bb[0][1]) * BOX_PER_UNIT

cx = np.zeros((bb_xnum))
cy = np.zeros((bb_ynum))

laser = Point()
camera = Point()

for vxi in range(bb_xnum):
    cx[vxi], _ = coords_from_indices(vxi, 0)
for vyi in range(bb_ynum):
    _, cy[vyi] = coords_from_indices(0, vyi)
