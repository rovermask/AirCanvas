import math

def distanceBetweenPoints(pt1,pt2):
    x_sq=(pt2[0]-pt1[0])**2
    y_sq=(pt2[1]-pt1[1])**2
    dist=math.sqrt(x_sq+y_sq)
    return dist