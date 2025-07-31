#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 12:33:23 2024

@author: glavrent
"""

import numpy as np

def rectangle_to_point_distance(rect_corner, rect_edge1, rect_edge2, point):
    """
    Compute the minimum distance between a rectangle and a point in 3D space.
    
    Parameters:
    - point: 3D coordinates of the point as a numpy array (e.g., np.array([x, y, z])).
    - rect_corner: 3D coordinates of the rectangle's corner (origin of the rectangle).
    - rect_edge1: 3D vector along one edge of the rectangle.
    - rect_edge2: 3D vector along the adjacent edge of the rectangle.
    
    Returns:
    - The minimum distance between the point and the rectangle.
    - The projection point on the rectangle.
    """

    # Vector from the corner of the rectangle to the point
    v = point - rect_corner
    
    # Project v onto the edges of the rectangle
    d1 = np.dot(v, rect_edge1)
    d2 = np.dot(v, rect_edge2)
    
    # Clamp the projections to the edges of the rectangle
    if d1 < 0:
        d1 = 0
    elif d1 > np.dot(rect_edge1, rect_edge1):
        d1 = np.dot(rect_edge1, rect_edge1)
    
    if d2 < 0:
        d2 = 0
    elif d2 > np.dot(rect_edge2, rect_edge2):
        d2 = np.dot(rect_edge2, rect_edge2)
    
    # The closest point on the rectangle
    closest_point = rect_corner + (d1 / np.dot(rect_edge1, rect_edge1)) * rect_edge1 \
                    + (d2 / np.dot(rect_edge2, rect_edge2)) * rect_edge2
    
    # Compute the distance between the point and the closest point on the rectangle
    distance = np.linalg.norm(point - closest_point)
    
    return distance, closest_point
