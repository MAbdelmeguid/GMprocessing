#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 12:18:13 2024

@author: glavrent
"""
#libraries
import numpy as np

def str_replace_nth_int(s, sub, repl, n):
    '''
    Replace sub-string every n^th occurrence.

    Parameters
    ----------
    s : string
        Main string.
    sub : string
        Substring for replacement.
    repl : string
        Substring to replace.
    n : int
        Interval.

    Returns
    -------
    string
        Updated string.
    '''
    
    find = s.find(sub)
    # If find is not -1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != n:
        # find + 1 means we start searching from after the last match
        find = s.find(sub, find + 1)
        i += 1
    # If i is equal to n we found nth match so replace
    if i == n:
        return s[:find] + repl + s[find+len(sub):]
    
    return s

def polar_to_cartesian(r, theta, degrees=False):
    """
    Converts polar coordinates to Cartesian coordinates.

    Parameters:
    - r (float): Radius or distance from the origin.
    - theta (float): Angle in radians by default.
    - degrees (bool): If True, treats theta as degrees.

    Returns:
    - (float, float): Tuple containing (x, y) coordinates.
    """
    if degrees:
        theta = np.deg2rad(theta)  # Convert degrees to radians

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return (x, y)

def get_evenly_spaced_elements(arr, num, return_index=False):
    """
    Retrieves `num` evenly spaced indices from the input array `arr`.
    Ensures that the first and last indices are always included if `num` > 2.
    
    Parameters:
    - arr (list or np.ndarray): The input array from which to retrieve indices.
    - num (int): The number of evenly spaced indices to retrieve.
    
    Returns:
    - np.ndarray: Array of evenly spaced indices.
    
    Raises:
    - ValueError: If `num` is not positive or exceeds the array length.
    """
    length = len(arr)
    
    if num <= 0:
        raise ValueError("Number of indices must be positive.")
    if num > length:
        raise ValueError("Number of indices requested exceeds array length.")
    
    if num == 1:
        # Return the first index. Alternatively, you could return the last index or the middle index.
        indices = np.array([0])
    elif num == 2:
        # Return the first and last indices
        indices = np.array([0, length - 1])
    else:
        # Generate `num` evenly spaced indices, ensuring first and last are included
        indices = np.linspace(0, length - 1, num)
        indices = np.round(indices).astype(int)
        
        # Ensure first and last indices are exactly 0 and length -1
        indices[0] = 0
        indices[-1] = length - 1
        
    if not return_index:
        return arr[indices]        
    else:
        return arr[indices], indices