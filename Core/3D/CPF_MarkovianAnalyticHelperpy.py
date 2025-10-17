#!usr/bin/env python
import numpy as np
from collections import defaultdict
from scipy.spatial import ConvexHull
from itertools import combinations
from functools import lru_cache
import itertools
import math
from sklearn.decomposition import PCA
from scipy.spatial import QhullError

## \brief Generates all possible connectivity tuples for n points, sorted lexicographically.
# 
# This function enumerates all set partitions of n elements and returns them
# as tuples of tuples, where each inner tuple is a connected component.
#
# \param n int, number of elements (points)
# \return list of all connectivity tuples sorted lexicographically
# 
# \example
# >>> generate_all_connectivity_tuples(3)
# [((0,), (1,), (2,)), ((0,), (1, 2)), ((0, 1), (2,)), ((0, 1, 2),), ((0, 2), (1,))]
@lru_cache(maxsize=None)
def generate_all_connectivity_tuples(n):
    """Generates all possible connectivity tuples for n points, sorted by lexicographical order.

    n: int, the number of elements (points)

    returns: list, the list of all connectivity tuples for the set of n points, sorted lexicographically

    >>> generate_all_connectivity_tuples(3)
    [((0,), (1,), (2,)), ((0,), (1, 2)), ((0, 1), (2,)), ((0, 1, 2),), ((0, 2), (1,))]
    """

    def partitions(set_):
        """Generate all partitions of a set, ensuring sorted partitions"""
        if len(set_) == 1:
            yield [tuple(set_)]
            return
        first = set_[0]
        for smaller in partitions(set_[1:]):
            for n, subset in enumerate(smaller):
                yield smaller[:n] + [(first,) + subset] + smaller[n+1:]
            yield [(first,)] + smaller

    elements = tuple(range(n))
    
    # Generate all connectivity tuples
    all_connectivity_tuples = [
        tuple(sorted(partition, key=lambda x: (x[0], x))) for partition in partitions(elements)
    ]
    
    # Sort the outer list of tuples
    all_connectivity_tuples.sort(key=lambda partition: (sorted(partition), partition))
    
    return all_connectivity_tuples

## \brief Filters connectivity tuples to exclude connections between differently colored points.
#
# Each tuple represents a partitioning of point indices into connected components.
# This function removes tuples where any component contains more than one color.
# If \p last_color_unknown is True, the final point is treated as having unknown color
# and is allowed to appear in multi-color components without constraint.
#
# \param[in] tuples List of connectivity tuples to filter (list of tuple of tuples).
# \param[in] colors Numpy array of integers specifying the color of each point.
# \param[in] last_color_unknown Boolean flag; if True, treats the last point as having unknown color.
# \return Filtered list of allowed connectivity tuples.
#
# \example
# >>> all_tuples = generate_all_connectivity_tuples(3)
# >>> colors1 = np.array([1, 1, 2])
# >>> allowed_tuples_colors(all_tuples, colors1)
# [((0,), (1,), (2,)), ((0, 1), (2,))]
def allowed_tuples_colors(tuples,colors, last_color_unknown = False):
    """Returns the allowed connectivity tuples for points with the given colors. Different colors can't be connected.

    tuples: list of tuples, the list of connectivity tuples to filter
    colors: np.array, the colors of the points
    last_color_unknown: bool, whether the last color is unknown and should be ignored (default False)

    returns: list, the list of allowed connectivity tuples

    >>> all_tuples = generate_all_connectivity_tuples(3)
    >>> colors1 = np.array([1, 1, 2])
    >>> result1 = allowed_tuples_colors(all_tuples, colors1)
    >>> result1.sort()
    >>> print(result1)
    [((0,), (1,), (2,)), ((0, 1), (2,))]

    >>> colors2 = np.array([1, 2, 1])
    >>> result2 = allowed_tuples_colors(all_tuples, colors2)
    >>> result2.sort()
    >>> print(result2)
    [((0,), (1,), (2,)), ((0, 2), (1,))]

    >>> colors3 = np.array([1, 1, 1])
    >>> result3 = allowed_tuples_colors(all_tuples, colors3)
    >>> result3.sort()
    >>> print(result3)
    [((0,), (1,), (2,)), ((0,), (1, 2)), ((0, 1), (2,)), ((0, 1, 2),), ((0, 2), (1,))]

    >>> colors4 = np.array([1, 2])
    >>> result4 = allowed_tuples_colors(all_tuples, colors4, last_color_unknown=True)
    >>> result4.sort()
    >>> print(result4)
    [((0,), (1,), (2,)), ((0,), (1, 2)), ((0, 2), (1,))]
    """
    if last_color_unknown:  # don't filter based on the last point's color
        final_point = len(colors) #not -1 since colors is for all the points which actually do have a color assigned
        allowed_tuples = [
            tup for tup in tuples
            if all(len(set(colors[i] for i in component if i != final_point)) == 1 
                for component in tup if len(component) > 1 or final_point not in component)
        ]
        return allowed_tuples
    
    # Filter and collect valid tuples using a list comprehension
    allowed_tuples = [
        tup for tup in tuples
        if all(len(set(colors[i] for i in component)) == 1 for component in tup)
    ]

    return allowed_tuples


## \brief Computes the connectivity tuple resulting from a sequence of graph cuts.
#
# Each cut results from a single hyperplane, and specifies a set of vertices that are 
# to be severed from all vertices outside the cut. This function returns the resulting
# connected components as a tuple of tuples containing the indices of the vertices.
#
# \param[in] num_points Total number of vertices (int).
# \param[in] cuts Tuple of tuples. Each inner tuple specifies vertices that should be disconnected
#                 from the rest of the graph.
# \return Tuple of tuples representing the resulting connected components after all cuts.
#
# \example
# >>> graph_cutter(5, ((0, 1), (2, 3)))
# ((0, 1), (2, 3), (4,))
#
# >>> graph_cutter(5, ((0, 1), (1, 2)))
# ((0,), (1,), (2,), (3, 4))
@lru_cache(maxsize = None) #cache the results of this function to avoid recalculating. must converet argument to hashable type
def graph_cutter(num_points, cuts):
    """Returns the connectivity tuple for a series of graph cuts.
    
    num_points: int, the number of points
    cuts: tuple of tuples, each tuple contains indices of points which are cut from every point OUTSIDE the tuple

    returns: tuple of tuples, the connectivity tuple

    >>> graph_cutter(5, ((0, 1), (2, 3)))
    ((0, 1), (2, 3), (4,))

    >>> graph_cutter(5, ((0, 1), (1, 2)))
    ((0,), (1,), (2,), (3, 4))
    """
    
    vertices = list(range(num_points))
    remaining = set(vertices)  # Use a set to track remaining vertices
    connectivity_tuple = []

    for v in vertices:
        if v not in remaining:  # Skip if already processed
            continue
        cc = [v]
        remaining.remove(v)  # Mark v as processed

        for w in list(remaining):  # Iterate over remaining vertices
            if all((v in cut) == (w in cut) for cut in cuts):  # Check if v and w belong to same side of all cuts
                cc.append(w)
                remaining.remove(w)  # Mark w as processed

        connectivity_tuple.append(cc)  # Keep as list for now

    return tuple(map(tuple, connectivity_tuple))  # Convert only once at the end


## \brief Calculates the Poisson rate of hyperplanes hitting the convex hull of 1D points.
#
# \param[in] points A 1D NumPy array of shape (n,) containing the coordinates of the points.
# \return Float value representing the length of the convex hull (i.e., the hit rate).
#
# \example
# >>> hitrate_1d(np.array([-2, -1, 0, 1, 2]))
# 4
def hitrate_1d(points):
    """Calculates the Poisson rate of hyperplanes hitting the convex hull of points in 1D.

    points: np.array, shape (n,), the points to hit

    returns: float, the rate of hyperplanes hitting the convex hull

    >>> hitrate_1d(np.array([-2, -1, 0, 1, 2]))
    4
    """

    return max(points)-min(points) #hitrate is the length of segment


## \brief Calculates the Poisson rate of hyperplanes hitting the convex hull of 2D points.
#
# \param[in] points NumPy array of shape (n, 2) representing the 2D coordinates of the points.
# \return Float value representing the estimated hit rate (half the perimeter of the convex hull).
#
# \example
# >>> hitrate_2d(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))
# 2.0
# >>> hitrate_2d(np.array([[0, 0], [1, 0]]))
# 1.0
def hitrate_2d(points):
    """Calculates the Poisson rate of hyperplanes hitting the convex hull of points in 2D.

    points: np.array, shape (n,2), the points to hit

    returns: float, the rate of hyperplanes hitting the convex hull

    >>> import numpy as np
    >>> hitrate_2d(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))
    2.0

    >>> hitrate_2d(np.array([[0, 0], [1, 0]]))
    1.0
    
    >>> hitrate_2d(np.array([[0,0],[0,0],[0,0],[0,0]]))
    0.0
    """
    if len(points) == 2:
        return np.linalg.norm(points[1] - points[0])
    elif len(points) == 3:
        return np.sum([np.linalg.norm(points[i] - points[(i+1)%3]) for i in range(3)]) / 2
    try:
        hull = ConvexHull(points)
        perimeter = hull.area
        return perimeter / 2
    except QhullError:
        # Degenerate in 2D → project to 1D and return segment length
        centered = points - np.mean(points, axis=0)
        if np.allclose(centered, 0):
            return 0.0  # all points coincide, no hitrate
        direction = PCA(n_components=1).fit(centered).components_[0]
        projected = np.dot(centered, direction)
        return np.ptp(projected)


## \brief Calculates the dihedral angle between two normal vectors.
#
# \param[in] norm1 NumPy array representing the normal vector of the first face.
# \param[in] norm2 NumPy array representing the normal vector of the second face.
# \return Float representing the dihedral angle in radians.
#
# \example
# >>> dihedral_angle(np.array([0, -1, 0]), np.array([1, 1, 1]))
# 2.1862760354652844
def dihedral_angle(norm1, norm2):
    """Calculates the dihedral angle between two normal vectors.

    Parameters:
    norm1 (np.array): Normal vector of the first face.
    norm2 (np.array): Normal vector of the second face.

    Returns:
    float: The dihedral angle in radians.

    >>> dihedral_angle(np.array([0, -1, 0]), np.array([1, 1, 1]))
    2.1862760354652844
    """

    cos_theta = np.dot(norm1, norm2) / (np.linalg.norm(norm1) * np.linalg.norm(norm2))
    # Ensure the cosine is within valid range due to potential numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return angle

## \brief Ensures that a normal vector points outward from a convex surface.
#
# \param[in] norm NumPy array representing the normal vector to check.
# \param[in] point_on_face NumPy array representing a point lying on the face.
# \param[in] centroid NumPy array representing the centroid of the convex object.
# \return NumPy array of the (possibly flipped) outward-facing normal vector.
def ensure_outward_facing(norm, point_on_face, centroid):
    """Ensure that the normal vector is outward-facing."""
    to_centroid = centroid - point_on_face
    if np.dot(norm, to_centroid) > 0:
        return -norm  # Flip the normal to point outward
    return norm


## \brief Calculates the Poisson rate, mu, of hyperplanes hitting the convex hull of 3D points.
#
# \param[in] points NumPy array of shape (n, 3) representing the 3D coordinates of the input points.
# \return Float value representing the estimated hyperplane hit rate.
#
# \example
# >>> hitrate_3d(np.array([[0, 0, 0], [1, 0, 0]]))
# 1.0
#
# >>> hitrate_3d(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.1, 0.1, 0.1]]))
# 2.2262549897645005
def hitrate_3d(points):
    """
    Calculates the Poisson rate of hyperplanes hitting the convex hull of points in 3D.

    Parameters:
    points (np.array): 3D array of points to hit, shape (n, 3).

    Returns:
    float: The rate of hyperplanes hitting the convex hull.

    Examples:
    >>> hitrate_3d(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.1, 0.1, 0.1]]))
    2.2262549897645005

    >>> hitrate_3d(np.array([[0, 0, 0], [1, 0, 0]]))
    1.0

    >>> hitrate_3d(np.array([[0, 0, 0], [1, 0, 0], [0,0,0]]))
    1.0

    >>> three_d_degenerate = hitrate_3d(np.array([[0, 0,5], [1, 0,5], [0, 1,5], [1, 1,5]]))
    >>> two_d = hitrate_2d(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))
    >>> three_d_degenerate == two_d
    True
    """
    if len(points) == 2:
        return np.linalg.norm(points[1] - points[0])
    elif len(points) == 3:
        return np.sum([np.linalg.norm(points[i] - points[(i+1)%3]) for i in range(3)]) / 2
    try:
        hull = ConvexHull(points)
    except QhullError:
        # Points are degenerate: project to 2D plane and try the 2D function
        centered = points - np.mean(points, axis=0)
        if np.allclose(centered, 0):
            return 0.0
        projected = PCA(n_components=2).fit_transform(centered)
        return hitrate_2d(projected)

    edge_contributions = []
    centroid = np.mean(points, axis=0)

    for simplex in hull.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = tuple(sorted((simplex[i], simplex[j])))
                if edge not in [e[0] for e in edge_contributions]:
                    adjacent_faces = [face for face in hull.simplices if edge[0] in face and edge[1] in face]
                    if len(adjacent_faces) < 2:
                        continue  # skip boundary edges
                    face1_edge1 = points[adjacent_faces[0][1]] - points[adjacent_faces[0][0]]
                    face1_edge2 = points[adjacent_faces[0][2]] - points[adjacent_faces[0][0]]
                    face2_edge1 = points[adjacent_faces[1][1]] - points[adjacent_faces[1][0]]
                    face2_edge2 = points[adjacent_faces[1][2]] - points[adjacent_faces[1][0]]

                    norm1 = ensure_outward_facing(np.cross(face1_edge1, face1_edge2), points[adjacent_faces[0][0]], centroid)
                    norm2 = ensure_outward_facing(np.cross(face2_edge1, face2_edge2), points[adjacent_faces[1][0]], centroid)

                    edge_length = np.linalg.norm(points[edge[1]] - points[edge[0]])
                    angle = dihedral_angle(norm1, norm2)
                    edge_contributions.append((edge, edge_length * angle))

    return sum(contribution for edge, contribution in edge_contributions) / (2 * np.pi)


## \brief Computes the single-hyperplane cut rates for sorted 1D points.
#
# This function calculates the rates of each single hyperplane cut of the points in 1D. In the 
# paper, these single hyperplane cuts are referred to using a slash "/" followed by a component.
#
# \param[in] points A sorted 1D NumPy array of shape (n,) representing the positions of the points.
# \return A dictionary mapping each prefix subset of indices to its associated partition rate.
#
# \example
# >>> points = np.array([0, 1, 2, 5])
# >>> slash_rates_1d(points)
# {(0,): 1, (0, 1): 1, (0, 1, 2): 3}
def slash_rates_1d(points):
    """Returns the rates of each single hyperplane partition of the points in 1D.

    points: sorted np.array, shape (n,), the points to partition

    returns: dict, the rates of each partition

    >>> points = np.array([0, 1, 2, 5])
    >>> slash_rates_1d(points)
    {(0,): 1, (0, 1): 1, (0, 1, 2): 3}
    """

    if not np.array_equal(points, np.sort(points)):
        raise ValueError("Points must be sorted")

    rates = {}
    #for each segment between points, the rate is the length of the segment
    for i in range(len(points)-1):
        connected_component = (tuple(range(0,i+1),))
        rates[connected_component] = points[i+1] - points[i]
    
    return rates


## \brief Computes the single-hyperplane partition rates for points in 1D, 2D, or 3D.
#
# This function calculates the rates of each single hyperplane cut of the points in 1,2, or 3D. In 
# the paper, these single hyperplane cuts are referred to using a slash "/" followed by a component.
#
# The method uses inclusion-exclusion to determine the rate associated with each subset 
# remaining connected post cut. Numerical noise is added to avoid degeneracies in convex hulls.
#
# \param[in] points A NumPy array of shape (n, d) where d ∈ {1, 2, 3} and n is the number of points.
# \return A dictionary mapping each subset of indices (as a tuple) to its associated hyperplane hit rate.
#
# \note If \p d == 1, the function defers to \c slash_rates_1d. If \p n == 2, it projects to 1D.
#
# \example
# >>> points2d = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
# >>> result = slash_rates(points2d)
# >>> result[(0,)]  # example access to the rate of a subset
# 0.292893
def slash_rates(points):
    """Returns the rates of each single hyperplane partition of the points.

    points: np.array, shape (n,d), the points to partition

    returns: dict, the rates of each partition

    >>> points2d = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    >>> result = slash_rates(points2d)
    >>> result_rounded = {k: round(v, 6) for k, v in result.items()}
    >>> print(len(result_rounded))
    7
    >>> print(result_rounded[(0,)])
    0.292893
    >>> print(result_rounded[(1,)])
    0.292893
    >>> print(result_rounded[(0, 2)])
    0.414214

    >>> points2d = np.array([[0, 0], [1, 0], [0, 1]])
    >>> result = slash_rates(points2d)
    >>> print(len(result))
    3

    >>> points2d = np.array([[0, 0], [1, 0]])
    >>> result = slash_rates(points2d)
    >>> print(len(result))
    1

    >>> points2d = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [.1,.1]])
    >>> result2d = slash_rates(points2d)
    >>> print(len(result2d))
    15
    >>> points3d = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [.1,.1, 0]])
    >>> result3d = slash_rates(points3d)
    >>> print(all([np.allclose(result2d[k],result3d[k]) for k in result2d]))
    True
    """
    dimension = points.shape[1] #this block makes the function general for any dimension
    if dimension == 1:
        return slash_rates_1d(points)
    elif dimension == 2:
        hitrate = hitrate_2d
    elif dimension == 3:
        hitrate = hitrate_3d

    n = len(points)

    #loop over all possible partitions of the points into two disjoint sets
    #start with just one point in the first set and the rest in the second set

    if n == 2:
        projection = (points[1]-points[0])/np.linalg.norm(points[1] - points[0])
        sorted_points = np.dot(points, projection)
        return slash_rates_1d(sorted_points)


    rates = defaultdict(int)
    whole_hitrate = hitrate(points)

    for i in range(n): #single point partitions base case
        rest_hitrate = hitrate(np.delete(points, i, axis=0))
        rates[(i,)] = whole_hitrate - rest_hitrate

    if n == 3:
        return rates
    
    def subset_rates(subset):
        m = len(subset)
        if m == 1:
            return rates[subset] #already calculated in base case
        
        rest_points = np.delete(points, list(subset), axis=0)
        rest_hitrate = hitrate(rest_points)
        
        rate = whole_hitrate - rest_hitrate #next we need to subtract the subset rates
        for sub_size in range(1,m):
            for sub_subset in combinations(subset, sub_size):
                rate -= rates[sub_subset] #subtracting off rate of smaller subsets
        
        rates[subset] = rate #can modify mutable dictionary in function scope
        return rate
    
    #calculate the rates of all other partitions up to the middle one which is an edge case
    for size in range(2,n//2):
        for subset in combinations(range(n), size):
            subset_rates(subset)
    
    if n % 2 == 0: #have to be careful with middle partitions so as to not be redundant
        for subset in combinations(range(n), n//2):
            complement = tuple(sorted(set(range(n)) - set(subset)))
            #only keep the subset if its first element is less than the complement's first element
            if subset < complement:
                subset_rates(subset)
    else: #n is odd
        for subset in combinations(range(n), n//2):
            subset_rates(subset)
    return rates


## \brief Computes the color distribution for the final point based on cuts.
#
# Given the single-hyperplane cuts, colors of the known points, and a prior color distribution,
# this function determines the probability distribution of the final point’s color.
#
# \param[in] partitions Tuple of tuples representing the connectivity tuples from single hyperplane cuts.
# \param[in] colors Tuple of length (num_points - 1) representing the known point colors.
# \param[in] num_points Total number of points, including the final point whose color is unknown.
# \param[in] color_dist Tuple representing the prior probability distribution over colors.
# \return NumPy array of posterior color probabilities for the final point.
#
# \example
# >>> partitions = ((0,), (1,), (0, 1))
# >>> colors = (1, 2, 1)
# >>> color_dist = (.2, .2, .2, .2, .2)
# >>> color_from_partitions(partitions, colors, 4, color_dist)
# array([0., 1., 0., 0., 0.])
def color_from_partitions(partitions, colors, num_points, color_dist):
    """Returns the color probability distribution for the final point given the partitions 
    and colors of the other points.
    
    partitions: tuple of tuples, the single cut connectivity tuples of the points
    colors: tuple, length (num_points-1), the colors of the points
    num_points: int, the number of points, including the final point whose color is unknown
    color_dist: tuple, the probabilities of each color

    returns: np.array, the calculated color probabilities for the final point

    >>> partitions = ((0,), (1,), (0, 1)) #the complete partition is ((0,), (1,), (2, 3)), so must be color 1
    >>> colors = (1, 2, 1)
    >>> num_points = 4
    >>> color_dist = (.2,.2,.2,.2,.2)
    >>> color_from_partitions(partitions, colors, num_points, color_dist)
    array([0., 1., 0., 0., 0.])

    >>> partitions = ((0,), (0,3)) #the complete partition is ((0,), (1,2), (3)), so uniform distribution
    >>> colors = (1, 2, 2)
    >>> num_points = 4
    >>> color_dist = (.4,.6)
    >>> got = color_from_partitions(partitions, colors, num_points, color_dist)
    >>> expected = np.array([0.4, 0.6])
    >>> np.allclose(got, expected)
    True

    >>> partitions = ()
    >>> colors = (1, 1, 1)
    >>> got = color_from_partitions(partitions, colors, num_points, color_dist)
    >>> expected = np.array([0., 1.])
    >>> np.allclose(got, expected)
    True
    """
    num_colors = len(color_dist)
    color_probs = np.zeros(num_colors)

    #find which point, if any, points[-1] is connected to.
    #points[-1] is connected to points[i] if each partition has ((num_points-1) in partition) == (i in partition)
    connected_point = None
    for i in range(num_points-1):
        if all(((num_points-1) in partition) == (i in partition) for partition in partitions):
            connected_point = i
            break
    if not connected_point == None:
        color_probs[colors[connected_point]] = 1
    else: #if points[-1] is isolated, the color is uniformly distributed
        color_probs = np.array(color_dist)
    return color_probs


## \brief Computes the CPF of the final point based on colors of the other points and color distributions.
#
# \param[in] points NumPy array of shape (n, d), where points[:-1] are known and points[-1] is the query point.
# \param[in] colors Tuple of length (n - 1) representing the known colors of the first n-1 points.
# \param[in] color_dist Tuple representing the prior color distribution over all possible colors.
# \param[in] noise Optional float (default 1e-8) controlling the magnitude of noise added to avoid degeneracies.
# \return NumPy array representing the posterior color probability distribution for the last point.
#
# \example
# >>> points = np.array([[0, 1], [1, 0], [0, 0]])
# >>> colors = (0, 1)
# >>> color_dist = (.5, .5)
# >>> color_distribution(points, colors, color_dist)
# array([0.5, 0.5])
def color_distribution(points, colors, color_dist):
    """Returns the color probability distribution for points[-1].
    
    points: np.array, shape (n,d). points[:-1] are the colored points, points[-1] is the point whose color is unknown
    colors: tuple of ints, shape (n-1). Colors of points[:-1].
    color_dist: tuple, the probabilities of each color

    returns: np.array, the calculated color distribution for the final point
    >>> points = np.array([[0, 1], [1, 0], [0, 0]])
    >>> colors = (0, 1)
    >>> color_dist = (.5,.5)
    >>> got = color_distribution(points, colors, color_dist)
    >>> expected = np.array([0.5, 0.5])
    >>> print(np.allclose(got, expected))
    True

    >>> points3d = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    >>> got3d = color_distribution(points3d, colors, color_dist)
    >>> print(np.allclose(got, got3d))
    True
    """
    n = len(points)
    rates = slash_rates(points)
    partitions = list(rates.keys())
    all_connectivity = generate_all_connectivity_tuples(n)
    allowed_partitions = allowed_tuples_colors(all_connectivity, colors, last_color_unknown=True)
    #allowed_partitions = allowed_tuples_geometric(allowed_partitions, points)
    #^^ the above line might be redundant, tests indicate it give the same result with minimal time difference
    num_p = len(partitions)
    ret = np.zeros(len(color_dist))

    # Precompute exp(-rates[partition]) for each partition
    exp_rates = {partition: np.exp(-rates[partition]) for partition in partitions}

    #calculate the probability of each member of the superset of all partitions
    #from there, use color_from_partitions to calculate the color distribution, then add it to the final distribution
    power_set = itertools.chain.from_iterable(itertools.combinations(partitions, r) for r in range(num_p+1))
    probcount = 0
    for subset in power_set: #remember, to convert a rate to a probability, do P(not happen) = e^(-rate)
        partition = graph_cutter(n, subset)
        if not partition in allowed_partitions:
            continue
        # Memoized product calculations
        subset_prob = math.prod([1 - exp_rates[partition] for partition in subset])
        complement_prob = math.prod([exp_rates[partition] for partition in partitions if partition not in subset])
        subset_prob *= complement_prob
        
        subset_colors = color_from_partitions(subset, colors, n, color_dist)
        ret += subset_prob * subset_colors    
        probcount += subset_prob
    ret = ret / probcount  # normalize
    ret = np.clip(ret, 0, None)  # remove any tiny negatives
    ret /= np.sum(ret)  # re-normalize after clipping
    return ret


if __name__ == "__main__":
    import doctest
    doctest.testmod()