import numpy as np


# Function to get the distance to the lth nearest neighbor for all the points
# Input:
#   nr_points: number of points
#   all_pair_distances: matrix of all pairs distances between the points
#   l: the param l
# Output:
#   radii: vector of size nr_points, ith element is point i's distance it's lth nearest neighbor
def get_dist_to_lth_NN(nr_points, all_pair_distances, l):
    radii = [0] * nr_points
    for p in range(nr_points):
        distances = np.array([all_pair_distances[p][q] for q in range(nr_points)])
        distances = sorted(distances)
        radii[p] = distances[l]
    return radii


# Function to get, for any point, its l nearest neighbors along with its distance to the lth nearest neighbor
# Input:
#   nr_points: number of points
#   all_pair_distances: matrix of all pairs distances between the points
#   l: the param l
# Output:
#   graph: a list of lists. for point i, graph[i] is the closest l points to i (including itself)
#   radii: vector of size nr_points, ith element is point i's distance it's lth nearest neighbor
def get_client_neighborhood(nr_points, all_pair_distances, l):
    radii = [0] * nr_points
    graph = []
    for p in range(nr_points):
        distances = np.array([all_pair_distances[p][q] for q in range(nr_points)])
        points_by_dist = np.argsort(distances)
        graph.append(points_by_dist[0:l + 1])
        radii[p] = all_pair_distances[p][points_by_dist[l]]

    return graph, radii


# Function to get, for any point, the list of neighbors in ball of a certain radius around it
# Input:
#   nr_points: number of points
#   all_pair_distances: matrix of all pairs distances between the points
#   radii: array of length nr_points. For point i, the radious around it in which we're looking for neighbors
# Output:
#   graph: a list of lists. for point i, graph[i] is the closest l points to i (including itself)
#   radii: vector of size nr_points, ith element is point i's distance it's lth nearest neighbor
def get_client_neighborhood_graph(nr_points, all_pair_distances, radii):
    graph = list()
    for p in range(nr_points):
        adj_list = list()
        for q in range(nr_points):
            if all_pair_distances[p][q] <= radii[p]:
                adj_list.append(q)
        graph.append(adj_list)

    return graph


# Function to get to the assignment of points to centers
# Input:
#   nr_points: number of points
#   all_pair_distances: matrix of all pairs distances between the points
#   centers: indices of points chosen as centers
# Output:
#   closest: array of length nr_points. closest[i] is the closest point in centers to i
#   dist_vec: vector of size nr_points. dist_vec[i] is point i's distance to closest center in centers
#   second_closest: array of length nr_points. closest[i] is the second closest point in centers to i
def get_center_assignments(nr_points, all_pair_distances, centers):
    closest = [-1] * nr_points
    second_closest = [-1] * nr_points
    dist_vec = [-1] * nr_points
    nr_centers = len(centers)
    for p in range(nr_points):
        # print("")
        # print("Clustering for point p = {}".format(p))
        center_distances = np.array([all_pair_distances[p][centers[c]] for c in range(nr_centers)])
        # print("Dist to centers = {}".format(center_distances))
        # List of centers INDICES sorted by increasing distance to p
        sorted_inds = np.argsort(center_distances)
        # print("Center indices sorted by inc distance: {}".format(sorted_inds))
        closest[p] = int(sorted_inds[0])
        dist_vec[p] = all_pair_distances[p][centers[closest[p]]]
        try:
            second_closest[p] = int(sorted_inds[1])
        except:
            second_closest[p] = None
    return closest, dist_vec, second_closest


# Function to get the index of the ball each point falls into, given disjoint balls.
# Input:
#   nr_points: total number of points
#   all_pair_distances: matrix of all pairs distances
#   balls: an array of tuples wehere tuple (c,r) indicates a ball of radius r around point index c. These balls are supposed to be disjoint
# Output:
#   ball_number: an array of length nr_points, at index i, the value is the number of the critical ball point i is in. -1 if not any.
def get_ball_assignments(nr_points, all_pair_distances, balls):
    nr_balls = len(balls)
    ball_number = [-1] * nr_points
    for b in range(nr_balls):
        (c, r) = balls[b]
        for p in range(nr_points):
            if all_pair_distances[p][c] <= r:
                # if ball_number[p] > -1:
                #     raise Exception(
                #         "Critical balls are not disjoint! Ball {} intersects ball {} at least in point {}".format(
                #             ball_number[p], b, p))
                ball_number[p] = b
                continue
    return ball_number


# Function that given a list of integers, returns a histogram of it a selected subset of it, only for positive values
# Input:
#   numbers: a list of integers
#   subset: a set of indices in range(0, len(numbers))
# Ouput: histogram of *positive* elements in <numbers> projected on indices from <subset>
def get_histogram_for_subset(numbers, subset):
    numbers_subset = np.array([numbers[p] for p in subset])
    # print("numbers projected by the subset = {}".format(numbers_subset))
    numbers_subset = numbers_subset[numbers_subset >= 0]
    # print("After removing negatives               = {}".format(numbers_subset))
    return np.bincount(numbers_subset)


# Function to get quality statistics of a clustering
# Input:
#   nr_points: number of points
#   all_pair_distances: matrix of all pairs distances between the points
#   radii: vector of size nr_points, ith element is a radius around point i
#   centers: indices of points chosen as centers
#   power: A number >= 1 the moment of the distances in the objective function. e.g. p=1 for k-med
# Output:
#   dist_vec: vector of size nr_points. dist_vec[i] is point i's distance to closest center in centers
#   radii_violations: vector of size nr_points. radii_violations[i] is the ratio to which i's neighborhood constraint is violated
#   cost: The clustering cost
#   max_violation: maximum neighborhood violation found, i.e. max_i radii_violations[i]
#   nr_fair: number of points with satisfied fairness constraints
def get_clustering_stats(nr_points, all_pair_distances, radii, centers, power):
    pred, dist_vec, _ = get_center_assignments(nr_points, all_pair_distances, centers)
    # dist_vec = np.array([all_pair_distances[p][centers[pred[p]]] for p in range(nr_points)])
    radii_violations = np.divide(dist_vec, radii)
    cost = np.power(np.sum(np.power(dist_vec, power)), 1 / power)
    # Some datasets have many duplicate points which forces neighborhood and cost to be zero at that index, resulting in NaN for violation
    max_violation = np.max([i for i in radii_violations if not np.isnan(i)])
    nr_fair = len([i for i in radii_violations if np.isnan(i) or i <= 1])
    return dist_vec, radii_violations, cost, max_violation, nr_fair
