import random
import numpy as np
from sklearn.cluster import KMeans
import time
import itertools
from priority_lp_solver import solve_priority_p_k_clustering_lp

from util.read_write_utils import write_output
from util.data_process_utils import get_clustering_stats, get_center_assignments, get_ball_assignments, \
    get_histogram_for_subset, get_client_neighborhood_graph


# A binary search on top of Filter to find the smallest dilation in range [<dial_min>, <dial_max>] for which you get at most <n_clusters> centers
# Input:
#   dial_min:
#   all_pair_distances: the metric. At index [i][j] is the distance between point i and j
#   n_clusters: the number of clusters
#   R: vector of size nr_points, ith element is the R_i from the algorithm
# Output:
#   centers, partition: Filter output
#   dial_max: The dilation used ultimately
def _bin_search_on_filter(dial_min, dial_max, all_pair_distances, n_clusters, R):
    if dial_min < 0 or dial_max <= dial_min:
        raise Exception(" Violated: 0 <= dial_min ({}) < dial_max ({})".format(dial_min, dial_max))

    while dial_max - dial_min > 1e-6:
        dilation = (dial_max + dial_min) / 2
        centers, _ = filter(all_pair_distances, R, dilation)

        # If found feasible solution, try to decrease dilation to get better cost
        if len(centers) <= n_clusters:
            dial_max = dilation
        else:
            dial_min = dilation

    centers, partition = filter(all_pair_distances, R, dial_max)

    if len(centers) > n_clusters:
        raise Exception(
            "Ended up with {} centers! dial_min = {} dial_max = {}".format(len(centers), dial_min, dial_max))
    return centers, partition, dial_max


# Jung et al's paper, a binary search on top of Filter
# Input:
#   nr_points: the total number of points
#   all_pair_distances: the metric. At index [i][j] is the distance between point i and j
#   n_clusters: the number of clusters
#   power: A number >= 1 the moment of the distances in the objective function. e.g. p=1 for k-med only used for computing the cost here
#   radii: vector of size nr_points, ith element is a radius around point i
# Output:
#   dictionary that contains various attributes of the solution
def jung_etal(nr_points, all_pair_distances, n_clusters, power, radii):
    time1 = time.monotonic()
    centers, _, dilation = _bin_search_on_filter(1, 2, all_pair_distances, n_clusters, radii)
    time2 = time.monotonic()

    dist_vec, radii_violations, cost, max_violation, nr_fair = get_clustering_stats(nr_points, all_pair_distances,
                                                                                    radii,
                                                                                    centers, power)

    output = dict()
    output["dilation"] = dilation
    output["centers"] = centers
    output["dist_vec"] = dist_vec
    output["cost"] = cost
    output["max_violation"] = max_violation
    output["radii_violations"] = radii_violations.tolist()
    output["nr_fair"] = nr_fair
    output["time"] = time2 - time1
    return output


# Function to wrap Arya et. al. 's local search algo as a vanilla clustering algo
# Input:
#   nr_points: the total number of points
#   all_pair_distances: the metric. At index [i][j] is the distance between point i and j
#   n_clusters: the number of clusters
#   power: A number >= 1 the moment of the distances in the objective function. e.g. p=1 for k-med
#   epsilon: small number < 1. local search terminates if solution doesn't improve by a factor of at least (1-epsilon)
#   max_iter: maximum number of iterations to run the local search. Useful if epsilon turns out to be too small
#   radii: vector of size nr_points, ith element is a radius around point i. Used for evaluation purposes only
#   centers: Initial set of centers. An array of point indices of size n_clusters
# Output:
#   dictionary that contains various attributes of the solution
def arya_etal_driver(nr_points, all_pair_distances, n_clusters, power, epsilon, max_iter, radii, centers=None):
    time1 = time.monotonic()
    vanilla_centers, _, _, _ = arya_etal(nr_points, all_pair_distances, n_clusters, power, epsilon, max_iter, centers,
                                         critical_balls=None,
                                         nr_swaps=1)
    time2 = time.monotonic()

    dist_vec, radii_violations, cost, max_violation, nr_fair = get_clustering_stats(nr_points,
                                                                                    all_pair_distances,
                                                                                    radii,
                                                                                    vanilla_centers,
                                                                                    power)

    output = dict()
    output["centers"] = vanilla_centers
    output["dist_vec"] = dist_vec
    output["cost"] = cost
    output["max_violation"] = max_violation
    output["radii_violations"] = radii_violations.tolist()
    output["nr_fair"] = nr_fair
    output["time"] = time2 - time1
    return output


# Function to wrap k-means++ algo as a vanilla clustering algo
# Input:
#   points: the actual points, in a dataframe
#   n_clusters: the number of clusters
#   epsilon: small number < 1. local search terminates if solution doesn't improve by a factor of at least (1-epsilon)
#   max_iter: maximum number of iterations to run the local search. Useful if epsilon turns out to be too small
#   radii: vector of size nr_points, ith element is a radius around point i. Used for evaluation purposes only
#   seed: integer for seeding randomness
# Output:
#   dictionary that contains various attributes of the solution
def kmeanspp_driver(points, n_clusters, epsilon, max_iter, radii, seed):
    time1 = time.monotonic()
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=1, max_iter=max_iter, tol=epsilon).fit(points)
    time2 = time.monotonic()

    nr_points = len(points.index)
    cost_vec = []
    for i in range(nr_points):
        row = points.loc[i].to_numpy()
        row = row.reshape(1, -1)
        cost_vec.append(-kmeans.score(row))
    dist_vec = np.sqrt(cost_vec)
    radii_violations = np.divide(dist_vec, radii)
    output = dict()
    output["dist_vec"] = dist_vec.tolist()
    output["cost"] = np.sqrt(np.sum(cost_vec))
    output["radii_violations"] = radii_violations.tolist()
    output["max_violation"] = np.max(radii_violations)
    output["time"] = time2 - time1

    return output


# Arya et. al.'s local search algorithm
# Input:
#   nr_points: the total number of points
#   all_pair_distances: the metric. At index [i][j] is the distance between point i and j
#   n_clusters: the number of clusters
#   power: A number >= 1 the moment of the distances in the objective function. e.g. p=1 for k-med
#   epsilon: small number < 1. local search terminates if solution doesn't improve by a factor of at least (1-epsilon)
#   max_iter: maximum number of iterations to run the local search. Useful if epsilon turns out to be too small
#   centers: Initial set of centers. An array of point indices of size n_clusters
#   critical_balls: an array of size <= n_clusters of tuples. Tuple (c,r) indicates a ball of radius r around point index c and if given, the algorithm would ensure to contain at least a center in each ball. These balls are supposed to be disjoint
#   nr_swaps: number of simultaneous swaps performed.
# Output:
#   centers: Indices of nr_clusters many points, decided to be cluster centers.
#   closest_center: Array of length <nr-points> at index i, is the index in <centers> indicating the closest center to point i
#   dist_vec: Array of length <nr_points> at index i, distance of point i to the point indicated by closest_center[i]
#   cost: The clustering objective
def arya_etal(nr_points, all_pair_distances, n_clusters, power, epsilon, max_iter, centers=None, critical_balls=None,
              nr_swaps=1):
    nr_balls = 0
    if critical_balls is not None:
        nr_balls = len(critical_balls)

    # This holds the number of the critical ball a point is in. There is at most one such ball. -1 if none.
    ball_number = [-1] * nr_points
    if nr_balls > 0:
        ball_number = get_ball_assignments(nr_points, all_pair_distances, critical_balls)

    # print("The ball numbers for each point = {}".format(ball_number))

    def _check_balls_are_hit(points):
        cnt_in_ball = get_histogram_for_subset(ball_number, points)
        return len([i for i in cnt_in_ball if i > 0]) == nr_balls

    def _validate_input():
        nonlocal centers
        # To speed things up, we keep track of the closest and the 2nd-closest center in the current solution to any point
        # Hence if n_clusters < 2 our implementation would not work
        if n_clusters < 2:
            raise Exception(
                "Current implementation of k-median does not support n_clusters = {} < 2".format(n_clusters))

        # Enforcing epsilon be in range (0,1)
        if epsilon > 1 - 1e-9 or epsilon < 1e-9:
            raise Exception("epsilon = {} not in range (0,1)".format(epsilon))

        if power < 1:
            raise Exception("power = {} < 1".format(power))

        # Number of critical balls, if any given
        if nr_balls > 0 and nr_balls > n_clusters:
            raise Exception(
                "Critical balls has size {} which is more than n_clusters = {}".format(len(centers),
                                                                                       n_clusters))

        # If centers is not given, initialize cluster centers
        if centers is None:
            centers = []
            # If critical balls are given, choose their centers as centres
            if nr_balls > 0:
                centers = [c for (c, r) in critical_balls]
            # Fill in the rest randomly to make sure there are exactly n_clusters centers (might have duplicates)
            centers = centers + random.sample(range(0, nr_points), n_clusters - len(centers))
        else:
            # If initial centers are given
            # Check there is exactly n_clusters of them
            if len(centers) != n_clusters:
                raise Exception(
                    "Initial centers has size {} which does not match n_clusters = {}".format(len(centers),
                                                                                              n_clusters))
            if not _check_balls_are_hit(centers):
                raise Exception("Given centers do not hit all critical balls.")
        # print("The center indices are: {}".format(centers))
        if nr_swaps > n_clusters - 1 or nr_swaps < 1:
            raise Exception("Number of simultaneous swaps is {} but has to be between 1 and k-1 = {}".format(nr_swaps,
                                                                                                             n_clusters - 1))

    _validate_input()

    # For each point, holds the index of the cluster it is assigned to
    closest_center = [None] * nr_points

    iter = 0
    updated_sln = True
    while updated_sln is True:
        updated_sln = False

        # Assign each point to the closest center and compute the solution cost
        # print("Centers now are {}".format(centers))
        closest_center, dist_vec, second_closest_center = get_center_assignments(nr_points, all_pair_distances, centers)
        cost_to_power = np.sum(np.power(dist_vec, power))
        # print("******* clustering cost is = {}".format(cost))

        cnt_in_ball = get_histogram_for_subset(ball_number, centers)
        if not _check_balls_are_hit(centers):
            raise Exception("Given centers do not hit all critical balls.")

        if iter > max_iter: break
        iter = iter + 1

        # print("-------iteration = {}".format(iter))

        def _single_swap():
            nonlocal updated_sln
            nonlocal centers
            # For the choice of the new center new_c
            for new_c in range(nr_points):
                # print("Looking to swap in point {}".format(new_c))
                # Running cost of swapping new_c with each of the current centers
                swap_cost_to_power = np.array([0] * n_clusters)

                # For all points, compute the connection cost of swapping in new_c with each one of the current centers
                for p in range(0, nr_points):
                    # print("     for point p = {}".format(p))
                    new_dist = all_pair_distances[p][new_c]
                    # Initially assume p goes to either one of the new centers or the original center it was assigned to, whichever is closer
                    connection_dist = np.array([min(new_dist, dist_vec[p])] * n_clusters)
                    # Unless if/when we swap out the original center c, then p has to choose between sub_c and new_c
                    sub_c = centers[second_closest_center[p]]
                    connection_dist[closest_center[p]] = min(all_pair_distances[p][sub_c], new_dist)
                    # print("overall its dist vector would be {}".format(connection_cost))
                    swap_cost_to_power = np.add(swap_cost_to_power, np.power(connection_dist, power))

                # If we are supposed to be feasible with respect to some critical balls
                if nr_balls > 0:
                    for c in range(n_clusters):
                        b = ball_number[centers[c]]
                        # If a center intersects a ball, and that ball has only this one center, and the new centers don't hit that ball
                        if b > -1 and cnt_in_ball[b] < 2 and ball_number[new_c] != b:
                            # Forget about swapping with this center. The cost remains the same
                            # print("Cannot swap out for center {} in ball {}".format(c, b))
                            swap_cost_to_power[c] = cost_to_power
                # Find the center for which the swapping cost of new_c is minimum
                # print("---- swap cost is = {}".format(swap_cost))
                new_cost_to_power, c = min(
                    (new_cost_to_power, c) for (c, new_cost_to_power) in enumerate(swap_cost_to_power))

                # print("---- swap cost is = {} so we might swap out with c = {} for a new cost of = {}".format(swap_cost,c,new_cost))
                # Check if this new_c is good for substitution
                if new_cost_to_power < (1 - epsilon) * cost_to_power:
                    centers[c] = new_c
                    # print("---- swapped! New centers are {}".format(centers))
                    updated_sln = True
                    # Break the loop to allow for iter to be incremented and allow for smaller improvements
                    break

        def _multi_swap():
            nonlocal updated_sln
            nonlocal centers
            # For all possible new centers, choices of <nr_swaps> points from the points
            # Old centers might be included. This accounts for considering swaps of length at most <nr_swaps>
            for new_centers in itertools.combinations(range(nr_points), nr_swaps):
                if updated_sln: break
                # old_center_inds are the indices of current centers that we are swapping out
                for old_center_inds in itertools.combinations(range(n_clusters), nr_swaps):
                    # print("New centers are {} current centers are {} old center inds are {}".format(new_centers, centers, old_center_inds))
                    # Keep the index of the points that are going to be the new centers
                    to_be_centers = list(new_centers)
                    for i in range(n_clusters):
                        if i not in old_center_inds:
                            to_be_centers.append(centers[i])
                    # to_be_centers = np.unique(to_be_centers) # remember some centers are originally duplicates

                    if len(to_be_centers) != n_clusters:
                        raise Exception(
                            "Ended up with {} centers but expected {} centers. New_centers = {}, Old_center_inds{}, centers = {}".format(
                                len(to_be_centers), n_clusters, new_centers, old_center_inds, centers))

                    if not _check_balls_are_hit(to_be_centers):
                        # print("Some ball would be left empty")
                        continue

                    new_closest, new_dist_vec, _ = get_center_assignments(nr_points, all_pair_distances, to_be_centers)
                    new_cost = np.sum(np.power(new_dist_vec, power))

                    # print("---- swap cost is = {} so we might swap out with c = {} for a new cost of = {}".format(swap_cost,c,new_cost))
                    # Check if this new_c is good for substitution
                    if new_cost < (1 - epsilon) * cost_to_power:
                        centers = to_be_centers
                        # print("---- swapped! New centers are {}".format(centers))
                        updated_sln = True
                        # Break the loop to allow for iter to be incremented and allow for smaller improvements
                        break

        if nr_swaps == 1:
            _single_swap()
        else:
            _multi_swap()

    # print("Number of iterations had to do = {}".format(iter))
    return centers, closest_center, dist_vec, np.power(cost_to_power, 1 / power)


# Just the filtering part of Plesnik, and Hochbaum and Shmoys
# Input:
#   all_pair_distances: the metric. At index [i][j] is the distance between point i and j
#   radii: vector of size nr_points, ith element is a radius around point i
#   dilation: algorithm parameter
# Output:
#   centers: array of point indices corresponding to centers
#   partition: list of the same size as centers. At index i, gives a list of points in the partition of center i
def filter(all_pair_distances, radii, dilation):
    # Indices of points sorted in increasing order of radii
    sorted_inds = np.argsort(radii).tolist()
    centers = []
    partition = []
    # print("Point indices in increasing order of radii = {}".format(sorted_inds))

    while len(sorted_inds) > 0:
        # The actual index of the point:
        u = sorted_inds[0]
        centers.append(u)
        # print("Looking at u = {}".format(u))
        # The partition correponding to this u (includes u itself)
        children = [u]
        # The points in sorted_inds that do not end up in children, in the same sorted order as sorted_inds
        not_in_children = []
        for j in range(1, len(sorted_inds)):
            v = sorted_inds[j]
            # print("     For v = {} dist = {} and radius is = {}".format(v,all_pair_distance[u][v],radii[v]))
            if all_pair_distances[u][v] <= dilation * radii[v]:
                children.append(v)
            else:
                not_in_children.append(v)

        # Only keep the points that do not appear in children[u]
        sorted_inds = not_in_children
        # print("List after removing the children = {}".format(sorted_inds))
        partition.append(children)

    return centers, partition


# Gonzales's greedy algorithm for k-center
# Input:
#   nr_points: the total number of points
#   all_pair_distances: the metric. At index [i][j] is the distance between point i and j
#   n_clusters: the number of clusters
#   centers: Initial set of centers. An array of point indices of size < n_clusters
#   critical_balls: an array of size <= n_clusters of tuples. Tuple (c,r) indicates a ball of radius r around point index c and if given, the algorithm would ensure to contain at least a center in each ball. These balls are supposed to be disjoint
# Output:
#   centers: an array of length <n_clusters> of point indices picked as centers
def gonzales(nr_points, all_pair_distances, n_clusters, centers=None):
    if centers is None:
        centers = [random.sample(range(nr_points))]
    elif len(centers) >= n_clusters:
        raise Exception(
            "Initial centers has size {} >= n_clusters = {}".format(len(centers), n_clusters))

    closest, distances, _ = get_center_assignments(nr_points, all_pair_distances, centers)
    while len(centers) < n_clusters:
        new_center = int(np.argmax(distances))
        centers.append(new_center)
        new_distances = np.array([all_pair_distances[new_center][q] for q in range(nr_points)])
        distances = np.minimum(distances, new_distances)

    return centers


# Mahabadi and Vakilian's algorithm
# Input:
#   nr_points: the total number of points
#   all_pair_distances: the metric. At index [i][j] is the distance between point i and j
#   n_clusters: the number of clusters
#   power: A number >= 1 the moment of the distances in the objective function. e.g. p=1 for k-med
#   radii: vector of size nr_points, ith element is point i's distance it's lth nearest neighbor
#   dilation: algo parameter passed down to filter, initial value is set according to their paper
#   ratio: algo parameter used for deciding the radius of the critical balls
#   epsilon: algo parameter passed down to arya_etal
#   iterations: algo parameter passed down to arya_etal
#   nr_swaps: algo parameter passed down to arya_etal. It is set to 1 in their experimental setting and 4 in their theoretical setting
# Output:
#   dictionary that contains various attributes of the solution
def mahabadi_vakilian(nr_points, all_pair_distances, n_clusters, power, radii, dilation=3, ratio=2, epsilon=0.02,
                      iterations=1000, nr_swaps=1):
    output = {}
    time1 = time.monotonic()
    centers, partition = filter(all_pair_distances, radii, dilation)
    # print("Filter centers = {} and Partition = {}".format(centers,partition))
    output["filter_centers"] = centers
    output["filter_partition"] = partition
    critical_balls = [(c, ratio * radii[c]) for c in centers]
    # print("Hence the critical balls are: {}".format(critical_balls))
    output["critical_balls"] = critical_balls
    centers = gonzales(nr_points, all_pair_distances, n_clusters, centers)
    # print("Gonzales centers = {}".format(centers))
    output["gonzales_centers"] = centers
    centers, _, _, _ = arya_etal(nr_points, all_pair_distances, n_clusters, power, epsilon, iterations, centers,
                                 critical_balls,
                                 nr_swaps)
    output["centers"] = centers
    time2 = time.monotonic()
    output["time"] = time2 - time1

    dist_vec, radii_violations, cost, max_violation, nr_fair = get_clustering_stats(nr_points, all_pair_distances,
                                                                                    radii,
                                                                                    centers, power)
    output["dist_vec"] = dist_vec
    output["radii_violations"] = radii_violations.tolist()
    output["cost"] = cost
    output["max_violation"] = max_violation
    output["nr_fair"] = nr_fair
    return output


# Fair Round algorithm
# Input:
#   nr_points: the total number of points
#   all_pair_distances: the metric. At index [i][j] is the distance between point i and j
#   n_clusters: the number of clusters
#   power: A number >= 1 the moment of the distances in the objective function. e.g. p=1 for k-med
#   radii: vector of size nr_points, ith element is point i's distance it's lth nearest neighbor
#   lp_y: vector of size <nr_points> fractional openings from the LP
#   lp_cost_shares: vector of size <nr_points> cost shares towards LP objective to the power of p.
#   dilation: algo parameter passed down to filter, initial value set according to their paper
#   ratio: algo parameter used for calculating radii passed down to filter, initial value set according to their paper
#   do_binsearch: Set to True if you'd like to find the value of ratio and dilation with binary search in intervals [0, <ratio>] and [0, <dilation>] respectively
#   n_filter_centers: Upperbound on the initial Filter centers you'd like to see. By default should be <nr_clusters> but anything up to 2<nr_clusters> is acceptable
# Output:
#   centers: array of length at most nr_clusters of point indices picked as centers
def fair_round(nr_points, all_pair_distances, nr_clusters, power, radii, lp_y, lp_cost_shares, dilation=2, ratio=2,
               do_binsearch=True):
    if ratio * dilation < 2:
        Warning("ratio x dilation < 2, Cannot guarantee y's are at least 0.5 after smudge 1")

    # Start the timer. Timing the whole function
    time1 = time.monotonic()
    output = {}

    # print("----- neighborhood radii, cost_shares {}".format([(radii[v], lp_cost_shares[v]) for v in range(nr_points)]))
    filter_radii = np.minimum(radii, np.power(np.multiply(lp_cost_shares, ratio), 1 / power))
    S, D = filter(all_pair_distances, filter_radii, dilation)
    print("----- Filter Centers ({} many) = {}".format(len(S), S))
    output["filter_centers"] = S
    output["nr_filter_centers"] = len(S)

    # output["filter_partitions"] = D

    # subroutine to stop the timer and fill out the output whenever we finish (there are many exit points for this algo)
    def _fill_output(centers):
        nonlocal output
        output["centers"] = centers
        time2 = time.monotonic()
        output["time"] = time2 - time1

        output["num_centers"] = len(centers)
        dist_vec, radii_violations, cost, max_violation, nr_fair = get_clustering_stats(nr_points, all_pair_distances,
                                                                                        radii,
                                                                                        centers, power)
        output["dist_vec"] = dist_vec
        output["radii_violations"] = radii_violations.tolist()
        output["cost"] = cost
        output["max_violation"] = max_violation.tolist()
        output["nr_fair"] = nr_fair
        return output

    def _bin_search():
        ratio_min = 0
        ratio_max = ratio

        while ratio_max - ratio_min > 1e-6:
            new_ratio = (ratio_max + ratio_min) / 2
            R = np.minimum(radii, np.power(np.multiply(lp_cost_shares, new_ratio), 1 / power))
            new_centers, _ = filter(all_pair_distances, R, dilation)

            # If found feasible solution, try to decrease dilation to get better cost
            if len(new_centers) <= nr_clusters:
                ratio_max = new_ratio
            else:
                ratio_min = new_ratio

        R = np.minimum(radii, np.power(np.multiply(lp_cost_shares, ratio_max), 1 / power))
        new_centers, _ = filter(all_pair_distances, R, dilation)
        # new_centers, _, binsearch_dilation = _bin_search_on_filter(0, dilation, all_pair_distances, new_centers, R)

        return new_centers, ratio_max  # , binsearch_dilation

    # If we don't have too many clusters, no need to round.
    if len(S) <= nr_clusters:
        if do_binsearch:
            S, binsearch_ratio = _bin_search()
            print("----- binsearch Filter Centers ({} many) = {}".format(len(S), S))
            output["binsearch_ratio"] = binsearch_ratio
        if len(S) <= nr_clusters:
            return _fill_output(S)

    closest, dist_vec, second_closest = get_center_assignments(nr_points, all_pair_distances, S)

    # print("-----Assignment to Fitler centers:")
    # print([(u, S[closest[u]]) for u in range(nr_points)])

    # The first smudging procedure: Take all y values from points outside of S to the closest point in S
    # Maintain 0 <= y_u <= 1, y values are only non-zero over elements in S
    def _smudge_1():
        y = lp_y
        # We only keep track of y values over memebers of S. Initialize.
        y_S = [y[c] for c in S]
        # print("Y values are = {}".format(y))
        # print("Sum of y values is {}".format(sum(y)))
        # For each point v, give away its y value to the closest point in S
        for v in range(nr_points):
            # i is the closest center to v
            i = closest[v]
            # If v is a center itself, it doesn't give its y to anyone
            if v == S[i]: continue
            # v gives its y value to the ith center
            y_S[i] = y_S[i] + y[v]

        excess = 0
        for i in range(len(S)):
            # maintain y <= 1
            if y_S[i] > 1:
                excess = excess + (y_S[i] - 1)
                y_S[i] = 1
                continue

        # Smudge the excess y on the rest
        for i in range(len(S)):
            if excess <= 0: break
            if y_S[i] < 1 - 1e-6:  # and y_S[i] > 0:
                given = min(1 - y_S[i], excess)
                y_S[i] = y_S[i] + given
                excess = excess - given

        sum_y_S = sum(y_S)
        # print("Sum of y values after smudging 1 is {}".format(sum_y_S))
        # print("And the y_S vector {}".format(y_S))
        if abs(sum_y_S - nr_clusters) > 0.01:
            write_output(output, "output/failure/", "duh", nr_points, 0, nr_clusters)
            raise Exception(
                "sum of y_S values after smudging 1 is {} which is not equal to nr_clusters = {} and excess is {}".format(
                    sum_y_S, nr_clusters, excess))
        if len([i for i in y_S if (i < 0.5 - 1e-6 or i > 1 + 1e-6)]) > 0:
            write_output(output, "output/failure/", "duh", nr_points, 0, nr_clusters)
            Warning("not all values of y_S are between 0.5 to 1. y_S = {}".format(y_S))
            added = 0
            for i in range(len(y_S)):
                if y_S[i] < 0.5:
                    added = added + (0.5 - y_S[i])
                    y_S[i] = 0.5
            if added >= 1:
                raise Exception("Resulted in adding {} much extra y value".format(added))

        return y_S

    y = _smudge_1()
    output["smudge_1_y"] = y

    # The second smudging procedure: Takes y values of S, given they are all at least 0.5 we take them to {0.5, 1}
    def _smudge_2(y):
        # For each center, what is the cost to redirect all of its partition to the closest center (other than itself)
        redirect_cost = [0] * len(S)
        for i in range(len(S)):
            u = S[i]
            v = S[second_closest[u]]
            redirect_cost[i] = len(D[i]) * (all_pair_distances[u][v] ** power)
            print(
                "center {} goes to {} at dist {} and since D[u] = {} it pays {}".format(u, v, all_pair_distances[u][v],
                                                                                        len(D[i]), redirect_cost[i]))

        # Keep track of members of S in {0.5,1}
        integral = [False] * len(S)
        half_integral = [False] * len(S)
        num_fractional = 0
        for i in range(len(S)):
            if abs(1 - y[i]) < 1e-6:
                integral[i] = True
                y[i] = 1
            elif abs(0.5 - y[i]) < 1e-6:
                half_integral[i] = True
                y[i] = 0.5
            else:
                num_fractional = num_fractional + 1

        # do while still have to round, expected at least one y to be rounded in each iteration
        while num_fractional > 0:
            # print("Entering the while loop!")
            for i in range(len(S)):  # giver
                if half_integral[i] or integral[i]: continue
                for j in range(len(S)):  # taker
                    if half_integral[j] or integral[j]: continue

                    # if i is less costly to redirect than j
                    if redirect_cost[i] < redirect_cost[j]:

                        # i gives some of its y value to j
                        given = min(y[i] - 0.5, 1 - y[j])
                        y[i] = y[i] - given
                        y[j] = y[j] + given

                        if abs(y[j] - 1) < 1e-6:
                            y[j] = 1
                            integral[j] = True
                            num_fractional = num_fractional - 1

                        if abs(y[i] - 0.5) < 1e-6:
                            y[i] = 0.5
                            half_integral[i] = True
                            num_fractional = num_fractional - 1
                            break  # since i wouldn't be able to give anything to anyone

        sum_y = sum(y)
        # print("Sum of y values after smudging 2 is {}".format(sum_y))
        # print("Y's on the S's {}".format(y))
        if abs(sum_y - nr_clusters) > 0.01:
            write_output(output, "output/failure/", "duh", nr_points, 0, nr_clusters)
            raise Exception(
                "sum of y values after smudging 2 is {} which is not equal to nr_clusters = {}".format(sum_y,
                                                                                                       nr_clusters))
        if len([i for i in range(len(S)) if not (integral[i] or half_integral[i])]) > 0:
            write_output(output, "output/failure/", "duh", nr_points, 0, nr_clusters)
            raise Exception("not all values of y are in {0.5,1}. y = {}".format(y))
        return y, integral, half_integral

    y, integral, half_integral = _smudge_2(y)
    output["smudge_2_y"] = y

    # DFS code for half-integrals ----------
    # Labels: 0, not visited. +1 or -1 means visited
    label = [0] * len(S)

    def _dfs(i, lab):
        nonlocal label
        label[i] = lab
        u = S[i]
        for j in range(len(S)):
            v = S[j]
            # If there is an edge between u and v
            if j == second_closest[u] or i == second_closest[v]:
                # But only if j is half integral and not visited
                if half_integral[j] and label[j] == 0:
                    _dfs(j, -lab)

    # Do DFS but only for half integral members of S
    # Alternating between +1 and -1 labels for the roots are better (gives more balanced splits in the end)
    lab = 1
    for i in range(len(S)):
        if half_integral[i] and label[i] == 0:
            _dfs(i, lab)
            lab = -lab

    # Which label is the minority? -1 or +1?
    sign_of_minority = -np.sign(sum(label))
    print("The DFS labels are {} and sign of minority is {}".format(label, sign_of_minority))

    # A center is a member of S that is either integral or minority in DFS
    # The integral ones are these:
    centers = [S[j] for j in range(len(S)) if integral[j]]
    print("Currently, the integral centres are: {} out of {}".format(centers, S))

    # If the number odd-distance and even-distanced points are equal
    if sign_of_minority == 0:
        even_distanced = [S[j] for j in range(len(S)) if label[j] == 1]
        odd_distanced = [S[j] for j in range(len(S)) if label[j] == -1]
        # print("centers with even_distanced centers {} odd_distanced centers {}".format(centers + even_distanced, centers + odd_distanced))

        _, dist_vec_even, _ = get_center_assignments(nr_points, all_pair_distances, centers + even_distanced)
        _, dist_vec_odd, _ = get_center_assignments(nr_points, all_pair_distances, centers + odd_distanced)

        if np.sum(np.power(dist_vec_odd, power)) < np.sum(np.power(dist_vec_even, power)):
            centers = centers + odd_distanced
        else:
            centers = centers + even_distanced
    else:
        # else, add the minority folks to the centers
        centers = centers + [S[j] for j in range(len(S)) if label[j] == sign_of_minority]

    print("After all this, finally the centers are {}".format(centers))

    return _fill_output(centers)


# Function to sparsify the instance, solve LP, and project results back to initial instance
# Input:
#   all_pair_distances: the metric. At index [i][j] is the distance between point i and j
#   radii: vector of size nr_points, ith element is point i's distance it's lth nearest neighbor
#   n_clusters: the number of clusters
#   power: A number >= 1 the moment of the distances in the objective function. e.g. p=1 for k-med
#   delta: a number > 0 running Filter with dilation=delta to sparsify the instance
#   lp_method: integer between 0 to 6 for LP solving method. See Cplex documentation
# Output:
#   output: dictionary of LP solving results. Contains fractional y values and cost shares for representatives
#   lp_y, lp_cost_shares: projected fractional openings and cost-shares into the initial instance
def sparsify_and_solve_lp(all_pair_distances, radii, nr_clusters, power, delta, lp_method=0):
    if delta <= 1e-9:
        raise Exception("Delta is {} too tiny or negative".format(delta))

    output = {}
    # Sparsify into reps
    reps, reps_partition = filter(all_pair_distances, radii, delta)
    # print("reps are {}".format(reps))
    # print("which partition the points into {}".format(reps_partition))

    # Compute weights, radii, neighborhood and distances
    reps_weights = [len(reps_partition[i]) for i in range(len(reps))]
    # print("So reps weights are {}".format(reps_weights))

    reps_radii = [radii[i] for i in reps]
    reps_all_pair_distances = [all_pair_distances[i] for i in reps]
    for i in range(len(reps)):
        reps_all_pair_distances[i] = [reps_all_pair_distances[i][j] for j in reps]
    reps_neighborhood = get_client_neighborhood_graph(len(reps), reps_all_pair_distances, reps_radii)
    # print("reps distances len = {}".format(len(reps_all_pair_distances)))
    # print("reps radii are = {}".format(reps_radii))
    # print("hence reps neighborhood is {}".format(reps_neighborhood))

    # Solve the lp
    res = solve_priority_p_k_clustering_lp(reps_weights, reps_radii, reps_neighborhood, reps_all_pair_distances,
                                           nr_clusters, power, lp_method)
    output["sparsified_reps"] = reps
    output["sparsified_partitions"] = reps_partition
    output["sparse_lp_solver_res"] = res

    reps_cost_shares = res["cost_shares"]
    reps_y = res["y"]
    # print("reps_y = {}".format(reps_y))
    # print("reps_cost shares {}".format(reps_cost_shares))

    # Now extend y and cost-shares to the entire instance
    lp_cost_shares = [0] * len(radii)
    lp_y = [0] * len(radii)
    for i in range(len(reps)):
        lp_y[reps[i]] = reps_y[i]
        for j in reps_partition[i]:
            # client j's projected cost_share is the cost_share of its rep plus its dist to the rep
            lp_cost_shares[j] = reps_cost_shares[i] + all_pair_distances[j][reps[i]] ** power
    output["projected_cost_shares"] = lp_cost_shares
    output["projected_y"] = lp_y

    return output, lp_y, lp_cost_shares
