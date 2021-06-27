'''
Based on codes by Suman K. Bera in https://github.com/nicolasjulioflores/fair_algorithms_for_clustering
'''
from cplex import Cplex
import numpy as np
import time


# Function to add constraints to the priority (p,k)-clustering LP
# Input:
#   radii: vector of size nr_points, ith element is point i's distance it's lth nearest neighbor
#   neighborhood: a list of lists. for point i, graph[i] is the closest l points to i (including itself)
#   n_clusters: the number of clusters
# Output:
#   rows: coefficients in each row of the coefficients matrix
#   senses: the <= or = or >= of the constraints
#   rhs: the RHS of the constraints
#   names: the constraint names
def prepare_to_add_constraints(radii, neighborhood, nr_clusters):
    nr_points = len(radii)
    rows = []
    rhs = []
    senses = []
    names = []

    def add_sums_to_one_constraints():
        nonlocal rows, rhs, senses, names
        rows.extend(
            [[["x_{}_{}".format(j, i) for i in neighborhood[j]], [1] * len(neighborhood[j])] for j in range(nr_points)])
        rhs.extend([1] * nr_points)
        senses.extend(["G"] * nr_points)
        names.extend(["cover_{}".format(j) for j in range(nr_points)])

    def add_k_fac_constraint():
        nonlocal rows, rhs, senses, names
        rows.append([["y_{}".format(i) for i in range(nr_points)], [1] * nr_points])
        rhs.append(nr_clusters)
        senses.append("L")
        names.append("At_most_k")

    def add_connection_constraints():
        nonlocal rows, rhs, senses, names
        new_constrains = [[["x_{}_{}".format(j, i), "y_{}".format(i)], [1, -1]] for j in range(nr_points) for i in
                          neighborhood[j]]
        nr_new_constraints = len(new_constrains)

        rows.extend(new_constrains)
        rhs.extend([0] * nr_new_constraints)
        senses.extend(["L"] * nr_new_constraints)
        names.extend(["cover_{}_{}".format(j, i) for j in range(nr_points) for i in neighborhood[j]])

    add_sums_to_one_constraints()
    add_k_fac_constraint()
    add_connection_constraints()

    return rows, senses, rhs, names


# Function to add variables to the priority (p,k)-clustering LP
# Input:
#   point_weights: vector of size <nr_points> elements are point weights in the objective
#   radii: vector of size nr_points, ith element is point i's distance it's lth nearest neighbor
#   neighborhood: a list of lists. for point i, graph[i] is the closest l points to i (including itself)
#   all_pair_costs: the metric. At index [i][j] is the cost of connecting points i and j
# Output:
#   objective: coefficients in obj fun
#   lower_bounds: variable lower_bounds
#   upper_bounds: variable upper_bounds
#   variable_names: variable names
def prepare_to_add_variables(point_weights, radii, neighborhood, all_pair_costs):
    nr_points = len(radii)

    # Preparing the y variables
    variable_names = ["y_{}".format(i) for i in range(nr_points)] + ["x_{}_{}".format(j, i) for j in range(nr_points)
                                                                     for i in neighborhood[j]]
    objective = [0 for i in range(nr_points)] + [all_pair_costs[j][i] * point_weights[j] for j in range(nr_points)
                                                 for i in
                                                 neighborhood[j]]

    total_variables = len(objective)
    lower_bounds = [0] * total_variables
    upper_bounds = [1] * total_variables

    return objective, lower_bounds, upper_bounds, variable_names


# LP solver for priority (p,k)-clustering LP
# Input:
#   point_weights: vector of size <nr_points> elements are point weights in the objective
#   radii: vector of size nr_points, ith element is point i's distance it's lth nearest neighbor
#   neighborhood: a list of lists. for point i, graph[i] is the closest l points to i (including itself)
#   all_pair_costs: the metric. At index [i][j] is the cost of connecting points i and j
#   n_clusters: the number of clusters
# Output:
#   problem: the cplex LP problem
#   variable_names: names we defined to address the variables in the LP
def get_priority_p_k_clustering_lp(point_weights, radii, neighborhood, all_pair_costs, nr_clusters):
    # There are primarily five steps:
    # 1. Initiate a model for cplex
    # 2. Declare if it is minimization or maximization problem
    # 3. Add variables to the model. The variables are generally named.
    #    The upper bounds and lower bounds on the range for the variables
    #    are also mentioned at this stage. The coefficient of the objective
    #    functions are also entered at this step
    # 4. Add the constraints to the model. The constraint matrix, denoted by A,
    #    can be added in three ways - row wise, column wise or non-zero entry wise.
    # 5. Finally, call the solver.

    # Step 1. Initiate a model for cplex.

    print("Initializing Cplex model")
    problem = Cplex()

    # Step 2. Declare that this is a minimization problem

    problem.objective.set_sense(problem.objective.sense.minimize)

    # Step 3.   Declare and  add variables to the model. The function
    #           prepare_to_add_variables (points, center) prepares all the
    #           required information for this stage.
    #
    #    objective: a list of coefficients (float) in the linear objective function
    #    lower bounds: a list of floats containing the lower bounds for each variable
    #    upper bounds: a list of floats containing the upper bounds for each variable
    #    variable_name: a list of strings that contains the name of the variables

    print("Starting to add variables...")
    t1 = time.monotonic()
    objective, lower_bounds, upper_bounds, variable_names = prepare_to_add_variables(point_weights, radii, neighborhood,
                                                                                     all_pair_costs)
    problem.variables.add(obj=objective,
                          lb=lower_bounds,
                          ub=upper_bounds,
                          names=variable_names)
    t2 = time.monotonic()
    print("Completed. Time for creating and adding variable = {}".format(t2 - t1))

    # Step 4.   Declare and add constraints to the model.
    #           There are few ways of adding constraints: rwo wise, col wise and non-zero entry wise.
    #           Assume the constraint matrix is A. We add the constraints row wise.
    #           The function prepare_to_add_constraints_by_entry(points,center,colors,alpha,beta)
    #           prepares the required data for this step.
    #
    #  constraints_row: Encoding of each row of the constraint matrix
    #  senses: a list of strings that identifies whether the corresponding constraint is
    #          an equality or inequality. "E" : equals to (=), "L" : less than (<=), "G" : greater than equals (>=)
    #  rhs: a list of floats corresponding to the rhs of the constraints.
    #  constraint_names: a list of string corresponding to the name of the constraint

    print("Starting to add constraints...")
    t1 = time.monotonic()
    objects_returned = prepare_to_add_constraints(radii, neighborhood, nr_clusters)
    constraints_row, senses, rhs, constraint_names = objects_returned
    problem.linear_constraints.add(lin_expr=constraints_row,
                                   senses=senses,
                                   rhs=rhs,
                                   names=constraint_names)
    t2 = time.monotonic()
    print("Completed. Time for creating and adding constraints = {}".format(t2 - t1))

    # Optional: We can set various parameters to optimize the performance of the lp solver
    # As an example, the following sets barrier method as the lp solving method
    # Available methods are: auto, primal, dual, sifting, concurrent, and barrier
    return problem, variable_names


# LP solver for priority (p,k)-Clustering LP
# Input:
#   point_weights: vector of size <nr_points> elements are point weights in the objective
#   radii: vector of size <nr_points>, ith element is point i's distance it's lth nearest neighbor
#   neighborhood: a list of lists. for point i, graph[i] is the closest l points to i (including itself)
#   all_pair_distances: the metric. At index [i][j] is the distance between point i and j
#   n_clusters: the number of clusters
#   power: A number >= 1 the moment of the distances in the objective function. e.g. p=1 for k-med
#   lp_method: integer between 0 to 6 for LP solving method. See Cplex documentation
# Output:
#   res: dictionary of LP solving results. Contains fractional y values and cost shares for points
def solve_priority_p_k_clustering_lp(point_weights, radii, neighborhood, all_pair_distances, nr_clusters, power,
                                     lp_method=0):
    t1 = time.monotonic()
    problem, variable_names = get_priority_p_k_clustering_lp(point_weights, radii, neighborhood,
                                                             np.power(all_pair_distances, power), nr_clusters)
    t2 = time.monotonic()
    lp_defining_time = t2 - t1

    t1 = time.monotonic()
    problem.parameters.lpmethod.set(lp_method)
    problem.solve()
    t2 = time.monotonic()
    lp_solving_time = t2 - t1

    # problem.solution is a weakly referenced object, so we must save its data
    #   in a dictionary so we can write it to a file later.
    values = problem.solution.get_values()

    nr_points = len(radii)
    y_values = values[0:nr_points]
    x_values = values[nr_points:]

    # --------- STUFF FOR LOGGING PURPOSE ---------
    # threshold = 0.01
    # y_names = variable_names[0:nr_points]
    x_names = variable_names[nr_points:]
    # fractional_openings = dict(zip(y_names, y_values))
    # # print("-----------openings-------------")
    # # print(fractional_openings)
    # fractional_openings = {key: val for key, val in fractional_openings.items() if val > 1e-5}
    # # print("-----------non-zero openings-------------")
    # # print(fractional_openings)
    # open_facilities = [int(key[2:]) for key, val in fractional_openings.items() if val > (1-threshold)]
    # print("-----------open facilities-------------")
    # print(open_facilities)
    #
    fractional_assignments = dict(zip(x_names, x_values))
    # # print("-----------assignments-------------")
    # # print(fractional_assignments)
    # fractional_assignments = {key:val for key, val in fractional_assignments.items() if val > threshold}
    # # print("-----------non-zero assignments-------------")
    # # print(fractional_assignments)
    # integral_assignments = []
    # non_zero_x_values = []
    cost_shares = [0] * nr_points
    for (name, val) in fractional_assignments.items():
        name_split = name.split('_')
        u = int(name_split[1])
        v = int(name_split[2])
        # non_zero_x_values.append(((u,v), val))
        cost_shares[u] = cost_shares[u] + (all_pair_distances[u][v] ** power) * val

    # if abs(np.sum(cost_shares) - problem.solution.get_objective_value()) > 0.01:
    #     raise Exception("Mismatch between sum of cost_shares {} and LP objective {}".format(np.sum(cost_shares),
    #                                                                                         problem.solution.get_objective_value()))

    # new_cost_shares = [0] * nr_points
    # for p in range(nr_points):
    #     connection_costs = np.array([all_pair_costs[p][c] for c in range(nr_points)])
    #     sorted_inds = np.argsort(connection_costs)
    #
    #     coverage = 0
    #     for i in range(nr_points):
    #         q = int(sorted_inds[i])
    #         gives_y = min(y_values[q], 1-coverage)
    #         new_cost_shares[p] = new_cost_shares[p] + gives_y * all_pair_costs[p][q]
    #         if coverage >= 1: break
    #
    #     if abs(new_cost_shares[p] - cost_shares[p]) > 1e-4:
    #         Warning("For points {}, new cost share is {} while LP cost share is {}".format(p, new_cost_shares[p], cost_shares[p]))

    #     if fractional_assignments[name] > (1-threshold):
    #         integral_assignments.append((u,v))
    # print("Sum of cost shares is  {} and the total objective reported is {}".format(np.sum(cost_shares), problem.solution.get_objective_value()))
    #
    # # print("-----------integral assignments-------------")
    # # print(integral_assignments)
    # # print("Number of fractional assignments: {} - {} = {}".format(nr_points, len(integral_assignments), nr_points-len(integral_assignments)))
    # print("Number of integrally open facilitie = {}".format(len(open_facilities)))

    res = {
        "lp_defining_time": lp_defining_time,
        "lp_solving_time": lp_solving_time,
        "time": lp_solving_time + lp_defining_time,
        "status": problem.solution.get_status(),
        "success": problem.solution.get_status_string(),
        "cost": np.power(problem.solution.get_objective_value(), 1 / power),
        # "fractional_openings":fractional_openings,
        # "fractional_assignments":fractional_assignments,
        # "open_facilities":open_facilities,
        # "integral_assignments":integral_assignments,
        "nr_x_ij_variables": len(x_names),
        "cost_shares": cost_shares,
        "y": y_values

    }
    return res
