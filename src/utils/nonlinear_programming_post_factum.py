import numpy as np
import pyscipopt as scip


def calculate_mean_and_variance(data):
    n_criteria = len(data)
    if n_criteria == 0:
        return None, None  # Handle empty list
    if n_criteria == 1:
        return data[0], 0  # Standard deviation is 0 for a single element

    mean = sum(data) / n_criteria

    # Calculate the sum of squared differences from the mean
    squared_differences = [(x - mean) ** 2 for x in data]
    sum_squared_differences = sum(squared_differences)

    # Calculate the variance
    variance = sum_squared_differences / n_criteria

    return mean, variance


def calculate_topsis_distances(performances, pis, nis):
    assert len(performances) == len(pis) == len(nis)

    n_criteria = len(performances)
    d_pos = sum((performances[idx] - pis[idx]) ** 2 for idx in range(n_criteria))
    d_neg = sum((performances[idx] - nis[idx]) ** 2 for idx in range(n_criteria))

    return d_pos, d_neg


def nonlinear_post_factum_scip(performances_US, weights, target_R_value, excluded_criteria_indices,
                               constant_WM=False):
    """
    performances_US -- performance vector in utility space US (performances are min-max scaled, all criteria are of a gain type)
    weights -- normalized weights (max of weights == 1)
    target_R -- expected value of R (TOPSIS Closeness Coefficient)
    constant_WM  -- set to True for `Retaining WM` method
    """

    n_criteria = len(performances_US)
    nis = np.zeros(n_criteria)  # negative ideal solution, vector of zeros in VS space
    pis = np.array(weights)  # positive ideal solution, in VS space it is identical to weights vector
    performances_US = np.array(performances_US)
    performances_VS = performances_US * weights  # performances in weighted utility space VS
    target_performances_US = None

    try:
        # Create a new model
        model = scip.Model("Euclidean_Distance_Optimization")
        model.setParam('limits/time', 6)
        model.hideOutput()

        # Define variables
        x = [model.addVar(vtype="C", lb=nis[idx], ub=pis[idx]) for idx in range(n_criteria)]

        # Constraint ensuring achievement of the performance target
        target_d_pos, target_d_neg = calculate_topsis_distances(x, pis, nis)
        model.addCons(target_R_value * (scip.sqrt(target_d_pos) + scip.sqrt(target_d_neg)) <= (scip.sqrt(target_d_neg)),
                      "Target_Constraint")

        # Constraints to disable modifications on specific criteria
        for idx in excluded_criteria_indices:
            model.addCons(abs(x[idx] - performances_VS[idx]) <= 1e-15, f"Exclude_Criterion_{idx}_Constraint")

        if constant_WM:
            # Constraints for `Retaining WM` method
            old_mean, old_variance = calculate_mean_and_variance(performances_VS)
            new_mean, new_variance = calculate_mean_and_variance(x)
            model.addCons(old_mean == new_mean, "Constant_WM_Constraint")

        # Objective function: Minimize Euclidean distance between current_performances and target performances
        # SCIP does not support nonlinear objective functions, so we need to reformulate the problem by moving the objective into a constraint.
        objective_variable = model.addVar(vtype="C", lb=0)
        model.setObjective(objective_variable, sense="minimize")
        objective_variable_constraint = scip.quicksum((x[idx] - performances_VS[idx]) ** 2 for idx in range(n_criteria))
        model.addCons(objective_variable >= objective_variable_constraint)

        # Optimize the model
        model.optimize()

        # Print the solution if found
        if model.getStatus() == "optimal":
            target_performances_VS = np.array([model.getVal(var) for var in x])
            target_performances_US = target_performances_VS / weights

    except Exception as e:
        print(f"An error occurred: {e}")

    return target_performances_US
