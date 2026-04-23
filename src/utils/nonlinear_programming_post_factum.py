import numpy as np
import pyscipopt as scip


DEFAULT_TIME_LIMIT = 6


def calculate_mean_and_variance(data):
    n_criteria = len(data)
    if n_criteria == 0:
        return None, None
    if n_criteria == 1:
        return data[0], 0

    mean = sum(data) / n_criteria
    squared_differences = [(x - mean) ** 2 for x in data]
    variance = sum(squared_differences) / n_criteria

    return mean, variance


def calculate_topsis_distances(performances, pis, nis):
    assert len(performances) == len(pis) == len(nis)

    n_criteria = len(performances)
    d_pos = sum((performances[idx] - pis[idx]) ** 2 for idx in range(n_criteria))
    d_neg = sum((performances[idx] - nis[idx]) ** 2 for idx in range(n_criteria))

    return d_pos, d_neg


class BaseNLPPostFactum:
    def __init__(
        self,
        current_vector,
        lower_bounds,
        upper_bounds,
        excluded_criteria_indices,
        constant_wm_coefficients=None,
        constant_wm_target=None,
        time_limit=DEFAULT_TIME_LIMIT,
    ):
        self.current_vector = np.asarray(current_vector, dtype=float)
        self.lower_bounds = np.asarray(lower_bounds, dtype=float)
        self.upper_bounds = np.asarray(upper_bounds, dtype=float)
        self.excluded_criteria_indices = list(excluded_criteria_indices or [])
        self.constant_wm_coefficients = (
            None
            if constant_wm_coefficients is None
            else np.asarray(constant_wm_coefficients, dtype=float)
        )
        self.constant_wm_target = (
            None if constant_wm_target is None else float(constant_wm_target)
        )
        self.time_limit = time_limit

        if not (
            len(self.current_vector)
            == len(self.lower_bounds)
            == len(self.upper_bounds)
        ):
            raise ValueError("Current vector and bounds must have the same length.")

    def add_target_constraint(self, model, x):
        raise NotImplementedError

    def extract_solution(self, solution_vector):
        return np.asarray(solution_vector, dtype=float)

    def build_model(self):
        model = scip.Model("PostFactumNLP")
        model.setParam("limits/time", self.time_limit)
        model.hideOutput()

        x = [
            model.addVar(
                vtype="C",
                lb=float(self.lower_bounds[idx]),
                ub=float(self.upper_bounds[idx]),
                name=f"x_{idx}",
            )
            for idx in range(len(self.current_vector))
        ]

        for idx in self.excluded_criteria_indices:
            model.addCons(
                x[idx] == float(self.current_vector[idx]),
                f"Exclude_Criterion_{idx}_Constraint",
            )

        if self.constant_wm_coefficients is not None:
            weighted_mean_expr = (
                scip.quicksum(
                    float(coeff) * x[idx]
                    for idx, coeff in enumerate(self.constant_wm_coefficients)
                )
                / len(x)
            )
            model.addCons(
                weighted_mean_expr == float(self.constant_wm_target),
                "Constant_WM_Constraint",
            )

        self.add_target_constraint(model, x)

        objective_variable = model.addVar(vtype="C", lb=0, name="objective")
        model.setObjective(objective_variable, sense="minimize")
        objective_expr = scip.quicksum(
            (x[idx] - float(self.current_vector[idx])) ** 2
            for idx in range(len(self.current_vector))
        )
        model.addCons(objective_variable >= objective_expr, "Objective_Constraint")

        return model, x

    def solve(self):
        model, x = self.build_model()
        model.optimize()

        if model.getStatus() != "optimal":
            return None

        solution = np.array([model.getVal(var) for var in x], dtype=float)
        return self.extract_solution(solution)


class UtilitySpaceNLPPostFactum(BaseNLPPostFactum):
    def __init__(
        self,
        performances_US,
        weights,
        target_score,
        excluded_criteria_indices,
        upper_bounds_US=None,
        constant_WM=False,
        time_limit=DEFAULT_TIME_LIMIT,
    ):
        self.performances_US = np.asarray(performances_US, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.target_score = float(target_score)

        if upper_bounds_US is None:
            upper_bounds_US = np.ones_like(self.performances_US, dtype=float)

        constant_wm_coefficients = None
        constant_wm_target = None
        if constant_WM:
            constant_wm_coefficients = self.weights
            constant_wm_target = np.mean(self.weights * self.performances_US)

        super().__init__(
            current_vector=self.performances_US,
            lower_bounds=np.zeros_like(self.performances_US, dtype=float),
            upper_bounds=np.asarray(upper_bounds_US, dtype=float),
            excluded_criteria_indices=excluded_criteria_indices,
            constant_wm_coefficients=constant_wm_coefficients,
            constant_wm_target=constant_wm_target,
            time_limit=time_limit,
        )

    def build_score_expression(self, x):
        raise NotImplementedError

    def add_target_constraint(self, model, x):
        model.addCons(
            self.build_score_expression(x) >= float(self.target_score),
            "Target_Constraint",
        )


class TopsisNLPPostFactum(BaseNLPPostFactum):
    def __init__(
        self,
        performances_US,
        weights,
        target_R_value,
        excluded_criteria_indices,
        upper_bounds_US=None,
        constant_WM=False,
        time_limit=DEFAULT_TIME_LIMIT,
    ):
        self.performances_US = np.asarray(performances_US, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.target_R_value = float(target_R_value)

        if upper_bounds_US is None:
            upper_bounds_US = np.ones_like(self.performances_US, dtype=float)

        performances_VS = self.performances_US * self.weights
        upper_bounds_VS = np.asarray(upper_bounds_US, dtype=float) * self.weights

        constant_wm_coefficients = None
        constant_wm_target = None
        if constant_WM:
            constant_wm_coefficients = np.ones_like(self.weights, dtype=float)
            constant_wm_target = np.mean(performances_VS)

        super().__init__(
            current_vector=performances_VS,
            lower_bounds=np.zeros_like(performances_VS, dtype=float),
            upper_bounds=upper_bounds_VS,
            excluded_criteria_indices=excluded_criteria_indices,
            constant_wm_coefficients=constant_wm_coefficients,
            constant_wm_target=constant_wm_target,
            time_limit=time_limit,
        )

    def add_target_constraint(self, model, x):
        nis = np.zeros(len(self.weights), dtype=float)
        target_d_pos, target_d_neg = calculate_topsis_distances(x, self.weights, nis)
        model.addCons(
            self.target_R_value
            * (scip.sqrt(target_d_pos) + scip.sqrt(target_d_neg))
            <= scip.sqrt(target_d_neg),
            "Target_Constraint",
        )

    def extract_solution(self, solution_vector):
        result = self.performances_US.copy()
        positive_weights = self.weights > 0
        result[positive_weights] = (
            solution_vector[positive_weights] / self.weights[positive_weights]
        )
        return result


class SAWNLPPostFactum(UtilitySpaceNLPPostFactum):
    def build_score_expression(self, x):
        return scip.quicksum(
            float(weight) * x[idx] for idx, weight in enumerate(self.weights)
        )


class ARASNLPPostFactum(UtilitySpaceNLPPostFactum):
    def build_score_expression(self, x):
        weight_sum = float(np.sum(self.weights))
        if weight_sum <= 0:
            return scip.quicksum(0.0 * var for var in x)
        normalized_weights = self.weights / weight_sum
        return scip.quicksum(
            float(weight) * x[idx] for idx, weight in enumerate(normalized_weights)
        )


class COPRASNLPPostFactum(UtilitySpaceNLPPostFactum):
    def __init__(
        self,
        performances_US,
        weights,
        objectives,
        target_score,
        excluded_criteria_indices,
        upper_bounds_US=None,
        constant_WM=False,
        time_limit=DEFAULT_TIME_LIMIT,
    ):
        self.objectives = np.asarray(objectives)
        super().__init__(
            performances_US=performances_US,
            weights=weights,
            target_score=target_score,
            excluded_criteria_indices=excluded_criteria_indices,
            upper_bounds_US=upper_bounds_US,
            constant_WM=constant_WM,
            time_limit=time_limit,
        )

    def add_target_constraint(self, model, x):
        gain_indices = np.where(self.objectives == "max")[0]
        cost_indices = np.where(self.objectives == "min")[0]

        sp = scip.quicksum(float(self.weights[idx]) * x[idx] for idx in gain_indices)
        if len(cost_indices) == 0:
            model.addCons(sp >= float(self.target_score), "Target_Constraint")
            return

        sm = scip.quicksum(
            float(self.weights[idx]) * (1 - x[idx]) for idx in cost_indices
        )
        model.addCons(
            sp >= float(self.target_score) * sm,
            "Target_Constraint",
        )


class WASPASNLPPostFactum(UtilitySpaceNLPPostFactum):
    def __init__(
        self,
        performances_US,
        weights,
        target_score,
        excluded_criteria_indices,
        upper_bounds_US=None,
        constant_WM=False,
        lam=0.5,
        time_limit=DEFAULT_TIME_LIMIT,
    ):
        self.lam = float(lam)
        super().__init__(
            performances_US=performances_US,
            weights=weights,
            target_score=target_score,
            excluded_criteria_indices=excluded_criteria_indices,
            upper_bounds_US=upper_bounds_US,
            constant_WM=constant_WM,
            time_limit=time_limit,
        )

    def build_score_expression(self, x):
        weight_sum = float(np.sum(self.weights))
        if weight_sum <= 0:
            return scip.quicksum(0.0 * var for var in x)

        normalized_weights = self.weights / weight_sum
        q_sum = scip.quicksum(
            float(weight) * x[idx] for idx, weight in enumerate(normalized_weights)
        )
        q_prod = None
        for idx, weight in enumerate(normalized_weights):
            if weight <= 0:
                continue
            term = x[idx] ** float(weight)
            q_prod = term if q_prod is None else q_prod * term

        if q_prod is None:
            q_prod = scip.quicksum(0.0 * var for var in x)

        return self.lam * q_sum + (1 - self.lam) * q_prod


class VIKORNLPPostFactum(BaseNLPPostFactum):
    def __init__(
        self,
        performances_US,
        target_score,
        excluded_criteria_indices,
        weights,
        fstar,
        fminus,
        sstar,
        sminus,
        rstar,
        rminus,
        v=0.5,
        upper_bounds_US=None,
        constant_WM=False,
        time_limit=DEFAULT_TIME_LIMIT,
    ):
        self.performances_US = np.asarray(performances_US, dtype=float)
        self.target_score = float(target_score)
        self.weights = np.asarray(weights, dtype=float)
        self.fstar = np.asarray(fstar, dtype=float)
        self.fminus = np.asarray(fminus, dtype=float)
        self.sstar = float(sstar)
        self.sminus = float(sminus)
        self.rstar = float(rstar)
        self.rminus = float(rminus)
        self.v = float(v)

        if upper_bounds_US is None:
            upper_bounds_US = np.ones_like(self.performances_US, dtype=float)

        constant_wm_coefficients = None
        constant_wm_target = None
        if constant_WM:
            constant_wm_coefficients = self.weights
            constant_wm_target = np.mean(self.weights * self.performances_US)

        super().__init__(
            current_vector=self.performances_US,
            lower_bounds=np.zeros_like(self.performances_US, dtype=float),
            upper_bounds=np.asarray(upper_bounds_US, dtype=float),
            excluded_criteria_indices=excluded_criteria_indices,
            constant_wm_coefficients=constant_wm_coefficients,
            constant_wm_target=constant_wm_target,
            time_limit=time_limit,
        )

    def build_model(self):
        model, x = super().build_model()
        return model, x

    def add_target_constraint(self, model, x):
        weighted_terms = []
        for idx, weight in enumerate(self.weights):
            denominator = float(self.fstar[idx] - self.fminus[idx])
            if np.isclose(denominator, 0.0):
                raise ValueError(
                    "VIKOR exact NLP cannot be applied when a criterion has zero fitted range."
                )
            weighted_terms.append(
                float(weight) * (float(self.fstar[idx]) - x[idx]) / denominator
            )

        t = model.addVar(vtype="C", lb=0.0, ub=1.0, name="vikor_regret")
        for idx, term in enumerate(weighted_terms):
            model.addCons(t >= term, f"VIKOR_Regret_{idx}_Constraint")

        s_expr = scip.quicksum(weighted_terms)
        q_expr = 0.0
        if np.isclose(self.sminus, self.sstar):
            q_expr += 0.0
        else:
            q_expr += self.v * (s_expr - self.sstar) / (self.sminus - self.sstar)

        if np.isclose(self.rminus, self.rstar):
            q_expr += 0.0
        else:
            q_expr += (1 - self.v) * (t - self.rstar) / (self.rminus - self.rstar)

        target_q = 1.0 - float(self.target_score)
        model.addCons(q_expr <= target_q, "Target_Constraint")


class FuzzyTOPSISNLPPostFactum(BaseNLPPostFactum):
    def __init__(
        self,
        current_matrix,
        target_score,
        lower_bounds_matrix,
        upper_bounds_matrix,
        excluded_criteria_indices,
        weights,
        objectives,
        gain_scales,
        cost_scales,
        fpis,
        fnis,
        time_limit=DEFAULT_TIME_LIMIT,
    ):
        self.current_matrix = np.asarray(current_matrix, dtype=float)
        self.target_score = float(target_score)
        self.weights = np.asarray(weights, dtype=float)
        self.objectives = np.asarray(objectives)
        self.gain_scales = np.asarray(gain_scales, dtype=float)
        self.cost_scales = np.asarray(cost_scales, dtype=float)
        self.fpis = np.asarray(fpis, dtype=float)
        self.fnis = np.asarray(fnis, dtype=float)

        if self.current_matrix.ndim != 2 or self.current_matrix.shape[1] != 3:
            raise ValueError("current_matrix must be a 2D array of triangular fuzzy numbers.")

        flattened_excluded = []
        for idx in excluded_criteria_indices:
            flattened_excluded.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])

        super().__init__(
            current_vector=self.current_matrix.reshape(-1),
            lower_bounds=np.asarray(lower_bounds_matrix, dtype=float).reshape(-1),
            upper_bounds=np.asarray(upper_bounds_matrix, dtype=float).reshape(-1),
            excluded_criteria_indices=flattened_excluded,
            time_limit=time_limit,
        )

    def build_model(self):
        model, x = super().build_model()
        n_criteria = self.current_matrix.shape[0]
        for idx in range(n_criteria):
            x_l, x_m, x_u = x[3 * idx : 3 * idx + 3]
            model.addCons(x_l <= x_m, f"Fuzzy_Order_LM_{idx}")
            model.addCons(x_m <= x_u, f"Fuzzy_Order_MU_{idx}")
        return model, x

    def add_target_constraint(self, model, x):
        d_pos_terms = []
        d_neg_terms = []

        for idx, weight in enumerate(self.weights):
            x_l, x_m, x_u = x[3 * idx : 3 * idx + 3]
            if self.objectives[idx] == "max":
                scale = float(self.gain_scales[idx])
                if np.isclose(scale, 0.0):
                    raise ValueError(
                        "Fuzzy TOPSIS exact NLP cannot be applied when a gain criterion has zero fitted scale."
                    )
                weighted_l = float(weight) * x_l / scale
                weighted_m = float(weight) * x_m / scale
                weighted_u = float(weight) * x_u / scale
            else:
                scale = float(self.cost_scales[idx])
                if np.isclose(scale, 0.0):
                    raise ValueError(
                        "Fuzzy TOPSIS exact NLP cannot be applied when a cost criterion has zero fitted scale."
                    )
                weighted_l = float(weight) * scale / x_u
                weighted_m = float(weight) * scale / x_m
                weighted_u = float(weight) * scale / x_l

            pos_sq = (
                (weighted_l - float(self.fpis[idx, 0])) ** 2
                + (weighted_m - float(self.fpis[idx, 1])) ** 2
                + (weighted_u - float(self.fpis[idx, 2])) ** 2
            ) / 3.0
            neg_sq = (
                (weighted_l - float(self.fnis[idx, 0])) ** 2
                + (weighted_m - float(self.fnis[idx, 1])) ** 2
                + (weighted_u - float(self.fnis[idx, 2])) ** 2
            ) / 3.0

            d_pos_terms.append(scip.sqrt(pos_sq))
            d_neg_terms.append(scip.sqrt(neg_sq))

        sum_d_pos = scip.quicksum(d_pos_terms)
        sum_d_neg = scip.quicksum(d_neg_terms)
        model.addCons(
            float(self.target_score) * sum_d_pos
            <= (1.0 - float(self.target_score)) * sum_d_neg,
            "Target_Constraint",
        )

    def extract_solution(self, solution_vector):
        return np.asarray(solution_vector, dtype=float).reshape(self.current_matrix.shape)

    def solve(self):
        model, x = self.build_model()
        model.optimize()

        if model.getStatus() not in {"optimal", "timelimit"} or model.getNSols() == 0:
            return None

        best_solution = model.getBestSol()
        solution = np.array(
            [model.getSolVal(best_solution, var) for var in x],
            dtype=float,
        )
        return self.extract_solution(solution)


def nonlinear_post_factum_scip(
    performances_US,
    weights,
    target_R_value,
    excluded_criteria_indices,
    constant_WM=False,
    upper_bounds_US=None,
):
    solver = TopsisNLPPostFactum(
        performances_US=performances_US,
        weights=weights,
        target_R_value=target_R_value,
        excluded_criteria_indices=excluded_criteria_indices,
        upper_bounds_US=upper_bounds_US,
        constant_WM=constant_WM,
    )
    return solver.solve()
