import numpy as np
import pandas as pd
from IPython.display import display
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from sklearn.base import TransformerMixin

from utils.nonlinear_programming_post_factum import FuzzyTOPSISNLPPostFactum

from .optimization import MyProgressBar

class FuzzyTOPSIS(TransformerMixin):
    """
    Library-only Fuzzy TOPSIS ranker for triangular fuzzy numbers.

    Accepted input formats:
    1. A DataFrame where each criterion cell is a triangular fuzzy number
       represented as a list/tuple/ndarray of length 3: (l, m, u).
    2. A flat DataFrame with columns named `<criterion>_l`, `<criterion>_m`,
       `<criterion>_u`.
    """

    score_column = "F"

    def __init__(self):
        self._isFitted = False

    @staticmethod
    def _coerce_fuzzy_value(value):
        if not isinstance(value, (list, tuple, np.ndarray, pd.Series)) or len(value) != 3:
            raise ValueError(
                "Each fuzzy criterion value must be a triangular fuzzy number with three components: (l, m, u)."
            )
        triple = np.asarray(value, dtype=float)
        if np.isnan(triple).any():
            raise ValueError("Fuzzy criterion values must not contain NaN.")
        if not (triple[0] <= triple[1] <= triple[2]):
            raise ValueError("Triangular fuzzy numbers must satisfy l <= m <= u.")
        return triple

    @staticmethod
    def _normalize_objectives(objectives, criteria):
        if objectives is None:
            normalized = ["max"] * len(criteria)
        elif isinstance(objectives, str):
            normalized = [objectives] * len(criteria)
        elif isinstance(objectives, dict):
            normalized = [objectives[name] for name in criteria]
        elif isinstance(objectives, list):
            normalized = objectives
        else:
            raise ValueError(
                "Invalid value at 'objectives': must be a list, dictionary, string, or None."
            )

        if len(normalized) != len(criteria):
            raise ValueError(
                "Invalid value at 'objectives': length must match the number of criteria."
            )

        normalized = [str(item).lower() for item in normalized]
        normalized = [item.replace("gain", "max").replace("g", "max") for item in normalized]
        normalized = [item.replace("cost", "min").replace("c", "min") for item in normalized]
        if not all(item in {"max", "min"} for item in normalized):
            raise ValueError(
                "Invalid value at 'objectives'. Use 'min', 'max', 'gain', 'cost', 'g' or 'c'."
            )
        return normalized

    @staticmethod
    def _normalize_weights(weights, criteria):
        if weights is None:
            normalized = np.ones(len(criteria), dtype=float)
        elif isinstance(weights, dict):
            normalized = np.asarray([weights[name] for name in criteria], dtype=float)
        else:
            normalized = np.asarray(weights, dtype=float)

        if normalized.shape[0] != len(criteria):
            raise ValueError(
                "Invalid value at 'weights': length must match the number of criteria."
            )
        if np.isnan(normalized).any():
            raise ValueError("Invalid value at 'weights': weights must not contain NaN.")
        if np.any(normalized < 0):
            raise ValueError("Invalid value at 'weights': weights must be non-negative.")
        if not np.any(normalized > 0):
            raise ValueError(
                "Invalid value at 'weights': at least one weight must be positive."
            )
        return normalized / np.sum(normalized)

    @classmethod
    def _coerce_fuzzy_dataframe(cls, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if X.empty:
            raise ValueError("X must not be empty.")

        flat_groups = {}
        flat_column_order = []
        for column in X.columns:
            if isinstance(column, str) and "_" in column:
                base, suffix = column.rsplit("_", 1)
                if suffix in {"l", "m", "u"}:
                    flat_groups.setdefault(base, {})[suffix] = column
                    if base not in flat_column_order:
                        flat_column_order.append(base)

        if flat_groups and len(flat_groups) * 3 == len(X.columns):
            if not all(set(group.keys()) == {"l", "m", "u"} for group in flat_groups.values()):
                raise ValueError(
                    "Flat fuzzy input must provide <criterion>_l, <criterion>_m, and <criterion>_u for every criterion."
                )

            matrix = np.zeros((len(X.index), len(flat_column_order), 3), dtype=float)
            for idx, criterion in enumerate(flat_column_order):
                matrix[:, idx, 0] = X[flat_groups[criterion]["l"]].to_numpy(dtype=float)
                matrix[:, idx, 1] = X[flat_groups[criterion]["m"]].to_numpy(dtype=float)
                matrix[:, idx, 2] = X[flat_groups[criterion]["u"]].to_numpy(dtype=float)
            if np.isnan(matrix).any():
                raise ValueError("Fuzzy criterion values must not contain NaN.")
            if np.any(matrix[:, :, 0] > matrix[:, :, 1]) or np.any(matrix[:, :, 1] > matrix[:, :, 2]):
                raise ValueError("Triangular fuzzy numbers must satisfy l <= m <= u.")
            return matrix, flat_column_order, "flat"

        criterion_names = list(X.columns)
        matrix = np.zeros((len(X.index), len(criterion_names), 3), dtype=float)
        for col_idx, criterion in enumerate(criterion_names):
            triples = [cls._coerce_fuzzy_value(value) for value in X[criterion].tolist()]
            matrix[:, col_idx, :] = np.vstack(triples)
        return matrix, criterion_names, "tuple"

    @staticmethod
    def _vertex_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2, axis=-1) / 3.0)

    @staticmethod
    def _matrix_to_frame(matrix, criterion_names, index, output_format):
        if output_format == "flat":
            data = {}
            for col_idx, criterion in enumerate(criterion_names):
                data[f"{criterion}_l"] = matrix[:, col_idx, 0]
                data[f"{criterion}_m"] = matrix[:, col_idx, 1]
                data[f"{criterion}_u"] = matrix[:, col_idx, 2]
            return pd.DataFrame(data, index=index)

        data = {}
        for col_idx, criterion in enumerate(criterion_names):
            data[criterion] = [tuple(row) for row in matrix[:, col_idx, :]]
        return pd.DataFrame(data, index=index)

    def _score_batch_from_matrix(self, matrix):
        normalized = np.zeros_like(matrix, dtype=float)
        for idx, objective in enumerate(self.objectives):
            if objective == "max":
                scale = self._gain_scales[idx]
                if scale <= 0:
                    raise ValueError(
                        "Gain-type fuzzy criteria must have a positive fitted upper bound."
                    )
                normalized[:, idx, :] = matrix[:, idx, :] / scale
            else:
                scale = self._cost_scales[idx]
                if scale <= 0 or np.any(matrix[:, idx, :] <= 0):
                    raise ValueError(
                        "Cost-type fuzzy criteria must be strictly positive for Fuzzy TOPSIS normalization."
                    )
                normalized[:, idx, 0] = scale / matrix[:, idx, 2]
                normalized[:, idx, 1] = scale / matrix[:, idx, 1]
                normalized[:, idx, 2] = scale / matrix[:, idx, 0]

        weighted = normalized * self.weights.reshape(1, -1, 1)
        d_pos = np.sum(self._vertex_distance(weighted, self.fpis_), axis=1)
        d_neg = np.sum(self._vertex_distance(weighted, self.fnis_), axis=1)
        denom = d_pos + d_neg
        scores = np.divide(
            d_neg,
            denom,
            out=np.zeros_like(d_neg, dtype=float),
            where=denom > 0,
        )
        return weighted, scores

    def _transform_matrix(self, matrix, index):
        weighted, scores = self._score_batch_from_matrix(matrix)
        result = self._matrix_to_frame(weighted, self.criteria_, index, self._input_format)
        result[self.score_column] = scores
        return result

    def fit(self, X, weights=None, objectives=None):
        self.X = X.copy()
        matrix, criteria, input_format = self._coerce_fuzzy_dataframe(X)
        self._matrix = matrix.copy()
        self.criteria_ = criteria
        self._input_format = input_format
        self.weights = self._normalize_weights(weights, criteria)
        self.objectives = self._normalize_objectives(objectives, criteria)
        self._criterion_index = {criterion: idx for idx, criterion in enumerate(criteria)}

        self._gain_scales = np.max(matrix[:, :, 2], axis=0)
        self._cost_scales = np.min(matrix[:, :, 0], axis=0)
        self._raw_min_bounds = np.min(matrix, axis=0)
        self._raw_max_bounds = np.max(matrix, axis=0)
        self._criterion_distance_scales = np.where(
            self._vertex_distance(self._raw_max_bounds, self._raw_min_bounds) > 0,
            self._vertex_distance(self._raw_max_bounds, self._raw_min_bounds),
            1.0,
        )

        if np.any(self._gain_scales <= 0):
            raise ValueError(
                "Fuzzy TOPSIS currently requires positive triangular fuzzy numbers for gain-type criteria."
            )
        for idx, objective in enumerate(self.objectives):
            if objective == "min" and self._cost_scales[idx] <= 0:
                raise ValueError(
                    "Fuzzy TOPSIS currently requires strictly positive triangular fuzzy numbers for cost-type criteria."
                )

        normalized = np.zeros_like(matrix, dtype=float)
        for idx, objective in enumerate(self.objectives):
            if objective == "max":
                normalized[:, idx, :] = matrix[:, idx, :] / self._gain_scales[idx]
            else:
                normalized[:, idx, 0] = self._cost_scales[idx] / matrix[:, idx, 2]
                normalized[:, idx, 1] = self._cost_scales[idx] / matrix[:, idx, 1]
                normalized[:, idx, 2] = self._cost_scales[idx] / matrix[:, idx, 0]

        weighted = normalized * self.weights.reshape(1, -1, 1)
        self.fpis_ = np.max(weighted, axis=0)
        self.fnis_ = np.min(weighted, axis=0)
        self.X_new = self._transform_matrix(matrix, X.index)
        self._ranked_alternatives = (
            self.X_new.sort_values(by=self.score_column, ascending=False).index.tolist()
        )
        self._isFitted = True
        return self

    def transform(self, X):
        if not self._isFitted:
            raise Exception("fit is required before transform")

        matrix, criteria, input_format = self._coerce_fuzzy_dataframe(X)
        if criteria != self.criteria_:
            raise ValueError(
                "New dataset must have the same fuzzy criteria layout as the dataset used to fit FuzzyTOPSIS."
            )
        if input_format != self._input_format:
            raise ValueError(
                "New dataset must use the same fuzzy input format as the dataset used to fit FuzzyTOPSIS."
            )
        return self._transform_matrix(matrix, X.index)

    def fit_transform(self, X, weights=None, objectives=None):
        self.fit(X, weights=weights, objectives=objectives)
        return self.X_new

    def return_ranking(self, normalized=True):
        ranking = self.X_new.copy() if normalized else self.X.copy()
        if not normalized:
            ranking[self.score_column] = self.X_new[self.score_column]

        ranking = ranking.assign(Rank=None)
        alternative_names = ranking.index.tolist()
        for alternative in alternative_names:
            ranking.loc[alternative, "Rank"] = (
                self._ranked_alternatives.index(alternative) + 1
            )

        columns = ranking.columns.tolist()
        columns = columns[-1:] + columns[:-1]
        ranking = ranking[columns]
        return ranking.sort_values(by=["Rank"])

    def show_ranking(self, mode="standard", first=1, last=None):
        ranking = self.return_ranking(normalized=True)
        if last is None:
            last = len(ranking.index)

        if not isinstance(first, int) or not isinstance(last, int):
            raise TypeError("'first' and 'last' must be integers.")
        if first < 1 or last < 1 or last < first or last > len(ranking.index):
            raise ValueError("'first' and 'last' must define a valid ranking slice.")

        ranking = ranking[(first - 1) : last]
        if mode == "minimal":
            display(ranking["Rank"])
        elif mode in {"standard", "full"}:
            display(ranking)
        else:
            raise ValueError(
                "Invalid value at 'mode': must be 'minimal', 'standard', or 'full'."
            )

    def __get_alternative_ID(self, alternative_id_or_rank):
        if type(alternative_id_or_rank) == int:
            return self._ranked_alternatives[alternative_id_or_rank - 1]
        elif type(alternative_id_or_rank) == str:
            return alternative_id_or_rank
        else:
            raise TypeError(
                f"Invalid alternative identifier: expected int or str, got {type(alternative_id_or_rank)}."
            )

    @staticmethod
    def __check_epsilon(epsilon):
        if not isinstance(epsilon, (float, int)):
            raise TypeError("Invalid value at 'epsilon': must be a float.")
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("Invalid value at 'epsilon': must be in range [0, 1].")

    def __check_features_to_change(self, features_to_change):
        if not isinstance(features_to_change, (list, tuple)) or not features_to_change:
            raise ValueError(
                "Invalid value at 'features_to_change': must be a non-empty list or tuple of fuzzy criteria names."
            )
        if len(set(features_to_change)) != len(features_to_change):
            raise ValueError("Invalid value at 'features_to_change': duplicates are not allowed.")
        invalid = [feature for feature in features_to_change if feature not in self._criterion_index]
        if invalid:
            raise ValueError(
                f"Invalid value at 'features_to_change': unknown fuzzy criteria {invalid}."
            )
        return list(features_to_change)

    def __check_boundary_values(
        self,
        current_matrix,
        features_to_change,
        boundary_values,
        allow_deterioration=False,
    ):
        if boundary_values is None:
            normalized = []
            for feature in features_to_change:
                idx = self._criterion_index[feature]
                if self.objectives[idx] == "max":
                    normalized.append(self._raw_max_bounds[idx].copy())
                else:
                    normalized.append(self._raw_min_bounds[idx].copy())
        elif isinstance(boundary_values, dict):
            normalized = [boundary_values[feature] for feature in features_to_change]
        elif isinstance(boundary_values, (list, tuple, np.ndarray)):
            normalized = list(boundary_values)
        else:
            raise TypeError(
                "Invalid value at 'boundary_values': must be None, a dict, or a list-like collection of triangular fuzzy numbers."
            )

        if len(normalized) != len(features_to_change):
            raise ValueError(
                "Invalid value at 'boundary_values': must be the same length as 'features_to_change'."
            )

        checked = np.zeros((len(features_to_change), 3), dtype=float)
        for idx, (feature, boundary) in enumerate(zip(features_to_change, normalized)):
            criterion_idx = self._criterion_index[feature]
            boundary = self._coerce_fuzzy_value(boundary)
            lower = self._raw_min_bounds[criterion_idx]
            upper = self._raw_max_bounds[criterion_idx]
            current = current_matrix[criterion_idx]

            if np.any(boundary < lower) or np.any(boundary > upper):
                raise ValueError(
                    "Invalid value at 'boundary_values': each fuzzy boundary must stay within the fitted dataset range for that criterion."
                )

            if not allow_deterioration:
                if self.objectives[criterion_idx] == "max" and np.any(boundary < current):
                    raise ValueError(
                        "Invalid value at 'boundary_values': for gain criteria the boundary must be better than or equal to the current fuzzy value."
                    )
                if self.objectives[criterion_idx] == "min" and np.any(boundary > current):
                    raise ValueError(
                        "Invalid value at 'boundary_values': for cost criteria the boundary must be better than or equal to the current fuzzy value."
                    )

            checked[idx] = boundary

        return checked

    def _build_improvement_bounds(self, current_matrix, criterion_indices, boundary_values):
        lower_bounds = current_matrix.copy()
        upper_bounds = current_matrix.copy()
        for criterion_idx, boundary in zip(criterion_indices, boundary_values):
            if self.objectives[criterion_idx] == "max":
                lower_bounds[criterion_idx] = current_matrix[criterion_idx]
                upper_bounds[criterion_idx] = boundary
            else:
                lower_bounds[criterion_idx] = boundary
                upper_bounds[criterion_idx] = current_matrix[criterion_idx]
        return lower_bounds, upper_bounds

    def _build_max_possible_matrix(self, current_matrix, criterion_indices, boundary_values):
        max_possible_matrix = current_matrix.copy()
        for feature_idx, boundary in zip(criterion_indices, boundary_values):
            max_possible_matrix[feature_idx] = boundary
        return max_possible_matrix

    def _matrix_delta_to_frame(self, delta_matrix):
        return self._matrix_to_frame(
            delta_matrix,
            self.criteria_,
            range(len(delta_matrix)),
            self._input_format,
        )

    def improvement(
        self,
        function_name,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon=1e-06,
        **kwargs,
    ):
        if not self._isFitted:
            raise Exception("fit is required before improvement")

        func = getattr(self, function_name)
        return func(
            self.__get_alternative_ID(alternative_to_improve),
            self.__get_alternative_ID(alternative_to_overcome),
            epsilon,
            **kwargs,
        )

    def improvement_genetic(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        features_to_change,
        boundary_values=None,
        allow_deterioration=False,
        popsize=None,
        n_generations=200,
        save_checkpoints=False,
    ):
        if allow_deterioration:
            raise NotImplementedError(
                "Fuzzy TOPSIS post-factum currently supports monotone improvements only."
            )

        self.__check_epsilon(epsilon)
        features_to_change = self.__check_features_to_change(features_to_change)

        current_score = self.X_new.loc[alternative_to_improve, self.score_column]
        target_score = self.X_new.loc[alternative_to_overcome, self.score_column]
        if current_score >= target_score:
            raise ValueError(
                "Invalid value at 'alternative_to_improve': must be worse than alternative_to_overcome'."
            )

        current_idx = self.X.index.get_loc(alternative_to_improve)
        current_matrix = self._matrix[current_idx].copy()
        criterion_indices = [self._criterion_index[feature] for feature in features_to_change]
        boundary_values = self.__check_boundary_values(
            current_matrix=current_matrix,
            features_to_change=features_to_change,
            boundary_values=boundary_values,
            allow_deterioration=allow_deterioration,
        )

        max_possible_matrix = self._build_max_possible_matrix(
            current_matrix,
            criterion_indices,
            boundary_values,
        )
        max_possible_score = self._score_batch_from_matrix(max_possible_matrix[np.newaxis, :, :])[1][0]
        if max_possible_score < target_score + epsilon:
            return None, None

        problem = FuzzyTOPSISPostFactumPymoo(
            aggregation_model=self,
            criterion_indices=criterion_indices,
            current_matrix=current_matrix,
            target_agg_value=target_score + epsilon,
            boundary_values=boundary_values,
        )

        if popsize is None:
            popsize_by_n_objectives = {1: 100, 2: 200, 3: 500, 4: 1000}
            popsize = popsize_by_n_objectives.get(len(features_to_change), 2000)

        algorithm = NSGA2(
            pop_size=popsize,
            crossover=SBX(eta=15, prob=0.9),
            mutation=PM(eta=20),
            save_history=False,
        )

        if save_checkpoints:
            my_callback = MyProgressBar(n_generations, " ".join(features_to_change))
            res = minimize(
                problem,
                algorithm,
                termination=("n_gen", n_generations),
                callback=my_callback,
                seed=42,
                verbose=False,
            )
            checkpoints = my_callback.checkpoints
            checkpoints[f"gen_{n_generations}_final"] = {
                "problem": res.problem,
                "exec_time": res.exec_time,
                "CV": res.CV,
                "F": res.F,
                "G": res.G,
                "X": res.X,
            }
        else:
            res = minimize(
                problem,
                algorithm,
                termination=("n_gen", n_generations),
                seed=42,
                verbose=False,
            )
            checkpoints = None

        if res.X is None:
            return None, checkpoints

        candidate_values = np.atleast_2d(np.asarray(res.X, dtype=float))
        candidate_objectives = np.atleast_2d(np.asarray(res.F, dtype=float))
        modified_matrices = np.repeat(current_matrix[np.newaxis, :, :], len(candidate_values), axis=0)
        modified_matrices[:, criterion_indices, :] = candidate_values.reshape(
            len(candidate_values), len(criterion_indices), 3
        )
        delta_matrices = modified_matrices - current_matrix[np.newaxis, :, :]
        sort_order = np.argsort(np.sum(candidate_objectives, axis=1))
        result = self._matrix_delta_to_frame(delta_matrices[sort_order])
        return result.reset_index(drop=True), checkpoints

    def improvement_non_linear_programming(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        features_to_change,
        boundary_values=None,
        constant_WM=False,
        **kwargs,
    ):
        if constant_WM:
            raise NotImplementedError(
                "Fuzzy TOPSIS exact NLP does not support constant_WM constraints."
            )

        self.__check_epsilon(epsilon)
        features_to_change = self.__check_features_to_change(features_to_change)

        current_score = self.X_new.loc[alternative_to_improve, self.score_column]
        target_score = self.X_new.loc[alternative_to_overcome, self.score_column]
        if current_score >= target_score:
            raise ValueError(
                "Invalid value at 'alternative_to_improve': must be worse than alternative_to_overcome'."
            )

        current_idx = self.X.index.get_loc(alternative_to_improve)
        current_matrix = self._matrix[current_idx].copy()
        criterion_indices = [self._criterion_index[feature] for feature in features_to_change]
        boundary_values = self.__check_boundary_values(
            current_matrix=current_matrix,
            features_to_change=features_to_change,
            boundary_values=boundary_values,
            allow_deterioration=False,
        )

        max_possible_matrix = self._build_max_possible_matrix(
            current_matrix,
            criterion_indices,
            boundary_values,
        )
        if self._score_batch_from_matrix(max_possible_matrix[np.newaxis, :, :])[1][0] < target_score + epsilon:
            return None

        lower_bounds, upper_bounds = self._build_improvement_bounds(
            current_matrix,
            criterion_indices,
            boundary_values,
        )
        excluded_criteria_indices = [
            idx for idx, criterion in enumerate(self.criteria_) if criterion not in features_to_change
        ]

        solver = FuzzyTOPSISNLPPostFactum(
            current_matrix=current_matrix,
            target_score=target_score + epsilon,
            lower_bounds_matrix=lower_bounds,
            upper_bounds_matrix=upper_bounds,
            excluded_criteria_indices=excluded_criteria_indices,
            weights=self.weights,
            objectives=self.objectives,
            gain_scales=self._gain_scales,
            cost_scales=self._cost_scales,
            fpis=self.fpis_,
            fnis=self.fnis_,
        )
        target_matrix = solver.solve()
        if target_matrix is None:
            return None
        achieved_score = self._score_batch_from_matrix(target_matrix[np.newaxis, :, :])[1][0]
        if achieved_score + 1e-6 < target_score + epsilon:
            return None

        return self._matrix_delta_to_frame(
            (target_matrix - current_matrix)[np.newaxis, :, :]
        ).reset_index(drop=True)


class FuzzyTOPSISPostFactumPymoo(Problem):
    def __init__(
        self,
        aggregation_model,
        criterion_indices,
        current_matrix,
        target_agg_value,
        boundary_values,
    ):
        self.aggregation_model = aggregation_model
        self.criterion_indices = np.asarray(criterion_indices, dtype=int)
        self.current_matrix = np.asarray(current_matrix, dtype=float)
        self.target_agg_value = float(target_agg_value)
        self.current_selected = self.current_matrix[self.criterion_indices]
        self.distance_scales = aggregation_model._criterion_distance_scales[self.criterion_indices]

        lower_bounds = []
        upper_bounds = []
        for criterion_idx, boundary in zip(self.criterion_indices, boundary_values):
            current = self.current_matrix[criterion_idx]
            objective = aggregation_model.objectives[criterion_idx]
            if objective == "max":
                lower_bounds.append(current)
                upper_bounds.append(boundary)
            else:
                lower_bounds.append(boundary)
                upper_bounds.append(current)

        lower_bounds = np.asarray(lower_bounds, dtype=float).reshape(-1)
        upper_bounds = np.asarray(upper_bounds, dtype=float).reshape(-1)

        n_criteria = len(self.criterion_indices)
        super().__init__(
            n_var=3 * n_criteria,
            n_obj=n_criteria,
            n_ieq_constr=1 + (2 * n_criteria),
            vtype=float,
        )
        self.xl = lower_bounds
        self.xu = upper_bounds

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        candidate_triples = x.reshape(len(x), len(self.criterion_indices), 3)
        modified_matrices = np.repeat(
            self.current_matrix[np.newaxis, :, :], repeats=len(candidate_triples), axis=0
        )
        modified_matrices[:, self.criterion_indices, :] = candidate_triples

        distances = self.aggregation_model._vertex_distance(
            candidate_triples,
            self.current_selected[np.newaxis, :, :],
        ) / self.distance_scales.reshape(1, -1)
        scores = self.aggregation_model._score_batch_from_matrix(modified_matrices)[1]

        g_target = self.target_agg_value - scores
        g_order_1 = candidate_triples[:, :, 0] - candidate_triples[:, :, 1]
        g_order_2 = candidate_triples[:, :, 1] - candidate_triples[:, :, 2]

        out["F"] = distances
        out["G"] = np.concatenate(
            [g_target[:, np.newaxis], g_order_1, g_order_2],
            axis=1,
        )


FTOPSIS = FuzzyTOPSIS

