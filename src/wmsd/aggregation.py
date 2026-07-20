import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from utils.nonlinear_programming_post_factum import (
    ARASNLPPostFactum,
    COPRASNLPPostFactum,
    SAWNLPPostFactum,
    TopsisNLPPostFactum,
    VIKORNLPPostFactum,
    WASPASNLPPostFactum,
)
from utils.population_reduction import reduce_population_agglomerative_clustering
from utils.single_criterion_exact_improvement import solve_quadratic_equation, choose_appropriate_solution

from .optimization import MyProgressBar, PostFactumAggregationPymoo

class AggregationFunction(ABC):
    """
    Base interface for scalar MCDA aggregation functions.
    """

    letter = None

    def __init__(self, wmsd_transformer=None):
        self.wmsd_transformer = None
        if wmsd_transformer is not None:
            self.fit(wmsd_transformer)

    def fit(self, wmsd_transformer):
        self.wmsd_transformer = wmsd_transformer
        return self

    def score(self, normalized_vector):
        matrix = np.atleast_2d(np.asarray(normalized_vector, dtype=float))
        return self.score_batch(matrix).item()

    @abstractmethod
    def score_batch(self, normalized_matrix):
        """Calculates aggregation scores for normalized alternatives."""
        pass

    def build_nlp_solver(
        self,
        performances_US,
        target_score,
        excluded_criteria_indices,
        upper_bounds_US,
        constant_WM=False,
    ):
        raise NotImplementedError(
            f"Non-linear programming is not supported for aggregation '{self.letter}'."
        )

    def _convert_us_solution_to_modifications(
        self,
        original_performances_US,
        target_performances_US,
    ):
        value_range = np.asarray(self.wmsd_transformer._value_range, dtype=float)
        performance_modifications = (
            np.asarray(target_performances_US, dtype=float)
            - np.asarray(original_performances_US, dtype=float)
        ) * value_range
        performance_modifications = performance_modifications.astype(float)
        performance_modifications[
            np.asarray(self.wmsd_transformer.objectives) == "min"
        ] *= -1
        return pd.DataFrame(
            [performance_modifications],
            columns=self.wmsd_transformer.X.columns,
        )

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
        if alternative_to_improve[str(self.letter)] >= alternative_to_overcome[str(self.letter)]:
            raise ValueError(
                "Invalid value at 'alternative_to_improve': must be worse than alternative_to_overcome'"
            )

        self.__check_epsilon(epsilon, self.wmsd_transformer.weights)
        boundary_values = self.__check_boundary_values(
            alternative_to_improve, features_to_change, boundary_values
        )

        criteria_columns = self.wmsd_transformer.X.columns.tolist()
        performances_US = (
            alternative_to_improve.loc[self.wmsd_transformer.X.columns]
            .to_numpy()
            .copy()
        )

        excluded_criteria_indices = [
            idx
            for idx, name in enumerate(criteria_columns)
            if name not in features_to_change
        ]
        upper_bounds_US = np.ones_like(performances_US, dtype=float)
        for feature_name, boundary in zip(features_to_change, boundary_values):
            upper_bounds_US[criteria_columns.index(feature_name)] = boundary

        solver = self.build_nlp_solver(
            performances_US=performances_US,
            target_score=alternative_to_overcome[str(self.letter)] + epsilon,
            excluded_criteria_indices=excluded_criteria_indices,
            upper_bounds_US=upper_bounds_US,
            constant_WM=constant_WM,
        )
        target_performances_US = solver.solve()

        if target_performances_US is None:
            return None

        return self._convert_us_solution_to_modifications(
            original_performances_US=performances_US,
            target_performances_US=np.clip(target_performances_US, 0.0, 1.0),
        )

    def improvement_single_feature(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        feature_to_change,
        **kwargs,
    ):
        """Calculates the required change on a single criterion."""
        return self.improvement_features(
            alternative_to_improve,
            alternative_to_overcome,
            epsilon,
            features_to_change=[feature_to_change],
            **kwargs,
        )

    def __check_boundary_values(self, alternative_to_improve, features_to_change, boundary_values):
        if boundary_values is None:
            boundary_values = np.ones(len(features_to_change))
        elif not isinstance(boundary_values, list):
            raise TypeError("Invalid value at 'boundary_values': must be a list")
        else:
            if len(features_to_change) != len(boundary_values):
                raise ValueError("Invalid value at 'boundary_values': must be same length as 'features_to_change'")

            lower_bounds = self.wmsd_transformer._lower_bounds
            upper_bounds = self.wmsd_transformer._upper_bounds
            value_range = self.wmsd_transformer._value_range
            criteria_columns = self.wmsd_transformer.X.columns

            for i, feature_name in enumerate(features_to_change):
                col = criteria_columns.get_loc(feature_name)
                if boundary_values[i] < lower_bounds[col] or boundary_values[i] > upper_bounds[col]:
                    raise ValueError("Invalid value at 'boundary_values': must be between defined 'expert_range'")
                else:
                    boundary_values[i] = (boundary_values[i] - lower_bounds[col]) / value_range[col]
                    if self.wmsd_transformer.objectives[col] == "min":
                        boundary_values[i] = 1 - boundary_values[i]
                    if alternative_to_improve[feature_name] > boundary_values[i]:
                        raise ValueError(
                            "Invalid value at 'boundary_values': must be better than or equal to the performances of the alternative being improved"
                        )

        return np.array(boundary_values)

    def __check_epsilon(self, epsilon, w):
        if not (isinstance(epsilon, float) or isinstance(epsilon, int)):
            raise ValueError("Invalid value at 'epsilon': must be a float")

        mean_weight = np.mean(w)
        if (epsilon < 0.0) or (epsilon > mean_weight/2):
            raise ValueError(f"Invalid value at 'epsilon': must be in range [0, {mean_weight/2}]")

    def improvement_features(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        features_to_change,
        boundary_values=None,
        **kwargs,
    ):
        if alternative_to_improve[str(self.letter)] >= alternative_to_overcome[str(self.letter)]:
            raise ValueError("Invalid value at 'alternative_to_improve': must be worse than alternative_to_overcome'")

        self.__check_epsilon(epsilon, self.wmsd_transformer.weights)
        boundary_values = self.__check_boundary_values(alternative_to_improve, features_to_change, boundary_values)

        initial_performances = alternative_to_improve.loc[self.wmsd_transformer.X.columns]
        current_performances = initial_performances.copy()

        is_improvement_satisfactory = False
        for i, k in zip(features_to_change, boundary_values):
            current_performances[i] = k
            agg_value = self.score(current_performances.to_numpy())

            if agg_value < alternative_to_overcome[str(self.letter)]:
                continue

            current_performances[i] = 0.5 * k
            agg_value = self.score(current_performances.to_numpy())
            change_ratio = 0.25 * k
            while True:
                if agg_value < alternative_to_overcome[str(self.letter)]:
                    current_performances[i] += change_ratio
                elif agg_value - alternative_to_overcome[str(self.letter)] > epsilon:
                    current_performances[i] -= change_ratio
                else:
                    is_improvement_satisfactory = True
                    break
                change_ratio = change_ratio / 2
                agg_value = self.score(current_performances.to_numpy())

            if is_improvement_satisfactory:
                value_range = self.wmsd_transformer._value_range
                performance_modifications = current_performances - initial_performances
                for j in range(len(performance_modifications)):
                    if performance_modifications.iloc[j] == 0:
                        continue
                    elif self.wmsd_transformer.objectives[j] == "max":
                        performance_modifications.iloc[j] = value_range[j] * performance_modifications.iloc[j]
                    else:
                        performance_modifications.iloc[j] = -value_range[j] * performance_modifications.iloc[j]
                result_df = performance_modifications.to_frame().transpose()
                result_df = result_df.reset_index(drop=True)
                return result_df
        else:
            return None

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
        seed=42,
    ):
        boundary_values = self.__check_boundary_values(
            alternative_to_improve, features_to_change, boundary_values
        )

        current_performances_US = (
            alternative_to_improve.loc[self.wmsd_transformer.X.columns].to_numpy().copy()
        )
        modified_criteria_subset = [
            x in features_to_change for x in self.wmsd_transformer.X.columns.tolist()
        ]

        max_possible_improved = current_performances_US.copy()
        max_possible_improved[modified_criteria_subset] = boundary_values
        max_possible_agg_value = self.score(max_possible_improved)
        if max_possible_agg_value < alternative_to_overcome[str(self.letter)]:
            return None

        problem = PostFactumAggregationPymoo(
            aggregation_model=self.wmsd_transformer,
            modified_criteria_subset=modified_criteria_subset,
            current_performances=current_performances_US,
            target_agg_value=alternative_to_overcome[str(self.letter)],
            upper_bounds=boundary_values,
            allow_deterioration=allow_deterioration,
        )

        if popsize is None:
            popsize_by_n_objectives = {2: 200, 3: 1000, 4: 2000}
            popsize = popsize_by_n_objectives.get(len(features_to_change), 5000)

        algorithm = NSGA2(
            pop_size=popsize,
            crossover=SBX(eta=15, prob=0.9),
            mutation=PM(eta=20),
            save_history=False,
        )

        if save_checkpoints:
            my_callback = MyProgressBar(n_generations, ' '.join(features_to_change))
            res = minimize(
                problem,
                algorithm,
                termination=('n_gen', n_generations),
                callback=my_callback,
                seed=seed,
                verbose=False,
            )
            print("Genetic algorithm execution time", res.exec_time)

            checkpoints = my_callback.checkpoints
            checkpoints[f"gen_{n_generations}_final"] = {
                "problem": res.problem,
                "exec_time": res.exec_time,
                "CV": res.CV,
                "F": res.F,
                "G": res.G,
                "X": res.X
            }
            result_path = f"checkpoints_popsize_{popsize}_n_gen_{n_generations}_{time.strftime('%Y_%m_%d_%H_%M_%S')}.pickle"
            print("Saving genetic algorithm checkpoints to:", result_path)
            pd.to_pickle(checkpoints, result_path)
        else:
            res = minimize(
                problem,
                algorithm,
                termination=('n_gen', n_generations),
                seed=seed,
                verbose=False,
            )
            checkpoints = None

        if res.F is not None:
            improvement_actions = np.zeros(
                shape=(len(res.F), len(current_performances_US))
            )
            improvement_actions[:, modified_criteria_subset] = (
                res.F - current_performances_US[modified_criteria_subset]
            )
            improvement_actions *= np.array(self.wmsd_transformer._value_range)
            improvement_actions[
                :, np.array(self.wmsd_transformer.objectives) == "min"
            ] *= -1
            return pd.DataFrame(
                sorted(improvement_actions.tolist(), key=lambda x: x[0]),
                columns=self.wmsd_transformer.X.columns,
            ), checkpoints
        else:
            return None, None


class WMSDAggregationFunction(AggregationFunction):
    """Base for aggregation methods that operate in WMSD space."""

    @abstractmethod
    def score_from_wmsd(self, mean_weight, w_means, w_stds):
        """Calculates scores from WMSD coordinates."""
        pass

    def improvement_mean(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        allow_std=False,
        solutions_number=5,
        **kwargs,
    ):
        """ Calculates minimal change in mean value of alternative's criteria in order to 
        let the alternative achieve the target position.
        Parameters
        ----------
        alternative_to_improve : int or str
            Name or position of the alternative which user wants to improve.
        alternative_to_overcome : int or str
            Name or position of the alternative which should be overcome by chosen alternative.
        epsilon : float
            Precision of calculations. Must be in range (0.0, 1.0>.
            (default : 0.000001)
        allow_std : bool
            If True then also possible proposition of changes in standard deviation.
            (default : False)
        solutions_number : int
            Maximal number of proposed solutions.
            (default : 5)
        Returns
        -------
        At most [solution_number] proposed solutions.
        """
        if alternative_to_improve[str(self.letter)] >= alternative_to_overcome[str(self.letter)]:
            raise ValueError(
                "Invalid value at 'alternative_to_improve': must be worse than alternative_to_overcome'"
            )

        w = np.mean(self.wmsd_transformer.weights)
        m_start = alternative_to_improve["Mean"]
        m_boundary = w
        std_start = alternative_to_improve["Std"]
        if (
            self.score_from_wmsd(w, m_boundary, alternative_to_improve["Std"])
            < alternative_to_overcome[str(self.letter)]
        ):
            return None
        else:
            change = (m_boundary - alternative_to_improve["Mean"]) / 2
            actual_aggfn = self.score_from_wmsd(
                w, alternative_to_improve["Mean"], alternative_to_improve["Std"]
            )
            while True:
                if actual_aggfn >= alternative_to_overcome[str(self.letter)]:
                    if (
                        actual_aggfn - alternative_to_overcome[str(self.letter)]
                        > epsilon
                    ):
                        alternative_to_improve["Mean"] -= change
                        change = change / 2
                        actual_aggfn = self.score_from_wmsd(
                            w,
                            alternative_to_improve["Mean"],
                            alternative_to_improve["Std"],
                        )
                    else:
                        break
                else:
                    alternative_to_improve["Mean"] += change
                    actual_aggfn = self.score_from_wmsd(
                        w, alternative_to_improve["Mean"], alternative_to_improve["Std"]
                    )
                    if actual_aggfn >= alternative_to_overcome[str(self.letter)]:
                        change = change / 2
            if alternative_to_improve["Std"] <= self.wmsd_transformer.max_std_calculator(
                alternative_to_improve["Mean"], self.wmsd_transformer.weights
            ):
                if solutions_number is None:
                    return pd.DataFrame(
                        [alternative_to_improve["Mean"] - m_start], columns=["Mean"]
                    )
                else:
                    inverse_solutions = self.wmsd_transformer.inverse_transform_numpy(alternative_to_improve["Mean"], alternative_to_improve["Std"], "==")
                    reduced_solutions = reduce_population_agglomerative_clustering(inverse_solutions, solutions_number)
                    result = reduced_solutions
            elif allow_std:
                alternative_to_improve["Std"] = self.wmsd_transformer.max_std_calculator(
                    alternative_to_improve["Mean"], self.wmsd_transformer.weights
                )
                actual_aggfn = self.score_from_wmsd(
                    w, alternative_to_improve["Mean"], alternative_to_improve["Std"]
                )
                if actual_aggfn >= alternative_to_overcome[str(self.letter)]:
                    if solutions_number is None:
                        return pd.DataFrame(
                            [
                                [
                                    alternative_to_improve["Mean"] - m_start,
                                    alternative_to_improve["Std"] - std_start,
                                ]
                            ],
                            columns=["Mean", "Std"],
                        )
                    else:
                        inverse_solutions = self.wmsd_transformer.inverse_transform_numpy(alternative_to_improve["Mean"], alternative_to_improve["Std"], "==")
                        reduced_solutions = reduce_population_agglomerative_clustering(inverse_solutions, solutions_number)
                        result = reduced_solutions
                else:
                    if solutions_number is None:
                        return pd.DataFrame(
                            [
                                [
                                    alternative_to_improve["Mean"] - m_start,
                                    alternative_to_improve["Std"] - std_start,
                                ]
                            ],
                            columns=["Mean", "Std"],
                        ) + self.improvement_mean(
                            alternative_to_improve,
                            alternative_to_overcome,
                            epsilon,
                            allow_std,
                            **kwargs,
                        )
                    else:
                        return self.improvement_mean(
                            alternative_to_improve,
                            alternative_to_overcome,
                            epsilon,
                            allow_std,
                            solutions_number,
                            **kwargs,
                        )
            else:
                while alternative_to_improve["Mean"] <= m_boundary:
                    if alternative_to_improve[
                        "Std"
                    ] <= self.wmsd_transformer.max_std_calculator(
                        alternative_to_improve["Mean"], self.wmsd_transformer.weights
                    ):
                        if solutions_number is None:
                            return pd.DataFrame(
                                [alternative_to_improve["Mean"] - m_start], columns=["Mean"]
                            )
                        else:
                            inverse_solutions = self.wmsd_transformer.inverse_transform_numpy(alternative_to_improve["Mean"], alternative_to_improve["Std"], "==")
                            reduced_solutions = reduce_population_agglomerative_clustering(inverse_solutions, solutions_number)
                            result = reduced_solutions
                            break
                    alternative_to_improve["Mean"] += epsilon
                else:
                    return None
            result_means, result_stds = self.wmsd_transformer.transform_US_to_wmsd(np.array(result))
            objectives = self.wmsd_transformer.objectives
            value_range = self.wmsd_transformer._value_range
            result -= alternative_to_improve[:-3]
            for i in result.index:
                for j in range(len(result.columns)):
                    if result[result.columns[j]][i] == 0:
                        continue
                    elif objectives[j] == "max":
                        result[result.columns[j]][i] = (
                            value_range[j] * result[result.columns[j]][i]
                        )
                    else:
                        result[result.columns[j]][i] = (
                            -value_range[j] * result[result.columns[j]][i]
                        )
            result['Mean'] = result_means - m_start
            result['Std'] = result_stds - std_start
            return result


class TOPSISAggregationFunction(WMSDAggregationFunction):
    """
    A class used to calculate TOPSIS ranking and perform improvement actions.
    ...
    Attributes
    ----------
    wmsd_transformer : WMSDTransformer object
    """

    @abstractmethod
    def TOPSIS_calculation(self, w, wm, wsd):
        """Calculates TOPSIS values according to chosen aggregation function."""
        pass

    def score_from_wmsd(self, mean_weight, w_means, w_stds):
        return self.TOPSIS_calculation(mean_weight, w_means, w_stds)

    def score_batch(self, normalized_matrix):
        normalized_matrix = np.atleast_2d(np.asarray(normalized_matrix, dtype=float))
        w_means, w_stds = self.wmsd_transformer.transform_US_to_wmsd(normalized_matrix)
        return self.score_from_wmsd(
            np.mean(self.wmsd_transformer.weights),
            w_means,
            w_stds,
        )

    def __check_boundary_values(self, alternative_to_improve, features_to_change, boundary_values):
        if boundary_values is None:
            boundary_values = np.ones(len(features_to_change))
        elif not isinstance(boundary_values, list):
            raise TypeError("Invalid value at 'boundary_values': must be a list")
        else:
            if len(features_to_change) != len(boundary_values):
                raise ValueError("Invalid value at 'boundary_values': must be same length as 'features_to_change'")

            lower_bounds = self.wmsd_transformer._lower_bounds
            upper_bounds = self.wmsd_transformer._upper_bounds
            value_range = self.wmsd_transformer._value_range
            criteria_columns = self.wmsd_transformer.X.columns

            for i, feature_name in enumerate(features_to_change):
                col = criteria_columns.get_loc(feature_name)
                if boundary_values[i] < lower_bounds[col] or boundary_values[i] > upper_bounds[col]:
                    raise ValueError("Invalid value at 'boundary_values': must be between defined 'expert_range'")
                else:
                    boundary_values[i] = (boundary_values[i] - lower_bounds[col]) / value_range[col]
                    if self.wmsd_transformer.objectives[col] == "min":
                        boundary_values[i] = 1 - boundary_values[i]
                    if alternative_to_improve[feature_name] > boundary_values[i]:
                        raise ValueError(
                            "Invalid value at 'boundary_values': must be better than or equal to the performances of the alternative being improved"
                        )

        return np.array(boundary_values)

    def __check_epsilon(self, epsilon, w):
        if not (isinstance(epsilon, float) or isinstance(epsilon, int)):
            raise ValueError("Invalid value at 'epsilon': must be a float")

        mean_weight = np.mean(w)
        if (epsilon < 0.0) or (epsilon > mean_weight/2):
            raise ValueError(f"Invalid value at 'epsilon': must be in range [0, {mean_weight/2}]")

    def improvement_features(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        features_to_change,
        boundary_values=None,
        **kwargs,
    ):
        """ Calculates minimal change in given criteria values in order to 
        let the alternative achieve the target position.
        Parameters
        ----------
        alternative_to_improve : int or str
            Name or position of the alternative which user wants to improve.
        alternative_to_overcome : int or str
            Name or position of the alternative which should be overcome by chosen alternative.
        epsilon : float
            Precision of calculations. Must be in range (0.0, 1.0>.
            (default : 0.000001)
        features_to_change : array of str
            Array containing names of criteria on which change should be calculated.
        boundary_values : 2D array of floats
            Array with dimensions number_of_features_to_change x 2. For each feature to change it should
            have provided 2 numbers: lower and upper boundaries of proposed values.
            (default : None)
        Returns
        -------
        Proposed solutions.
        """
        if alternative_to_improve[str(self.letter)] >= alternative_to_overcome[str(self.letter)]:
            raise ValueError("Invalid value at 'alternative_to_improve': must be worse than alternative_to_overcome'")

        self.__check_epsilon(epsilon, self.wmsd_transformer.weights)
        boundary_values = self.__check_boundary_values(alternative_to_improve, features_to_change, boundary_values)

        initial_performances = alternative_to_improve.loc[self.wmsd_transformer.X.columns]
        current_performances = initial_performances.copy()

        is_improvement_satisfactory = False
        for i, k in zip(features_to_change, boundary_values):
            # Applying the maximum allowable improvement of the alternative's evaluation on the i-th criterion
            current_performances[i] = k
            agg_value = self.score(current_performances.to_numpy())

            # If the maximum allowable improvement on this criterion is not sufficient to achieve the target,
            # then it is necessary to improve on the next criterion.
            if agg_value < alternative_to_overcome[str(self.letter)]:
                continue

            # If the maximum allowable improvement of this criterion is sufficient to achieve the goal,
            # perform the binary search algorithm to achieve the target by means of the minimal improvement.
            current_performances[i] = 0.5 * k
            agg_value = self.score(current_performances.to_numpy())
            change_ratio = 0.25 * k
            while True:
                if agg_value < alternative_to_overcome[str(self.letter)]:
                    current_performances[i] += change_ratio
                elif agg_value - alternative_to_overcome[str(self.letter)] > epsilon:
                    current_performances[i] -= change_ratio
                else:
                    is_improvement_satisfactory = True
                    break
                change_ratio = change_ratio / 2
                agg_value = self.score(current_performances.to_numpy())

            if is_improvement_satisfactory:
                value_range = self.wmsd_transformer._value_range
                performance_modifications = current_performances - initial_performances
                for j in range(len(performance_modifications)):
                    if performance_modifications.iloc[j] == 0:
                        continue
                    elif self.wmsd_transformer.objectives[j] == "max":
                        performance_modifications.iloc[j] = value_range[j] * performance_modifications.iloc[j]
                    else:
                        performance_modifications.iloc[j] = -value_range[j] * performance_modifications.iloc[j]
                result_df = performance_modifications.to_frame().transpose()
                result_df = result_df.reset_index(drop=True)
                return result_df
        else:
            return None

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
        seed=42,
    ):
        """ Use genetic algorithm to create propositions of changes to
        let the chosen alternative achieve the target position.
        Parameters
        ----------
        alternative_to_improve : int or str
            Name or position of the alternative which user wants to improve.
        alternative_to_overcome : int or str
            Name or position of the alternative which should be overcome by chosen alternative.
        epsilon : float
            Precision of calculations. Must be in range (0.0, 1.0>.
            (default : 0.000001)
        features_to_change : array of str
            Array containing names of criteria on which change should be calculated.
        boundary_values : 2D array of floats
            Array with dimensions number_of_features_to_change x 2. For each feature to change it should
            have provided 2 numbers: lower and upper boundaries of proposed values.
            (default : None)
        allow_deterioration : bool
            TODO description
            (default : False)
        popsize : int
            Size of the population.
            (default : None)
        n_generations : int
            Number of generations (iterations).
            (default : 200)
        Returns
        -------
        Proposed solutions.
        """
        boundary_values = self.__check_boundary_values(
            alternative_to_improve, features_to_change, boundary_values
        )
        # TODO check if criteria names are correct (I misspelled once)

        current_performances_US = (
            alternative_to_improve.loc[self.wmsd_transformer.X.columns].to_numpy().copy()
        )
        modified_criteria_subset = [
            x in features_to_change for x in self.wmsd_transformer.X.columns.tolist()
        ]

        max_possible_improved = current_performances_US.copy()
        max_possible_improved[modified_criteria_subset] = boundary_values
        max_possible_agg_value = self.score(max_possible_improved)
        if max_possible_agg_value < alternative_to_overcome[str(self.letter)]:
            # print(f"Not possible to achieve target {alternative_to_overcome['AggFn']} with specified features and boundary_values. Max possible agg value is {max_possible_agg_value}")
            return None

        problem = PostFactumAggregationPymoo(
            aggregation_model=self.wmsd_transformer,
            modified_criteria_subset=modified_criteria_subset,
            current_performances=current_performances_US,
            target_agg_value=alternative_to_overcome[str(self.letter)],
            upper_bounds=boundary_values,
            allow_deterioration=allow_deterioration,
        )

        if popsize is None:
            popsize_by_n_objectives = {2: 200, 3: 1000, 4: 2000}
            popsize = popsize_by_n_objectives.get(len(features_to_change), 5000)

        algorithm = NSGA2(
            pop_size=popsize,
            crossover=SBX(eta=15, prob=0.9),
            mutation=PM(eta=20),
            save_history=False,
        )


        if save_checkpoints:
            my_callback = MyProgressBar(n_generations, ' '.join(features_to_change))
            res = minimize(
                problem,
                algorithm,
                termination=('n_gen', n_generations),
                callback=my_callback,
                seed=seed,
                verbose=False,
            )
            print("Genetic algorithm execution time", res.exec_time)

            checkpoints = my_callback.checkpoints
            checkpoints[f"gen_{n_generations}_final"] = {
                "problem": res.problem,
                "exec_time": res.exec_time,
                "CV": res.CV,
                "F": res.F,
                "G": res.G,
                "X": res.X
            }
            result_path = f"checkpoints_popsize_{popsize}_n_gen_{n_generations}_{time.strftime('%Y_%m_%d_%H_%M_%S')}.pickle"
            print("Saving genetic algorithm checkpoints to:", result_path)
            pd.to_pickle(checkpoints, result_path)
        else:
            res = minimize(
                problem,
                algorithm,
                termination=('n_gen', n_generations),
                seed=seed,
                verbose=False,
            )
            checkpoints = None

        if res.F is not None:
            improvement_actions = np.zeros(
                shape=(len(res.F), len(current_performances_US))
            )
            improvement_actions[:, modified_criteria_subset] = (
                res.F - current_performances_US[modified_criteria_subset]
            )
            improvement_actions *= np.array(self.wmsd_transformer._value_range)
            improvement_actions[
                :, np.array(self.wmsd_transformer.objectives) == "min"
            ] *= -1
            return pd.DataFrame(
                sorted(improvement_actions.tolist(), key=lambda x: x[0]),
                columns=self.wmsd_transformer.X.columns,
            ), checkpoints
        else:
            return None, None

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
        if alternative_to_improve[str(self.letter)] >= alternative_to_overcome[str(self.letter)]:
            raise ValueError(
                "Invalid value at 'alternative_to_improve': must be worse than alternative_to_overcome'"
            )

        self.__check_epsilon(epsilon, self.wmsd_transformer.weights)
        boundary_values = self.__check_boundary_values(
            alternative_to_improve, features_to_change, boundary_values
        )

        criteria_columns = self.wmsd_transformer.X.columns.tolist()
        performances_US = (
            alternative_to_improve.loc[self.wmsd_transformer.X.columns]
            .to_numpy()
            .copy()
        )

        excluded_criteria_indices = [
            idx
            for idx, name in enumerate(criteria_columns)
            if name not in features_to_change
        ]
        upper_bounds_US = np.ones_like(performances_US, dtype=float)
        for feature_name, boundary in zip(features_to_change, boundary_values):
            upper_bounds_US[criteria_columns.index(feature_name)] = boundary

        solver = self.build_nlp_solver(
            performances_US=performances_US,
            target_score=alternative_to_overcome[str(self.letter)] + epsilon,
            excluded_criteria_indices=excluded_criteria_indices,
            upper_bounds_US=upper_bounds_US,
            constant_WM=constant_WM,
        )
        target_performances_US = solver.solve()

        if target_performances_US is None:
            return None

        return self._convert_us_solution_to_modifications(
            original_performances_US=performances_US,
            target_performances_US=np.clip(target_performances_US, 0.0, 1.0),
        )





class SAW(AggregationFunction):
    """Simple Additive Weighting aggregation."""

    def __init__(self, wmsd_transformer=None):
        super().__init__(wmsd_transformer)
        self.letter = "U"

    def score_batch(self, normalized_matrix):
        normalized_matrix = np.atleast_2d(np.asarray(normalized_matrix, dtype=float))
        return normalized_matrix @ np.asarray(self.wmsd_transformer.weights, dtype=float)

    def build_nlp_solver(
        self,
        performances_US,
        target_score,
        excluded_criteria_indices,
        upper_bounds_US,
        constant_WM=False,
    ):
        return SAWNLPPostFactum(
            performances_US=performances_US,
            weights=self.wmsd_transformer.weights,
            target_score=target_score,
            excluded_criteria_indices=excluded_criteria_indices,
            upper_bounds_US=upper_bounds_US,
            constant_WM=constant_WM,
        )

    def improvement_single_feature(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        feature_to_change,
        **kwargs,
    ):
        if alternative_to_improve[str(self.letter)] >= alternative_to_overcome[str(self.letter)]:
            raise ValueError(
                "Invalid value at 'alternative_to_improve': must be worse than alternative_to_overcome'"
            )

        criteria_columns = self.wmsd_transformer.X.columns
        performances_US = alternative_to_improve.loc[criteria_columns].to_numpy().copy()
        modified_criterion_idx = list(criteria_columns).index(feature_to_change)
        criterion_weight = self.wmsd_transformer.weights[modified_criterion_idx]
        if criterion_weight <= 0:
            return None

        target_score = alternative_to_overcome[str(self.letter)] + epsilon / 2
        current_score = self.score(performances_US)
        required_delta = target_score - current_score
        if required_delta <= 0:
            return pd.DataFrame(
                [np.zeros_like(performances_US)],
                columns=criteria_columns,
            )

        normalized_delta = required_delta / criterion_weight
        improved_value = performances_US[modified_criterion_idx] + normalized_delta
        if improved_value > 1:
            return None

        feature_modification = normalized_delta * self.wmsd_transformer._value_range[modified_criterion_idx]
        if self.wmsd_transformer.objectives[modified_criterion_idx] == "min":
            feature_modification *= -1

        modification_vector = np.zeros_like(performances_US)
        modification_vector[modified_criterion_idx] = feature_modification
        return pd.DataFrame([modification_vector], columns=criteria_columns)


class ARAS(AggregationFunction):
    """Additive Ratio Assessment aggregation on normalized utility data."""

    def __init__(self, wmsd_transformer=None):
        super().__init__(wmsd_transformer)
        self.letter = "K"

    def score_batch(self, normalized_matrix):
        normalized_matrix = np.atleast_2d(np.asarray(normalized_matrix, dtype=float))
        weights = np.asarray(self.wmsd_transformer.weights, dtype=float)
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            return np.zeros(normalized_matrix.shape[0], dtype=float)
        return (normalized_matrix @ weights) / weight_sum

    def build_nlp_solver(
        self,
        performances_US,
        target_score,
        excluded_criteria_indices,
        upper_bounds_US,
        constant_WM=False,
    ):
        return ARASNLPPostFactum(
            performances_US=performances_US,
            weights=self.wmsd_transformer.weights,
            target_score=target_score,
            excluded_criteria_indices=excluded_criteria_indices,
            upper_bounds_US=upper_bounds_US,
            constant_WM=constant_WM,
        )

    def improvement_single_feature(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        feature_to_change,
        **kwargs,
    ):
        if alternative_to_improve[str(self.letter)] >= alternative_to_overcome[str(self.letter)]:
            raise ValueError(
                "Invalid value at 'alternative_to_improve': must be worse than alternative_to_overcome'"
            )

        criteria_columns = self.wmsd_transformer.X.columns
        performances_US = alternative_to_improve.loc[criteria_columns].to_numpy().copy()
        modified_criterion_idx = list(criteria_columns).index(feature_to_change)
        weights = np.asarray(self.wmsd_transformer.weights, dtype=float)
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            return None

        criterion_weight = weights[modified_criterion_idx] / weight_sum
        if criterion_weight <= 0:
            return None

        target_score = alternative_to_overcome[str(self.letter)] + epsilon / 2
        current_score = self.score(performances_US)
        required_delta = target_score - current_score
        if required_delta <= 0:
            return pd.DataFrame(
                [np.zeros_like(performances_US)],
                columns=criteria_columns,
            )

        normalized_delta = required_delta / criterion_weight
        improved_value = performances_US[modified_criterion_idx] + normalized_delta
        if improved_value > 1:
            return None

        feature_modification = normalized_delta * self.wmsd_transformer._value_range[modified_criterion_idx]
        if self.wmsd_transformer.objectives[modified_criterion_idx] == "min":
            feature_modification *= -1

        modification_vector = np.zeros_like(performances_US)
        modification_vector[modified_criterion_idx] = feature_modification
        return pd.DataFrame([modification_vector], columns=criteria_columns)


class COPRAS(AggregationFunction):
    """COPRAS-inspired score on normalized utility data."""

    def __init__(self, wmsd_transformer=None):
        super().__init__(wmsd_transformer)
        self.letter = "C"

    def score_batch(self, normalized_matrix):
        normalized_matrix = np.atleast_2d(np.asarray(normalized_matrix, dtype=float))
        weights = np.asarray(self.wmsd_transformer.weights, dtype=float)
        objectives = np.asarray(self.wmsd_transformer.objectives)

        gain_mask = objectives == "max"
        cost_mask = objectives == "min"

        sp = np.sum(normalized_matrix[:, gain_mask] * weights[gain_mask], axis=1)
        if not np.any(cost_mask):
            return sp

        sm = np.sum((1 - normalized_matrix[:, cost_mask]) * weights[cost_mask], axis=1)
        sm = np.maximum(sm, 1e-12)
        return sp / sm

    def build_nlp_solver(
        self,
        performances_US,
        target_score,
        excluded_criteria_indices,
        upper_bounds_US,
        constant_WM=False,
    ):
        return COPRASNLPPostFactum(
            performances_US=performances_US,
            weights=self.wmsd_transformer.weights,
            objectives=self.wmsd_transformer.objectives,
            target_score=target_score,
            excluded_criteria_indices=excluded_criteria_indices,
            upper_bounds_US=upper_bounds_US,
            constant_WM=constant_WM,
        )


class WASPAS(AggregationFunction):
    """Weighted Aggregated Sum Product Assessment on normalized utility data."""

    def __init__(self, wmsd_transformer=None, lam=0.5):
        self.lam = lam
        super().__init__(wmsd_transformer)
        self.letter = "W"

    def score_batch(self, normalized_matrix):
        normalized_matrix = np.atleast_2d(np.asarray(normalized_matrix, dtype=float))
        weights = np.asarray(self.wmsd_transformer.weights, dtype=float)
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            return np.zeros(normalized_matrix.shape[0], dtype=float)

        normalized_weights = weights / weight_sum
        q_sum = np.sum(normalized_matrix * normalized_weights, axis=1)
        q_prod = np.prod(normalized_matrix ** normalized_weights, axis=1)
        return self.lam * q_sum + (1 - self.lam) * q_prod

    def build_nlp_solver(
        self,
        performances_US,
        target_score,
        excluded_criteria_indices,
        upper_bounds_US,
        constant_WM=False,
    ):
        return WASPASNLPPostFactum(
            performances_US=performances_US,
            weights=self.wmsd_transformer.weights,
            target_score=target_score,
            excluded_criteria_indices=excluded_criteria_indices,
            upper_bounds_US=upper_bounds_US,
            constant_WM=constant_WM,
            lam=self.lam,
        )


class VIKOR(AggregationFunction):
    """VIKOR compromise ranking expressed as a higher-is-better score (1 - Q)."""

    def __init__(self, wmsd_transformer=None, v=0.5):
        self.v = v
        super().__init__(wmsd_transformer)
        self.letter = "V"

    def _get_reference_context(self):
        reference_matrix = np.asarray(
            self.wmsd_transformer.X_new.loc[:, self.wmsd_transformer.X.columns],
            dtype=float,
        )
        fstar = np.max(reference_matrix, axis=0)
        fminus = np.min(reference_matrix, axis=0)
        if np.any(np.isclose(fstar, fminus)):
            raise ValueError(
                "VIKOR cannot be applied when a criterion has identical values for all alternatives."
            )

        weights = np.asarray(self.wmsd_transformer.weights, dtype=float)
        weight_sum = np.sum(weights)
        if weight_sum <= 0:
            raise ValueError("VIKOR requires at least one positive criterion weight.")
        normalized_weights = weights / weight_sum

        weighted_ff = normalized_weights * ((fstar - reference_matrix) / (fstar - fminus))
        S = np.sum(weighted_ff, axis=1)
        R = np.max(weighted_ff, axis=1)

        return {
            "fstar": fstar,
            "fminus": fminus,
            "weights": normalized_weights,
            "Sstar": float(np.min(S)),
            "Sminus": float(np.max(S)),
            "Rstar": float(np.min(R)),
            "Rminus": float(np.max(R)),
        }

    def score_batch(self, normalized_matrix):
        normalized_matrix = np.atleast_2d(np.asarray(normalized_matrix, dtype=float))
        context = self._get_reference_context()

        weighted_ff = context["weights"] * (
            (context["fstar"] - normalized_matrix)
            / (context["fstar"] - context["fminus"])
        )
        S = np.sum(weighted_ff, axis=1)
        R = np.max(weighted_ff, axis=1)

        if np.isclose(context["Sminus"], context["Sstar"]):
            s_term = np.zeros_like(S)
        else:
            s_term = (S - context["Sstar"]) / (context["Sminus"] - context["Sstar"])

        if np.isclose(context["Rminus"], context["Rstar"]):
            r_term = np.zeros_like(R)
        else:
            r_term = (R - context["Rstar"]) / (context["Rminus"] - context["Rstar"])

        q_values = self.v * s_term + (1 - self.v) * r_term
        return 1 - q_values

    def build_nlp_solver(
        self,
        performances_US,
        target_score,
        excluded_criteria_indices,
        upper_bounds_US,
        constant_WM=False,
    ):
        context = self._get_reference_context()
        return VIKORNLPPostFactum(
            performances_US=performances_US,
            target_score=target_score,
            excluded_criteria_indices=excluded_criteria_indices,
            weights=context["weights"],
            fstar=context["fstar"],
            fminus=context["fminus"],
            sstar=context["Sstar"],
            sminus=context["Sminus"],
            rstar=context["Rstar"],
            rminus=context["Rminus"],
            v=self.v,
            upper_bounds_US=upper_bounds_US,
            constant_WM=constant_WM,
        )


class ATOPSIS(TOPSISAggregationFunction):
    """
    A class used to calculate TOPSIS ranking and perform improvement actions for A() aggregation function.
    ...
    Attributes
    ----------
    wmsd_transformer : WMSDTransformer object
    """

    def __init__(self, wmsd_transformer):
        super().__init__(wmsd_transformer)
        self.letter = 'A'

    def TOPSIS_calculation(self, w, wm, wsd):
        """Calculates TOPSIS values according to A() aggregation function.
        Parameters
        ----------
        w : TODO
            Weights.
        wm : TODO
            Weighted mean.
        wsd : TODO
            Weighted standard deviation.
        Returns
        -------
        Calculated aggregation function value.
        """
        return np.sqrt(wm * wm + wsd * wsd) / w

    def improvement_single_feature(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        feature_to_change,
        **kwargs,
    ):
        """ Exact algorithm dedicated to the aggregation `A` for achieving the target by modifying the performance on a single criterion.
        Calculates minimal change in given criterion value in order to let the alternative achieve the target position.
        Parameters
        ----------
        alternative_to_improve : int or str
            Name or position of the alternative which user wants to improve.
        alternative_to_overcome : int or str
            Name or position of the alternative which should be overcome by chosen alternative.
        epsilon : float
            Precision of calculations. Must be in range (0.0, 1.0>.
            (default : 0.000001)
        feature_to_change : str
            Name of criterion on which change should be calculated.
        Returns
        -------
        Calculated minimal change in given criterion.
        """

        performances_US = (
            alternative_to_improve.drop(labels=["Mean", "Std", str(self.letter)])
            .to_numpy()
            .copy()
        )
        performances_CS = (
            performances_US * self.wmsd_transformer._value_range
            + self.wmsd_transformer._lower_bounds
        )
        weights = self.wmsd_transformer.weights
        target_agg_value = (
            alternative_to_overcome[str(self.letter)] + epsilon / 2
        ) * np.linalg.norm(weights)

        modified_criterion_idx = list(
            alternative_to_improve.drop(labels=["Mean", "Std", str(self.letter)]).index
        ).index(feature_to_change)
        criterion_range = self.wmsd_transformer._value_range[modified_criterion_idx]
        lower_bound = self.wmsd_transformer._lower_bounds[modified_criterion_idx]
        upper_bound = lower_bound + criterion_range
        objective = self.wmsd_transformer.objectives[modified_criterion_idx]

        # Negative Ideal Solution (utility space)
        NIS = np.zeros_like(performances_US)

        v_ij = performances_US * weights
        j = modified_criterion_idx

        v_ij_excluding_j = np.delete(v_ij, j)
        NIS_excluding_j = np.delete(NIS, j)

        a = 1
        b = -2 * NIS[j]
        c = (
            NIS[j] ** 2
            + np.sum((v_ij_excluding_j - NIS_excluding_j) ** 2)
            - target_agg_value**2
        )

        solutions = solve_quadratic_equation(a, b, c)  # solutions are new performances in VS, not modifications
        if solutions is None:
            # print("Not possible to achieve target")
            return None
        else:
            # solution_1 and solution_2 -- new performances in CS
            solution_1 = ((solutions[0] / weights[j]) * criterion_range) + lower_bound
            solution_2 = ((solutions[1] / weights[j]) * criterion_range) + lower_bound

            # solution -- new performances in CS
            solution = choose_appropriate_solution(
                solution_1, solution_2, lower_bound, upper_bound, objective
            )
            if solution is None:
                return None
            else:
                feature_modification = solution - performances_CS[j]
                if self.wmsd_transformer.objectives[modified_criterion_idx] == 'min':
                    feature_modification *= -1
                modification_vector = np.zeros_like(performances_US)
                modification_vector[modified_criterion_idx] = feature_modification
                result_df = pd.DataFrame(
                    [modification_vector], columns=self.wmsd_transformer.X.columns
                )
                return result_df

    def improvement_std(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        solutions_number = 5,
        **kwargs,
    ):
        """ Calculates minimal change in standard deviation value of alternative's criteria in order to 
        let the alternative achieve the target position.
        Parameters
        ----------
        alternative_to_improve : int or str 
            Name or position of the alternative which user wants to improve.
        alternative_to_overcome : int or str 
            Name or position of the alternative which should be overcome by chosen alternative.
        epsilon : float
            Precision of calculations. Must be in range (0.0, 1.0>.
            (default : 0.000001)
        solutions_number : int
            Maximal number of proposed solutions.
            (default : 5)
        Returns
        -------
        At most [solution_number] proposed solutions.
        """
        if alternative_to_improve[str(self.letter)] >= alternative_to_overcome[str(self.letter)]:
            raise ValueError(
                "Invalid value at 'alternative_to_improve': must be worse than alternative_to_overcome'"
            )

        w = np.mean(self.wmsd_transformer.weights)
        std_start = alternative_to_improve["Std"]
        m_start = alternative_to_improve["Mean"]
        sd_boundary = self.wmsd_transformer.max_std_calculator(
            alternative_to_improve["Mean"], self.wmsd_transformer.weights
        )
        if (
            self.TOPSIS_calculation(w, alternative_to_improve["Mean"], sd_boundary)
            < alternative_to_overcome[str(self.letter)]
        ):
            return None
        else:
            change = (sd_boundary - alternative_to_improve["Std"]) / 2
            actual_aggfn = self.TOPSIS_calculation(
                w, alternative_to_improve["Mean"], alternative_to_improve["Std"]
            )
            while True:
                if actual_aggfn > alternative_to_overcome[str(self.letter)]:
                    if (
                        actual_aggfn - alternative_to_overcome[str(self.letter)]
                        > epsilon
                    ):
                        alternative_to_improve["Std"] -= change
                        change = change / 2
                        actual_aggfn = self.TOPSIS_calculation(
                            w,
                            alternative_to_improve["Mean"],
                            alternative_to_improve["Std"],
                        )
                    else:
                        break
                else:
                    alternative_to_improve["Std"] += change
                    change = change / 2
                    actual_aggfn = self.TOPSIS_calculation(
                        w, alternative_to_improve["Mean"], alternative_to_improve["Std"]
                    )
            if solutions_number is None:
                return pd.DataFrame(
                    [alternative_to_improve["Std"] - std_start], columns=["Std"]
                )
            else:
                inverse_solutions = self.wmsd_transformer.inverse_transform_numpy(alternative_to_improve["Mean"], alternative_to_improve["Std"], "==")
                reduced_solutions = reduce_population_agglomerative_clustering(inverse_solutions, solutions_number)
                result = reduced_solutions
            result_means, result_stds = self.wmsd_transformer.transform_US_to_wmsd(np.array(result))
            objectives = self.wmsd_transformer.objectives
            value_range = self.wmsd_transformer._value_range
            result -= alternative_to_improve[:-3]
            for i in result.index:
                for j in range(len(result.columns)):
                    if result[result.columns[j]][i] == 0:
                        continue
                    elif objectives[j] == "max":
                        result[result.columns[j]][i] = (
                            value_range[j] * result[result.columns[j]][i]
                        )
                    else:
                        result[result.columns[j]][i] = (
                            -value_range[j] * result[result.columns[j]][i]
                        )
            result['Mean'] = result_means - m_start
            result['Std'] = result_stds - std_start
            return result


class ITOPSIS(TOPSISAggregationFunction):
    """
    A class used to calculate TOPSIS ranking and perform improvement actions for I() aggregation function.
    ...
    Attributes
    ----------
    wmsd_transformer : WMSDTransformer object
    """

    def __init__(self, wmsd_transformer):
        super().__init__(wmsd_transformer)
        self.letter = 'I'

    def TOPSIS_calculation(self, w, wm, wsd):
        """Calculates TOPSIS values according to I() aggregation function.
        Parameters
        ----------
        w : TODO
            Weights.
        wm : TODO
            Weighted mean.
        wsd : TODO
            Weighted standard deviation.
        Returns
        -------
        Calculated aggregation function value.
        """
        return 1 - np.sqrt((w - wm) * (w - wm) + wsd * wsd) / w

    def improvement_single_feature(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        feature_to_change,
        **kwargs,
    ):
        """ 
        Exact algorithm dedicated to the aggregation `A` for achieving the target by modifying the performance on a single criterion.
        Calculates minimal change in given criterion value in order to let the alternative achieve the target position.
        Parameters
        ----------
        alternative_to_improve : int or str
            Name or position of the alternative which user wants to improve.
        alternative_to_overcome : int or str
            Name or position of the alternative which should be overcome by chosen alternative.
        epsilon : float
            Precision of calculations. Must be in range (0.0, 1.0>.
            (default : 0.000001)
        feature_to_change : str
            Name of criterion on which change should be calculated.
        Returns
        -------
        Calculated minimal change in given criterion.
        """

        performances_US = (
            alternative_to_improve.drop(labels=["Mean", "Std", str(self.letter)])
            .to_numpy()
            .copy()
        )
        performances_CS = (
            performances_US * self.wmsd_transformer._value_range
            + self.wmsd_transformer._lower_bounds
        )
        weights = self.wmsd_transformer.weights
        target_agg_value = (
            1 - (alternative_to_overcome[str(self.letter)] + epsilon / 2)
        ) * np.linalg.norm(weights)

        modified_criterion_idx = list(
            alternative_to_improve.drop(labels=["Mean", "Std", str(self.letter)]).index
        ).index(feature_to_change)
        criterion_range = self.wmsd_transformer._value_range[modified_criterion_idx]
        lower_bound = self.wmsd_transformer._lower_bounds[modified_criterion_idx]
        upper_bound = lower_bound + criterion_range
        objective = self.wmsd_transformer.objectives[modified_criterion_idx]

        # Positive Ideal Solution (utility space)
        PIS = weights

        v_ij = performances_US * weights
        j = modified_criterion_idx

        v_ij_excluding_j = np.delete(v_ij, j)
        PIS_excluding_j = np.delete(PIS, j)

        a = 1
        b = -2 * PIS[j]
        c = (
            PIS[j] ** 2
            + np.sum((v_ij_excluding_j - PIS_excluding_j) ** 2)
            - target_agg_value**2
        )

        solutions = solve_quadratic_equation(a, b, c)  # solutions are new performances in VS, not modifications
        if solutions is None:
            # print("Not possible to achieve target")
            return None
        else:
            # solution_1 and solution_2 -- new performances in CS
            solution_1 = ((solutions[0] / weights[j]) * criterion_range) + lower_bound
            solution_2 = ((solutions[1] / weights[j]) * criterion_range) + lower_bound

            # solution -- new performances in CS
            solution = choose_appropriate_solution(
                solution_1, solution_2, lower_bound, upper_bound, objective
            )
            if solution is None:
                return None
            else:
                feature_modification = solution - performances_CS[j]
                if self.wmsd_transformer.objectives[modified_criterion_idx] == 'min':
                    feature_modification *= -1
                modification_vector = np.zeros_like(performances_US)
                modification_vector[modified_criterion_idx] = feature_modification
                result_df = pd.DataFrame(
                    [modification_vector], columns=self.wmsd_transformer.X.columns
                )
                return result_df

    def improvement_std(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        solutions_number=5,
        **kwargs,
    ):
        """ Calculates minimal change in standard deviation value of alternative's criteria in order to 
        let the alternative achieve the target position.
        Parameters
        ----------
        alternative_to_improve : int or str
            Name or position of the alternative which user wants to improve.
        alternative_to_overcome : int or str
            Name or position of the alternative which should be overcome by chosen alternative.
        epsilon : float
            Precision of calculations. Must be in range (0.0, 1.0>.
            (default : 0.000001)
        solutions_number : int
            Maximal number of proposed solutions.
            (default : 5)
        Returns
        -------
        At most [solution_number] proposed solutions.
        """
        if alternative_to_improve[str(self.letter)] >= alternative_to_overcome[str(self.letter)]:
            raise ValueError(
                "Invalid value at 'alternative_to_improve': must be worse than alternative_to_overcome'"
            )

        w = np.mean(self.wmsd_transformer.weights)
        std_start = alternative_to_improve["Std"]
        m_start = alternative_to_improve["Mean"]
        sd_boundary = self.wmsd_transformer.max_std_calculator(
            alternative_to_improve["Mean"], self.wmsd_transformer.weights
        )
        if (
            self.TOPSIS_calculation(w, alternative_to_improve["Mean"], 0)
            < alternative_to_overcome[str(self.letter)]
        ):
            return None
        else:
            change = alternative_to_improve["Std"] / 2
            actual_aggfn = self.TOPSIS_calculation(
                w, alternative_to_improve["Mean"], alternative_to_improve["Std"]
            )
            while True:
                if actual_aggfn > alternative_to_overcome[str(self.letter)]:
                    if (
                        actual_aggfn - alternative_to_overcome[str(self.letter)]
                        > epsilon
                    ):
                        alternative_to_improve["Std"] += change
                        change = change / 2
                        actual_aggfn = self.TOPSIS_calculation(
                            w,
                            alternative_to_improve["Mean"],
                            alternative_to_improve["Std"],
                        )
                    else:
                        break
                else:
                    alternative_to_improve["Std"] -= change
                    change = change / 2
                    actual_aggfn = self.TOPSIS_calculation(
                        w, alternative_to_improve["Mean"], alternative_to_improve["Std"]
                    )
            if solutions_number is None:
                return pd.DataFrame(
                    [alternative_to_improve["Std"] - std_start], columns=["Std"]
                )
            else:
                inverse_solutions = self.wmsd_transformer.inverse_transform_numpy(alternative_to_improve["Mean"], alternative_to_improve["Std"], "==")
                reduced_solutions = reduce_population_agglomerative_clustering(inverse_solutions, solutions_number)
                result = reduced_solutions
            result_means, result_stds = self.wmsd_transformer.transform_US_to_wmsd(np.array(result))
            objectives = self.wmsd_transformer.objectives
            value_range = self.wmsd_transformer._value_range
            result -= alternative_to_improve[:-3]
            for i in result.index:
                for j in range(len(result.columns)):
                    if result[result.columns[j]][i] == 0:
                        continue
                    elif objectives[j] == "max":
                        result[result.columns[j]][i] = (
                            value_range[j] * result[result.columns[j]][i]
                        )
                    else:
                        result[result.columns[j]][i] = (
                            -value_range[j] * result[result.columns[j]][i]
                        )
            result['Mean'] = result_means - m_start
            result['Std'] = result_stds - std_start
            return result


class RTOPSIS(TOPSISAggregationFunction):
    """
    A class used to calculate TOPSIS ranking and perform improvement actions for R() aggregation function.
    ...
    Attributes
    ----------
    wmsd_transformer : WMSDTransformer object
    """

    def __init__(self, wmsd_transformer):
        super().__init__(wmsd_transformer)
        self.letter = 'R'

    def TOPSIS_calculation(self, w, wm, wsd):
        """Calculates TOPSIS values according to R() aggregation function.
        Parameters
        ----------
        w : TODO
            Weights.
        wm : TODO
            Weighted mean.
        wsd : TODO
            Weighted standard deviation.
        Returns
        -------
        Calculated aggregation function value.
        """
        return np.sqrt(wm * wm + wsd * wsd) / (
            np.sqrt(wm * wm + wsd * wsd) + np.sqrt((w - wm) * (w - wm) + wsd * wsd)
        )

    def build_nlp_solver(
        self,
        performances_US,
        target_score,
        excluded_criteria_indices,
        upper_bounds_US,
        constant_WM=False,
    ):
        return TopsisNLPPostFactum(
            performances_US=performances_US,
            weights=self.wmsd_transformer.weights,
            target_R_value=target_score,
            excluded_criteria_indices=excluded_criteria_indices,
            upper_bounds_US=upper_bounds_US,
            constant_WM=constant_WM,
        )

    def improvement_single_feature(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        feature_to_change,
        **kwargs,
    ):
        """
        Exact algorithm dedicated to the aggregation `R` for achieving the target by modifying the performance on a single criterion.
        Calculates minimal change in given criterion value in order to let the alternative achieve the target position.
        Parameters
        ----------
        alternative_to_improve : int or str
            Name or position of the alternative which user wants to improve.
        alternative_to_overcome : int or str
            Name or position of the alternative which should be overcome by chosen alternative.
        epsilon : float
            Precision of calculations. Must be in range (0.0, 1.0>.
            (default : 0.000001)
        feature_to_change : str
            Name of criterion on which change should be calculated.
        Returns
        -------
        Calculated minimal change in given criterion.
        """
        performances_US = (
            alternative_to_improve.drop(labels=["Mean", "Std", str(self.letter)])
            .to_numpy()
            .copy()
        )
        performances_CS = (
            performances_US * self.wmsd_transformer._value_range
            + self.wmsd_transformer._lower_bounds
        )
        weights = self.wmsd_transformer.weights
        target_agg_value = alternative_to_overcome[str(self.letter)] + epsilon / 2

        modified_criterion_idx = list(
            alternative_to_improve.drop(labels=["Mean", "Std", str(self.letter)]).index
        ).index(feature_to_change)
        criterion_range = self.wmsd_transformer._value_range[modified_criterion_idx]
        lower_bound = self.wmsd_transformer._lower_bounds[modified_criterion_idx]
        upper_bound = lower_bound + criterion_range
        objective = self.wmsd_transformer.objectives[modified_criterion_idx]

        # Positive and Negative Ideal Solution (utility space)
        PIS = weights
        NIS = np.zeros_like(performances_US)

        v_ij = performances_US * weights
        j = modified_criterion_idx

        # Calculate the sum of squared distances for the remaining (unmodified) criteria
        v_ij_excluding_j = np.delete(v_ij, j)
        PIS_excluding_j = np.delete(PIS, j)
        NIS_excluding_j = np.delete(NIS, j)
        k = (target_agg_value / (1 - target_agg_value)) ** 2
        p = k * np.sum((v_ij_excluding_j - PIS_excluding_j) ** 2) - np.sum(
            (v_ij_excluding_j - NIS_excluding_j) ** 2
        )

        a = (1 - k) * (weights[j] / criterion_range) ** 2
        b = (
            2
            * (weights[j] / criterion_range)
            * (v_ij[j] - NIS[j] - k * (v_ij[j] - PIS[j]))
        )
        c = (v_ij[j] - NIS[j]) ** 2 - k * (v_ij[j] - PIS[j]) ** 2 - p

        solutions = solve_quadratic_equation(a, b, c)  # solutions are performance modifications in CS !!!
        if solutions is None:
            # print("Not possible to achieve target")
            return None
        else:
            # solution_1 and solution_2 -- new performances in CS
            solution_1 = solutions[0] + performances_CS[j]
            solution_2 = solutions[1] + performances_CS[j]

        # solution -- new performances in CS
        solution = choose_appropriate_solution(
            solution_1, solution_2, lower_bound, upper_bound, objective
        )
        if solution is None:
            return None
        else:
            feature_modification = solution - performances_CS[j]
            if self.wmsd_transformer.objectives[modified_criterion_idx] == 'min':
                feature_modification *= -1
            modification_vector = np.zeros_like(performances_US)
            modification_vector[modified_criterion_idx] = feature_modification
            result_df = pd.DataFrame(
                [modification_vector], columns=self.wmsd_transformer.X.columns
            )
            return result_df

    def improvement_non_linear_programming(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        features_to_change,
        boundary_values=None, # TODO
        constant_WM=False,
        **kwargs,
    ):


        # alternative_to_improve : int or str
        #     Name or position of the alternative which user wants to improve.
        # alternative_to_overcome : int or str
        #     Name or position of the alternative which should be overcome by chosen alternative.

        # epsilon : float
        #     Precision of calculations. Must be in range (0.0, 1.0>.
        #     (default : 0.000001)
        # features_to_change : array of str
        #     Array containing names of criteria on which change should be calculated.
        # boundary_values : 2D array of floats
        #     Array with dimensions number_of_features_to_change x 2. For each feature to change it should
        #     have provided 2 numbers: lower and upper boundaries of proposed values.
        #     (default : None)

        """
        Non-linear programming based exact algorithm dedicated to the aggregation `R` for achieving the target
        by modifying the performance on multiple criteria
        Calculates minimal change in given criterion value in order to let the alternative achieve the target position.

        Parameters
        ----------
        alternative_to_improve : int or str
            Name or position of the alternative which user wants to improve.
        alternative_to_overcome : int or str
            Name or position of the alternative which should be overcome by chosen alternative.
        epsilon : float
            Precision of calculations. Must be in range (0.0, 1.0>.
            (default : 0.000001)
        features_to_change : array of str
            Array containing names of criteria on which change should be calculated.
        boundary_values : 2D array of floats
            Array with dimensions number_of_features_to_change x 2. For each feature to change it should
            have provided 2 numbers: lower and upper boundaries of proposed values.
            (default : None)
        constant_WM : bool
            Indicates whether weight scale mean should remain unchanged after applying proposed modifications
            (default : False)

        Returns
        -------
        Calculated minimum change on given criteria.
        """
        return super().improvement_non_linear_programming(
            alternative_to_improve=alternative_to_improve,
            alternative_to_overcome=alternative_to_overcome,
            epsilon=epsilon,
            features_to_change=features_to_change,
            boundary_values=boundary_values,
            constant_WM=constant_WM,
            **kwargs,
        )


    def improvement_std(
        self,
        alternative_to_improve,
        alternative_to_overcome,
        epsilon,
        solutions_number=5,
        **kwargs,
    ):
        """ Calculates minimal change in standard deviation value of alternative's criteria in order to 
        let the alternative achieve the target position.
        Parameters
        ----------
        alternative_to_improve : int or str
            Name or position of the alternative which user wants to improve.
        alternative_to_overcome : int or str
            Name or position of the alternative which should be overcome by chosen alternative.
        epsilon : float
            Precision of calculations. Must be in range (0.0, 1.0>.
            (default : 0.000001)
        solutions_number : int
            Maximal number of proposed solutions.
            (default : 5)
        Returns
        -------
        At most [solution_number] proposed solutions.
        """
        if alternative_to_improve[str(self.letter)] >= alternative_to_overcome[str(self.letter)]:
            raise ValueError(
                "Invalid value at 'alternatie_to_improve': must be worse than alternative_to_overcome'"
            )

        w = np.mean(self.wmsd_transformer.weights)
        std_start = alternative_to_improve["Std"]
        m_start = alternative_to_improve["Mean"]
        sd_boundary = self.wmsd_transformer.max_std_calculator(
            alternative_to_improve["Mean"], self.wmsd_transformer.weights
        )
        if alternative_to_improve["Mean"] < w / 2:
            if (
                self.TOPSIS_calculation(w, alternative_to_improve["Mean"], sd_boundary)
                < alternative_to_overcome[str(self.letter)]
            ):
                return None
            else:
                change = (sd_boundary - alternative_to_improve["Std"]) / 2
                actual_aggfn = self.TOPSIS_calculation(
                    w, alternative_to_improve["Mean"], alternative_to_improve["Std"]
                )
                while True:
                    if actual_aggfn > alternative_to_overcome[str(self.letter)]:
                        if (
                            actual_aggfn - alternative_to_overcome[str(self.letter)]
                            > epsilon
                        ):
                            alternative_to_improve["Std"] -= change
                            change = change / 2
                            actual_aggfn = self.TOPSIS_calculation(
                                w,
                                alternative_to_improve["Mean"],
                                alternative_to_improve["Std"],
                            )
                        else:
                            break
                    else:
                        alternative_to_improve["Std"] += change
                        change = change / 2
                        actual_aggfn = self.TOPSIS_calculation(
                            w,
                            alternative_to_improve["Mean"],
                            alternative_to_improve["Std"],
                        )
                if solutions_number is None:
                    return pd.DataFrame(
                        [alternative_to_improve["Std"] - std_start],
                        columns=["Improvement rate"],
                        index=["Std"],
                    )
                else:
                    inverse_solutions = self.wmsd_transformer.inverse_transform_numpy(alternative_to_improve["Mean"], alternative_to_improve["Std"], "==")
                    reduced_solutions = reduce_population_agglomerative_clustering(inverse_solutions, solutions_number)
                    result = reduced_solutions
        else:
            if (
                self.TOPSIS_calculation(w, alternative_to_improve["Mean"], 0)
                < alternative_to_overcome[str(self.letter)]
            ):
                return None
            else:
                change = alternative_to_improve["Std"] / 2
                actual_aggfn = self.TOPSIS_calculation(
                    w, alternative_to_improve["Mean"], alternative_to_improve["Std"]
                )
                while True:
                    if actual_aggfn > alternative_to_overcome[str(self.letter)]:
                        if (
                            actual_aggfn - alternative_to_overcome[str(self.letter)]
                            > epsilon
                        ):
                            alternative_to_improve["Std"] += change
                            change = change / 2
                            actual_aggfn = self.TOPSIS_calculation(
                                w,
                                alternative_to_improve["Mean"],
                                alternative_to_improve["Std"],
                            )
                        else:
                            break
                    else:
                        alternative_to_improve["Std"] -= change
                        change = change / 2
                        actual_aggfn = self.TOPSIS_calculation(
                            w,
                            alternative_to_improve["Mean"],
                            alternative_to_improve["Std"],
                        )
                if solutions_number is None:
                    return pd.DataFrame(
                        [alternative_to_improve["Std"] - std_start], columns=["Std"]
                    )
                else:
                    inverse_solutions = self.wmsd_transformer.inverse_transform_numpy(alternative_to_improve["Mean"], alternative_to_improve["Std"], "==")
                    reduced_solutions = reduce_population_agglomerative_clustering(inverse_solutions, solutions_number)
                    result = reduced_solutions
        result_means, result_stds = self.wmsd_transformer.transform_US_to_wmsd(np.array(result))
        objectives = self.wmsd_transformer.objectives
        value_range = self.wmsd_transformer._value_range
        result -= alternative_to_improve[:-3]
        for i in result.index:
            for j in range(len(result.columns)):
                if result[result.columns[j]][i] == 0:
                    continue
                elif objectives[j] == "max":
                    result[result.columns[j]][i] = (
                        value_range[j] * result[result.columns[j]][i]
                    )
                else:
                    result[result.columns[j]][i] = (
                        -value_range[j] * result[result.columns[j]][i]
                    )
        result['Mean'] = result_means - m_start
        result['Std'] = result_stds - std_start
        return result



