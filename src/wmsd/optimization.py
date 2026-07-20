import numpy as np
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from tqdm import tqdm

class TqdmProgressBar(tqdm):
    def update_to(self, current, total):
        self.total = total
        self.update(current - self.n)


class MyProgressBar(Callback):
    def __init__(self, n_gen, subset_str) -> None:
        super().__init__()
        self.progress_bar = TqdmProgressBar()
        self.progress_bar.set_description(f"{subset_str}")
        self.n_gen = n_gen
        # self.checkpoint_every = self.n_gen // 20 if self.n_gen >= 40 else 2
        self.checkpoint_every = 1
        self.checkpoints = {}

    def notify(self, algorithm):
        if algorithm.n_iter % self.checkpoint_every == 1 or self.checkpoint_every == 1:
            self.checkpoints[f"gen_{algorithm.n_gen}_checkpoint"] = {
                "CV": algorithm.opt.get("CV"),
                "F": algorithm.opt.get("F"),
                "G": algorithm.opt.get("G"),
                "X": algorithm.opt.get("X")
            }
        self.progress_bar.update_to(algorithm.n_iter, self.n_gen)


class PostFactumAggregationPymoo(Problem):
    """
    Class description
    ...
    Attributes
    ----------

    topsis_model : object
        Object with methods to calculate weighted means, weighted standard deviations and aggregation values (e.g. WMSDTransformer object).
    modified_criteria_subset : numpy array of bools
        Used to slice numpy arrays.
    current_performances : object
        Current performances of alternative in US (utility space)
    target_agg_value : object
        TODO description
    upper_bounds : 2D array of floats
        Array with dimensions number_of_features_to_change x 2. For each feature to change it should
        have provided 2 numbers: lower and upper boundaries of proposed values.
        (default : None)
    """

    def __init__(
        self,
        aggregation_model,
        modified_criteria_subset,
        current_performances,
        target_agg_value,
        upper_bounds,
        allow_deterioration=False,
    ):
        n_criteria = np.array(modified_criteria_subset).astype(bool).sum()
        super().__init__(
            n_var=n_criteria, n_obj=n_criteria, n_ieq_constr=1, vtype=float
        )

        self.aggregation_model = aggregation_model
        self.modified_criteria_subset = np.array(modified_criteria_subset).astype(bool)
        self.current_performances = current_performances.copy()
        self.target_agg_value = target_agg_value

        # Lower and upper bounds in Utility Space
        self.xl = (
            np.zeros(n_criteria)
            if allow_deterioration
            else self.current_performances[self.modified_criteria_subset]
        )
        self.xu = upper_bounds

    def _evaluate(self, x, out, *args, **kwargs):
        # In Utility Space variables and objectives are the same values
        out["F"] = x.copy()  # this copy might be redundant

        # Topsis target constraint
        modified_performances = np.repeat(
            [self.current_performances], repeats=len(x), axis=0
        )
        modified_performances[
            :, self.modified_criteria_subset
        ] = x.copy()  # this copy might be redundant
        agg_values = self.aggregation_model.agg_fn.score_batch(modified_performances)
        g1 = (
            self.target_agg_value - agg_values
        )  # In Pymoo positive values indicate constraint violation
        out["G"] = np.array([g1])


class PostFactumTopsisPymoo(PostFactumAggregationPymoo):
    pass

