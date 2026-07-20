import math
import numba
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
import plotly.graph_objects as go
from joblib import Parallel, delayed

from .aggregation import (
    AggregationFunction,
    ARAS,
    ATOPSIS,
    COPRAS,
    ITOPSIS,
    RTOPSIS,
    SAW,
    VIKOR,
    WASPAS,
    WMSDAggregationFunction,
)


class WMSDTransformer(TransformerMixin):
    """
    A class used to calculate TOPSIS ranking,
    plot positions of alternatives in WMSD space,
    perform improvement actions on selected alternative.

    X : data-frame
        Pandas data-frame provided by the user.
    X_new : data-frame
        Pandas data-frame, normalized X.
    data : data-frame
        A copy of self.X, on which all calculations are performed.
    n : int
        Number of data-frame's columns
    m : int
        Number of data-frame's rows
    weights : np.array of float
        Array containing normalized weights.
    objectives : np.array of str
        Numpy array informing which criteria are cost type
        and which are gain type.
    expert_range : 2D list of floats
        2D list containing normalized expert range.
    """

    def __init__(self, agg_fn, max_std_calculator="scip", n_jobs=None):
        self.agg_fn = self.__check_agg_fn(agg_fn)
        self.max_std_calculator = self.__check_max_std_calculator(max_std_calculator)
        self._isFitted = False
        self.n_jobs = n_jobs

    def fit(self, X, weights=None, objectives=None, expert_range=None):
        """Checks input data and normalizes it.

        Parameters
        ----------
        X : pandas DataFrame
            Pandas data-frame provided by the user.
            All values must be numerical, excluding column and row names.
        weights : list or dict, default=None
            Weights of the criteria - preference information provided by the user.
            Numpy array of a length equal to the number of criteria.
            TODO dict ...
            If None, then criteria are equally weighted.
        objectives : list or dict or str, default=None
            Specifies whether criteria are cost or gain type.
            It can be passed as:
            - a list of length equal to the number of criteria, in which each element describes type of one criterion:
            'cost'/'c'/'min' for cost type criteria and 'gain'/'g'/'max' for gain type criteria.
            - dictionary of size equal to the number of criteria in which each key is the criterion name and each value
            takes one of the following values: 'cost'/'c'/'min' for cost type criteria
            and 'gain'/'g'/'max' for gain type criteria.
            - a string which describes type of all criteria:
            'cost'/'c'/'min' if criteria are cost type and 'gain'/'g'/'max' if criteria are gain type.
            If None, then all criteria are of a gain type.
        expert_range : 2D list or dictionary, default=None
            For each criterion must be provided minimal and maximal value.
            All criteria must fit in range [minimal, maximal]
            If None, then expert_range is derived from the extreme values observed in the dataset

        Returns
        -------
        self : WMSDTransformer
            Fitted transformer.
        """

        self.X = X
        self.n = X.shape[0]  # n_alternatives
        self.m = X.shape[1]  # n_criteria

        self._original_weights = self.__check_weights(weights)
        self.weights = self._original_weights.copy()
        self.weights = self.__normalize_weights(self.weights)

        # preference directions of criteria (max for gain-type criteria, min for cost-type criteria)
        self.objectives = self.__check_objectives(objectives)
        self.objectives = list(map(lambda x: x.replace("gain", "max"), self.objectives))
        self.objectives = list(map(lambda x: x.replace("g", "max"), self.objectives))
        self.objectives = list(map(lambda x: x.replace("cost", "min"), self.objectives))
        self.objectives = list(map(lambda x: x.replace("c", "min"), self.objectives))

        # domains of criteria values, either provided by user or derived from dataset
        self.expert_range = self.__check_expert_range(expert_range)

        self.__check_input()

        self._value_range = []
        self._lower_bounds = []
        self._upper_bounds = []
        for c in range(self.m):
            self._lower_bounds.append(self.expert_range[c][0])
            self._upper_bounds.append(self.expert_range[c][1])
            self._value_range.append(self.expert_range[c][1] - self.expert_range[c][0])

        self.X_new = self.__normalize_data(X.copy()) # min-max scaling
        self.__wmstd() # adds two columns to DataFrame Mean and Std
        self.X_new[str(self.agg_fn.letter)] = self.agg_fn.score_batch(
            np.array(self.X_new.loc[:, self.X.columns], dtype=float)
        )
        self._ranked_alternatives = self.__ranking()
        self._isFitted = True

        return self

    def transform(self, X):
        """Transform data from data-frame X to WMSD space.

        Parameters
        ----------
        X : pandas DataFrame
            Pandas data-frame provided by the user.
            All values must be numerical, excluding column and row names.

        Returns
        -------
        X_transformed: pandas DataFrame
        """

        if not self._isFitted:
            raise Exception("fit is required before transform")

        self.__check_input_after_transform(X)
        X_transformed = self.__normalize_data(X.copy())
        w_means, w_stds = self.transform_US_to_wmsd(np.array(X_transformed))
        agg_values = self.agg_fn.score_batch(
            np.array(X_transformed.loc[:, self.X.columns], dtype=float)
        )
        X_transformed["Mean"] = w_means
        X_transformed["Std"] = w_stds
        X_transformed[str(self.agg_fn.letter)] = agg_values

        return X_transformed
    
    def fit_transform(self, X, weights=None, objectives=None, expert_range=None):
        """Fit to data, then transform it.

        Parameters
        ----------
        X : pandas DataFrame
            Pandas data-frame provided by the user.
            All values must be numerical, excluding column and row names.
        weights : np.array of dict, default=None
            Weights of the criteria - preference information provided by the user.
            Numpy array of a length equal to the number of criteria.
            TODO dict ...
            If None, then criteria are equally weighted.
        objectives : list or dict or str, default=None
            Specifies whether criteria are cost or gain type.
            It can be passed as:
            - a list of length equal to the number of criteria, in which each element describes type of one criterion:
            'cost'/'c'/'min' for cost type criteria and 'gain'/'g'/'max' for gain type criteria.
            - dictionary of size equal to the number of criteria in which each key is the criterion name and each value
            takes one of the following values: 'cost'/'c'/'min' for cost type criteria
            and 'gain'/'g'/'max' for gain type criteria.
            - a string which describes type of all criteria:
            'cost'/'c'/'min' if criteria are cost type and 'gain'/'g'/'max' if criteria are gain type.
            If None, then all criteria are of a gain type.
        expert_range : 2D list or dictionary, default=None
            For each criterion must be provided minimal and maximal value.
            All criteria must fit in range [minimal, maximal]
            If None, then expert_range is derived from the extreme values observed in the dataset

        Returns
        -------
        X_new : pandas DataFrame
        """
        self.fit(X, weights, objectives, expert_range)
        return self.X_new

    def transform_US_to_wmsd(self, X_US):
        """Transforms data from Utility Space to weighted MSD Space.

        Parameters
        ----------
        X_US : pandas DataFrame
            Pandas data-frame where data are presented in Utility Space.

        Returns
        -------
        Norms w_means and w_stds.
        """
        # transform data from Utility Space to WMSD Space
        w = self.weights
        s = np.linalg.norm(w) / np.mean(w)
        v = X_US * w

        vw = (np.sum(v * w, axis=1) / np.sum(w**2)).reshape(-1, 1) @ w.reshape(1, -1)
        w_means = np.linalg.norm(vw, axis=1) / s
        w_stds = np.linalg.norm(v - vw, axis=1) / s
        return w_means, w_stds

    def inverse_transform_numpy(self, target_mean, target_std, std_type='==', sampling_density=None, epsilon=0.01, verbose=False):
        """
        Find possible performance vectors (i.e., vectors of artificial alternatives' evaluations) in US space
        for which the weighted mean (WM) and weight-scaled standard deviation (WSD) are close to the expected values.
        The number of feasible solutions grows exponentially with the dimensionality (number of criteria) of
        the dataset. Computing the exact values of all solutions is computationally expensive, particularly
        for high-dimensional datasets. Therefore, this method samples the US space to identify a set of points
        that are sufficiently close (within distance of `epsilon` or less) to the expected values of `target_mean`
        and `target_std`.
        This method utilizes NumPy vectorization to enhance performance. However, it requires storing all samples in RAM,
        which may not be possible for a large number of dimensions and a high value of the `sampling_density` parameter.

        Parameters
        ----------
        target_mean : float
            The expected value of the weighted mean score for the returned solutions (performance vectors).

        target_std : float
            The expected value of the weight-scaled standard deviation for the returned solutions (performance vectors).

        std_type : str, default='=='
            The nature WSD criterion varies depending on the aggregation function used and the WM. It might be
            considered as a gain-type or a cost-type criterion. By default, the method assumes that the WSD should
            be as close as possible to `target_std' ('=='). The value '<=' means that the WSD is a cost-type criterion,
            and therefore solutions that do not exceed `target_std` will be returned (larger deviations in the other
            direction, i.e. towards smallerWSD values, are acceptable). The symbol '>=' indicates that WSD is a gain-type
            criterion. Therefore, the returned solutions will exceed `target_std` (larger deviations in the other direction,
            i.e. towards larger WSD values, are acceptable).
            Must be one of following strings '==', '<=', '>='.

        sampling_density : int or None, default=None
            The `sampling_density` parameter determines how densely the Utility Space is sampled.
            By default, i.e., when the `sampling_density=None`, the value of `sampling_density`
            is calculated based on the dimensionality of the dataset.

        epsilon : float, default=0.01
            Maximum deviation of WM and WSD (when `std_type='==') or WM (otherwise) from target values.
            Must be in range (0.0, 1.0].

        verbose : bool, default=False
            When the value of this parameter is set to True, the method provides information about the total
            number of sampled solutions, RAM consumption, and the number of returned solutions.

        Returns
        -------
        solutions: DataFrame or None
            The method returns a DataFrame containing performance vectors that meet the requirements specified by
            `target_mean`, `target_std`, `std_type` and `epsilon`, or None if no points satisfying these requirements
            are found. In the latter scenario, it may be helpful to increase the value of `epsilon` at the expense of
            lower accuracy.
        """

        if std_type not in ['==', '<=', '>=']:
            raise ValueError("Invalid value at `std_type`, should be one of the following strings '==', '<=', '>='")

        if sampling_density is None:
            sampling_density = math.ceil(5000000 ** (1 / self.m))

        # TODO Possible enhancement: The method can automatically select the epsilon parameter.
        #  If many points meet the requirements, the method can be rerun with a smaller tolerance.
        #  If no solution is found, the method can be rerun with a larger tolerance.

        dims = [np.linspace(0, 1, sampling_density, dtype=np.float32) for i in range(self.m)]
        grid = np.meshgrid(*dims)
        points = np.column_stack([xx.ravel() for xx in grid])
        if verbose:
            print(f"inverse_transform_numpy: sampling_density: {sampling_density}")
            print(f"inverse_transform_numpy: {len(points)} samples generated in total")
            print(f"inverse_transform_numpy: RAM usage for points: {points.nbytes / 1024 / 1024} MiB")
        w_means, w_stds = self.transform_US_to_wmsd(points)

        if std_type == "==":
            filtered_points = points[np.bitwise_and(abs(w_means - target_mean) < epsilon, abs(target_std - w_stds) < epsilon)]
        elif std_type == "<=":
            filtered_points = points[np.bitwise_and(abs(w_means - target_mean) < epsilon, w_stds <= target_std)]
        else:  # std_type == ">="
            filtered_points = points[np.bitwise_and(abs(w_means - target_mean) < epsilon, w_stds >= target_std)]

        if verbose:
            print(f"inverse_transform_numpy: Returning {filtered_points.shape[0]} solutions")

        if filtered_points.shape[0] == 0:
            solutions = None
        else:
            solutions = pd.DataFrame(filtered_points, columns=self.X.columns)

        return solutions

    def inverse_transform_numba(self, target_mean, target_std, std_type='==', sampling_density=None, epsilon=0.01, verbose=False):
        """
        Find possible performance vectors (i.e., vectors of artificial alternatives' evaluations) in US space
        for which the weighted mean (WM) and weight-scaled standard deviation (WSD) are close to the expected values.
        The number of feasible solutions grows exponentially with the dimensionality (number of criteria) of
        the dataset. Computing the exact values of all solutions is computationally expensive, particularly
        for high-dimensional datasets. Therefore, this method samples the US space to identify a set of points
        that are sufficiently close (within distance of `epsilon` or less) to the expected values of `target_mean`
        and `target_std`.
        This method utilizes a just-in-time compiler Numba to enhance performance without requiring to store all samples in RAM.

        Parameters
        ----------
        target_mean : float
            The expected value of the weighted mean score for the returned solutions (performance vectors).

        target_std : float
            The expected value of the weight-scaled standard deviation for the returned solutions (performance vectors).

        std_type : str, default='=='
            The nature WSD criterion varies depending on the aggregation function used and the WM. It might be
            considered as a gain-type or a cost-type criterion. By default, the method assumes that the WSD should
            be as close as possible to `target_std' ('=='). The value '<=' means that the WSD is a cost-type criterion,
            and therefore solutions that do not exceed `target_std` will be returned (larger deviations in the other
            direction, i.e. towards smallerWSD values, are acceptable). The symbol '>=' indicates that WSD is a gain-type
            criterion. Therefore, the returned solutions will exceed `target_std` (larger deviations in the other direction,
            i.e. towards larger WSD values, are acceptable).
            Must be one of following strings '==', '<=', '>='.

        sampling_density : int or None, default=None
            The `sampling_density` parameter determines how densely the Utility Space is sampled.
            By default, i.e., when the `sampling_density=None`, the value of `sampling_density`
            is calculated based on the dimensionality of the dataset.

        epsilon : float, default=0.01
            Maximum deviation of WM and WSD (when `std_type='==') or WM (otherwise) from target values.
            Must be in range (0.0, 1.0].

        verbose : bool, default=False
            When the value of this parameter is set to True, the method provides information about the total
            number of sampled solutions, RAM consumption, and the number of returned solutions.

        Returns
        -------
        solutions: DataFrame or None
            The method returns a DataFrame containing performance vectors that meet the requirements specified by
            `target_mean`, `target_std`, `std_type` and `epsilon`, or None if no points satisfying these requirements
            are found. In the latter scenario, it may be helpful to increase the value of `epsilon` at the expense of
            lower accuracy.
        """

        from utils.numba_inverse_transform import inverse_transform
        filtered_points = inverse_transform(target_mean, target_std, self.weights, std_type, sampling_density, epsilon, verbose)

        if len(filtered_points) == 0:
            solutions = None
        else:
            solutions = pd.DataFrame(filtered_points, columns=self.X.columns)

        return solutions

    def __ensure_wmsd_aggregation(self, function_name):
        if not isinstance(self.agg_fn, WMSDAggregationFunction):
            raise NotImplementedError(
                f"{function_name} is available only for WMSD-based aggregation functions."
            )

    def plot(self, heatmap_quality=500, show_names=False, plot_name=None, color='jet'):
        """Plots positions of alternatives in WMSD space.

        Parameters
        ----------
        heatmap_quality : int, optional
            Integer value of the precision of the heatmap. The higher, the heatmap more fills the plot, but the plot generates longer.
            (default: 500)
        show_names : bool, optional
            Boolean value, if true, then points labels are showed on a plot.
            (default: False)
        plot_name : str, optional
            String that contains a title of a plot. If None, then in place of title, weights of criteria will be shown.
            (default: None)
        color : str, optional
            String that contains a name of a color-scale in which the plot will be presented.
            (default: 'jet')

        Returns
        -------
        Plot as a plotly figure.
        """
        self.__ensure_wmsd_aggregation("plot")

        # for all possible mean and std count aggregation value and color it by it
        mean_ticks = np.linspace(0, 1, heatmap_quality + 1)
        std_ticks = np.linspace(0, 0.5, heatmap_quality // 2 + 1)
        grid = np.meshgrid(mean_ticks, std_ticks)
        w_means = grid[0].ravel()
        w_stds = grid[1].ravel()
        agg_values = self.agg_fn.TOPSIS_calculation(
            np.mean(self.weights), np.array(w_means), np.array(w_stds)
        )
        if plot_name is None:
            plot_name = "Weights: " + ",".join([f"{x:.3f}" for x in self.weights])

        fig = go.Figure(
            data=go.Contour(
                x=w_means,
                y=w_stds,
                z=agg_values,
                zmin=0.0,
                zmax=1.0,
                colorscale=color,
                contours_coloring="heatmap",
                line_width=0,
                colorbar=dict(
                    title=str(self.agg_fn.letter + '(v)'),
                    titleside="right",
                    outlinewidth=1,
                    title_font_size=22,
                    tickfont_size=15,
                ),
                hoverinfo="none",
            ),
            layout=go.Layout(
                title=go.layout.Title(text=plot_name, font_size=30),
                title_x=0.5,
                xaxis_range=[0.0, np.mean(self.weights)],
                yaxis_range=[0.0, np.mean(self.weights)/2],
            ),
        )

        fig.update_xaxes(
            title_text="M: mean",
            title_font_size=22,
            tickfont_size=15,
            tickmode="auto",
            showline=True,
            linewidth=1.25,
            linecolor="black",
            minor=dict(ticklen=6, ticks="inside", tickcolor="black", showgrid=True),
        )
        fig.update_yaxes(
            title_text="SD: std",
            title_font_size=22,
            tickfont_size=15,
            showline=True,
            linewidth=1.25,
            linecolor="black",
            minor=dict(ticklen=6, ticks="inside", tickcolor="black", showgrid=True),
        )

        # calculate upper perimeter
        if len(set(self.weights)) == 1:

            def max_std(m, n):
                floor_mn = np.floor(m * n)
                nm = n * m
                value_under_sqrt = n * (floor_mn + (floor_mn - nm) ** 2) - nm**2
                return np.sqrt(value_under_sqrt) / n

            means = np.linspace(0, 1, 10000)
            half_perimeter = max_std(means[:len(means)//2], self.m)
            perimeter = np.concatenate((half_perimeter, np.flip(half_perimeter)))   
        else:
            quality_exact = {
                2: 125,
                3: 100,
                4: 75,
            }
            means = np.linspace(0, np.mean(self.weights), quality_exact.get(self.m, 50))
            half_perimeter = Parallel(n_jobs=self.n_jobs)(delayed(self.max_std_calculator)(mean, self.weights) for mean in means[:len(means)//2])
            perimeter = np.concatenate((half_perimeter, np.flip(half_perimeter)))

        # draw upper perimeter
        fig.add_trace(
            go.Scatter(
                x=means,
                y=perimeter,
                mode="lines",
                showlegend=False,
                hoverinfo="none",
                line_color="black",
            )
        )

        # fill between the line and the std = 0.5
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0.5, 0.5],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(255, 255, 255, 1)",
                showlegend=False,
                hoverinfo="none",
                line_color="white",
            )
        )

        # fill from the end of the graph to mean = 1
        fig.add_trace(
            go.Scatter(
                x=[max(means), max(means), 1],
                y=[0, 0.5, 0.5],
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(255, 255, 255, 1)",
                showlegend=False,
                hoverinfo="none",
                line_color="white",
            )
        )

        self.plot_background = go.Figure(fig)
        values = self.X_new[str(self.agg_fn.letter)].to_list()

        ### plot the ranked data
        custom = []
        for i in self.X_new.index.values:
            custom.append(1 + self._ranked_alternatives.index(i))

        fig.add_trace(
            go.Scatter(
                x=self.X_new["Mean"].tolist(),
                y=self.X_new["Std"].tolist(),
                showlegend=False,
                mode="markers",
                marker=dict(color="black", size=10),
                customdata=np.stack((custom, values), axis=1),
                text=self.X_new.index.values,
                hovertemplate="<b>ID</b>: %{text}<br>"
                + "<b>Rank</b>: %{customdata[0]:f}<br>"
                + f"<b>{str(self.agg_fn.letter + '(v)')}</b>: " "%{customdata[1]:f}<br>"
                + "<extra></extra>",
            )
        )
        ### add names
        if show_names == True:
            for i, label in enumerate(self.X_new.index.values):
                fig.add_annotation(
                    x=self.X_new["Mean"].tolist()[i] + 0.01,
                    y=self.X_new["Std"].tolist()[i] + 0.01,
                    text=label,
                    showarrow=False,
                    font=dict(size=12),
                )
        return fig

    def __update_for_plot(self, id, changes, change_number):

        if "Mean" in changes.columns and "Std" in changes.columns:
            self.X_newPoint = self.X_new.copy()
            self.X_newPoint.loc["NEW " + id] = self.X_newPoint.loc[id]
            self.X_newPoint.loc["NEW " + id, "Mean"] += changes["Mean"].values[0]
            self.X_newPoint.loc["NEW " + id, "Std"] += changes["Std"].values[0]
            agg_value = self.agg_fn.TOPSIS_calculation(
                np.mean(self.weights),
                self.X_newPoint.loc["NEW " + id, "Mean"],
                self.X_newPoint.loc["NEW " + id, "Std"],
            )
            self.X_newPoint.loc["NEW " + id, str(self.agg_fn.letter)] = agg_value
        elif "Mean" in changes.columns:
            self.X_newPoint = self.X_new.copy()
            self.X_newPoint.loc["NEW " + id] = self.X_newPoint.loc[id]
            self.X_newPoint.loc["NEW " + id, "Mean"] += changes["Mean"].values[0]
            agg_value = self.agg_fn.TOPSIS_calculation(
                np.mean(self.weights),
                self.X_newPoint.loc["NEW " + id, "Mean"],
                self.X_newPoint.loc["NEW " + id, "Std"],
            )
            self.X_newPoint.loc["NEW " + id, str(self.agg_fn.letter)] = agg_value
        elif "Std" in changes.columns:
            self.X_newPoint = self.X_new.copy()
            self.X_newPoint.loc["NEW " + id] = self.X_newPoint.loc[id]
            self.X_newPoint.loc["NEW " + id, "Std"] += changes["Std"].values[0]
            agg_value = self.agg_fn.TOPSIS_calculation(
                np.mean(self.weights),
                self.X_newPoint.loc["NEW " + id, "Mean"],
                self.X_newPoint.loc["NEW " + id, "Std"],
            )
            self.X_newPoint.loc["NEW " + id, str(self.agg_fn.letter)] = agg_value
        else:
            row_to_add = changes.iloc[change_number]
            result = self.X.loc[id].add(row_to_add, fill_value=0)
            new_row = pd.Series(result, name="NEW " + id)
            self.X_newPoint = self.X.copy()
            self.X_newPoint = pd.concat(
                [self.X_newPoint, new_row.to_frame().T]
            )
            self.X_newPoint = self.__normalize_data(self.X_newPoint)
            w_means, w_stds = self.transform_US_to_wmsd(np.array(self.X_newPoint))
            agg_values = self.agg_fn.score_batch(
                np.array(self.X_newPoint.loc[:, self.X.columns], dtype=float)
            )
            self.X_newPoint["Mean"] = w_means
            self.X_newPoint["Std"] = w_stds
            self.X_newPoint[str(self.agg_fn.letter)] = agg_values
        return self.X_newPoint

    def plot_improvement(self, id, changes, show_names=False, change_number=0):
        """Plots positions of alternatives in WMSD space and visualize the change after applying improvement action.

        Parameters
        ----------
        id : string
            String value containing the name of the alternative to improve.
        changes : pandas Data-frame
            Data-frame with the improvement action changes
        show_names : bool, optional
            Boolean value, if true, then points labels are showed on a plot.
                (default: False)
        change_number : int, optional
            Integer value indicating which change should be applied to plot, if there are more than 1.
            (default: 0)

        Returns
        -------
        Plot as a plotly figure.
        """
        self.__ensure_wmsd_aggregation("plot_improvement")
        self.__update_for_plot(id, changes, change_number)
        old_rank = (
            self.X_new.sort_values(by=str(self.agg_fn.letter), ascending=False).index.get_loc(id) + 1
        )
        old_value = self.X_new.loc[id, str(self.agg_fn.letter)]
        fig = go.Figure(self.plot_background)

        ### add old point
        fig.add_trace(
            go.Scatter(
                x=[self.X_new.loc[id, "Mean"]],
                y=[self.X_new.loc[id, "Std"]],
                showlegend=False,
                mode="markers",
                marker=dict(color="black", size=10),
                customdata=np.stack(([old_rank], [old_value]), axis=1),
                text=["OLD " + id],
                hovertemplate="<b>ID</b>: %{text}<br>"
                + "<b>Old Rank</b>: %{customdata[0]:f}<br>"
                + "<b>New Rank</b>: -<br>"
                + f"<b>{str(self.agg_fn.letter + '(v)')}</b>: " "%{customdata[1]:f}<br>"
                + "<extra></extra>",
            )
        )
        ### add name for old point
        if show_names == True:
            fig.add_annotation(
                x=self.X_new.loc[id, "Mean"] + 0.01,
                y=self.X_new.loc[id, "Std"] + 0.01,
                text="OLD " + id,
                showarrow=False,
                font=dict(size=12),
            )

        self.X_newPoint = self.X_newPoint.drop(index=(id))
        self.X_newPoint = self.X_newPoint.sort_values(by=str(self.agg_fn.letter), ascending=False)
        new_rank = self.X_newPoint.index.get_loc("NEW " + id) + 1
        new_value = self.X_newPoint.loc["NEW " + id, str(self.agg_fn.letter)]

        numbers = [i for i in range(1, self.n + 1)]
        custom0 = numbers.copy()
        custom0.remove(old_rank)
        custom1 = numbers.copy()
        custom1.remove(new_rank)

        ### define new point
        new_point = go.Scatter(
            x=[self.X_newPoint.loc["NEW " + id, "Mean"]],
            y=[self.X_newPoint.loc["NEW " + id, "Std"]],
            showlegend=False,
            mode="markers",
            marker=dict(color="white", size=10, line=dict(color='black', width=2)),
            customdata=np.stack(([new_rank], [new_value]), axis=1),
            text=["NEW " + id],
            hovertemplate="<b>ID</b>: %{text}<br>"
            + "<b>Old Rank</b>: -<br>"
            + "<b>New Rank</b>: %{customdata[0]:f}<br>"
            + f"<b>{str(self.agg_fn.letter + '(v)')}</b>: " "%{customdata[1]:f}<br>"
            + "<extra></extra>",
        )
        ### add names for new point and other points
        if show_names == True:
            for i, label in enumerate(self.X_newPoint.index.values):
                fig.add_annotation(
                    x=self.X_newPoint["Mean"].tolist()[i] + 0.01,
                    y=self.X_newPoint["Std"].tolist()[i] + 0.01,
                    text=label,
                    showarrow=False,
                    font=dict(size=12),
                )

        ### add arrow between old point and new point
        fig.add_annotation(
            x=self.X_newPoint.loc["NEW " + id, "Mean"],
            y=self.X_newPoint.loc["NEW " + id, "Std"],
            ax=self.X_new.loc[id, "Mean"],
            ay=self.X_new.loc[id, "Std"],
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            text='',
            showarrow=True,
            arrowhead=2,
            arrowwidth=2,
            arrowcolor='black'
        )

        self.X_newPoint = self.X_newPoint.drop(index=("NEW " + id))
        values = self.X_newPoint[str(self.agg_fn.letter)].to_list()

        ### add other points
        fig.add_trace(
            go.Scatter(
                x=self.X_newPoint["Mean"].tolist(),
                y=self.X_newPoint["Std"].tolist(),
                showlegend=False,
                mode="markers",
                marker=dict(color="black", size=10),
                customdata=np.stack((custom0, custom1, values), axis=1),
                text=self.X_newPoint.index.values,
                hovertemplate="<b>ID</b>: %{text}<br>"
                + "<b>Old Rank</b>: %{customdata[0]:f}<br>"
                + "<b>New Rank</b>: %{customdata[1]:f}<br>"
                + f"<b>{str(self.agg_fn.letter + '(v)')}</b>: " "%{customdata[2]:f}<br>"
                + "<extra></extra>",
            )
        )
        ### add new point
        fig.add_trace(new_point)
        return fig

    def return_ranking(self, normalized=True):
        """Returns the TOPSIS ranking

        Parameters
        ----------
        normalized : boolean, default=True
            If True, then all criteria values will be returned in their normalized form.
            If False, then all criteria values will be returned as they were passed by the user.

        Returns
        -------
        ranking : Pandas data-frame.
        """

        if not isinstance(normalized, bool):
            raise ValueError(
                "Invalid value at 'normalized': must be a bool."
            )

        if normalized:
            ranking = self.X_new.copy()
        else:
            ranking = self.X.copy()
            ranking["Mean"] = self.X_new["Mean"]
            ranking["Std"] = self.X_new["Std"]
            ranking[str(self.agg_fn.letter)] = self.X_new[str(self.agg_fn.letter)]

        ranking = ranking.assign(Rank=None)
        columns = ranking.columns.tolist()
        columns = columns[-1:] + columns[:-1]
        ranking = ranking[columns]

        alternative_names = ranking.index.tolist()
        for alternative in alternative_names:
            ranking.loc[alternative, "Rank"] = (
                self._ranked_alternatives.index(alternative) + 1
            )

        ranking = ranking.sort_values(by=["Rank"])

        return ranking

    def show_ranking(self, mode="standard", first=1, last=None):
        """Displays the TOPSIS ranking

        Parameters
        ----------
        mode : 'minimal'/'standard'/'full', default="standard"
            Specifies the level of detail in the ranking display.
            mode='minimal' shows only the positions of ranked alternatives.
            mode='standard' shows the positions of ranked alternatives and criteria values
            mode='full', shows the positions of ranked alternatives, criteria values,
            and also values of mean, standard deviation and aggregation function
        first : int, default=1
            Rank from which the ranking should be displayed.
        last : int, default=None
            Rank to which the ranking should be displayed.
        """

        if last is None:
            last = len(self.X_new.index)

        self.__check_show_ranking(first, last)

        ranking = self.X_new.copy()
        ranking = ranking.assign(Rank=None)
        columns = ranking.columns.tolist()
        columns = columns[-1:] + columns[:-1]
        ranking = ranking[columns]

        alternative_names = ranking.index.tolist()
        for alternative in alternative_names:
            ranking.loc[alternative, "Rank"] = (
                self._ranked_alternatives.index(alternative) + 1
            )

        ranking = ranking.sort_values(by=["Rank"])
        # ranking = ranking.loc[max(first-1, 0):last]
        ranking = ranking[(first - 1) : last]

        if isinstance(mode, str):
            if mode == "minimal":
                display(ranking["Rank"])
            elif mode == "standard":
                display(ranking.drop(["Mean", "Std", str(self.agg_fn.letter)], axis=1))
            elif mode == "full":
                display(ranking)
            else:
                raise ValueError(
                    "Invalid value at 'mode': must be a string (minimal, standard, or full)."
                )
            return

        display(ranking.drop(["Mean", "Std", str(self.agg_fn.letter)], axis=1))
        return

    def __get_alternative_ID(self, alternative_id_or_rank):
        if type(alternative_id_or_rank) == int:
            # return alternative ID according to the position in the ranking specified by user
            return self._ranked_alternatives[alternative_id_or_rank-1]
        elif type(alternative_id_or_rank) == str:
            # return alternative ID specified by user
            return alternative_id_or_rank
        else:
            raise TypeError(f"Invalid value at 'alternative_to_improve': must be int or str, not {type(alternative_id_or_rank)}")

    def improvement(self, function_name, alternative_to_improve, alternative_to_overcome, epsilon=1e-06, **kwargs):
        """ Runs chosen by the user improvement function.

        Parameters
        ----------
        function_name : str
            Name of the function (improvement action) to perform on given alternative.
            It must be one of the following strings:
            'improvement_single_feature',
            'improvement_mean',
            'improvement_features',
            'improvement_genetic',
            'improvement_std'

        alternative_to_improve : int or str
            Name or position of the alternative which user wants to improve.
        alternative_to_overcome : int or str
            Name or position of the alternative which should be overcome by chosen alternative.
        epsilon : float
            Precision of calculations. Must be in range (0.0, 1.0>.
            (default : 0.000001)

        Returns
        -------
        Output returned by the [function_name] function.
        """

        alternative_to_improve = self.X_new.loc[self.__get_alternative_ID(alternative_to_improve)].copy()
        alternative_to_overcome = self.X_new.loc[self.__get_alternative_ID(alternative_to_overcome)].copy()

        func = getattr(self.agg_fn, function_name)
        return func(alternative_to_improve, alternative_to_overcome, epsilon, **kwargs)

    def __check_max_std_calculator(self, max_std_calculator):
        if isinstance(max_std_calculator, str):
            if max_std_calculator == "scip":
                from utils.max_std_calculator_scip import max_std_scip

                return max_std_scip
            elif max_std_calculator == "gurobi":
                from utils.max_std_calculator_gurobi import max_std_gurobi

                return max_std_gurobi
            else:
                raise ValueError(
                    "Invalid value at 'agg_fn': must be string (gurobi or scip) or function."
                )
        elif callable(max_std_calculator):
            return max_std_calculator
        else:
            raise ValueError(
                "Invalid value at 'agg_fn': must be string (scip or gurobi) or function."
            )

    def __check_agg_fn(self, agg_fn):
        if isinstance(agg_fn, str):
            if agg_fn == "A":
                return ATOPSIS(self)
            elif agg_fn == "I":
                return ITOPSIS(self)
            elif agg_fn == "R":
                return RTOPSIS(self)
            elif agg_fn == "U":
                return SAW(self)
            elif agg_fn == "C":
                return COPRAS(self)
            elif agg_fn == "K":
                return ARAS(self)
            elif agg_fn == "W":
                return WASPAS(self)
            elif agg_fn == "V":
                return VIKOR(self)
            else:
                raise ValueError(
                    "Invalid value at 'agg_fn': must be string (A, I, R, U, C, K, W, or V) or class implementing AggregationFunction."
                )
        elif isinstance(agg_fn, AggregationFunction):
            return agg_fn.fit(self)
        elif isinstance(agg_fn, type) and issubclass(agg_fn, AggregationFunction):
            return agg_fn(self)
        else:
            raise ValueError(
                "Invalid value at 'agg_fn': must be string (A, I, R, U, C, K, W, or V) or class implementing AggregationFunction."
            )

    def __check_weights(self, weights):
        if isinstance(weights, list):
            return weights

        elif isinstance(weights, dict):
            return self.__dict_to_list(weights)

        elif weights is None:
            return np.ones(self.m)

        else:
            raise ValueError(
                "Invalid value at 'weights': must be a list or a dictionary"
            )

    def __check_objectives(self, objectives):
        if isinstance(objectives, list):
            return objectives
        elif isinstance(objectives, str):
            return np.repeat(objectives, self.m)
        elif isinstance(objectives, dict):
            return self.__dict_to_list(objectives)
        elif objectives is None:
            return np.repeat("max", self.m)
        else:
            raise ValueError(
                "Invalid value at 'objectives': must be a list or a string (gain, g, cost, c, min or max) or a dictionary"
            )

    def __check_expert_range(self, expert_range):
        if isinstance(expert_range, dict):
            expert_range = self.__dict_to_list(expert_range)

        if isinstance(expert_range, list):
            if all(isinstance(e, list) for e in expert_range):
                return expert_range

            elif all(isinstance(e, (int, float, np.float64)) for e in expert_range):
                expert_range = [expert_range]
                numpy_expert_range = np.repeat(expert_range, self.m, axis=0)
                return numpy_expert_range.tolist()

            else:
                raise ValueError(
                    "Invalid value at 'expert_range': must be a homogenous list (1D or 2D) or a dictionary"
                )

        elif expert_range is None:
            lower_bounds = self.X.min()
            upper_bounds = self.X.max()
            expert_range = [lower_bounds, upper_bounds]
            numpy_expert_range = np.array(expert_range).T
            return numpy_expert_range.tolist()

        else:
            raise ValueError(
                "Invalid value at 'expert_range': must be a homogenous list (1D or 2D) or a dictionary"
            )

    def __check_input(self):
        if self.X.isnull().values.any():
            raise ValueError(
                "Dataframe must not contain any none/nan values, but found at least one"
            )

        if len(self.weights) != self.m:
            raise ValueError("Invalid value 'weights'.")

        if not all(type(item) in [int, float, np.float64] for item in self.weights):
            raise ValueError(
                "Invalid value 'weights'. Expected numerical value (int or float)."
            )

        if not all(item >= 0 for item in self.weights):
            raise ValueError(
                "Invalid value 'weights'. Expected value must be non-negative."
            )

        if not any(item > 0 for item in self.weights):
            raise ValueError(
                "Invalid value 'weights'. At least one weight must be positive."
            )

        if len(self.objectives) != self.m:
            raise ValueError("Invalid value 'objectives'.")

        if not all(item in ["min", "max"] for item in self.objectives):
            raise ValueError(
                "Invalid value at 'objectives'. Use 'min', 'max', 'gain', 'cost', 'g' or 'c'."
            )

        if len(self.expert_range) != len(self.objectives):
            raise ValueError(
                "Invalid value at 'expert_range'. Length of should be equal to number of criteria."
            )

        for col in self.expert_range:
            if len(col) != 2:
                raise ValueError(
                    "Invalid value at 'expert_range'. Every criterion has to have minimal and maximal value."
                )
            if not all(type(item) in [int, float] for item in col):
                raise ValueError(
                    "Invalid value at 'expert_range'. Expected numerical value (int or float)."
                )
            if col[0] > col[1]:
                raise ValueError(
                    "Invalid value at 'expert_range'. Minimal value  is bigger then maximal value."
                )

        lower_bound = np.array(self.X.min()).tolist()
        upper_bound = np.array(self.X.max()).tolist()

        for val, mini, maxi in zip(self.expert_range, lower_bound, upper_bound):
            if not (val[0] <= mini and val[1] >= maxi):
                raise ValueError(
                    "Invalid value at 'expert_range'. All values from original data must be in a range of expert_range."
                )

    def __check_input_after_transform(self, X):
        n = X.shape[0]
        m = X.shape[1]

        if X.isnull().values.any():
            raise ValueError(
                "Dataframe must not contain any none/nan values, but found at least one"
            )

        if self.m != m:
            raise ValueError(
                "Invalid number of columns. Number of criteria must be the same as in previous dataframe."
            )

        if not all(X.columns.values == self.X.columns.values):
            raise ValueError(
                "New dataset must have the same columns as the dataset used to fit WMSDTransformer"
            )

        lower_bound = np.array(X.min()).tolist()
        upper_bound = np.array(X.max()).tolist()

        for val, mini, maxi in zip(self.expert_range, lower_bound, upper_bound):
            if not (val[0] <= mini and val[1] >= maxi):
                raise ValueError(
                    "Invalid value at 'expert_range'. All values from original data must be in a range of expert_range."
                )

    def __check_show_ranking(self, first, last):
        if isinstance(first, int):
            if first < 1 or first > len(self.X_new.index):
                raise ValueError(
                    f"Invalid value at 'first': must be in range [1:{len(self.X_new.index)}]"
                )
        else:
            raise TypeError("Invalid type of 'first': must be an int")

        if isinstance(last, int):
            if last < 1 or last > len(self.X_new.index):
                raise ValueError(
                    f"Invalid value at 'last': must be in range [1:{len(self.X_new.index)}]"
                )
        else:
            raise TypeError("Invalid type of 'last': must be an int")

        if last < first:
            raise ValueError("'first' must be not greater than 'last'")

    def __normalize_data(self, data):
        for col_idx, colname in enumerate(data.columns):
            data[colname] = (data[colname] - self._lower_bounds[col_idx]) / (self._value_range[col_idx])

        for i in range(self.m):
            if self.objectives[i] == "min":
                data[data.columns[i]] = 1 - data[data.columns[i]]

        return data

    def __normalize_weights(self, weights):
        weights = np.array([float(i) / max(weights) for i in weights])
        return weights

    def __wmstd(self):
        w = self.weights
        s = np.sqrt(sum(w * w)) / np.mean(w)
        wm = []
        wsd = []
        for index, row in self.X_new.iterrows():
            v = row * w
            vw = (sum(v * w) / sum(w * w)) * w
            wm.append(np.sqrt(sum(vw * vw)) / s)
            wsd.append(np.sqrt(sum((v - vw) * (v - vw))) / s)

        self.X_new["Mean"] = wm
        self.X_new["Std"] = wsd

    def __ranking(self):
        data__ = self.X_new.copy()
        data__ = data__.sort_values(by=str(self.agg_fn.letter), ascending=False)
        arranged = data__.index.tolist()
        return arranged

    def __dict_to_list(self, dictionary):
        new_list = []

        for col_name in self.X.columns:
            new_list.append(dictionary[col_name])

        return new_list



