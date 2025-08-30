from formulaic import Formula
import pandas as pd
import statsmodels.api as sm
from typing import Literal, Optional, Any, Dict
import warnings

# Import OaxacaResults from the new results module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .results import OaxacaResults

from .formulaic_utils import (
    dummies,
    term_dummies,
    term_dummies_gu_adjusted,
    get_base_category,
)


class Oaxaca:
    """Oaxaca-Blinder decomposition for analyzing group differences.

    The Oaxaca-Blinder decomposition is a statistical method used to explain
    the difference in outcomes between two groups by decomposing it into
    explained and unexplained components.

    Attributes
    ----------
    coef_ : dict
        Dictionary mapping group values to their coefficients (pd.Series).
    models_ : dict
        Dictionary mapping group values to their fitted OLS models.
    group_stats_ : dict
        Dictionary mapping group values to their statistics including n_obs,
        mean_y, mean_X, std_y, and r_squared.
    group_variable_ : str
        The name of the column in X that contains the group indicator.
    groups_ : list
        The unique groups identified in the data.
    """

    def __init__(self):
        """Initialize the Oaxaca-Blinder decomposition model."""
        pass

    def fit(self, formula: str, data: pd.DataFrame, group_variable: str) -> "Oaxaca":
        """
        Fit the Oaxaca-Blinder decomposition model.

        Parameters
        ----------
        formula : str
            The formula for the regression model
        data : pd.DataFrame
            The data containing all variables
        group_variable : str
            The name of the column in data that contains the group indicator

        Returns
        -------
        self : Oaxaca
            The fitted Oaxaca object for method chaining
        """

        # Store user input
        self.formula = formula
        self.group_variable = group_variable

        # Get unique groups
        self.groups_ = sorted(data[group_variable].unique().tolist())
        if len(self.groups_) != 2:
            raise ValueError("Group variable must have exactly 2 unique values")

        # Get rid of missing data
        data = data.dropna(subset=Formula(self.formula).required_variables)
        # Ensure common support between two groups
        data = self._harmonize_common_support(data)

        # Initialize group-specific attributes
        self.coef_ = {}
        self.models_ = {}
        self.group_stats_ = {}

        # Fit separate models for each group
        for group in self.groups_:
            group_mask = data[group_variable] == group
            # ensure_full_rank=True since we want the full-rank model for OLS
            y_group, X_group = Formula(formula).get_model_matrix(data[group_mask], output="pandas", ensure_full_rank=True)
            self.X_model_spec = X_group.model_spec

            # Check for zero variance columns, which statsmodels.OLS surprisingly just let through silently
            # errors="ignore" because some models may not have an Intercept
            variances = X_group.drop("Intercept", axis=1, errors="ignore").var()
            # Check if any column has zero variance
            if (variances == 0).any():
                # Identify the problematic columns
                zero_variance_cols = variances[variances == 0].index.tolist()
                X_group = X_group.drop(zero_variance_cols, axis=1)
                warnings.warn(f"Warning: The following columns have zero variance and were removed: {zero_variance_cols}")

            model = sm.OLS(y_group, X_group).fit()

            # Store coefficients and stats before removing data since remove_data() corrupts the params index
            self.coef_[group] = model.params.copy()
            self.group_stats_[group] = {
                "n_obs": len(y_group),
                "mean_y": float(y_group.mean().iloc[0]),
                "mean_X": X_group.mean(),
                "std_y": float(y_group.std().iloc[0]),
                "r_squared": model.rsquared,
            }

            # Remove training data from model object to reduce memory usage
            model.remove_data()

            self.models_[group] = model
        # Store the model specification for later tying back the dummies to the categorical terms
        # in the output table
        # At this point, the two groups have the same categories, so it doesn't matter which one we take
        del y_group, X_group  # Release memory

        # Return self to allow method chaining
        return self

    def two_fold(
        self,
        weights: Optional[Dict[Any, float]] = None,
        gu_adjustment: Literal["none", "unweighted", "weighted"] = "none",
        direction: Literal["group0 - group1", "group1 - group0"] = "group0 - group1",
    ) -> "OaxacaResults":
        """
        Perform two-fold decomposition with customizable weights.

        Parameters
        ----------
        weights : dict of {group_value: float}, optional
            Weights for the non-discriminatory coefficient vector, where keys are
            the group values and values are the corresponding weights.

        gu_adjustment : bool, default False
            If True, apply Gardeazabal and Ugidos (2004) adjustment for omitted group problem.

        direction : str, default "group0 - group1"
            Direction of the decomposition. Options are:
            - "group0 - group1": Decompose group0 - group1 (default)
            - "group1 - group0": Decompose group1 - group0
            Where group0 is the first group alphabetically and group1 is the second.

        Returns
        -------
        OaxacaResults
            A new OaxacaResults object with decomposition results
        """
        # Validate inputs
        if weights is None:
            raise ValueError("Weights must be provided")
        if not isinstance(weights, dict):
            raise ValueError("Weights must be a dictionary with group values as keys")
        if set(weights.keys()) != set(self.groups_):
            raise ValueError(f"Weights keys must match group values: {self.groups_}")
        if abs(sum(weights.values()) - 1.0) > 1e-10:
            raise ValueError("Weights must sum to 1.0")

        if gu_adjustment not in ["none", "unweighted", "weighted"]:
            raise ValueError("gu_adjustment must be one of: 'none', 'unweighted', 'weighted'")
        if direction not in ["group0 - group1", "group1 - group0"]:
            raise ValueError("Direction must be either 'group0 - group1' or 'group1 - group0'")

        # Get group references
        group_0, group_1 = self.groups_

        # Get coefficients and mean X values
        coef_0 = self.coef_[group_0]
        coef_1 = self.coef_[group_1]
        mean_X_0 = self.group_stats_[group_0]["mean_X"]
        mean_X_1 = self.group_stats_[group_1]["mean_X"]

        # Apply GU adjustment if needed (after ensuring conformable dimensions)
        if gu_adjustment != "none":
            mean_X_0 = self.group_stats_all_categories_[group_0]["mean_X"]
            mean_X_1 = self.group_stats_all_categories_[group_1]["mean_X"]
            coef_0 = self._apply_gu_adjustment(coef_0, weight=mean_X_0 if gu_adjustment == "weighted" else None)
            coef_1 = self._apply_gu_adjustment(coef_1, weight=mean_X_1 if gu_adjustment == "weighted" else None)

        # Since we potentially manipulated the indices of coef and mean_X, let's check that their indices
        # are the same, only out of order. pandas won't do so for us
        assert set(mean_X_0.index) == set(mean_X_1.index) == set(coef_0.index) == set(coef_1.index), (
            f"mean_X_0 vs mean_X_1 {set(mean_X_0.index).symmetric_difference(set(mean_X_1.index))}",
            f"coef_0 vs coef_1 {set(coef_0.index).symmetric_difference(set(coef_1.index))}",
            f"mean_X_0 vs coef_0 {set(mean_X_0.index).symmetric_difference(set(coef_0.index))}",
        )

        # Get mean Y values
        mean_y_0 = self.group_stats_[group_0]["mean_y"]
        mean_y_1 = self.group_stats_[group_1]["mean_y"]

        # Calculate non-discriminatory coefficient vector using dictionary weights
        coef_nd = weights[group_0] * coef_0 + weights[group_1] * coef_1

        # Calculate decomposition components, default: group0 - group1
        total_diff = float(mean_y_0 - mean_y_1)
        explained = float((mean_X_0 - mean_X_1) @ coef_nd)
        explained_detailed = (mean_X_0 - mean_X_1) * coef_nd
        unexplained = float(mean_X_0 @ (coef_0 - coef_nd) + mean_X_1 @ (coef_nd - coef_1))
        unexplained_detailed = mean_X_0 * (coef_0 - coef_nd) + mean_X_1 * (coef_nd - coef_1)
        X_diff = mean_X_0 - mean_X_1
        if direction == "group1 - group0":
            total_diff, explained, unexplained = -total_diff, -explained, -unexplained
            explained_detailed, unexplained_detailed, X_diff = -explained_detailed, -unexplained_detailed, -X_diff
        # Get the appropriate categorical mapping based on whether GU adjustment was applied
        if gu_adjustment != "none":
            categorical_mapping = term_dummies_gu_adjusted(self.X_model_spec)
        else:
            categorical_mapping = term_dummies(self.X_model_spec)

        # Import here to avoid circular imports
        from .results import OaxacaResults

        # Create and return new OaxacaResults object
        return OaxacaResults(
            oaxaca_instance=self,
            total_difference=total_diff,
            explained=explained,
            unexplained=unexplained,
            explained_detailed=explained_detailed,
            unexplained_detailed=unexplained_detailed,
            X_diff=X_diff,
            coef_nondiscriminatory=coef_nd,
            weights=weights,
            mean_X_0=mean_X_0,
            mean_X_1=mean_X_1,
            categorical_mapping=categorical_mapping,
            direction=direction,
        )

    def _harmonize_common_support(self, data: pd.DataFrame):
        """
        Solve the common support problem by removing rows
        so that the two groups have the same set of dummies/categories
        """
        y = {}
        X = {}
        X_model_spec = {}
        for group in self.groups_:
            group_mask = data[self.group_variable] == group
            # ensure_full_rank=False since we're doing data clean up here, not modeling
            # We don't want the base to interfere with the harmonization
            # For example, when a base is excluded from a group's model matrix, making it appear to not be exclusive to that group
            y[group], X[group] = Formula(self.formula).get_model_matrix(data.loc[group_mask, :], output="pandas", ensure_full_rank=False)
            X_model_spec[group] = X[group].model_spec
            # Sometimes the user-supplied formula can result in all-0 dummies, such as when they
            #   specify a categorical level that doesn't exist in the data
            columns_that_are_all_0 = X[group].columns[(X[group] == 0).all(axis=0)]
            X[group] = X[group].drop(columns_that_are_all_0, axis=1)

        # Figure out which rows need to be removed to ensure common support
        self.dummy_removal_result_ = {}
        self.group_stats_all_categories_ = {}
        for this, other in zip(self.groups_, self.groups_[::-1]):
            # Remove dummies that are just all 0

            # Convert to list since pandas can't accept set as index
            dummies_exclusive_to_this_group = list(set(dummies(X_model_spec[this])) - set(dummies(X_model_spec[other])))
            rows_to_remove = (X[this].loc[:, dummies_exclusive_to_this_group] == 1).any(axis=1)

            # Compute scalar outcomes and share as floats for easier downstream use
            outcome_pre_removal_val = float(y[this].mean().iloc[0])
            outcome_post_removal_val = float(y[this][~rows_to_remove].mean().iloc[0])
            # May be NaN if no rows removed; float() preserves NaN
            outcome_among_removed_val = float(y[this][rows_to_remove].mean().iloc[0]) if len(y[this][rows_to_remove]) > 0 else float("nan")
            share_removed_val = float(rows_to_remove.mean())
            mean_adjustment_val = outcome_pre_removal_val - outcome_post_removal_val

            self.dummy_removal_result_[this] = {
                "removed_dummies": dummies_exclusive_to_this_group,
                "rows_to_remove": rows_to_remove,
                "outcome_pre_removal": outcome_pre_removal_val,
                "outcome_post_removal": outcome_post_removal_val,
                "outcome_among_removed": outcome_among_removed_val,
                "share_removed": share_removed_val,
                "mean_adjustment": mean_adjustment_val,
            }
            # In addition to the full-rank model matrix in OLS below,
            #   calculate the mean of all categories for GU adjustment
            # We do this opportunistically by using the cleaned data
            cleaned_X = X[this].loc[~rows_to_remove, :].drop(dummies_exclusive_to_this_group, axis=1)
            self.group_stats_all_categories_[this] = {
                "mean_X": cleaned_X.mean(),
            }

        harmonized_data_list = []
        for group in self.groups_:
            group_mask = data[self.group_variable] == group
            data_group = data[group_mask]
            harmonized_data_list.append(data_group.loc[~self.dummy_removal_result_[group]["rows_to_remove"], :])
        return pd.concat(harmonized_data_list, axis=0, ignore_index=True)

    def _apply_gu_adjustment(self, coef: pd.Series, weight: Optional[pd.Series] = None) -> pd.Series:
        """
        Apply Gardeazabal and Ugidos (2004) adjustment for omitted group problem.

        For each categorical variable:
        1. Insert coefficient of 0 for omitted base category
        2. Calculate mean of all dummy coefficients for that categorical variable
        3. Subtract this mean from each dummy coefficient
        4. Add this mean to the intercept coefficient

        Parameters
        ----------
        coef : pd.Series
            Original coefficients from OLS regression
        weight : pd.Series, optional
            If not set, perform the "classic" GU adjustment.
            If set, a useful set of weights is the relative frequency of the categories,
              which result in the adjusted Intercept equalling the overall mean outcome,
              and consequently the coef as deviation from the overall mean


        Returns
        -------
        pd.Series
            Adjusted coefficients
        """

        new_coef = pd.Series(dtype=float)
        for term, term_slice in self.X_model_spec.term_slices.items():
            if term not in term_dummies(self.X_model_spec):
                # Not a categorical term, so just append the original coef
                new_coef = pd.concat([new_coef, coef[term_slice]])
            else:
                # It's a categorical term, so let's apply GU adjustment
                if len(term.factors) > 1:
                    raise ValueError("We only support single categorical variable, not interaction")
                factor = term.factors[0]
                contrast_state = self.X_model_spec.factor_contrasts[factor]
                base_category = get_base_category(contrast_state)
                base_category_column_name = contrast_state.contrasts.get_factor_format(levels=contrast_state.levels).format(
                    name=repr(factor), field=base_category
                )

                # Create extended coefficient series including base category (coefficient = 0)
                extended_coefs = pd.concat([coef[term_slice], pd.Series({base_category_column_name: 0.0})])
                # The non-full-rank X model-matrix will be named slightly different, e.g.
                # edu[high_school] instead of edu[T.high_school]
                # so we reformat the coefficient here to match
                extended_coefs.index = extended_coefs.index.str.replace("[T.", "[", regex=False)

                # Calculate mean of all coefficients (including base = 0)
                if weight is None:
                    mean_coef = extended_coefs.mean()
                else:
                    # The multiplication of weight and coef relies on pandas index alignment
                    #    if there are mismatched indices, fill with NaN then drop them
                    mean_coef = weight.mul(extended_coefs, fill_value=None).dropna().sum()

                # Adjust the coefficients, including the intercept
                extended_coefs -= mean_coef
                new_coef = pd.concat([new_coef, extended_coefs])
                new_coef["Intercept"] += mean_coef

        return new_coef

    def _ensure_conformable_dimensions(self, *series: pd.Series) -> tuple[pd.Series, ...]:
        """
        Ensure multiple pandas Series have conformable dimensions

        This solves the "common support" problem where categorical variables may have different
        sets of categories across the two groups.
        Missing categories are inserted with value 0.

        See a discussion of the common support issue in Lemieux p16 and
        Nopo, Hugo (2008) ìMatching as a Tool to Decompose Wage Gaps," Review of Economics and Statistics 90: 290-299.
        """
        target_index = self.X_model_spec.column_names
        aligned_series = tuple(s.reindex(target_index, fill_value=0.0) for s in series)

        return aligned_series
