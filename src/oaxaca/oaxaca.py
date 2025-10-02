import warnings

# Import OaxacaResults from the new results module
from typing import TYPE_CHECKING, Any, Literal, Optional

import pandas as pd
import statsmodels.api as sm
from formulaic import Formula

if TYPE_CHECKING:
    from .results import ThreeFoldResults, TwoFoldResults

from .formulaic_utils import (
    dummies,
    get_base_category,
    term_dummies,
    term_dummies_gu_adjusted,
)


class Oaxaca:
    """Oaxaca-Blinder decomposition for analyzing group differences.

    The Oaxaca-Blinder decomposition is a statistical method used to explain
    the difference in outcomes between two groups by decomposing it into
    explained and unexplained components.

    Attributes:
        coef_: Dictionary mapping group values to their coefficients (pd.Series).
        models_: Dictionary mapping group values to their fitted OLS models.
        group_stats_: Dictionary mapping group values to their statistics including n_obs,
            mean_y, mean_X, std_y, and r_squared.
        group_variable_: The name of the column in X that contains the group indicator.
        groups_: The unique groups identified in the data.
    """

    def __init__(self):
        """Initialize the Oaxaca-Blinder decomposition model."""
        pass

    def fit(self, formula: str, data: pd.DataFrame, group_variable: str) -> "Oaxaca":
        """Fit the Oaxaca-Blinder decomposition model.

        Args:
            formula: R-style formula for the regression model.
            data:
            group_variable: The column that contains the group indicator.

        Returns:
            The fitted Oaxaca object for method chaining.
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
        self.model_summary_stats_ = {}

        # Fit separate models for each group
        for group in self.groups_:
            group_mask = data[group_variable] == group
            # ensure_full_rank=True since we want the full-rank model for OLS
            y_group, X_group = Formula(formula).get_model_matrix(
                data[group_mask], output="pandas", ensure_full_rank=True
            )
            self.X_model_spec = X_group.model_spec

            # Check for zero variance columns, which statsmodels.OLS surprisingly just let through silently
            # errors="ignore" because some models may not have an Intercept
            variances = X_group.drop("Intercept", axis=1, errors="ignore").var()
            # Check if any column has zero variance
            if (variances == 0).any():
                # Identify the problematic columns
                zero_variance_cols = variances[variances == 0].index.tolist()
                X_group = X_group.drop(zero_variance_cols, axis=1)
                warnings.warn(
                    f"Warning: The following columns have zero variance and were removed: {zero_variance_cols}",
                    stacklevel=2,
                )

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

            # Store model summary statistics before removing data
            self.model_summary_stats_[group] = {
                "bse": model.bse.copy(),
                "tvalues": model.tvalues.copy(),
                "pvalues": model.pvalues.copy(),
            }

            # Remove training data from model object to reduce memory usage
            model.remove_data()

            self.models_[group] = model
        # Store the model specification for later tying back the dummies to the categorical terms
        # in the output table
        # At this point, the two groups have the same categories, so it doesn't matter which one we take
        del y_group, X_group  # Release memory

        # Check for zero total difference early to avoid division by zero issues
        group_0, group_1 = self.groups_
        mean_y_0 = self.group_stats_[group_0]["mean_y"]
        mean_y_1 = self.group_stats_[group_1]["mean_y"]
        total_difference = mean_y_0 - mean_y_1

        if abs(total_difference) < 1e-10:
            raise ValueError(
                f"Total difference between groups is effectively zero ({total_difference:.2e}). "
                f"Group {group_0} mean: {mean_y_0:.6f}, Group {group_1} mean: {mean_y_1:.6f}. "
                "Decomposition is not meaningful when group means are identical."
            )

        # Return self to allow method chaining
        return self

    def _validate_weights_input(self, weights):
        if weights is None:
            raise ValueError("Weights must be provided")
        if not isinstance(weights, dict):
            raise TypeError("Weights must be a dictionary with group values as keys")
        if set(weights.keys()) != set(self.groups_):
            raise ValueError(f"Weights keys must match group values: {self.groups_}")
        if abs(sum(weights.values()) - 1.0) > 1e-10:
            raise ValueError("Weights must sum to 1.0")

    def _compute_x_and_coef(self, gu_adjustment: Literal["none", "unweighted", "weighted"] = "none"):
        """Compute E(X) and β for both groups, which is all that is needed for both two-fold and three-fold decompositions.

        Args:
            gu_adjustment: Type of adjustment to apply. Options are:
                - "none": No adjustment (default)
                - "unweighted": Apply Gardeazabal and Ugidos (2004) adjustment. This is equivalent to running the
                    decomposition leaving out one category at a time, then take the average contributions
                - "weighted": Apply Gardeazabal and Ugidos (2004) adjustment with
                  weights based on category frequencies. This is equivalent to making the intercept the overall mean outcome,
                  leaving the coefficients as deviations from the overall mean.
        """
        if gu_adjustment not in ["none", "unweighted", "weighted"]:
            raise ValueError("gu_adjustment must be one of: 'none', 'unweighted', 'weighted'")

        group_0, group_1 = self.groups_
        coef_0 = self.coef_[group_0]
        coef_1 = self.coef_[group_1]
        mean_X_0 = self.group_stats_[group_0]["mean_X"]
        mean_X_1 = self.group_stats_[group_1]["mean_X"]

        if gu_adjustment != "none":
            mean_X_0 = self.group_stats_all_categories_[group_0]["mean_X"]
            mean_X_1 = self.group_stats_all_categories_[group_1]["mean_X"]
            coef_0 = self._apply_gu_adjustment(coef_0, weight=mean_X_0 if gu_adjustment == "weighted" else None)
            coef_1 = self._apply_gu_adjustment(coef_1, weight=mean_X_1 if gu_adjustment == "weighted" else None)

        # Since we potentially manipulated the indices of coef and mean_X, let's check that their indices
        # are the same, only out of order. pandas won't do so for us
        if not set(mean_X_0.index) == set(mean_X_1.index) == set(coef_0.index) == set(coef_1.index):
            raise ValueError("Incompatible indices detected")

        return mean_X_0, mean_X_1, coef_0, coef_1

    def two_fold(
        self,
        weights: dict[Any, float],
        gu_adjustment: Literal["none", "unweighted", "weighted"] = "none",
        direction: Literal["group0 - group1", "group1 - group0"] = "group0 - group1",
    ) -> "TwoFoldResults":
        """Perform two-fold decomposition with customizable weights.

        Args:
            weights: Weights for the non-discriminatory coefficient vector, where keys are
                the group values and values are the corresponding weights.
            gu_adjustment: Type of [Gardeazabal and Ugidos (2004)](index.md#the-omitted-base-category-problem) adjustment to apply.

                - "none": No adjustment
                - "unweighted": Apply unweighted GU adjustment
                - "weighted": Apply weighted GU adjustment


            direction: Direction of the decomposition.

                - "group0 - group1"
                - "group1 - group0"

        Returns:
            A new TwoFoldResults object with decomposition results.
        """
        self._validate_weights_input(weights)
        if direction not in ["group0 - group1", "group1 - group0"]:
            raise ValueError("Direction must be either 'group0 - group1' or 'group1 - group0'")

        group_0, group_1 = self.groups_
        mean_y_0 = self.group_stats_[group_0]["mean_y"]
        mean_y_1 = self.group_stats_[group_1]["mean_y"]

        mean_X_0, mean_X_1, coef_0, coef_1 = self._compute_x_and_coef(gu_adjustment=gu_adjustment)
        # Calculate non-discriminatory coefficient vector
        coef_nd = weights[group_0] * coef_0 + weights[group_1] * coef_1

        # Calculate decomposition components
        total_diff = float(mean_y_0 - mean_y_1)
        explained = float((mean_X_0 - mean_X_1) @ coef_nd)
        explained_detailed = (mean_X_0 - mean_X_1) * coef_nd
        unexplained = float(mean_X_0 @ (coef_0 - coef_nd) + mean_X_1 @ (coef_nd - coef_1))
        unexplained_detailed = mean_X_0 * (coef_0 - coef_nd) + mean_X_1 * (coef_nd - coef_1)
        if direction == "group1 - group0":
            total_diff, explained, unexplained = -total_diff, -explained, -unexplained
            explained_detailed, unexplained_detailed = -explained_detailed, -unexplained_detailed
        # Get the appropriate categorical mapping based on whether GU adjustment was applied
        if gu_adjustment != "none":
            categorical_to_dummy = term_dummies_gu_adjusted(self.X_model_spec)
        else:
            categorical_to_dummy = term_dummies(self.X_model_spec)

        # Import here to avoid circular imports
        from .results import TwoFoldResults

        return TwoFoldResults(
            oaxaca_instance=self,
            total_difference=total_diff,
            explained=explained,
            unexplained=unexplained,
            explained_detailed=explained_detailed,
            unexplained_detailed=unexplained_detailed,
            coef_nondiscriminatory=coef_nd,
            weights=weights,
            mean_X_0=mean_X_0,
            mean_X_1=mean_X_1,
            categorical_to_dummy=categorical_to_dummy,
            direction=direction,
        )

    def three_fold(
        self,
        gu_adjustment: Literal["none", "unweighted", "weighted"] = "none",
        direction: Literal["group0 - group1", "group1 - group0"] = "group0 - group1",
    ) -> "ThreeFoldResults":
        """Perform three-fold decomposition.

        Args:
            gu_adjustment: Type of [Gardeazabal and Ugidos (2004)](index.md#the-omitted-base-category-problem) adjustment to apply.

                - "none": No adjustment
                - "unweighted": Apply unweighted GU adjustment
                - "weighted": Apply weighted GU adjustment


            direction: Direction of the decomposition.

                - "group0 - group1"
                - "group1 - group0"
        Returns:
            A new ThreeFoldResults object with decomposition results.
        """
        if direction not in ["group0 - group1", "group1 - group0"]:
            raise ValueError("Direction must be either 'group0 - group1' or 'group1 - group0'")

        group_0, group_1 = self.groups_
        mean_y_0 = self.group_stats_[group_0]["mean_y"]
        mean_y_1 = self.group_stats_[group_1]["mean_y"]
        mean_X_0, mean_X_1, coef_0, coef_1 = self._compute_x_and_coef(gu_adjustment=gu_adjustment)

        # Calculate decomposition components
        total_diff = float(mean_y_0 - mean_y_1)

        # 1. Endowment effect: (X_0 - X_1) * β_1
        endowment = float((mean_X_0 - mean_X_1) @ coef_1)
        endowment_detailed = (mean_X_0 - mean_X_1) * coef_1
        # 2. Coefficient effect: X_1 * (β_0 - β_1)
        coefficient = float(mean_X_1 @ (coef_0 - coef_1))
        coefficient_detailed = mean_X_1 * (coef_0 - coef_1)
        # 3. Interaction effect: (X_0 - X_1) * (β_0 - β_1)
        interaction = float((mean_X_0 - mean_X_1) @ (coef_0 - coef_1))
        interaction_detailed = (mean_X_0 - mean_X_1) * (coef_0 - coef_1)

        X_diff = mean_X_0 - mean_X_1

        # Apply direction adjustment if needed
        if direction == "group1 - group0":
            total_diff = -total_diff
            X_diff = -X_diff
            endowment = -endowment
            coefficient = -coefficient
            interaction = -interaction
            endowment_detailed = -endowment_detailed
            coefficient_detailed = -coefficient_detailed
            interaction_detailed = -interaction_detailed

        # Get the appropriate categorical mapping based on whether GU adjustment was applied
        if gu_adjustment != "none":
            categorical_to_dummy = term_dummies_gu_adjusted(self.X_model_spec)
        else:
            categorical_to_dummy = term_dummies(self.X_model_spec)

        # Import here to avoid circular imports
        from .results import ThreeFoldResults

        return ThreeFoldResults(
            oaxaca_instance=self,
            total_difference=total_diff,
            endowment=endowment,
            coefficient=coefficient,
            interaction=interaction,
            endowment_detailed=endowment_detailed,
            coefficient_detailed=coefficient_detailed,
            interaction_detailed=interaction_detailed,
            mean_X_0=mean_X_0,
            mean_X_1=mean_X_1,
            categorical_to_dummy=categorical_to_dummy,
            direction=direction,
        )

    def _harmonize_common_support(self, data: pd.DataFrame):
        """Solve the common support problem by removing rows so that the two groups have the same set of dummies/categories."""
        y = {}
        X = {}
        X_model_spec = {}
        for group in self.groups_:
            group_mask = data[self.group_variable] == group
            # ensure_full_rank=False since we're doing data clean up here, not modeling
            # We don't want the base to interfere with the harmonization
            # For example, when a base is excluded from a group's model matrix, making it appear to not be exclusive to that group
            y[group], X[group] = Formula(self.formula).get_model_matrix(
                data.loc[group_mask, :], output="pandas", ensure_full_rank=False
            )
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
            outcome_among_removed_val = (
                float(y[this][rows_to_remove].mean().iloc[0]) if len(y[this][rows_to_remove]) > 0 else float("nan")
            )
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
        """Apply Gardeazabal and Ugidos (2004) adjustment for omitted group problem.

        For each categorical variable:
        1. Insert coefficient of 0 for omitted base category
        2. Calculate mean of all dummy coefficients for that categorical variable
        3. Subtract this mean from each dummy coefficient
        4. Add this mean to the intercept coefficient

        Args:
            coef: Original coefficients from OLS regression.
            weight: If not set, perform the "classic" GU adjustment.
                If set, a useful set of weights is the relative frequency of the categories,
                which result in the adjusted Intercept equalling the overall mean outcome,
                and consequently the coef as deviation from the overall mean.

        Returns:
            Adjusted coefficients.
        """

        new_coef = pd.Series(dtype=float)
        for term, term_slice in self.X_model_spec.term_slices.items():
            if term not in term_dummies(self.X_model_spec):
                # Not a categorical term, so just append the original coef
                new_coef = pd.concat([new_coef, coef[term_slice]], axis=0)
            else:
                # It's a categorical term, so let's apply GU adjustment
                if len(term.factors) > 1:
                    raise ValueError("We only support single categorical variable, not interaction")
                factor = term.factors[0]
                contrast_state = self.X_model_spec.factor_contrasts[factor]
                base_category = get_base_category(contrast_state)
                base_category_column_name = contrast_state.contrasts.get_factor_format(
                    levels=contrast_state.levels
                ).format(name=repr(factor), field=base_category)

                # Create extended coefficient series including base category (coefficient = 0)
                extended_coefs = pd.concat([coef[term_slice], pd.Series({base_category_column_name: 0.0})], axis=0)
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
                new_coef = pd.concat([new_coef, extended_coefs], axis=0)
                if "Intercept" in new_coef.index:
                    new_coef["Intercept"] += mean_coef

        # Ensure return type is Series (pd.concat can infer Series | DataFrame)
        return pd.Series(new_coef)
