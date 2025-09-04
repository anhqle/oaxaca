from functools import cached_property
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

if TYPE_CHECKING:
    from .oaxaca import Oaxaca


def create_removal_details_html(removal_details: dict) -> str:
    """Create HTML for detailed dummy variable removal results.

    Parameters
    ----------
    removal_details : dict
        Result from removal_info property containing removal information

    Returns
    -------
    str
        HTML string for removal details
    """
    if not removal_details.get("has_removals"):
        return ""

    removal_html = """
    <div style="margin-top: 15px; padding: 10px; background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px;">
        <h5 style="color: #495057; margin-bottom: 10px;">Dummy Variable Removal Details</h5>
    """

    for group, g in removal_details.get("by_group", {}).items():
        removed = g.get("removed_dummies", [])
        if not removed:
            continue

        share_removed = g.get("share_removed", 0.0)
        outcome_among = g.get("outcome_among_removed", 0.0)
        outcome_pre = g.get("outcome_pre_removal", 0.0)
        outcome_post = g.get("outcome_post_removal", 0.0)

        removal_html += f"""
        <div style="margin-bottom: 10px; padding: 8px; background-color: white; border-radius: 3px;">
            <strong>Group {group}:</strong><br>
            <span style="margin-left: 15px;">Removed dummies: {", ".join(removed)}</span><br>
            <span style="margin-left: 15px;">Share removed: {share_removed:.4f}, Mean among removed: {outcome_among:.4f}</span><br>
            <span style="margin-left: 15px;">Group mean adjustment: {outcome_pre:.4f} → {outcome_post:.4f}</span>
        </div>
        """

    removal_html += "</div>"
    return removal_html


def _format_cell(value, cell_type: str = "data", is_percentage: bool = False) -> str:
    """Format a single table cell for HTML output."""
    if cell_type == "header":
        return f'<th style="border: 1px solid #ddd; padding: 8px; text-align: {"left" if value == "Variable" else "right"}; {"width: 35%;" if value == "Variable" else ""}">{value}</th>'
    elif cell_type == "name":
        return f'<td style="border: 1px solid #ddd; padding: 8px;">{value}</td>'
    elif cell_type == "name_sub":
        return f'<td style="border: 1px solid #ddd; padding: 8px; padding-left: 24px; font-style: italic; color: #666;">{value}</td>'
    else:  # data cell
        if is_percentage:
            return f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{value:.1f}%</td>'
        else:
            return f'<td style="border: 1px solid #ddd; padding: 8px; text-align: right;">{value:.4f}</td>'


def _truncate_variable_name(var_name: str, display_len: Optional[int] = None) -> str:
    """
    Truncate variable name to specified length if display_len is provided.
    For categorical variables like 'education[T.LTHS]', only truncate the base name
    before the bracket and keep the bracket part intact.

    Parameters
    ----------
    var_name : str
        The variable name to potentially truncate
    display_len : int, optional
        Maximum length for variable names. If None, no truncation is applied.

    Returns
    -------
    str
        The original or truncated variable name
    """
    if display_len is None or len(var_name) <= display_len:
        return var_name

    # Check if this is a categorical variable with brackets
    # Find the LAST bracket pair in the variable name
    last_bracket_pos = var_name.rfind("[")
    if last_bracket_pos != -1:
        # Split into base name and last bracket part
        base_name = var_name[:last_bracket_pos]
        bracket_part = var_name[last_bracket_pos:]

        # Only truncate the base name if it's longer than display_len
        if len(base_name) > display_len:
            truncated_base = base_name[:display_len]
            return f"{truncated_base}{bracket_part}"
        else:
            return var_name
    else:
        # Regular variable, truncate normally
        return var_name[:display_len]


def _extract_common_variable_name(dummy_vars):
    """
    Extract the common variable name from a list of dummy variable names.

    For example, given ['education[T.high_school]', 'education[T.college]', 'education[T.graduate]'],
    this would return 'education'.

    Parameters
    ----------
    dummy_vars : list
        List of dummy variable names

    Returns
    -------
    str
        The common variable name extracted from the dummy variables
    """
    if not dummy_vars:
        return ""

    # Take the first dummy variable as reference
    first_dummy = dummy_vars[0]

    # Find the common prefix by looking for the opening bracket
    # This handles formats like 'education[T.high_school]' or 'education[high_school]'
    bracket_pos = first_dummy.find("[")
    if bracket_pos != -1:
        common_name = first_dummy[:bracket_pos]
    else:
        # If no bracket found, try to find common prefix among all dummy names
        common_name = first_dummy
        for dummy in dummy_vars[1:]:
            # Find longest common prefix
            i = 0
            while i < min(len(common_name), len(dummy)) and common_name[i] == dummy[i]:
                i += 1
            common_name = common_name[:i]

    return common_name.strip()


class OaxacaResults:
    def __init__(
        self,
        oaxaca_instance: "Oaxaca",
        total_difference: float,
        mean_X_0: pd.Series,
        mean_X_1: pd.Series,
        categorical_to_dummy: dict,
        direction: str,
    ):
        self._oaxaca = oaxaca_instance
        self.total_difference = total_difference
        self.mean_X_0 = mean_X_0
        self.mean_X_1 = mean_X_1
        self.X_diff = mean_X_0 - mean_X_1 if direction == "group0 - group1" else mean_X_1 - mean_X_0
        self.direction = direction
        self.categorical_to_dummy = categorical_to_dummy
        self.dummy_to_categorical = {}
        for categorical_term, dummy_vars in self.categorical_to_dummy.items():
            for dummy_var in dummy_vars:
                self.dummy_to_categorical[dummy_var] = categorical_term

    @cached_property
    def removal_info(self) -> dict:
        """Get detailed impact of removals per group and net impact.

        Returns
        -------
        dict
            {
              'by_group': {
                 group_value: {
                    'removed_dummies': List[str],
                    'share_removed': float,
                    'outcome_pre_removal': float,
                    'outcome_post_removal': float,
                    'outcome_among_removed': float,
                    'mean_adjustment': float,
                 },
                 ...
              },
              'removal_contribution': float,
              'removal_contribution_pct': float,
              'has_removals': bool,
            }
        """
        by_group: dict[Any, dict[str, Any]] = {}
        if not hasattr(self._oaxaca, "dummy_removal_result_") or not self._oaxaca.dummy_removal_result_:
            return {
                "by_group": by_group,
                "removal_contribution": 0.0,
                "removal_contribution_pct": 0.0,
                "has_removals": False,
            }

        # Build per-group details
        for group in self._oaxaca.groups_:
            info = self._oaxaca.dummy_removal_result_.get(group, {})
            removed = info.get("removed_dummies", []) or []
            pre = info.get("outcome_pre_removal")
            post = info.get("outcome_post_removal")
            among = info.get("outcome_among_removed")
            share = info.get("share_removed", 0.0)
            mean_adjustment = info.get("mean_adjustment")

            if mean_adjustment is None:
                if removed:
                    # Calculate how much removals shifted this group's mean
                    # Handle pandas Series case
                    pre_val = float(pre.iloc[0]) if hasattr(pre, "iloc") else float(pre) if pre is not None else 0.0

                    if hasattr(post, "iloc"):
                        post_val = float(post.iloc[0])
                    else:
                        post_val = float(post) if post is not None else 0.0
                    mean_adjustment = pre_val - post_val
                else:
                    mean_adjustment = 0.0

            by_group[group] = {
                "removed_dummies": removed,
                "share_removed": float(share) if share is not None else 0.0,
                "outcome_pre_removal": float(pre) if pre is not None else 0.0,
                "outcome_post_removal": float(post) if post is not None else 0.0,
                "outcome_among_removed": float(among) if among is not None else 0.0,
                "mean_adjustment": float(mean_adjustment),
            }

        # Compute net contribution based on direction
        groups = self._oaxaca.groups_
        adj_0 = by_group.get(groups[0], {}).get("mean_adjustment", 0.0)
        adj_1 = by_group.get(groups[1], {}).get("mean_adjustment", 0.0)
        removal_contribution = adj_0 - adj_1 if self.direction == "group0 - group1" else adj_1 - adj_0

        removal_contribution_pct = removal_contribution / abs(self.total_difference) * 100

        has_removals = (
            any(len(g.get("removed_dummies", [])) > 0 for g in by_group.values()) and abs(removal_contribution) > 1e-10
        )

        return {
            "by_group": by_group,
            "removal_contribution": float(removal_contribution),
            "removal_contribution_pct": float(removal_contribution_pct),
            "has_removals": bool(has_removals),
        }

    def detailed_contributions(self, decomposition_components: dict[str, pd.Series]) -> pd.DataFrame:
        """
        decomposition_components are the decomposition terms
        - for two_fold: explained + unexplained
        - for three_fold: endowment + coefficient + interaction

        These are pandas Series indexed by variable names.
        """
        index_tuples = []
        result_rows = []

        # Ignore type checking because type checker mistakenly thought sum() returns a float
        decomposition_components["total"] = sum(decomposition_components.values())  # type: ignore[invalid-assignment]

        for var_name in self.mean_X_0.index:
            result_row = {}
            for component in decomposition_components:
                result_row[component] = decomposition_components[component][var_name]
                result_row[f"{component}_pct"] = (
                    decomposition_components[component][var_name] / abs(self.total_difference) * 100
                )

            if var_name in self.dummy_to_categorical:
                # Categorical dummy variable
                categorical_term = self.dummy_to_categorical[var_name]
                common_var_name = _extract_common_variable_name(self.categorical_to_dummy[categorical_term])
                index_tuples.append((common_var_name, var_name))
                result_row["variable_type"] = "categorical"
            else:
                # Continuous variable
                index_tuples.append((var_name, var_name))
                result_row["variable_type"] = "continuous"

            result_rows.append(result_row)

        # Create MultiIndex DataFrame
        index = pd.MultiIndex.from_tuples(index_tuples, names=("variable_group", "category"))
        df = pd.DataFrame(result_rows, index=index)

        return df

    def contributions(self) -> pd.DataFrame:
        """
        Create a table showing aggregated contributions (i.e. categorical variable gets the sum of its dummies).
        """
        detailed_df = self.detailed_contributions().drop(columns=["variable_type"])

        # Group by the first level of MultiIndex (variable_group) and sum
        # Use sort=False to preserve the original order from explained_detailed.index
        aggregated = detailed_df.groupby(level="variable_group", sort=False).sum()

        aggregated = aggregated.reset_index()
        aggregated = aggregated.rename(columns={"variable_group": "variable"})

        return aggregated

    def __repr__(self):
        return f"OaxacaResults(groups={self._oaxaca.groups_}, group_variable='{self._oaxaca.group_variable}')"

    def _repr_html_(self) -> str:
        """Rich HTML display for Jupyter notebooks with detailed information."""
        return self.to_html(display_len=None, sort=True)


class TwoFoldResults(OaxacaResults):
    """Results class for Oaxaca-Blinder decomposition."""

    def __init__(
        self,
        oaxaca_instance: "Oaxaca",
        total_difference: float,
        explained: float,
        unexplained: float,
        explained_detailed: pd.Series,
        unexplained_detailed: pd.Series,
        coef_nondiscriminatory: pd.Series,
        weights: dict[Any, float],
        mean_X_0: pd.Series,
        mean_X_1: pd.Series,
        categorical_to_dummy: dict,
        direction: str,
    ):
        super().__init__(oaxaca_instance, total_difference, mean_X_0, mean_X_1, categorical_to_dummy, direction)
        self.explained = explained
        self.unexplained = unexplained
        self.explained_detailed = explained_detailed
        self.unexplained_detailed = unexplained_detailed
        self.coef_nondiscriminatory = coef_nondiscriminatory
        self.weights = weights

    def contributions(self) -> pd.DataFrame:
        """
        Create a table showing only categorical variable contributions (aggregated).

        Returns
        -------
        pd.DataFrame
            A table with columns for Variable, Mix-shift, Within-slice, Total,
            and their corresponding percentages. Only includes categorical variables
            and continuous variables, not individual dummy categories.
        """
        return super().contributions()

    def detailed_contributions(self) -> pd.DataFrame:
        """
        Create a table showing detailed contributions with proper hierarchical structure.

        Returns
        -------
        pd.DataFrame
            A table with MultiIndex (variable_group, Category) showing individual
            category contributions with their parent categorical variable.
        """
        return super().detailed_contributions({
            "explained_detailed": self.explained_detailed,
            "unexplained_detailed": self.unexplained_detailed,
        })

    def _create_detailed_contributions_table(self, display_len: Optional[int] = None, sort: bool = True) -> str:
        """Create detailed contributions table in HTML format.

        Parameters
        ----------
        display_len : int, optional
            Maximum length for variable names in output tables. If provided,
            variable names will be truncated to this length.
        sort : bool, default True
            Whether to sort variables by their absolute total contributions.
        """
        # Get both dataframes from class properties
        detailed_df = self.detailed_contributions()
        categorical_df = self.contributions()

        # Apply sorting if requested
        if sort and len(categorical_df) > 0:
            categorical_df = categorical_df.reindex(categorical_df["total"].abs().sort_values(ascending=False).index)

        lines = []
        # Table header
        lines.append('<table style="border-collapse: collapse; width: 100%; font-size: 0.9em;">')
        lines.append("<thead>")
        lines.append('<tr style="background-color: #f0f0f0;">')

        for header in ["Variable", "Mix-shift", "Mix %", "Within-slice", "Within %", "Total", "Tot %"]:
            lines.append(_format_cell(header, "header"))

        lines.append("</tr>")
        lines.append("</thead>")
        lines.append("<tbody>")

        # Process each categorical variable and its details directly
        for _, cat_row in categorical_df.iterrows():
            var_name = cat_row["variable"]
            explained = cat_row["explained_detailed"]
            explained_pct = cat_row["explained_detailed_pct"]
            unexplained = cat_row["unexplained_detailed"]
            unexplained_pct = cat_row["unexplained_detailed_pct"]
            total = cat_row["total"]
            total_pct = cat_row["total_pct"]

            # Apply truncation if specified
            display_var_name = _truncate_variable_name(var_name, display_len)

            # Add main variable row
            lines.append('<tr style="font-weight: bold; background-color: #f8f9fa;">')
            lines.append(_format_cell(display_var_name, "name"))
            lines.append(_format_cell(explained))
            lines.append(_format_cell(explained_pct, is_percentage=True))
            lines.append(_format_cell(unexplained))
            lines.append(_format_cell(unexplained_pct, is_percentage=True))
            lines.append(_format_cell(total))
            lines.append(_format_cell(total_pct, is_percentage=True))
            lines.append("</tr>")

            # Individual categories only if it's truly categorical (multiple categories)
            var_metadata = detailed_df.loc[detailed_df.index.get_level_values("variable_group") == var_name]
            if len(var_metadata) > 1:  # More than one category means it's categorical
                for category_name, detail_row in var_metadata.iterrows():
                    category_explained = detail_row["explained_detailed"]
                    category_explained_pct = detail_row["explained_detailed_pct"]
                    category_unexplained = detail_row["unexplained_detailed"]
                    category_unexplained_pct = detail_row["unexplained_detailed_pct"]
                    category_total = detail_row["total"]
                    category_total_pct = detail_row["total_pct"]

                    # category_name[1] is the actual category name (second part of MultiIndex)
                    display_category = _truncate_variable_name(category_name[1], display_len)

                    # Add subcategory row
                    lines.append("<tr>")
                    lines.append(_format_cell(display_category, "name_sub"))
                    lines.append(_format_cell(category_explained))
                    lines.append(_format_cell(category_explained_pct, is_percentage=True))
                    lines.append(_format_cell(category_unexplained))
                    lines.append(_format_cell(category_unexplained_pct, is_percentage=True))
                    lines.append(_format_cell(category_total))
                    lines.append(_format_cell(category_total_pct, is_percentage=True))
                    lines.append("</tr>")

        # Calculate and add total row
        total_row_explained = self.explained
        total_row_unexplained = self.unexplained
        total_row_total = self.total_difference
        total_explained_pct = total_row_explained / abs(self.total_difference) * 100
        total_unexplained_pct = total_row_unexplained / abs(self.total_difference) * 100
        total_pct = 100.0

        lines.append('<tr style="font-weight: bold; background-color: #e9ecef; border-top: 2px solid #333;">')
        lines.append(_format_cell("total", "name"))
        lines.append(_format_cell(total_row_explained))
        lines.append(_format_cell(total_explained_pct, is_percentage=True))
        lines.append(_format_cell(total_row_unexplained))
        lines.append(_format_cell(total_unexplained_pct, is_percentage=True))
        lines.append(_format_cell(total_row_total))
        lines.append(_format_cell(total_pct, is_percentage=True))
        lines.append("</tr>")
        lines.append("</tbody>")
        lines.append("</table>")

        return "".join(lines)

    def to_html(self, display_len: Optional[int] = None, sort: bool = True) -> str:
        """Generate HTML representation with optional variable name truncation.

        Parameters
        ----------
        display_len : int, optional
            Maximum length for variable names in output tables. If provided,
            variable names will be truncated to this length.
        sort : bool, default True
            Whether to sort variables by their absolute total contributions.

        Returns
        -------
        str
            HTML string representation of the decomposition results
        """
        if not hasattr(self, "total_difference"):
            return "<p><strong>OaxacaResults</strong> (not yet computed - call two_fold() first)</p>"

        if self.direction == "group0 - group1":
            direction_text = f"{self._oaxaca.groups_[0]} - {self._oaxaca.groups_[1]}"
        else:
            direction_text = f"{self._oaxaca.groups_[1]} - {self._oaxaca.groups_[0]}"

        detailed_table_html = self._create_detailed_contributions_table(display_len, sort)

        removal_section = ""
        if self.removal_info["has_removals"]:
            removal_section = f"""
            <div style="margin-top: 15px; padding: 10px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px;">
                <strong>⚠️ Category Removal Impact:</strong> {self.removal_info["removal_contribution"]:.4f}
                ({self.removal_info["removal_contribution_pct"]:.1f}% of total difference)
            </div>
            """

            # Add detailed removal results if available
            removal_details = create_removal_details_html(self.removal_info)
            removal_section += removal_details

        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 1000px;">
            <h3 style="color: #2c3e50; margin-bottom: 15px;">Oaxaca-Blinder Decomposition Results</h3>

            <div style="margin-bottom: 15px;">
                <p style="margin: 3px 0;"><strong>Group Variable:</strong> {self._oaxaca.group_variable} | <strong>Groups:</strong> {self._oaxaca.groups_[0]} vs {self._oaxaca.groups_[1]} | <strong>Direction:</strong> {direction_text} | <strong>Weights:</strong> {self.weights}</p>
                <p style="margin: 3px 0;"><strong>Total Difference:</strong> {self.total_difference:.4f} | <strong>Mix-shift:</strong> {self.explained:.4f} ({self.explained / abs(self.total_difference) * 100:.1f}%) | <strong>Within-slice:</strong> {self.unexplained:.4f} ({self.unexplained / abs(self.total_difference) * 100:.1f}%)</p>
            </div>

            {removal_section}

            <h4 style="color: #2c3e50; margin-top: 20px; margin-bottom: 10px;">Detailed Variable Contributions</h4>
            {detailed_table_html}

            <div style="margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 4px;">
                <p style="margin: 0; font-size: 0.9em; color: #666;">
                    <strong>💡 For programmatic access:</strong><br>
                    • <code>contributions</code> - aggregated categorical variables<br>
                    • <code>detailed_contributions</code> - individual categories with hierarchy<br>
                    • <code>removal_info</code> - per-group removal impact details
                </p>
            </div>
        </div>
        """
        return html

    def print_ols(self, display_len: Optional[int] = None):
        """Print OLS regression results for each group.

        Parameters
        ----------
        out : List[str], default ["diff"]
            Controls which sections to print. Options are:
            - "group0": Print results for group 0
            - "group1": Print results for group 1
            - "diff": Print coefficient comparison table
        display_len : int, optional
            Maximum length for variable names in output tables. If provided,
            variable names will be truncated to this length.
        """
        print("OLS Regression Results by Group")
        print("=" * 60)

        for _, group in enumerate(self._oaxaca.groups_):
            model = self._oaxaca.models_[group]
            group_stats = self._oaxaca.group_stats_[group]

            print(f"\nGroup: {group}")
            print("-" * 40)
            print(f"Number of observations: {group_stats['n_obs']}")
            print(f"R-squared: {group_stats['r_squared']:.4f}")
            print(f"Mean of dependent variable: {group_stats['mean_y']:.4f}")
            print(f"Std of dependent variable: {group_stats['std_y']:.4f}")

            print("\nCoefficients:")
            print(f"{'Variable':<40} {'Coeff':>10} {'Std Err':>10} {'t':>8} {'P>|t|':>8}")
            print("-" * 61)

            for var_name, coeff in model.params.items():
                std_err = model.bse[var_name]
                t_stat = model.tvalues[var_name]
                p_value = model.pvalues[var_name]

                # Truncate variable name if display_len is specified
                display_var_name = _truncate_variable_name(var_name, display_len)

                print(f"{display_var_name:<40} {coeff:>10.4f} {std_err:>10.4f} {t_stat:>8.3f} {p_value:>8.3f}")

        print("Coefficient Comparison Between Groups")
        print("=" * 80)

        # Get the two groups
        group_0, group_1 = self._oaxaca.groups_

        # Determine column order based on direction
        if self.direction == "group0 - group1":
            first_group, second_group = group_0, group_1
            first_label, second_label = f"Group {group_0}", f"Group {group_1}"
            diff_label = f"{group_0} - {group_1}"
        else:  # "group1 - group0"
            first_group, second_group = group_1, group_0
            first_label, second_label = f"Group {group_1}", f"Group {group_0}"
            diff_label = f"{group_1} - {group_0}"

        # Get coefficients for both groups
        coef_first = self._oaxaca.coef_[first_group]
        coef_second = self._oaxaca.coef_[second_group]

        # Ensure conformable dimensions (handle missing coefficients)
        coef_first, coef_second = self._oaxaca._ensure_conformable_dimensions(coef_first, coef_second)

        # Calculate difference
        coef_diff = coef_first - coef_second

        print(f"Direction: {diff_label}")
        print()

        # Print table header
        header = f"{'Variable':<40} {first_label:>12} {second_label:>12} {'Difference':>12}"
        print(header)
        print("-" * len(header))

        # Print coefficients for each variable
        for var_name in coef_first.index:
            first_coef = coef_first[var_name]
            second_coef = coef_second[var_name]
            diff_coef = coef_diff[var_name]

            # Truncate variable name if display_len is specified
            display_var_name = _truncate_variable_name(var_name, display_len)

            print(f"{display_var_name:<40} {first_coef:>12.4f} {second_coef:>12.4f} {diff_coef:>12.4f}")

    def x_difference_table(self) -> pd.DataFrame:
        """
        Create a table showing the difference in X (predictor variables) between the two groups.

        Returns
        -------
        pd.DataFrame
            A table with columns for Variable, Group 0 Mean, Group 1 Mean, and Difference
        """
        # Create DataFrame using the actual mean_X values used in decomposition
        groups = self._oaxaca.groups_
        df = pd.DataFrame({
            "Variable": self.explained_detailed.index.tolist(),
            f"Group {groups[0]} Mean": self.mean_X_0,
            f"Group {groups[1]} Mean": self.mean_X_1,
            "Difference": self.X_diff,
        })

        return df

    def print_x(self, display_len: Optional[int] = None):
        """
        Print a formatted table showing the difference in X (predictor variables) between the two groups.

        Parameters
        ----------
        display_len : int, optional
            Maximum length for variable names in output tables. If provided,
            variable names will be truncated to this length.
        """
        print("Difference in X (Predictor Variables) Between Groups")
        print("=" * 80)
        groups = self._oaxaca.groups_
        print(f"Group Variable: {self._oaxaca.group_variable}")
        print(f"Groups: {groups[0]} (Group 0) vs {groups[1]} (Group 1)")

        # Show direction information based on self.direction
        if self.direction == "group0 - group1":
            difference_text = f"Group {groups[0]} Mean - Group {groups[1]} Mean"
        else:
            difference_text = f"Group {groups[1]} Mean - Group {groups[0]} Mean"

        print(f"Difference = {difference_text}")
        print()

        df = self.x_difference_table()

        # Print formatted table
        if self.direction == "group0 - group1":
            header = (
                f"{'Variable':<40} {str(groups[0]) + ' Mean':>15} {str(groups[1]) + ' Mean':>15} {'Difference':>15}"
            )
        elif self.direction == "group1 - group0":
            header = (
                f"{'Variable':<40} {str(groups[1]) + ' Mean':>15} {str(groups[0]) + ' Mean':>15} {'Difference':>15}"
            )
        print(header)
        print("-" * len(header))

        for _, row in df.iterrows():
            var_name = row["Variable"]
            group_0_mean = row[f"Group {self._oaxaca.groups_[0]} Mean"]
            group_1_mean = row[f"Group {self._oaxaca.groups_[1]} Mean"]
            difference = row["Difference"]

            # Truncate variable name if display_len is specified
            display_var_name = _truncate_variable_name(var_name, display_len)

            if self.direction == "group0 - group1":
                print(f"{display_var_name:<40} {group_0_mean:>15.4f} {group_1_mean:>15.4f} {difference:>15.4f}")
            elif self.direction == "group1 - group0":
                print(f"{display_var_name:<40} {group_1_mean:>15.4f} {group_0_mean:>15.4f} {difference:>15.4f}")


class ThreeFoldResults(OaxacaResults):
    def __init__(
        self,
        oaxaca_instance: "Oaxaca",
        total_difference: float,
        endowment: float,
        coefficient: float,
        interaction: float,
        endowment_detailed: pd.Series,
        coefficient_detailed: pd.Series,
        interaction_detailed: pd.Series,
        mean_X_0: pd.Series,
        mean_X_1: pd.Series,
        categorical_to_dummy: dict,
        direction: str,
    ):
        super().__init__(oaxaca_instance, total_difference, mean_X_0, mean_X_1, categorical_to_dummy, direction)
        self.endowment = endowment
        self.coefficient = coefficient
        self.interaction = interaction
        self.endowment_detailed = endowment_detailed
        self.coefficient_detailed = coefficient_detailed
        self.interaction_detailed = interaction_detailed

    def contributions(self) -> pd.DataFrame:
        return super().contributions()

    def detailed_contributions(self) -> pd.DataFrame:
        """
        Create a table showing detailed contributions with proper hierarchical structure.

        Returns
        -------
        pd.DataFrame
            A table with MultiIndex (variable_group, Category) showing individual
            category contributions with their parent categorical variable.
        """
        df = pd.DataFrame({
            "Variable": self.endowment_detailed.index.tolist(),
            "Endowment": self.endowment_detailed,
            "Coefficient": self.coefficient_detailed,
            "Interaction": self.interaction_detailed,
        })

        return df
