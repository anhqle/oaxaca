from formulaic.utils.sentinels import Sentinel


def get_base_category(contrast_state):
    """Get the base category for a categorical variable, handling UNSET case.

    Args:
        contrast_state: The contrast state from formulaic model spec.

    Returns:
        The base category (first level if UNSET, otherwise the specified base).
    """
    base_category = contrast_state.contrasts.base
    if base_category == Sentinel.UNSET:
        base_category = contrast_state.levels[0]
    return base_category


def term_dummies(model_spec) -> dict:
    """Create mapping from categorical terms to their dummy column names.

    Args:
        model_spec: The model specification from formulaic containing terms, factors,
            and column names.

    Returns:
        Dictionary mapping terms (representing categorical variables) to lists of
        dummy column names.
    """
    results = {}
    for term in model_spec.terms:
        for factor in term.factors:
            if factor in model_spec.factor_contrasts:
                results[term] = model_spec.column_names[model_spec.term_slices[term]]
    return results


def dummies(model_spec) -> list:
    """Create a list of dummy column names from the model specification.

    Args:
        model_spec: The model specification from formulaic.

    Returns:
        List of dummy column names.
    """
    result = []
    for dummies in term_dummies(model_spec).values():
        result.extend(dummies)
    return result


def term_dummies_gu_adjusted(model_spec) -> dict:
    """Generate categorical to dummy mapping for GU-adjusted coefficients.

    This includes base categories and uses reformatted variable names.

    Args:
        model_spec: The model specification from formulaic containing terms, factors,
            and column names.

    Returns:
        Dictionary mapping terms (representing categorical variables) to lists of
        dummy column names (including base category, with reformatted names).
    """
    results = {}
    for term in model_spec.terms:
        for factor in term.factors:
            if factor in model_spec.factor_contrasts:
                dummies = list(model_spec.column_names[model_spec.term_slices[term]])
                # Add base category
                if len(term.factors) == 1:
                    contrast_state = model_spec.factor_contrasts[factor]
                    base_category = get_base_category(contrast_state)
                    factor_format = contrast_state.contrasts.get_factor_format(levels=contrast_state.levels)
                    base_category_column_name = factor_format.format(name=repr(factor), field=base_category)
                    dummies.append(base_category_column_name)

                dummies = [dummy.replace("[T.", "[", 1) for dummy in dummies]
                results[term] = dummies
    return results
