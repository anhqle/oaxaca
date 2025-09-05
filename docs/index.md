# Oaxaca-Blinder Decomposition

[![Release](https://img.shields.io/github/v/release/anhqle/oaxaca)](https://img.shields.io/github/v/release/anhqle/oaxaca)
[![Build status](https://img.shields.io/github/actions/workflow/status/anhqle/oaxaca/main.yml?branch=main)](https://github.com/anhqle/oaxaca/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/anhqle/oaxaca)](https://img.shields.io/github/commit-activity/m/anhqle/oaxaca)
[![License](https://img.shields.io/github/license/anhqle/oaxaca)](https://img.shields.io/github/license/anhqle/oaxaca)

The Oaxaca-Blinder decomposition is a statistical method used to explain the difference in outcomes between two groups by decomposing it into:

1. A part that is "explained" by differences in group predictor
2. A part that remains "unexplained"

For example, the gender wage gap can be partly "explained" by the difference in education and work experience between men and women. The remaining "unexplained" part is typically considered discrimination.

For a methodological review, see Jann (2008) and Fortin et al. (2011).

## Demo

[Notebook](https://github.com/anhqle/oaxaca/blob/main/notebooks/minimal_oaxaca_demo.ipynb)

## Why use this package?

The most feature-rich implementation of Oaxaca-Blinder is in Stata (Jann, 2008). In contrast, existing implementations in R and Python are lacking:

1. The R [`oaxaca` package](https://cran.r-project.org/web/packages/oaxaca/index.html) does not permit more than 1 categorical variable ([discussion](https://stats.stackexchange.com/questions/543828/blinder-oaxaca-decomposition-and-gardeazabal-and-ugidos-2004-correction-in-the))
2. The Python [implementation in `statsmodels`](https://www.statsmodels.org/dev/generated/statsmodels.stats.oaxaca.OaxacaBlinder.html) only decomposes into the explained and unexplained part, without a "detailed decomposition" into the contribution of each predictor

For industry data science work, these limitations are prohibitive. This package thus fills in the gap by providing:

1. As table stakes, two-fold and three-fold decomposition, with detailed decomposition for each predictor
2. Multiple ways to deal with the "omitted base category problem" (see [below](#the-omitted-base-category-problem))
3. Automatic handling of the case when the two groups don't have a common support. For example, some occupations may only exist in 1975 and not 2025, and vice versa
4. Rich HTML table output

The main "limitation" of this package is the lack of standard error. While possible to add, this feature is deprioritized for the application of industry data science because:

1. The number of observation is often large enough that the standard error of the coefficient is negligible
2. Since the goal is to explain an observed difference between two groups (as opposed to proving some hypotheses about the world), the difference in predictors should be considered fixed. Therefore, the standard error of the predictors should be 0

## The omitted base category problem

In short, the choice of the omitted base category in a regression affects the value of the other coefficients, which in turn affects the contribution of a predictor. This has the disturbing implication that, depending on the analyst's choice of the omitted base category, the same predictor may appear more or less important.

This is a well-known problem in the literature (see Jann, 2008, p. 9 for a discussion).

The package offers three solutions (via the `gu_adjustment` option):

1. Not do anything. The analyst can choose a business-relevant category to omit (conveniently via [R-style formula](https://matthewwardrop.github.io/formulaic/latest/guides/contrasts/#treatment-aka-dummy)). The intercept then represents the mean of omitted category, and the remaining dummy coefficients are deviation from this mean.
2. Restrict the coefficients for the single categories to sum to zero. The intercept then represents the mean of the categories. This is the common approach in the academic literature, proposed by Gardeazabal and Ugidos (2004) and Yun (2005).
3. Restrict the coefficients for the single categories to *weighted* sum to zero. The intercept then represents the overall mean. This probably makes the most sense in an industry data science application.

## References

Fortin, N., Lemieux, T., & Firpo, S. (2011). Decomposition methods in economics. In *Handbook of Labor Economics* (Vol. 4, pp. 1-102). Elsevier.

Gardeazabal, J., & Ugidos, A. (2004). More on identification in detailed wage decompositions. *The Review of Economics and Statistics*, 86(4), 1034–1036.

Jann, B. (2008). A Stata implementation of the Blinder-Oaxaca decomposition. *Stata Journal*, 8(4), 453-479.

Yun, M.-S. (2005). A simple solution to the identification problem in detailed wage decompositions. *Economic Inquiry*, 43(4), 766–772.
