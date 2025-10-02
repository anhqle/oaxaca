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

## Why use this package?

If possible, you should use the Stata package `oaxaca`, which is the most feature-rich implementation (Jann, 2008). If you can't, existing implementations in R and Python are lacking:

1. The R [`oaxaca` package](https://cran.r-project.org/web/packages/oaxaca/index.html) does not permit more than 1 categorical variable ([discussion](https://stats.stackexchange.com/questions/543828/blinder-oaxaca-decomposition-and-gardeazabal-and-ugidos-2004-correction-in-the))
2. The Python [implementation in `statsmodels`](https://www.statsmodels.org/dev/generated/statsmodels.stats.oaxaca.OaxacaBlinder.html) only decomposes into the explained and unexplained part, without a "detailed decomposition" into the contribution of each predictor

For industry data science work, these limitations are prohibitive. This package thus fills in the gap by providing:

1. As table stakes, two-fold and three-fold decomposition, with detailed decomposition for each predictor
2. Multiple ways to deal with the "omitted base category problem" (see [below](#the-omitted-base-category-problem))
3. Automatic handling of the case when the two groups don't have a common support. For example, some occupations may only exist in 1975 and not 2025, and vice versa
4. Rich HTML table output

This package does not produce standard error. While possible to add, this feature is deprioritized for the application of industry data science because:

1. The standard error of the OLS coefficient is often negligible given the number of observation in industry.
2. The standard error of the covariates is 0 given the goal of explaining the difference between two groups in a fixed population. This goal contrasts with that of academics, which is to prove some hypotheses about the world. There, one can imagine a new sample from a superpopulation that has a different covariate distribution.

## References

Fortin, N., Lemieux, T., & Firpo, S. (2011). Decomposition methods in economics. In *Handbook of Labor Economics* (Vol. 4, pp. 1-102). Elsevier.

Jann, B. (2008). A Stata implementation of the Blinder-Oaxaca decomposition. *Stata Journal*, 8(4), 453-479.
