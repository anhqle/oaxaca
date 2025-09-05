# Oaxaca-Blinder Decomposition

[![Release](https://img.shields.io/github/v/release/anhqle/oaxaca)](https://img.shields.io/github/v/release/anhqle/oaxaca)
[![Build status](https://img.shields.io/github/actions/workflow/status/anhqle/oaxaca/main.yml?branch=main)](https://github.com/anhqle/oaxaca/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/anhqle/oaxaca)](https://img.shields.io/github/commit-activity/m/anhqle/oaxaca)
[![License](https://img.shields.io/github/license/anhqle/oaxaca)](https://img.shields.io/github/license/anhqle/oaxaca)

The Oaxaca-Blinder decomposition is a statistical method used to explain the difference in outcomes between two groups by decomposing it into:

1. a part that is "explained" by differences in group covariate
2. a part that remains "unexplained"

For example, the gender wage gap can be partly "explained" by the difference in education and work experience between men and women. The remaining "unexplained" part is typically considered discrimination.

For a methodological review, see Jann (2008) and Fortin (2011).

## Why use this package?

The most feature-rich implementation of Oaxaca-Blinder is in Stata (Jann 2008). However, existing implementations in R and Python are lacking:

1. The R [`oaxaca` package](https://cran.r-project.org/web/packages/oaxaca/index.html) does not permit more than 1 categorical variable ([discussion](https://stats.stackexchange.com/questions/543828/blinder-oaxaca-decomposition-and-gardeazabal-and-ugidos-2004-correction-in-the))
2. The Python [implementation in `statsmodel`](https://www.statsmodels.org/dev/generated/statsmodels.stats.oaxaca.OaxacaBlinder.html) only decomposes into the explained and unexplained part, without a "detailed decomposition" into the contribution of each predictor

For industry data science work, these limitations are prohibitive.

This package thus implements the following features

1. As table stakes, two-fold and three-fold decomposition, with detailed decomposition for each predictor
2. Multiple ways to deal with the "omitted base category problem."

In addition, since this package is developed in the context of private industry (as opposed to academic research), it makes a few design trade-offs:

1. Multiple ways to deal with the "omitted base category problem" are available. See Jann (2008, p9) for a discussion of this problem
2. No standard error is reported.
    - In the context of industry, the number of observation is often large enough that the standard error of the coefficient is negligible
    - Since the goal is to explain an observed difference between two groups (as opposed to proving some hypotheses about the world), the difference in covariates should be considered fixed with no standard error

## The omitted base category problem

TBD

## References

Jann, Ben. "A Stata implementation of the Blinder-Oaxaca decomposition." Stata journal 8, no. 4 (2008): 453-479.

Fortin, Nicole, Thomas Lemieux, and Sergio Firpo. "Decomposition methods in economics." In Handbook of labor economics, vol. 4, pp. 1-102. Elsevier, 2011.
