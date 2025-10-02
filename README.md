# oaxaca

[![Release](https://img.shields.io/github/v/release/anhqle/oaxaca)](https://img.shields.io/github/v/release/anhqle/oaxaca)
[![Build status](https://img.shields.io/github/actions/workflow/status/anhqle/oaxaca/main.yml?branch=main)](https://github.com/anhqle/oaxaca/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/anhqle/oaxaca/branch/main/graph/badge.svg)](https://codecov.io/gh/anhqle/oaxaca)
[![Commit activity](https://img.shields.io/github/commit-activity/m/anhqle/oaxaca)](https://img.shields.io/github/commit-activity/m/anhqle/oaxaca)
[![License](https://img.shields.io/github/license/anhqle/oaxaca)](https://img.shields.io/github/license/anhqle/oaxaca)

- **Github repository**: <https://github.com/anhqle/oaxaca/>
- **Documentation** <https://anhqle.github.io/oaxaca/>

The Oaxaca-Blinder decomposition is a statistical method used to explain the difference in outcomes between two groups by decomposing it into:

- A part that is "explained" by differences in group predictor
- A part that remains "unexplained"

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
