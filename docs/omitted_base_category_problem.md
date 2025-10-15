The choice of the omitted base category in a regression affects the value of the other coefficients, which in turn affects the contribution of a predictor. This has the disturbing implication that, depending on the analyst's choice of the omitted base category, the same predictor may appear more or less important.

This is a well-known problem in the literature (see Jann, 2008, p. 9 for a discussion). Some important implications:

1. The total `explained` and `unexplained` parts are invariant to the omitted group
2. The `explained` part of a categorical variable is invariant. The `unexplained` part is NOT invariant.
3. Both the `explained` and `unexplained` parts of the dummy within the categorical variable is NOT invariant.

The package offers three solutions (via the `gu_adjustment` option):

1. Not do anything. The analyst can choose a business-relevant category to omit (conveniently via [R-style formula](https://matthewwardrop.github.io/formulaic/latest/guides/contrasts/#treatment-aka-dummy)). The intercept then represents the mean of omitted category, and the remaining dummy coefficients are deviation from this mean.
2. Restrict the coefficients for the single categories to sum to zero. The intercept then represents the mean of the categories. This is the common approach in the academic literature, proposed by Gardeazabal and Ugidos (2004) and Yun (2005).
3. Restrict the coefficients for the single categories to *weighted* sum to zero. The intercept then represents the overall mean. This probably makes the most sense in an industry data science application.

## Proof

Following the notation in Jann (2008), we have the two-fold decomposition formula:

$m_A - m_B = (\bar{X}_{A} - \bar{X}_{B}) \, \beta_{A k}$

$\sum_{k \neq 1} (\bar{X}_{A k} - \bar{X}_{B k}) \, \beta_{A k}$

\[
= \sum_{k \neq 3} (\bar{X}_{A k} - \bar{X}_{B k}) \, \beta'_{A k}
\]

\[
= \sum_{k \neq 3} (\bar{X}_{A k} - \bar{X}_{B k}) (\beta_{A k} - \beta_{A 3})
\]

\[
= \sum_{k \neq 3} (\bar{X}_{A k} - \bar{X}_{B k}) \, \beta_{A k}
- \left( \sum_{k \neq 3} (\bar{X}_{A k} - \bar{X}_{B k}) \right) \beta_{A 3}
\]

\[
= (\bar{X}_{A 1} - \bar{X}_{B 1}) \beta_{A 1}
+ \beta_{A 3} \sum_{k \neq 3} (\bar{X}_{A k} - \bar{X}_{B k}) \, \beta_{A k}
\]


## References

Jann, B. (2008). A Stata implementation of the Blinder-Oaxaca decomposition. *Stata Journal*, 8(4), 453-479.

Gardeazabal, J., & Ugidos, A. (2004). More on identification in detailed wage decompositions. *The Review of Economics and Statistics*, 86(4), 1034–1036.

Yun, M.-S. (2005). A simple solution to the identification problem in detailed wage decompositions. *Economic Inquiry*, 43(4), 766–772.
