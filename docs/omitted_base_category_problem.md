The choice of the omitted base category in a regression affects the value of the other coefficients, which in turn affects the contribution of a predictor. This has the disturbing implication that, depending on the analyst's choice of the omitted base category, the same predictor may appear more or less important.

This is a well-known problem in the literature (see Jann, 2008, p. 9 for a discussion).

The package offers three solutions (via the `gu_adjustment` option):

1. Not do anything. The analyst can choose a business-relevant category to omit (conveniently via [R-style formula](https://matthewwardrop.github.io/formulaic/latest/guides/contrasts/#treatment-aka-dummy)). The intercept then represents the mean of omitted category, and the remaining dummy coefficients are deviation from this mean.
2. Restrict the coefficients for the single categories to sum to zero. The intercept then represents the mean of the categories. This is the common approach in the academic literature, proposed by Gardeazabal and Ugidos (2004) and Yun (2005).
3. Restrict the coefficients for the single categories to *weighted* sum to zero. The intercept then represents the overall mean. This probably makes the most sense in an industry data science application.


## References

Jann, B. (2008). A Stata implementation of the Blinder-Oaxaca decomposition. *Stata Journal*, 8(4), 453-479.

Gardeazabal, J., & Ugidos, A. (2004). More on identification in detailed wage decompositions. *The Review of Economics and Statistics*, 86(4), 1034–1036.

Yun, M.-S. (2005). A simple solution to the identification problem in detailed wage decompositions. *Economic Inquiry*, 43(4), 766–772.
