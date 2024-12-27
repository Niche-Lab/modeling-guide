Below is a common scenario illustrating how to compute the expected ￼ and the expected RMSE when you have two independent variables ￼ and ￼ drawn i.i.d. from ￼ and you fit a simple linear regression of ￼ on ￼ (with an intercept):
	•	Because ￼ and ￼ are truly independent, the true (population) slope is ￼.
	•	In each finite sample of size ￼, you get a sample slope ￼ and intercept ￼, hence a fitted model ￼.
	•	The sample correlation ￼ between ￼ and ￼ will fluctuate around 0 but typically will not be exactly 0; thus the sample-based ￼ also fluctuates.

Below we work out:
	1.	￼
	2.	￼

under the null scenario that ￼ (i.e., ￼ and ￼ really are independent).

1. The expected ￼

For simple linear regression with a single predictor ￼, the regression ￼ is exactly the square of the sample correlation ￼.  Thus,

￼

When ￼ and ￼ are i.i.d. normal ￼ and truly independent (￼), the distribution of the sample correlation ￼ is well known.  A standard (and somewhat surprising) closed-form result is:


\[
\boxed{
\mathbb{E}\bigl[r^2\bigr]
\;=\;
\frac{1}{\,n-1\,}.
}
\]


Hence,

￼

This result holds exactly for samples from a bivariate normal with true correlation 0.  It also serves as a good approximation in many near-normal or large-sample settings.

Intuition

Even though the true correlation is 0, random sampling induces small spurious correlations, and on average ￼ comes out to ￼.  In other words, in a regression of ￼ on an unrelated ￼, you still expect a small positive ￼ in your sample purely by chance, but that shrinks as ￼ grows larger.

1. The expected RMSE

In the same setup, we typically estimate the regression model

￼

where ￼ and ￼ are found by ordinary least squares.  Because ￼ and ￼ are truly independent, the true slope is 0, and the true noise variance is ￼.

The formula for RMSE

By definition,

\[
\mathrm{RMSE}
\;=\;
\sqrt{
\frac{1}{n}\sum_{i=1}^n
\bigl(Y_i - \hat{Y}_i\bigr)^2
}
\;=\;
\sqrt{\frac{\mathrm{SSE}}{n}},
\]

where
￼

For simple linear regression with an intercept, under the “null model” (true slope = 0), the usual theory tells us:
	1.	￼,  because we are estimating 2 parameters (intercept and slope).
	2.	Here, ￼.  So ￼.

Hence,

￼

Thus its expectation is

\[
\mathbb{E}[\mathrm{RMSE}]
\;=\;
\mathbb{E}\!\Bigl[\sqrt{\tfrac{\chi^2_{n-2}}{n}}\Bigr]
\;=\;
\frac{1}{\sqrt{n}}\,
\mathbb{E}\!\bigl[\chi_{n-2}\bigr],
\]

where ￼ denotes a chi distribution with ￼ degrees of freedom (the square root of a ￼-distributed variable).

Closed-form and approximations

The mean of a ￼ random variable (chi distribution with ￼ degrees of freedom) is

\[
\mathbb{E}[\chi_k]
\;=\;
\sqrt{2}\,\frac{\Gamma\!\bigl(\tfrac{k+1}{2}\bigr)}{\Gamma\!\bigl(\tfrac{k}{2}\bigr)}.
\]

So we get

\[
\boxed{
\mathbb{E}[\mathrm{RMSE}]
\;=\;
\frac{1}{\sqrt{n}}
\,\sqrt{2}\,
\frac{\Gamma\!\Bigl(\tfrac{n-1}{2}\Bigr)}{\Gamma\!\Bigl(\tfrac{n-2}{2}\Bigr)}.
}
\]

That is an exact expression for all ￼.  Although it looks a bit complicated, it can be evaluated numerically for any ￼.

Large-￼ approximation

For large ￼, ￼ is close to a normal with mean ￼.  More precisely, a common asymptotic approximation is

\[
\mathbb{E}[\chi_{n-2}]
\,\approx\,
\sqrt{n-2 \,}\,\bigl(1 - \tfrac{1}{4(n-2)}\bigr),
\]

or more simply ￼.  Thus,

￼

In other words, if ￼ is unrelated to ￼, you will typically get an RMSE near 1 (the true ￼) once ￼ is reasonably large.

Final results

Putting it all together for two i.i.d. standard normal variables ￼ (independent) and fitting a simple linear regression of ￼ on ￼:
	1.	Expected ￼:
￼
Even though the true correlation is zero, on average you will see an ￼ of ￼ just by random chance.
	2.	Expected RMSE:
\[
\boxed{\mathbb{E}[\mathrm{RMSE}]
\;=\;
\frac{1}{\sqrt{n}}\;\mathbb{E}[\chi_{n-2}]
\;=\;
\frac{\sqrt{2}}{\sqrt{n}}
\,\frac{\Gamma\bigl(\frac{n-1}{2}\bigr)}{\Gamma\bigl(\frac{n-2}{2}\bigr)}.
}
\]
	•	For large ￼, this is very close to 1 (the true noise std.\ dev.).

These formulas highlight that (i) the sample ￼ is positively biased away from zero, and (ii) the sample RMSE is a slightly biased estimator of the true ￼ (though it converges quickly to 1 as ￼ grows).