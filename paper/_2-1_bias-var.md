# Bias and Variance of Cross Validation

## Background

The choice of $K$ has a trade-off between the bias and the variance of estimating the true model generalization error.
A large $K$ will reduce the bias, as a larger size of training set represents more of the entire dataset. However, a large $K$ also increases the estimation variance as the test set is smaller in size and therefore the prediction error is more sensitive to the randomness of the test set. An extreme example of large $K$ is the leave-one-out cross validation (LOOCV) where $K$ is equal to the number of samples in the dataset. Since in each iteration, only one sample is evaluated for the prediction error. The error is sensitive to the choice of the sample and therefore has a high variance. LOOCV usually requires a large number of iterations to reduce the variance and achieve an unbiased estimation of the prediction errors. Hence, large $K$ or LOOCV is usually not recommended with small datasets. A detailed discussion of the trade-off has been discussed in (ref1, ref2).

In this section, a simulation study is conducted investigate how the interaction between the sample sizes and the number of folds affects the bias and variance of the cross validation procedure. An "In-Sample" metric is also reported to demonstrate an overoptimistic estimation when cross validation is not correctly implemented.

## Objectives

The goal of this study is to investigate the bias and variance of each performance estimators, which include "In-Sample" that validates the model on the same dataset used for training, K-fold cross validation where K is assigned as 2, 5, and 10, and LOOCV. The bias and variance of the performance estimation are calculated based on the following equations:

$$
\begin{align}
Bias &= \mathbb{E}\left[\hat{f}(x) - f(x)\right] \\
Var &= \mathbb{E}\left[\left(\hat{f}(x) - \mathbb{E}\left[\hat{f}(x)\right]\right)^2\right]
\end{align}
$$

where $\hat{f}(x)$ is estimated performance and $f(x)$ is the expected performance. Since it is difficult to obtain the expected performance if

the predicted value of the model and $f(x)$ is the ground truth. The bias and variance are calculated for each sample in the dataset and then averaged across all samples. The bias and variance are calculated for each sample in the dataset and then averaged across all samples. The bias and variance are calculated for each sample in the dataset and then averaged across all samples. The bias and variance are calculated for each sample in the dataset and then averaged across all samples. The bias and variance are calculated for each sample in the dataset and then averaged across all samples. The bias and variance are calculated for each sample in the dataset and then averaged across all samples. The bias and variance are calculated for each sample in the dataset and then averaged across all samples. The bias and variance are calculated for each sample in the dataset and then averaged across all samples. The bias and variance are calculated for each sample in the dataset and then averaged across all samples.


