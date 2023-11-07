# Validation Bias and Variance

## Definition

The true generalization performance of the model $g(f_{\mathcal{D}})$ can only be approximated by averaging the performance metrics over infinite unseen datasets. However, in practice, the dataset $\mathcal{D}$ is finite and therefore there is always a bias in the estimation of $g(f_{\mathcal{D}})$. The validation bias can be defined as:

$$
\text{Bias} = \mathbb{E}[\hat{g}(f_{\mathcal{D}})] - g(f_{\mathcal{D}})
$$

For example, if RMSE is used as the performance metric, a positive validation bias suggests that the model validation procedure concludes an pessimistic estimation of the model performance, since the true performance is expected to be lower than the estimated performance. Another aspect of model validation is the variance of the estimated performance. For example, in a 5-fold cross validation, there are five estimates of the model performance. The variance among these five estimates is the validation variance. A high validation variance suggests that the performance is sensitive to the choice of the test set $\mathcal{D}_{\text{k}}$. The validation variance can be defined as:

$$
\text{Variance} = \mathbb{E}[(\hat{g}(f_{\mathcal{D}_{\text{-k}}}) - \mathbb{E}[\hat{g}(f_{\mathcal{D}})])^2]
$$


## Bias-Variance Trade-off

The choice of $K$ has a trade-off between the bias and the variance of estimating the true model generalization error.
A large $K$ will reduce the bias, as a larger size of training set represents more of the entire dataset. However, a large $K$ also increases the estimation variance as the test set is smaller in size and therefore the prediction error is more sensitive to the randomness of the test set. An extreme example of large $K$ is the leave-one-out cross validation (LOOCV) where $K$ is equal to the number of samples in the dataset. Since in each iteration, only one sample is evaluated for the prediction error. The error is sensitive to the choice of the sample and therefore has a high variance. LOOCV usually requires a large number of iterations to reduce the variance and achieve an unbiased estimation of the prediction errors. Hence, large $K$ or LOOCV is usually not recommended with small datasets. A detailed discussion of the trade-off has been discussed in (ref1, ref2).


In this section, a simulation study is conducted investigate how the interaction between the sample sizes and the number of folds affects the bias and variance of the cross validation procedure. An "In-Sample" metric is also reported to demonstrate an overoptimistic estimation when cross validation is not correctly implemented.

## Objectives

The goal of this study is to investigate the bias and variance of each performance estimators, which include "In-Sample" that validates the model on the same dataset used for training, K-fold cross validation where K is assigned as 2, 5, and 10, and LOOCV. The bias and variance of the performance estimation are calculated based on the following equations:

the predicted value of the model and $f(x)$ is the ground truth. The bias and variance are calculated for each sample in the dataset and then averaged across all samples. The bias and variance are calculated for each sample in the dataset and then averaged across all samples. The bias and variance are calculated for each sample in the dataset and then averaged across all samples. The bias and variance are calculated for each sample in the dataset and then averaged across all samples. The bias and variance are calculated for each sample in the dataset and then averaged across all samples. The bias and variance are calculated for each sample in the dataset and then averaged across all samples. The bias and variance are calculated for each sample in the dataset and then averaged across all samples. The bias and variance are calculated for each sample in the dataset and then averaged across all samples. The bias and variance are calculated for each sample in the dataset and then averaged across all samples.


