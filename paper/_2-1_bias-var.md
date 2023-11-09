# Validation Bias and Variance

## Definition

The true generalization performance of the model $G(f_{\mathcal{D}})$ can only be approximated by averaging the performance metrics over infinite unseen datasets. However, in practice, the dataset $\mathcal{D}$ is finite and therefore there is always a bias in the estimation of $G(f_{\mathcal{D}})$. The validation bias can be defined as:

$$
\text{Bias} = \mathbb{E}[\hat{g}(f_{\mathcal{D}})] - G(f_{\mathcal{D}})
$$

For example, if RMSE is used as the performance metric, a positive validation bias suggests that the model validation procedure concludes an pessimistic estimation of the model performance, since the true performance is expected to be lower than the estimated performance. Another aspect of model validation is the variance of the estimated performance. For example, in a 5-fold cross validation, there are five estimates of the model performance. The variance among these five estimates is the validation variance. A high validation variance suggests that the performance is sensitive to the choice of the test set $\mathcal{D}_{\text{k}}$, which may be caused by small sample size or over-complex model. The validation variance can be defined as:

$$
\begin{align*}
\text{Variance}
    &= \mathbb{E}[(\hat{g}(f_{\mathcal{D}_{\text{-k}}}) - \mathbb{E}[\hat{g}(f_{\mathcal{D}})])^{2}]\\
    &= \mathbb{E}[\hat{g}^{2}(f_{\mathcal{D}_\text{-k}}) - 2\hat{g}(f_{\mathcal{D}_{\text{-k}}})\mathbb{E}[\hat{g}(f_{\mathcal{D}})] + \mathbb{E}^{2}[\hat{g}(f_{\mathcal{D}})]]\\
    &= \mathbb{E}[\hat{g}^{2}(f_{\mathcal{D}_\text{-k}})] - 2\mathbb{E}[\hat{g}(f_{\mathcal{D}_{\text{-k}}})]\mathbb{E}[\hat{g}(f_{\mathcal{D}})] + \mathbb{E}^2[\hat{g}(f_{\mathcal{D}})]\\
    &= \mathbb{E}[\hat{g}^{2}(f_{\mathcal{D}_\text{-k}})] - \mathbb{E}^{2}[\hat{g}(f_{\mathcal{D}})]
\end{align*}
$$

It is noted that $\mathbb{E}[\hat{g}(f_{\mathcal{D}_\text{-k}})]$ is equivalent to $\mathbb{E}[\hat{g}(f_{\mathcal{D}})]$ in K-fold CV, since $\mathbb{E}[\hat{g}(f_{\mathcal{D}})]$ is estimated by averaging all $\hat{g}(f_{\mathcal{D}_\text{-k}})$ over $K$ folds, which is the definition of $\mathbb{E}[\hat{g}(f_{\mathcal{D}_\text{-k}})]$. Combining the bias and variance, the mean squared error (MSE) of the model validation can be defined as:

$$
\begin{align*}
\text{Validation MSE}
                    &= \mathbb{E}[(\hat{g}(f_{\mathcal{D}_{\text{-k}}}) - G(f_{\mathcal{D}}))^2]\\
                    &= \mathbb{E}[\hat{g}^2(f_{\mathcal{D_{-k}}})] - 2\mathbb{E}[\hat{g}(f_{\mathcal{D_{-k}}})]G(f_{\mathcal{D}}) + G^2(f_{\mathcal{D}}) + \mathbb{E}^2[\hat{g}(f_{\mathcal{D_{-k}}})] - \mathbb{E}^2[\hat{g}(f_{\mathcal{D_{-k}}})]\\
                    &= (\mathbb{E}^{2}[\hat{g}(f_{\mathcal{D_{-k}}})] - 2\mathbb{E}[\hat{g}(f_{\mathcal{D_{-k}}})]G(f_{\mathcal{D}}) + G^2(f_{\mathcal{D}})) + (\mathbb{E}[\hat{g}^2(f_{\mathcal{D_{-k}}})] - \mathbb{E}^2[\hat{g}(f_{\mathcal{D_{-k}}})])\\
                    &= (\mathbb{E}[\hat{g}(f_{\mathcal{D}})] - G(f_{\mathcal{D}}))^2 + (\mathbb{E}[\hat{g}^2(f_{\mathcal{D_{-k}}})] - \mathbb{E}^2[\hat{g}(f_{\mathcal{D}})])\\
                    &= \text{Bias}^2 + \text{Variance}
\end{align*}
$$


## Bias-Variance Trade-off

From the equpation above, a trade-off relationship between the bias and variance can be observed given a constant validation MSE. With the fixed sample size and model complexity in K-fold CV, the choice of $K$ is the major factor that affects the bias and variance of the model validation. When the $K$ is set larger, each training set $\mathcal{D}_{\text{-k}}$ is larger in size, whcih means the model is trained on a dataset that is more representative of the population of interest, leading to lower bias. However, because the test set $\mathcal{D}_{\text{k}}$ is relatively small in size, the validation variance can be high due to the high sensitivity to the specific data points in the test set $\mathcal{D}_{\text{k}}$. On the other hand, with fewer folds when $K$ is set smaller, each training set $\mathcal{D}_{\text{-k}}$ is smaller, leading to worse representation of the population and higher bias. However, the test set $\mathcal{D}_{\text{k}}$ is larger in size, in which the estimate from each fold is more stable and therefore the validation variance is lower.

Leave-one-out cross validation (LOOCV) is a special case of K-fold CV where $K$ is equal to sample size $\mathcal{N}$ in the complete dataset $\mathcal{D}$. It is known for an unbiased estimation of the model performance, since the training set $\mathcal{D}_{\text{-k}}$ has good representative sample size $\mathcal{N} - 1$. However, as the trade-off discussion suggested, the validation variance can be also high as there is only one sample being tested in each fold. It is worth noting that the unbiased estimation in LOOCV can only be achieved when all $K$ folds are tested. If an incomplete LOOCV is conducted, the validation bias can also be high due to its nature of high validation variance. Therefore, LOOCV should be avoided when the dataset is large and the training process is costly in time or computational resources. A detailed discussion of the trade-off has been discussed in (ref1, ref2).

## Objectives and Hypothesis

A simulation study is conducted to examine the interaction between sample sizes and various performance estimators, as well as how this interaction influences the bias and variance in model validation. It is hypothesized that both bias and variance will diminish as the sample size grows. Furthermore, it is anticipated that the variance will escalate with an increase in the number of folds used by the estimator, although this will concurrently decrease bias. Given that K-fold cross-validation (CV) utilizes only a portion (i.e., \( K - 1 \) folds) of data points for training, it is considered a pessimistic estimate of model performance. The study also aims to quantify the extent of performance underestimation for each CV estimator.

## Simulation Design and Procedure

The studied performance estimators include K-fold cross validation where K is assigned as 2, 5, and 10, and LOOCV, which is a special case of K-fold CV where K is equal to the sample size, and "In-Sample" that validates the model on the same dataset used for training. The "In-Sample" metric is presented to demonstrate an overoptimistic estimation of the model performance without conducting model validation. Three performance metrics are used to evaluate the model performance, including the RMSE (eq x), $R^2$ (eq y), and $r$ (eq z). The validating model in this simulation is multivariate linear regression which takes ten features as the input regressors and one target variable as the output. Both the input regressors and target variable are generated from a normal distribution with mean 0 and standard deviation 1. Hence there should have no linear association being observed from the data. The sample size $\text{n}$ is set as 50, 100, and 500 to observe the interaction between the sample size and the performance estimators. The simulation was iterated 1000 times for each setting (i.e., sample size and estimator) to observe the distribution of bias and variance.

In each iteration of the simulation, the dataset $\mathcal{D}=\{X, Y\}$ is sampled accordingly to the simulation assumption as described. If the estimator is a K-fold CV, the dataset $\mathcal{D}$ is partitioned into $K$ folds in which each fold is $\mathcal{D_k}=\{X_k, Y_k\}$. Otherwise, the dataset $\mathcal{D}$ is not partitioned in the "In-Sample" estimator. The linear model $f$ is trained on the training set $\mathcal{D}_{\text{-k}}$ to estimate the coefficients $\beta$, which is then used to predict the target variable $\hat{Y_k}$ in the test set $\mathcal{D}_{\text{k}}$. The procedure of K-fold CV can be expressed as:

$$
\begin{align*}
Y_{\text{-k}} &= f_{\mathcal{D_{-k}}}(X_{\text{-k}}) + \epsilon = X_{\text{-k}}\beta + \epsilon, \quad k = 1, 2, \dots, K\\
\hat{Y}_{\text{k}} &= f_{\mathcal{D_{-k}}}(X_{\text{k}}) = X_{\text{k}}\beta
\end{align*}
$$

Since there is no split in the "In-Sample" estimator, the prediction of the target variable $\hat{Y}$ in "In-Sample" is obtained as:

$$
\begin{align*}
Y &= f_{\mathcal{D}}(X) = X\beta + \epsilon\\
\hat{Y} &= f_{\mathcal{D}}(X) = X\beta
\end{align*}
$$

where:

- $X$ is the input regressors sampled from a standard normal distribution $\mathcal{N}(0, 1)$ with dimensions $\text{n} \times 10$.
- $Y$ is the target variable sampled from a standard normal distribution $\mathcal{N}(0, 1)$ and belongs to $\mathbb{R}^{\text{n} \times 1}$.
- $X_{\text{-k}}$ and $Y_{\text{-k}}$ are the input regressors and target variable in the training set $\mathcal{D}_{\text{-k}}$.
- $X_{\text{k}}$ is the input regressors in the test set $\mathcal{D}_{\text{k}}$.
- $\hat{Y}_{\text{k}}$ is the predicted target variable in the test set $\mathcal{D}_{\text{k}}$.
- $\beta$ is the estimated regression coefficients and belongs to $\mathbb{R}^{10 \times 1}$.
- $\epsilon$ is the error term assumed to be normally distributed with mean 0 and standard deviation 1.

After obtaining the estimated target variable $\hat{Y_k}$ or $\hat{Y}$, the estimated performance $\mathbb{E}[\hat{g}(f_{\mathcal{D}})]$ can be derived as described in the previous section. To simulate the true model performance $G(f_{\mathcal{D}})$, one-hundread of unseen datasets $\mathcal{D}^{*}$ are sampled in the same manner as the dataset $\mathcal{D}$, and the model performance $G(f_{\mathcal{D}})$ is approximated by averaging the performance metrics over all $\mathcal{D}^{*}$. The bias and variance of the model validation are then calculated as eq x and eq y, respectively.

## Results

The 100-iteration simulation results were summarized in the box plots to examine the validation bias and variance distribution. The figure 1 focuses on examining the bias changes across different estimators and sample sizes. Regardless of estimator and metric, the bias decreases as the sample size escalated. Although LOOCV has been considered as the 


When the metrics are $R^2$ or RMSE, LOOCV 





The to observe the trade-off relatinoship

Considering that there is only one data point tested in LOOCV, the validation variance is only applicable to the metric RMSE. Figure 1.1 inllustrates the bias and variance in the RMSE across different performanc estimators as a function of sample size $\text{N}$. Both the bias and variance in RMSE are observed to decrease as the sample size increases, which meet the hypothesis. The LOOCV is found to have the least biased estimation among all the estimators. Although 2-fold CV shows the highest bias, however, the bias did not show a significant decrease when the sample size increases. And all estimator shows similar bias when the sample size reaches 500. Rregarding validation variance, LOOCV exhibits a consistently higher value as compared to other estimators across all sample sizes. Furthermore, it is observed that a lower number of folds $K$ correlates with reduced variance, which is also consistent with the hypothesis.


When including other metrics suchs as $r$ and $R^2$ (figure 1.2), and In-Sample approach into the discussion, the 


## Conclusion