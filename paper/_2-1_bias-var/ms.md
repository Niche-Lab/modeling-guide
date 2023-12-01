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

Leave-one-out cross validation (LOOCV) is a variant of K-fold CV where $K$ equals the sample size $\mathcal{N}$ of the complete dataset $\mathcal{D}$. It provides an unbiased estimation of model performance because the training set $\mathcal{D}_{\text{-k}}$ closely resembles the unseen population of interest, given its size of $\mathcal{N} - 1$. However, as the trade-off discussion suggested, this method can lead to high validation vairance due to the evaluation of the model on only one sample at a time. The true unbiased nature of LOOCV is fully realized only when all K folds are utilized. Performing an incomplete LOOCV can introduce significant bias because of the inherent high validation variance, which often occurs when training each model iteration is prohibitively time-consuming or computationally demanding. In certain contexts, such as genomic prediction, strategies like the one described by Cheng et al. leverage the matrix inverse lemma, which allows for computational savings by avoiding the inversion of large matrices in each fold. This technique significantly reduces the dependency of computational resources on sample size (Cheng et al., 2017). Van Dixhoorn et al. exemplify the use of LOOCV with a small dataset, aiming to predict cow resilience (van Dixhoorn et al., 2018). Nevertheless, for large datasets, LOOCV is generally not recommended due to computational inefficiency. The bias-variance trade-off associated with LOOCV has been extensively explored in the statistical literature (Hastie et al., 2009; Cowley and Talbot, 2010).

## Simulation Objectives and Hypothesis

A simulation study is conducted to examine the interaction between sample sizes and various performance estimators, as well as how this interaction influences the bias and variance in model validation. It is hypothesized that both bias and variance will diminish as the sample size grows. Furthermore, it is anticipated that the variance will escalate with an increase in the number of folds used by the estimator, although this will concurrently decrease bias. Given that K-fold cross-validation (CV) utilizes only a portion (i.e., \( K - 1 \) folds) of data points for training, it is considered a pessimistic estimate of model performance. The study also aims to quantify the extent of performance underestimation for each CV estimator.

## Simulation Design

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

Figure 5. Simulation results of validation bias from 1000 sampling iterations. Multiple performance estimators across different sample sizes were color-coded. Three metrics: r, R^2, and RMSE, were displayed in the column facets.

Figure 6. Simulation results of validation bias and variance from 1000 sampling iterations. Multiple performance estimators across different sample sizes were color-coded. Only RMSE was displayed. Bias and variance were listed in the left and right facets, respectively.

The simulation results, depicted in box plots (Figure 5, 6), explored the validation bias and variance distribution. Figure 5 examines the bias alterations across various estimators and sample sizes. Independent of the estimator and metric, the bias diminishes with increasing sample sizes. The in-sample estimator consistently overestimates across all metrics and sample sizes, underscoring the necessity of cross-validation (CV) for unbiased performance evaluation.
In CV estimators, although LOOCV is traditionally viewed as unbiased, it shows underestimation in model performance, especially when the metric is correlation coefficient (r). Comparatively, 2-, 5-, and 10-fold CV provide a more unbiased estimation than LOOCV for all sample sizes. However, for metrics like $R^2$ or RMSE, LOOCV emerges as the least biased estimator. While K-fold CV exhibits higher bias than LOOCV, this difference dwindles when the sample size exceeds 500. Notably, 10-fold CV, contrary to expectations, demonstrates higher bias than 5-fold CV for small sample sizes (50 and 100) in the $R^2$ metric, though this disparity also becomes insignificant at larger sample sizes (N = 500).



Considering that there is only one data point tested in LOOCV, the validation variance is only applicable to the metric RMSE, which allows conducting an evaluation on a single data point. Figure 6 inllustrates the bias and variance in the RMSE across different performance estimators as a function of sample size $\text{N}$. Both the bias and variance in RMSE are observed to decrease as the sample size increases, which meet the hypothesis. The LOOCV is found to have the least biased estimation among all the estimators. Although 2-fold CV shows the highest bias, however, the bias did not show a significant decrease when the sample size increases. And all estimator shows similar bias when the sample size reaches 500. Regarding validation variance, LOOCV exhibits a consistently higher value as compared to other estimators across all sample sizes. Furthermore, it is observed that a lower number of folds $K$ correlates with reduced variance, which is also consistent with the hypothesis.

Considering LOOCV’s singular data point testing, its validation variance is pertinent only for RMSE, which permits single data point evaluations. Figure 6 illustrates the bias and variance in RMSE across different performance estimators as a function of sample size (N). Both bias and variance in RMSE decrease as sample size increases, aligning with the hypothesis. LOOCV provides the least biased estimation, while 2-fold CV exhibits the highest bias without significant reduction at larger sample sizes. However, biases across all estimators converge at a sample size of 500. In terms of validation variance, LOOCV consistently shows higher values than other estimators for all sample sizes. Additionally, a lower number of folds (K) correlates with reduced variance, which is also in line with the hypothesized trend.

### Conclusion

In conclusion, when conducting model validation, it is crucial to consider the estimator and sample size, as they significantly influence bias and variance. Larger sample sizes generally lead to reduced bias and variance, enhancing the reliability of the validation process. For unbiased performance estimation, CV methods, such as K-fold CV and LOOCV, are preferable than in-sample estimation. LOOCV often provides less biased estimations for certain metrics but can exhibit higher variance. It is also noteworthy that the number of folds in K-fold CV can affect bias and variance; thus, experimenting with different numbers of folds, especially in smaller sample sizes, can be beneficial. Ultimately, the selection of appropriate validation techniques should be tailored to the specific context of the dataset and the objectives of the modeling exercise, ensuring a robust and reliable assessment of model performance.
