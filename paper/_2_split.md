# Cross Validation

Cross validation is the most common method for evaluating an expected model performance. This approach examines how the model performs on an unseen dataset. The implementation of cross validation is to split the dataset into $K$ equal sized subsets. One subset is holded as the test set which serves as the unseen data. The remaining subsets are used as the training set to estimate the model parameters. The prediction errors are then calculated on the test set. This process will repeat $K$ times, each time holding out a different subset as the test set. A expected prediction error in a cross validation procedure is given by

$$
\text{E(CV)} = \frac{1}{K}\sum_{k=1}^K \mathcal{L}\left(Y_k, \hat{f}^{-k}(X_k)\right)
$$

where $L$ measures the prediction errors between two variables. $Y_k$ and $X_k$ are the ground truths and the input features in the hold-out test set, respectively. $\hat{f}^{-k}(X_k)$ is the predicted value of the model trained on the training set without the $k$-th subset. The choice of $K$ has a trade-off between the bias and the variance of estimating the true model generalization error.
A large $K$ will reduce the bias, as a larger size of training set represents more of the entire dataset. However, a large $K$ also increases the estimation variance as the test set is smaller in size and therefore the prediction error is more sensitive to the randomness of the test set. An extreme example of large $K$ is the leave-one-out cross validation (LOOCV) where $K$ is equal to the number of samples in the dataset. Since in each iteration, only one sample is evaluated for the prediction error. The error is sensitive to the choice of the sample and therefore has a high variance. LOOCV usually requires a large number of iterations to reduce the variance and achieve an unbiased estimation of the prediction errors. Hence, large $K$ or LOOCV is usually not recommended with small datasets. A detailed discussion of the trade-off has been discussed in (ref1, ref2).

In this section, a simulation study is conducted to demonstrate the bias-variance trade-off of cross validation. We will also show the overoptimistics 