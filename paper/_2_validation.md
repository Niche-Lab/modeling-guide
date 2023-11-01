# Model Validation

One of the most prevalent methods for assessing a model's performance is K-fold Cross-Validation (K-fold CV). This technique aims to evaluate how well a given model generalizes to an independent dataset that it has not seen during training. To implement K-fold CV, the available dataset, denoted as \( \mathcal{D} \), is partitioned into \( K \) equally-sized folds. Mathematically, we can represent the dataset as:

\[
\mathcal{D} = \{(X, Y)\} = \{(X_1, Y_1), (X_2, Y_2), \dots, (X_K, Y_K)\}
\]

Here, \( X \in \mathbb{R}^{n \times p} \) represents the input features, and \( Y \in \mathbb{R}^{n \times 1} \) symbolizes the labels. \( n \) is the total number of samples, while \( p \) indicates the number of features. In each iteration of the K-fold CV, one fold is reserved as the test set, \( \mathcal{D}_{\text{test}} \), to act as unseen data, while the remaining folds make up the training set \( \mathcal{D}_{\text{train}} \). More formally:

\[
\begin{align*}
\mathcal{D}_{\text{train}} &= \mathcal{D}_{-k} = \{(X_1, Y_1), (X_2, Y_2), \dots, (X_{k-1}, Y_{k-1}), (X_{k+1}, Y_{k+1}), \dots, (X_K, Y_K)\}\\
\mathcal{D}_{\text{test}} &= \mathcal{D}_{k} = \{(X_k, Y_k)\}
\end{align*}
\]

Once the model is trained on \( \mathcal{D}_{\text{train}} \), its performance is evaluated on \( \mathcal{D}_{\text{test}} \) by calculating the prediction errors. This procedure is repeated \( K \) times, with a different fold serving as the test set each time.The generalization error of the model $g(f_{\mathcal{D}})$ using the complete dataset $\mathcal{D}$ is computed as:

\[
\mathcal{g}(f_{\mathcal{D}}) = \frac{1}{K}\sum_{k=1}^{K} \mathcal{L}\left(Y_k, f_{\mathcal{D}_{-k}}(X_k)\right)
\]

In this equation, \( \mathcal{L} \) measures the prediction errors between the ground truth \( Y_k \) and the predicted values \( f_{\mathcal{D}_{-k}}(X_k) \). Here, \( f_{\mathcal{D}_{-k}} \) is the model trained on the dataset that excludes the \( k^{\text{th}} \) fold \( \mathcal{D}_{k} \). Finally, we average these \( K \) prediction errors to yield an estimate of the model's prediction error $g(f_{\mathcal{D}})$, thereby providing an assessment of its performance.


## Bias and Variance

The ideal estimation 

The choice of $K$ has a trade-off between the bias and the variance of estimating the true model generalization error.
A large $K$ will reduce the bias, as a larger size of training set represents more of the entire dataset. However, a large $K$ also increases the estimation variance as the test set is smaller in size and therefore the prediction error is more sensitive to the randomness of the test set. An extreme example of large $K$ is the leave-one-out cross validation (LOOCV) where $K$ is equal to the number of samples in the dataset. Since in each iteration, only one sample is evaluated for the prediction error. The error is sensitive to the choice of the sample and therefore has a high variance. LOOCV usually requires a large number of iterations to reduce the variance and achieve an unbiased estimation of the prediction errors. Hence, large $K$ or LOOCV is usually not recommended with small datasets. A detailed discussion of the trade-off has been discussed in (ref1, ref2).


$$
\begin{align}
Bias &= \mathbb{E}\left[\hat{f}(x) - f(x)\right] \\
Var &= \mathbb{E}\left[\left(\hat{f}(x) - \mathbb{E}\left[\hat{f}(x)\right]\right)^2\right]
\end{align}
$$