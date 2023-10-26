# Introduction

In agricultural research, modeling serves as an indispensable tool for hypothesis formulation and decision-making. It functions as a structured framework that validates system understanding through the analysis of empirical data, and further extends this understanding by enabling the extrapolation of results to novel trials and conditions. The advancement of research is fundamentally built upon the accumulation of prior knowledge within the scientific community. Therefore, the evaluation of model performance becomes particularly critical, necessitating a rigorous and standardized approach that allows for both reproducibility and comparability. The failure to adhere to these standards, by reporting model performance through ill-defined metrics or non-rigorous procedures, has the potential to introduce misinterpretations and miscommunications. Such lapses not only impede scientific progress but can also compromise the integrity of the collective body of research in the field.

This review aims to scrutinize four common pitfalls in model evaluation. The first pitfall is the failure to validate model performance on unseen or test datasets; the second involves indiscriminate feature selection across the full dataset; the third emerges when tuning hyperparameters with the same data designated for model evaluation; the fourth is neglecting block effects in model validation. In each section, current relevant practices in dairy science are examined to offer practical examples for clearer illustration. To underscore the importance of these pitfalls, a minimally reproducible simulation is presented to demonstrate their impact on model evaluation.