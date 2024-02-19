## Introduction (Jonny & Dani)
#### Goal 

- **Reason for the Presentation**: Explain why Deep Networks work well in practice at approximating low-rank functions. 
- **TakeAways:**  Introduce why this is important: 
	- Overfitting due to larger capacity: deeper networks can require far fewer parameters than shallower networks to obtain the same modeling "capacity". (p. 1). 
	- "The simplest solution is often the best solution". For networks trained on *natural data*, the goal is often to discover a low-rank relationship between the input and the label.  (p. 9).

Introduction & 

Deeper networks have *a greater proportion of parameter space* that maps the input data to lower-rank embeddings; hence, deeper networks are more likely to converge to functions that learn simpler embeddings (at initialisation and after training).

Deeper networks are implicitly biased to find lower effective rank embeddings because the volume of functions that map to lower effective rank embeddings increases with depth. 
#### Terminology 
- Overparameterisation
$$ \hat y = W_d W_{d-1} ... W_1 x = W_e x $$
- Continuous Rank Embeddings
- Gram matrices 
- Conventional, explicit regularisation (as opposed to implicit through deeper networks)

> Place the relevant terminology on the slide it is used in. 

#### Related Work
- Balancing Overfitting with Generalisability 

## Contributions (Ugo)
Introduce the following two headings.

#### Parameterisation Bias of Depth 
Observations in section 3. 

- **Theoretical Results** Over-parameterisation, in theory, acts as regularisation

#### Over-Parameterisation as Implicit Regularisation (Aral)

> Include results on origianl and over-parameterised CV models. 

A technique used to prevent overfitting (by spreading decisions/reliance over multiple weights instead of focusing on one). This can be done explicitly (like $l_1, l_2$norm-based and commonly-used pseudo-measures of rank) and implicitly. 

- **Empirical Results** Over-parameterisation acts as implicit regularisation and improves generalisation ability. 

Over-paraeterisation receives a combined effect of both gradient descent's implicit bias and model parameterisation's inductive bias.

## Research that builds upon this paper and competing methods (Kasper)
E.g. https://arxiv.org/abs/1805.08522 (2018)

## SWOT (TODO: Discuss)
Strengths, Weaknesses, Improvements

- RMT correlates well with empricial observations. However, it considers matrices of infinite dimension, so future work could attempt to properly apply RMT to finite matrices. 
- 

#### Discussion Points 

#### Formal Conclusion