# Nomenclature

- $\tau$: State-Action sequence $s_0, u_0, \ldots, s_H, u_H$ trajectory. 
- $R(\tau)$ Reward for trajectory $\tau$. $R(\tau)=\sum_{t=0}^HR(s_t, u_t)$.
- $U(\theta)$ Utility (objective). 
$$
U(\theta) = \mathbb E \left[ \sum_{t=0}^H R(s_t, u_t); \pi_\theta \right] = \sum_\tau P(\tau;\theta) R(\tau)
$$

- Goal: Find $\theta$ such that 
$$
\max_\theta U(\theta) = \max_\theta \sum_\tau P(\tau;\theta) R(\tau)
$$

# Policy Gradient
## Derivation
Taking the gradient with respect to $\theta$ gives

$$
\begin{align*}
\nabla_\theta U(\theta) 
&= \nabla_\theta \sum_\tau P(\tau; \theta) R(\tau) \\
&= \sum_{\tau} \nabla_\theta P(\tau; \theta) R(\tau) \\
&= \sum_{\tau} \frac{P(\tau; \theta)}{P(\tau; \theta)} \nabla_\theta P(\tau; \theta) R(\tau) \\
&= \sum_{\tau} P(\tau; \theta) \frac{\nabla_\theta P(\tau; \theta)}{P(\tau; \theta)} R(\tau) \\
&= \sum_{\tau} P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) R(\tau)
\end{align*}
$$

Approximate with empirical estimate for $m$ sample paths under policy $\pi_\theta$:

$$
\nabla_\theta U(\theta) \approx \hat{g} = \frac{1}{m} \sum_{i=1}^m \nabla_\theta \log P(\tau^{(i)}; \theta) R(\tau^{(i)})
$$

The above is valid even for discontinuous or unknown $R$ and when the sample space of paths is a discrete set.


The gradient tries to increase the probability of paths with "more favorable" rewards and decrease the probability of paths with negative rewards (It is more subtle than that since the probabilities have to sum to one).

Looking more closely into the term for trajectory probability, we can decompose the path into states and actions and show that the policy gradient estimator does not need a dynamics model:

$$
\begin{align*}
\nabla_\theta \log P(\tau^{(i)}; \theta)
&= \nabla_\theta \log \left[
\prod_{t=0}^{H} P\left(s_{t+1}^{(i)} \mid s_t^{(i)}, u_t^{(i)}\right)
\pi_\theta\left(u_t^{(i)} \mid s_t^{(i)}\right)
\right] \\
&= \nabla_\theta \left[
\sum_{t=0}^{H} \log P\left(s_{t+1}^{(i)} \mid s_t^{(i)}, u_t^{(i)}\right)
+ \sum_{t=0}^{H} \log \pi_\theta\left(u_t^{(i)} \mid s_t^{(i)}\right)
\right] \\
&= \sum_{t=0}^{H} \nabla_\theta \log \pi_\theta\left(u_t^{(i)} \mid s_t^{(i)}\right)
\end{align*}
$$

The estimator thus becomes:

$$
\hat{g} \approx \frac{1}{m} \sum_{i=1}^m \sum_{t=0}^{H} \nabla_\theta \log \pi_\theta\left(u_t^{(i)} \mid s_t^{(i)}\right) R(\tau^{(i)})
$$

Since this is an unbiased but noisy estimator (proof?), next some remedies are introduced. 

## Baseline Subtraction
Consider the following: In an environment with only positive rewards, even the poor trajectories receive positive reinforcement, which is inefficient at best, if not detrimental. This motivates the introduction of baseline:
$$
\nabla_\theta U(\theta) \approx \hat{g} = \frac{1}{m} \sum_{i=1}^m \nabla_\theta \log P(\tau^{(i)}; \theta) \left(R(\tau^{(i)}) - b\right)
$$

Does the new estimate hold, in a sense that it is still unbiased? It is easy to show that the expectation of the new term, $\nabla_\theta \log P(\tau; \theta) b$, is zero, given that $b$ is independent of the samples actions:

$$
\mathbb E \left[ \nabla_\theta \log P(\tau; \theta) b\right]
= \sum_\tau P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) b 
= \sum_\tau \nabla_\theta P(\tau; \theta) b 
= \nabla_\theta \sum_\tau P(\tau; \theta) b 
= b \cdot \nabla_\theta 1
= 0
$$

Even though this shows that on expectation the contribution of baseline has no effect, in the case of finite sampling, the new baseline leads to reduced variance and a better gradient estimate. 
When can we pull $b$ out of the sum and not having it depend on $\theta$? This is only allowed, as mentioned above, as long as $b$ is independent sampled action. To show that $b$ cannot depend on action, assume $b = b(u_t)$. Since $u_t \sim \pi_\theta(\cdot|s_t)$, then $b$'s distribution depend on $\theta$/policy and so it is no longer a constant with respect to gradient and trajectories generated. What are some good estimates for the baseline? 

- $b$ that depends on the state $s_t$, motivating the choice of $b=V(s_t)$.
- $b$ as an average of sampled minibatch of trajectories.
- Constant $b$

We will revisit these choices again in the next section.

## Temporal Decomposion

The current estimate can be written like the following:
$$
\begin{align*}

\hat{g} &= \frac{1}{m} \sum_{i=1}^m \nabla_\theta \log P(\tau^{(i)}; \theta) \left( R(\tau^{(i)}) - b \right) \\

&= \frac{1}{m} \sum_{i=1}^m \left( \sum_{t=0}^{H-1} \nabla_\theta \log \pi_\theta\!\left(u_t^{(i)} \mid s_t^{(i)}\right) \right) 
\left( \sum_{t=0}^{H-1} R\!\left(s_t^{(i)}, u_t^{(i)}\right) - b \right)  \\

&= \frac{1}{m} \sum_{i=1}^m \left( \sum_{t=0}^{H-1} \nabla_\theta \log \pi_\theta\!\left(u_t^{(i)} \mid s_t^{(i)}\right) 
\left[ \sum_{k=0}^{t-1} R\!\left(s_k^{(i)}, u_k^{(i)}\right) 
+ \sum_{k=t}^{H-1} R\!\left(s_k^{(i)}, u_k^{(i)}\right) - b \right] \right)
\end{align*}
$$

Note that the first term in the square bracket does not depend on current action $u_t^{(i)}$. This justifies removal of the first term from the gradient estimate, and thus lowering of the variance. 

$$
\hat{g} = \frac{1}{m} \sum_{i=1}^m \sum_{t=0}^{H-1} \nabla_\theta \log \pi_\theta\!\left(u_t^{(i)} \mid s_t^{(i)}\right) 
\left( 
\sum_{\color{blue} k=t}^{H-1} R\!\left(s_k^{(i)}, u_k^{(i)}\right) - b(s_t^{(i)}) \right)
$$


## Value Functions Estimation
## Advantage Estimation


# Policy Gradient with GAE or A3C
$$
\begin{align*}
\mathbb{E}_{\tau \sim P_\theta} \left[ R(\tau) \right]
&= \sum_{\tau} P_\theta(\tau) R(\tau) \\
&= \sum_{\tau} \frac{P_\theta(\tau)}{P_{\theta_{\text{old}}}(\tau)} P_{\theta_{\text{old}}}(\tau) R(\tau) \\
&= \mathbb{E}_{\tau \sim P_{\theta_{\text{old}}}} \left[ \frac{P_\theta(\tau)}{P_{\theta_{\text{old}}}(\tau)} R(\tau) \right]
\end{align*}
$$

