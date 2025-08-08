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

## Temporal Decomposion
## Baseline Subtraction
Consider the following: In an environment with only positive rewards, even the poor trajectories receive positive reinforcement, which is inefficient at best, if not detrimental. This motivates the introduction of baseline. 
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

