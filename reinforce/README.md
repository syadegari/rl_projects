# REINFORCE

### Introduction


### Getting Started


### Instructions

### Theory
We start by writing the gradient of the expected return in its general form. We then proceed by moving the gradient inside the sum to get:

$$
U(\theta) = \sum_{\tau}P(\tau;\theta)R(\tau)
$$
$$
\begin{align}
\nabla_{\theta} U(\theta) &= \sum_{\tau} \nabla_{\theta} P(\tau;\theta)R(\tau) \\\\
&=\sum_{\tau} \frac{P(\tau;\theta)}{P(\tau;\theta)} \nabla_{\theta} P(\tau;\theta)R(\tau) \\\\
&=\sum_{\tau} P(\tau;\theta) \nabla_{\theta} {\rm{log}}(P(\tau;\theta))R(\tau)
\end{align}
$$


The first part is very straightforward. This is also called **the likelihood ratio trock**. We then approximate the gardient using the sample based approach:

$$
\begin{align}
\nabla_{\theta} U(\theta) \approx \frac{1}{m} \sum_{i=1}^m \nabla_{\theta} {\rm log} \mathbb{P}(\tau^{(i)};\theta)R(\tau^{(i)})
\end{align}
$$

I am not sure why the uniform distribution assumption is used here. We assume all the sampled trajectories have the same probability of occuring ($\frac{1}{m}$ on the right hand side of the above expression). Next we will focus on the first term in the sum, $\nabla_{\theta}{\rm log}\mathbb{P}(\tau^{(i)}; \theta)$. We use the state transition dynamics of the MDP to get:

$$
\begin{align}
\nabla_{\theta}{\rm log}\mathbb{P}(\tau^{(i)}; \theta) &= \nabla_{\theta} {\rm log} \left[
    \prod_{t=0}^H \mathbb{P}(s_{t+1}^{(i)}|s_t^{(i)},a_t^{(i)}) \pi_{\theta}(a_t^{(i)}|s_t^{(i)})
\right]\\
\end{align}
$$

We note that the state transitions are not parametrized by the the parameters $\theta$ and so they won't make a contribution to our final result. Continuing the derivation, we get:

$$
\begin{align}
\nabla_{\theta}{\rm log}\mathbb{P}(\tau^{(i)}; \theta) &= \nabla_{\theta} {\rm log} \left[
    \sum_{t=0}^H \pi_{\theta}(a_t^{(i)}|s_t^{(i)})
\right]\\
&= \sum_{t=0}^H \nabla_{\theta} {\rm log} \pi_{\theta}(a_t^{(i)}|s_t^{(i)})
\end{align}
$$

Pluggin (7) into (4) we get the final expression for estimate of the gradient in REINFORCE method:

$$
\nabla_{\theta} U(\theta) \approx \frac{1}{m} \sum_{i=1}^m \sum_{t=0}^H \nabla_{\theta} {\rm log} \pi_{\theta}(a_t^{(i)}|s_t^{(i)})R(\tau^{(i)}).
$$
