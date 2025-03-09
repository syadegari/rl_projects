# A Survey of DQN Methods

This is a list of literature review and some explanations of various methods from DQN method. It starts with basic methods for DQN, as explained in Deepmind paper:
- Two separate networks for argmax value (called target network) and one for learning called local network
- Soft update of the network $\theta_{\rm target} \leftarrow \tau \theta_{\rm target} + (1 - \tau) \theta_{\rm local}$ for some $0 \lt \tau \ll 1$.
- Priotorized experience replay buffer


$$
\large
\begin{align*}
Q_{\rm{local}}(s, a; \theta) \leftarrow Q_{\rm{local}}(s, a; \theta) 
+ \alpha \big[ Q_{\rm{target}}(s, a; \theta) - Q_{\rm{local}}(s, a; \theta) \big],
\end{align*}

\\

\begin{align*}
Q_{\rm{target}}(s, a; \theta) &= r + \gamma \max_{a'} Q_{\rm{local}}(s', a'; \theta),
\end{align*}
$$

The loss and gradient are calculated as follows:


$$
l = \left\Vert \left( Q_{\rm local}  - Q_{\rm target} \right)^2 \right\Vert \\
l = \nabla_{\theta_{\rm local}}\left\Vert \left( Q_{\rm local}  - Q_{\rm target} \right)^2 \right\Vert \\
g = \nabla_{\theta_{\rm local}} l
$$
We then gradually build our way to a rainbow DQN method. 

## Double Q-Learning
## Dueling Networks
## Multistep Learning
## Noisy Networks
## Distributional RL (C51)


let's try to write some equations here

$$
\huge
\begin{align*}
3x^2 \in R \subset Q \\
\mathnormal{3x^2 \in R \subset Q} \\
\mathrm{3x^2 \in R \subset Q} \\
\mathit{3x^2 \in R \subset Q} \\
\mathbf{3x^2 \in R \subset Q} \\
\mathsf{3x^2 \in R \subset Q} \\
\mathtt{3x^2 \in R \subset Q}
\end{align*}
$$
