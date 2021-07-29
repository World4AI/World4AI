================
Policy Iteration
================

The policy iteration algorithm is an iterative method. Iterative methods start with initial (usually random or 0) values as approximations and improve the subsequent approximations with each iteration using the previous approximations as input. The policy iteration algorithm consists of two steps. The policy evaluation step calculates the value function for a given policy. The policy improvement step improves the given policy. Both steps run after each other to form the policy iteration algorithm. 

Policy Evaluation
=================

.. math::
    :nowrap:

    \begin{align*}
    v_{\pi}(s) & \doteq \mathbb{E}_{\pi}[G_t \mid S_t = s] \\
    & = \mathbb{E}_{\pi}[R_{t+1} + \gamma G_{t+1} \mid S_t = s] \\
    & = \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s] \\
    & = \sum_{a}\pi(a \mid s)\sum_{s', a}p(s', a \mid s, a)[r + \gamma v_{\pi}(s')]
    \end{align*}


.. math::
    v_{k+1}(s) \doteq \sum_{a}\pi(a \mid s)\sum_{s', a}p(s', a \mid s, a)[r + \gamma v_{\pi}(s')]


.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{Iterative Policy Evaluation}
        \label{alg1}
    \begin{algorithmic}
        \STATE Input: policy $\pi$, stop criterion $\theta > 0$, discount factor $\gamma$
        \STATE Initialize: $V(s)$ and $V_{old}(s)$, for all $s \in \mathcal{S}$ with zeros
        \REPEAT
            \STATE $\Delta \leftarrow 0$
            \STATE $V_{old}(s) = V(s)$ for all $s \in \mathcal{S}$
            \FORALL{$s \in \mathcal{S}$}
                \STATE $V(s) \leftarrow \sum_{a}\pi(a \mid s)\sum_{s', a}p(s', a \mid s, a)[r + \gamma V_{old}(s')]$
                \STATE $\Delta \leftarrow \max(\Delta,|V_{old}(s) - V(s)|)$
            \ENDFOR
        \UNTIL{$\Delta < \theta$}
    \end{algorithmic}
    \end{algorithm}




Policy Improvement
==================


