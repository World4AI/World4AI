import{S as se,i as re,s as pe,l as y,a as $,r as u,w as L,T as me,m as g,h as t,c as _,n as w,u as c,x as M,p as Z,G as d,b as s,y as W,f as D,t as R,B as G,E as ee}from"../../../../chunks/index-caa95cd4.js";import{C as ue}from"../../../../chunks/Container-5c6b7f6d.js";import{L as F}from"../../../../chunks/Latex-bf74aeea.js";function ce(f){let l;return{c(){l=u("v_*(s)")},l(i){l=c(i,"v_*(s)")},m(i,a){s(i,l,a)},d(i){i&&t(l)}}}function fe(f){let l=String.raw`
\begin{aligned}
  v_*(s) & = \max_a q_*(s, a) \\
  & = \max_a \mathbb{E}_{\pi}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a] \\ 
  & = \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_*(s')]
\end{aligned}
`+"",i;return{c(){i=u(l)},l(a){i=c(a,l)},m(a,r){s(a,i,r)},p:ee,d(a){a&&t(i)}}}function he(f){let l=String.raw`
      v_{k+1}(s) \doteq \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_k (s')]
  `+"",i;return{c(){i=u(l)},l(a){i=c(a,l)},m(a,r){s(a,i,r)},p:ee,d(a){a&&t(i)}}}function ve(f){let l=String.raw`
  \begin{aligned}
    \text{(1: Policy Evaluation) } & q_{k+1}(s, a) = \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_k (s')] \\
    \text{(2: Policy Improvement) }& v_{k+1}(s) = \max_a q_{k+1}(s, a)
  \end{aligned}
`+"",i;return{c(){i=u(l)},l(a){i=c(a,l)},m(a,r){s(a,i,r)},p:ee,d(a){a&&t(i)}}}function $e(f){let l=String.raw`\theta`+"",i;return{c(){i=u(l)},l(a){i=c(a,l)},m(a,r){s(a,i,r)},p:ee,d(a){a&&t(i)}}}function _e(f){let l,i,a,r,x,m,T,h,v,o,p,b,J,k,K,S,te,N,E,O,B,ae,Q,I,ne,P,ie,U,V,le,X,C,Y;return m=new F({props:{$$slots:{default:[ce]},$$scope:{ctx:f}}}),v=new F({props:{$$slots:{default:[fe]},$$scope:{ctx:f}}}),k=new F({props:{$$slots:{default:[he]},$$scope:{ctx:f}}}),E=new F({props:{$$slots:{default:[ve]},$$scope:{ctx:f}}}),P=new F({props:{$$slots:{default:[$e]},$$scope:{ctx:f}}}),{c(){l=y("p"),i=u(`When we consider policy iteration again, we should remember that there are
    two distinct steps, policy evaluation and policy improvement. The policy
    improvement step is a single step, where the new policy is derived by acting
    greedily. The policy evaluation on the other hand is a longer iterative
    process. It turns out that it is not necessary to wait for the policy
    evaluation algorithm to finish converging to the true value function. In
    fact the value iteration algorithm works with only one single policy
    evaluation step.`),a=$(),r=y("p"),x=u("The main goal of value iteration is to find the optimal value function "),L(m.$$.fragment),T=u(`, that can be used to derive the optimal policy. The optimal value function
    can be expressed as a Bellman equation that looks as follows.`),h=$(),L(v.$$.fragment),o=$(),p=y("p"),b=u(`The value iteration is essentially the Bellman optimality equation, that has
    been transformed to an iterative algorithm.`),J=$(),L(k.$$.fragment),K=$(),S=y("p"),te=u(`Although the update step looks like a single step at first glance, it
    actually combines truncated (one step) policy evaluation and policy
    improvement in a single step.`),N=$(),L(E.$$.fragment),O=$(),B=y("p"),ae=u(`In the first step the action-value function is calculated based on the old
    state-value function and the model of the Markov decision process. In the
    second step a max over the action-value function is taken in order to
    generate the new state-value function. That implicitly generates a new
    policy as a value function is always calculated for a particular policy.`),Q=$(),I=y("p"),ne=u(`The combination of both steps is the value iteration algorithm. The
    iterative process continues until the difference between the old and the new
    state-value function is smaller than some parameter theta `),L(P.$$.fragment),ie=u(`. As the final step the optimal policy can be deduced by always selecting
    the greedy action.`),U=$(),V=y("p"),le=u(`Below is the Python implementation of the value iteration algorithm.
    Compared to policy iteration the implementation is more compact, because
    policy evaluation and policy improvement can be implemented in a single
    function.`),X=$(),C=y("div"),this.h()},l(e){l=g(e,"P",{});var n=w(l);i=c(n,`When we consider policy iteration again, we should remember that there are
    two distinct steps, policy evaluation and policy improvement. The policy
    improvement step is a single step, where the new policy is derived by acting
    greedily. The policy evaluation on the other hand is a longer iterative
    process. It turns out that it is not necessary to wait for the policy
    evaluation algorithm to finish converging to the true value function. In
    fact the value iteration algorithm works with only one single policy
    evaluation step.`),n.forEach(t),a=_(e),r=g(e,"P",{});var q=w(r);x=c(q,"The main goal of value iteration is to find the optimal value function "),M(m.$$.fragment,q),T=c(q,`, that can be used to derive the optimal policy. The optimal value function
    can be expressed as a Bellman equation that looks as follows.`),q.forEach(t),h=_(e),M(v.$$.fragment,e),o=_(e),p=g(e,"P",{});var H=w(p);b=c(H,`The value iteration is essentially the Bellman optimality equation, that has
    been transformed to an iterative algorithm.`),H.forEach(t),J=_(e),M(k.$$.fragment,e),K=_(e),S=g(e,"P",{});var j=w(S);te=c(j,`Although the update step looks like a single step at first glance, it
    actually combines truncated (one step) policy evaluation and policy
    improvement in a single step.`),j.forEach(t),N=_(e),M(E.$$.fragment,e),O=_(e),B=g(e,"P",{});var z=w(B);ae=c(z,`In the first step the action-value function is calculated based on the old
    state-value function and the model of the Markov decision process. In the
    second step a max over the action-value function is taken in order to
    generate the new state-value function. That implicitly generates a new
    policy as a value function is always calculated for a particular policy.`),z.forEach(t),Q=_(e),I=g(e,"P",{});var A=w(I);ne=c(A,`The combination of both steps is the value iteration algorithm. The
    iterative process continues until the difference between the old and the new
    state-value function is smaller than some parameter theta `),M(P.$$.fragment,A),ie=c(A,`. As the final step the optimal policy can be deduced by always selecting
    the greedy action.`),A.forEach(t),U=_(e),V=g(e,"P",{});var oe=w(V);le=c(oe,`Below is the Python implementation of the value iteration algorithm.
    Compared to policy iteration the implementation is more compact, because
    policy evaluation and policy improvement can be implemented in a single
    function.`),oe.forEach(t),X=_(e),C=g(e,"DIV",{class:!0}),w(C).forEach(t),this.h()},h(){Z(C,"class","separator")},m(e,n){s(e,l,n),d(l,i),s(e,a,n),s(e,r,n),d(r,x),W(m,r,null),d(r,T),s(e,h,n),W(v,e,n),s(e,o,n),s(e,p,n),d(p,b),s(e,J,n),W(k,e,n),s(e,K,n),s(e,S,n),d(S,te),s(e,N,n),W(E,e,n),s(e,O,n),s(e,B,n),d(B,ae),s(e,Q,n),s(e,I,n),d(I,ne),W(P,I,null),d(I,ie),s(e,U,n),s(e,V,n),d(V,le),s(e,X,n),s(e,C,n),Y=!0},p(e,n){const q={};n&1&&(q.$$scope={dirty:n,ctx:e}),m.$set(q);const H={};n&1&&(H.$$scope={dirty:n,ctx:e}),v.$set(H);const j={};n&1&&(j.$$scope={dirty:n,ctx:e}),k.$set(j);const z={};n&1&&(z.$$scope={dirty:n,ctx:e}),E.$set(z);const A={};n&1&&(A.$$scope={dirty:n,ctx:e}),P.$set(A)},i(e){Y||(D(m.$$.fragment,e),D(v.$$.fragment,e),D(k.$$.fragment,e),D(E.$$.fragment,e),D(P.$$.fragment,e),Y=!0)},o(e){R(m.$$.fragment,e),R(v.$$.fragment,e),R(k.$$.fragment,e),R(E.$$.fragment,e),R(P.$$.fragment,e),Y=!1},d(e){e&&t(l),e&&t(a),e&&t(r),G(m),e&&t(h),G(v,e),e&&t(o),e&&t(p),e&&t(J),G(k,e),e&&t(K),e&&t(S),e&&t(N),G(E,e),e&&t(O),e&&t(B),e&&t(Q),e&&t(I),G(P),e&&t(U),e&&t(V),e&&t(X),e&&t(C)}}}function de(f){let l,i,a,r,x,m,T,h,v;return h=new ue({props:{$$slots:{default:[_e]},$$scope:{ctx:f}}}),{c(){l=y("meta"),i=$(),a=y("h1"),r=u("Value Iteration"),x=$(),m=y("div"),T=$(),L(h.$$.fragment),this.h()},l(o){const p=me('[data-svelte="svelte-1n8ictx"]',document.head);l=g(p,"META",{name:!0,content:!0}),p.forEach(t),i=_(o),a=g(o,"H1",{});var b=w(a);r=c(b,"Value Iteration"),b.forEach(t),x=_(o),m=g(o,"DIV",{class:!0}),w(m).forEach(t),T=_(o),M(h.$$.fragment,o),this.h()},h(){document.title="World4AI | Reinforcement Learning | Value Iteration Algorithm",Z(l,"name","description"),Z(l,"content","Value iteration is an iterative (dynamic programming) algorithm. The algorithm alternates between (truncated) policy evaluation and policy improvement to arrive at the optimal policy and value functions"),Z(m,"class","separator")},m(o,p){d(document.head,l),s(o,i,p),s(o,a,p),d(a,r),s(o,x,p),s(o,m,p),s(o,T,p),W(h,o,p),v=!0},p(o,[p]){const b={};p&1&&(b.$$scope={dirty:p,ctx:o}),h.$set(b)},i(o){v||(D(h.$$.fragment,o),v=!0)},o(o){R(h.$$.fragment,o),v=!1},d(o){t(l),o&&t(i),o&&t(a),o&&t(x),o&&t(m),o&&t(T),G(h,o)}}}class be extends se{constructor(l){super(),re(this,l,null,de,pe,{})}}export{be as default};
