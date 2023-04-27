import{S as se,i as re,s as pe,k as y,a as $,q as c,y as W,W as me,l as g,h as t,c as _,m as w,r as u,z as L,n as Z,N as d,b as s,A as M,g as D,d as R,B as z,C as ee}from"../chunks/index.4d92b023.js";import{C as ce}from"../chunks/Container.b0705c7b.js";import{L as F}from"../chunks/Latex.e0b308c0.js";function ue(f){let l;return{c(){l=c("v_*(s)")},l(i){l=u(i,"v_*(s)")},m(i,a){s(i,l,a)},d(i){i&&t(l)}}}function fe(f){let l=String.raw`
\begin{aligned}
  v_*(s) & = \max_a q_*(s, a) \\
  & = \max_a \mathbb{E}_{\pi}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a] \\ 
  & = \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_*(s')]
\end{aligned}
`+"",i;return{c(){i=c(l)},l(a){i=u(a,l)},m(a,r){s(a,i,r)},p:ee,d(a){a&&t(i)}}}function he(f){let l=String.raw`
      v_{k+1}(s) \doteq \max_a \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_k (s')]
  `+"",i;return{c(){i=c(l)},l(a){i=u(a,l)},m(a,r){s(a,i,r)},p:ee,d(a){a&&t(i)}}}function ve(f){let l=String.raw`
  \begin{aligned}
    \text{(1: Policy Evaluation) } & q_{k+1}(s, a) = \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_k (s')] \\
    \text{(2: Policy Improvement) }& v_{k+1}(s) = \max_a q_{k+1}(s, a)
  \end{aligned}
`+"",i;return{c(){i=c(l)},l(a){i=u(a,l)},m(a,r){s(a,i,r)},p:ee,d(a){a&&t(i)}}}function $e(f){let l=String.raw`\theta`+"",i;return{c(){i=c(l)},l(a){i=u(a,l)},m(a,r){s(a,i,r)},p:ee,d(a){a&&t(i)}}}function _e(f){let l,i,a,r,x,m,k,h,v,o,p,b,G,T,J,S,te,K,I,O,B,ae,Q,E,ne,P,ie,U,C,le,X,V,Y;return m=new F({props:{$$slots:{default:[ue]},$$scope:{ctx:f}}}),v=new F({props:{$$slots:{default:[fe]},$$scope:{ctx:f}}}),T=new F({props:{$$slots:{default:[he]},$$scope:{ctx:f}}}),I=new F({props:{$$slots:{default:[ve]},$$scope:{ctx:f}}}),P=new F({props:{$$slots:{default:[$e]},$$scope:{ctx:f}}}),{c(){l=y("p"),i=c(`When we consider policy iteration again, we should remember that there are
    two distinct steps, policy evaluation and policy improvement. The policy
    improvement step is a single step, where the new policy is derived by acting
    greedily. The policy evaluation on the other hand is a longer iterative
    process. It turns out that it is not necessary to wait for the policy
    evaluation algorithm to finish converging to the true value function. In
    fact the value iteration algorithm works with only one single policy
    evaluation step.`),a=$(),r=y("p"),x=c("The main goal of value iteration is to find the optimal value function "),W(m.$$.fragment),k=c(`, that can be used to derive the optimal policy. The optimal value function
    can be expressed as a Bellman equation that looks as follows.`),h=$(),W(v.$$.fragment),o=$(),p=y("p"),b=c(`The value iteration is essentially the Bellman optimality equation, that has
    been transformed to an iterative algorithm.`),G=$(),W(T.$$.fragment),J=$(),S=y("p"),te=c(`Although the update step looks like a single step at first glance, it
    actually combines truncated (one step) policy evaluation and policy
    improvement in a single step.`),K=$(),W(I.$$.fragment),O=$(),B=y("p"),ae=c(`In the first step the action-value function is calculated based on the old
    state-value function and the model of the Markov decision process. In the
    second step a max over the action-value function is taken in order to
    generate the new state-value function. That implicitly generates a new
    policy as a value function is always calculated for a particular policy.`),Q=$(),E=y("p"),ne=c(`The combination of both steps is the value iteration algorithm. The
    iterative process continues until the difference between the old and the new
    state-value function is smaller than some parameter theta `),W(P.$$.fragment),ie=c(`. As the final step the optimal policy can be deduced by always selecting
    the greedy action.`),U=$(),C=y("p"),le=c(`Below is the Python implementation of the value iteration algorithm.
    Compared to policy iteration the implementation is more compact, because
    policy evaluation and policy improvement can be implemented in a single
    function.`),X=$(),V=y("div"),this.h()},l(e){l=g(e,"P",{});var n=w(l);i=u(n,`When we consider policy iteration again, we should remember that there are
    two distinct steps, policy evaluation and policy improvement. The policy
    improvement step is a single step, where the new policy is derived by acting
    greedily. The policy evaluation on the other hand is a longer iterative
    process. It turns out that it is not necessary to wait for the policy
    evaluation algorithm to finish converging to the true value function. In
    fact the value iteration algorithm works with only one single policy
    evaluation step.`),n.forEach(t),a=_(e),r=g(e,"P",{});var q=w(r);x=u(q,"The main goal of value iteration is to find the optimal value function "),L(m.$$.fragment,q),k=u(q,`, that can be used to derive the optimal policy. The optimal value function
    can be expressed as a Bellman equation that looks as follows.`),q.forEach(t),h=_(e),L(v.$$.fragment,e),o=_(e),p=g(e,"P",{});var H=w(p);b=u(H,`The value iteration is essentially the Bellman optimality equation, that has
    been transformed to an iterative algorithm.`),H.forEach(t),G=_(e),L(T.$$.fragment,e),J=_(e),S=g(e,"P",{});var N=w(S);te=u(N,`Although the update step looks like a single step at first glance, it
    actually combines truncated (one step) policy evaluation and policy
    improvement in a single step.`),N.forEach(t),K=_(e),L(I.$$.fragment,e),O=_(e),B=g(e,"P",{});var j=w(B);ae=u(j,`In the first step the action-value function is calculated based on the old
    state-value function and the model of the Markov decision process. In the
    second step a max over the action-value function is taken in order to
    generate the new state-value function. That implicitly generates a new
    policy as a value function is always calculated for a particular policy.`),j.forEach(t),Q=_(e),E=g(e,"P",{});var A=w(E);ne=u(A,`The combination of both steps is the value iteration algorithm. The
    iterative process continues until the difference between the old and the new
    state-value function is smaller than some parameter theta `),L(P.$$.fragment,A),ie=u(A,`. As the final step the optimal policy can be deduced by always selecting
    the greedy action.`),A.forEach(t),U=_(e),C=g(e,"P",{});var oe=w(C);le=u(oe,`Below is the Python implementation of the value iteration algorithm.
    Compared to policy iteration the implementation is more compact, because
    policy evaluation and policy improvement can be implemented in a single
    function.`),oe.forEach(t),X=_(e),V=g(e,"DIV",{class:!0}),w(V).forEach(t),this.h()},h(){Z(V,"class","separator")},m(e,n){s(e,l,n),d(l,i),s(e,a,n),s(e,r,n),d(r,x),M(m,r,null),d(r,k),s(e,h,n),M(v,e,n),s(e,o,n),s(e,p,n),d(p,b),s(e,G,n),M(T,e,n),s(e,J,n),s(e,S,n),d(S,te),s(e,K,n),M(I,e,n),s(e,O,n),s(e,B,n),d(B,ae),s(e,Q,n),s(e,E,n),d(E,ne),M(P,E,null),d(E,ie),s(e,U,n),s(e,C,n),d(C,le),s(e,X,n),s(e,V,n),Y=!0},p(e,n){const q={};n&1&&(q.$$scope={dirty:n,ctx:e}),m.$set(q);const H={};n&1&&(H.$$scope={dirty:n,ctx:e}),v.$set(H);const N={};n&1&&(N.$$scope={dirty:n,ctx:e}),T.$set(N);const j={};n&1&&(j.$$scope={dirty:n,ctx:e}),I.$set(j);const A={};n&1&&(A.$$scope={dirty:n,ctx:e}),P.$set(A)},i(e){Y||(D(m.$$.fragment,e),D(v.$$.fragment,e),D(T.$$.fragment,e),D(I.$$.fragment,e),D(P.$$.fragment,e),Y=!0)},o(e){R(m.$$.fragment,e),R(v.$$.fragment,e),R(T.$$.fragment,e),R(I.$$.fragment,e),R(P.$$.fragment,e),Y=!1},d(e){e&&t(l),e&&t(a),e&&t(r),z(m),e&&t(h),z(v,e),e&&t(o),e&&t(p),e&&t(G),z(T,e),e&&t(J),e&&t(S),e&&t(K),z(I,e),e&&t(O),e&&t(B),e&&t(Q),e&&t(E),z(P),e&&t(U),e&&t(C),e&&t(X),e&&t(V)}}}function de(f){let l,i,a,r,x,m,k,h,v;return h=new ce({props:{$$slots:{default:[_e]},$$scope:{ctx:f}}}),{c(){l=y("meta"),i=$(),a=y("h1"),r=c("Value Iteration"),x=$(),m=y("div"),k=$(),W(h.$$.fragment),this.h()},l(o){const p=me("svelte-1n8ictx",document.head);l=g(p,"META",{name:!0,content:!0}),p.forEach(t),i=_(o),a=g(o,"H1",{});var b=w(a);r=u(b,"Value Iteration"),b.forEach(t),x=_(o),m=g(o,"DIV",{class:!0}),w(m).forEach(t),k=_(o),L(h.$$.fragment,o),this.h()},h(){document.title="World4AI | Reinforcement Learning | Value Iteration Algorithm",Z(l,"name","description"),Z(l,"content","Value iteration is an iterative (dynamic programming) algorithm. The algorithm alternates between (truncated) policy evaluation and policy improvement to arrive at the optimal policy and value functions"),Z(m,"class","separator")},m(o,p){d(document.head,l),s(o,i,p),s(o,a,p),d(a,r),s(o,x,p),s(o,m,p),s(o,k,p),M(h,o,p),v=!0},p(o,[p]){const b={};p&1&&(b.$$scope={dirty:p,ctx:o}),h.$set(b)},i(o){v||(D(h.$$.fragment,o),v=!0)},o(o){R(h.$$.fragment,o),v=!1},d(o){t(l),o&&t(i),o&&t(a),o&&t(x),o&&t(m),o&&t(k),z(h,o)}}}class be extends se{constructor(l){super(),re(this,l,null,de,pe,{})}}export{be as default};
