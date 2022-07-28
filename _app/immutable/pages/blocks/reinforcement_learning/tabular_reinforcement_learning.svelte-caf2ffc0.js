import{S as Z,i as ee,s as te,l as u,a as d,r as w,w as j,T as ae,m as v,h as t,c as y,n as b,u as g,x as z,p as W,G as p,b as i,y as F,f as G,t as H,B as N,E as ne}from"../../../chunks/index-caa95cd4.js";import{C as ie}from"../../../chunks/Container-5c6b7f6d.js";import{T as oe}from"../../../chunks/Table-a51fb5ea.js";import{L as se}from"../../../chunks/Latex-bf74aeea.js";/* empty css                                                              */function re($){let o=String.raw`
  \begin{aligned}
    \text{(1: Policy Evaluation) } & q_{k+1}(s, a) = \sum_{s', r} p(s', r \mid s, a) [r + \gamma v_k (s')] \\
    \text{(2: Policy Improvement) }& v_{k+1}(s) = \max_a q_{k+1}(s, a)
  \end{aligned}
`+"",m;return{c(){m=w(o)},l(s){m=g(s,o)},m(s,h){i(s,m,h)},p:ne,d(s){s&&t(m)}}}function le($){let o,m,s,h,_,l,T,c,x,a,r,f,V,I,k,S,q,B,A,D,C,L,P,O,M,Q,R;return h=new oe({props:{header:$[0],data:$[1]}}),k=new se({props:{$$slots:{default:[re]},$$scope:{ctx:$}}}),{c(){o=u("p"),m=w(`Dynamic programming offers us valuable theoretical foundations, but
    generally speaking we do not have access to the model of the environment.
    The only viable solution that remains is learning through interaction with
    the environment. This chapter is dedicated to reinforcement learning in
    finite MDPs. Finite Markov decision processes have a finite state and action
    space. This allows for the implementation of state-value and action-value
    functions that can be stored in lists or tables. Those implementations are
    appropriately called tabular methods.`),s=d(),j(h.$$.fragment),_=d(),l=u("p"),T=w("Above we can see a so called "),c=u("strong"),x=w("Q-table"),a=w(` for an imaginary gridworld
    with 3 states and 2 actions. The available actions are used as the column names,
    the available states are used as row names and the intersection contains the
    estimated Q-value for that specific state-action pair.`),r=d(),f=u("p"),V=w(`To understand why Q-tables are essential tools that can help us find the
    optimal policy, let us remind ourselves how the value iteration algorithm
    works.`),I=d(),j(k.$$.fragment),S=d(),q=u("p"),B=w(`Value iteration consists of policy evaluation (that requires the knowledge
    of the model) and policy improvement (that takes the max over the
    action-value function). Ignoring the first step for now, once we have access
    to the action-value function (a Q-table in our case), all we really need to
    do in each state is to choose the action with the maximum value. Different
    tabular reinforcement learning algorithms utilize different methods to
    estimate the Q-values, but the policy improvement steps remain similar.`),A=d(),D=u("p"),C=w(`Before we move on to the discussion of tabular reinforcement learning
    algorithms it is important to discuss the difference between on-policy and
    off-policy methods. We could ask ourselves: \u201CDo we need to improve the same
    policy that is used to generate actions or can we learn the optimal policy
    while using the data that was produced by a different policy?\u201D. To frame the
    question differently \u201CIs it possible to learn the optimal policy while only
    selecting random actions?\u201D. That depends on the design of the algorithm.
    On-policy methods improve the same policy that is also used to generate the
    actions, while off-policy methods improve a policy that is not the one that
    is used to generate the trajectories. We will encounter and implement both
    types of algorithms.`),L=d(),P=u("p"),O=w(`The current state of the art reinforcement learning rarely deals with
    tabular methods any more, but it is still more convenient to start the
    exploration of reinforcement learning techniques with those, as the general
    ideas are extremely relevant to modern (approximative) methods which are
    going to be introduced in future sections.`),M=d(),Q=u("div"),this.h()},l(e){o=v(e,"P",{});var n=b(o);m=g(n,`Dynamic programming offers us valuable theoretical foundations, but
    generally speaking we do not have access to the model of the environment.
    The only viable solution that remains is learning through interaction with
    the environment. This chapter is dedicated to reinforcement learning in
    finite MDPs. Finite Markov decision processes have a finite state and action
    space. This allows for the implementation of state-value and action-value
    functions that can be stored in lists or tables. Those implementations are
    appropriately called tabular methods.`),n.forEach(t),s=y(e),z(h.$$.fragment,e),_=y(e),l=v(e,"P",{});var E=b(l);T=g(E,"Above we can see a so called "),c=v(E,"STRONG",{});var J=b(c);x=g(J,"Q-table"),J.forEach(t),a=g(E,` for an imaginary gridworld
    with 3 states and 2 actions. The available actions are used as the column names,
    the available states are used as row names and the intersection contains the
    estimated Q-value for that specific state-action pair.`),E.forEach(t),r=y(e),f=v(e,"P",{});var K=b(f);V=g(K,`To understand why Q-tables are essential tools that can help us find the
    optimal policy, let us remind ourselves how the value iteration algorithm
    works.`),K.forEach(t),I=y(e),z(k.$$.fragment,e),S=y(e),q=v(e,"P",{});var U=b(q);B=g(U,`Value iteration consists of policy evaluation (that requires the knowledge
    of the model) and policy improvement (that takes the max over the
    action-value function). Ignoring the first step for now, once we have access
    to the action-value function (a Q-table in our case), all we really need to
    do in each state is to choose the action with the maximum value. Different
    tabular reinforcement learning algorithms utilize different methods to
    estimate the Q-values, but the policy improvement steps remain similar.`),U.forEach(t),A=y(e),D=v(e,"P",{});var X=b(D);C=g(X,`Before we move on to the discussion of tabular reinforcement learning
    algorithms it is important to discuss the difference between on-policy and
    off-policy methods. We could ask ourselves: \u201CDo we need to improve the same
    policy that is used to generate actions or can we learn the optimal policy
    while using the data that was produced by a different policy?\u201D. To frame the
    question differently \u201CIs it possible to learn the optimal policy while only
    selecting random actions?\u201D. That depends on the design of the algorithm.
    On-policy methods improve the same policy that is also used to generate the
    actions, while off-policy methods improve a policy that is not the one that
    is used to generate the trajectories. We will encounter and implement both
    types of algorithms.`),X.forEach(t),L=y(e),P=v(e,"P",{});var Y=b(P);O=g(Y,`The current state of the art reinforcement learning rarely deals with
    tabular methods any more, but it is still more convenient to start the
    exploration of reinforcement learning techniques with those, as the general
    ideas are extremely relevant to modern (approximative) methods which are
    going to be introduced in future sections.`),Y.forEach(t),M=y(e),Q=v(e,"DIV",{class:!0}),b(Q).forEach(t),this.h()},h(){W(Q,"class","separator")},m(e,n){i(e,o,n),p(o,m),i(e,s,n),F(h,e,n),i(e,_,n),i(e,l,n),p(l,T),p(l,c),p(c,x),p(l,a),i(e,r,n),i(e,f,n),p(f,V),i(e,I,n),F(k,e,n),i(e,S,n),i(e,q,n),p(q,B),i(e,A,n),i(e,D,n),p(D,C),i(e,L,n),i(e,P,n),p(P,O),i(e,M,n),i(e,Q,n),R=!0},p(e,n){const E={};n&4&&(E.$$scope={dirty:n,ctx:e}),k.$set(E)},i(e){R||(G(h.$$.fragment,e),G(k.$$.fragment,e),R=!0)},o(e){H(h.$$.fragment,e),H(k.$$.fragment,e),R=!1},d(e){e&&t(o),e&&t(s),N(h,e),e&&t(_),e&&t(l),e&&t(r),e&&t(f),e&&t(I),N(k,e),e&&t(S),e&&t(q),e&&t(A),e&&t(D),e&&t(L),e&&t(P),e&&t(M),e&&t(Q)}}}function me($){let o,m,s,h,_,l,T,c,x;return c=new ie({props:{$$slots:{default:[le]},$$scope:{ctx:$}}}),{c(){o=u("meta"),m=d(),s=u("h1"),h=w("Tabular Reinforcement Learning"),_=d(),l=u("div"),T=d(),j(c.$$.fragment),this.h()},l(a){const r=ae('[data-svelte="svelte-1clvkx4"]',document.head);o=v(r,"META",{name:!0,content:!0}),r.forEach(t),m=y(a),s=v(a,"H1",{});var f=b(s);h=g(f,"Tabular Reinforcement Learning"),f.forEach(t),_=y(a),l=v(a,"DIV",{class:!0}),b(l).forEach(t),T=y(a),z(c.$$.fragment,a),this.h()},h(){document.title="World4AI | Reinforcement Learning | Tabular Reinforcement Learning",W(o,"name","description"),W(o,"content","Tabular reinforcement learning deals with finding optimal value functions and policies for finite markov decision processes."),W(l,"class","separator")},m(a,r){p(document.head,o),i(a,m,r),i(a,s,r),p(s,h),i(a,_,r),i(a,l,r),i(a,T,r),F(c,a,r),x=!0},p(a,[r]){const f={};r&4&&(f.$$scope={dirty:r,ctx:a}),c.$set(f)},i(a){x||(G(c.$$.fragment,a),x=!0)},o(a){H(c.$$.fragment,a),x=!1},d(a){t(o),a&&t(m),a&&t(s),a&&t(_),a&&t(l),a&&t(T),N(c,a)}}}function ce($){return[["","A_0","A_1"],[["S_0","0","2"],["S_1","1","3"],["S_2","2","4"]]]}class ve extends Z{constructor(o){super(),ee(this,o,ce,me,te,{})}}export{ve as default};
