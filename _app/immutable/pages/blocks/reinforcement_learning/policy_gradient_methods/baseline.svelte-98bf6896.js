import{S as ne,i as ie,s as se,l as R,a as v,r as h,w as V,T as re,m as I,h as n,c as g,n as N,u as f,x as B,p as K,G as b,b as l,y as P,f as H,t as A,B as G,E as ae}from"../../../../chunks/index-caa95cd4.js";import{C as le}from"../../../../chunks/Container-5c6b7f6d.js";import{L as W}from"../../../../chunks/Latex-bf74aeea.js";function oe(u){let a;return{c(){a=h("b")},l(t){a=f(t,"b")},m(t,i){l(t,a,i)},d(t){t&&n(a)}}}function he(u){let a=String.raw`
    \begin{aligned}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) R(\tau)] \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) \sum_{k=t+1}^{H} \gamma^{k-t-1} R_k - b
    \end{aligned}
`+"",t;return{c(){t=h(a)},l(i){t=f(i,a)},m(i,c){l(i,t,c)},p:ae,d(i){i&&n(t)}}}function fe(u){let a;return{c(){a=h("b")},l(t){a=f(t,"b")},m(t,i){l(t,a,i)},d(t){t&&n(a)}}}function ue(u){let a;return{c(){a=h("V(S_t)")},l(t){a=f(t,"V(S_t)")},m(t,i){l(t,a,i)},d(t){t&&n(a)}}}function ce(u){let a;return{c(){a=h("V(S_t)")},l(t){a=f(t,"V(S_t)")},m(t,i){l(t,a,i)},d(t){t&&n(a)}}}function me(u){let a=String.raw`
    \begin{aligned}
    \nabla_{\theta} J(\theta) & = \mathbb{E}_{\tau \sim \pi_{\theta}}[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) R(\tau)] \\
    & \approx \frac{1}{m}\sum_i^m\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t^{(i)} \mid S_t^{(i)}) \sum_{k=t+1}^{H} \gamma^{k-t-1} R_k - V(S_t^{(i)})
    \end{aligned}
  `+"",t;return{c(){t=h(a)},l(i){t=f(i,a)},m(i,c){l(i,t,c)},p:ae,d(i){i&&n(t)}}}function pe(u){let a,t,i,c,T,d,p,_,$,r,o,m,z,y,Q,x,U,D,w,X,C,Y,k,Z,J,F,M,L,j;return $=new W({props:{$$slots:{default:[oe]},$$scope:{ctx:u}}}),m=new W({props:{$$slots:{default:[he]},$$scope:{ctx:u}}}),x=new W({props:{$$slots:{default:[fe]},$$scope:{ctx:u}}}),C=new W({props:{$$slots:{default:[ue]},$$scope:{ctx:u}}}),k=new W({props:{$$slots:{default:[ce]},$$scope:{ctx:u}}}),F=new W({props:{$$slots:{default:[me]},$$scope:{ctx:u}}}),{c(){a=R("p"),t=h(`The reinforce algorithm that we developed in the last chapter is the first
    viable variant of a policy gradient method, yet not the one with the least
    amount of variance. In this chapter we introduce an improvement to
    REINFORCE. The algorithm is called REINFORCE with baseline, but sometimes
    the name "vanilla policy gradient" (VPG) is also used.`),i=v(),c=R("p"),T=h(`The gradient calculation for REINFORCE can be interpreted as follows. The
    log probability of actions that generate high returns is increased, while
    the log probability of actions that generate negative returns is decreased.
    But what if all returns are positive or clustered? The probability of
    actions with highest returns will increase more than those with lower
    returns. But the process is slow.`),d=v(),p=R("p"),_=h(`To reduce the variance we are going to introduce what is called the
    "baseline". The baseline
    `),V($.$$.fragment),r=h(` is deducted from the return, which has not bias in expectation,
    but reduces the variance.`),o=v(),V(m.$$.fragment),z=v(),y=R("p"),Q=h(`Intuitively some of the positive returns might stay positive, while others
    are pushed below the zero line. That makes the gradient positive for returns
    above the baseline and negative for returns below the baseline. There are
    different choices for the baseline `),V(x.$$.fragment),U=h(`, which has an impact on
    how much variance and bias the algorithm has.`),D=v(),w=R("p"),X=h("REINFORCE with baseline uses the state value function "),V(C.$$.fragment),Y=h(` as
    the baseline. This makes perfect sense as only the probability of those actions
    are increased that generate returns that are above the expected sum of rewards.
    In our implementation `),V(k.$$.fragment),Z=h(` is going to be a learned neural network,
    meaning that we have two separate functions, one for the policy and one for the
    state value. In VPG the policy and value functions are updated with monte carlo
    simulations, therefore this algorithm is only suited for episodic tasks.`),J=v(),V(F.$$.fragment),M=v(),L=R("div"),this.h()},l(e){a=I(e,"P",{});var s=N(a);t=f(s,`The reinforce algorithm that we developed in the last chapter is the first
    viable variant of a policy gradient method, yet not the one with the least
    amount of variance. In this chapter we introduce an improvement to
    REINFORCE. The algorithm is called REINFORCE with baseline, but sometimes
    the name "vanilla policy gradient" (VPG) is also used.`),s.forEach(n),i=g(e),c=I(e,"P",{});var q=N(c);T=f(q,`The gradient calculation for REINFORCE can be interpreted as follows. The
    log probability of actions that generate high returns is increased, while
    the log probability of actions that generate negative returns is decreased.
    But what if all returns are positive or clustered? The probability of
    actions with highest returns will increase more than those with lower
    returns. But the process is slow.`),q.forEach(n),d=g(e),p=I(e,"P",{});var O=N(p);_=f(O,`To reduce the variance we are going to introduce what is called the
    "baseline". The baseline
    `),B($.$$.fragment,O),r=f(O,` is deducted from the return, which has not bias in expectation,
    but reduces the variance.`),O.forEach(n),o=g(e),B(m.$$.fragment,e),z=g(e),y=I(e,"P",{});var S=N(y);Q=f(S,`Intuitively some of the positive returns might stay positive, while others
    are pushed below the zero line. That makes the gradient positive for returns
    above the baseline and negative for returns below the baseline. There are
    different choices for the baseline `),B(x.$$.fragment,S),U=f(S,`, which has an impact on
    how much variance and bias the algorithm has.`),S.forEach(n),D=g(e),w=I(e,"P",{});var E=N(w);X=f(E,"REINFORCE with baseline uses the state value function "),B(C.$$.fragment,E),Y=f(E,` as
    the baseline. This makes perfect sense as only the probability of those actions
    are increased that generate returns that are above the expected sum of rewards.
    In our implementation `),B(k.$$.fragment,E),Z=f(E,` is going to be a learned neural network,
    meaning that we have two separate functions, one for the policy and one for the
    state value. In VPG the policy and value functions are updated with monte carlo
    simulations, therefore this algorithm is only suited for episodic tasks.`),E.forEach(n),J=g(e),B(F.$$.fragment,e),M=g(e),L=I(e,"DIV",{class:!0}),N(L).forEach(n),this.h()},h(){K(L,"class","separator")},m(e,s){l(e,a,s),b(a,t),l(e,i,s),l(e,c,s),b(c,T),l(e,d,s),l(e,p,s),b(p,_),P($,p,null),b(p,r),l(e,o,s),P(m,e,s),l(e,z,s),l(e,y,s),b(y,Q),P(x,y,null),b(y,U),l(e,D,s),l(e,w,s),b(w,X),P(C,w,null),b(w,Y),P(k,w,null),b(w,Z),l(e,J,s),P(F,e,s),l(e,M,s),l(e,L,s),j=!0},p(e,s){const q={};s&1&&(q.$$scope={dirty:s,ctx:e}),$.$set(q);const O={};s&1&&(O.$$scope={dirty:s,ctx:e}),m.$set(O);const S={};s&1&&(S.$$scope={dirty:s,ctx:e}),x.$set(S);const E={};s&1&&(E.$$scope={dirty:s,ctx:e}),C.$set(E);const ee={};s&1&&(ee.$$scope={dirty:s,ctx:e}),k.$set(ee);const te={};s&1&&(te.$$scope={dirty:s,ctx:e}),F.$set(te)},i(e){j||(H($.$$.fragment,e),H(m.$$.fragment,e),H(x.$$.fragment,e),H(C.$$.fragment,e),H(k.$$.fragment,e),H(F.$$.fragment,e),j=!0)},o(e){A($.$$.fragment,e),A(m.$$.fragment,e),A(x.$$.fragment,e),A(C.$$.fragment,e),A(k.$$.fragment,e),A(F.$$.fragment,e),j=!1},d(e){e&&n(a),e&&n(i),e&&n(c),e&&n(d),e&&n(p),G($),e&&n(o),G(m,e),e&&n(z),e&&n(y),G(x),e&&n(D),e&&n(w),G(C),G(k),e&&n(J),G(F,e),e&&n(M),e&&n(L)}}}function $e(u){let a,t,i,c,T,d,p,_,$;return _=new le({props:{$$slots:{default:[pe]},$$scope:{ctx:u}}}),{c(){a=R("meta"),t=v(),i=R("h1"),c=h("REINFORCE With Baseline"),T=v(),d=R("div"),p=v(),V(_.$$.fragment),this.h()},l(r){const o=re('[data-svelte="svelte-x7lhvr"]',document.head);a=I(o,"META",{name:!0,content:!0}),o.forEach(n),t=g(r),i=I(r,"H1",{});var m=N(i);c=f(m,"REINFORCE With Baseline"),m.forEach(n),T=g(r),d=I(r,"DIV",{class:!0}),N(d).forEach(n),p=g(r),B(_.$$.fragment,r),this.h()},h(){document.title="World4AI | Reinforcement Learning | REINFORCE with Baseline",K(a,"name","description"),K(a,"content","REINFORCE with baseline also calle vanilla policy gradient reduces the variance of the REINFORCE algorithm by introducing a baseline. The baseline is often a neural network, specifically a value function."),K(d,"class","separator")},m(r,o){b(document.head,a),l(r,t,o),l(r,i,o),b(i,c),l(r,T,o),l(r,d,o),l(r,p,o),P(_,r,o),$=!0},p(r,[o]){const m={};o&1&&(m.$$scope={dirty:o,ctx:r}),_.$set(m)},i(r){$||(H(_.$$.fragment,r),$=!0)},o(r){A(_.$$.fragment,r),$=!1},d(r){n(a),r&&n(t),r&&n(i),r&&n(T),r&&n(d),r&&n(p),G(_,r)}}}class we extends ne{constructor(a){super(),ie(this,a,null,$e,se,{})}}export{we as default};
