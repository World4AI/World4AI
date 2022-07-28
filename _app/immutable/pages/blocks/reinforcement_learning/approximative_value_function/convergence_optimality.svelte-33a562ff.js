import{S as fe,i as pe,s as de,l as d,a as c,r as u,w as O,T as ge,m as g,h as t,c as m,n as b,u as f,x as A,p as Z,G as p,b as o,y as H,f as P,t as V,B as I}from"../../../../chunks/index-caa95cd4.js";import{C as ve}from"../../../../chunks/Container-5c6b7f6d.js";import{T as ue}from"../../../../chunks/Table-a51fb5ea.js";import{L as me}from"../../../../chunks/Latex-bf74aeea.js";function we(v){let r;return{c(){r=u("\\pi")},l(i){r=f(i,"\\pi")},m(i,s){o(i,r,s)},d(i){i&&t(r)}}}function be(v){let r;return{c(){r=u("\\pi")},l(i){r=f(i,"\\pi")},m(i,s){o(i,r,s)},d(i){i&&t(r)}}}function $e(v){let r,i,s,$,x,y,T,w,_,n,l,h,ee,R,k,Q,C,te,E,ne,B,F,ae,G,M,j,L,oe,z,S,re,J,q,ie,K,N,U;return h=new me({props:{$$slots:{default:[we]},$$scope:{ctx:v}}}),k=new ue({props:{header:v[0],data:v[1]}}),E=new me({props:{$$slots:{default:[be]},$$scope:{ctx:v}}}),M=new ue({props:{header:v[2],data:v[3]}}),{c(){r=d("p"),i=u(`The algorithms that we discussed during the last chapters attempt to find
    weights that create an approximate function that is as close as possible to
    the true state or action value function. The measurement of closeness that
    is used throughout reinforcement learning is the mean squared error (MSE).
    But in what way does finding the weights that produce the minimal mean
    squared error contribute to a value function that is close to the optimal
    function and are we guaranteed to find such weights?`),s=c(),$=d("h2"),x=u("Convergence"),y=c(),T=d("p"),w=u(`When we talk about convergence we usually mean that as time moves along, the
    value function of the agent changes towards some specific form. The steps
    towards that form get smaller and smaller and our function should have the
    desired form in the limit.`),_=c(),n=d("p"),l=u(`What does convergence mean for the prediction problem? For tabular methods
    we aspire to find the true value function of a policy `),O(h.$$.fragment),ee=u(`.
    Therefore convergence means that the value function of the agent converges
    towards the true value function. For approximative methods the agent adjusts
    the weight vector through gradient descent to reduce the mean squared error
    and if convergence is possible the weights move towards a specific vector.
    That does not necessarily mean that the agent finds a weight vector that
    generates the smallest possible MSE, as gradient descent might get stuck in
    a local minimum.`),R=c(),O(k.$$.fragment),Q=c(),C=d("p"),te=u("Monte Carlo and TD algorithms converge towards the true value function of "),O(E.$$.fragment),ne=u(`
    when the agent deals with finite MDPs and uses tabular methods. When we talk
    about approximate solutions the answers to the question whether prediction algorithms
    converge depend strongly on the type of algorithm. Monte Carlo algorithms use
    returns as a proxy for the true value function. Returns are unbiased but noisy
    estimates of the true value function, therefore we have a guarantee of convergence
    while using gradient descent. Linear methods converge to global optimum while
    non-linear methods (like neural networks) converge to a local optimum. The MSE
    for linear monte carlo approximators is convex, which means that there is a single
    optimum which is guaranteed to be found. The MSE for non-linear monte carlo approximators
    is non-convex, therefore gradient descent might get stuck in a local optimum.
    Temporal difference methods use bootstrapping. These algorithms use estimates
    for the target values in the update step. That makes them biased estimators.
    Q-Learning especially is problematic as there is no convergence guarantee even
    for linear methods.`),B=c(),F=d("p"),ae=u(`What does convergence mean for control? For tabular methods that means to
    find the optimal value function and thereby policy. Therefore convergence
    means that the value function of the agent converges towards the optimal
    value function. For approximative methods convergence means that gradient
    descent finds either a local or a global optimum, while trying to find the
    weights that minimize the mean squared error between the approximate value
    function and the optimal value function.`),G=c(),O(M.$$.fragment),j=c(),L=d("p"),oe=u(`Linear functions (MC and SARSA) oscillate around the near optimal value. For
    off-policy learning and non-linear methods no convergence guarantees exist.`),z=c(),S=d("h2"),re=u("Optimality"),J=c(),q=d("p"),ie=u(`Finding the true optimal value function is not possible with function
    approximators, because the state space is continuous or very large. How
    should we decide then which algorithms to use? The most important takeaway
    should be that when deciding between algorithms, convergence should not be
    the primary decision factor. If it was then linear approximators would be
    the first choice. In practice off-policy temporal difference algorithms are
    often the first choice, even though according to the table above there is no
    convergence guarantee. The truth of the matter is that in practice neural
    networks work well, provided we use some particular techniques to prevent
    divergence. We will learn more about those in the next chapters.`),K=c(),N=d("div"),this.h()},l(e){r=g(e,"P",{});var a=b(r);i=f(a,`The algorithms that we discussed during the last chapters attempt to find
    weights that create an approximate function that is as close as possible to
    the true state or action value function. The measurement of closeness that
    is used throughout reinforcement learning is the mean squared error (MSE).
    But in what way does finding the weights that produce the minimal mean
    squared error contribute to a value function that is close to the optimal
    function and are we guaranteed to find such weights?`),a.forEach(t),s=m(e),$=g(e,"H2",{});var W=b($);x=f(W,"Convergence"),W.forEach(t),y=m(e),T=g(e,"P",{});var D=b(T);w=f(D,`When we talk about convergence we usually mean that as time moves along, the
    value function of the agent changes towards some specific form. The steps
    towards that form get smaller and smaller and our function should have the
    desired form in the limit.`),D.forEach(t),_=m(e),n=g(e,"P",{});var X=b(n);l=f(X,`What does convergence mean for the prediction problem? For tabular methods
    we aspire to find the true value function of a policy `),A(h.$$.fragment,X),ee=f(X,`.
    Therefore convergence means that the value function of the agent converges
    towards the true value function. For approximative methods the agent adjusts
    the weight vector through gradient descent to reduce the mean squared error
    and if convergence is possible the weights move towards a specific vector.
    That does not necessarily mean that the agent finds a weight vector that
    generates the smallest possible MSE, as gradient descent might get stuck in
    a local minimum.`),X.forEach(t),R=m(e),A(k.$$.fragment,e),Q=m(e),C=g(e,"P",{});var Y=b(C);te=f(Y,"Monte Carlo and TD algorithms converge towards the true value function of "),A(E.$$.fragment,Y),ne=f(Y,`
    when the agent deals with finite MDPs and uses tabular methods. When we talk
    about approximate solutions the answers to the question whether prediction algorithms
    converge depend strongly on the type of algorithm. Monte Carlo algorithms use
    returns as a proxy for the true value function. Returns are unbiased but noisy
    estimates of the true value function, therefore we have a guarantee of convergence
    while using gradient descent. Linear methods converge to global optimum while
    non-linear methods (like neural networks) converge to a local optimum. The MSE
    for linear monte carlo approximators is convex, which means that there is a single
    optimum which is guaranteed to be found. The MSE for non-linear monte carlo approximators
    is non-convex, therefore gradient descent might get stuck in a local optimum.
    Temporal difference methods use bootstrapping. These algorithms use estimates
    for the target values in the update step. That makes them biased estimators.
    Q-Learning especially is problematic as there is no convergence guarantee even
    for linear methods.`),Y.forEach(t),B=m(e),F=g(e,"P",{});var se=b(F);ae=f(se,`What does convergence mean for control? For tabular methods that means to
    find the optimal value function and thereby policy. Therefore convergence
    means that the value function of the agent converges towards the optimal
    value function. For approximative methods convergence means that gradient
    descent finds either a local or a global optimum, while trying to find the
    weights that minimize the mean squared error between the approximate value
    function and the optimal value function.`),se.forEach(t),G=m(e),A(M.$$.fragment,e),j=m(e),L=g(e,"P",{});var le=b(L);oe=f(le,`Linear functions (MC and SARSA) oscillate around the near optimal value. For
    off-policy learning and non-linear methods no convergence guarantees exist.`),le.forEach(t),z=m(e),S=g(e,"H2",{});var he=b(S);re=f(he,"Optimality"),he.forEach(t),J=m(e),q=g(e,"P",{});var ce=b(q);ie=f(ce,`Finding the true optimal value function is not possible with function
    approximators, because the state space is continuous or very large. How
    should we decide then which algorithms to use? The most important takeaway
    should be that when deciding between algorithms, convergence should not be
    the primary decision factor. If it was then linear approximators would be
    the first choice. In practice off-policy temporal difference algorithms are
    often the first choice, even though according to the table above there is no
    convergence guarantee. The truth of the matter is that in practice neural
    networks work well, provided we use some particular techniques to prevent
    divergence. We will learn more about those in the next chapters.`),ce.forEach(t),K=m(e),N=g(e,"DIV",{class:!0}),b(N).forEach(t),this.h()},h(){Z(N,"class","separator")},m(e,a){o(e,r,a),p(r,i),o(e,s,a),o(e,$,a),p($,x),o(e,y,a),o(e,T,a),p(T,w),o(e,_,a),o(e,n,a),p(n,l),H(h,n,null),p(n,ee),o(e,R,a),H(k,e,a),o(e,Q,a),o(e,C,a),p(C,te),H(E,C,null),p(C,ne),o(e,B,a),o(e,F,a),p(F,ae),o(e,G,a),H(M,e,a),o(e,j,a),o(e,L,a),p(L,oe),o(e,z,a),o(e,S,a),p(S,re),o(e,J,a),o(e,q,a),p(q,ie),o(e,K,a),o(e,N,a),U=!0},p(e,a){const W={};a&16&&(W.$$scope={dirty:a,ctx:e}),h.$set(W);const D={};a&16&&(D.$$scope={dirty:a,ctx:e}),E.$set(D)},i(e){U||(P(h.$$.fragment,e),P(k.$$.fragment,e),P(E.$$.fragment,e),P(M.$$.fragment,e),U=!0)},o(e){V(h.$$.fragment,e),V(k.$$.fragment,e),V(E.$$.fragment,e),V(M.$$.fragment,e),U=!1},d(e){e&&t(r),e&&t(s),e&&t($),e&&t(y),e&&t(T),e&&t(_),e&&t(n),I(h),e&&t(R),I(k,e),e&&t(Q),e&&t(C),I(E),e&&t(B),e&&t(F),e&&t(G),I(M,e),e&&t(j),e&&t(L),e&&t(z),e&&t(S),e&&t(J),e&&t(q),e&&t(K),e&&t(N)}}}function ye(v){let r,i,s,$,x,y,T,w,_;return w=new ve({props:{$$slots:{default:[$e]},$$scope:{ctx:v}}}),{c(){r=d("meta"),i=c(),s=d("h1"),$=u("Convergence and Optimality"),x=c(),y=d("div"),T=c(),O(w.$$.fragment),this.h()},l(n){const l=ge('[data-svelte="svelte-6100ga"]',document.head);r=g(l,"META",{name:!0,content:!0}),l.forEach(t),i=m(n),s=g(n,"H1",{});var h=b(s);$=f(h,"Convergence and Optimality"),h.forEach(t),x=m(n),y=g(n,"DIV",{class:!0}),b(y).forEach(t),T=m(n),A(w.$$.fragment,n),this.h()},h(){document.title=`World4AI | Reinforcement Learning | Approximation Convergence and
    Optimality
  `,Z(r,"name","description"),Z(r,"content","Non linear function approximators, especially in combination with off policy temporal difference learning exhibit very poor convergence properties."),Z(y,"class","separator")},m(n,l){p(document.head,r),o(n,i,l),o(n,s,l),p(s,$),o(n,x,l),o(n,y,l),o(n,T,l),H(w,n,l),_=!0},p(n,[l]){const h={};l&16&&(h.$$scope={dirty:l,ctx:n}),w.$set(h)},i(n){_||(P(w.$$.fragment,n),_=!0)},o(n){V(w.$$.fragment,n),_=!1},d(n){t(r),n&&t(i),n&&t(s),n&&t(x),n&&t(y),n&&t(T),I(w,n)}}}function Te(v){return[["Algorithm","Tabular","Linear","Non-Linear"],[["Monte Carlo","Convergence (True Value Function)","Convergence (Global Optimum)","Convergence (Local Optimum)"],["Sarsa","Convergence (True Value Function)","Convergence (Near Global Optimum)","No Convergence"],["Q-Learning","Convergence (True Value Function)","No Convergence","No Convergence"]],["Algorithm","Tabular","Linear","Non-Linear"],[["Monte Carlo","Convergence (True Value Function)","Oscilates","No Convergence"],["Sarsa","Convergence (True Value Function)","Oscilates","No Convergence"],["Q-Learning","Convergence (True Value Function)","No Convergence","No Convergence"]]]}class ke extends fe{constructor(r){super(),pe(this,r,Te,ye,de,{})}}export{ke as default};
