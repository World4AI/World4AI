import{S as j,i as B,s as C,l as u,a as g,r as I,w as A,T as D,m as f,h as i,c as w,n as v,u as $,x as S,p as P,G as _,b as o,y as V,f as H,t as L,B as R,E as W}from"../../../chunks/index-caa95cd4.js";import{C as F}from"../../../chunks/Container-5c6b7f6d.js";function J(k){let n,y,l,d,b,h,c,r,p,t,a,m,x,z,E,T;return{c(){n=u("p"),y=I(`The methods we have considered so far were designed to estimate the value
    function of a policy. The policy of the agent was determined implicitly by
    evaluating state-action pairs and taking the actions with the highest action
    value. In this chapter we are going to study methods that will allow us to
    learn the policy directly without using value functions. There are several
    reasons why we would prefer policy based methods over value based methods.
    Below is a subset of those reasons.`),l=g(),d=u("p"),b=I(`Q-learning does not easily work with a large numer of actions or with
    continuous action spaces, because of the max operation that is required to
    determine the best action. If the action space is large or continuous the
    maximization operation becomes very involved and the algorithm becomes very
    inefficient. In policy gradient methods we sample an action from a
    probability distribution, thereby removing the need for a maximization
    operation.`),h=g(),c=u("p"),r=I(`It is easy to implement a stochastic policy with policy gradient methods.
    This avoids the need for an additional exploration strategy, as we can
    randomly sample from the distribution and thereby explore the environment.
    Through gradient descent better actions are going to be assigned higher
    probabilities while bad actions will become unlikely.`),p=g(),t=u("p"),a=I(`In Q-learning the action which constitutes the greedy action might change
    due to gradient descent. The change of the policy might be very abrupt and
    might thus destabilize training. In policy gradient methods we apply
    gradient descent to the policy directly, therefore the change of the
    probability distribution of actions is relatively smooth.`),m=g(),x=u("p"),z=I(`The big disadvantage of policy gradient methods is the high variance, but
    adjustments to the naive implementation can be made to decrease the
    variance.`),E=g(),T=u("div"),this.h()},l(e){n=f(e,"P",{});var s=v(n);y=$(s,`The methods we have considered so far were designed to estimate the value
    function of a policy. The policy of the agent was determined implicitly by
    evaluating state-action pairs and taking the actions with the highest action
    value. In this chapter we are going to study methods that will allow us to
    learn the policy directly without using value functions. There are several
    reasons why we would prefer policy based methods over value based methods.
    Below is a subset of those reasons.`),s.forEach(i),l=w(e),d=f(e,"P",{});var q=v(d);b=$(q,`Q-learning does not easily work with a large numer of actions or with
    continuous action spaces, because of the max operation that is required to
    determine the best action. If the action space is large or continuous the
    maximization operation becomes very involved and the algorithm becomes very
    inefficient. In policy gradient methods we sample an action from a
    probability distribution, thereby removing the need for a maximization
    operation.`),q.forEach(i),h=w(e),c=f(e,"P",{});var G=v(c);r=$(G,`It is easy to implement a stochastic policy with policy gradient methods.
    This avoids the need for an additional exploration strategy, as we can
    randomly sample from the distribution and thereby explore the environment.
    Through gradient descent better actions are going to be assigned higher
    probabilities while bad actions will become unlikely.`),G.forEach(i),p=w(e),t=f(e,"P",{});var M=v(t);a=$(M,`In Q-learning the action which constitutes the greedy action might change
    due to gradient descent. The change of the policy might be very abrupt and
    might thus destabilize training. In policy gradient methods we apply
    gradient descent to the policy directly, therefore the change of the
    probability distribution of actions is relatively smooth.`),M.forEach(i),m=w(e),x=f(e,"P",{});var Q=v(x);z=$(Q,`The big disadvantage of policy gradient methods is the high variance, but
    adjustments to the naive implementation can be made to decrease the
    variance.`),Q.forEach(i),E=w(e),T=f(e,"DIV",{class:!0}),v(T).forEach(i),this.h()},h(){P(T,"class","separator")},m(e,s){o(e,n,s),_(n,y),o(e,l,s),o(e,d,s),_(d,b),o(e,h,s),o(e,c,s),_(c,r),o(e,p,s),o(e,t,s),_(t,a),o(e,m,s),o(e,x,s),_(x,z),o(e,E,s),o(e,T,s)},p:W,d(e){e&&i(n),e&&i(l),e&&i(d),e&&i(h),e&&i(c),e&&i(p),e&&i(t),e&&i(m),e&&i(x),e&&i(E),e&&i(T)}}}function K(k){let n,y,l,d,b,h,c,r,p;return r=new F({props:{$$slots:{default:[J]},$$scope:{ctx:k}}}),{c(){n=u("meta"),y=g(),l=u("h1"),d=I("Policy Gradient Methods"),b=g(),h=u("div"),c=g(),A(r.$$.fragment),this.h()},l(t){const a=D('[data-svelte="svelte-sp5dwj"]',document.head);n=f(a,"META",{name:!0,content:!0}),a.forEach(i),y=w(t),l=f(t,"H1",{});var m=v(l);d=$(m,"Policy Gradient Methods"),m.forEach(i),b=w(t),h=f(t,"DIV",{class:!0}),v(h).forEach(i),c=w(t),S(r.$$.fragment,t),this.h()},h(){document.title="World4AI | Reinforcement Learning | Policy Gradient Methods",P(n,"name","description"),P(n,"content","Policy gradient methods allow us to estimate and improve the policy directly without the need for a value function. Policy gradient methods are usually more stable than value based methods, but show hight variance."),P(h,"class","separator")},m(t,a){_(document.head,n),o(t,y,a),o(t,l,a),_(l,d),o(t,b,a),o(t,h,a),o(t,c,a),V(r,t,a),p=!0},p(t,[a]){const m={};a&1&&(m.$$scope={dirty:a,ctx:t}),r.$set(m)},i(t){p||(H(r.$$.fragment,t),p=!0)},o(t){L(r.$$.fragment,t),p=!1},d(t){i(n),t&&i(y),t&&i(l),t&&i(b),t&&i(h),t&&i(c),R(r,t)}}}class U extends j{constructor(n){super(),B(this,n,null,K,C,{})}}export{U as default};
