import{S as oe,i as le,s as re,l as N,a as _,r as f,w as P,T as ue,m as T,h as i,c as v,n as U,u as h,x as W,p as R,G as w,b as l,y as M,f as C,t as H,B as L,E as B}from"../../../../chunks/index-caa95cd4.js";import{C as fe}from"../../../../chunks/Container-5c6b7f6d.js";import{L as V}from"../../../../chunks/Latex-bf74aeea.js";function he(c){let n=String.raw`\mathbf{w}^-`+"",a;return{c(){a=f(n)},l(t){a=h(t,n)},m(t,r){l(t,a,r)},p:B,d(t){t&&i(a)}}}function ce(c){let n=String.raw`
r + \gamma \max_{a'} Q(s', a', \mathbf{w}^-)
  `+"",a;return{c(){a=f(n)},l(t){a=h(t,n)},m(t,r){l(t,a,r)},p:B,d(t){t&&i(a)}}}function me(c){let n=String.raw`Q(s, a, \mathbf{w}^-)`+"",a;return{c(){a=f(n)},l(t){a=h(t,n)},m(t,r){l(t,a,r)},p:B,d(t){t&&i(a)}}}function $e(c){let n;return{c(){n=f("a'")},l(a){n=h(a,"a'")},m(a,t){l(a,n,t)},d(a){a&&i(n)}}}function pe(c){let n=String.raw`
    MSE = \mathbb{E}_{(s, a, r, s', t) \sim U(D)}[(r + \gamma Q(s', \arg\max_{a'} Q(s', a', \mathbf{w}), \mathbf{w}^-) - Q(s, a, \mathbf{w}))^2]
  `+"",a;return{c(){a=f(n)},l(t){a=h(t,n)},m(t,r){l(t,a,r)},p:B,d(t){t&&i(a)}}}function we(c){let n=String.raw`Q(s, a, \mathbf{w})`+"",a;return{c(){a=f(n)},l(t){a=h(t,n)},m(t,r){l(t,a,r)},p:B,d(t){t&&i(a)}}}function ge(c){let n=String.raw`Q(s, a, \mathbf{w}^-)`+"",a;return{c(){a=f(n)},l(t){a=h(t,n)},m(t,r){l(t,a,r)},p:B,d(t){t&&i(a)}}}function de(c){let n,a,t,r,b,m,z,$,p,o,u,Q,j,E,F,g,Y,k,Z,S,y,J,I,K,d,ee,q,te,A,ae,X;return u=new V({props:{$$slots:{default:[he]},$$scope:{ctx:c}}}),E=new V({props:{$$slots:{default:[ce]},$$scope:{ctx:c}}}),k=new V({props:{$$slots:{default:[me]},$$scope:{ctx:c}}}),S=new V({props:{$$slots:{default:[$e]},$$scope:{ctx:c}}}),I=new V({props:{$$slots:{default:[pe]},$$scope:{ctx:c}}}),q=new V({props:{$$slots:{default:[we]},$$scope:{ctx:c}}}),A=new V({props:{$$slots:{default:[ge]},$$scope:{ctx:c}}}),{c(){n=N("p"),a=f(`"We first show that the recent DQN algorithm, which combines Q-learning with
    a deep neural network, suffers from substantial overestimations in some
    games in the Atari 2600 domain. We then show that the idea behind the Double
    Q-learning algorithm, which was introduced in a tabular setting, can be
    generalized to work with large-scale function approximation."`),t=_(),r=N("div"),b=_(),m=N("p"),z=f(`The DQN algorithm suffers from a similar maximization bias that is present
    in tabular Q-learning. The output of Q-functions is an estimate that might
    contain some noise. The noise that produces the highest number will be
    preferred in a max operation, even if the true action values are equal. The
    researchers at DeepMind showed that applying double learning to the DQN
    algorithms improves the performance of the agent for Atari games. This gives
    rise to double DQN (DDQN).`),$=_(),p=N("p"),o=f(`In the DQN algorithm the target value is calculated by utilizing the neural
    network with frozen weights `),P(u.$$.fragment),Q=f("."),j=_(),P(E.$$.fragment),F=_(),g=N("p"),Y=f("It is noticeble that the same Q-function "),P(k.$$.fragment),Z=f(" is used to select the next action "),P(S.$$.fragment),y=f(` and to calculate the ation-value.
    This is consistent with the classical definition of Q-learning. In double Q-learning
    two separate acion value functions are used. One is used for action selection
    while the other is used for the calculation of the target value. Using the same
    approach in DQN would not be efficient, as it would require the training of two
    action value function. However the original DQN algorithm already uses two action
    value funcitons: the neural network that is used to estimate the action values
    and the neural network with frozen weights used for bootsrapping. The two functions
    will allow us to to separate action selection and action value calculation.`),J=_(),P(I.$$.fragment),K=_(),d=N("p"),ee=f(`The action in the next state is selected by utilizing the action value
    function `),P(q.$$.fragment),te=f(`, while the
    calculation of the action value is performed using the frozen weights `),P(A.$$.fragment),ae=f("."),this.h()},l(e){n=T(e,"P",{class:!0});var s=U(n);a=h(s,`"We first show that the recent DQN algorithm, which combines Q-learning with
    a deep neural network, suffers from substantial overestimations in some
    games in the Atari 2600 domain. We then show that the idea behind the Double
    Q-learning algorithm, which was introduced in a tabular setting, can be
    generalized to work with large-scale function approximation."`),s.forEach(i),t=v(e),r=T(e,"DIV",{class:!0}),U(r).forEach(i),b=v(e),m=T(e,"P",{});var G=U(m);z=h(G,`The DQN algorithm suffers from a similar maximization bias that is present
    in tabular Q-learning. The output of Q-functions is an estimate that might
    contain some noise. The noise that produces the highest number will be
    preferred in a max operation, even if the true action values are equal. The
    researchers at DeepMind showed that applying double learning to the DQN
    algorithms improves the performance of the agent for Atari games. This gives
    rise to double DQN (DDQN).`),G.forEach(i),$=v(e),p=T(e,"P",{});var O=U(p);o=h(O,`In the DQN algorithm the target value is calculated by utilizing the neural
    network with frozen weights `),W(u.$$.fragment,O),Q=h(O,"."),O.forEach(i),j=v(e),W(E.$$.fragment,e),F=v(e),g=T(e,"P",{});var D=U(g);Y=h(D,"It is noticeble that the same Q-function "),W(k.$$.fragment,D),Z=h(D," is used to select the next action "),W(S.$$.fragment,D),y=h(D,` and to calculate the ation-value.
    This is consistent with the classical definition of Q-learning. In double Q-learning
    two separate acion value functions are used. One is used for action selection
    while the other is used for the calculation of the target value. Using the same
    approach in DQN would not be efficient, as it would require the training of two
    action value function. However the original DQN algorithm already uses two action
    value funcitons: the neural network that is used to estimate the action values
    and the neural network with frozen weights used for bootsrapping. The two functions
    will allow us to to separate action selection and action value calculation.`),D.forEach(i),J=v(e),W(I.$$.fragment,e),K=v(e),d=T(e,"P",{});var x=U(d);ee=h(x,`The action in the next state is selected by utilizing the action value
    function `),W(q.$$.fragment,x),te=h(x,`, while the
    calculation of the action value is performed using the frozen weights `),W(A.$$.fragment,x),ae=h(x,"."),x.forEach(i),this.h()},h(){R(n,"class","info"),R(r,"class","separator")},m(e,s){l(e,n,s),w(n,a),l(e,t,s),l(e,r,s),l(e,b,s),l(e,m,s),w(m,z),l(e,$,s),l(e,p,s),w(p,o),M(u,p,null),w(p,Q),l(e,j,s),M(E,e,s),l(e,F,s),l(e,g,s),w(g,Y),M(k,g,null),w(g,Z),M(S,g,null),w(g,y),l(e,J,s),M(I,e,s),l(e,K,s),l(e,d,s),w(d,ee),M(q,d,null),w(d,te),M(A,d,null),w(d,ae),X=!0},p(e,s){const G={};s&1&&(G.$$scope={dirty:s,ctx:e}),u.$set(G);const O={};s&1&&(O.$$scope={dirty:s,ctx:e}),E.$set(O);const D={};s&1&&(D.$$scope={dirty:s,ctx:e}),k.$set(D);const x={};s&1&&(x.$$scope={dirty:s,ctx:e}),S.$set(x);const ne={};s&1&&(ne.$$scope={dirty:s,ctx:e}),I.$set(ne);const ie={};s&1&&(ie.$$scope={dirty:s,ctx:e}),q.$set(ie);const se={};s&1&&(se.$$scope={dirty:s,ctx:e}),A.$set(se)},i(e){X||(C(u.$$.fragment,e),C(E.$$.fragment,e),C(k.$$.fragment,e),C(S.$$.fragment,e),C(I.$$.fragment,e),C(q.$$.fragment,e),C(A.$$.fragment,e),X=!0)},o(e){H(u.$$.fragment,e),H(E.$$.fragment,e),H(k.$$.fragment,e),H(S.$$.fragment,e),H(I.$$.fragment,e),H(q.$$.fragment,e),H(A.$$.fragment,e),X=!1},d(e){e&&i(n),e&&i(t),e&&i(r),e&&i(b),e&&i(m),e&&i($),e&&i(p),L(u),e&&i(j),L(E,e),e&&i(F),e&&i(g),L(k),L(S),e&&i(J),L(I,e),e&&i(K),e&&i(d),L(q),L(A)}}}function _e(c){let n,a,t,r,b,m,z,$,p;return $=new fe({props:{$$slots:{default:[de]},$$scope:{ctx:c}}}),{c(){n=N("meta"),a=_(),t=N("h1"),r=f("Double DQN"),b=_(),m=N("div"),z=_(),P($.$$.fragment),this.h()},l(o){const u=ue('[data-svelte="svelte-nuh6od"]',document.head);n=T(u,"META",{name:!0,content:!0}),u.forEach(i),a=v(o),t=T(o,"H1",{});var Q=U(t);r=h(Q,"Double DQN"),Q.forEach(i),b=v(o),m=T(o,"DIV",{class:!0}),U(m).forEach(i),z=v(o),W($.$$.fragment,o),this.h()},h(){document.title="World4AI | Reinforcement Learning | Double DQN",R(n,"name","description"),R(n,"content","The double deep Q-network (DQN) is an improvemnt of DQN. Similar to double Q-learning, the agent utilizes two value functions: one for action selection and the other for value calculation. The implementation reduces the overestimation bias."),R(m,"class","separator")},m(o,u){w(document.head,n),l(o,a,u),l(o,t,u),w(t,r),l(o,b,u),l(o,m,u),l(o,z,u),M($,o,u),p=!0},p(o,[u]){const Q={};u&1&&(Q.$$scope={dirty:u,ctx:o}),$.$set(Q)},i(o){p||(C($.$$.fragment,o),p=!0)},o(o){H($.$$.fragment,o),p=!1},d(o){i(n),o&&i(a),o&&i(t),o&&i(b),o&&i(m),o&&i(z),L($,o)}}}class De extends oe{constructor(n){super(),le(this,n,null,_e,re,{})}}export{De as default};
