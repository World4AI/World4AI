import{S as dt,i as vt,s as wt,k as $,a as c,q as r,y as d,W as _t,l as g,h as a,c as u,m as T,r as h,z as v,n as Y,N as m,b as l,A as w,g as _,d as b,B as y,V as $t}from"../chunks/index.4d92b023.js";import{C as bt}from"../chunks/Container.b0705c7b.js";import{H as M}from"../chunks/Highlight.b7c1de53.js";import{A as be}from"../chunks/Alert.25a852b3.js";import{I as yt}from"../chunks/Interaction.6c345f67.js";import{D as Tt}from"../chunks/DeterministicAgent.a1dd8b4b.js";import{G as kt,g as Et,a as gt}from"../chunks/maps.0f079072.js";function It(i){let n;return{c(){n=r("model")},l(t){n=h(t,"model")},m(t,s){l(t,n,s)},d(t){t&&a(n)}}}function Pt(i){let n;return{c(){n=r("value function")},l(t){n=h(t,"value function")},m(t,s){l(t,n,s)},d(t){t&&a(n)}}}function St(i){let n;return{c(){n=r("policy")},l(t){n=h(t,"policy")},m(t,s){l(t,n,s)},d(t){t&&a(n)}}}function xt(i){let n;return{c(){n=r("model")},l(t){n=h(t,"model")},m(t,s){l(t,n,s)},d(t){t&&a(n)}}}function Vt(i){let n;return{c(){n=r(`The environment has one component called model, while the agent might
    contain a value function, a policy and a model.`)},l(t){n=h(t,`The environment has one component called model, while the agent might
    contain a value function, a policy and a model.`)},m(t,s){l(t,n,s)},d(t){t&&a(n)}}}function At(i){let n;return{c(){n=r("transition function")},l(t){n=h(t,"transition function")},m(t,s){l(t,n,s)},d(t){t&&a(n)}}}function Mt(i){let n;return{c(){n=r("reward function")},l(t){n=h(t,"reward function")},m(t,s){l(t,n,s)},d(t){t&&a(n)}}}function Ht(i){let n;return{c(){n=r("The model consists of the transition function and the reward function.")},l(t){n=h(t,"The model consists of the transition function and the reward function.")},m(t,s){l(t,n,s)},d(t){t&&a(n)}}}function Dt(i){let n;return{c(){n=r(`The model of the agent is an approximation of the true model of the
    environment.`)},l(t){n=h(t,`The model of the agent is an approximation of the true model of the
    environment.`)},m(t,s){l(t,n,s)},d(t){t&&a(n)}}}function qt(i){let n;return{c(){n=r("model based")},l(t){n=h(t,"model based")},m(t,s){l(t,n,s)},d(t){t&&a(n)}}}function Ft(i){let n;return{c(){n=r("model free")},l(t){n=h(t,"model free")},m(t,s){l(t,n,s)},d(t){t&&a(n)}}}function zt(i){let n;return{c(){n=r("The policy of the agent maps states to actions.")},l(t){n=h(t,"The policy of the agent maps states to actions.")},m(t,s){l(t,n,s)},d(t){t&&a(n)}}}function Ct(i){let n;return{c(){n=r("The value function of the agent maps states to values.")},l(t){n=h(t,"The value function of the agent maps states to values.")},m(t,s){l(t,n,s)},d(t){t&&a(n)}}}function Wt(i){let n;return{c(){n=r("value based methods")},l(t){n=h(t,"value based methods")},m(t,s){l(t,n,s)},d(t){t&&a(n)}}}function Gt(i){let n;return{c(){n=r("policy based methods")},l(t){n=h(t,"policy based methods")},m(t,s){l(t,n,s)},d(t){t&&a(n)}}}function jt(i){let n;return{c(){n=r("actor-critic methods")},l(t){n=h(t,"actor-critic methods")},m(t,s){l(t,n,s)},d(t){t&&a(n)}}}function Rt(i){let n,t,s,H,E,S,k,x,P,f,p,I,C,Z,Be,ye,ee,Te,te,Ne,ke,D,Oe,W,Je,G,Ke,Ee,j,Ie,ne,Le,Pe,R,Se,q,Qe,U,Xe,B,Ye,xe,oe,Ve,ae,Ze,Ae,N,Me,le,et,He,O,De,se,tt,qe,ie,Fe,re,nt,ze,J,Ce,he,ot,We,fe,at,Ge,K,je,V,lt,L,st,Q,it,X,rt,Re,me,Ue;return s=new M({props:{$$slots:{default:[It]},$$scope:{ctx:i}}}),E=new M({props:{$$slots:{default:[Pt]},$$scope:{ctx:i}}}),k=new M({props:{$$slots:{default:[St]},$$scope:{ctx:i}}}),P=new M({props:{$$slots:{default:[xt]},$$scope:{ctx:i}}}),I=new be({props:{type:"info",$$slots:{default:[Vt]},$$scope:{ctx:i}}}),W=new M({props:{$$slots:{default:[At]},$$scope:{ctx:i}}}),G=new M({props:{$$slots:{default:[Mt]},$$scope:{ctx:i}}}),j=new be({props:{type:"info",$$slots:{default:[Ht]},$$scope:{ctx:i}}}),R=new be({props:{type:"info",$$slots:{default:[Dt]},$$scope:{ctx:i}}}),U=new M({props:{$$slots:{default:[qt]},$$scope:{ctx:i}}}),B=new M({props:{$$slots:{default:[Ft]},$$scope:{ctx:i}}}),N=new be({props:{type:"info",$$slots:{default:[zt]},$$scope:{ctx:i}}}),O=new gt({props:{player:i[0],cells:i[1],policy:i[4]}}),J=new be({props:{type:"info",$$slots:{default:[Ct]},$$scope:{ctx:i}}}),K=new gt({props:{player:i[0],cells:i[1],showColoredValues:!0}}),L=new M({props:{$$slots:{default:[Wt]},$$scope:{ctx:i}}}),Q=new M({props:{$$slots:{default:[Gt]},$$scope:{ctx:i}}}),X=new M({props:{$$slots:{default:[jt]},$$scope:{ctx:i}}}),{c(){n=$("p"),t=r(`In order for the agent to determine the next action and for the environment
    to calculate the next state and the corresponding reward, the two make use
    of their respective internal components. The environment utilizes the `),d(s.$$.fragment),H=r(`, while the agent might use up to three components: the
    `),d(E.$$.fragment),S=r(`, the
    `),d(k.$$.fragment),x=r(`
    and the
    `),d(P.$$.fragment),f=r("."),p=c(),d(I.$$.fragment),C=c(),Z=$("p"),Be=r(`The agent only requires the policy to work, nevertheless the model and the
    value function are major parts of many modern reinforcement learning
    algorithms, as these additional components can make the agent a lot more
    competent at solving a task.`),ye=c(),ee=$("div"),Te=c(),te=$("h2"),Ne=r("The Model"),ke=c(),D=$("p"),Oe=r(`The model is the only component of the environment. The model takes the
    current state and the action chosen by the agent as inputs and outputs the
    next state and the reward. Usually it consists of two distnict functions:
    the `),d(W.$$.fragment),Je=r(` calculates the next state, while
    the `),d(G.$$.fragment),Ke=r(" calculates the corresponding reward."),Ee=c(),d(j.$$.fragment),Ie=c(),ne=$("p"),Le=r(`How exactly the model looks depends on the environment. Sometimes a simple
    table is all that is required. For our gridworld with 25 possible states and
    4 possible actions a table with 25 rows and 4 columns could be used to
    represent the model. The inner cells at the intersection between the current
    state and the action would contain the corresponding probabilities to
    transition into the next state and the reward. More complex environments
    like the Atari games would implement the model using a game engine.`),Pe=c(),d(R.$$.fragment),Se=c(),q=$("p"),Qe=r(`In reinforcement learning the model of the environment is usually not
    something that the agent has access to. The agent has to learn to navigate
    in an environment where the rules of the game are not known. The agent can
    theoretically learn about the model by interacting with the environment.
    Essentially the agent creates some sort of an approximation of the true
    model of the environment. Each interaction allows the agent to improve its
    knowledge. Algorithms that implement a model for the agent are called `),d(U.$$.fragment),Xe=r(`, otherwise the algorithms are called
    `),d(B.$$.fragment),Ye=r("."),xe=c(),oe=$("div"),Ve=c(),ae=$("h2"),Ze=r("Policy"),Ae=c(),d(N.$$.fragment),Me=c(),le=$("p"),et=r(`The policy calculates the action based the current state of the environment.
    For very simple environments the policy function might be simple a table
    that contains all possible states and the corresponding action for that
    state. The policy of the 5x5 grid world we used so far would also be
    contained in a mapping table, where the corresponding optimal policy might
    look as follows.`),He=c(),d(O.$$.fragment),De=c(),se=$("p"),tt=r(`In more complex environments it is not possible to construct a mapping
    table, as the number of possible states is extremely high. In that case
    other solutions like neural networks are used.`),qe=c(),ie=$("div"),Fe=c(),re=$("h2"),nt=r("Value Function"),ze=c(),d(J.$$.fragment),Ce=c(),he=$("p"),ot=r(`The value function gets a state as an input and generates a single scalar
    value. The value function plays an important role in most state of the art
    reinforcement learning algorithms. Intuitively speaking the agent looks at
    the state of the environment and assigns a value of "goodness" to the state.
    The higher the value, the better the state. With the help of the value
    function the agent tries to locate and move towards better and better
    states.`),We=c(),fe=$("p"),at=r(`Similar to the policy for simple environments the value function can be
    calculated with the help of a table or in more complex environments using a
    neural network. The grid world example below shows color coded values in the
    grid world environment. The orange value in the top left corner is the
    farthers away from the goal. The blue value in the bottom left corner is the
    goal that provides a positive reward. The colors inbetween are interpolated
    based on the distance from the goal. The closer the agent is to the goal the
    higher the values are expected to be. Therefore the agent could
    theoretically look around the current state and look for states with more
    "blueish" values and move into that direction to arrive at the goal.`),Ge=c(),d(K.$$.fragment),je=c(),V=$("p"),lt=r(`Especially beginner level reinforcement learning agents have only a value
    function. In that case the policy of the agent is implicitly derived from
    the value function. Reinforcement learning algorithms that only utilize a
    value function are called `),d(L.$$.fragment),st=r(`. If on
    the other hand the agent derives the policy directly without using value
    functions the methods are called `),d(Q.$$.fragment),it=r(`. Most modern algorithms have agents with both components. Those are called
    `),d(X.$$.fragment),rt=r("."),Re=c(),me=$("div"),this.h()},l(e){n=g(e,"P",{});var o=T(n);t=h(o,`In order for the agent to determine the next action and for the environment
    to calculate the next state and the corresponding reward, the two make use
    of their respective internal components. The environment utilizes the `),v(s.$$.fragment,o),H=h(o,`, while the agent might use up to three components: the
    `),v(E.$$.fragment,o),S=h(o,`, the
    `),v(k.$$.fragment,o),x=h(o,`
    and the
    `),v(P.$$.fragment,o),f=h(o,"."),o.forEach(a),p=u(e),v(I.$$.fragment,e),C=u(e),Z=g(e,"P",{});var ce=T(Z);Be=h(ce,`The agent only requires the policy to work, nevertheless the model and the
    value function are major parts of many modern reinforcement learning
    algorithms, as these additional components can make the agent a lot more
    competent at solving a task.`),ce.forEach(a),ye=u(e),ee=g(e,"DIV",{class:!0}),T(ee).forEach(a),Te=u(e),te=g(e,"H2",{});var ue=T(te);Ne=h(ue,"The Model"),ue.forEach(a),ke=u(e),D=g(e,"P",{});var F=T(D);Oe=h(F,`The model is the only component of the environment. The model takes the
    current state and the action chosen by the agent as inputs and outputs the
    next state and the reward. Usually it consists of two distnict functions:
    the `),v(W.$$.fragment,F),Je=h(F,` calculates the next state, while
    the `),v(G.$$.fragment,F),Ke=h(F," calculates the corresponding reward."),F.forEach(a),Ee=u(e),v(j.$$.fragment,e),Ie=u(e),ne=g(e,"P",{});var pe=T(ne);Le=h(pe,`How exactly the model looks depends on the environment. Sometimes a simple
    table is all that is required. For our gridworld with 25 possible states and
    4 possible actions a table with 25 rows and 4 columns could be used to
    represent the model. The inner cells at the intersection between the current
    state and the action would contain the corresponding probabilities to
    transition into the next state and the reward. More complex environments
    like the Atari games would implement the model using a game engine.`),pe.forEach(a),Pe=u(e),v(R.$$.fragment,e),Se=u(e),q=g(e,"P",{});var z=T(q);Qe=h(z,`In reinforcement learning the model of the environment is usually not
    something that the agent has access to. The agent has to learn to navigate
    in an environment where the rules of the game are not known. The agent can
    theoretically learn about the model by interacting with the environment.
    Essentially the agent creates some sort of an approximation of the true
    model of the environment. Each interaction allows the agent to improve its
    knowledge. Algorithms that implement a model for the agent are called `),v(U.$$.fragment,z),Xe=h(z,`, otherwise the algorithms are called
    `),v(B.$$.fragment,z),Ye=h(z,"."),z.forEach(a),xe=u(e),oe=g(e,"DIV",{class:!0}),T(oe).forEach(a),Ve=u(e),ae=g(e,"H2",{});var $e=T(ae);Ze=h($e,"Policy"),$e.forEach(a),Ae=u(e),v(N.$$.fragment,e),Me=u(e),le=g(e,"P",{});var ge=T(le);et=h(ge,`The policy calculates the action based the current state of the environment.
    For very simple environments the policy function might be simple a table
    that contains all possible states and the corresponding action for that
    state. The policy of the 5x5 grid world we used so far would also be
    contained in a mapping table, where the corresponding optimal policy might
    look as follows.`),ge.forEach(a),He=u(e),v(O.$$.fragment,e),De=u(e),se=g(e,"P",{});var de=T(se);tt=h(de,`In more complex environments it is not possible to construct a mapping
    table, as the number of possible states is extremely high. In that case
    other solutions like neural networks are used.`),de.forEach(a),qe=u(e),ie=g(e,"DIV",{class:!0}),T(ie).forEach(a),Fe=u(e),re=g(e,"H2",{});var ve=T(re);nt=h(ve,"Value Function"),ve.forEach(a),ze=u(e),v(J.$$.fragment,e),Ce=u(e),he=g(e,"P",{});var we=T(he);ot=h(we,`The value function gets a state as an input and generates a single scalar
    value. The value function plays an important role in most state of the art
    reinforcement learning algorithms. Intuitively speaking the agent looks at
    the state of the environment and assigns a value of "goodness" to the state.
    The higher the value, the better the state. With the help of the value
    function the agent tries to locate and move towards better and better
    states.`),we.forEach(a),We=u(e),fe=g(e,"P",{});var _e=T(fe);at=h(_e,`Similar to the policy for simple environments the value function can be
    calculated with the help of a table or in more complex environments using a
    neural network. The grid world example below shows color coded values in the
    grid world environment. The orange value in the top left corner is the
    farthers away from the goal. The blue value in the bottom left corner is the
    goal that provides a positive reward. The colors inbetween are interpolated
    based on the distance from the goal. The closer the agent is to the goal the
    higher the values are expected to be. Therefore the agent could
    theoretically look around the current state and look for states with more
    "blueish" values and move into that direction to arrive at the goal.`),_e.forEach(a),Ge=u(e),v(K.$$.fragment,e),je=u(e),V=g(e,"P",{});var A=T(V);lt=h(A,`Especially beginner level reinforcement learning agents have only a value
    function. In that case the policy of the agent is implicitly derived from
    the value function. Reinforcement learning algorithms that only utilize a
    value function are called `),v(L.$$.fragment,A),st=h(A,`. If on
    the other hand the agent derives the policy directly without using value
    functions the methods are called `),v(Q.$$.fragment,A),it=h(A,`. Most modern algorithms have agents with both components. Those are called
    `),v(X.$$.fragment,A),rt=h(A,"."),A.forEach(a),Re=u(e),me=g(e,"DIV",{class:!0}),T(me).forEach(a),this.h()},h(){Y(ee,"class","separator"),Y(oe,"class","separator"),Y(ie,"class","separator"),Y(me,"class","separator")},m(e,o){l(e,n,o),m(n,t),w(s,n,null),m(n,H),w(E,n,null),m(n,S),w(k,n,null),m(n,x),w(P,n,null),m(n,f),l(e,p,o),w(I,e,o),l(e,C,o),l(e,Z,o),m(Z,Be),l(e,ye,o),l(e,ee,o),l(e,Te,o),l(e,te,o),m(te,Ne),l(e,ke,o),l(e,D,o),m(D,Oe),w(W,D,null),m(D,Je),w(G,D,null),m(D,Ke),l(e,Ee,o),w(j,e,o),l(e,Ie,o),l(e,ne,o),m(ne,Le),l(e,Pe,o),w(R,e,o),l(e,Se,o),l(e,q,o),m(q,Qe),w(U,q,null),m(q,Xe),w(B,q,null),m(q,Ye),l(e,xe,o),l(e,oe,o),l(e,Ve,o),l(e,ae,o),m(ae,Ze),l(e,Ae,o),w(N,e,o),l(e,Me,o),l(e,le,o),m(le,et),l(e,He,o),w(O,e,o),l(e,De,o),l(e,se,o),m(se,tt),l(e,qe,o),l(e,ie,o),l(e,Fe,o),l(e,re,o),m(re,nt),l(e,ze,o),w(J,e,o),l(e,Ce,o),l(e,he,o),m(he,ot),l(e,We,o),l(e,fe,o),m(fe,at),l(e,Ge,o),w(K,e,o),l(e,je,o),l(e,V,o),m(V,lt),w(L,V,null),m(V,st),w(Q,V,null),m(V,it),w(X,V,null),m(V,rt),l(e,Re,o),l(e,me,o),Ue=!0},p(e,o){const ce={};o&1024&&(ce.$$scope={dirty:o,ctx:e}),s.$set(ce);const ue={};o&1024&&(ue.$$scope={dirty:o,ctx:e}),E.$set(ue);const F={};o&1024&&(F.$$scope={dirty:o,ctx:e}),k.$set(F);const pe={};o&1024&&(pe.$$scope={dirty:o,ctx:e}),P.$set(pe);const z={};o&1024&&(z.$$scope={dirty:o,ctx:e}),I.$set(z);const $e={};o&1024&&($e.$$scope={dirty:o,ctx:e}),W.$set($e);const ge={};o&1024&&(ge.$$scope={dirty:o,ctx:e}),G.$set(ge);const de={};o&1024&&(de.$$scope={dirty:o,ctx:e}),j.$set(de);const ve={};o&1024&&(ve.$$scope={dirty:o,ctx:e}),R.$set(ve);const we={};o&1024&&(we.$$scope={dirty:o,ctx:e}),U.$set(we);const _e={};o&1024&&(_e.$$scope={dirty:o,ctx:e}),B.$set(_e);const A={};o&1024&&(A.$$scope={dirty:o,ctx:e}),N.$set(A);const ht={};o&1&&(ht.player=e[0]),o&2&&(ht.cells=e[1]),O.$set(ht);const mt={};o&1024&&(mt.$$scope={dirty:o,ctx:e}),J.$set(mt);const ft={};o&1&&(ft.player=e[0]),o&2&&(ft.cells=e[1]),K.$set(ft);const ct={};o&1024&&(ct.$$scope={dirty:o,ctx:e}),L.$set(ct);const ut={};o&1024&&(ut.$$scope={dirty:o,ctx:e}),Q.$set(ut);const pt={};o&1024&&(pt.$$scope={dirty:o,ctx:e}),X.$set(pt)},i(e){Ue||(_(s.$$.fragment,e),_(E.$$.fragment,e),_(k.$$.fragment,e),_(P.$$.fragment,e),_(I.$$.fragment,e),_(W.$$.fragment,e),_(G.$$.fragment,e),_(j.$$.fragment,e),_(R.$$.fragment,e),_(U.$$.fragment,e),_(B.$$.fragment,e),_(N.$$.fragment,e),_(O.$$.fragment,e),_(J.$$.fragment,e),_(K.$$.fragment,e),_(L.$$.fragment,e),_(Q.$$.fragment,e),_(X.$$.fragment,e),Ue=!0)},o(e){b(s.$$.fragment,e),b(E.$$.fragment,e),b(k.$$.fragment,e),b(P.$$.fragment,e),b(I.$$.fragment,e),b(W.$$.fragment,e),b(G.$$.fragment,e),b(j.$$.fragment,e),b(R.$$.fragment,e),b(U.$$.fragment,e),b(B.$$.fragment,e),b(N.$$.fragment,e),b(O.$$.fragment,e),b(J.$$.fragment,e),b(K.$$.fragment,e),b(L.$$.fragment,e),b(Q.$$.fragment,e),b(X.$$.fragment,e),Ue=!1},d(e){e&&a(n),y(s),y(E),y(k),y(P),e&&a(p),y(I,e),e&&a(C),e&&a(Z),e&&a(ye),e&&a(ee),e&&a(Te),e&&a(te),e&&a(ke),e&&a(D),y(W),y(G),e&&a(Ee),y(j,e),e&&a(Ie),e&&a(ne),e&&a(Pe),y(R,e),e&&a(Se),e&&a(q),y(U),y(B),e&&a(xe),e&&a(oe),e&&a(Ve),e&&a(ae),e&&a(Ae),y(N,e),e&&a(Me),e&&a(le),e&&a(He),y(O,e),e&&a(De),e&&a(se),e&&a(qe),e&&a(ie),e&&a(Fe),e&&a(re),e&&a(ze),y(J,e),e&&a(Ce),e&&a(he),e&&a(We),e&&a(fe),e&&a(Ge),y(K,e),e&&a(je),e&&a(V),y(L),y(Q),y(X),e&&a(Re),e&&a(me)}}}function Ut(i){let n,t,s,H,E,S,k,x,P;return x=new bt({props:{$$slots:{default:[Rt]},$$scope:{ctx:i}}}),{c(){n=$("meta"),t=c(),s=$("h1"),H=r("Value, Policy, Model"),E=c(),S=$("div"),k=c(),d(x.$$.fragment),this.h()},l(f){const p=_t("svelte-1fvxist",document.head);n=g(p,"META",{name:!0,content:!0}),p.forEach(a),t=u(f),s=g(f,"H1",{});var I=T(s);H=h(I,"Value, Policy, Model"),I.forEach(a),E=u(f),S=g(f,"DIV",{class:!0}),T(S).forEach(a),k=u(f),v(x.$$.fragment,f),this.h()},h(){document.title="Value, Policy, Model - World4AI",Y(n,"name","description"),Y(n,"content","In reinforcement learning the value function, the policy and the model are essential components of agent. The environment on the other hand has a single component, the model."),Y(S,"class","separator")},m(f,p){m(document.head,n),l(f,t,p),l(f,s,p),m(s,H),l(f,E,p),l(f,S,p),l(f,k,p),w(x,f,p),P=!0},p(f,[p]){const I={};p&1027&&(I.$$scope={dirty:p,ctx:f}),x.$set(I)},i(f){P||(_(x.$$.fragment,f),P=!0)},o(f){b(x.$$.fragment,f),P=!1},d(f){a(n),f&&a(t),f&&a(s),f&&a(E),f&&a(S),f&&a(k),y(x,f)}}}function Bt(i,n,t){let s,H,E,S,k=new kt(Et),x=new Tt(k.getObservationSpace(),k.getActionSpace()),P=new yt(x,k,2),f=k.cellsStore;$t(i,f,C=>t(6,S=C));let p=P.observationStore;$t(i,p,C=>t(5,E=C));let I={0:{0:2,1:2,2:2,3:2,4:2},1:{0:1,1:1,2:1,3:2,4:2},2:{0:0,1:0,2:0,3:2,4:2},3:{0:2,1:3,2:3,3:3,4:2},4:{0:0,1:3,2:3,3:3,4:3}};return i.$$.update=()=>{i.$$.dirty&64&&t(1,s=S),i.$$.dirty&32&&t(0,H=E)},[H,s,f,p,I,E,S]}class Yt extends dt{constructor(n){super(),vt(this,n,Bt,Ut,wt,{})}}export{Yt as default};
