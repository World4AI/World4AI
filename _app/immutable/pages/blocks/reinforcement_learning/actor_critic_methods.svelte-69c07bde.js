import{S as pt,i as _t,s as dt,l as q,a as R,r,w as g,T as gt,m as V,h as i,c as T,n as M,u as c,x as v,p as Z,G as $,b as l,y as w,f as b,t as E,B as y,E as L}from"../../../chunks/index-caa95cd4.js";import{C as vt}from"../../../chunks/Container-5c6b7f6d.js";import{L as I}from"../../../chunks/Latex-bf74aeea.js";function wt(f){let n;return{c(){n=r("\\pi")},l(a){n=c(a,"\\pi")},m(a,e){l(a,n,e)},d(a){a&&i(n)}}}function bt(f){let n;return{c(){n=r("V")},l(a){n=c(a,"V")},m(a,e){l(a,n,e)},d(a){a&&i(n)}}}function Et(f){let n=String.raw`
 \nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}\Big[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) [R(\tau)_{t:H} - V_w(S_t)]\Big] \\
`+"",a;return{c(){a=r(n)},l(e){a=c(e,n)},m(e,u){l(e,a,u)},p:L,d(e){e&&i(a)}}}function yt(f){let n=String.raw`\pi_{\theta}`+"",a;return{c(){a=r(n)},l(e){a=c(e,n)},m(e,u){l(e,a,u)},p:L,d(e){e&&i(a)}}}function xt(f){let n=String.raw`V_w`+"",a;return{c(){a=r(n)},l(e){a=c(e,n)},m(e,u){l(e,a,u)},p:L,d(e){e&&i(a)}}}function St(f){let n=String.raw`
 \nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}}\Big[\sum_t^H \nabla_{\theta} \log \pi_{\theta}(A_t \mid S_t) [R_{t+1} + V_w(S_{t+1}) - V_w(S_t)]\Big] 
`+"",a;return{c(){a=r(n)},l(e){a=c(e,n)},m(e,u){l(e,a,u)},p:L,d(e){e&&i(a)}}}function kt(f){let n=String.raw`S_{t + 1}`+"",a;return{c(){a=r(n)},l(e){a=c(e,n)},m(e,u){l(e,a,u)},p:L,d(e){e&&i(a)}}}function It(f){let n=String.raw`V(S_{t + 1})`+"",a;return{c(){a=r(n)},l(e){a=c(e,n)},m(e,u){l(e,a,u)},p:L,d(e){e&&i(a)}}}function Rt(f){let n=String.raw`S_{t+1}`+"",a;return{c(){a=r(n)},l(e){a=c(e,n)},m(e,u){l(e,a,u)},p:L,d(e){e&&i(a)}}}function Tt(f){let n=String.raw`A_t`+"",a;return{c(){a=r(n)},l(e){a=c(e,n)},m(e,u){l(e,a,u)},p:L,d(e){e&&i(a)}}}function Ct(f){let n,a,e,u,_,x,C,m,N,o,h,S,k,tt,W,et,F,at,z,O,K,p,nt,B,st,P,it,D,ot,H,lt,U,J,rt,X,Q,Y;return e=new I({props:{$$slots:{default:[wt]},$$scope:{ctx:f}}}),_=new I({props:{$$slots:{default:[bt]},$$scope:{ctx:f}}}),h=new I({props:{$$slots:{default:[Et]},$$scope:{ctx:f}}}),W=new I({props:{$$slots:{default:[yt]},$$scope:{ctx:f}}}),F=new I({props:{$$slots:{default:[xt]},$$scope:{ctx:f}}}),O=new I({props:{$$slots:{default:[St]},$$scope:{ctx:f}}}),B=new I({props:{$$slots:{default:[kt]},$$scope:{ctx:f}}}),P=new I({props:{$$slots:{default:[It]},$$scope:{ctx:f}}}),D=new I({props:{$$slots:{default:[Rt]},$$scope:{ctx:f}}}),H=new I({props:{$$slots:{default:[Tt]},$$scope:{ctx:f}}}),{c(){n=q("p"),a=r(`So far we have studied value based methods like DQN, which estimate state or
    action value functions and determine the policy implicitly. We additionally
    derived policy based methods like REINFORCE, which estimate the policy
    directly. It turns out that combining both types of methods can result in
    so-called actor-critic methods. The actor is the decision maker and the
    policy `),g(e.$$.fragment),u=r(`
    of the agent. The critic is the value function `),g(_.$$.fragment),x=r(` that estimates
    how good or bad the decisions are that the actor makes. We will implement both
    functions as neural networks and train them simultaneously. The actor-critic
    methods can have significant improvements over pure value or policy based methods
    and in many cases constitute state of the art methods.`),C=R(),m=q("p"),N=r(`At this point in time a natural question might occur. Is the REINFORCE
    algorithm with baseline (vpg) an actor-critic algorithm?`),o=R(),g(h.$$.fragment),S=R(),k=q("p"),tt=r("REINFORCE with baseline has a policy "),g(W.$$.fragment),et=r(`
    and a value function `),g(F.$$.fragment),at=r(`. Should that be
    sufficient to classify the algorithm as actor-critic? It turns out that not
    all agents that have separate policy and value functions are defined as
    actor-critic methods. The key component that is required is bootstrapping.`),z=R(),g(O.$$.fragment),K=R(),p=q("p"),nt=r("The next state "),g(B.$$.fragment),st=r(` that results from the action
    of the actor has to be evaluated by the value function, the critic. When we calculate
    `),g(P.$$.fragment),it=r(` we ask the critic to calculate the
    expected value of the next state`),g(D.$$.fragment),ot=r(` that resulted
    from the actor taking an action `),g(H.$$.fragment),lt=r(`. That way
    the critic essentially "critiques" the previous action of the actor.`),U=R(),J=q("p"),rt=r(`In the following sections we will discuss some of the actor-critic
    algorithms. Especially we will focus on so called advantage actor-critic
    methods.`),X=R(),Q=q("div"),this.h()},l(t){n=V(t,"P",{});var s=M(n);a=c(s,`So far we have studied value based methods like DQN, which estimate state or
    action value functions and determine the policy implicitly. We additionally
    derived policy based methods like REINFORCE, which estimate the policy
    directly. It turns out that combining both types of methods can result in
    so-called actor-critic methods. The actor is the decision maker and the
    policy `),v(e.$$.fragment,s),u=c(s,`
    of the agent. The critic is the value function `),v(_.$$.fragment,s),x=c(s,` that estimates
    how good or bad the decisions are that the actor makes. We will implement both
    functions as neural networks and train them simultaneously. The actor-critic
    methods can have significant improvements over pure value or policy based methods
    and in many cases constitute state of the art methods.`),s.forEach(i),C=T(t),m=V(t,"P",{});var G=M(m);N=c(G,`At this point in time a natural question might occur. Is the REINFORCE
    algorithm with baseline (vpg) an actor-critic algorithm?`),G.forEach(i),o=T(t),v(h.$$.fragment,t),S=T(t),k=V(t,"P",{});var A=M(k);tt=c(A,"REINFORCE with baseline has a policy "),v(W.$$.fragment,A),et=c(A,`
    and a value function `),v(F.$$.fragment,A),at=c(A,`. Should that be
    sufficient to classify the algorithm as actor-critic? It turns out that not
    all agents that have separate policy and value functions are defined as
    actor-critic methods. The key component that is required is bootstrapping.`),A.forEach(i),z=T(t),v(O.$$.fragment,t),K=T(t),p=V(t,"P",{});var d=M(p);nt=c(d,"The next state "),v(B.$$.fragment,d),st=c(d,` that results from the action
    of the actor has to be evaluated by the value function, the critic. When we calculate
    `),v(P.$$.fragment,d),it=c(d,` we ask the critic to calculate the
    expected value of the next state`),v(D.$$.fragment,d),ot=c(d,` that resulted
    from the actor taking an action `),v(H.$$.fragment,d),lt=c(d,`. That way
    the critic essentially "critiques" the previous action of the actor.`),d.forEach(i),U=T(t),J=V(t,"P",{});var j=M(J);rt=c(j,`In the following sections we will discuss some of the actor-critic
    algorithms. Especially we will focus on so called advantage actor-critic
    methods.`),j.forEach(i),X=T(t),Q=V(t,"DIV",{class:!0}),M(Q).forEach(i),this.h()},h(){Z(Q,"class","separator")},m(t,s){l(t,n,s),$(n,a),w(e,n,null),$(n,u),w(_,n,null),$(n,x),l(t,C,s),l(t,m,s),$(m,N),l(t,o,s),w(h,t,s),l(t,S,s),l(t,k,s),$(k,tt),w(W,k,null),$(k,et),w(F,k,null),$(k,at),l(t,z,s),w(O,t,s),l(t,K,s),l(t,p,s),$(p,nt),w(B,p,null),$(p,st),w(P,p,null),$(p,it),w(D,p,null),$(p,ot),w(H,p,null),$(p,lt),l(t,U,s),l(t,J,s),$(J,rt),l(t,X,s),l(t,Q,s),Y=!0},p(t,s){const G={};s&1&&(G.$$scope={dirty:s,ctx:t}),e.$set(G);const A={};s&1&&(A.$$scope={dirty:s,ctx:t}),_.$set(A);const d={};s&1&&(d.$$scope={dirty:s,ctx:t}),h.$set(d);const j={};s&1&&(j.$$scope={dirty:s,ctx:t}),W.$set(j);const ct={};s&1&&(ct.$$scope={dirty:s,ctx:t}),F.$set(ct);const ft={};s&1&&(ft.$$scope={dirty:s,ctx:t}),O.$set(ft);const ut={};s&1&&(ut.$$scope={dirty:s,ctx:t}),B.$set(ut);const ht={};s&1&&(ht.$$scope={dirty:s,ctx:t}),P.$set(ht);const mt={};s&1&&(mt.$$scope={dirty:s,ctx:t}),D.$set(mt);const $t={};s&1&&($t.$$scope={dirty:s,ctx:t}),H.$set($t)},i(t){Y||(b(e.$$.fragment,t),b(_.$$.fragment,t),b(h.$$.fragment,t),b(W.$$.fragment,t),b(F.$$.fragment,t),b(O.$$.fragment,t),b(B.$$.fragment,t),b(P.$$.fragment,t),b(D.$$.fragment,t),b(H.$$.fragment,t),Y=!0)},o(t){E(e.$$.fragment,t),E(_.$$.fragment,t),E(h.$$.fragment,t),E(W.$$.fragment,t),E(F.$$.fragment,t),E(O.$$.fragment,t),E(B.$$.fragment,t),E(P.$$.fragment,t),E(D.$$.fragment,t),E(H.$$.fragment,t),Y=!1},d(t){t&&i(n),y(e),y(_),t&&i(C),t&&i(m),t&&i(o),y(h,t),t&&i(S),t&&i(k),y(W),y(F),t&&i(z),y(O,t),t&&i(K),t&&i(p),y(B),y(P),y(D),y(H),t&&i(U),t&&i(J),t&&i(X),t&&i(Q)}}}function At(f){let n,a,e,u,_,x,C,m,N;return m=new vt({props:{$$slots:{default:[Ct]},$$scope:{ctx:f}}}),{c(){n=q("meta"),a=R(),e=q("h1"),u=r("Actor Critic Methods"),_=R(),x=q("div"),C=R(),g(m.$$.fragment),this.h()},l(o){const h=gt('[data-svelte="svelte-1khpp3y"]',document.head);n=V(h,"META",{name:!0,content:!0}),h.forEach(i),a=T(o),e=V(o,"H1",{});var S=M(e);u=c(S,"Actor Critic Methods"),S.forEach(i),_=T(o),x=V(o,"DIV",{class:!0}),M(x).forEach(i),C=T(o),v(m.$$.fragment,o),this.h()},h(){document.title="World4AI | Reinforcement Learning | Actor-Critic Methods",Z(n,"name","description"),Z(n,"content","Actor-Critic methods combine value based and policy based algorithms. The actor is the decision maker, the policy and the critique is the bootstrapped value function that evaluates the actions of the actor."),Z(x,"class","separator")},m(o,h){$(document.head,n),l(o,a,h),l(o,e,h),$(e,u),l(o,_,h),l(o,x,h),l(o,C,h),w(m,o,h),N=!0},p(o,[h]){const S={};h&1&&(S.$$scope={dirty:h,ctx:o}),m.$set(S)},i(o){N||(b(m.$$.fragment,o),N=!0)},o(o){E(m.$$.fragment,o),N=!1},d(o){i(n),o&&i(a),o&&i(e),o&&i(_),o&&i(x),o&&i(C),y(m,o)}}}class Wt extends pt{constructor(n){super(),_t(this,n,null,At,dt,{})}}export{Wt as default};
