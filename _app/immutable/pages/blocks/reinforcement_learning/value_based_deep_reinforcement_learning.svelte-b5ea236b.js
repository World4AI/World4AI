import{S as j,i as M,s as P,l as b,a as D,r as N,w as z,T as C,m as _,h as l,c as $,n as y,u as E,x as V,p as Q,G as g,b as v,y as T,f as A,t as B,B as S,R as L,O as x,E as O}from"../../../chunks/index-caa95cd4.js";import{C as W}from"../../../chunks/Container-5c6b7f6d.js";function I(d,t,r){const e=d.slice();return e[1]=t[r],e[3]=r,e}function q(d){let t,r,e,o=d[1].time+"",h,n,p,f=d[1].content+"",u,a;return{c(){t=b("div"),r=b("div"),e=b("h6"),h=N(o),n=D(),p=b("p"),u=N(f),a=D(),this.h()},l(s){t=_(s,"DIV",{class:!0});var c=y(t);r=_(c,"DIV",{class:!0});var w=y(r);e=_(w,"H6",{class:!0});var i=y(e);h=E(i,o),i.forEach(l),n=$(w),p=_(w,"P",{});var m=y(p);u=E(m,f),m.forEach(l),w.forEach(l),a=$(c),c.forEach(l),this.h()},h(){Q(e,"class","title svelte-1e3qbp5"),Q(r,"class","content"),Q(t,"class","container svelte-1e3qbp5"),x(t,"left",d[3]%2===0),x(t,"right",d[3]%2!=0)},m(s,c){v(s,t,c),g(t,r),g(r,e),g(e,h),g(r,n),g(r,p),g(p,u),g(t,a)},p:O,d(s){s&&l(t)}}}function F(d){let t,r=d[0],e=[];for(let o=0;o<r.length;o+=1)e[o]=q(I(d,r,o));return{c(){t=b("div");for(let o=0;o<e.length;o+=1)e[o].c();this.h()},l(o){t=_(o,"DIV",{class:!0});var h=y(t);for(let n=0;n<e.length;n+=1)e[n].l(h);h.forEach(l),this.h()},h(){Q(t,"class","timeline svelte-1e3qbp5")},m(o,h){v(o,t,h);for(let n=0;n<e.length;n+=1)e[n].m(t,null)},p(o,h){if(h&1){r=o[0];let n;for(n=0;n<r.length;n+=1){const p=I(o,r,n);e[n]?e[n].p(p,h):(e[n]=q(p),e[n].c(),e[n].m(t,null))}for(;n<e.length;n+=1)e[n].d(1);e.length=r.length}},d(o){o&&l(t),L(e,o)}}}function H(d){let t,r,e,o,h,n,p,f,u,a,s,c,w;return o=new W({props:{maxWidth:"300px",$$slots:{default:[F]},$$scope:{ctx:d}}}),{c(){t=b("p"),r=N(`The naive implementation of approximate value based reinforcement learning
    algorithms can lead to divergence, especially when combining non-linear
    function approximators, temporal difference learning and off-policy
    learning. Yet particularly over the last decade researchers have developed
    techniques to reduce the probability of divergence dramatically, making
    off-policy temporal difference learning algorithms the first choice for many
    problemns. This trajectory started with the development of the deep
    Q-network (DQN) by DeepMind. Since then each new iteration of the algorithm
    provided a new improvement, a piece of the puzzle. All of those pieces were
    eventually combined by DeepMind into the so called Rainbow algorithm.`),e=D(),z(o.$$.fragment),h=D(),n=b("p"),p=N(`In this chapter we are basically going to take a ride down the history lane
    of modern value based deep reinforcement learning. We will start the journey
    by impelementing the DQN algorithm. After that we will cover each subsequent
    improvement separately until we are able to implement a fully featured
    Rainbow algorithm.`),f=D(),u=b("p"),a=N(`Before we move on to the discussions and implementations of the individual
    algorithms let us shortly discuss the approach that we are going to take in
    the subsequent chapters. Some of the discussed algorithms built upon the
    previous findings. For example the duelling DQN used the double DQN and not
    the original vanilla DQN as the basis for improvement. In research this
    approach is desirable, as you need to show that your contributions are able
    to improve the current state of the art implementation. We are going to
    cover each of the sections independently. Only the original DQN is going to
    serve as the basis for each of the the improvements. In our opinion this
    makes didactically more sense, as we only need to focus on one piece of the
    puzzle. The combination of the different improvements is going to be
    implemented in the final chapter of this section, the Rainbow algorithm.`),s=D(),c=b("div"),this.h()},l(i){t=_(i,"P",{});var m=y(t);r=E(m,`The naive implementation of approximate value based reinforcement learning
    algorithms can lead to divergence, especially when combining non-linear
    function approximators, temporal difference learning and off-policy
    learning. Yet particularly over the last decade researchers have developed
    techniques to reduce the probability of divergence dramatically, making
    off-policy temporal difference learning algorithms the first choice for many
    problemns. This trajectory started with the development of the deep
    Q-network (DQN) by DeepMind. Since then each new iteration of the algorithm
    provided a new improvement, a piece of the puzzle. All of those pieces were
    eventually combined by DeepMind into the so called Rainbow algorithm.`),m.forEach(l),e=$(i),V(o.$$.fragment,i),h=$(i),n=_(i,"P",{});var k=y(n);p=E(k,`In this chapter we are basically going to take a ride down the history lane
    of modern value based deep reinforcement learning. We will start the journey
    by impelementing the DQN algorithm. After that we will cover each subsequent
    improvement separately until we are able to implement a fully featured
    Rainbow algorithm.`),k.forEach(l),f=$(i),u=_(i,"P",{});var R=y(u);a=E(R,`Before we move on to the discussions and implementations of the individual
    algorithms let us shortly discuss the approach that we are going to take in
    the subsequent chapters. Some of the discussed algorithms built upon the
    previous findings. For example the duelling DQN used the double DQN and not
    the original vanilla DQN as the basis for improvement. In research this
    approach is desirable, as you need to show that your contributions are able
    to improve the current state of the art implementation. We are going to
    cover each of the sections independently. Only the original DQN is going to
    serve as the basis for each of the the improvements. In our opinion this
    makes didactically more sense, as we only need to focus on one piece of the
    puzzle. The combination of the different improvements is going to be
    implemented in the final chapter of this section, the Rainbow algorithm.`),R.forEach(l),s=$(i),c=_(i,"DIV",{class:!0}),y(c).forEach(l),this.h()},h(){Q(c,"class","separator")},m(i,m){v(i,t,m),g(t,r),v(i,e,m),T(o,i,m),v(i,h,m),v(i,n,m),g(n,p),v(i,f,m),v(i,u,m),g(u,a),v(i,s,m),v(i,c,m),w=!0},p(i,m){const k={};m&16&&(k.$$scope={dirty:m,ctx:i}),o.$set(k)},i(i){w||(A(o.$$.fragment,i),w=!0)},o(i){B(o.$$.fragment,i),w=!1},d(i){i&&l(t),i&&l(e),S(o,i),i&&l(h),i&&l(n),i&&l(f),i&&l(u),i&&l(s),i&&l(c)}}}function Y(d){let t,r,e,o,h,n,p,f,u;return f=new W({props:{$$slots:{default:[H]},$$scope:{ctx:d}}}),{c(){t=b("meta"),r=D(),e=b("h1"),o=N("Value Based Deep Reinforcement Learning"),h=D(),n=b("div"),p=D(),z(f.$$.fragment),this.h()},l(a){const s=C('[data-svelte="svelte-jyzkp3"]',document.head);t=_(s,"META",{name:!0,content:!0}),s.forEach(l),r=$(a),e=_(a,"H1",{});var c=y(e);o=E(c,"Value Based Deep Reinforcement Learning"),c.forEach(l),h=$(a),n=_(a,"DIV",{class:!0}),y(n).forEach(l),p=$(a),V(f.$$.fragment,a),this.h()},h(){document.title="World4AI | Reinforcement Learning | Value Based Deep Reinforcement Learning",Q(t,"name","description"),Q(t,"content","Value based deep reinforcement learning algorithms like DQN, DDQN, duelling DDQN and so on have achieved state of the art results over the last 10 years."),Q(n,"class","separator")},m(a,s){g(document.head,t),v(a,r,s),v(a,e,s),g(e,o),v(a,h,s),v(a,n,s),v(a,p,s),T(f,a,s),u=!0},p(a,[s]){const c={};s&16&&(c.$$scope={dirty:s,ctx:a}),f.$set(c)},i(a){u||(A(f.$$.fragment,a),u=!0)},o(a){B(f.$$.fragment,a),u=!1},d(a){l(t),a&&l(r),a&&l(e),a&&l(h),a&&l(n),a&&l(p),S(f,a)}}}function G(d){return[[{time:"2013, 2015",content:"DQN"},{time:2016,content:"Double DQN"},{time:2016,content:"Duelling DQN"},{time:2015,content:"Prioritized Experience Replay"},{time:2016,content:"AC3"},{time:2017,content:"Distributional DQN"},{time:2017,content:"Noisy DQN"},{time:2017,content:"\u{1F308} Rainbow"}]]}class U extends j{constructor(t){super(),M(this,t,G,Y,P,{})}}export{U as default};
