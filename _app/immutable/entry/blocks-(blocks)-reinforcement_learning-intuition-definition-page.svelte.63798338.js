import{S as Kt,i as Zt,s as xt,y as S,z as L,A as P,g as _,d as y,B as R,Q as he,R as ce,m as b,h as i,n as T,b as h,v as qt,f as Et,P as St,e as De,N as k,C as en,k as q,a as $,q as w,W as _n,l as E,c as d,r as v,V as It}from"../chunks/index.4d92b023.js";import{C as bn}from"../chunks/Container.b0705c7b.js";import{A as Me}from"../chunks/Alert.25a852b3.js";import{F as yn,I as kn}from"../chunks/InternalLink.7deb899c.js";import{H as Re}from"../chunks/Highlight.b7c1de53.js";import{S as Tn}from"../chunks/SvgContainer.f70b5745.js";import{B as wn}from"../chunks/Block.059eddcd.js";import{A as In}from"../chunks/Arrow.ae91874c.js";import{R as qn}from"../chunks/RandomAgent.2a3051e9.js";import{D as En}from"../chunks/DeterministicAgent.a1dd8b4b.js";import{G as sn,g as ln,a as Qt}from"../chunks/maps.0f079072.js";import{I as fn}from"../chunks/Interaction.6c345f67.js";function hn(o,t,n){const r=o.slice();return r[2]=t[n],r[4]=n,r}function cn(o){let t,n;return t=new wn({props:{x:1+o[4]*Fe+Fe/2+o[4]*4,y:1+Fe/2,width:Fe,height:Fe,text:o[4]+1,fontSize:12,class:o[4]===o[0]-1?"fill-blue-400":"fill-red-400"}}),{c(){S(t.$$.fragment)},l(r){L(t.$$.fragment,r)},m(r,a){P(t,r,a),n=!0},p(r,a){const c={};a&1&&(c.class=r[4]===r[0]-1?"fill-blue-400":"fill-red-400"),t.$set(c)},i(r){n||(_(t.$$.fragment,r),n=!0)},o(r){y(t.$$.fragment,r),n=!1},d(r){R(t,r)}}}function Sn(o){let t,n,r=o[1],a=[];for(let f=0;f<r.length;f+=1)a[f]=cn(hn(o,r,f));const c=f=>y(a[f],1,1,()=>{a[f]=null});return{c(){t=he("svg");for(let f=0;f<a.length;f+=1)a[f].c();this.h()},l(f){t=ce(f,"svg",{viewBox:!0});var g=b(t);for(let s=0;s<a.length;s+=1)a[s].l(g);g.forEach(i),this.h()},h(){T(t,"viewBox","0 0 370 22")},m(f,g){h(f,t,g);for(let s=0;s<a.length;s+=1)a[s]&&a[s].m(t,null);n=!0},p(f,g){if(g&1){r=f[1];let s;for(s=0;s<r.length;s+=1){const I=hn(f,r,s);a[s]?(a[s].p(I,g),_(a[s],1)):(a[s]=cn(I),a[s].c(),_(a[s],1),a[s].m(t,null))}for(qt(),s=r.length;s<a.length;s+=1)c(s);Et()}},i(f){if(!n){for(let g=0;g<r.length;g+=1)_(a[g]);n=!0}},o(f){a=a.filter(Boolean);for(let g=0;g<a.length;g+=1)y(a[g]);n=!1},d(f){f&&i(t),St(a,f)}}}function Ln(o){let t,n;return t=new Tn({props:{maxWidth:"500px",$$slots:{default:[Sn]},$$scope:{ctx:o}}}),{c(){S(t.$$.fragment)},l(r){L(t.$$.fragment,r)},m(r,a){P(t,r,a),n=!0},p(r,[a]){const c={};a&33&&(c.$$scope={dirty:a,ctx:r}),t.$set(c)},i(r){n||(_(t.$$.fragment,r),n=!0)},o(r){y(t.$$.fragment,r),n=!1},d(r){R(t,r)}}}const Fe=20;function Pn(o,t,n){let{sequenceLength:r}=t,a=[];for(let c=0;c<r;c++)a.push(c);return o.$$set=c=>{"sequenceLength"in c&&n(0,r=c.sequenceLength)},[r,a]}class Jt extends Kt{constructor(t){super(),Zt(this,t,Pn,Ln,xt,{sequenceLength:0})}}function mn(o,t,n){const r=o.slice();return r[2]=t[n].d,r[3]=t[n]._,r[5]=n,r}function un(o,t,n){const r=o.slice();return r[5]=n,r}function gn(o,t,n){const r=o.slice();return r[8]=n,r}function Rn(o){let t,n;return t=new In({props:{strokeWidth:.7,dashed:!0,strokeDashArray:"6 6",showMarker:!1,moving:!0,data:[{x:H+o[5]*(o[1]+x)+o[1]/2,y:H+o[1]},{x:H+o[8]*(o[1]+x)+o[1]/2,y:Ae-o[1]-o[1]/2}]}}),{c(){S(t.$$.fragment)},l(r){L(t.$$.fragment,r)},m(r,a){P(t,r,a),n=!0},p:en,i(r){n||(_(t.$$.fragment,r),n=!0)},o(r){y(t.$$.fragment,r),n=!1},d(r){R(t,r)}}}function pn(o){let t,n,r=o[8]<=o[5]&&Rn(o);return{c(){r&&r.c(),t=De()},l(a){r&&r.l(a),t=De()},m(a,c){r&&r.m(a,c),h(a,t,c),n=!0},p(a,c){a[8]<=a[5]&&r.p(a,c)},i(a){n||(_(r),n=!0)},o(a){y(r),n=!1},d(a){r&&r.d(a),a&&i(t)}}}function $n(o){let t,n,r=o[0],a=[];for(let f=0;f<r.length;f+=1)a[f]=pn(gn(o,r,f));const c=f=>y(a[f],1,1,()=>{a[f]=null});return{c(){for(let f=0;f<a.length;f+=1)a[f].c();t=De()},l(f){for(let g=0;g<a.length;g+=1)a[g].l(f);t=De()},m(f,g){for(let s=0;s<a.length;s+=1)a[s]&&a[s].m(f,g);h(f,t,g),n=!0},p(f,g){if(g&2){r=f[0];let s;for(s=0;s<r.length;s+=1){const I=gn(f,r,s);a[s]?(a[s].p(I,g),_(a[s],1)):(a[s]=pn(I),a[s].c(),_(a[s],1),a[s].m(t.parentNode,t))}for(qt(),s=r.length;s<a.length;s+=1)c(s);Et()}},i(f){if(!n){for(let g=0;g<r.length;g+=1)_(a[g]);n=!0}},o(f){a=a.filter(Boolean);for(let g=0;g<a.length;g+=1)y(a[g]);n=!1},d(f){St(a,f),f&&i(t)}}}function dn(o){let t,n,r,a,c,f,g;return t=new wn({props:{x:H+o[5]*(o[1]+x)+o[1]/2,y:H+o[1]/2,width:o[1],height:o[1],text:o[5]+1,fontSize:13,class:o[5]<o[0].length-1?"fill-red-400":"fill-blue-400"}}),{c(){S(t.$$.fragment),n=he("defs"),r=he("marker"),a=he("polygon"),c=he("circle"),f=he("line"),this.h()},l(s){L(t.$$.fragment,s),n=ce(s,"defs",{});var I=b(n);r=ce(I,"marker",{id:!0,markerWidth:!0,markerHeight:!0,refX:!0,refY:!0,orient:!0,class:!0});var m=b(r);a=ce(m,"polygon",{points:!0}),b(a).forEach(i),m.forEach(i),I.forEach(i),c=ce(s,"circle",{class:!0,cx:!0,cy:!0,r:!0}),b(c).forEach(i),f=ce(s,"line",{x1:!0,y1:!0,x2:!0,y2:!0,transform:!0,"stroke-width":!0,"marker-end":!0,class:!0}),b(f).forEach(i),this.h()},h(){T(a,"points","0 0, 10 3, 0 6"),T(r,"id","arrowhead"),T(r,"markerWidth","10"),T(r,"markerHeight","6"),T(r,"refX","0"),T(r,"refY","3"),T(r,"orient","auto"),T(r,"class","fill-black"),T(c,"class","fill-slate-200 stroke-black"),T(c,"cx",H+o[5]*(o[1]+x)+o[1]/2),T(c,"cy",Ae-o[1]),T(c,"r",o[1]/2),T(f,"x1",H+o[5]*(o[1]+x)),T(f,"y1",Ae-o[1]),T(f,"x2",H+o[5]*(o[1]+x)+o[1]-5),T(f,"y2",Ae-o[1]),T(f,"transform","rotate("+o[2]+", "+(H+o[5]*(o[1]+x)+o[1]/2)+", "+(Ae-o[1])+")"),T(f,"stroke-width","0.5"),T(f,"marker-end","url(#arrowhead)"),T(f,"class","stroke-black")},m(s,I){P(t,s,I),h(s,n,I),k(n,r),k(r,a),h(s,c,I),h(s,f,I),g=!0},p:en,i(s){g||(_(t.$$.fragment,s),g=!0)},o(s){y(t.$$.fragment,s),g=!1},d(s){R(t,s),s&&i(n),s&&i(c),s&&i(f)}}}function An(o){let t,n,r,a=o[0],c=[];for(let m=0;m<a.length;m+=1)c[m]=$n(un(o,a,m));const f=m=>y(c[m],1,1,()=>{c[m]=null});let g=o[0],s=[];for(let m=0;m<g.length;m+=1)s[m]=dn(mn(o,g,m));const I=m=>y(s[m],1,1,()=>{s[m]=null});return{c(){t=he("svg");for(let m=0;m<c.length;m+=1)c[m].c();n=De();for(let m=0;m<s.length;m+=1)s[m].c();this.h()},l(m){t=ce(m,"svg",{viewBox:!0});var p=b(t);for(let u=0;u<c.length;u+=1)c[u].l(p);n=De();for(let u=0;u<s.length;u+=1)s[u].l(p);p.forEach(i),this.h()},h(){T(t,"viewBox","0 0 "+vn+" "+Ae)},m(m,p){h(m,t,p);for(let u=0;u<c.length;u+=1)c[u]&&c[u].m(t,null);k(t,n);for(let u=0;u<s.length;u+=1)s[u]&&s[u].m(t,null);r=!0},p(m,[p]){if(p&3){a=m[0];let u;for(u=0;u<a.length;u+=1){const D=un(m,a,u);c[u]?(c[u].p(D,p),_(c[u],1)):(c[u]=$n(D),c[u].c(),_(c[u],1),c[u].m(t,n))}for(qt(),u=a.length;u<c.length;u+=1)f(u);Et()}if(p&3){g=m[0];let u;for(u=0;u<g.length;u+=1){const D=mn(m,g,u);s[u]?(s[u].p(D,p),_(s[u],1)):(s[u]=dn(D),s[u].c(),_(s[u],1),s[u].m(t,null))}for(qt(),u=g.length;u<s.length;u+=1)I(u);Et()}},i(m){if(!r){for(let p=0;p<a.length;p+=1)_(c[p]);for(let p=0;p<g.length;p+=1)_(s[p]);r=!0}},o(m){c=c.filter(Boolean);for(let p=0;p<c.length;p+=1)y(c[p]);s=s.filter(Boolean);for(let p=0;p<s.length;p+=1)y(s[p]);r=!1},d(m){m&&i(t),St(c,m),St(s,m)}}}let vn=400,Ae=120,H=.5,x=15;function Dn(o){let t=[{d:0,r:"negative"},{d:90,r:"negative"},{d:180,r:"negative"},{d:270,r:"negative"},{d:270,r:"negative"},{d:0,r:"negative"},{d:0,r:"negative"},{d:90,r:"negative"},{d:180,r:"negative"},{d:0,r:"positive"}],n=vn/t.length-x;return[t,n]}class zn extends Kt{constructor(t){super(),Zt(this,t,Dn,An,xt,{})}}function Cn(o){let t,n,r,a;return n=new kn({props:{type:"note",id:1}}),{c(){t=w(`Reinforcement Learning is characterized by learning through trial and error
    and delayed rewards`),S(n.$$.fragment),r=w(".")},l(c){t=v(c,`Reinforcement Learning is characterized by learning through trial and error
    and delayed rewards`),L(n.$$.fragment,c),r=v(c,".")},m(c,f){h(c,t,f),P(n,c,f),h(c,r,f),a=!0},p:en,i(c){a||(_(n.$$.fragment,c),a=!0)},o(c){y(n.$$.fragment,c),a=!1},d(c){c&&i(t),R(n,c),c&&i(r)}}}function Bn(o){let t;return{c(){t=w("Learning")},l(n){t=v(n,"Learning")},m(n,r){h(n,t,r)},d(n){n&&i(t)}}}function Wn(o){let t;return{c(){t=w("Trial and Error")},l(n){t=v(n,"Trial and Error")},m(n,r){h(n,t,r)},d(n){n&&i(t)}}}function Hn(o){let t;return{c(){t=w("Delayed Rewards")},l(n){t=v(n,"Delayed Rewards")},m(n,r){h(n,t,r)},d(n){n&&i(t)}}}function Nn(o){let t;return{c(){t=w(`Learning means that the agent gets better at achieving the goal of the
    environment over time.`)},l(n){t=v(n,`Learning means that the agent gets better at achieving the goal of the
    environment over time.`)},m(n,r){h(n,t,r)},d(n){n&&i(t)}}}function On(o){let t;return{c(){t=w("reward")},l(n){t=v(n,"reward")},m(n,r){h(n,t,r)},d(n){n&&i(t)}}}function Gn(o){let t;return{c(){t=w(`In reinforcement learning the agent learns to maximize rewards. The goal of
    the environment has to be implicitly contained in the rewards.`)},l(n){t=v(n,`In reinforcement learning the agent learns to maximize rewards. The goal of
    the environment has to be implicitly contained in the rewards.`)},m(n,r){h(n,t,r)},d(n){n&&i(t)}}}function Vn(o){let t;return{c(){t=w("trial and error")},l(n){t=v(n,"trial and error")},m(n,r){h(n,t,r)},d(n){n&&i(t)}}}function jn(o){let t;return{c(){t=w(`In the context of reinforcement learning, trial and error means trying out
    different sequences of actions and compare the resulting sum of rewards to
    learn optimal behaviour.`)},l(n){t=v(n,`In the context of reinforcement learning, trial and error means trying out
    different sequences of actions and compare the resulting sum of rewards to
    learn optimal behaviour.`)},m(n,r){h(n,t,r)},d(n){n&&i(t)}}}function Mn(o){let t;return{c(){t=w(`In reinforcement learning rewards for an action are often delayed, which
    leads to the credit assignment problem.`)},l(n){t=v(n,`In reinforcement learning rewards for an action are often delayed, which
    leads to the credit assignment problem.`)},m(n,r){h(n,t,r)},d(n){n&&i(t)}}}function Fn(o){let t;return{c(){t=w("the credit assignment problem")},l(n){t=v(n,"the credit assignment problem")},m(n,r){h(n,t,r)},d(n){n&&i(t)}}}function Un(o){let t,n,r,a,c,f,g,s,I,m,p,u,D,me,B,ee,z,ze,ue,W,ge,A,N,Ue,pe,Lt,Xe,O,Ye,$e,Pt,Qe,G,Je,de,Ke,we,Rt,Ze,V,At,j,Dt,xe,M,et,ve,zt,tt,F,nt,_e,Ct,rt,be,it,ye,Bt,ot,U,Wt,X,Ht,at,ke,Nt,st,te,Ot,lt,ne,ft,re,Gt,ht,ie,ct,oe,Vt,mt,ae,ut,Te,jt,gt,Y,pt,Ie,$t,qe,Mt,dt,Ee,Ft,wt,Q,vt,J,Ut,K,Xt,_t,se,bt,Se,Yt,yt,le,kt;return a=new Me({props:{type:"info",$$slots:{default:[Cn]},$$scope:{ctx:o}}}),s=new Re({props:{$$slots:{default:[Bn]},$$scope:{ctx:o}}}),m=new Re({props:{$$slots:{default:[Wn]},$$scope:{ctx:o}}}),u=new Re({props:{$$slots:{default:[Hn]},$$scope:{ctx:o}}}),N=new Qt({props:{cells:o[3],player:o[2]}}),O=new Me({props:{type:"info",$$slots:{default:[Nn]},$$scope:{ctx:o}}}),G=new Qt({props:{cells:o[1],player:o[0]}}),j=new Re({props:{$$slots:{default:[On]},$$scope:{ctx:o}}}),M=new Me({props:{type:"info",$$slots:{default:[Gn]},$$scope:{ctx:o}}}),F=new Qt({props:{cells:o[1],player:o[0],showColoredReward:!0}}),X=new Re({props:{$$slots:{default:[Vn]},$$scope:{ctx:o}}}),ne=new Jt({props:{sequenceLength:"13"}}),ie=new Jt({props:{sequenceLength:"10"}}),ae=new Jt({props:{sequenceLength:"15"}}),Y=new Me({props:{type:"info",$$slots:{default:[jn]},$$scope:{ctx:o}}}),Q=new Me({props:{type:"info",$$slots:{default:[Mn]},$$scope:{ctx:o}}}),K=new Re({props:{$$slots:{default:[Fn]},$$scope:{ctx:o}}}),se=new zn({}),le=new yn({props:{notes:o[4]}}),{c(){t=q("p"),n=w(`There are probably dozens of formal definitions of reinforcement learning.
    These definitions do not necessarily contradict each other, but rather
    explain something similar when we look a little deeper at what the
    definitions are trying to convey. In this section we are going to look at
    the one definition that should capture the essence of reinforcement learning
    in a very clear way.`),r=$(),S(a.$$.fragment),c=$(),f=q("p"),g=w("The definition consists of three distinct parts: "),S(s.$$.fragment),I=w(`,
    `),S(m.$$.fragment),p=w(`
    and `),S(u.$$.fragment),D=w(`. In order to understand the
    complete definition we will deconstruct the sentence and look at each part
    individually.`),me=$(),B=q("div"),ee=$(),z=q("h2"),ze=w("Learning"),ue=$(),W=q("p"),ge=w(`Learning is probably the most obvious part of the definition. When the agent
    starts to interact with the environment the agent does not know anything
    about that environment, but the environment contains some goal that the
    agent has to achieve.`),A=$(),S(N.$$.fragment),Ue=$(),pe=q("p"),Lt=w(`In the example above the agent is expected to move the circle from the
    starting cell (top left corner) to the goal cell (bottom left corner).`),Xe=$(),S(O.$$.fragment),Ye=$(),$e=q("p"),Pt=w(`When we talk about learning we imply that the agent gets better at achieving
    that particular goal over time. The agent would probably move randomly at
    first, but over time learn the best possible (meaning the shortest) route.`),Qe=$(),S(G.$$.fragment),Je=$(),de=q("div"),Ke=$(),we=q("h2"),Rt=w("Rewards"),Ze=$(),V=q("p"),At=w(`The question still remains how exactly does the agent figure out what the
    goal of the environment actually is? The environment with which the agent
    interacts gives feedback about the behaviour of the agent by giving out a
    `),S(j.$$.fragment),Dt=w(" after each single step that the agent takes."),xe=$(),S(M.$$.fragment),et=$(),ve=q("p"),zt=w(`If the goal of the grid world environment is to move the circle to the cell
    with the triangle as fast as possible the environment could for example give
    a positive reward when the agent reaches the goal cell and punish the agent
    in any other case.`),tt=$(),S(F.$$.fragment),nt=$(),_e=q("p"),Ct=w(`The above animation represents that idea by color-coding rewards. The red
    grid cells give a reward of -1. The blue grid cell gives a reward of +1. If
    the agent takes a random route to the triangle, then the sum of rewards is
    going to be very negative. If on the other hand like in the animation above
    the agent takes the direct route to the triangle, the sum of rewards is
    going to be larger (but still negative). The agent learns through the reward
    feedback that some sequences of actions are better than others. Generally
    speaking the agent needs to find the routes that produce high sum of
    rewards.`),rt=$(),be=q("div"),it=$(),ye=q("h2"),Bt=w("Trial and Error"),ot=$(),U=q("p"),Wt=w(`The problem with rewards is that it is not clear from the very beginning
    what path produces the highest possible sum of rewards. It is therefore not
    clear which sequence of actions the agent needs to take. In reinforcement
    learning the only feedback the agent receives is the reward signal and even
    if the agent receives a positive sum of rewards it never knows if it could
    have done better. Unlike in supervised learning, there is no teacher (a.k.a.
    supervisor) to tell the agent what the best behaviour is. So how can the
    agent figure out what sequence of actions produces the highest sum of
    rewards? The only way it can: by `),S(X.$$.fragment),Ht=w("."),at=$(),ke=q("p"),Nt=w(`The agent has to try out different behaviour and produce different sequences
    of rewards to figure out which sequence of actions is the optimal one. How
    long it takes the agent to find a good sequence of decisions depends on the
    complexity of the environment and the employed learning algorithm.`),st=$(),te=q("p"),Ot=w("Trial Nr. 1"),lt=$(),S(ne.$$.fragment),ft=$(),re=q("p"),Gt=w("Trial Nr. 2"),ht=$(),S(ie.$$.fragment),ct=$(),oe=q("p"),Vt=w("Trial Nr. 3"),mt=$(),S(ae.$$.fragment),ut=$(),Te=q("p"),jt=w(`The above figures show how the sequences of actions might look like in the
    gridworld environment after three trials. In the second trial the agend
    takes the shortest route and has therefore the highest sum of rewards. It
    might therefore be a good idea to follow the first sequence of actions more
    often that the sequence of actions taken in the first and third trial.`),gt=$(),S(Y.$$.fragment),pt=$(),Ie=q("div"),$t=$(),qe=q("h2"),Mt=w("Delayed"),dt=$(),Ee=q("p"),Ft=w(`In reinforcement learning the agent often needs to take dozens or even
    thousands of steps before a reward is achieved. In that case there has been
    a succession of many steps and the agent has to decide which step and in
    which proportion is responsible for the reward, so that the agent could
    select the decisions that lead to a good sequence of rewards more often.`),wt=$(),S(Q.$$.fragment),vt=$(),J=q("p"),Ut=w(`Which of the steps is responsible for a particular reward? Is it the action
    just prior to the reward? Or the one before that? Or the one before that?
    Reinforcement Learning has no easy answer to the question which decision
    gets the credit for the reward. This problem is called `),S(K.$$.fragment),Xt=w("."),_t=$(),S(se.$$.fragment),bt=$(),Se=q("p"),Yt=w(`Let's assume that in the grid world example the agent took 10 steps to reach
    the goal. The first reward can only be assigned to the first action. The
    second reward can be assigned to the first and the second action. And so on.
    The last (and positive) reward can theoretically be assigned to any of the
    actions taken prior to the reward.`),yt=$(),S(le.$$.fragment),this.h()},l(e){t=E(e,"P",{});var l=b(t);n=v(l,`There are probably dozens of formal definitions of reinforcement learning.
    These definitions do not necessarily contradict each other, but rather
    explain something similar when we look a little deeper at what the
    definitions are trying to convey. In this section we are going to look at
    the one definition that should capture the essence of reinforcement learning
    in a very clear way.`),l.forEach(i),r=d(e),L(a.$$.fragment,e),c=d(e),f=E(e,"P",{});var C=b(f);g=v(C,"The definition consists of three distinct parts: "),L(s.$$.fragment,C),I=v(C,`,
    `),L(m.$$.fragment,C),p=v(C,`
    and `),L(u.$$.fragment,C),D=v(C,`. In order to understand the
    complete definition we will deconstruct the sentence and look at each part
    individually.`),C.forEach(i),me=d(e),B=E(e,"DIV",{class:!0}),b(B).forEach(i),ee=d(e),z=E(e,"H2",{});var Ce=b(z);ze=v(Ce,"Learning"),Ce.forEach(i),ue=d(e),W=E(e,"P",{});var Be=b(W);ge=v(Be,`Learning is probably the most obvious part of the definition. When the agent
    starts to interact with the environment the agent does not know anything
    about that environment, but the environment contains some goal that the
    agent has to achieve.`),Be.forEach(i),A=d(e),L(N.$$.fragment,e),Ue=d(e),pe=E(e,"P",{});var We=b(pe);Lt=v(We,`In the example above the agent is expected to move the circle from the
    starting cell (top left corner) to the goal cell (bottom left corner).`),We.forEach(i),Xe=d(e),L(O.$$.fragment,e),Ye=d(e),$e=E(e,"P",{});var Le=b($e);Pt=v(Le,`When we talk about learning we imply that the agent gets better at achieving
    that particular goal over time. The agent would probably move randomly at
    first, but over time learn the best possible (meaning the shortest) route.`),Le.forEach(i),Qe=d(e),L(G.$$.fragment,e),Je=d(e),de=E(e,"DIV",{class:!0}),b(de).forEach(i),Ke=d(e),we=E(e,"H2",{});var He=b(we);Rt=v(He,"Rewards"),He.forEach(i),Ze=d(e),V=E(e,"P",{});var Z=b(V);At=v(Z,`The question still remains how exactly does the agent figure out what the
    goal of the environment actually is? The environment with which the agent
    interacts gives feedback about the behaviour of the agent by giving out a
    `),L(j.$$.fragment,Z),Dt=v(Z," after each single step that the agent takes."),Z.forEach(i),xe=d(e),L(M.$$.fragment,e),et=d(e),ve=E(e,"P",{});var Ne=b(ve);zt=v(Ne,`If the goal of the grid world environment is to move the circle to the cell
    with the triangle as fast as possible the environment could for example give
    a positive reward when the agent reaches the goal cell and punish the agent
    in any other case.`),Ne.forEach(i),tt=d(e),L(F.$$.fragment,e),nt=d(e),_e=E(e,"P",{});var Oe=b(_e);Ct=v(Oe,`The above animation represents that idea by color-coding rewards. The red
    grid cells give a reward of -1. The blue grid cell gives a reward of +1. If
    the agent takes a random route to the triangle, then the sum of rewards is
    going to be very negative. If on the other hand like in the animation above
    the agent takes the direct route to the triangle, the sum of rewards is
    going to be larger (but still negative). The agent learns through the reward
    feedback that some sequences of actions are better than others. Generally
    speaking the agent needs to find the routes that produce high sum of
    rewards.`),Oe.forEach(i),rt=d(e),be=E(e,"DIV",{class:!0}),b(be).forEach(i),it=d(e),ye=E(e,"H2",{});var Pe=b(ye);Bt=v(Pe,"Trial and Error"),Pe.forEach(i),ot=d(e),U=E(e,"P",{});var fe=b(U);Wt=v(fe,`The problem with rewards is that it is not clear from the very beginning
    what path produces the highest possible sum of rewards. It is therefore not
    clear which sequence of actions the agent needs to take. In reinforcement
    learning the only feedback the agent receives is the reward signal and even
    if the agent receives a positive sum of rewards it never knows if it could
    have done better. Unlike in supervised learning, there is no teacher (a.k.a.
    supervisor) to tell the agent what the best behaviour is. So how can the
    agent figure out what sequence of actions produces the highest sum of
    rewards? The only way it can: by `),L(X.$$.fragment,fe),Ht=v(fe,"."),fe.forEach(i),at=d(e),ke=E(e,"P",{});var Ge=b(ke);Nt=v(Ge,`The agent has to try out different behaviour and produce different sequences
    of rewards to figure out which sequence of actions is the optimal one. How
    long it takes the agent to find a good sequence of decisions depends on the
    complexity of the environment and the employed learning algorithm.`),Ge.forEach(i),st=d(e),te=E(e,"P",{class:!0});var Ve=b(te);Ot=v(Ve,"Trial Nr. 1"),Ve.forEach(i),lt=d(e),L(ne.$$.fragment,e),ft=d(e),re=E(e,"P",{class:!0});var je=b(re);Gt=v(je,"Trial Nr. 2"),je.forEach(i),ht=d(e),L(ie.$$.fragment,e),ct=d(e),oe=E(e,"P",{class:!0});var tn=b(oe);Vt=v(tn,"Trial Nr. 3"),tn.forEach(i),mt=d(e),L(ae.$$.fragment,e),ut=d(e),Te=E(e,"P",{});var nn=b(Te);jt=v(nn,`The above figures show how the sequences of actions might look like in the
    gridworld environment after three trials. In the second trial the agend
    takes the shortest route and has therefore the highest sum of rewards. It
    might therefore be a good idea to follow the first sequence of actions more
    often that the sequence of actions taken in the first and third trial.`),nn.forEach(i),gt=d(e),L(Y.$$.fragment,e),pt=d(e),Ie=E(e,"DIV",{class:!0}),b(Ie).forEach(i),$t=d(e),qe=E(e,"H2",{});var rn=b(qe);Mt=v(rn,"Delayed"),rn.forEach(i),dt=d(e),Ee=E(e,"P",{});var on=b(Ee);Ft=v(on,`In reinforcement learning the agent often needs to take dozens or even
    thousands of steps before a reward is achieved. In that case there has been
    a succession of many steps and the agent has to decide which step and in
    which proportion is responsible for the reward, so that the agent could
    select the decisions that lead to a good sequence of rewards more often.`),on.forEach(i),wt=d(e),L(Q.$$.fragment,e),vt=d(e),J=E(e,"P",{});var Tt=b(J);Ut=v(Tt,`Which of the steps is responsible for a particular reward? Is it the action
    just prior to the reward? Or the one before that? Or the one before that?
    Reinforcement Learning has no easy answer to the question which decision
    gets the credit for the reward. This problem is called `),L(K.$$.fragment,Tt),Xt=v(Tt,"."),Tt.forEach(i),_t=d(e),L(se.$$.fragment,e),bt=d(e),Se=E(e,"P",{});var an=b(Se);Yt=v(an,`Let's assume that in the grid world example the agent took 10 steps to reach
    the goal. The first reward can only be assigned to the first action. The
    second reward can be assigned to the first and the second action. And so on.
    The last (and positive) reward can theoretically be assigned to any of the
    actions taken prior to the reward.`),an.forEach(i),yt=d(e),L(le.$$.fragment,e),this.h()},h(){T(B,"class","separator"),T(de,"class","separator"),T(be,"class","separator"),T(te,"class","flex justify-center items-center font-bold"),T(re,"class","flex justify-center items-center font-bold"),T(oe,"class","flex justify-center items-center font-bold"),T(Ie,"class","separator")},m(e,l){h(e,t,l),k(t,n),h(e,r,l),P(a,e,l),h(e,c,l),h(e,f,l),k(f,g),P(s,f,null),k(f,I),P(m,f,null),k(f,p),P(u,f,null),k(f,D),h(e,me,l),h(e,B,l),h(e,ee,l),h(e,z,l),k(z,ze),h(e,ue,l),h(e,W,l),k(W,ge),h(e,A,l),P(N,e,l),h(e,Ue,l),h(e,pe,l),k(pe,Lt),h(e,Xe,l),P(O,e,l),h(e,Ye,l),h(e,$e,l),k($e,Pt),h(e,Qe,l),P(G,e,l),h(e,Je,l),h(e,de,l),h(e,Ke,l),h(e,we,l),k(we,Rt),h(e,Ze,l),h(e,V,l),k(V,At),P(j,V,null),k(V,Dt),h(e,xe,l),P(M,e,l),h(e,et,l),h(e,ve,l),k(ve,zt),h(e,tt,l),P(F,e,l),h(e,nt,l),h(e,_e,l),k(_e,Ct),h(e,rt,l),h(e,be,l),h(e,it,l),h(e,ye,l),k(ye,Bt),h(e,ot,l),h(e,U,l),k(U,Wt),P(X,U,null),k(U,Ht),h(e,at,l),h(e,ke,l),k(ke,Nt),h(e,st,l),h(e,te,l),k(te,Ot),h(e,lt,l),P(ne,e,l),h(e,ft,l),h(e,re,l),k(re,Gt),h(e,ht,l),P(ie,e,l),h(e,ct,l),h(e,oe,l),k(oe,Vt),h(e,mt,l),P(ae,e,l),h(e,ut,l),h(e,Te,l),k(Te,jt),h(e,gt,l),P(Y,e,l),h(e,pt,l),h(e,Ie,l),h(e,$t,l),h(e,qe,l),k(qe,Mt),h(e,dt,l),h(e,Ee,l),k(Ee,Ft),h(e,wt,l),P(Q,e,l),h(e,vt,l),h(e,J,l),k(J,Ut),P(K,J,null),k(J,Xt),h(e,_t,l),P(se,e,l),h(e,bt,l),h(e,Se,l),k(Se,Yt),h(e,yt,l),P(le,e,l),kt=!0},p(e,l){const C={};l&524288&&(C.$$scope={dirty:l,ctx:e}),a.$set(C);const Ce={};l&524288&&(Ce.$$scope={dirty:l,ctx:e}),s.$set(Ce);const Be={};l&524288&&(Be.$$scope={dirty:l,ctx:e}),m.$set(Be);const We={};l&524288&&(We.$$scope={dirty:l,ctx:e}),u.$set(We);const Le={};l&8&&(Le.cells=e[3]),l&4&&(Le.player=e[2]),N.$set(Le);const He={};l&524288&&(He.$$scope={dirty:l,ctx:e}),O.$set(He);const Z={};l&2&&(Z.cells=e[1]),l&1&&(Z.player=e[0]),G.$set(Z);const Ne={};l&524288&&(Ne.$$scope={dirty:l,ctx:e}),j.$set(Ne);const Oe={};l&524288&&(Oe.$$scope={dirty:l,ctx:e}),M.$set(Oe);const Pe={};l&2&&(Pe.cells=e[1]),l&1&&(Pe.player=e[0]),F.$set(Pe);const fe={};l&524288&&(fe.$$scope={dirty:l,ctx:e}),X.$set(fe);const Ge={};l&524288&&(Ge.$$scope={dirty:l,ctx:e}),Y.$set(Ge);const Ve={};l&524288&&(Ve.$$scope={dirty:l,ctx:e}),Q.$set(Ve);const je={};l&524288&&(je.$$scope={dirty:l,ctx:e}),K.$set(je)},i(e){kt||(_(a.$$.fragment,e),_(s.$$.fragment,e),_(m.$$.fragment,e),_(u.$$.fragment,e),_(N.$$.fragment,e),_(O.$$.fragment,e),_(G.$$.fragment,e),_(j.$$.fragment,e),_(M.$$.fragment,e),_(F.$$.fragment,e),_(X.$$.fragment,e),_(ne.$$.fragment,e),_(ie.$$.fragment,e),_(ae.$$.fragment,e),_(Y.$$.fragment,e),_(Q.$$.fragment,e),_(K.$$.fragment,e),_(se.$$.fragment,e),_(le.$$.fragment,e),kt=!0)},o(e){y(a.$$.fragment,e),y(s.$$.fragment,e),y(m.$$.fragment,e),y(u.$$.fragment,e),y(N.$$.fragment,e),y(O.$$.fragment,e),y(G.$$.fragment,e),y(j.$$.fragment,e),y(M.$$.fragment,e),y(F.$$.fragment,e),y(X.$$.fragment,e),y(ne.$$.fragment,e),y(ie.$$.fragment,e),y(ae.$$.fragment,e),y(Y.$$.fragment,e),y(Q.$$.fragment,e),y(K.$$.fragment,e),y(se.$$.fragment,e),y(le.$$.fragment,e),kt=!1},d(e){e&&i(t),e&&i(r),R(a,e),e&&i(c),e&&i(f),R(s),R(m),R(u),e&&i(me),e&&i(B),e&&i(ee),e&&i(z),e&&i(ue),e&&i(W),e&&i(A),R(N,e),e&&i(Ue),e&&i(pe),e&&i(Xe),R(O,e),e&&i(Ye),e&&i($e),e&&i(Qe),R(G,e),e&&i(Je),e&&i(de),e&&i(Ke),e&&i(we),e&&i(Ze),e&&i(V),R(j),e&&i(xe),R(M,e),e&&i(et),e&&i(ve),e&&i(tt),R(F,e),e&&i(nt),e&&i(_e),e&&i(rt),e&&i(be),e&&i(it),e&&i(ye),e&&i(ot),e&&i(U),R(X),e&&i(at),e&&i(ke),e&&i(st),e&&i(te),e&&i(lt),R(ne,e),e&&i(ft),e&&i(re),e&&i(ht),R(ie,e),e&&i(ct),e&&i(oe),e&&i(mt),R(ae,e),e&&i(ut),e&&i(Te),e&&i(gt),R(Y,e),e&&i(pt),e&&i(Ie),e&&i($t),e&&i(qe),e&&i(dt),e&&i(Ee),e&&i(wt),R(Q,e),e&&i(vt),e&&i(J),R(K),e&&i(_t),R(se,e),e&&i(bt),e&&i(Se),e&&i(yt),R(le,e)}}}function Xn(o){let t,n,r,a,c,f,g,s,I;return s=new bn({props:{$$slots:{default:[Un]},$$scope:{ctx:o}}}),{c(){t=q("meta"),n=$(),r=q("h1"),a=w("Definition Of Reinforcement Learning"),c=$(),f=q("div"),g=$(),S(s.$$.fragment),this.h()},l(m){const p=_n("svelte-9m1lh4",document.head);t=E(p,"META",{name:!0,content:!0}),p.forEach(i),n=d(m),r=E(m,"H1",{});var u=b(r);a=v(u,"Definition Of Reinforcement Learning"),u.forEach(i),c=d(m),f=E(m,"DIV",{class:!0}),b(f).forEach(i),g=d(m),L(s.$$.fragment,m),this.h()},h(){document.title="Definition of Reinforcement Learning - World4AI",T(t,"name","description"),T(t,"content","Reinforcement learning is defined as learning through trial and error and delayed rewards."),T(f,"class","separator")},m(m,p){k(document.head,t),h(m,n,p),h(m,r,p),k(r,a),h(m,c,p),h(m,f,p),h(m,g,p),P(s,m,p),I=!0},p(m,[p]){const u={};p&524303&&(u.$$scope={dirty:p,ctx:m}),s.$set(u)},i(m){I||(_(s.$$.fragment,m),I=!0)},o(m){y(s.$$.fragment,m),I=!1},d(m){i(t),m&&i(n),m&&i(r),m&&i(c),m&&i(f),m&&i(g),R(s,m)}}}function Yn(o,t,n){let r,a,c,f,g,s,I,m;const p=["This definition is highly inspired by the book 'Reinforcement Learning: An Introduction' by Richard S. Sutton and Andrew G. Barto."];let u=new sn(ln),D=new qn(u.getObservationSpace(),u.getActionSpace()),me=new fn(D,u,3);const B=u.cellsStore;It(o,B,A=>n(12,m=A));const ee=me.observationStore;It(o,ee,A=>n(11,I=A));let z=new sn(ln),ze=new En(z.getObservationSpace(),z.getActionSpace()),ue=new fn(ze,z,3);const W=z.cellsStore;It(o,W,A=>n(10,s=A));const ge=ue.observationStore;return It(o,ge,A=>n(9,g=A)),o.$$.update=()=>{o.$$.dirty&4096&&n(3,r=m),o.$$.dirty&2048&&n(2,a=I),o.$$.dirty&1024&&n(1,c=s),o.$$.dirty&512&&n(0,f=g)},[f,c,a,r,p,B,ee,W,ge,g,s,I,m]}class sr extends Kt{constructor(t){super(),Zt(this,t,Yn,Xn,xt,{})}}export{sr as default};
