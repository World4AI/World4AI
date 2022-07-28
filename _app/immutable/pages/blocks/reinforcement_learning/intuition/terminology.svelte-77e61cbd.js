import{S as mt,i as pt,s as vt,l as i,r as o,a as m,w as ut,m as r,n as h,u as a,h as t,c as p,x as dt,p as S,b as l,G as n,y as gt,f as yt,t as _t,B as Et,E as wt}from"../../../../chunks/index-caa95cd4.js";import{C as kt}from"../../../../chunks/Container-5c6b7f6d.js";function bt(pe){let v,$,w,d,k,u,b,f,c,G,A,ve,ue,L,de,ge,W,ye,_e,ee,N,te,O,Ee,ne,_,we,F,ke,be,q,Ie,Te,oe,P,ae,C,Re,le,E,$e,j,Se,Ge,J,Ne,Oe,se,V,ie,x,Pe,re,g,Ce,K,Ve,xe,Q,Be,De,U,Me,He,he,B,fe,D,ze,ce,y,Ae,X,Le,We,Y,Fe,qe,Z,je,Je,me,M;return{c(){v=i("p"),$=o(`In this chapter we are going to look at some terminology that is used
    throughout reinforcement learning and at how reinforcement learning
    algorithms can be classified based on the components the agent has.`),w=m(),d=i("div"),k=m(),u=i("h2"),b=o("Value Based vs Policy Based"),f=m(),c=i("p"),G=o(`Especially beginner level reinforcement learning agents have only a value
    function. In that case the policy of the agent is implicitly derived from
    the value function. Reinforcement learning methods that only utilize the
    value function are called `),A=i("strong"),ve=o("value based methods"),ue=o(`. If on the
    other hand the agent derives the policy directly without using value
    functions the methods are called `),L=i("strong"),de=o("policy based methods"),ge=o(`. Most
    modern algorithms have agents with both components. Those are called
    `),W=i("strong"),ye=o("actor-critic methods"),_e=o("."),ee=m(),N=i("div"),te=m(),O=i("h2"),Ee=o("Model Based vs Model Free"),ne=m(),_=i("p"),we=o(`If the agent has an internal representation of the model as a component, the
    algorithms are called `),F=i("strong"),ke=o("model based"),be=o(`. If the agent learns the
    policy without a model, the algorithms are called
    `),q=i("strong"),Ie=o("model free"),Te=o("."),oe=m(),P=i("div"),ae=m(),C=i("h2"),Re=o("Learning vs Planning"),le=m(),E=i("p"),$e=o(`In many cases the agent has no access to the model of the environment or has
    no internal representation of the model. Therefore the agent has to interact
    with the environment to improve the policy. In reinforcement learning this
    is called `),j=i("strong"),Se=o("learning"),Ge=o(`. If the agent on the other hand utilizes
    the model to improve the value function or the policy, we call it
    `),J=i("strong"),Ne=o("planning"),Oe=o("."),se=m(),V=i("div"),ie=m(),x=i("h2"),Pe=o("Prediction vs Improvement vs Control"),re=m(),g=i("p"),Ce=o("There are several tasks an agent might need to perform. We talk about the "),K=i("strong"),Ve=o("prediction"),xe=o(`
    task when we have a certain policy and the agent has to calculate the exact value
    function for that policy. We talk about the `),Q=i("strong"),Be=o("improvement"),De=o(`
    task when we want to improve a given policy. And we talk about the
    `),U=i("strong"),Me=o("control"),He=o(` task when the agent has to find the best possible policy.
    As you can imagine, prediction and improvement are necessary steps to solve the
    control problem.`),he=m(),B=i("div"),fe=m(),D=i("h2"),ze=o("Episodic vs Continuing Tasks"),ce=m(),y=i("p"),Ae=o(`In reinforcement learning we distinguish between episodic and continuing
    tasks. `),X=i("strong"),Le=o("Episodic"),We=o(` tasks are tasks that have a natural ending.
    The last state in an episodic task is called a `),Y=i("em"),Fe=o("terminal state"),qe=o(`.
    `),Z=i("strong"),je=o("Continuing"),Je=o(` tasks are tasks that do not have a natural ending
    and may theoretically go on forever.`),me=m(),M=i("div"),this.h()},l(e){v=r(e,"P",{});var s=h(v);$=a(s,`In this chapter we are going to look at some terminology that is used
    throughout reinforcement learning and at how reinforcement learning
    algorithms can be classified based on the components the agent has.`),s.forEach(t),w=p(e),d=r(e,"DIV",{class:!0}),h(d).forEach(t),k=p(e),u=r(e,"H2",{});var Ke=h(u);b=a(Ke,"Value Based vs Policy Based"),Ke.forEach(t),f=p(e),c=r(e,"P",{});var I=h(c);G=a(I,`Especially beginner level reinforcement learning agents have only a value
    function. In that case the policy of the agent is implicitly derived from
    the value function. Reinforcement learning methods that only utilize the
    value function are called `),A=r(I,"STRONG",{});var Qe=h(A);ve=a(Qe,"value based methods"),Qe.forEach(t),ue=a(I,`. If on the
    other hand the agent derives the policy directly without using value
    functions the methods are called `),L=r(I,"STRONG",{});var Ue=h(L);de=a(Ue,"policy based methods"),Ue.forEach(t),ge=a(I,`. Most
    modern algorithms have agents with both components. Those are called
    `),W=r(I,"STRONG",{});var Xe=h(W);ye=a(Xe,"actor-critic methods"),Xe.forEach(t),_e=a(I,"."),I.forEach(t),ee=p(e),N=r(e,"DIV",{class:!0}),h(N).forEach(t),te=p(e),O=r(e,"H2",{});var Ye=h(O);Ee=a(Ye,"Model Based vs Model Free"),Ye.forEach(t),ne=p(e),_=r(e,"P",{});var H=h(_);we=a(H,`If the agent has an internal representation of the model as a component, the
    algorithms are called `),F=r(H,"STRONG",{});var Ze=h(F);ke=a(Ze,"model based"),Ze.forEach(t),be=a(H,`. If the agent learns the
    policy without a model, the algorithms are called
    `),q=r(H,"STRONG",{});var et=h(q);Ie=a(et,"model free"),et.forEach(t),Te=a(H,"."),H.forEach(t),oe=p(e),P=r(e,"DIV",{class:!0}),h(P).forEach(t),ae=p(e),C=r(e,"H2",{});var tt=h(C);Re=a(tt,"Learning vs Planning"),tt.forEach(t),le=p(e),E=r(e,"P",{});var z=h(E);$e=a(z,`In many cases the agent has no access to the model of the environment or has
    no internal representation of the model. Therefore the agent has to interact
    with the environment to improve the policy. In reinforcement learning this
    is called `),j=r(z,"STRONG",{});var nt=h(j);Se=a(nt,"learning"),nt.forEach(t),Ge=a(z,`. If the agent on the other hand utilizes
    the model to improve the value function or the policy, we call it
    `),J=r(z,"STRONG",{});var ot=h(J);Ne=a(ot,"planning"),ot.forEach(t),Oe=a(z,"."),z.forEach(t),se=p(e),V=r(e,"DIV",{class:!0}),h(V).forEach(t),ie=p(e),x=r(e,"H2",{});var at=h(x);Pe=a(at,"Prediction vs Improvement vs Control"),at.forEach(t),re=p(e),g=r(e,"P",{});var T=h(g);Ce=a(T,"There are several tasks an agent might need to perform. We talk about the "),K=r(T,"STRONG",{});var lt=h(K);Ve=a(lt,"prediction"),lt.forEach(t),xe=a(T,`
    task when we have a certain policy and the agent has to calculate the exact value
    function for that policy. We talk about the `),Q=r(T,"STRONG",{});var st=h(Q);Be=a(st,"improvement"),st.forEach(t),De=a(T,`
    task when we want to improve a given policy. And we talk about the
    `),U=r(T,"STRONG",{});var it=h(U);Me=a(it,"control"),it.forEach(t),He=a(T,` task when the agent has to find the best possible policy.
    As you can imagine, prediction and improvement are necessary steps to solve the
    control problem.`),T.forEach(t),he=p(e),B=r(e,"DIV",{class:!0}),h(B).forEach(t),fe=p(e),D=r(e,"H2",{});var rt=h(D);ze=a(rt,"Episodic vs Continuing Tasks"),rt.forEach(t),ce=p(e),y=r(e,"P",{});var R=h(y);Ae=a(R,`In reinforcement learning we distinguish between episodic and continuing
    tasks. `),X=r(R,"STRONG",{});var ht=h(X);Le=a(ht,"Episodic"),ht.forEach(t),We=a(R,` tasks are tasks that have a natural ending.
    The last state in an episodic task is called a `),Y=r(R,"EM",{});var ft=h(Y);Fe=a(ft,"terminal state"),ft.forEach(t),qe=a(R,`.
    `),Z=r(R,"STRONG",{});var ct=h(Z);je=a(ct,"Continuing"),ct.forEach(t),Je=a(R,` tasks are tasks that do not have a natural ending
    and may theoretically go on forever.`),R.forEach(t),me=p(e),M=r(e,"DIV",{class:!0}),h(M).forEach(t),this.h()},h(){S(d,"class","separator"),S(N,"class","separator"),S(P,"class","separator"),S(V,"class","separator"),S(B,"class","separator"),S(M,"class","separator")},m(e,s){l(e,v,s),n(v,$),l(e,w,s),l(e,d,s),l(e,k,s),l(e,u,s),n(u,b),l(e,f,s),l(e,c,s),n(c,G),n(c,A),n(A,ve),n(c,ue),n(c,L),n(L,de),n(c,ge),n(c,W),n(W,ye),n(c,_e),l(e,ee,s),l(e,N,s),l(e,te,s),l(e,O,s),n(O,Ee),l(e,ne,s),l(e,_,s),n(_,we),n(_,F),n(F,ke),n(_,be),n(_,q),n(q,Ie),n(_,Te),l(e,oe,s),l(e,P,s),l(e,ae,s),l(e,C,s),n(C,Re),l(e,le,s),l(e,E,s),n(E,$e),n(E,j),n(j,Se),n(E,Ge),n(E,J),n(J,Ne),n(E,Oe),l(e,se,s),l(e,V,s),l(e,ie,s),l(e,x,s),n(x,Pe),l(e,re,s),l(e,g,s),n(g,Ce),n(g,K),n(K,Ve),n(g,xe),n(g,Q),n(Q,Be),n(g,De),n(g,U),n(U,Me),n(g,He),l(e,he,s),l(e,B,s),l(e,fe,s),l(e,D,s),n(D,ze),l(e,ce,s),l(e,y,s),n(y,Ae),n(y,X),n(X,Le),n(y,We),n(y,Y),n(Y,Fe),n(y,qe),n(y,Z),n(Z,je),n(y,Je),l(e,me,s),l(e,M,s)},p:wt,d(e){e&&t(v),e&&t(w),e&&t(d),e&&t(k),e&&t(u),e&&t(f),e&&t(c),e&&t(ee),e&&t(N),e&&t(te),e&&t(O),e&&t(ne),e&&t(_),e&&t(oe),e&&t(P),e&&t(ae),e&&t(C),e&&t(le),e&&t(E),e&&t(se),e&&t(V),e&&t(ie),e&&t(x),e&&t(re),e&&t(g),e&&t(he),e&&t(B),e&&t(fe),e&&t(D),e&&t(ce),e&&t(y),e&&t(me),e&&t(M)}}}function It(pe){let v,$,w,d,k,u,b;return u=new kt({props:{$$slots:{default:[bt]},$$scope:{ctx:pe}}}),{c(){v=i("h1"),$=o("Reinforcement Learning Terminology"),w=m(),d=i("div"),k=m(),ut(u.$$.fragment),this.h()},l(f){v=r(f,"H1",{});var c=h(v);$=a(c,"Reinforcement Learning Terminology"),c.forEach(t),w=p(f),d=r(f,"DIV",{class:!0}),h(d).forEach(t),k=p(f),dt(u.$$.fragment,f),this.h()},h(){S(d,"class","separator")},m(f,c){l(f,v,c),n(v,$),l(f,w,c),l(f,d,c),l(f,k,c),gt(u,f,c),b=!0},p(f,[c]){const G={};c&1&&(G.$$scope={dirty:c,ctx:f}),u.$set(G)},i(f){b||(yt(u.$$.fragment,f),b=!0)},o(f){_t(u.$$.fragment,f),b=!1},d(f){f&&t(v),f&&t(w),f&&t(d),f&&t(k),Et(u,f)}}}class $t extends mt{constructor(v){super(),pt(this,v,null,It,vt,{})}}export{$t as default};
