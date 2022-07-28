import{S as lt,i as pt,s as ft,l as _,a as m,r as s,w as $,T as ht,m as v,h as n,c,n as T,u as l,x as w,p as ye,G as h,b as o,y as b,f as y,t as g,B as x,E as fe}from"../../../../chunks/index-caa95cd4.js";import{C as ut}from"../../../../chunks/Container-5c6b7f6d.js";import{L as z}from"../../../../chunks/Latex-bf74aeea.js";import{M as st}from"../../../../chunks/MemoryBuffer-d03c34ea.js";import"../../../../chunks/Button-13b097c1.js";function mt(p){let r=String.raw`\delta = r + \gamma \max_{a'} Q(s', a', \mathbf{w}^-) - Q(s, a, \mathbf{w})`+"",t;return{c(){t=s(r)},l(a){t=l(a,r)},m(a,u){o(a,t,u)},p:fe,d(a){a&&n(t)}}}function ct(p){let r;return{c(){r=s("| \\delta |")},l(t){r=l(t,"| \\delta |")},m(t,a){o(t,r,a)},d(t){t&&n(r)}}}function dt(p){let r;return{c(){r=s("p_i = |\\delta_i| + \\epsilon")},l(t){r=l(t,"p_i = |\\delta_i| + \\epsilon")},m(t,a){o(t,r,a)},d(t){t&&n(r)}}}function $t(p){let r;return{c(){r=s("\\epsilon")},l(t){r=l(t,"\\epsilon")},m(t,a){o(t,r,a)},d(t){t&&n(r)}}}function wt(p){let r=String.raw`p_i = \frac{1}{rank(i)}`+"",t;return{c(){t=s(r)},l(a){t=l(a,r)},m(a,u){o(a,t,u)},p:fe,d(a){a&&n(t)}}}function bt(p){let r;return{c(){r=s("p_i")},l(t){r=l(t,"p_i")},m(t,a){o(t,r,a)},d(t){t&&n(r)}}}function yt(p){let r;return{c(){r=s("\\alpha")},l(t){r=l(t,"\\alpha")},m(t,a){o(t,r,a)},d(t){t&&n(r)}}}function gt(p){let r;return{c(){r=s("\\alpha")},l(t){r=l(t,"\\alpha")},m(t,a){o(t,r,a)},d(t){t&&n(r)}}}function xt(p){let r;return{c(){r=s("\\alpha")},l(t){r=l(t,"\\alpha")},m(t,a){o(t,r,a)},d(t){t&&n(r)}}}function _t(p){let r=String.raw`P(i) = (p^{\alpha}_i) / (\sum_k p^{\alpha}_k)`+"",t;return{c(){t=s(r)},l(a){t=l(a,r)},m(a,u){o(a,t,u)},p:fe,d(a){a&&n(t)}}}function vt(p){let r=String.raw`w_i = (N \cdot P(i))^{-\beta}`+"",t;return{c(){t=s(r)},l(a){t=l(a,r)},m(a,u){o(a,t,u)},p:fe,d(a){a&&n(t)}}}function Tt(p){let r=String.raw`{P(i)} `+"",t;return{c(){t=s(r)},l(a){t=l(a,r)},m(a,u){o(a,t,u)},p:fe,d(a){a&&n(t)}}}function zt(p){let r=String.raw`{\frac{1}{N}}`+"",t;return{c(){t=s(r)},l(a){t=l(a,r)},m(a,u){o(a,t,u)},p:fe,d(a){a&&n(t)}}}function Et(p){let r;return{c(){r=s("\\beta")},l(t){r=l(t,"\\beta")},m(t,a){o(t,r,a)},d(t){t&&n(r)}}}function kt(p){let r;return{c(){r=s("\\beta")},l(t){r=l(t,"\\beta")},m(t,a){o(t,r,a)},d(t){t&&n(r)}}}function Dt(p){let r,t,a,u,S,D,A,P,R,f,d,L,ge,ne,Me,xe,te,_e,ae,Ae,ve,B,Te,oe,Be,ze,F,Fe,N,Ne,Ee,se,Qe,ke,j,Ce,Q,He,C,We,De,H,Oe,W,Ve,Pe,E,Ye,O,Ge,V,Je,Y,Ke,G,Ue,Ie,J,qe,le,Xe,Re,pe,Ze,je,K,Se,k,et,U,tt,X,rt,Z,it,ee,nt,Le;return R=new st({props:{prioritized:!1}}),te=new st({props:{prioritized:!0}}),B=new z({props:{$$slots:{default:[mt]},$$scope:{ctx:p}}}),N=new z({props:{$$slots:{default:[ct]},$$scope:{ctx:p}}}),Q=new z({props:{$$slots:{default:[dt]},$$scope:{ctx:p}}}),C=new z({props:{$$slots:{default:[$t]},$$scope:{ctx:p}}}),W=new z({props:{$$slots:{default:[wt]},$$scope:{ctx:p}}}),O=new z({props:{$$slots:{default:[bt]},$$scope:{ctx:p}}}),V=new z({props:{$$slots:{default:[yt]},$$scope:{ctx:p}}}),Y=new z({props:{$$slots:{default:[gt]},$$scope:{ctx:p}}}),G=new z({props:{$$slots:{default:[xt]},$$scope:{ctx:p}}}),J=new z({props:{$$slots:{default:[_t]},$$scope:{ctx:p}}}),K=new z({props:{$$slots:{default:[vt]},$$scope:{ctx:p}}}),U=new z({props:{$$slots:{default:[Tt]},$$scope:{ctx:p}}}),X=new z({props:{$$slots:{default:[zt]},$$scope:{ctx:p}}}),Z=new z({props:{$$slots:{default:[Et]},$$scope:{ctx:p}}}),ee=new z({props:{$$slots:{default:[kt]},$$scope:{ctx:p}}}),{c(){r=_("p"),t=s(`"In this paper we develop a framework for prioritizing experience, so as to
    replay important transitions more frequently, and therefore learn more
    efficiently.`),a=m(),u=_("div"),S=m(),D=_("p"),A=s(`The experience replay is one of the major DQN components that make the
    algorithm so efficient. The agent is able to store already seen experiences
    and to reuse them several times in training before they are discarded. Each
    of the past experiences has the same chance of being drawn and being used
    for training.`),P=m(),$(R.$$.fragment),f=m(),d=_("p"),L=s(`But the uniform distribution with with the experiences are drawn is also the
    drawback of the experience replay. It is reasonable to assume that some
    experiences are more important and therefore better suited to learn from.
    This is where the prioritized experience replay (PER) comes into play. Each
    of the experience tuples has a priority assigned to it and the probability
    with which the tuple is likely to be drawn from the replay buffer and be
    used in training increases with higher priority. That way more important
    experiences are used more often and contribute to faster learning of the
    agent.`),ge=m(),ne=_("p"),Me=s(`The interactive example below is an attempt to visualize to procedure of the
    prioritized experience replay technique. You will notice that each of the
    experiences in the memory buffer have a different size. This size represents
    the priority of the experiece and has therefore a higher probability to be
    drawn. Each time an experience is drawn, the priority is reduced, because we
    generally expect that the agent has already learned from that particular
    experience. Additionally we can observe that the newest experiences have
    generally extremely high priority. This is because each new experience gets
    automatically a the highest priority to make sure that it is used at least
    once for learning.`),xe=m(),$(te.$$.fragment),_e=m(),ae=_("p"),Ae=s(`The creators of PER mention, that the ideal quantity of priority would be a
    measure of how much an agent can learn from a given experience. Such a
    measure is obviously not available and TD error is used as a proxy of
    priority.`),ve=m(),$(B.$$.fragment),Te=m(),oe=_("p"),Be=s(`This is obviously not an ideal measure, because the TD error is based on a
    noisy estimate, but we will assume that this is valid approximation of the
    importance of an experience tuple.`),ze=m(),F=_("p"),Fe=s(`TD error can be positive or negative, but we are only interested in the
    magnitude of the error and not in the direction. Therefore the absolute
    value of the error, `),$(N.$$.fragment),Ne=s(", is going to be used."),Ee=m(),se=_("p"),Qe=s(`If we sampled only according to the magnitude of TD error, some experiences
    would not be sampled at all before they are discarded. This is especially
    problematic when we consider that we bootstrap and have only access to
    estimates of TD errors. Therefore we generally want to sample experiences
    with high TD error more often, but still have a non zero probability for
    experiences with low TD errors. DeepMind proposes two approaches to
    calculate priorities.`),ke=m(),j=_("p"),Ce=s("Proportilan priorization: "),$(Q.$$.fragment),He=s(", where "),$(C.$$.fragment),We=s(` is a positive constant that makes sure that experience tuples with a TD error
    of 0 still have a non-zero percent probability of being selected.`),De=m(),H=_("p"),Oe=s("Ranked-based priorization: "),$(W.$$.fragment),Ve=s(`, where rank(i) is the index number of an experience tuple in a list, in
    which all absolute TD errors are sorted in descending order. Ranked-based
    prioritization is expected to be less sensitive to outliers, therefore this
    approach is going to be utilized in this chapter.`),Pe=m(),E=_("p"),Ye=s(`The distribution is not only determined by the priority
    `),$(O.$$.fragment),Ge=s(`, but is additionally controlled by a constant
    `),$(V.$$.fragment),Je=s(". If "),$(Y.$$.fragment),Ke=s(` is 0 we are essentially facing
    a uniform distribution. Higher numbers of `),$(G.$$.fragment),Ue=s(` increase the
    to higher importance of priorities.`),Ie=m(),$(J.$$.fragment),qe=m(),le=_("p"),Xe=s(`Measuring TD errors for all experience tuples at each time step would be
    extremely inefficient, therefore the updates are done only periodically. The
    TD errors are updated only once they are drawn from the memory buffer and
    used in the training step. This is due to the fact that TD errors have to be
    calculated at the training step anyway and no additional computational power
    is therefore required. The calculations are not done for new experiences
    therefore each new experience tuple will receive the highest possible
    priority.`),Re=m(),pe=_("p"),Ze=s(`If we are not careful and keep using the prioritized experience replay
    without any adjustment to the update step, we will introduce a bias. Let us
    assume that we possess the weights of the policy that minimize the mean
    squared error for the optimal policy. We utilize the policy and interact
    with the environment to fill the replay buffer. Lastly we want to recreate
    the weights for the above mentioned policy using the filled replay buffer.
    If we use the prioritized experience replay we utilize a different
    distribution than the one that is implied by the optimal weights, which is
    the uniform distribution. For example we might draw rare experiences more
    often, which would imply gradient descent steps calculated based on rare
    experiences more often. On the one hand we want to use important experiences
    more often, but we would also like to avoid the bias, especially in the long
    run. For that purpose we adjust the gradient descent step by a weight
    factor.`),je=m(),$(K.$$.fragment),Se=m(),k=_("p"),et=s(`The simplest way to imagine why the adjustment works is to imagine that we
    have uniform distribution.`),$(U.$$.fragment),tt=s(` becomes
    `),$(X.$$.fragment),rt=s(` and the whole expression amounts to
    1, indicating that the uniform distribution is already the correct one and we
    do not need any adjustments. The `),$(Z.$$.fragment),it=s(` factor is used to control
    the correction factor. The requirement that we would like to impose is the uniform
    distribution at the end of the training. Therefore we start with a low `),$(ee.$$.fragment),nt=s(`
    and allow for stronger updates towards the rare experiences and increase the
    value over time to make full corrections.`),this.h()},l(e){r=v(e,"P",{class:!0});var i=T(r);t=l(i,`"In this paper we develop a framework for prioritizing experience, so as to
    replay important transitions more frequently, and therefore learn more
    efficiently.`),i.forEach(n),a=c(e),u=v(e,"DIV",{class:!0}),T(u).forEach(n),S=c(e),D=v(e,"P",{});var he=T(D);A=l(he,`The experience replay is one of the major DQN components that make the
    algorithm so efficient. The agent is able to store already seen experiences
    and to reuse them several times in training before they are discarded. Each
    of the past experiences has the same chance of being drawn and being used
    for training.`),he.forEach(n),P=c(e),w(R.$$.fragment,e),f=c(e),d=v(e,"P",{});var ue=T(d);L=l(ue,`But the uniform distribution with with the experiences are drawn is also the
    drawback of the experience replay. It is reasonable to assume that some
    experiences are more important and therefore better suited to learn from.
    This is where the prioritized experience replay (PER) comes into play. Each
    of the experience tuples has a priority assigned to it and the probability
    with which the tuple is likely to be drawn from the replay buffer and be
    used in training increases with higher priority. That way more important
    experiences are used more often and contribute to faster learning of the
    agent.`),ue.forEach(n),ge=c(e),ne=v(e,"P",{});var me=T(ne);Me=l(me,`The interactive example below is an attempt to visualize to procedure of the
    prioritized experience replay technique. You will notice that each of the
    experiences in the memory buffer have a different size. This size represents
    the priority of the experiece and has therefore a higher probability to be
    drawn. Each time an experience is drawn, the priority is reduced, because we
    generally expect that the agent has already learned from that particular
    experience. Additionally we can observe that the newest experiences have
    generally extremely high priority. This is because each new experience gets
    automatically a the highest priority to make sure that it is used at least
    once for learning.`),me.forEach(n),xe=c(e),w(te.$$.fragment,e),_e=c(e),ae=v(e,"P",{});var ce=T(ae);Ae=l(ce,`The creators of PER mention, that the ideal quantity of priority would be a
    measure of how much an agent can learn from a given experience. Such a
    measure is obviously not available and TD error is used as a proxy of
    priority.`),ce.forEach(n),ve=c(e),w(B.$$.fragment,e),Te=c(e),oe=v(e,"P",{});var de=T(oe);Be=l(de,`This is obviously not an ideal measure, because the TD error is based on a
    noisy estimate, but we will assume that this is valid approximation of the
    importance of an experience tuple.`),de.forEach(n),ze=c(e),F=v(e,"P",{});var re=T(F);Fe=l(re,`TD error can be positive or negative, but we are only interested in the
    magnitude of the error and not in the direction. Therefore the absolute
    value of the error, `),w(N.$$.fragment,re),Ne=l(re,", is going to be used."),re.forEach(n),Ee=c(e),se=v(e,"P",{});var $e=T(se);Qe=l($e,`If we sampled only according to the magnitude of TD error, some experiences
    would not be sampled at all before they are discarded. This is especially
    problematic when we consider that we bootstrap and have only access to
    estimates of TD errors. Therefore we generally want to sample experiences
    with high TD error more often, but still have a non zero probability for
    experiences with low TD errors. DeepMind proposes two approaches to
    calculate priorities.`),$e.forEach(n),ke=c(e),j=v(e,"P",{});var M=T(j);Ce=l(M,"Proportilan priorization: "),w(Q.$$.fragment,M),He=l(M,", where "),w(C.$$.fragment,M),We=l(M,` is a positive constant that makes sure that experience tuples with a TD error
    of 0 still have a non-zero percent probability of being selected.`),M.forEach(n),De=c(e),H=v(e,"P",{});var ie=T(H);Oe=l(ie,"Ranked-based priorization: "),w(W.$$.fragment,ie),Ve=l(ie,`, where rank(i) is the index number of an experience tuple in a list, in
    which all absolute TD errors are sorted in descending order. Ranked-based
    prioritization is expected to be less sensitive to outliers, therefore this
    approach is going to be utilized in this chapter.`),ie.forEach(n),Pe=c(e),E=v(e,"P",{});var I=T(E);Ye=l(I,`The distribution is not only determined by the priority
    `),w(O.$$.fragment,I),Ge=l(I,`, but is additionally controlled by a constant
    `),w(V.$$.fragment,I),Je=l(I,". If "),w(Y.$$.fragment,I),Ke=l(I,` is 0 we are essentially facing
    a uniform distribution. Higher numbers of `),w(G.$$.fragment,I),Ue=l(I,` increase the
    to higher importance of priorities.`),I.forEach(n),Ie=c(e),w(J.$$.fragment,e),qe=c(e),le=v(e,"P",{});var we=T(le);Xe=l(we,`Measuring TD errors for all experience tuples at each time step would be
    extremely inefficient, therefore the updates are done only periodically. The
    TD errors are updated only once they are drawn from the memory buffer and
    used in the training step. This is due to the fact that TD errors have to be
    calculated at the training step anyway and no additional computational power
    is therefore required. The calculations are not done for new experiences
    therefore each new experience tuple will receive the highest possible
    priority.`),we.forEach(n),Re=c(e),pe=v(e,"P",{});var be=T(pe);Ze=l(be,`If we are not careful and keep using the prioritized experience replay
    without any adjustment to the update step, we will introduce a bias. Let us
    assume that we possess the weights of the policy that minimize the mean
    squared error for the optimal policy. We utilize the policy and interact
    with the environment to fill the replay buffer. Lastly we want to recreate
    the weights for the above mentioned policy using the filled replay buffer.
    If we use the prioritized experience replay we utilize a different
    distribution than the one that is implied by the optimal weights, which is
    the uniform distribution. For example we might draw rare experiences more
    often, which would imply gradient descent steps calculated based on rare
    experiences more often. On the one hand we want to use important experiences
    more often, but we would also like to avoid the bias, especially in the long
    run. For that purpose we adjust the gradient descent step by a weight
    factor.`),be.forEach(n),je=c(e),w(K.$$.fragment,e),Se=c(e),k=v(e,"P",{});var q=T(k);et=l(q,`The simplest way to imagine why the adjustment works is to imagine that we
    have uniform distribution.`),w(U.$$.fragment,q),tt=l(q,` becomes
    `),w(X.$$.fragment,q),rt=l(q,` and the whole expression amounts to
    1, indicating that the uniform distribution is already the correct one and we
    do not need any adjustments. The `),w(Z.$$.fragment,q),it=l(q,` factor is used to control
    the correction factor. The requirement that we would like to impose is the uniform
    distribution at the end of the training. Therefore we start with a low `),w(ee.$$.fragment,q),nt=l(q,`
    and allow for stronger updates towards the rare experiences and increase the
    value over time to make full corrections.`),q.forEach(n),this.h()},h(){ye(r,"class","info"),ye(u,"class","separator")},m(e,i){o(e,r,i),h(r,t),o(e,a,i),o(e,u,i),o(e,S,i),o(e,D,i),h(D,A),o(e,P,i),b(R,e,i),o(e,f,i),o(e,d,i),h(d,L),o(e,ge,i),o(e,ne,i),h(ne,Me),o(e,xe,i),b(te,e,i),o(e,_e,i),o(e,ae,i),h(ae,Ae),o(e,ve,i),b(B,e,i),o(e,Te,i),o(e,oe,i),h(oe,Be),o(e,ze,i),o(e,F,i),h(F,Fe),b(N,F,null),h(F,Ne),o(e,Ee,i),o(e,se,i),h(se,Qe),o(e,ke,i),o(e,j,i),h(j,Ce),b(Q,j,null),h(j,He),b(C,j,null),h(j,We),o(e,De,i),o(e,H,i),h(H,Oe),b(W,H,null),h(H,Ve),o(e,Pe,i),o(e,E,i),h(E,Ye),b(O,E,null),h(E,Ge),b(V,E,null),h(E,Je),b(Y,E,null),h(E,Ke),b(G,E,null),h(E,Ue),o(e,Ie,i),b(J,e,i),o(e,qe,i),o(e,le,i),h(le,Xe),o(e,Re,i),o(e,pe,i),h(pe,Ze),o(e,je,i),b(K,e,i),o(e,Se,i),o(e,k,i),h(k,et),b(U,k,null),h(k,tt),b(X,k,null),h(k,rt),b(Z,k,null),h(k,it),b(ee,k,null),h(k,nt),Le=!0},p(e,i){const he={};i&1&&(he.$$scope={dirty:i,ctx:e}),B.$set(he);const ue={};i&1&&(ue.$$scope={dirty:i,ctx:e}),N.$set(ue);const me={};i&1&&(me.$$scope={dirty:i,ctx:e}),Q.$set(me);const ce={};i&1&&(ce.$$scope={dirty:i,ctx:e}),C.$set(ce);const de={};i&1&&(de.$$scope={dirty:i,ctx:e}),W.$set(de);const re={};i&1&&(re.$$scope={dirty:i,ctx:e}),O.$set(re);const $e={};i&1&&($e.$$scope={dirty:i,ctx:e}),V.$set($e);const M={};i&1&&(M.$$scope={dirty:i,ctx:e}),Y.$set(M);const ie={};i&1&&(ie.$$scope={dirty:i,ctx:e}),G.$set(ie);const I={};i&1&&(I.$$scope={dirty:i,ctx:e}),J.$set(I);const we={};i&1&&(we.$$scope={dirty:i,ctx:e}),K.$set(we);const be={};i&1&&(be.$$scope={dirty:i,ctx:e}),U.$set(be);const q={};i&1&&(q.$$scope={dirty:i,ctx:e}),X.$set(q);const at={};i&1&&(at.$$scope={dirty:i,ctx:e}),Z.$set(at);const ot={};i&1&&(ot.$$scope={dirty:i,ctx:e}),ee.$set(ot)},i(e){Le||(y(R.$$.fragment,e),y(te.$$.fragment,e),y(B.$$.fragment,e),y(N.$$.fragment,e),y(Q.$$.fragment,e),y(C.$$.fragment,e),y(W.$$.fragment,e),y(O.$$.fragment,e),y(V.$$.fragment,e),y(Y.$$.fragment,e),y(G.$$.fragment,e),y(J.$$.fragment,e),y(K.$$.fragment,e),y(U.$$.fragment,e),y(X.$$.fragment,e),y(Z.$$.fragment,e),y(ee.$$.fragment,e),Le=!0)},o(e){g(R.$$.fragment,e),g(te.$$.fragment,e),g(B.$$.fragment,e),g(N.$$.fragment,e),g(Q.$$.fragment,e),g(C.$$.fragment,e),g(W.$$.fragment,e),g(O.$$.fragment,e),g(V.$$.fragment,e),g(Y.$$.fragment,e),g(G.$$.fragment,e),g(J.$$.fragment,e),g(K.$$.fragment,e),g(U.$$.fragment,e),g(X.$$.fragment,e),g(Z.$$.fragment,e),g(ee.$$.fragment,e),Le=!1},d(e){e&&n(r),e&&n(a),e&&n(u),e&&n(S),e&&n(D),e&&n(P),x(R,e),e&&n(f),e&&n(d),e&&n(ge),e&&n(ne),e&&n(xe),x(te,e),e&&n(_e),e&&n(ae),e&&n(ve),x(B,e),e&&n(Te),e&&n(oe),e&&n(ze),e&&n(F),x(N),e&&n(Ee),e&&n(se),e&&n(ke),e&&n(j),x(Q),x(C),e&&n(De),e&&n(H),x(W),e&&n(Pe),e&&n(E),x(O),x(V),x(Y),x(G),e&&n(Ie),x(J,e),e&&n(qe),e&&n(le),e&&n(Re),e&&n(pe),e&&n(je),x(K,e),e&&n(Se),e&&n(k),x(U),x(X),x(Z),x(ee)}}}function Pt(p){let r,t,a,u,S,D,A,P,R;return P=new ut({props:{$$slots:{default:[Dt]},$$scope:{ctx:p}}}),{c(){r=_("meta"),t=m(),a=_("h1"),u=s("Prioritized Experience Replay"),S=m(),D=_("div"),A=m(),$(P.$$.fragment),this.h()},l(f){const d=ht('[data-svelte="svelte-cwpdpa"]',document.head);r=v(d,"META",{name:!0,content:!0}),d.forEach(n),t=c(f),a=v(f,"H1",{});var L=T(a);u=l(L,"Prioritized Experience Replay"),L.forEach(n),S=c(f),D=v(f,"DIV",{class:!0}),T(D).forEach(n),A=c(f),w(P.$$.fragment,f),this.h()},h(){document.title="World4AI | Reinforcement Learning | Prioritized Experience Replay",ye(r,"name","description"),ye(r,"content","The prioritized experience replay (PER) adjust the memory buffer in such a way, that assigns each experience tuple a priority. Experience tuples with higher priorities are used more often for training."),ye(D,"class","separator")},m(f,d){h(document.head,r),o(f,t,d),o(f,a,d),h(a,u),o(f,S,d),o(f,D,d),o(f,A,d),b(P,f,d),R=!0},p(f,[d]){const L={};d&1&&(L.$$scope={dirty:d,ctx:f}),P.$set(L)},i(f){R||(y(P.$$.fragment,f),R=!0)},o(f){g(P.$$.fragment,f),R=!1},d(f){n(r),f&&n(t),f&&n(a),f&&n(S),f&&n(D),f&&n(A),x(P,f)}}}class Lt extends lt{constructor(r){super(),pt(this,r,null,Pt,ft,{})}}export{Lt as default};
