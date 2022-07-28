import{S as ce,i as me,s as pe,l as c,a as l,r as $,w as S,T as ve,m,h as t,c as r,n as d,u as x,x as V,p as O,G as w,b as o,y as R,f as W,t as Y,B as F,E as de}from"../../../chunks/index-caa95cd4.js";import{C as be}from"../../../chunks/Container-5c6b7f6d.js";import{T as ne}from"../../../chunks/Table-a51fb5ea.js";import{C as we}from"../../../chunks/CartPole-8bd7166b.js";function ge(g){let s,_,p,v,y,u,k,f,b,a,i,h,q,E,U,z,A,X,Q,I,B,D,Z,G,P,ee,j,M,te,L,T,J,C,ae,K,H,N;return v=new ne({props:{header:g[0],data:g[1]}}),h=new ne({props:{header:g[2],data:g[3]}}),I=new ne({props:{header:g[4],data:g[5]}}),T=new we({}),{c(){s=c("p"),_=$(`So far we have dealt with tabular reinforcement learning and finite Markov
    decision processes.`),p=l(),S(v.$$.fragment),y=l(),u=c("p"),k=$(`The number of rows and columns in the Q-Table was finite. This allowed us to
    loop over all state-action pairs and apply Monte Carlo or temporal
    difference learning. Given enough iterations we were guaranteed to arrive at
    the optimal solution.`),f=l(),b=c("p"),a=$(`Most interesting reinforcement learning problems do not have such nice
    properties. In case state or action sets are infinite or extremely large it
    becomes impossible to store the value function as a table.`),i=l(),S(h.$$.fragment),q=l(),E=c("p"),U=$(`The above table shows action-values for 1,000,000,000 discrete states. Even
    if we possessed a computer which could efficiently store a high amount of
    states, we still need to loop over all these states to improve the
    estimation and thus convergence would be extremely slow.`),z=l(),A=c("p"),X=$(`Representing a value function in Q-tables become almost impossible when the
    agent has to deal with continuous variables. When the agent encounters a
    continuous observation it is extremely unlikely that the same exact value
    will be seen again. Yet the agent will need to learn how to deal with future
    unseen observations. We expect from the agent to find a policy that is
    \u201Cgood\u201D across many different observations. The key word that we are looking
    for is generalization.`),Q=l(),S(I.$$.fragment),B=l(),D=c("p"),Z=$(`The example above shows how generalization might look like. The state is
    represented by 1 single continuous variable and there are only 2 discrete
    actions available (left and right). If you look at the state representation
    with the value of 1.7, could you approximate the action-values and determine
    which action has the higher value? You will probably not get the exact
    correct answer but the value for the left action should probably be
    somewhere between 1.5 and 1.8. Real reinforcement learning tasks might be a
    lot more complex, but I it is not a bad mental model to imagine the agent
    interpolating between states that were already encountered.`),G=l(),P=c("p"),ee=$(`In the following sections we are going to deal with Markov decision
    processes that have a large or continuous state space, but a small discrete
    number of actions. Due to the unlimited number of states it becomes
    necessary to create value functions that are not exact, but approximative,
    meaning that the value function does not return the true value of a policy
    but a value that is hopefully close enough. Finding the optimal policy and
    value function is often not possible, but generalally we will attempt to
    find a policy that still performs in a way that generates a relatively high
    expected sum of rewards.`),j=l(),M=c("p"),te=$(`In the cart pole environment below the task is to balance the pole and not
    to move too far away from the center of the screen. At each timestep the
    agent can either move left or right and gets a reward of +1. If the angle of
    the pole is too large or the cart goes offscreen the game ends. The agent
    can observe the car position, the cart velocity, the angle of the pole and
    the angular velocity. Each of the variables is continuous and it is
    therefore becomes necessary to approximate the value function.`),L=l(),S(T.$$.fragment),J=l(),C=c("p"),ae=$(`Before we move on to the next section, we should mention the limitation of
    approximative value based methods. Value based approximative methods are
    able to deal with a continuous state space, but not continuous actions. Yet
    as you can imagine many of the tasks a robot for example would need to
    perform require continuous actions. In later chapters we will discuss how we
    can deal with these limitations.`),K=l(),H=c("div"),this.h()},l(e){s=m(e,"P",{});var n=d(s);_=x(n,`So far we have dealt with tabular reinforcement learning and finite Markov
    decision processes.`),n.forEach(t),p=r(e),V(v.$$.fragment,e),y=r(e),u=m(e,"P",{});var oe=d(u);k=x(oe,`The number of rows and columns in the Q-Table was finite. This allowed us to
    loop over all state-action pairs and apply Monte Carlo or temporal
    difference learning. Given enough iterations we were guaranteed to arrive at
    the optimal solution.`),oe.forEach(t),f=r(e),b=m(e,"P",{});var ie=d(b);a=x(ie,`Most interesting reinforcement learning problems do not have such nice
    properties. In case state or action sets are infinite or extremely large it
    becomes impossible to store the value function as a table.`),ie.forEach(t),i=r(e),V(h.$$.fragment,e),q=r(e),E=m(e,"P",{});var se=d(E);U=x(se,`The above table shows action-values for 1,000,000,000 discrete states. Even
    if we possessed a computer which could efficiently store a high amount of
    states, we still need to loop over all these states to improve the
    estimation and thus convergence would be extremely slow.`),se.forEach(t),z=r(e),A=m(e,"P",{});var le=d(A);X=x(le,`Representing a value function in Q-tables become almost impossible when the
    agent has to deal with continuous variables. When the agent encounters a
    continuous observation it is extremely unlikely that the same exact value
    will be seen again. Yet the agent will need to learn how to deal with future
    unseen observations. We expect from the agent to find a policy that is
    \u201Cgood\u201D across many different observations. The key word that we are looking
    for is generalization.`),le.forEach(t),Q=r(e),V(I.$$.fragment,e),B=r(e),D=m(e,"P",{});var re=d(D);Z=x(re,`The example above shows how generalization might look like. The state is
    represented by 1 single continuous variable and there are only 2 discrete
    actions available (left and right). If you look at the state representation
    with the value of 1.7, could you approximate the action-values and determine
    which action has the higher value? You will probably not get the exact
    correct answer but the value for the left action should probably be
    somewhere between 1.5 and 1.8. Real reinforcement learning tasks might be a
    lot more complex, but I it is not a bad mental model to imagine the agent
    interpolating between states that were already encountered.`),re.forEach(t),G=r(e),P=m(e,"P",{});var ue=d(P);ee=x(ue,`In the following sections we are going to deal with Markov decision
    processes that have a large or continuous state space, but a small discrete
    number of actions. Due to the unlimited number of states it becomes
    necessary to create value functions that are not exact, but approximative,
    meaning that the value function does not return the true value of a policy
    but a value that is hopefully close enough. Finding the optimal policy and
    value function is often not possible, but generalally we will attempt to
    find a policy that still performs in a way that generates a relatively high
    expected sum of rewards.`),ue.forEach(t),j=r(e),M=m(e,"P",{});var fe=d(M);te=x(fe,`In the cart pole environment below the task is to balance the pole and not
    to move too far away from the center of the screen. At each timestep the
    agent can either move left or right and gets a reward of +1. If the angle of
    the pole is too large or the cart goes offscreen the game ends. The agent
    can observe the car position, the cart velocity, the angle of the pole and
    the angular velocity. Each of the variables is continuous and it is
    therefore becomes necessary to approximate the value function.`),fe.forEach(t),L=r(e),V(T.$$.fragment,e),J=r(e),C=m(e,"P",{});var he=d(C);ae=x(he,`Before we move on to the next section, we should mention the limitation of
    approximative value based methods. Value based approximative methods are
    able to deal with a continuous state space, but not continuous actions. Yet
    as you can imagine many of the tasks a robot for example would need to
    perform require continuous actions. In later chapters we will discuss how we
    can deal with these limitations.`),he.forEach(t),K=r(e),H=m(e,"DIV",{class:!0}),d(H).forEach(t),this.h()},h(){O(H,"class","separator")},m(e,n){o(e,s,n),w(s,_),o(e,p,n),R(v,e,n),o(e,y,n),o(e,u,n),w(u,k),o(e,f,n),o(e,b,n),w(b,a),o(e,i,n),R(h,e,n),o(e,q,n),o(e,E,n),w(E,U),o(e,z,n),o(e,A,n),w(A,X),o(e,Q,n),R(I,e,n),o(e,B,n),o(e,D,n),w(D,Z),o(e,G,n),o(e,P,n),w(P,ee),o(e,j,n),o(e,M,n),w(M,te),o(e,L,n),R(T,e,n),o(e,J,n),o(e,C,n),w(C,ae),o(e,K,n),o(e,H,n),N=!0},p:de,i(e){N||(W(v.$$.fragment,e),W(h.$$.fragment,e),W(I.$$.fragment,e),W(T.$$.fragment,e),N=!0)},o(e){Y(v.$$.fragment,e),Y(h.$$.fragment,e),Y(I.$$.fragment,e),Y(T.$$.fragment,e),N=!1},d(e){e&&t(s),e&&t(p),F(v,e),e&&t(y),e&&t(u),e&&t(f),e&&t(b),e&&t(i),F(h,e),e&&t(q),e&&t(E),e&&t(z),e&&t(A),e&&t(Q),F(I,e),e&&t(B),e&&t(D),e&&t(G),e&&t(P),e&&t(j),e&&t(M),e&&t(L),F(T,e),e&&t(J),e&&t(C),e&&t(K),e&&t(H)}}}function ye(g){let s,_,p,v,y,u,k,f,b;return f=new be({props:{$$slots:{default:[ge]},$$scope:{ctx:g}}}),{c(){s=c("meta"),_=l(),p=c("h1"),v=$("Approximative Value Function"),y=l(),u=c("div"),k=l(),S(f.$$.fragment),this.h()},l(a){const i=ve('[data-svelte="svelte-27lgjb"]',document.head);s=m(i,"META",{name:!0,content:!0}),i.forEach(t),_=r(a),p=m(a,"H1",{});var h=d(p);v=x(h,"Approximative Value Function"),h.forEach(t),y=r(a),u=m(a,"DIV",{class:!0}),d(u).forEach(t),k=r(a),V(f.$$.fragment,a),this.h()},h(){document.title="World4AI | Reinforcement Learning | Value Function Approximation",O(s,"name","description"),O(s,"content","When the state space is too large or continuous, it gets impossible to represent a value function as a table. In that case the agent needs to use an approximate value function."),O(u,"class","separator")},m(a,i){w(document.head,s),o(a,_,i),o(a,p,i),w(p,v),o(a,y,i),o(a,u,i),o(a,k,i),R(f,a,i),b=!0},p(a,[i]){const h={};i&64&&(h.$$scope={dirty:i,ctx:a}),f.$set(h)},i(a){b||(W(f.$$.fragment,a),b=!0)},o(a){Y(f.$$.fragment,a),b=!1},d(a){t(s),a&&t(_),a&&t(p),a&&t(y),a&&t(u),a&&t(k),F(f,a)}}}function $e(g){return[["State","Action 1","Action 2"],[[0,0,1],[1,0,1]],["State","Action 1","Action 2"],[[0,0,1],[1,0,1],[2,0,1],[3,2,2],[4,-1,1],["...","...","..."],[1e9,22,25]],["State Representation","Action 1","Action 2"],[[1.1,1,2],[1.3,1.2,1.8],[1.5,1.5,1.2],[1.7,"?","?"],[2.1,1.8,1.3],[2.2,1.7,1.8],[2.5,1.5,2]]]}class Te extends ce{constructor(s){super(),me(this,s,$e,ye,pe,{})}}export{Te as default};
