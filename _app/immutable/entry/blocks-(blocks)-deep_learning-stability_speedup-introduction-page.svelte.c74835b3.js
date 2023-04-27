import{S,i as q,s as P,k as d,a as w,q as b,y as x,W as B,l as m,h as a,c as v,m as y,r as k,z as C,n as $,N as _,b as l,A as T,g as z,d as A,B as M,C as N}from"../chunks/index.4d92b023.js";import{C as D}from"../chunks/Container.b0705c7b.js";function G(I){let o,h,s,u,f,r,c,i,p,e;return{c(){o=d("p"),h=b(`It takes a lot of time and patience to train a neural network. In practice
    you usually want to iterate often to find a good approach to a particular
    problem. But if a single iteration takes hours (or even days) you might not
    be able to solve your task. Especially when you need to train large models
    on a cluster of GPUs, each saved hour of training will additionally
    correspond to monetary savings.`),s=w(),u=d("p"),f=b(`In this chapter we will look at different techniques that will either
    improve your training time significantly or allow you to build models that
    are less prone to instability during training.`),r=w(),c=d("p"),i=b(`Be aware, that many of the techniques that we will introduce in this chapter
    are not strictly necessary to sovle MNIST, yet they will be essential down
    the line once we start dealing with convolutional neural networks or
    transformer models.`),p=w(),e=d("div"),this.h()},l(t){o=m(t,"P",{});var n=y(o);h=k(n,`It takes a lot of time and patience to train a neural network. In practice
    you usually want to iterate often to find a good approach to a particular
    problem. But if a single iteration takes hours (or even days) you might not
    be able to solve your task. Especially when you need to train large models
    on a cluster of GPUs, each saved hour of training will additionally
    correspond to monetary savings.`),n.forEach(a),s=v(t),u=m(t,"P",{});var g=y(u);f=k(g,`In this chapter we will look at different techniques that will either
    improve your training time significantly or allow you to build models that
    are less prone to instability during training.`),g.forEach(a),r=v(t),c=m(t,"P",{});var E=y(c);i=k(E,`Be aware, that many of the techniques that we will introduce in this chapter
    are not strictly necessary to sovle MNIST, yet they will be essential down
    the line once we start dealing with convolutional neural networks or
    transformer models.`),E.forEach(a),p=v(t),e=m(t,"DIV",{class:!0}),y(e).forEach(a),this.h()},h(){$(e,"class","separator")},m(t,n){l(t,o,n),_(o,h),l(t,s,n),l(t,u,n),_(u,f),l(t,r,n),l(t,c,n),_(c,i),l(t,p,n),l(t,e,n)},p:N,d(t){t&&a(o),t&&a(s),t&&a(u),t&&a(r),t&&a(c),t&&a(p),t&&a(e)}}}function U(I){let o,h,s,u,f,r,c,i,p;return i=new D({props:{$$slots:{default:[G]},$$scope:{ctx:I}}}),{c(){o=d("meta"),h=w(),s=d("h1"),u=b("Stability and Speedup"),f=w(),r=d("div"),c=w(),x(i.$$.fragment),this.h()},l(e){const t=B("svelte-12nv6a",document.head);o=m(t,"META",{name:!0,content:!0}),t.forEach(a),h=v(e),s=m(e,"H1",{});var n=y(s);u=k(n,"Stability and Speedup"),n.forEach(a),f=v(e),r=m(e,"DIV",{class:!0}),y(r).forEach(a),c=v(e),C(i.$$.fragment,e),this.h()},h(){document.title="Stability and Speedup - World4AI",$(o,"name","description"),$(o,"content","Over the years several techniques, like new optimizers, skip connections and batch normalization have been developed to combat long or unstable training. This chapter is dedicated to those techniques."),$(r,"class","separator")},m(e,t){_(document.head,o),l(e,h,t),l(e,s,t),_(s,u),l(e,f,t),l(e,r,t),l(e,c,t),T(i,e,t),p=!0},p(e,[t]){const n={};t&1&&(n.$$scope={dirty:t,ctx:e}),i.$set(n)},i(e){p||(z(i.$$.fragment,e),p=!0)},o(e){A(i.$$.fragment,e),p=!1},d(e){a(o),e&&a(h),e&&a(s),e&&a(f),e&&a(r),e&&a(c),M(i,e)}}}class H extends S{constructor(o){super(),q(this,o,null,U,P,{})}}export{H as default};
