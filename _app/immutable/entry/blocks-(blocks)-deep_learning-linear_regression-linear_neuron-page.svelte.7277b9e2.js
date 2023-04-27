import{S as ve,i as be,s as xe,k as N,a as H,y as g,W as ze,l as D,h as i,c as V,z as w,n as Z,N as m,b as l,A as d,g as v,d as b,B as x,q as o,m as M,r,C as K}from"../chunks/index.4d92b023.js";import{C as ye}from"../chunks/Container.b0705c7b.js";import{L as y}from"../chunks/Latex.e0b308c0.js";import{H as Le}from"../chunks/Highlight.b7c1de53.js";function Ee(f){let n=String.raw`\mathbf{x}`+"",t;return{c(){t=o(n)},l(a){t=r(a,n)},m(a,u){l(a,t,u)},p:K,d(a){a&&i(t)}}}function Se(f){let n=String.raw`\mathbf{w}`+"",t;return{c(){t=o(n)},l(a){t=r(a,n)},m(a,u){l(a,t,u)},p:K,d(a){a&&i(t)}}}function Ae(f){let n;return{c(){n=o("b")},l(t){n=r(t,"b")},m(t,a){l(t,n,a)},d(t){t&&i(n)}}}function Ie(f){let n=String.raw`z = \mathbf{x} \mathbf{w^T} + b`+"",t;return{c(){t=o(n)},l(a){t=r(a,n)},m(a,u){l(a,t,u)},p:K,d(a){a&&i(t)}}}function Te(f){let n=String.raw`
      z = 
      \begin{bmatrix}
      x_1 & x_2 & x_3 & \cdots & x_n 
      \end{bmatrix}
      \begin{bmatrix}
      w_1 \\ 
      w_2 \\  
      w_3 \\ 
      \vdots \\ 
      w_n
      \end{bmatrix}
      + b
      =
      x_1w_1 + x_2w_2+ x_3w_3+ \cdots + x_nw_n + b
`+"",t;return{c(){t=o(n)},l(a){t=r(a,n)},m(a,u){l(a,t,u)},p:K,d(a){a&&i(t)}}}function We(f){let n;return{c(){n=o("f(z)")},l(t){n=r(t,"f(z)")},m(t,a){l(t,n,a)},d(t){t&&i(n)}}}function ke(f){let n;return{c(){n=o("z")},l(t){n=r(t,"z")},m(t,a){l(t,n,a)},d(t){t&&i(n)}}}function Pe(f){let n;return{c(){n=o("net input")},l(t){n=r(t,"net input")},m(t,a){l(t,n,a)},d(t){t&&i(n)}}}function je(f){let n;return{c(){n=o("f(z) = z")},l(t){n=r(t,"f(z) = z")},m(t,a){l(t,n,a)},d(t){t&&i(n)}}}function qe(f){let n=String.raw`a = f(\mathbf{x} \mathbf{w}^T  + b)`+"",t;return{c(){t=o(n)},l(a){t=r(a,n)},m(a,u){l(a,t,u)},p:K,d(a){a&&i(t)}}}function Ce(f){let n;return{c(){n=o("f(z)")},l(t){n=r(t,"f(z)")},m(t,a){l(t,n,a)},d(t){t&&i(n)}}}function Ne(f){let n,t,a,u,$,h,F,O,_,ee,L,te,E,ne,S,ae,Q,A,se,I,ie,R,T,U,p,oe,W,re,k,le,P,fe,j,ue,q,$e,C,pe,X,G,Y;return L=new y({props:{$$slots:{default:[Ee]},$$scope:{ctx:f}}}),E=new y({props:{$$slots:{default:[Se]},$$scope:{ctx:f}}}),S=new y({props:{$$slots:{default:[Ae]},$$scope:{ctx:f}}}),I=new y({props:{$$slots:{default:[Ie]},$$scope:{ctx:f}}}),T=new y({props:{$$slots:{default:[Te]},$$scope:{ctx:f}}}),W=new y({props:{$$slots:{default:[We]},$$scope:{ctx:f}}}),k=new y({props:{$$slots:{default:[ke]},$$scope:{ctx:f}}}),P=new Le({props:{$$slots:{default:[Pe]},$$scope:{ctx:f}}}),j=new y({props:{$$slots:{default:[je]},$$scope:{ctx:f}}}),q=new y({props:{$$slots:{default:[qe]},$$scope:{ctx:f}}}),C=new y({props:{$$slots:{default:[Ce]},$$scope:{ctx:f}}}),{c(){n=N("h1"),t=o("Linear Neuron"),a=H(),u=N("div"),$=H(),h=N("p"),F=o(`If we look at linear regression from a different perspective, we will
    realize that linear regression is just a neuron with a linear activation
    function.`),O=H(),_=N("p"),ee=o(`Let us remind ourselves, that a neuron is a computational unit. The output
    of the neuron is based on three distinct calculation steps: scaling of
    inputs `),g(L.$$.fragment),te=o(" with weights "),g(E.$$.fragment),ne=o(", summation of the scaled inputs (plus bias "),g(S.$$.fragment),ae=o(`) and the
    application of an activation function.`),Q=H(),A=N("p"),se=o(`Linear regression
    `),g(I.$$.fragment),ie=o(`
    essentially performs all three parts in a single step.`),R=H(),g(T.$$.fragment),U=H(),p=N("p"),oe=o(`At this point you might interject, that we do not have an activation
    function `),g(W.$$.fragment),re=o(". Instead we end up with "),g(k.$$.fragment),le=o(`, the
    so called `),g(P.$$.fragment),fe=o(`. So let us introduce one that
    does not change the nature of linear regression. We are going to use the so
    called identity function, where the input equals the output `),g(j.$$.fragment),ue=o(`. When we apply the identity function as an activation, we end up with a
    linear neuron, where
    `),g(q.$$.fragment),$e=o(`. This might
    seem like an unnecessary step, but by enforcing the usage of an identity
    function, we put ourselves into a position where we can start to understand
    different types of neurons. All we have to do is to replace the identity
    function by any other activation function `),g(C.$$.fragment),pe=o(` to describe any
    other type of neuron.`),X=H(),G=N("div"),this.h()},l(e){n=D(e,"H1",{});var s=M(n);t=r(s,"Linear Neuron"),s.forEach(i),a=V(e),u=D(e,"DIV",{class:!0}),M(u).forEach(i),$=V(e),h=D(e,"P",{});var J=M(h);F=r(J,`If we look at linear regression from a different perspective, we will
    realize that linear regression is just a neuron with a linear activation
    function.`),J.forEach(i),O=V(e),_=D(e,"P",{});var z=M(_);ee=r(z,`Let us remind ourselves, that a neuron is a computational unit. The output
    of the neuron is based on three distinct calculation steps: scaling of
    inputs `),w(L.$$.fragment,z),te=r(z," with weights "),w(E.$$.fragment,z),ne=r(z,", summation of the scaled inputs (plus bias "),w(S.$$.fragment,z),ae=r(z,`) and the
    application of an activation function.`),z.forEach(i),Q=V(e),A=D(e,"P",{});var B=M(A);se=r(B,`Linear regression
    `),w(I.$$.fragment,B),ie=r(B,`
    essentially performs all three parts in a single step.`),B.forEach(i),R=V(e),w(T.$$.fragment,e),U=V(e),p=D(e,"P",{});var c=M(p);oe=r(c,`At this point you might interject, that we do not have an activation
    function `),w(W.$$.fragment,c),re=r(c,". Instead we end up with "),w(k.$$.fragment,c),le=r(c,`, the
    so called `),w(P.$$.fragment,c),fe=r(c,`. So let us introduce one that
    does not change the nature of linear regression. We are going to use the so
    called identity function, where the input equals the output `),w(j.$$.fragment,c),ue=r(c,`. When we apply the identity function as an activation, we end up with a
    linear neuron, where
    `),w(q.$$.fragment,c),$e=r(c,`. This might
    seem like an unnecessary step, but by enforcing the usage of an identity
    function, we put ourselves into a position where we can start to understand
    different types of neurons. All we have to do is to replace the identity
    function by any other activation function `),w(C.$$.fragment,c),pe=r(c,` to describe any
    other type of neuron.`),c.forEach(i),X=V(e),G=D(e,"DIV",{class:!0}),M(G).forEach(i),this.h()},h(){Z(u,"class","separator"),Z(G,"class","separator")},m(e,s){l(e,n,s),m(n,t),l(e,a,s),l(e,u,s),l(e,$,s),l(e,h,s),m(h,F),l(e,O,s),l(e,_,s),m(_,ee),d(L,_,null),m(_,te),d(E,_,null),m(_,ne),d(S,_,null),m(_,ae),l(e,Q,s),l(e,A,s),m(A,se),d(I,A,null),m(A,ie),l(e,R,s),d(T,e,s),l(e,U,s),l(e,p,s),m(p,oe),d(W,p,null),m(p,re),d(k,p,null),m(p,le),d(P,p,null),m(p,fe),d(j,p,null),m(p,ue),d(q,p,null),m(p,$e),d(C,p,null),m(p,pe),l(e,X,s),l(e,G,s),Y=!0},p(e,s){const J={};s&1&&(J.$$scope={dirty:s,ctx:e}),L.$set(J);const z={};s&1&&(z.$$scope={dirty:s,ctx:e}),E.$set(z);const B={};s&1&&(B.$$scope={dirty:s,ctx:e}),S.$set(B);const c={};s&1&&(c.$$scope={dirty:s,ctx:e}),I.$set(c);const ce={};s&1&&(ce.$$scope={dirty:s,ctx:e}),T.$set(ce);const me={};s&1&&(me.$$scope={dirty:s,ctx:e}),W.$set(me);const he={};s&1&&(he.$$scope={dirty:s,ctx:e}),k.$set(he);const _e={};s&1&&(_e.$$scope={dirty:s,ctx:e}),P.$set(_e);const ge={};s&1&&(ge.$$scope={dirty:s,ctx:e}),j.$set(ge);const we={};s&1&&(we.$$scope={dirty:s,ctx:e}),q.$set(we);const de={};s&1&&(de.$$scope={dirty:s,ctx:e}),C.$set(de)},i(e){Y||(v(L.$$.fragment,e),v(E.$$.fragment,e),v(S.$$.fragment,e),v(I.$$.fragment,e),v(T.$$.fragment,e),v(W.$$.fragment,e),v(k.$$.fragment,e),v(P.$$.fragment,e),v(j.$$.fragment,e),v(q.$$.fragment,e),v(C.$$.fragment,e),Y=!0)},o(e){b(L.$$.fragment,e),b(E.$$.fragment,e),b(S.$$.fragment,e),b(I.$$.fragment,e),b(T.$$.fragment,e),b(W.$$.fragment,e),b(k.$$.fragment,e),b(P.$$.fragment,e),b(j.$$.fragment,e),b(q.$$.fragment,e),b(C.$$.fragment,e),Y=!1},d(e){e&&i(n),e&&i(a),e&&i(u),e&&i($),e&&i(h),e&&i(O),e&&i(_),x(L),x(E),x(S),e&&i(Q),e&&i(A),x(I),e&&i(R),x(T,e),e&&i(U),e&&i(p),x(W),x(k),x(P),x(j),x(q),x(C),e&&i(X),e&&i(G)}}}function He(f){let n,t,a,u;return a=new ye({props:{$$slots:{default:[Ne]},$$scope:{ctx:f}}}),{c(){n=N("meta"),t=H(),g(a.$$.fragment),this.h()},l($){const h=ze("svelte-ulf785",document.head);n=D(h,"META",{name:!0,content:!0}),h.forEach(i),t=V($),w(a.$$.fragment,$),this.h()},h(){document.title="Linear Neuron - World4AI",Z(n,"name","description"),Z(n,"content","Linear regression constitutes the simplest neuron imaginable: a neuron with a linear activation function.")},m($,h){m(document.head,n),l($,t,h),d(a,$,h),u=!0},p($,[h]){const F={};h&1&&(F.$$scope={dirty:h,ctx:$}),a.$set(F)},i($){u||(v(a.$$.fragment,$),u=!0)},o($){b(a.$$.fragment,$),u=!1},d($){i(n),$&&i(t),x(a,$)}}}class Fe extends ve{constructor(n){super(),be(this,n,null,He,xe,{})}}export{Fe as default};
