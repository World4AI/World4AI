import{S as me,i as he,s as _e,l as P,a as C,w as g,T as we,m as D,h as o,c as N,x as d,p as X,G as m,b as i,y as v,f as b,t as x,B as y,r,n as B,u as l,E as F}from"../../../../chunks/index-caa95cd4.js";import{C as ge}from"../../../../chunks/Container-5c6b7f6d.js";import{L as z}from"../../../../chunks/Latex-bf74aeea.js";function de(f){let a=String.raw`\mathbf{x}`+"",t;return{c(){t=r(a)},l(n){t=l(n,a)},m(n,u){i(n,t,u)},p:F,d(n){n&&o(t)}}}function ve(f){let a=String.raw`\mathbf{w}`+"",t;return{c(){t=r(a)},l(n){t=l(n,a)},m(n,u){i(n,t,u)},p:F,d(n){n&&o(t)}}}function be(f){let a;return{c(){a=r("b")},l(t){a=l(t,"b")},m(t,n){i(t,a,n)},d(t){t&&o(a)}}}function xe(f){let a=String.raw`z = \mathbf{x} \mathbf{w^T} + b`+"",t;return{c(){t=r(a)},l(n){t=l(n,a)},m(n,u){i(n,t,u)},p:F,d(n){n&&o(t)}}}function ye(f){let a=String.raw`
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
`+"",t;return{c(){t=r(a)},l(n){t=l(n,a)},m(n,u){i(n,t,u)},p:F,d(n){n&&o(t)}}}function ze(f){let a;return{c(){a=r("a(z)")},l(t){a=l(t,"a(z)")},m(t,n){i(t,a,n)},d(t){t&&o(a)}}}function Ee(f){let a;return{c(){a=r("a(z) = z")},l(t){a=l(t,"a(z) = z")},m(t,n){i(t,a,n)},d(t){t&&o(a)}}}function Le(f){let a=String.raw`\hat{y} = a(\mathbf{x} \mathbf{w}^T  + b)`+"",t;return{c(){t=r(a)},l(n){t=l(n,a)},m(n,u){i(n,t,u)},p:F,d(n){n&&o(t)}}}function Te(f){let a;return{c(){a=r("a(z)")},l(t){a=l(t,"a(z)")},m(t,n){i(t,a,n)},d(t){t&&o(a)}}}function ke(f){let a,t,n,u,$,c,G,J,_,Y,E,Z,L,ee,T,te,K,k,ne,S,ae,O,A,Q,p,se,q,oe,I,ie,W,re,j,le,R,H,U;return E=new z({props:{$$slots:{default:[de]},$$scope:{ctx:f}}}),L=new z({props:{$$slots:{default:[ve]},$$scope:{ctx:f}}}),T=new z({props:{$$slots:{default:[be]},$$scope:{ctx:f}}}),S=new z({props:{$$slots:{default:[xe]},$$scope:{ctx:f}}}),A=new z({props:{$$slots:{default:[ye]},$$scope:{ctx:f}}}),q=new z({props:{$$slots:{default:[ze]},$$scope:{ctx:f}}}),I=new z({props:{$$slots:{default:[Ee]},$$scope:{ctx:f}}}),W=new z({props:{$$slots:{default:[Le]},$$scope:{ctx:f}}}),j=new z({props:{$$slots:{default:[Te]},$$scope:{ctx:f}}}),{c(){a=P("h1"),t=r("Linear Neuron"),n=C(),u=P("div"),$=C(),c=P("p"),G=r(`If we look at linear regression from the perspective of a neural network, we
    will observe that linear regression is just a neuron with a linear
    activation function.`),J=C(),_=P("p"),Y=r(`Let us remind ourselves, that a neuron is a computational unit. The output
    of the neuron is based on three distinct calculations: scaling of inputs `),g(E.$$.fragment),Z=r(" with weights "),g(L.$$.fragment),ee=r(`, summation of the
    scaled inputs (plus bias `),g(T.$$.fragment),te=r(`) and the application of an
    activation function.`),K=C(),k=P("p"),ne=r(`The calculation from linear regression
    `),g(S.$$.fragment),ae=r(`
    essentially performs all three parts in a single step.`),O=C(),g(A.$$.fragment),Q=C(),p=P("p"),se=r(`At this point you might interject, that we do not have an activation
    function `),g(q.$$.fragment),oe=r(`, so let us introduce one that does not change
    the nature of linear regression. We are going to use the so called identity
    function, where the input equals the output `),g(I.$$.fragment),ie=r(`. When we
    apply the identity function as an activation, we end up with a linear
    neuron, where
    `),g(W.$$.fragment),re=r(`. This
    might seem like an unnecessary step, but by enforcing the usage of an
    identity function, we put ourselves into a position where we can start to
    understand different types of neurons. All we have to do is to replace the
    identity function by any other activation function `),g(j.$$.fragment),le=r(` to describe
    any other type of neuron.`),R=C(),H=P("div"),this.h()},l(e){a=D(e,"H1",{});var s=B(a);t=l(s,"Linear Neuron"),s.forEach(o),n=N(e),u=D(e,"DIV",{class:!0}),B(u).forEach(o),$=N(e),c=D(e,"P",{});var M=B(c);G=l(M,`If we look at linear regression from the perspective of a neural network, we
    will observe that linear regression is just a neuron with a linear
    activation function.`),M.forEach(o),J=N(e),_=D(e,"P",{});var w=B(_);Y=l(w,`Let us remind ourselves, that a neuron is a computational unit. The output
    of the neuron is based on three distinct calculations: scaling of inputs `),d(E.$$.fragment,w),Z=l(w," with weights "),d(L.$$.fragment,w),ee=l(w,`, summation of the
    scaled inputs (plus bias `),d(T.$$.fragment,w),te=l(w,`) and the application of an
    activation function.`),w.forEach(o),K=N(e),k=D(e,"P",{});var V=B(k);ne=l(V,`The calculation from linear regression
    `),d(S.$$.fragment,V),ae=l(V,`
    essentially performs all three parts in a single step.`),V.forEach(o),O=N(e),d(A.$$.fragment,e),Q=N(e),p=D(e,"P",{});var h=B(p);se=l(h,`At this point you might interject, that we do not have an activation
    function `),d(q.$$.fragment,h),oe=l(h,`, so let us introduce one that does not change
    the nature of linear regression. We are going to use the so called identity
    function, where the input equals the output `),d(I.$$.fragment,h),ie=l(h,`. When we
    apply the identity function as an activation, we end up with a linear
    neuron, where
    `),d(W.$$.fragment,h),re=l(h,`. This
    might seem like an unnecessary step, but by enforcing the usage of an
    identity function, we put ourselves into a position where we can start to
    understand different types of neurons. All we have to do is to replace the
    identity function by any other activation function `),d(j.$$.fragment,h),le=l(h,` to describe
    any other type of neuron.`),h.forEach(o),R=N(e),H=D(e,"DIV",{class:!0}),B(H).forEach(o),this.h()},h(){X(u,"class","separator"),X(H,"class","separator")},m(e,s){i(e,a,s),m(a,t),i(e,n,s),i(e,u,s),i(e,$,s),i(e,c,s),m(c,G),i(e,J,s),i(e,_,s),m(_,Y),v(E,_,null),m(_,Z),v(L,_,null),m(_,ee),v(T,_,null),m(_,te),i(e,K,s),i(e,k,s),m(k,ne),v(S,k,null),m(k,ae),i(e,O,s),v(A,e,s),i(e,Q,s),i(e,p,s),m(p,se),v(q,p,null),m(p,oe),v(I,p,null),m(p,ie),v(W,p,null),m(p,re),v(j,p,null),m(p,le),i(e,R,s),i(e,H,s),U=!0},p(e,s){const M={};s&1&&(M.$$scope={dirty:s,ctx:e}),E.$set(M);const w={};s&1&&(w.$$scope={dirty:s,ctx:e}),L.$set(w);const V={};s&1&&(V.$$scope={dirty:s,ctx:e}),T.$set(V);const h={};s&1&&(h.$$scope={dirty:s,ctx:e}),S.$set(h);const fe={};s&1&&(fe.$$scope={dirty:s,ctx:e}),A.$set(fe);const ue={};s&1&&(ue.$$scope={dirty:s,ctx:e}),q.$set(ue);const $e={};s&1&&($e.$$scope={dirty:s,ctx:e}),I.$set($e);const ce={};s&1&&(ce.$$scope={dirty:s,ctx:e}),W.$set(ce);const pe={};s&1&&(pe.$$scope={dirty:s,ctx:e}),j.$set(pe)},i(e){U||(b(E.$$.fragment,e),b(L.$$.fragment,e),b(T.$$.fragment,e),b(S.$$.fragment,e),b(A.$$.fragment,e),b(q.$$.fragment,e),b(I.$$.fragment,e),b(W.$$.fragment,e),b(j.$$.fragment,e),U=!0)},o(e){x(E.$$.fragment,e),x(L.$$.fragment,e),x(T.$$.fragment,e),x(S.$$.fragment,e),x(A.$$.fragment,e),x(q.$$.fragment,e),x(I.$$.fragment,e),x(W.$$.fragment,e),x(j.$$.fragment,e),U=!1},d(e){e&&o(a),e&&o(n),e&&o(u),e&&o($),e&&o(c),e&&o(J),e&&o(_),y(E),y(L),y(T),e&&o(K),e&&o(k),y(S),e&&o(O),y(A,e),e&&o(Q),e&&o(p),y(q),y(I),y(W),y(j),e&&o(R),e&&o(H)}}}function Se(f){let a,t,n,u;return n=new ge({props:{$$slots:{default:[ke]},$$scope:{ctx:f}}}),{c(){a=P("meta"),t=C(),g(n.$$.fragment),this.h()},l($){const c=we('[data-svelte="svelte-veg9kq"]',document.head);a=D(c,"META",{name:!0,content:!0}),c.forEach(o),t=N($),d(n.$$.fragment,$),this.h()},h(){document.title="World4AI | Deep Learning | Linear Neuron",X(a,"name","description"),X(a,"content","Linear regression constitutes the simplest neuron imaginable: a neuron with a linear activation function.")},m($,c){m(document.head,a),i($,t,c),v(n,$,c),u=!0},p($,[c]){const G={};c&1&&(G.$$scope={dirty:c,ctx:$}),n.$set(G)},i($){u||(b(n.$$.fragment,$),u=!0)},o($){x(n.$$.fragment,$),u=!1},d($){o(a),$&&o(t),y(n,$)}}}class We extends me{constructor(a){super(),he(this,a,null,Se,_e,{})}}export{We as default};
