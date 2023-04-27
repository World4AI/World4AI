import{S as ce,i as me,s as pe,k as D,a as V,q as l,y as v,W as he,l as F,h as o,c as B,m as O,r as f,z as x,n as U,N as g,b as $,A as b,g as k,d as S,B as E,C as G}from"../chunks/index.4d92b023.js";import{C as ge}from"../chunks/Container.b0705c7b.js";import{L as z}from"../chunks/Latex.e0b308c0.js";function _e(u){let n=String.raw`\mathbf{x}`+"",e;return{c(){e=l(n)},l(a){e=f(a,n)},m(a,r){$(a,e,r)},p:G,d(a){a&&o(e)}}}function de(u){let n=String.raw`\mathbf{w}`+"",e;return{c(){e=l(n)},l(a){e=f(a,n)},m(a,r){$(a,e,r)},p:G,d(a){a&&o(e)}}}function we(u){let n;return{c(){n=l("b")},l(e){n=f(e,"b")},m(e,a){$(e,n,a)},d(e){e&&o(n)}}}function ve(u){let n;return{c(){n=l("f")},l(e){n=f(e,"f")},m(e,a){$(e,n,a)},d(e){e&&o(n)}}}function xe(u){let n=String.raw`f(z)`+"",e;return{c(){e=l(n)},l(a){e=f(a,n)},m(a,r){$(a,e,r)},p:G,d(a){a&&o(e)}}}function be(u){let n;return{c(){n=l("f")},l(e){n=f(e,"f")},m(e,a){$(e,n,a)},d(e){e&&o(n)}}}function ke(u){let n;return{c(){n=l("\\sigma")},l(e){n=f(e,"\\sigma")},m(e,a){$(e,n,a)},d(e){e&&o(n)}}}function Se(u){let n;return{c(){n=l("z")},l(e){n=f(e,"z")},m(e,a){$(e,n,a)},d(e){e&&o(n)}}}function Ee(u){let n=String.raw`\mathbf{xw}^T + b`+"",e;return{c(){e=l(n)},l(a){e=f(a,n)},m(a,r){$(a,e,r)},p:G,d(a){a&&o(e)}}}function ze(u){let n=String.raw`\dfrac{1}{1+e^{-(w_1x_1 + \cdots + w_nx_n + b)}}`+"",e;return{c(){e=l(n)},l(a){e=f(a,n)},m(a,r){$(a,e,r)},p:G,d(a){a&&o(e)}}}function We(u){let n,e,a,r,W,_,I,m,T,s,h,d,X,J,c,Y,A,Z,L,y,P,ee,C,te,N,ne,q,ae,K,H,se,Q,M,R;return _=new z({props:{$$slots:{default:[_e]},$$scope:{ctx:u}}}),m=new z({props:{$$slots:{default:[de]},$$scope:{ctx:u}}}),s=new z({props:{$$slots:{default:[we]},$$scope:{ctx:u}}}),d=new z({props:{$$slots:{default:[ve]},$$scope:{ctx:u}}}),A=new z({props:{$$slots:{default:[xe]},$$scope:{ctx:u}}}),L=new z({props:{$$slots:{default:[be]},$$scope:{ctx:u}}}),P=new z({props:{$$slots:{default:[ke]},$$scope:{ctx:u}}}),C=new z({props:{$$slots:{default:[Se]},$$scope:{ctx:u}}}),N=new z({props:{$$slots:{default:[Ee]},$$scope:{ctx:u}}}),q=new z({props:{$$slots:{default:[ze]},$$scope:{ctx:u}}}),{c(){n=D("p"),e=l(`Let us make the same exercise we did with linear regression. We can look at
    logistic regression from the perspective of a neural network. If we do, we
    will realize that logistic regression is a neuron with a sigmoid activation
    function.`),a=V(),r=D("p"),W=l(`Once again let us remind ourselves, that a neuron is a computational unit
    that is based on three distinct steps. First: the inputs `),v(_.$$.fragment),I=l(" are scaled by weights "),v(m.$$.fragment),T=l(`. Second: the
    scaled inputs (plus bias `),v(s.$$.fragment),h=l(`) are aggregated via a sum. Third:
    an activation function `),v(d.$$.fragment),X=l(" is applied to the sum."),J=V(),c=D("p"),Y=l(`In logistic regression all three steps can be described by
    `),v(A.$$.fragment),Z=l(", where "),v(L.$$.fragment),y=l(` is the sigmoid activation
    function `),v(P.$$.fragment),ee=l(" and "),v(C.$$.fragment),te=l(" is the net input "),v(N.$$.fragment),ne=l(" . Written in a more familiar manner the output of the neuron amounts to: "),v(q.$$.fragment),ae=l("."),K=V(),H=D("p"),se=l(`This type of a neuron is extremely powerful. When we combine different
    sigmoid neurons, such that the output of a neuron is used as an input to the
    neurons in the next layer, we essentially create a neural network.
    Activation functions like the sigmoid are often called nonlinear
    activations, because they can be utilized in a neural network to solve
    nonlinear problems (more on that in the next chapter).`),Q=V(),M=D("div"),this.h()},l(t){n=F(t,"P",{});var i=O(n);e=f(i,`Let us make the same exercise we did with linear regression. We can look at
    logistic regression from the perspective of a neural network. If we do, we
    will realize that logistic regression is a neuron with a sigmoid activation
    function.`),i.forEach(o),a=B(t),r=F(t,"P",{});var w=O(r);W=f(w,`Once again let us remind ourselves, that a neuron is a computational unit
    that is based on three distinct steps. First: the inputs `),x(_.$$.fragment,w),I=f(w," are scaled by weights "),x(m.$$.fragment,w),T=f(w,`. Second: the
    scaled inputs (plus bias `),x(s.$$.fragment,w),h=f(w,`) are aggregated via a sum. Third:
    an activation function `),x(d.$$.fragment,w),X=f(w," is applied to the sum."),w.forEach(o),J=B(t),c=F(t,"P",{});var p=O(c);Y=f(p,`In logistic regression all three steps can be described by
    `),x(A.$$.fragment,p),Z=f(p,", where "),x(L.$$.fragment,p),y=f(p,` is the sigmoid activation
    function `),x(P.$$.fragment,p),ee=f(p," and "),x(C.$$.fragment,p),te=f(p," is the net input "),x(N.$$.fragment,p),ne=f(p," . Written in a more familiar manner the output of the neuron amounts to: "),x(q.$$.fragment,p),ae=f(p,"."),p.forEach(o),K=B(t),H=F(t,"P",{});var j=O(H);se=f(j,`This type of a neuron is extremely powerful. When we combine different
    sigmoid neurons, such that the output of a neuron is used as an input to the
    neurons in the next layer, we essentially create a neural network.
    Activation functions like the sigmoid are often called nonlinear
    activations, because they can be utilized in a neural network to solve
    nonlinear problems (more on that in the next chapter).`),j.forEach(o),Q=B(t),M=F(t,"DIV",{class:!0}),O(M).forEach(o),this.h()},h(){U(M,"class","separator")},m(t,i){$(t,n,i),g(n,e),$(t,a,i),$(t,r,i),g(r,W),b(_,r,null),g(r,I),b(m,r,null),g(r,T),b(s,r,null),g(r,h),b(d,r,null),g(r,X),$(t,J,i),$(t,c,i),g(c,Y),b(A,c,null),g(c,Z),b(L,c,null),g(c,y),b(P,c,null),g(c,ee),b(C,c,null),g(c,te),b(N,c,null),g(c,ne),b(q,c,null),g(c,ae),$(t,K,i),$(t,H,i),g(H,se),$(t,Q,i),$(t,M,i),R=!0},p(t,i){const w={};i&1&&(w.$$scope={dirty:i,ctx:t}),_.$set(w);const p={};i&1&&(p.$$scope={dirty:i,ctx:t}),m.$set(p);const j={};i&1&&(j.$$scope={dirty:i,ctx:t}),s.$set(j);const ie={};i&1&&(ie.$$scope={dirty:i,ctx:t}),d.$set(ie);const oe={};i&1&&(oe.$$scope={dirty:i,ctx:t}),A.$set(oe);const re={};i&1&&(re.$$scope={dirty:i,ctx:t}),L.$set(re);const le={};i&1&&(le.$$scope={dirty:i,ctx:t}),P.$set(le);const fe={};i&1&&(fe.$$scope={dirty:i,ctx:t}),C.$set(fe);const $e={};i&1&&($e.$$scope={dirty:i,ctx:t}),N.$set($e);const ue={};i&1&&(ue.$$scope={dirty:i,ctx:t}),q.$set(ue)},i(t){R||(k(_.$$.fragment,t),k(m.$$.fragment,t),k(s.$$.fragment,t),k(d.$$.fragment,t),k(A.$$.fragment,t),k(L.$$.fragment,t),k(P.$$.fragment,t),k(C.$$.fragment,t),k(N.$$.fragment,t),k(q.$$.fragment,t),R=!0)},o(t){S(_.$$.fragment,t),S(m.$$.fragment,t),S(s.$$.fragment,t),S(d.$$.fragment,t),S(A.$$.fragment,t),S(L.$$.fragment,t),S(P.$$.fragment,t),S(C.$$.fragment,t),S(N.$$.fragment,t),S(q.$$.fragment,t),R=!1},d(t){t&&o(n),t&&o(a),t&&o(r),E(_),E(m),E(s),E(d),t&&o(J),t&&o(c),E(A),E(L),E(P),E(C),E(N),E(q),t&&o(K),t&&o(H),t&&o(Q),t&&o(M)}}}function Ie(u){let n,e,a,r,W,_,I,m,T;return m=new ge({props:{$$slots:{default:[We]},$$scope:{ctx:u}}}),{c(){n=D("meta"),e=V(),a=D("h1"),r=l("Sigmoid Neuron"),W=V(),_=D("div"),I=V(),v(m.$$.fragment),this.h()},l(s){const h=he("svelte-1yvskow",document.head);n=F(h,"META",{name:!0,content:!0}),h.forEach(o),e=B(s),a=F(s,"H1",{});var d=O(a);r=f(d,"Sigmoid Neuron"),d.forEach(o),W=B(s),_=F(s,"DIV",{class:!0}),O(_).forEach(o),I=B(s),x(m.$$.fragment,s),this.h()},h(){document.title="Sigmoid Neuron - World4AI",U(n,"name","description"),U(n,"content","Logistic regression constitutes the simplest non-linear neuron: a neuron with the sigmoid activation function."),U(_,"class","separator")},m(s,h){g(document.head,n),$(s,e,h),$(s,a,h),g(a,r),$(s,W,h),$(s,_,h),$(s,I,h),b(m,s,h),T=!0},p(s,[h]){const d={};h&1&&(d.$$scope={dirty:h,ctx:s}),m.$set(d)},i(s){T||(k(m.$$.fragment,s),T=!0)},o(s){S(m.$$.fragment,s),T=!1},d(s){o(n),s&&o(e),s&&o(a),s&&o(W),s&&o(_),s&&o(I),E(m,s)}}}class Pe extends ce{constructor(n){super(),me(this,n,null,Ie,pe,{})}}export{Pe as default};
