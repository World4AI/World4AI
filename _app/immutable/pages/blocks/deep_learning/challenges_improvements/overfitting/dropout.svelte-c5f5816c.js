import{S as ce,i as de,s as he,w as g,a as x,x as w,c as E,y as _,b as s,f as y,t as b,B as k,h as i,l as P,r as $,T as ve,m as z,n as Y,u as c,p as ue,G as v,E as ne}from"../../../../../chunks/index-caa95cd4.js";import{C as ge}from"../../../../../chunks/Container-5c6b7f6d.js";import"../../../../../chunks/SvgContainer-cd77374a.js";import{P as we}from"../../../../../chunks/PlayButton-ea5f48d9.js";import{N as _e}from"../../../../../chunks/NeuralNetwork-14818774.js";import{F as ye,I as $e}from"../../../../../chunks/InternalLink-f9299a76.js";import{L as K}from"../../../../../chunks/Latex-bf74aeea.js";function be(f){let n,a,t,u;return n=new we({props:{f:f[1],delta:800}}),t=new _e({props:{width:ke,maxWidth:xe,height:Ee,rectSize:De,layers:f[0]}}),{c(){g(n.$$.fragment),a=x(),g(t.$$.fragment)},l(l){w(n.$$.fragment,l),a=E(l),w(t.$$.fragment,l)},m(l,p){_(n,l,p),s(l,a,p),_(t,l,p),u=!0},p(l,[p]){const D={};p&1&&(D.layers=l[0]),t.$set(D)},i(l){u||(y(n.$$.fragment,l),y(t.$$.fragment,l),u=!0)},o(l){b(n.$$.fragment,l),b(t.$$.fragment,l),u=!1},d(l){k(n,l),l&&i(a),k(t,l)}}}const ke=500,xe="700px",Ee=250,De=25;let Se=.2;function Te(f,n,a){let t=[{title:"",nodes:[{value:"",fill:"none"},{value:"",fill:"none"},{value:"",fill:"none"},{value:"",fill:"none"},{value:"",fill:"none"}]},{title:"",nodes:[{value:"",fill:"none"},{value:"",fill:"none"},{value:"",fill:"none"},{value:"",fill:"none"},{value:"",fill:"none"},{value:"",fill:"none"}]},{title:"",nodes:[{value:"",fill:"none"},{value:"",fill:"none"},{value:"",fill:"none"},{value:"",fill:"none"},{value:"",fill:"none"}]},{title:"",nodes:[{value:"",fill:"none"}]}];function u(){t.forEach((l,p)=>{p!==t.length-1&&l.nodes.forEach((D,m)=>{Math.random()>=Se?a(0,t[p].nodes[m].fill="none",t):a(0,t[p].nodes[m].fill="var(--main-color-1)",t)})})}return[t,u]}class Pe extends ce{constructor(n){super(),de(this,n,Te,be,he,{})}}function ze(f){let n;return{c(){n=$("p")},l(a){n=c(a,"p")},m(a,t){s(a,n,t)},d(a){a&&i(n)}}}function Be(f){let n;return{c(){n=$("p")},l(a){n=c(a,"p")},m(a,t){s(a,n,t)},d(a){a&&i(n)}}}function Ae(f){let n=String.raw`\dfrac{1}{1-p}`+"",a;return{c(){a=$(n)},l(t){a=c(t,n)},m(t,u){s(t,a,u)},p:ne,d(t){t&&i(a)}}}function Le(f){let n;return{c(){n=$("p")},l(a){n=c(a,"p")},m(a,t){s(a,n,t)},d(a){a&&i(n)}}}function Ne(f){let n=String.raw`
    \begin{bmatrix}
    1 \\
    1 \\
    1 \\
    1 \\
    1 \\
    1 \\
    \end{bmatrix}
    `+"",a;return{c(){a=$(n)},l(t){a=c(t,n)},m(t,u){s(t,a,u)},p:ne,d(t){t&&i(a)}}}function We(f){let n=String.raw`\dfrac{1}{1-0.5} = 2`+"",a;return{c(){a=$(n)},l(t){a=c(t,n)},m(t,u){s(t,a,u)},p:ne,d(t){t&&i(a)}}}function Ie(f){let n=String.raw`
    \begin{bmatrix}
    2 \\
    0 \\
    0 \\
    2 \\
    0 \\
    2 \\
    \end{bmatrix}
    `+"",a;return{c(){a=$(n)},l(t){a=c(t,n)},m(t,u){s(t,a,u)},p:ne,d(t){t&&i(a)}}}function qe(f){let n,a,t,u,l,p,D,m,B,h,A,r,d,S,C,V,L,ae,N,re,Q,W,oe,I,ie,U,q,X,G,le,H,se,Z,R,ee,O,fe,te;return t=new $e({props:{id:1,type:"reference"}}),l=new $e({props:{id:2,type:"reference"}}),h=new K({props:{$$slots:{default:[ze]},$$scope:{ctx:f}}}),r=new K({props:{$$slots:{default:[Be]},$$scope:{ctx:f}}}),C=new Pe({}),N=new K({props:{$$slots:{default:[Ae]},$$scope:{ctx:f}}}),I=new K({props:{$$slots:{default:[Le]},$$scope:{ctx:f}}}),q=new K({props:{$$slots:{default:[Ne]},$$scope:{ctx:f}}}),H=new K({props:{$$slots:{default:[We]},$$scope:{ctx:f}}}),R=new K({props:{$$slots:{default:[Ie]},$$scope:{ctx:f}}}),{c(){n=P("p"),a=$(`Dropout is a regularization technique was developed by Geoffrey Hinton and
    his colleagues at the university of Toronto `),g(t.$$.fragment),u=x(),g(l.$$.fragment),p=$(`. The idea seems so simple, that it
    is almost preposterous that it works at all.`),D=x(),m=P("p"),B=$("At trainig time at each training step with a probability of "),g(h.$$.fragment),A=$(`
    a neuron is deactivated, its value is set to 0. You can start the interactive
    example and observe how it looks like with a `),g(r.$$.fragment),d=$(" value of 0.2."),S=x(),g(C.$$.fragment),V=x(),L=P("p"),ae=$(`When we use our model for inference, we do not remove any of the nodes. If
    we did that, we would get different results each time we run a model. By not
    deactivating any nodes we introduce a problem though. Because more neurons
    are active, each layer has to deal with an input that is on a different
    scale, than the one the neural network has seen during training. Different
    conditions during training and inference will prevent the neural network
    from generating good predictions. To prevent that from happening the active
    nodes are scaled by
    `),g(N.$$.fragment),re=$(` during training. We skip that scaling
    during inference and thus create similar conditions for training and inference.`),Q=x(),W=P("p"),oe=$(`Let us assume for example that the activation layer contains only 1's and
    `),g(I.$$.fragment),ie=$(" is 0.5."),U=x(),g(q.$$.fragment),X=x(),G=P("p"),le=$(`The dropout layer will zero out roughly half of the activations and multiply
    the rest by `),g(H.$$.fragment),se=$("."),Z=x(),g(R.$$.fragment),ee=x(),O=P("p"),fe=$(`But why is the dropout procedure helpful in avoiding overfitting? Each time
    we remove a set of activations from training, we essentially create a
    different model. This simplified model has to learn to deal with the task at
    hand without overrelying on any of the previous activations, because any of
    those might get deactivated at any time. The final model can be seen as an
    ensemble of an immensely huge collection of simplified models. Ensembles
    models tend to produce better results and reduce overfitting. You will
    notice that in practice dropout works extremely well.`)},l(e){n=z(e,"P",{});var o=Y(n);a=c(o,`Dropout is a regularization technique was developed by Geoffrey Hinton and
    his colleagues at the university of Toronto `),w(t.$$.fragment,o),u=E(o),w(l.$$.fragment,o),p=c(o,`. The idea seems so simple, that it
    is almost preposterous that it works at all.`),o.forEach(i),D=E(e),m=z(e,"P",{});var T=Y(m);B=c(T,"At trainig time at each training step with a probability of "),w(h.$$.fragment,T),A=c(T,`
    a neuron is deactivated, its value is set to 0. You can start the interactive
    example and observe how it looks like with a `),w(r.$$.fragment,T),d=c(T," value of 0.2."),T.forEach(i),S=E(e),w(C.$$.fragment,e),V=E(e),L=z(e,"P",{});var M=Y(L);ae=c(M,`When we use our model for inference, we do not remove any of the nodes. If
    we did that, we would get different results each time we run a model. By not
    deactivating any nodes we introduce a problem though. Because more neurons
    are active, each layer has to deal with an input that is on a different
    scale, than the one the neural network has seen during training. Different
    conditions during training and inference will prevent the neural network
    from generating good predictions. To prevent that from happening the active
    nodes are scaled by
    `),w(N.$$.fragment,M),re=c(M,` during training. We skip that scaling
    during inference and thus create similar conditions for training and inference.`),M.forEach(i),Q=E(e),W=z(e,"P",{});var j=Y(W);oe=c(j,`Let us assume for example that the activation layer contains only 1's and
    `),w(I.$$.fragment,j),ie=c(j," is 0.5."),j.forEach(i),U=E(e),w(q.$$.fragment,e),X=E(e),G=z(e,"P",{});var F=Y(G);le=c(F,`The dropout layer will zero out roughly half of the activations and multiply
    the rest by `),w(H.$$.fragment,F),se=c(F,"."),F.forEach(i),Z=E(e),w(R.$$.fragment,e),ee=E(e),O=z(e,"P",{});var J=Y(O);fe=c(J,`But why is the dropout procedure helpful in avoiding overfitting? Each time
    we remove a set of activations from training, we essentially create a
    different model. This simplified model has to learn to deal with the task at
    hand without overrelying on any of the previous activations, because any of
    those might get deactivated at any time. The final model can be seen as an
    ensemble of an immensely huge collection of simplified models. Ensembles
    models tend to produce better results and reduce overfitting. You will
    notice that in practice dropout works extremely well.`),J.forEach(i)},m(e,o){s(e,n,o),v(n,a),_(t,n,null),v(n,u),_(l,n,null),v(n,p),s(e,D,o),s(e,m,o),v(m,B),_(h,m,null),v(m,A),_(r,m,null),v(m,d),s(e,S,o),_(C,e,o),s(e,V,o),s(e,L,o),v(L,ae),_(N,L,null),v(L,re),s(e,Q,o),s(e,W,o),v(W,oe),_(I,W,null),v(W,ie),s(e,U,o),_(q,e,o),s(e,X,o),s(e,G,o),v(G,le),_(H,G,null),v(G,se),s(e,Z,o),_(R,e,o),s(e,ee,o),s(e,O,o),v(O,fe),te=!0},p(e,o){const T={};o&2&&(T.$$scope={dirty:o,ctx:e}),h.$set(T);const M={};o&2&&(M.$$scope={dirty:o,ctx:e}),r.$set(M);const j={};o&2&&(j.$$scope={dirty:o,ctx:e}),N.$set(j);const F={};o&2&&(F.$$scope={dirty:o,ctx:e}),I.$set(F);const J={};o&2&&(J.$$scope={dirty:o,ctx:e}),q.$set(J);const me={};o&2&&(me.$$scope={dirty:o,ctx:e}),H.$set(me);const pe={};o&2&&(pe.$$scope={dirty:o,ctx:e}),R.$set(pe)},i(e){te||(y(t.$$.fragment,e),y(l.$$.fragment,e),y(h.$$.fragment,e),y(r.$$.fragment,e),y(C.$$.fragment,e),y(N.$$.fragment,e),y(I.$$.fragment,e),y(q.$$.fragment,e),y(H.$$.fragment,e),y(R.$$.fragment,e),te=!0)},o(e){b(t.$$.fragment,e),b(l.$$.fragment,e),b(h.$$.fragment,e),b(r.$$.fragment,e),b(C.$$.fragment,e),b(N.$$.fragment,e),b(I.$$.fragment,e),b(q.$$.fragment,e),b(H.$$.fragment,e),b(R.$$.fragment,e),te=!1},d(e){e&&i(n),k(t),k(l),e&&i(D),e&&i(m),k(h),k(r),e&&i(S),k(C,e),e&&i(V),e&&i(L),k(N),e&&i(Q),e&&i(W),k(I),e&&i(U),k(q,e),e&&i(X),e&&i(G),k(H),e&&i(Z),k(R,e),e&&i(ee),e&&i(O)}}}function Ge(f){let n,a,t,u,l,p,D,m,B,h,A;return m=new ge({props:{$$slots:{default:[qe]},$$scope:{ctx:f}}}),h=new ye({props:{references:f[0]}}),{c(){n=P("meta"),a=x(),t=P("h1"),u=$("Dropout"),l=x(),p=P("div"),D=x(),g(m.$$.fragment),B=x(),g(h.$$.fragment),this.h()},l(r){const d=ve('[data-svelte="svelte-16g4sf3"]',document.head);n=z(d,"META",{name:!0,content:!0}),d.forEach(i),a=E(r),t=z(r,"H1",{});var S=Y(t);u=c(S,"Dropout"),S.forEach(i),l=E(r),p=z(r,"DIV",{class:!0}),Y(p).forEach(i),D=E(r),w(m.$$.fragment,r),B=E(r),w(h.$$.fragment,r),this.h()},h(){document.title="World4AI | Deep Learning | Dropout",ue(n,"name","description"),ue(n,"content","Dropout is a regularization technique that works by removing different activation nodes at each training step."),ue(p,"class","separator")},m(r,d){v(document.head,n),s(r,a,d),s(r,t,d),v(t,u),s(r,l,d),s(r,p,d),s(r,D,d),_(m,r,d),s(r,B,d),_(h,r,d),A=!0},p(r,[d]){const S={};d&2&&(S.$$scope={dirty:d,ctx:r}),m.$set(S)},i(r){A||(y(m.$$.fragment,r),y(h.$$.fragment,r),A=!0)},o(r){b(m.$$.fragment,r),b(h.$$.fragment,r),A=!1},d(r){i(n),r&&i(a),r&&i(t),r&&i(l),r&&i(p),r&&i(D),k(m,r),r&&i(B),k(h,r)}}}function He(f){return[[{author:"G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever and R. R. Salakhutdivno",title:"Improving neural networks by preventing co-adaptation of feature detectors",journal:"",year:"2012",pages:"",volume:"",issue:""},{author:"Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov",title:"Dropout: A Simple Way to Prevent Neural Networks from Overfitting",journal:"Journal Of Machine Learning Research",year:"2014",pages:"1929-1958",volume:"15",issue:"1"}]]}class Oe extends ce{constructor(n){super(),de(this,n,He,Ge,he,{})}}export{Oe as default};
