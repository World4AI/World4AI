import{S as De,i as He,s as Ke,l as T,a as b,r as l,w as g,T as Re,m as P,h as i,c as z,n as S,u as f,x as c,p as fe,G as $,b as s,y as w,f as _,t as v,B as y,E as N}from"../../../../../chunks/index-caa95cd4.js";import{C as Xe}from"../../../../../chunks/Container-5c6b7f6d.js";/* empty css                                                                    */import{N as Ue}from"../../../../../chunks/NeuralNetwork-14818774.js";import{L as I}from"../../../../../chunks/Latex-bf74aeea.js";import{F as Ve,I as Be}from"../../../../../chunks/InternalLink-f9299a76.js";import"../../../../../chunks/SvgContainer-cd77374a.js";function Ye(u){let n;return{c(){n=l("w")},l(t){n=f(t,"w")},m(t,a){s(t,n,a)},d(t){t&&i(n)}}}function Je(u){let n=String.raw`
  \begin{aligned}
    a_1 &= x_1 * w + x_2 * w \\
    a_2 &= x_1 * w + x_2 * w
  \end{aligned}
    `+"",t;return{c(){t=l(n)},l(a){t=f(a,n)},m(a,h){s(a,t,h)},p:N,d(a){a&&i(t)}}}function Me(u){let n=String.raw`
    \dfrac{\partial L}{\partial o}
    \dfrac{\partial o}{\partial z}
    \dfrac{\partial z}{\partial a_1}
    =
    \dfrac{\partial L}{\partial o}
    \dfrac{\partial o}{\partial z}
    \dfrac{\partial z}{\partial a_2}
    `+"",t;return{c(){t=l(n)},l(a){t=f(a,n)},m(a,h){s(a,t,h)},p:N,d(a){a&&i(t)}}}function Ze(u){let n;return{c(){n=l("\\mu = 0")},l(t){n=f(t,"\\mu = 0")},m(t,a){s(t,n,a)},d(t){t&&i(n)}}}function Oe(u){let n;return{c(){n=l("\\sigma = 0.1")},l(t){n=f(t,"\\sigma = 0.1")},m(t,a){s(t,n,a)},d(t){t&&i(n)}}}function Qe(u){let n=String.raw`-0.5 \text{ to } 0.5`+"",t;return{c(){t=l(n)},l(a){t=f(a,n)},m(a,h){s(a,t,h)},p:N,d(a){a&&i(t)}}}function et(u){let n=String.raw`\mathcal{U}(-a, a)`+"",t;return{c(){t=l(n)},l(a){t=f(a,n)},m(a,h){s(a,t,h)},p:N,d(a){a&&i(t)}}}function tt(u){let n=String.raw`a = \sqrt{\frac{6}{fan_{in} + fan_{out}}}`+"",t;return{c(){t=l(n)},l(a){t=f(a,n)},m(a,h){s(a,t,h)},p:N,d(a){a&&i(t)}}}function at(u){let n=String.raw`\mathcal{N}(0, std^2)`+"",t;return{c(){t=l(n)},l(a){t=f(a,n)},m(a,h){s(a,t,h)},p:N,d(a){a&&i(t)}}}function nt(u){let n=String.raw` std = \sqrt{\frac{2}{fan_{in} + fan_{out}}}`+"",t;return{c(){t=l(n)},l(a){t=f(a,n)},m(a,h){s(a,t,h)},p:N,d(a){a&&i(t)}}}function it(u){let n=String.raw`fan_{in}`+"",t;return{c(){t=l(n)},l(a){t=f(a,n)},m(a,h){s(a,t,h)},p:N,d(a){a&&i(t)}}}function rt(u){let n=String.raw`fan_{out}`+"",t;return{c(){t=l(n)},l(a){t=f(a,n)},m(a,h){s(a,t,h)},p:N,d(a){a&&i(t)}}}function ot(u){let n,t,a,h,q,L,F,E,W,k,C,o,d,x,_e,ue,A,he,te,ve,pe,G,me,ae,ye,$e,J,be,de,p,ze,j,ke,B,xe,D,Te,M,Pe,H,Ee,K,Ie,R,Se,X,Le,U,Fe,V,We,ge,Y,Ce,Z,Ne,ce,O,qe,we;return k=new Ue({}),x=new I({props:{$$slots:{default:[Ye]},$$scope:{ctx:u}}}),A=new I({props:{$$slots:{default:[Je]},$$scope:{ctx:u}}}),G=new I({props:{$$slots:{default:[Me]},$$scope:{ctx:u}}}),j=new I({props:{$$slots:{default:[Ze]},$$scope:{ctx:u}}}),B=new I({props:{$$slots:{default:[Oe]},$$scope:{ctx:u}}}),D=new I({props:{$$slots:{default:[Qe]},$$scope:{ctx:u}}}),M=new Be({props:{id:1,type:"reference"}}),H=new I({props:{$$slots:{default:[et]},$$scope:{ctx:u}}}),K=new I({props:{$$slots:{default:[tt]},$$scope:{ctx:u}}}),R=new I({props:{$$slots:{default:[at]},$$scope:{ctx:u}}}),X=new I({props:{$$slots:{default:[nt]},$$scope:{ctx:u}}}),U=new I({props:{$$slots:{default:[it]},$$scope:{ctx:u}}}),V=new I({props:{$$slots:{default:[rt]},$$scope:{ctx:u}}}),Z=new Be({props:{id:"2",type:"reference"}}),{c(){n=T("p"),t=l(`Previously we had mentioned that weights can contribute to vanishing and
    exploding gradients. For the most part we adjust weights in a completely
    automated process by using backpropagation and applying gradient descent.
    For that reason we do not have a lot of influence on how the weights
    develop. The one place where we directly determine the distribution of
    weights is during the initialization process. This section is going to be
    dedicated to weight initialization: the pitfalls and best practices.`),a=b(),h=T("p"),q=l(`The first idea we might come up with is to initialize all weights equally,
    specifically to use 0 as the starting value for all weights and biases.`),L=b(),F=T("p"),E=l(`We will use this simple neural network to demonstrate the danger of such
    initialization. All we need to do is to work through a single forward and
    backward pass to realize the problem.`),W=b(),g(k.$$.fragment),C=b(),o=T("p"),d=l("If we have the same weight "),g(x.$$.fragment),_e=l(` for all nodes and layers, then in
    the very first forward pass all the neurons from the same layer will produce
    the same value.`),ue=b(),g(A.$$.fragment),he=b(),te=T("p"),ve=l(`When we apply backpropagation we will quickly realize that the gradients for
    each of the weights are identical for each node in a particular layer.`),pe=b(),g(G.$$.fragment),me=b(),ae=T("p"),ye=l(`The same starting values and the same gradients can only mean that all nodes
    in a layer will always have the same value. This is no different than having
    a neural network with a single neuron per layer. The network will never be
    able to solve complex problems. And if you initialize all your weights with
    zero, the network will always have dead neurons, always staying at the 0
    value.`),$e=b(),J=T("p"),be=l("Never initialize your weights uniformly. Break the symmetry!"),de=b(),p=T("p"),ze=l("For a long time researchers were using either a normal distribution (e.g. "),g(j.$$.fragment),ke=l(" and "),g(B.$$.fragment),xe=l(`) or a uniform distribution (e.g in
    the range `),g(D.$$.fragment),Te=l(`) to initialize
    weights. This might seem reasonable, but Glorot and Bengio`),g(M.$$.fragment),Pe=l(` showed that it is much more preferable to initialize weights in such a way,
    that during the forward pass the variance of input neurons and the variance of
    output neurons stays the same and during the backward pass the gradients keep
    a constant variance from layer to layer. That condition reduces the likelihood
    of vanishing or exploding gradients. The authors proposed to initialize weights
    either using a uniform distribution
    `),g(H.$$.fragment),Ee=l(" where "),g(K.$$.fragment),Ie=l(" or the normal distribution "),g(R.$$.fragment),Se=l(`, where
    `),g(X.$$.fragment),Le=l(`.
    The words `),g(U.$$.fragment),Fe=l(" and "),g(V.$$.fragment),We=l(` stand for the number of neurons that go into the layer as input and the number
    of neurons that are in the layer respectively.`),ge=b(),Y=T("p"),Ce=l(`While the Xavier/Glorot initialization was studied in conjunction with the
    sigmoind and the tanh activation function, the Kaiming/He initialization was
    designed to work with the ReLU activation`),g(Z.$$.fragment),Ne=l(". This is the standard initialization mode used in PyTorch."),ce=b(),O=T("p"),qe=l(`For the most part you will not spend a lot of time dealing with weight
    initializations. Libraries like PyTorch and Keras have good common sense
    initialization values and allow you to switch between the initialization
    modes relatively easy. You do not nead to memorize those formulas. If you
    implement backpropagation on your own don't forget to at least break the
    symmetry.`),this.h()},l(e){n=P(e,"P",{});var r=S(n);t=f(r,`Previously we had mentioned that weights can contribute to vanishing and
    exploding gradients. For the most part we adjust weights in a completely
    automated process by using backpropagation and applying gradient descent.
    For that reason we do not have a lot of influence on how the weights
    develop. The one place where we directly determine the distribution of
    weights is during the initialization process. This section is going to be
    dedicated to weight initialization: the pitfalls and best practices.`),r.forEach(i),a=z(e),h=P(e,"P",{});var ne=S(h);q=f(ne,`The first idea we might come up with is to initialize all weights equally,
    specifically to use 0 as the starting value for all weights and biases.`),ne.forEach(i),L=z(e),F=P(e,"P",{});var ie=S(F);E=f(ie,`We will use this simple neural network to demonstrate the danger of such
    initialization. All we need to do is to work through a single forward and
    backward pass to realize the problem.`),ie.forEach(i),W=z(e),c(k.$$.fragment,e),C=z(e),o=P(e,"P",{});var Q=S(o);d=f(Q,"If we have the same weight "),c(x.$$.fragment,Q),_e=f(Q,` for all nodes and layers, then in
    the very first forward pass all the neurons from the same layer will produce
    the same value.`),Q.forEach(i),ue=z(e),c(A.$$.fragment,e),he=z(e),te=P(e,"P",{});var re=S(te);ve=f(re,`When we apply backpropagation we will quickly realize that the gradients for
    each of the weights are identical for each node in a particular layer.`),re.forEach(i),pe=z(e),c(G.$$.fragment,e),me=z(e),ae=P(e,"P",{});var oe=S(ae);ye=f(oe,`The same starting values and the same gradients can only mean that all nodes
    in a layer will always have the same value. This is no different than having
    a neural network with a single neuron per layer. The network will never be
    able to solve complex problems. And if you initialize all your weights with
    zero, the network will always have dead neurons, always staying at the 0
    value.`),oe.forEach(i),$e=z(e),J=P(e,"P",{class:!0});var se=S(J);be=f(se,"Never initialize your weights uniformly. Break the symmetry!"),se.forEach(i),de=z(e),p=P(e,"P",{});var m=S(p);ze=f(m,"For a long time researchers were using either a normal distribution (e.g. "),c(j.$$.fragment,m),ke=f(m," and "),c(B.$$.fragment,m),xe=f(m,`) or a uniform distribution (e.g in
    the range `),c(D.$$.fragment,m),Te=f(m,`) to initialize
    weights. This might seem reasonable, but Glorot and Bengio`),c(M.$$.fragment,m),Pe=f(m,` showed that it is much more preferable to initialize weights in such a way,
    that during the forward pass the variance of input neurons and the variance of
    output neurons stays the same and during the backward pass the gradients keep
    a constant variance from layer to layer. That condition reduces the likelihood
    of vanishing or exploding gradients. The authors proposed to initialize weights
    either using a uniform distribution
    `),c(H.$$.fragment,m),Ee=f(m," where "),c(K.$$.fragment,m),Ie=f(m," or the normal distribution "),c(R.$$.fragment,m),Se=f(m,`, where
    `),c(X.$$.fragment,m),Le=f(m,`.
    The words `),c(U.$$.fragment,m),Fe=f(m," and "),c(V.$$.fragment,m),We=f(m,` stand for the number of neurons that go into the layer as input and the number
    of neurons that are in the layer respectively.`),m.forEach(i),ge=z(e),Y=P(e,"P",{});var ee=S(Y);Ce=f(ee,`While the Xavier/Glorot initialization was studied in conjunction with the
    sigmoind and the tanh activation function, the Kaiming/He initialization was
    designed to work with the ReLU activation`),c(Z.$$.fragment,ee),Ne=f(ee,". This is the standard initialization mode used in PyTorch."),ee.forEach(i),ce=z(e),O=P(e,"P",{class:!0});var le=S(O);qe=f(le,`For the most part you will not spend a lot of time dealing with weight
    initializations. Libraries like PyTorch and Keras have good common sense
    initialization values and allow you to switch between the initialization
    modes relatively easy. You do not nead to memorize those formulas. If you
    implement backpropagation on your own don't forget to at least break the
    symmetry.`),le.forEach(i),this.h()},h(){fe(J,"class","danger"),fe(O,"class","info")},m(e,r){s(e,n,r),$(n,t),s(e,a,r),s(e,h,r),$(h,q),s(e,L,r),s(e,F,r),$(F,E),s(e,W,r),w(k,e,r),s(e,C,r),s(e,o,r),$(o,d),w(x,o,null),$(o,_e),s(e,ue,r),w(A,e,r),s(e,he,r),s(e,te,r),$(te,ve),s(e,pe,r),w(G,e,r),s(e,me,r),s(e,ae,r),$(ae,ye),s(e,$e,r),s(e,J,r),$(J,be),s(e,de,r),s(e,p,r),$(p,ze),w(j,p,null),$(p,ke),w(B,p,null),$(p,xe),w(D,p,null),$(p,Te),w(M,p,null),$(p,Pe),w(H,p,null),$(p,Ee),w(K,p,null),$(p,Ie),w(R,p,null),$(p,Se),w(X,p,null),$(p,Le),w(U,p,null),$(p,Fe),w(V,p,null),$(p,We),s(e,ge,r),s(e,Y,r),$(Y,Ce),w(Z,Y,null),$(Y,Ne),s(e,ce,r),s(e,O,r),$(O,qe),we=!0},p(e,r){const ne={};r&2&&(ne.$$scope={dirty:r,ctx:e}),x.$set(ne);const ie={};r&2&&(ie.$$scope={dirty:r,ctx:e}),A.$set(ie);const Q={};r&2&&(Q.$$scope={dirty:r,ctx:e}),G.$set(Q);const re={};r&2&&(re.$$scope={dirty:r,ctx:e}),j.$set(re);const oe={};r&2&&(oe.$$scope={dirty:r,ctx:e}),B.$set(oe);const se={};r&2&&(se.$$scope={dirty:r,ctx:e}),D.$set(se);const m={};r&2&&(m.$$scope={dirty:r,ctx:e}),H.$set(m);const ee={};r&2&&(ee.$$scope={dirty:r,ctx:e}),K.$set(ee);const le={};r&2&&(le.$$scope={dirty:r,ctx:e}),R.$set(le);const Ae={};r&2&&(Ae.$$scope={dirty:r,ctx:e}),X.$set(Ae);const Ge={};r&2&&(Ge.$$scope={dirty:r,ctx:e}),U.$set(Ge);const je={};r&2&&(je.$$scope={dirty:r,ctx:e}),V.$set(je)},i(e){we||(_(k.$$.fragment,e),_(x.$$.fragment,e),_(A.$$.fragment,e),_(G.$$.fragment,e),_(j.$$.fragment,e),_(B.$$.fragment,e),_(D.$$.fragment,e),_(M.$$.fragment,e),_(H.$$.fragment,e),_(K.$$.fragment,e),_(R.$$.fragment,e),_(X.$$.fragment,e),_(U.$$.fragment,e),_(V.$$.fragment,e),_(Z.$$.fragment,e),we=!0)},o(e){v(k.$$.fragment,e),v(x.$$.fragment,e),v(A.$$.fragment,e),v(G.$$.fragment,e),v(j.$$.fragment,e),v(B.$$.fragment,e),v(D.$$.fragment,e),v(M.$$.fragment,e),v(H.$$.fragment,e),v(K.$$.fragment,e),v(R.$$.fragment,e),v(X.$$.fragment,e),v(U.$$.fragment,e),v(V.$$.fragment,e),v(Z.$$.fragment,e),we=!1},d(e){e&&i(n),e&&i(a),e&&i(h),e&&i(L),e&&i(F),e&&i(W),y(k,e),e&&i(C),e&&i(o),y(x),e&&i(ue),y(A,e),e&&i(he),e&&i(te),e&&i(pe),y(G,e),e&&i(me),e&&i(ae),e&&i($e),e&&i(J),e&&i(de),e&&i(p),y(j),y(B),y(D),y(M),y(H),y(K),y(R),y(X),y(U),y(V),e&&i(ge),e&&i(Y),y(Z),e&&i(ce),e&&i(O)}}}function st(u){let n,t,a,h,q,L,F,E,W,k,C;return E=new Xe({props:{$$slots:{default:[ot]},$$scope:{ctx:u}}}),k=new Ve({props:{references:u[0]}}),{c(){n=T("meta"),t=b(),a=T("h1"),h=l("Weight Initialization"),q=b(),L=T("div"),F=b(),g(E.$$.fragment),W=b(),g(k.$$.fragment),this.h()},l(o){const d=Re('[data-svelte="svelte-1ysb1ub"]',document.head);n=P(d,"META",{name:!0,content:!0}),d.forEach(i),t=z(o),a=P(o,"H1",{});var x=S(a);h=f(x,"Weight Initialization"),x.forEach(i),q=z(o),L=P(o,"DIV",{class:!0}),S(L).forEach(i),F=z(o),c(E.$$.fragment,o),W=z(o),c(k.$$.fragment,o),this.h()},h(){document.title="World4AI | Deep Learning | Weight Initialization",fe(n,"name","description"),fe(n,"content","Proper weight initialization, like (Xavier/Glorot and Kaiming/He) can decrease the chances of exploding or vanishing gradients."),fe(L,"class","separator")},m(o,d){$(document.head,n),s(o,t,d),s(o,a,d),$(a,h),s(o,q,d),s(o,L,d),s(o,F,d),w(E,o,d),s(o,W,d),w(k,o,d),C=!0},p(o,[d]){const x={};d&2&&(x.$$scope={dirty:d,ctx:o}),E.$set(x)},i(o){C||(_(E.$$.fragment,o),_(k.$$.fragment,o),C=!0)},o(o){v(E.$$.fragment,o),v(k.$$.fragment,o),C=!1},d(o){i(n),o&&i(t),o&&i(a),o&&i(q),o&&i(L),o&&i(F),y(E,o),o&&i(W),y(k,o)}}}function lt(u){return[[{author:"Glorot, Xavier and Bengio Yoshua",title:"Understanding the Difficulty of Training Deep Feedforward Neural Networks",journal:"Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, Journal of Machine Learning Research",year:"2010",pages:"249-256",volume:"9",issue:""},{author:"K. He, X. Zhang, S. Ren and J. Sun",title:" Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification",journal:"2015 IEEE International Conference on Computer Vision (ICCV)",year:"2015",pages:"1026-1024",volume:"",issue:""}]]}class gt extends De{constructor(n){super(),He(this,n,lt,st,Ke,{})}}export{gt as default};
