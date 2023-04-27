import{S as yt,i as bt,s as Et,y as w,a as $,z as v,c as h,A as _,b as o,g,d as y,B as b,h as a,C as ye,k as T,q as u,W as Tt,l as D,m as k,r as m,n as ie,N as c,L as Dt}from"../chunks/index.4d92b023.js";import{C as At}from"../chunks/Container.b0705c7b.js";import{B as kt}from"../chunks/ButtonContainer.e9aac418.js";import{P as St}from"../chunks/PlayButton.85103c5a.js";import{N as Lt}from"../chunks/NeuralNetwork.9b1e2957.js";import{F as Pt,I as gt}from"../chunks/InternalLink.7deb899c.js";import{L as oe}from"../chunks/Latex.e0b308c0.js";import{H as It}from"../chunks/Highlight.b7c1de53.js";import{A as Ct}from"../chunks/Alert.25a852b3.js";import{P as le}from"../chunks/PythonCode.212ba7a6.js";function xt(f){let n,t;return n=new St({props:{f:f[1],delta:800}}),{c(){w(n.$$.fragment)},l(r){v(n.$$.fragment,r)},m(r,d){_(n,r,d),t=!0},p:ye,i(r){t||(g(n.$$.fragment,r),t=!0)},o(r){y(n.$$.fragment,r),t=!1},d(r){b(n,r)}}}function Nt(f){let n,t,r,d;return n=new kt({props:{$$slots:{default:[xt]},$$scope:{ctx:f}}}),r=new Lt({props:{width:Vt,maxWidth:zt,height:Mt,rectSize:Wt,layers:f[0]}}),{c(){w(n.$$.fragment),t=$(),w(r.$$.fragment)},l(i){v(n.$$.fragment,i),t=h(i),v(r.$$.fragment,i)},m(i,p){_(n,i,p),o(i,t,p),_(r,i,p),d=!0},p(i,[p]){const S={};p&4&&(S.$$scope={dirty:p,ctx:i}),n.$set(S);const A={};p&1&&(A.layers=i[0]),r.$set(A)},i(i){d||(g(n.$$.fragment,i),g(r.$$.fragment,i),d=!0)},o(i){y(n.$$.fragment,i),y(r.$$.fragment,i),d=!1},d(i){b(n,i),i&&a(t),b(r,i)}}}const Vt=500,zt="700px",Mt=450,Wt=20;let Ot=.2;function Rt(f,n,t){let r=[{title:"",nodes:[{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"}]},{title:"",nodes:[{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"}]},{title:"",nodes:[{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"}]},{title:"",nodes:[{value:"",class:"fill-white"}]}];function d(){r.forEach((i,p)=>{p!==r.length-1&&p!==0&&i.nodes.forEach((S,A)=>{Math.random()>=Ot?t(0,r[p].nodes[A].class="fill-white",r):t(0,r[p].nodes[A].class="fill-w4ai-red",r)})})}return[r,d]}class qt extends yt{constructor(n){super(),bt(this,n,Rt,Nt,Et,{})}}const Bt=""+new URL("../assets/dropout_overfitting.1f080882.webp",import.meta.url).href;function Ht(f){let n;return{c(){n=u("Dropout")},l(t){n=m(t,"Dropout")},m(t,r){o(t,n,r)},d(t){t&&a(n)}}}function Gt(f){let n;return{c(){n=u("p")},l(t){n=m(t,"p")},m(t,r){o(t,n,r)},d(t){t&&a(n)}}}function Ut(f){let n,t,r,d;return t=new oe({props:{$$slots:{default:[Gt]},$$scope:{ctx:f}}}),{c(){n=u("At each training step dropout deactivates a neuron with a probability of "),w(t.$$.fragment),r=u(". The deactivated neurons are set to a value at 0.")},l(i){n=m(i,"At each training step dropout deactivates a neuron with a probability of "),v(t.$$.fragment,i),r=m(i,". The deactivated neurons are set to a value at 0.")},m(i,p){o(i,n,p),_(t,i,p),o(i,r,p),d=!0},p(i,p){const S={};p&256&&(S.$$scope={dirty:p,ctx:i}),t.$set(S)},i(i){d||(g(t.$$.fragment,i),d=!0)},o(i){y(t.$$.fragment,i),d=!1},d(i){i&&a(n),b(t,i),i&&a(r)}}}function jt(f){let n;return{c(){n=u("p")},l(t){n=m(t,"p")},m(t,r){o(t,n,r)},d(t){t&&a(n)}}}function Ft(f){let n=String.raw`\dfrac{1}{1-p}`+"",t;return{c(){t=u(n)},l(r){t=m(r,n)},m(r,d){o(r,t,d)},p:ye,d(r){r&&a(t)}}}function Yt(f){let n;return{c(){n=u("p")},l(t){n=m(t,"p")},m(t,r){o(t,n,r)},d(t){t&&a(n)}}}function Zt(f){let n=String.raw`
    \begin{bmatrix}
    1 \\
    1 \\
    1 \\
    1 \\
    1 \\
    1 \\
    \end{bmatrix}
    `+"",t;return{c(){t=u(n)},l(r){t=m(r,n)},m(r,d){o(r,t,d)},p:ye,d(r){r&&a(t)}}}function Kt(f){let n=String.raw`\dfrac{1}{1-0.5} = 2`+"",t;return{c(){t=u(n)},l(r){t=m(r,n)},m(r,d){o(r,t,d)},p:ye,d(r){r&&a(t)}}}function Jt(f){let n=String.raw`
    \begin{bmatrix}
    2 \\
    0 \\
    0 \\
    2 \\
    0 \\
    2 \\
    \end{bmatrix}
    `+"",t;return{c(){t=u(n)},l(r){t=m(r,n)},m(r,d){o(r,t,d)},p:ye,d(r){r&&a(t)}}}function Qt(f){let n,t,r,d,i,p,S,A,L,I,P,s,E,x,be,G,Ee,se,je,Te,N,Fe,V,Ye,De,z,Ze,M,Ke,Ae,U,W,ke,O,Je,R,Qe,Se,j,q,Le,fe,Xe,Pe,C,et,de,tt,nt,ce,at,rt,Ie,F,Ce,ue,lt,xe,B,ot,$e,it,st,Ne,Y,Ve,H,ft,he,ut,mt,ze,Z,Me,K,We,me,pt,Oe,J,Re,Q,qe,X,Be,ee,dt,He;return t=new It({props:{$$slots:{default:[Ht]},$$scope:{ctx:f}}}),d=new gt({props:{id:1,type:"reference"}}),p=new gt({props:{id:2,type:"reference"}}),L=new Ct({props:{type:"info",$$slots:{default:[Ut]},$$scope:{ctx:f}}}),E=new oe({props:{$$slots:{default:[jt]},$$scope:{ctx:f}}}),G=new qt({}),V=new oe({props:{$$slots:{default:[Ft]},$$scope:{ctx:f}}}),M=new oe({props:{$$slots:{default:[Yt]},$$scope:{ctx:f}}}),W=new oe({props:{$$slots:{default:[Zt]},$$scope:{ctx:f}}}),R=new oe({props:{$$slots:{default:[Kt]},$$scope:{ctx:f}}}),q=new oe({props:{$$slots:{default:[Jt]},$$scope:{ctx:f}}}),F=new le({props:{code:f[3]}}),Y=new le({props:{code:f[1]}}),Z=new le({props:{code:f[2]}}),K=new le({props:{code:f[4]}}),J=new le({props:{code:f[5]}}),Q=new le({props:{code:f[6],isOutput:!0}}),X=new le({props:{code:f[7]}}),{c(){n=T("p"),w(t.$$.fragment),r=u(` is a regularization technique, that was developed
    by Geoffrey Hinton and his colleagues at the university of Toronto `),w(d.$$.fragment),i=$(),w(p.$$.fragment),S=u(`
    .`),A=$(),w(L.$$.fragment),I=$(),P=T("p"),s=u(`You can use the interactive example below. We use a relatively small neural
    network below and apply dropout with a `),w(E.$$.fragment),x=u(` value of 0.2 to the two
    hidden layer.`),be=$(),w(G.$$.fragment),Ee=$(),se=T("p"),je=u(`Theoretically you can apply dropout to any layer you desire, but most likely
    you will not want to deactivate the input and the output layers.`),Te=$(),N=T("p"),Fe=u(`When we use our model for inference, we do not remove any of the nodes
    randomly. If we did that, we would get different results each time we run a
    model. By not deactivating any nodes we introduce a problem though. More
    neurons are active during inference, therefore each layer has to deal with
    an input that is on a different scale, than the one the neural network has
    seen during training. Different conditions during training and inference
    will prevent the neural network from generating good predictions. To prevent
    that from happening the active nodes are scaled by
    `),w(V.$$.fragment),Ye=u(` during training, which increases
    the magnitude of the input signal. We skip that scaling during inference, but
    the average signal strength remains fairly similar, as the network now has more
    neurons as inputs.`),De=$(),z=T("p"),Ze=u(`Let us assume for example that a layer of neurons contains only 1's and
    `),w(M.$$.fragment),Ke=u(" is 0.5."),Ae=$(),U=T("div"),w(W.$$.fragment),ke=$(),O=T("p"),Je=u(`The dropout layer will zero out roughly half of the activations and multiply
    the rest by `),w(R.$$.fragment),Qe=u("."),Se=$(),j=T("div"),w(q.$$.fragment),Le=$(),fe=T("p"),Xe=u(`But why is the dropout procedure helpful in avoiding overfitting? Each time
    we remove a set of neurons from training, we essentially create a different,
    simplified model. This simplified model has to learn to deal with the task
    at hand without overrelying on any of the neurons from the previous layer,
    because any of those might get deactivated at any time. The final model can
    be seen as an ensemble of an immensely huge collection of simplified models.
    Ensembles tend to produce better results and reduce overfitting. You will
    notice that in practice dropout works extremely well.`),Pe=$(),C=T("p"),et=u("As expected PyTorch provides built-in "),de=T("code"),tt=u("nn.Dropout()"),nt=u(` modules with
    a probability parameter `),ce=T("code"),at=u("p"),rt=u(`. We add those after each of the
    hidden layers and we are good to go.`),Ie=$(),w(F.$$.fragment),Ce=$(),ue=T("p"),lt=u(`There are a couple more adjustments we need to make, to actually make our
    code behave the way we desire. We can set modules in training and evaluation
    modes. Modules might behave differently, depending on the mode they are in.
    As mentioned before dropout needs to behave differently depending on whether
    we are training or evaluating, but there are more layers in PyTorch, that
    require that distinction.`),xe=$(),B=T("p"),ot=u(`To actually set the modules in different modes is actually qute easy. Below
    we use `),$e=T("code"),it=u("model.eval()"),st=u(` to start evaluation mode at the start of the
    function.`),Ne=$(),w(Y.$$.fragment),Ve=$(),H=T("p"),ft=u("The method "),he=T("code"),ut=u("model.train()"),mt=u(` on the other hand puts all the modules
    in training mode.`),ze=$(),w(Z.$$.fragment),Me=$(),w(K.$$.fragment),We=$(),me=T("p"),pt=u(`There is still some distance between the performance of the training and the
    validation datasets, but we are manage to reduce overfitting by using
    dropout.`),Oe=$(),w(J.$$.fragment),Re=$(),w(Q.$$.fragment),qe=$(),w(X.$$.fragment),Be=$(),ee=T("img"),this.h()},l(e){n=D(e,"P",{});var l=k(n);v(t.$$.fragment,l),r=m(l,` is a regularization technique, that was developed
    by Geoffrey Hinton and his colleagues at the university of Toronto `),v(d.$$.fragment,l),i=h(l),v(p.$$.fragment,l),S=m(l,`
    .`),l.forEach(a),A=h(e),v(L.$$.fragment,e),I=h(e),P=D(e,"P",{});var te=k(P);s=m(te,`You can use the interactive example below. We use a relatively small neural
    network below and apply dropout with a `),v(E.$$.fragment,te),x=m(te,` value of 0.2 to the two
    hidden layer.`),te.forEach(a),be=h(e),v(G.$$.fragment,e),Ee=h(e),se=D(e,"P",{});var we=k(se);je=m(we,`Theoretically you can apply dropout to any layer you desire, but most likely
    you will not want to deactivate the input and the output layers.`),we.forEach(a),Te=h(e),N=D(e,"P",{});var ne=k(N);Fe=m(ne,`When we use our model for inference, we do not remove any of the nodes
    randomly. If we did that, we would get different results each time we run a
    model. By not deactivating any nodes we introduce a problem though. More
    neurons are active during inference, therefore each layer has to deal with
    an input that is on a different scale, than the one the neural network has
    seen during training. Different conditions during training and inference
    will prevent the neural network from generating good predictions. To prevent
    that from happening the active nodes are scaled by
    `),v(V.$$.fragment,ne),Ye=m(ne,` during training, which increases
    the magnitude of the input signal. We skip that scaling during inference, but
    the average signal strength remains fairly similar, as the network now has more
    neurons as inputs.`),ne.forEach(a),De=h(e),z=D(e,"P",{});var ae=k(z);Ze=m(ae,`Let us assume for example that a layer of neurons contains only 1's and
    `),v(M.$$.fragment,ae),Ke=m(ae," is 0.5."),ae.forEach(a),Ae=h(e),U=D(e,"DIV",{class:!0});var ve=k(U);v(W.$$.fragment,ve),ve.forEach(a),ke=h(e),O=D(e,"P",{});var re=k(O);Je=m(re,`The dropout layer will zero out roughly half of the activations and multiply
    the rest by `),v(R.$$.fragment,re),Qe=m(re,"."),re.forEach(a),Se=h(e),j=D(e,"DIV",{class:!0});var _e=k(j);v(q.$$.fragment,_e),_e.forEach(a),Le=h(e),fe=D(e,"P",{});var ge=k(fe);Xe=m(ge,`But why is the dropout procedure helpful in avoiding overfitting? Each time
    we remove a set of neurons from training, we essentially create a different,
    simplified model. This simplified model has to learn to deal with the task
    at hand without overrelying on any of the neurons from the previous layer,
    because any of those might get deactivated at any time. The final model can
    be seen as an ensemble of an immensely huge collection of simplified models.
    Ensembles tend to produce better results and reduce overfitting. You will
    notice that in practice dropout works extremely well.`),ge.forEach(a),Pe=h(e),C=D(e,"P",{});var pe=k(C);et=m(pe,"As expected PyTorch provides built-in "),de=D(pe,"CODE",{});var ct=k(de);tt=m(ct,"nn.Dropout()"),ct.forEach(a),nt=m(pe,` modules with
    a probability parameter `),ce=D(pe,"CODE",{});var $t=k(ce);at=m($t,"p"),$t.forEach(a),rt=m(pe,`. We add those after each of the
    hidden layers and we are good to go.`),pe.forEach(a),Ie=h(e),v(F.$$.fragment,e),Ce=h(e),ue=D(e,"P",{});var ht=k(ue);lt=m(ht,`There are a couple more adjustments we need to make, to actually make our
    code behave the way we desire. We can set modules in training and evaluation
    modes. Modules might behave differently, depending on the mode they are in.
    As mentioned before dropout needs to behave differently depending on whether
    we are training or evaluating, but there are more layers in PyTorch, that
    require that distinction.`),ht.forEach(a),xe=h(e),B=D(e,"P",{});var Ge=k(B);ot=m(Ge,`To actually set the modules in different modes is actually qute easy. Below
    we use `),$e=D(Ge,"CODE",{});var wt=k($e);it=m(wt,"model.eval()"),wt.forEach(a),st=m(Ge,` to start evaluation mode at the start of the
    function.`),Ge.forEach(a),Ne=h(e),v(Y.$$.fragment,e),Ve=h(e),H=D(e,"P",{});var Ue=k(H);ft=m(Ue,"The method "),he=D(Ue,"CODE",{});var vt=k(he);ut=m(vt,"model.train()"),vt.forEach(a),mt=m(Ue,` on the other hand puts all the modules
    in training mode.`),Ue.forEach(a),ze=h(e),v(Z.$$.fragment,e),Me=h(e),v(K.$$.fragment,e),We=h(e),me=D(e,"P",{});var _t=k(me);pt=m(_t,`There is still some distance between the performance of the training and the
    validation datasets, but we are manage to reduce overfitting by using
    dropout.`),_t.forEach(a),Oe=h(e),v(J.$$.fragment,e),Re=h(e),v(Q.$$.fragment,e),qe=h(e),v(X.$$.fragment,e),Be=h(e),ee=D(e,"IMG",{src:!0,alt:!0}),this.h()},h(){ie(U,"class","flex justify-center"),ie(j,"class","flex justify-center"),Dt(ee.src,dt=Bt)||ie(ee,"src",dt),ie(ee,"alt","Overfitting with dropout")},m(e,l){o(e,n,l),_(t,n,null),c(n,r),_(d,n,null),c(n,i),_(p,n,null),c(n,S),o(e,A,l),_(L,e,l),o(e,I,l),o(e,P,l),c(P,s),_(E,P,null),c(P,x),o(e,be,l),_(G,e,l),o(e,Ee,l),o(e,se,l),c(se,je),o(e,Te,l),o(e,N,l),c(N,Fe),_(V,N,null),c(N,Ye),o(e,De,l),o(e,z,l),c(z,Ze),_(M,z,null),c(z,Ke),o(e,Ae,l),o(e,U,l),_(W,U,null),o(e,ke,l),o(e,O,l),c(O,Je),_(R,O,null),c(O,Qe),o(e,Se,l),o(e,j,l),_(q,j,null),o(e,Le,l),o(e,fe,l),c(fe,Xe),o(e,Pe,l),o(e,C,l),c(C,et),c(C,de),c(de,tt),c(C,nt),c(C,ce),c(ce,at),c(C,rt),o(e,Ie,l),_(F,e,l),o(e,Ce,l),o(e,ue,l),c(ue,lt),o(e,xe,l),o(e,B,l),c(B,ot),c(B,$e),c($e,it),c(B,st),o(e,Ne,l),_(Y,e,l),o(e,Ve,l),o(e,H,l),c(H,ft),c(H,he),c(he,ut),c(H,mt),o(e,ze,l),_(Z,e,l),o(e,Me,l),_(K,e,l),o(e,We,l),o(e,me,l),c(me,pt),o(e,Oe,l),_(J,e,l),o(e,Re,l),_(Q,e,l),o(e,qe,l),_(X,e,l),o(e,Be,l),o(e,ee,l),He=!0},p(e,l){const te={};l&256&&(te.$$scope={dirty:l,ctx:e}),t.$set(te);const we={};l&256&&(we.$$scope={dirty:l,ctx:e}),L.$set(we);const ne={};l&256&&(ne.$$scope={dirty:l,ctx:e}),E.$set(ne);const ae={};l&256&&(ae.$$scope={dirty:l,ctx:e}),V.$set(ae);const ve={};l&256&&(ve.$$scope={dirty:l,ctx:e}),M.$set(ve);const re={};l&256&&(re.$$scope={dirty:l,ctx:e}),W.$set(re);const _e={};l&256&&(_e.$$scope={dirty:l,ctx:e}),R.$set(_e);const ge={};l&256&&(ge.$$scope={dirty:l,ctx:e}),q.$set(ge)},i(e){He||(g(t.$$.fragment,e),g(d.$$.fragment,e),g(p.$$.fragment,e),g(L.$$.fragment,e),g(E.$$.fragment,e),g(G.$$.fragment,e),g(V.$$.fragment,e),g(M.$$.fragment,e),g(W.$$.fragment,e),g(R.$$.fragment,e),g(q.$$.fragment,e),g(F.$$.fragment,e),g(Y.$$.fragment,e),g(Z.$$.fragment,e),g(K.$$.fragment,e),g(J.$$.fragment,e),g(Q.$$.fragment,e),g(X.$$.fragment,e),He=!0)},o(e){y(t.$$.fragment,e),y(d.$$.fragment,e),y(p.$$.fragment,e),y(L.$$.fragment,e),y(E.$$.fragment,e),y(G.$$.fragment,e),y(V.$$.fragment,e),y(M.$$.fragment,e),y(W.$$.fragment,e),y(R.$$.fragment,e),y(q.$$.fragment,e),y(F.$$.fragment,e),y(Y.$$.fragment,e),y(Z.$$.fragment,e),y(K.$$.fragment,e),y(J.$$.fragment,e),y(Q.$$.fragment,e),y(X.$$.fragment,e),He=!1},d(e){e&&a(n),b(t),b(d),b(p),e&&a(A),b(L,e),e&&a(I),e&&a(P),b(E),e&&a(be),b(G,e),e&&a(Ee),e&&a(se),e&&a(Te),e&&a(N),b(V),e&&a(De),e&&a(z),b(M),e&&a(Ae),e&&a(U),b(W),e&&a(ke),e&&a(O),b(R),e&&a(Se),e&&a(j),b(q),e&&a(Le),e&&a(fe),e&&a(Pe),e&&a(C),e&&a(Ie),b(F,e),e&&a(Ce),e&&a(ue),e&&a(xe),e&&a(B),e&&a(Ne),b(Y,e),e&&a(Ve),e&&a(H),e&&a(ze),b(Z,e),e&&a(Me),b(K,e),e&&a(We),e&&a(me),e&&a(Oe),b(J,e),e&&a(Re),b(Q,e),e&&a(qe),b(X,e),e&&a(Be),e&&a(ee)}}}function Xt(f){let n,t,r,d,i,p,S,A,L,I,P;return A=new At({props:{$$slots:{default:[Qt]},$$scope:{ctx:f}}}),I=new Pt({props:{references:f[0]}}),{c(){n=T("meta"),t=$(),r=T("h1"),d=u("Dropout"),i=$(),p=T("div"),S=$(),w(A.$$.fragment),L=$(),w(I.$$.fragment),this.h()},l(s){const E=Tt("svelte-1qhlwaf",document.head);n=D(E,"META",{name:!0,content:!0}),E.forEach(a),t=h(s),r=D(s,"H1",{});var x=k(r);d=m(x,"Dropout"),x.forEach(a),i=h(s),p=D(s,"DIV",{class:!0}),k(p).forEach(a),S=h(s),v(A.$$.fragment,s),L=h(s),v(I.$$.fragment,s),this.h()},h(){document.title="Dropout - World4AI",ie(n,"name","description"),ie(n,"content","Dropout is a regularization technique that works by randomly deactivating different neurons at each training step. Dropout essentially reduces the overreliance on any particular neuron and generates a different (simpler) model at each training step, thereby reducing overfitting."),ie(p,"class","separator")},m(s,E){c(document.head,n),o(s,t,E),o(s,r,E),c(r,d),o(s,i,E),o(s,p,E),o(s,S,E),_(A,s,E),o(s,L,E),_(I,s,E),P=!0},p(s,[E]){const x={};E&256&&(x.$$scope={dirty:E,ctx:s}),A.$set(x)},i(s){P||(g(A.$$.fragment,s),g(I.$$.fragment,s),P=!0)},o(s){y(A.$$.fragment,s),y(I.$$.fragment,s),P=!1},d(s){a(n),s&&a(t),s&&a(r),s&&a(i),s&&a(p),s&&a(S),b(A,s),s&&a(L),b(I,s)}}}function en(f){return[[{author:"G. E. Hinton, N. Srivastava, A. Krizhevsky, I. Sutskever and R. R. Salakhutdivno",title:"Improving neural networks by preventing co-adaptation of feature detectors",journal:"",year:"2012",pages:"",volume:"",issue:""},{author:"Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov",title:"Dropout: A Simple Way to Prevent Neural Networks from Overfitting",journal:"Journal Of Machine Learning Research",year:"2014",pages:"1929-1958",volume:"15",issue:"1"}],`def track_performance(dataloader, model, criterion):
    # switch to evaluation mode
    model.eval()

    num_samples = 0
    num_correct = 0
    loss_sum = 0
    
    # no need to calculate gradients
    with torch.inference_mode():
        for batch_idx, (features, labels) in enumerate(dataloader):
            features = features.view(-1, NUM_FEATURES).to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(features)
            
            predictions = logits.max(dim=1)[1]
            num_correct += (predictions == labels).sum().item()
            
            loss = criterion(logits, labels)
            loss_sum += loss.cpu().item()
            num_samples += len(features)
    
    # we return the average loss and the accuracy
    return loss_sum/num_samples, num_correct/num_samples`,`def train_epoch(dataloader, model, criterion, optimizer):
    # switch to training mode
    model.train()
    for batch_idx, (features, labels) in enumerate(train_dataloader):
        # move features and labels to GPU
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        # ------ FORWARD PASS --------
        output = model(features)

        # ------CALCULATE LOSS --------
        loss = criterion(output, labels)

        # ------BACKPROPAGATION --------
        loss.backward()

        # ------GRADIENT DESCENT --------
        optimizer.step()

        # ------CLEAR GRADIENTS --------
        optimizer.zero_grad()`,`class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(NUM_FEATURES, HIDDEN_SIZE_1),
                nn.Sigmoid(),
                nn.Dropout(p=0.6),
                nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
                nn.Sigmoid(),
                nn.Dropout(p=0.6),
                nn.Linear(HIDDEN_SIZE_2, NUM_LABELS),
            )
    
    def forward(self, features):
        return self.layers(features)`,`model = Model().to(DEVICE)
criterion = nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.SGD(model.parameters(), lr=0.005)`,"history = train(NUM_EPOCHS, train_dataloader, val_dataloader, model, criterion, optimizer)",`Epoch: 1/50|Train Loss: 0.9384 |Val Loss: 0.9489 |Train Acc: 0.6870 |Val Acc: 0.6770
Epoch: 10/50|Train Loss: 0.2586 |Val Loss: 0.2698 |Train Acc: 0.9232 |Val Acc: 0.9202
Epoch: 20/50|Train Loss: 0.1905 |Val Loss: 0.2089 |Train Acc: 0.9427 |Val Acc: 0.9363
Epoch: 30/50|Train Loss: 0.1658 |Val Loss: 0.1869 |Train Acc: 0.9479 |Val Acc: 0.9417
Epoch: 40/50|Train Loss: 0.1450 |Val Loss: 0.1732 |Train Acc: 0.9547 |Val Acc: 0.9458
Epoch: 50/50|Train Loss: 0.1313 |Val Loss: 0.1624 |Train Acc: 0.9590 |Val Acc: 0.9483
`,"plot_history(history, 'dropout_overfitting')"]}class pn extends yt{constructor(n){super(),bt(this,n,en,Xt,Et,{})}}export{pn as default};
