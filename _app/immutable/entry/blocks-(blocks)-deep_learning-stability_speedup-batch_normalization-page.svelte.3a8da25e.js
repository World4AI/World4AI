import{S as ta,i as aa,s as na,k as b,a as g,q as i,y as h,W as la,l as y,h as n,c as v,m as z,r as o,z as c,n as je,N as u,b as r,A as d,g as p,d as w,B as _,C as Ie}from"../chunks/index.4d92b023.js";import{C as sa}from"../chunks/Container.b0705c7b.js";import{N as Yt}from"../chunks/NeuralNetwork.9b1e2957.js";import{H as ra}from"../chunks/Highlight.b7c1de53.js";import{L as E}from"../chunks/Latex.e0b308c0.js";import{F as ia,I as oa}from"../chunks/InternalLink.7deb899c.js";import{P as ea}from"../chunks/PythonCode.212ba7a6.js";function fa(f){let a;return{c(){a=i("batch normalizaton")},l(t){a=o(t,"batch normalizaton")},m(t,s){r(t,a,s)},d(t){t&&n(a)}}}function ua(f){let a;return{c(){a=i("l")},l(t){a=o(t,"l")},m(t,s){r(t,a,s)},d(t){t&&n(a)}}}function ma(f){let a=String.raw`\mu_j`+"",t;return{c(){t=i(a)},l(s){t=o(s,a)},m(s,k){r(s,t,k)},p:Ie,d(s){s&&n(t)}}}function $a(f){let a=String.raw`\sigma_j^2`+"",t;return{c(){t=i(a)},l(s){t=o(s,a)},m(s,k){r(s,t,k)},p:Ie,d(s){s&&n(t)}}}function ha(f){let a;return{c(){a=i("j")},l(t){a=o(t,"j")},m(t,s){r(t,a,s)},d(t){t&&n(a)}}}function ca(f){let a=String.raw`
  \begin{aligned}
    \mu_j &= \dfrac{1}{n}\sum_{i=1}^n a_j^{(i)} \\
    \sigma_j^2 &= \dfrac{1}{n}\sum_{i=1}^n (a_j^{(i)} - \mu_j)
  \end{aligned}
    `+"",t;return{c(){t=i(a)},l(s){t=o(s,a)},m(s,k){r(s,t,k)},p:Ie,d(s){s&&n(t)}}}function da(f){let a=String.raw`\hat{a}_j^{(i)} = \dfrac{a_j^{(i)} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}`+"",t;return{c(){t=i(a)},l(s){t=o(s,a)},m(s,k){r(s,t,k)},p:Ie,d(s){s&&n(t)}}}function pa(f){let a=String.raw`\bar{a}_j^{(i)} = \gamma_j \hat{a}_j^{(i)} + \beta_j`+"",t;return{c(){t=i(a)},l(s){t=o(s,a)},m(s,k){r(s,t,k)},p:Ie,d(s){s&&n(t)}}}function wa(f){let a;return{c(){a=i("\\gamma")},l(t){a=o(t,"\\gamma")},m(t,s){r(t,a,s)},d(t){t&&n(a)}}}function _a(f){let a;return{c(){a=i("\\beta")},l(t){a=o(t,"\\beta")},m(t,s){r(t,a,s)},d(t){t&&n(a)}}}function ga(f){let a;return{c(){a=i("\\gamma_j")},l(t){a=o(t,"\\gamma_j")},m(t,s){r(t,a,s)},d(t){t&&n(a)}}}function va(f){let a;return{c(){a=i("\\sigma_j")},l(t){a=o(t,"\\sigma_j")},m(t,s){r(t,a,s)},d(t){t&&n(a)}}}function ba(f){let a;return{c(){a=i("\\beta_j")},l(t){a=o(t,"\\beta_j")},m(t,s){r(t,a,s)},d(t){t&&n(a)}}}function ya(f){let a;return{c(){a=i("\\mu_j")},l(t){a=o(t,"\\mu_j")},m(t,s){r(t,a,s)},d(t){t&&n(a)}}}function za(f){let a;return{c(){a=i("b")},l(t){a=o(t,"b")},m(t,s){r(t,a,s)},d(t){t&&n(a)}}}function ka(f){let a=String.raw`z = \mathbf{xw^T} + b`+"",t;return{c(){t=i(a)},l(s){t=o(s,a)},m(s,k){r(s,t,k)},p:Ie,d(s){s&&n(t)}}}function Ea(f){let a;return{c(){a=i("\\beta")},l(t){a=o(t,"\\beta")},m(t,s){r(t,a,s)},d(t){t&&n(a)}}}function xa(f){let a;return{c(){a=i("\\mu_j")},l(t){a=o(t,"\\mu_j")},m(t,s){r(t,a,s)},d(t){t&&n(a)}}}function Ta(f){let a;return{c(){a=i("\\sigma_j")},l(t){a=o(t,"\\sigma_j")},m(t,s){r(t,a,s)},d(t){t&&n(a)}}}function ja(f){let a;return{c(){a=i("\\mu")},l(t){a=o(t,"\\mu")},m(t,s){r(t,a,s)},d(t){t&&n(a)}}}function Ia(f){let a;return{c(){a=i("\\sigma")},l(t){a=o(t,"\\sigma")},m(t,s){r(t,a,s)},d(t){t&&n(a)}}}function Na(f){let a,t,s,k,W,T,R,I,V,N,G,m,$,L,J,ct,K,dt,Q,pt,X,wt,Ge,ce,Y,Je,be,_t,Ke,de,Z,Qe,ye,gt,Xe,pe,ee,Ye,x,vt,te,bt,ae,yt,ne,zt,le,kt,se,Et,re,xt,Ze,ze,Tt,et,we,tt,ke,jt,at,_e,nt,Ee,It,lt,D,Nt,ie,St,oe,Dt,fe,Pt,st,B,Bt,ue,Ft,me,At,rt,F,Ct,$e,Wt,he,Lt,it,xe,qt,ot,A,Mt,Ne,Ht,Ut,Se,Ot,Rt,ft,ge,ut,Te,Vt,mt,ve,$t,C,Gt,De,Jt,Kt,Pe,Qt,Xt,ht;return k=new Yt({props:{height:100,width:250,maxWidth:"500px",layers:f[1]}}),I=new oa({props:{type:"reference",id:"1"}}),N=new ra({props:{$$slots:{default:[fa]},$$scope:{ctx:f}}}),J=new E({props:{$$slots:{default:[ua]},$$scope:{ctx:f}}}),K=new E({props:{$$slots:{default:[ma]},$$scope:{ctx:f}}}),Q=new E({props:{$$slots:{default:[$a]},$$scope:{ctx:f}}}),X=new E({props:{$$slots:{default:[ha]},$$scope:{ctx:f}}}),Y=new E({props:{$$slots:{default:[ca]},$$scope:{ctx:f}}}),Z=new E({props:{$$slots:{default:[da]},$$scope:{ctx:f}}}),ee=new E({props:{$$slots:{default:[pa]},$$scope:{ctx:f}}}),te=new E({props:{$$slots:{default:[wa]},$$scope:{ctx:f}}}),ae=new E({props:{$$slots:{default:[_a]},$$scope:{ctx:f}}}),ne=new E({props:{$$slots:{default:[ga]},$$scope:{ctx:f}}}),le=new E({props:{$$slots:{default:[va]},$$scope:{ctx:f}}}),se=new E({props:{$$slots:{default:[ba]},$$scope:{ctx:f}}}),re=new E({props:{$$slots:{default:[ya]},$$scope:{ctx:f}}}),we=new Yt({props:{height:50,width:250,maxWidth:"600px",layers:f[2],padding:{left:0,right:30}}}),_e=new Yt({props:{height:50,width:250,maxWidth:"600px",layers:f[3],padding:{left:0,right:30}}}),ie=new E({props:{$$slots:{default:[za]},$$scope:{ctx:f}}}),oe=new E({props:{$$slots:{default:[ka]},$$scope:{ctx:f}}}),fe=new E({props:{$$slots:{default:[Ea]},$$scope:{ctx:f}}}),ue=new E({props:{$$slots:{default:[xa]},$$scope:{ctx:f}}}),me=new E({props:{$$slots:{default:[Ta]},$$scope:{ctx:f}}}),$e=new E({props:{$$slots:{default:[ja]},$$scope:{ctx:f}}}),he=new E({props:{$$slots:{default:[Ia]},$$scope:{ctx:f}}}),ge=new ea({props:{code:`HIDDEN_FEATURES = 70
class BatchModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(HIDDEN_FEATURES, HIDDEN_FEATURES, bias=False),
            nn.BatchNorm1d(HIDDEN),
            nn.ReLU()
        )
    
    def forward(self, features):
        return self.layers(features)`}}),ve=new ea({props:{code:`class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(NUM_FEATURES, HIDDEN_FEATURES),
                BatchModule(),
                BatchModule(),
                BatchModule(),
                nn.Linear(HIDDEN, NUM_LABELS),
            )
    
    def forward(self, features):
        return self.layers(features)`}}),{c(){a=b("p"),t=i(`In a previous chapter we have discussed feature scaling. Feature scaling
    only applies to the input layer, so should we try to scale the intermediary
    features that are produced by hidden units? Would that be in any form
    benefitiary for trainig?`),s=g(),h(k.$$.fragment),W=g(),T=b("p"),R=i(`Sergey Ioffe and Christian Szegedy answered the question with a definitive
    yes`),h(I.$$.fragment),V=i(". When we add so called "),h(N.$$.fragment),G=i(`
    to hidden features, we can speed up the training process significantly, while
    gaining other additional advantages.`),m=g(),$=b("p"),L=i("Consider a particular layer "),h(J.$$.fragment),ct=i(`, to which output we would like
    to apply batch normalization. Using a batch of data we calculate the mean `),h(K.$$.fragment),dt=i(` and the variance
    `),h(Q.$$.fragment),pt=i(" for each hidden unit "),h(X.$$.fragment),wt=i(" in the layer."),Ge=g(),ce=b("div"),h(Y.$$.fragment),Je=g(),be=b("p"),_t=i(`Given those parameters we can normalize the hidden features, using the same
    procedure we used for feature scaling.`),Ke=g(),de=b("div"),h(Z.$$.fragment),Qe=g(),ye=b("p"),gt=i(`The authors argued that this normalization procedure might theoretically be
    detremental to the performance, because it might reduce the expressiveness
    of the neural network. To combat that they introduced an additional step
    that allowed the neural network to reverse the standardization.`),Xe=g(),pe=b("div"),h(ee.$$.fragment),Ye=g(),x=b("p"),vt=i("The feature specific parameters "),h(te.$$.fragment),bt=i(" and "),h(ae.$$.fragment),yt=i(" are learned by the neural network. If the network decides to set "),h(ne.$$.fragment),zt=i(` to
    `),h(le.$$.fragment),kt=i(" and "),h(se.$$.fragment),Et=i(" to "),h(re.$$.fragment),xt=i(` that
    essentially neutralizes the normalization. If normalization indeed worsens the
    performance, the neural network has the option to reverse the normalization step.`),Ze=g(),ze=b("p"),Tt=i(`Our formulations above indicated that batch normalization is applied to
    activations. This procedure is similar to input feature scaling, because you
    normalize the data that is processed in the next layer.`),et=g(),h(we.$$.fragment),tt=g(),ke=b("p"),jt=i(`In practice though batch norm is often applied to the net inputs and the
    result is forwarded to the activation function.`),at=g(),h(_e.$$.fragment),nt=g(),Ee=b("p"),It=i(`There is no real consensus about how you should apply batch normalization,
    but this decision in all likelihood should not make or break your project.`),lt=g(),D=b("p"),Nt=i("There is an additional caveat. In practice we often remove the bias term "),h(ie.$$.fragment),St=i(" from the calculation of the net input "),h(oe.$$.fragment),Dt=i(". This is done due to the assumption, that "),h(fe.$$.fragment),Pt=i(` essentially
    does the same operation. Both are used to shift the calculation by a constant
    and there is hardly a reason to do that calculation twice.`),st=g(),B=b("p"),Bt=i(`The authors observed several adantages that batch normalization provides.
    For once batch norm makes the model less sensitive to the choice of the
    learning rate, which allows us to increase the learning rate and thereby
    increase the speed of convergence. Second, the model is more forgiving when
    choosing bad initial weights. Third, batch normalization seems to help with
    the vanishing gradients problem. Overall the authors observed a significant
    increase in training speed, thus requiring less epochs to reach the desired
    performance. Finally batch norm seems to act as a regularizer. When we train
    the neural network we calculate the mean `),h(ue.$$.fragment),Ft=i(` and the standard
    deviation `),h(me.$$.fragment),At=i(` one batch at a time. This calculation is noisy
    and the neural network has to learn to tune out that noise in order to achieve
    a reasonable performance.`),rt=g(),F=b("p"),Ct=i(`During inference the procedure of calculating per batch statistics would
    cause problems, because different inference runs would generate different
    means and standard deviations and therefore generate different outputs. We
    want the neural network to be deterministic during inference. The same
    inputs should always lead to the same outputs. For that reason during
    training the batch norm layer calculates a moving average of `),h($e.$$.fragment),Wt=i(" and "),h(he.$$.fragment),Lt=i(" that can be used at inference time."),it=g(),xe=b("p"),qt=i(`Let also mention that no one really seems to know why batch norm works.
    Different hypotheses have been formulated over the years, but there seems to
    be no clear consensus on the matter. All you have to know is that batch
    normalization works well and is almost a requirement when training modern
    deep neural networks. This technique will become one of your main tools when
    designing modern neural network architectures.`),ot=g(),A=b("p"),Mt=i("PyToch has an explicit "),Ne=b("code"),Ht=i("BatchNorm1d"),Ut=i(` module that can be applied
    to a flattened tensor, like the flattened MNIST image. The 2d version will
    become important when we start dealing with 2d images. Below we create a
    small module that combines a linear mapping, batch normalization and a
    non-linear activation. Notice that we we provide the linear module with the
    argument `),Se=b("code"),Ot=i("bias=False"),Rt=i(" in order to deactivate the bias calculation."),ft=g(),h(ge.$$.fragment),ut=g(),Te=b("p"),Vt=i("We can reuse the above defined module several times."),mt=g(),h(ve.$$.fragment),$t=g(),C=b("p"),Gt=i(`As the batch normalization layer behaves differently during training and
    evalutation, don't forget to switch between `),De=b("code"),Jt=i("model.train()"),Kt=i(`
    and `),Pe=b("code"),Qt=i("model.eval()"),Xt=i("."),this.h()},l(e){a=y(e,"P",{});var l=z(a);t=o(l,`In a previous chapter we have discussed feature scaling. Feature scaling
    only applies to the input layer, so should we try to scale the intermediary
    features that are produced by hidden units? Would that be in any form
    benefitiary for trainig?`),l.forEach(n),s=v(e),c(k.$$.fragment,e),W=v(e),T=y(e,"P",{});var q=z(T);R=o(q,`Sergey Ioffe and Christian Szegedy answered the question with a definitive
    yes`),c(I.$$.fragment,q),V=o(q,". When we add so called "),c(N.$$.fragment,q),G=o(q,`
    to hidden features, we can speed up the training process significantly, while
    gaining other additional advantages.`),q.forEach(n),m=v(e),$=y(e,"P",{});var S=z($);L=o(S,"Consider a particular layer "),c(J.$$.fragment,S),ct=o(S,`, to which output we would like
    to apply batch normalization. Using a batch of data we calculate the mean `),c(K.$$.fragment,S),dt=o(S,` and the variance
    `),c(Q.$$.fragment,S),pt=o(S," for each hidden unit "),c(X.$$.fragment,S),wt=o(S," in the layer."),S.forEach(n),Ge=v(e),ce=y(e,"DIV",{class:!0});var Be=z(ce);c(Y.$$.fragment,Be),Be.forEach(n),Je=v(e),be=y(e,"P",{});var Fe=z(be);_t=o(Fe,`Given those parameters we can normalize the hidden features, using the same
    procedure we used for feature scaling.`),Fe.forEach(n),Ke=v(e),de=y(e,"DIV",{class:!0});var Ae=z(de);c(Z.$$.fragment,Ae),Ae.forEach(n),Qe=v(e),ye=y(e,"P",{});var Ce=z(ye);gt=o(Ce,`The authors argued that this normalization procedure might theoretically be
    detremental to the performance, because it might reduce the expressiveness
    of the neural network. To combat that they introduced an additional step
    that allowed the neural network to reverse the standardization.`),Ce.forEach(n),Xe=v(e),pe=y(e,"DIV",{class:!0});var We=z(pe);c(ee.$$.fragment,We),We.forEach(n),Ye=v(e),x=y(e,"P",{});var j=z(x);vt=o(j,"The feature specific parameters "),c(te.$$.fragment,j),bt=o(j," and "),c(ae.$$.fragment,j),yt=o(j," are learned by the neural network. If the network decides to set "),c(ne.$$.fragment,j),zt=o(j,` to
    `),c(le.$$.fragment,j),kt=o(j," and "),c(se.$$.fragment,j),Et=o(j," to "),c(re.$$.fragment,j),xt=o(j,` that
    essentially neutralizes the normalization. If normalization indeed worsens the
    performance, the neural network has the option to reverse the normalization step.`),j.forEach(n),Ze=v(e),ze=y(e,"P",{});var Le=z(ze);Tt=o(Le,`Our formulations above indicated that batch normalization is applied to
    activations. This procedure is similar to input feature scaling, because you
    normalize the data that is processed in the next layer.`),Le.forEach(n),et=v(e),c(we.$$.fragment,e),tt=v(e),ke=y(e,"P",{});var qe=z(ke);jt=o(qe,`In practice though batch norm is often applied to the net inputs and the
    result is forwarded to the activation function.`),qe.forEach(n),at=v(e),c(_e.$$.fragment,e),nt=v(e),Ee=y(e,"P",{});var Me=z(Ee);It=o(Me,`There is no real consensus about how you should apply batch normalization,
    but this decision in all likelihood should not make or break your project.`),Me.forEach(n),lt=v(e),D=y(e,"P",{});var P=z(D);Nt=o(P,"There is an additional caveat. In practice we often remove the bias term "),c(ie.$$.fragment,P),St=o(P," from the calculation of the net input "),c(oe.$$.fragment,P),Dt=o(P,". This is done due to the assumption, that "),c(fe.$$.fragment,P),Pt=o(P,` essentially
    does the same operation. Both are used to shift the calculation by a constant
    and there is hardly a reason to do that calculation twice.`),P.forEach(n),st=v(e),B=y(e,"P",{});var M=z(B);Bt=o(M,`The authors observed several adantages that batch normalization provides.
    For once batch norm makes the model less sensitive to the choice of the
    learning rate, which allows us to increase the learning rate and thereby
    increase the speed of convergence. Second, the model is more forgiving when
    choosing bad initial weights. Third, batch normalization seems to help with
    the vanishing gradients problem. Overall the authors observed a significant
    increase in training speed, thus requiring less epochs to reach the desired
    performance. Finally batch norm seems to act as a regularizer. When we train
    the neural network we calculate the mean `),c(ue.$$.fragment,M),Ft=o(M,` and the standard
    deviation `),c(me.$$.fragment,M),At=o(M,` one batch at a time. This calculation is noisy
    and the neural network has to learn to tune out that noise in order to achieve
    a reasonable performance.`),M.forEach(n),rt=v(e),F=y(e,"P",{});var H=z(F);Ct=o(H,`During inference the procedure of calculating per batch statistics would
    cause problems, because different inference runs would generate different
    means and standard deviations and therefore generate different outputs. We
    want the neural network to be deterministic during inference. The same
    inputs should always lead to the same outputs. For that reason during
    training the batch norm layer calculates a moving average of `),c($e.$$.fragment,H),Wt=o(H," and "),c(he.$$.fragment,H),Lt=o(H," that can be used at inference time."),H.forEach(n),it=v(e),xe=y(e,"P",{});var He=z(xe);qt=o(He,`Let also mention that no one really seems to know why batch norm works.
    Different hypotheses have been formulated over the years, but there seems to
    be no clear consensus on the matter. All you have to know is that batch
    normalization works well and is almost a requirement when training modern
    deep neural networks. This technique will become one of your main tools when
    designing modern neural network architectures.`),He.forEach(n),ot=v(e),A=y(e,"P",{});var U=z(A);Mt=o(U,"PyToch has an explicit "),Ne=y(U,"CODE",{});var Ue=z(Ne);Ht=o(Ue,"BatchNorm1d"),Ue.forEach(n),Ut=o(U,` module that can be applied
    to a flattened tensor, like the flattened MNIST image. The 2d version will
    become important when we start dealing with 2d images. Below we create a
    small module that combines a linear mapping, batch normalization and a
    non-linear activation. Notice that we we provide the linear module with the
    argument `),Se=y(U,"CODE",{});var Oe=z(Se);Ot=o(Oe,"bias=False"),Oe.forEach(n),Rt=o(U," in order to deactivate the bias calculation."),U.forEach(n),ft=v(e),c(ge.$$.fragment,e),ut=v(e),Te=y(e,"P",{});var Re=z(Te);Vt=o(Re,"We can reuse the above defined module several times."),Re.forEach(n),mt=v(e),c(ve.$$.fragment,e),$t=v(e),C=y(e,"P",{});var O=z(C);Gt=o(O,`As the batch normalization layer behaves differently during training and
    evalutation, don't forget to switch between `),De=y(O,"CODE",{});var Ve=z(De);Jt=o(Ve,"model.train()"),Ve.forEach(n),Kt=o(O,`
    and `),Pe=y(O,"CODE",{});var Zt=z(Pe);Qt=o(Zt,"model.eval()"),Zt.forEach(n),Xt=o(O,"."),O.forEach(n),this.h()},h(){je(ce,"class","flex justify-center"),je(de,"class","flex justify-center"),je(pe,"class","flex justify-center")},m(e,l){r(e,a,l),u(a,t),r(e,s,l),d(k,e,l),r(e,W,l),r(e,T,l),u(T,R),d(I,T,null),u(T,V),d(N,T,null),u(T,G),r(e,m,l),r(e,$,l),u($,L),d(J,$,null),u($,ct),d(K,$,null),u($,dt),d(Q,$,null),u($,pt),d(X,$,null),u($,wt),r(e,Ge,l),r(e,ce,l),d(Y,ce,null),r(e,Je,l),r(e,be,l),u(be,_t),r(e,Ke,l),r(e,de,l),d(Z,de,null),r(e,Qe,l),r(e,ye,l),u(ye,gt),r(e,Xe,l),r(e,pe,l),d(ee,pe,null),r(e,Ye,l),r(e,x,l),u(x,vt),d(te,x,null),u(x,bt),d(ae,x,null),u(x,yt),d(ne,x,null),u(x,zt),d(le,x,null),u(x,kt),d(se,x,null),u(x,Et),d(re,x,null),u(x,xt),r(e,Ze,l),r(e,ze,l),u(ze,Tt),r(e,et,l),d(we,e,l),r(e,tt,l),r(e,ke,l),u(ke,jt),r(e,at,l),d(_e,e,l),r(e,nt,l),r(e,Ee,l),u(Ee,It),r(e,lt,l),r(e,D,l),u(D,Nt),d(ie,D,null),u(D,St),d(oe,D,null),u(D,Dt),d(fe,D,null),u(D,Pt),r(e,st,l),r(e,B,l),u(B,Bt),d(ue,B,null),u(B,Ft),d(me,B,null),u(B,At),r(e,rt,l),r(e,F,l),u(F,Ct),d($e,F,null),u(F,Wt),d(he,F,null),u(F,Lt),r(e,it,l),r(e,xe,l),u(xe,qt),r(e,ot,l),r(e,A,l),u(A,Mt),u(A,Ne),u(Ne,Ht),u(A,Ut),u(A,Se),u(Se,Ot),u(A,Rt),r(e,ft,l),d(ge,e,l),r(e,ut,l),r(e,Te,l),u(Te,Vt),r(e,mt,l),d(ve,e,l),r(e,$t,l),r(e,C,l),u(C,Gt),u(C,De),u(De,Jt),u(C,Kt),u(C,Pe),u(Pe,Qt),u(C,Xt),ht=!0},p(e,l){const q={};l&16&&(q.$$scope={dirty:l,ctx:e}),N.$set(q);const S={};l&16&&(S.$$scope={dirty:l,ctx:e}),J.$set(S);const Be={};l&16&&(Be.$$scope={dirty:l,ctx:e}),K.$set(Be);const Fe={};l&16&&(Fe.$$scope={dirty:l,ctx:e}),Q.$set(Fe);const Ae={};l&16&&(Ae.$$scope={dirty:l,ctx:e}),X.$set(Ae);const Ce={};l&16&&(Ce.$$scope={dirty:l,ctx:e}),Y.$set(Ce);const We={};l&16&&(We.$$scope={dirty:l,ctx:e}),Z.$set(We);const j={};l&16&&(j.$$scope={dirty:l,ctx:e}),ee.$set(j);const Le={};l&16&&(Le.$$scope={dirty:l,ctx:e}),te.$set(Le);const qe={};l&16&&(qe.$$scope={dirty:l,ctx:e}),ae.$set(qe);const Me={};l&16&&(Me.$$scope={dirty:l,ctx:e}),ne.$set(Me);const P={};l&16&&(P.$$scope={dirty:l,ctx:e}),le.$set(P);const M={};l&16&&(M.$$scope={dirty:l,ctx:e}),se.$set(M);const H={};l&16&&(H.$$scope={dirty:l,ctx:e}),re.$set(H);const He={};l&16&&(He.$$scope={dirty:l,ctx:e}),ie.$set(He);const U={};l&16&&(U.$$scope={dirty:l,ctx:e}),oe.$set(U);const Ue={};l&16&&(Ue.$$scope={dirty:l,ctx:e}),fe.$set(Ue);const Oe={};l&16&&(Oe.$$scope={dirty:l,ctx:e}),ue.$set(Oe);const Re={};l&16&&(Re.$$scope={dirty:l,ctx:e}),me.$set(Re);const O={};l&16&&(O.$$scope={dirty:l,ctx:e}),$e.$set(O);const Ve={};l&16&&(Ve.$$scope={dirty:l,ctx:e}),he.$set(Ve)},i(e){ht||(p(k.$$.fragment,e),p(I.$$.fragment,e),p(N.$$.fragment,e),p(J.$$.fragment,e),p(K.$$.fragment,e),p(Q.$$.fragment,e),p(X.$$.fragment,e),p(Y.$$.fragment,e),p(Z.$$.fragment,e),p(ee.$$.fragment,e),p(te.$$.fragment,e),p(ae.$$.fragment,e),p(ne.$$.fragment,e),p(le.$$.fragment,e),p(se.$$.fragment,e),p(re.$$.fragment,e),p(we.$$.fragment,e),p(_e.$$.fragment,e),p(ie.$$.fragment,e),p(oe.$$.fragment,e),p(fe.$$.fragment,e),p(ue.$$.fragment,e),p(me.$$.fragment,e),p($e.$$.fragment,e),p(he.$$.fragment,e),p(ge.$$.fragment,e),p(ve.$$.fragment,e),ht=!0)},o(e){w(k.$$.fragment,e),w(I.$$.fragment,e),w(N.$$.fragment,e),w(J.$$.fragment,e),w(K.$$.fragment,e),w(Q.$$.fragment,e),w(X.$$.fragment,e),w(Y.$$.fragment,e),w(Z.$$.fragment,e),w(ee.$$.fragment,e),w(te.$$.fragment,e),w(ae.$$.fragment,e),w(ne.$$.fragment,e),w(le.$$.fragment,e),w(se.$$.fragment,e),w(re.$$.fragment,e),w(we.$$.fragment,e),w(_e.$$.fragment,e),w(ie.$$.fragment,e),w(oe.$$.fragment,e),w(fe.$$.fragment,e),w(ue.$$.fragment,e),w(me.$$.fragment,e),w($e.$$.fragment,e),w(he.$$.fragment,e),w(ge.$$.fragment,e),w(ve.$$.fragment,e),ht=!1},d(e){e&&n(a),e&&n(s),_(k,e),e&&n(W),e&&n(T),_(I),_(N),e&&n(m),e&&n($),_(J),_(K),_(Q),_(X),e&&n(Ge),e&&n(ce),_(Y),e&&n(Je),e&&n(be),e&&n(Ke),e&&n(de),_(Z),e&&n(Qe),e&&n(ye),e&&n(Xe),e&&n(pe),_(ee),e&&n(Ye),e&&n(x),_(te),_(ae),_(ne),_(le),_(se),_(re),e&&n(Ze),e&&n(ze),e&&n(et),_(we,e),e&&n(tt),e&&n(ke),e&&n(at),_(_e,e),e&&n(nt),e&&n(Ee),e&&n(lt),e&&n(D),_(ie),_(oe),_(fe),e&&n(st),e&&n(B),_(ue),_(me),e&&n(rt),e&&n(F),_($e),_(he),e&&n(it),e&&n(xe),e&&n(ot),e&&n(A),e&&n(ft),_(ge,e),e&&n(ut),e&&n(Te),e&&n(mt),_(ve,e),e&&n($t),e&&n(C)}}}function Sa(f){let a,t,s,k,W,T,R,I,V,N,G;return I=new sa({props:{$$slots:{default:[Na]},$$scope:{ctx:f}}}),N=new ia({props:{references:f[0]}}),{c(){a=b("meta"),t=g(),s=b("h1"),k=i("Batch Normalization"),W=g(),T=b("div"),R=g(),h(I.$$.fragment),V=g(),h(N.$$.fragment),this.h()},l(m){const $=la("svelte-1o2shcx",document.head);a=y($,"META",{name:!0,content:!0}),$.forEach(n),t=v(m),s=y(m,"H1",{});var L=z(s);k=o(L,"Batch Normalization"),L.forEach(n),W=v(m),T=y(m,"DIV",{class:!0}),z(T).forEach(n),R=v(m),c(I.$$.fragment,m),V=v(m),c(N.$$.fragment,m),this.h()},h(){document.title="World4AI | Deep Learning | Batch Normalization",je(a,"name","description"),je(a,"content","Similar to feature scaling, batch normalization normalizes hidden layers, thereby speeding up the training process, reducing overfitting and decreasin the chances of vanishing gradients."),je(T,"class","separator")},m(m,$){u(document.head,a),r(m,t,$),r(m,s,$),u(s,k),r(m,W,$),r(m,T,$),r(m,R,$),d(I,m,$),r(m,V,$),d(N,m,$),G=!0},p(m,[$]){const L={};$&16&&(L.$$scope={dirty:$,ctx:m}),I.$set(L)},i(m){G||(p(I.$$.fragment,m),p(N.$$.fragment,m),G=!0)},o(m){w(I.$$.fragment,m),w(N.$$.fragment,m),G=!1},d(m){n(a),m&&n(t),m&&n(s),m&&n(W),m&&n(T),m&&n(R),_(I,m),m&&n(V),_(N,m)}}}function Da(f){return[[{author:"Sergey Ioffe, Christian Szegedy",title:"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",journal:"",year:"2005",pages:"",volume:"",issue:""}],[{title:"Input",nodes:[{value:"x_1",class:"fill-white"},{value:"x_2",class:"fill-white"},{value:"x_3",class:"fill-white"}]},{title:"Hidden Layer 1",nodes:[{value:"a_1",class:"fill-w4ai-yellow"},{value:"a_2",class:"fill-w4ai-yellow"},{value:"a_3",class:"fill-w4ai-yellow"}]},{title:"Hidden Layer 2",nodes:[{value:"a_1",class:"fill-w4ai-yellow"},{value:"a_2",class:"fill-w4ai-yellow"},{value:"a_3",class:"fill-w4ai-yellow"}]},{title:"Out",nodes:[{value:"a_1",class:"fill-white"},{value:"a_2",class:"fill-white"},{value:"a_3",class:"fill-white"}]}],[{title:"Net Input",nodes:[{value:"\\mathbf{z}",class:"fill-w4ai-yellow"}]},{title:"Activations",nodes:[{value:"\\mathbf{a}",class:"fill-w4ai-yellow"}]},{title:"Batch Norm",nodes:[{value:"\\mathbf{\\bar{a}}",class:"fill-w4ai-yellow"}]}],[{title:"Net Input",nodes:[{value:"\\mathbf{z}",class:"fill-w4ai-yellow"}]},{title:"Batch Norm",nodes:[{value:"\\mathbf{\\bar{z}}",class:"fill-w4ai-yellow"}]},{title:"Activations",nodes:[{value:"\\mathbf{a}",class:"fill-w4ai-yellow"}]}]]}class qa extends ta{constructor(a){super(),aa(this,a,Da,Sa,na,{})}}export{qa as default};
