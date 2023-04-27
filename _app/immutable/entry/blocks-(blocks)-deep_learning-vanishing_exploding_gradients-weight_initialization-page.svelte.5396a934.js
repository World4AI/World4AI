import{S as Qt,i as en,s as tn,k as y,a as p,q as s,y as d,W as nn,l as b,h as i,c as $,m as E,r as l,z as w,n as me,N as m,b as r,A as c,g,d as _,B as v,L as an,C}from"../chunks/index.4d92b023.js";import{C as on}from"../chunks/Container.b0705c7b.js";import{A as jt}from"../chunks/Alert.25a852b3.js";import{N as Jt}from"../chunks/NeuralNetwork.9b1e2957.js";import{L as S}from"../chunks/Latex.e0b308c0.js";import{F as rn,I as Zt}from"../chunks/InternalLink.7deb899c.js";import{B as sn}from"../chunks/BackpropGraph.6a7a3666.js";import{P as Ee}from"../chunks/PythonCode.212ba7a6.js";import{V as fe}from"../chunks/Network.03de8e4c.js";const ln=""+new URL("../assets/xavier_metrics.ec7b538b.webp",import.meta.url).href;function fn(u){let o;return{c(){o=s("w")},l(n){o=l(n,"w")},m(n,a){r(n,o,a)},d(n){n&&i(o)}}}function mn(u){let o=String.raw`
  \begin{aligned}
    a_1 &= x_1 * w_1 + x_2 * w_2 \\
    a_2 &= x_1 * w_3 + x_2 * w_4 \\
    a_1 &= a_2 
  \end{aligned}
    `+"",n;return{c(){n=s(o)},l(a){n=l(a,o)},m(a,h){r(a,n,h)},p:C,d(a){a&&i(n)}}}function un(u){let o=String.raw`
    \dfrac{\partial o}{\partial a_1}
    \dfrac{\partial a_1}{\partial w_1}
    =
    \dfrac{\partial o}{\partial a_2}
    \dfrac{\partial a_2}{\partial w_3}
    `+"",n;return{c(){n=s(o)},l(a){n=l(a,o)},m(a,h){r(a,n,h)},p:C,d(a){a&&i(n)}}}function hn(u){let o;return{c(){o=s("Never initialize your weights uniformly. Break the symmetry!")},l(n){o=l(n,"Never initialize your weights uniformly. Break the symmetry!")},m(n,a){r(n,o,a)},d(n){n&&i(o)}}}function pn(u){let o;return{c(){o=s("\\mu = 0")},l(n){o=l(n,"\\mu = 0")},m(n,a){r(n,o,a)},d(n){n&&i(o)}}}function $n(u){let o;return{c(){o=s("\\sigma = 0.1")},l(n){o=l(n,"\\sigma = 0.1")},m(n,a){r(n,o,a)},d(n){n&&i(o)}}}function dn(u){let o=String.raw`-0.5 \text{ to } 0.5`+"",n;return{c(){n=s(o)},l(a){n=l(a,o)},m(a,h){r(a,n,h)},p:C,d(a){a&&i(n)}}}function wn(u){let o=String.raw`\mathcal{U}(-a, a)`+"",n;return{c(){n=s(o)},l(a){n=l(a,o)},m(a,h){r(a,n,h)},p:C,d(a){a&&i(n)}}}function cn(u){let o=String.raw`a = \sqrt{\dfrac{6}{fan_{in} + fan_{out}}}`+"",n;return{c(){n=s(o)},l(a){n=l(a,o)},m(a,h){r(a,n,h)},p:C,d(a){a&&i(n)}}}function gn(u){let o=String.raw`\mathcal{N}(0, \sigma^2)`+"",n;return{c(){n=s(o)},l(a){n=l(a,o)},m(a,h){r(a,n,h)},p:C,d(a){a&&i(n)}}}function _n(u){let o=String.raw`\sigma = \sqrt{\dfrac{2}{fan_{in} + fan_{out}}}`+"",n;return{c(){n=s(o)},l(a){n=l(a,o)},m(a,h){r(a,n,h)},p:C,d(a){a&&i(n)}}}function vn(u){let o=String.raw`fan_{in}`+"",n;return{c(){n=s(o)},l(a){n=l(a,o)},m(a,h){r(a,n,h)},p:C,d(a){a&&i(n)}}}function yn(u){let o=String.raw`fan_{out}`+"",n;return{c(){n=s(o)},l(a){n=l(a,o)},m(a,h){r(a,n,h)},p:C,d(a){a&&i(n)}}}function bn(u){let o=String.raw`fan_{in}`+"",n;return{c(){n=s(o)},l(a){n=l(a,o)},m(a,h){r(a,n,h)},p:C,d(a){a&&i(n)}}}function zn(u){let o=String.raw`fan_{out}`+"",n;return{c(){n=s(o)},l(a){n=l(a,o)},m(a,h){r(a,n,h)},p:C,d(a){a&&i(n)}}}function kn(u){let o;return{c(){o=s(`For the most part you will not spend a lot of time dealing with weight
    initializations. Libraries like PyTorch and Keras have good common sense
    initialization values and allow you to switch between the initialization
    modes relatively easy. You do not nead to memorize those formulas. If you
    implement backpropagation on your own don't forget to at least break the
    symmetry.`)},l(n){o=l(n,`For the most part you will not spend a lot of time dealing with weight
    initializations. Libraries like PyTorch and Keras have good common sense
    initialization values and allow you to switch between the initialization
    modes relatively easy. You do not nead to memorize those formulas. If you
    implement backpropagation on your own don't forget to at least break the
    symmetry.`)},m(n,a){r(n,o,a)},d(n){n&&i(o)}}}function Tn(u){let o,n,a,h,q,I,P,L,N,x,H,f,k,A,B,Ne,F,xe,ue,K,Ie,U,gt,Oe,V,Ye,Pe,_t,je,X,Je,Le,vt,Ze,M,Qe,De,yt,et,z,bt,O,zt,Y,kt,j,Tt,he,Et,J,xt,Z,It,Q,Pt,ee,Lt,tt,D,Dt,te,St,ne,Wt,ie,Nt,ae,Ht,nt,pe,it,oe,At,$e,Ct,at,re,ot,G,qt,He,Bt,Ft,se,Kt,Ut,rt,de,st,we,lt,le,Gt,Ae,Rt,Vt,ft,ce,mt,ge,ut,Se,Xt,ht,_e,pt,ve,$t,ye,dt,be,Ot,wt,We,Mt,ct;return x=new Jt({props:{layers:u[1],padding:{left:0,right:10}}}),F=new S({props:{$$slots:{default:[fn]},$$scope:{ctx:u}}}),K=new S({props:{$$slots:{default:[mn]},$$scope:{ctx:u}}}),V=new S({props:{$$slots:{default:[un]},$$scope:{ctx:u}}}),X=new jt({props:{type:"danger",$$slots:{default:[hn]},$$scope:{ctx:u}}}),M=new sn({props:{graph:u[0],maxWidth:900,width:1180,height:920}}),O=new S({props:{$$slots:{default:[pn]},$$scope:{ctx:u}}}),Y=new S({props:{$$slots:{default:[$n]},$$scope:{ctx:u}}}),j=new S({props:{$$slots:{default:[dn]},$$scope:{ctx:u}}}),he=new Zt({props:{id:1,type:"reference"}}),J=new S({props:{$$slots:{default:[wn]},$$scope:{ctx:u}}}),Z=new S({props:{$$slots:{default:[cn]},$$scope:{ctx:u}}}),Q=new S({props:{$$slots:{default:[gn]},$$scope:{ctx:u}}}),ee=new S({props:{$$slots:{default:[_n]},$$scope:{ctx:u}}}),te=new S({props:{$$slots:{default:[vn]},$$scope:{ctx:u}}}),ne=new S({props:{$$slots:{default:[yn]},$$scope:{ctx:u}}}),ie=new S({props:{$$slots:{default:[bn]},$$scope:{ctx:u}}}),ae=new S({props:{$$slots:{default:[zn]},$$scope:{ctx:u}}}),pe=new Jt({props:{layers:u[2],padding:{left:0,right:20},height:100}}),$e=new Zt({props:{id:"2",type:"reference"}}),re=new jt({props:{type:"info",$$slots:{default:[kn]},$$scope:{ctx:u}}}),de=new Ee({props:{code:`W = torch.empty(5, 5)
# initializations are defined in nn.init
nn.init.kaiming_uniform_(W)`}}),we=new Ee({props:{code:`tensor([[-0.9263, -0.0416,  0.0063, -0.8040,  0.8433],
        [ 0.3724, -0.9250, -0.2109, -0.1961,  0.3596],
        [ 0.6127,  0.2282,  0.1292,  0.8036,  0.8993],
        [-0.3890,  0.8515,  0.2224,  0.6172,  0.0440],
        [ 1.0282, -0.7566, -0.0305, -0.4382, -0.0368]])
`,isOutput:!0}}),ce=new Ee({props:{code:`class SigmoidModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(NUM_FEATURES, HIDDEN),
            nn.Sigmoid(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.Sigmoid(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.Sigmoid(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.Sigmoid(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.Sigmoid(),
            nn.Linear(HIDDEN, NUM_LABELS)
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        with torch.inference_mode():
            for param in self.parameters():
                if param.ndim > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    param.zero_()
            
    def forward(self, features):
        return self.layers(features)`}}),ge=new Ee({props:{code:`model = SigmoidModel().to(DEVICE)
criterion = nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.SGD(model.parameters(), lr=ALPHA)`}}),_e=new Ee({props:{code:"history = train(train_dataloader, val_dataloader, model, criterion, optimizer)"}}),ve=new Ee({props:{code:`Epoch: 1/30 | Train Loss: 2.308537656906164 | Val Loss: 2.308519915898641
Epoch: 10/30 | Train Loss: 0.20434634447115443 | Val Loss: 0.2530187802116076
Epoch: 20/30 | Train Loss: 0.07002673296947375 | Val Loss: 0.15957480862239998
Epoch: 30/30 | Train Loss: 0.04259131510365781 | Val Loss: 0.1637446080291023
`,isOutput:!0}}),ye=new Ee({props:{code:'plot_history(history, "xavier_metrics")'}}),{c(){o=y("p"),n=s(`Previously we had mentioned that weights can contribute to vanishing and
    exploding gradients. For the most part we adjust weights in a completely
    automated process by using backpropagation and applying gradient descent.
    For that reason we do not have a lot of influence on how the weights
    develop. The one place where we directly determine the distribution of
    weights is during the initialization process. This section is going to be
    dedicated to weight initialization: the pitfalls and best practices.`),a=p(),h=y("p"),q=s(`The first idea we might come up with is to initialize all weights equally,
    specifically to use 0 as the starting value for all weights and biases.`),I=p(),P=y("p"),L=s(`We will use this simple neural network to demonstrate the danger of such
    initialization. All we need to do is to work through a single forward and
    backward pass to realize the problem.`),N=p(),d(x.$$.fragment),H=p(),f=y("p"),k=s(`We will make the simplifying assumption, that there are no activation
    functions in the neural network and that we want to minimize the value of
    the output neuron and not some loss function. These assumptions do not have
    any effect on the results, but simplify notation and the depiction of the
    computational graph.`),A=p(),B=y("p"),Ne=s("If we have the same weight "),d(F.$$.fragment),xe=s(` for all nodes and layers, then in
    the very first forward pass all the neurons from the same layer will produce
    the same value.`),ue=p(),d(K.$$.fragment),Ie=p(),U=y("p"),gt=s(`When we apply backpropagation we will quickly notice that the gradients with
    respect to the weights of the same feature are identical in each node.`),Oe=p(),d(V.$$.fragment),Ye=p(),Pe=y("p"),_t=s(`The same starting values and the same gradients can only mean that all nodes
    in a layer will always have the same value. This is no different than having
    a neural network with a single neuron per layer. The network will never be
    able to solve complex problems. And if you initialize all your weights with
    zero, the network will always have dead neurons, always staying at the 0
    value.`),je=p(),d(X.$$.fragment),Je=p(),Le=y("p"),vt=s(`Now let's use the same neural network and actually work though a dummy
    example. We assume feature values of 5 and 2 respectively and initialize all
    weights to 1.`),Ze=p(),d(M.$$.fragment),Qe=p(),De=y("p"),yt=s(`Essentially you can observe two paths (left path and right path) in the
    computational graph above, representing the two neurons. But the paths are
    identical in their values and in their gradients. Even though there are 6
    weights in the neural network, half of them are basically clones.`),et=p(),z=y("p"),bt=s(`In order to break the symmetry researchers used to apply either a normal
    distribution (e.g. `),d(O.$$.fragment),zt=s(" and "),d(Y.$$.fragment),kt=s(`)
    or a uniform distribution (e.g in the range `),d(j.$$.fragment),Tt=s(") to initialize weights. This might seem reasonable, but Glorot and Bengio"),d(he.$$.fragment),Et=s(` showed that it is much more preferable to initialize weights based on the
    number of neurons that are used as input into the layer and the number of neurons
    that are inside a layer. This initializiation technique makes sure, that during
    the forward pass the variance of neurons stays similar from layer to layer and
    during the backward pass the gradients keep a constant variance from layer to
    layer. That condition reduces the likelihood of vanishing or exploding gradients.
    The authors proposed to initialize weights either using a uniform distribution
    `),d(J.$$.fragment),xt=s(" where "),d(Z.$$.fragment),It=s(" or the normal distribution "),d(Q.$$.fragment),Pt=s(`, where
    `),d(ee.$$.fragment),Lt=s("."),tt=p(),D=y("p"),Dt=s("The words "),d(te.$$.fragment),St=s(" and "),d(ne.$$.fragment),Wt=s(` stand for the number of neurons that go into the layer as input and the number
    of neurons that are in the layer respectively. In the below example in the first
    hidden layer `),d(ie.$$.fragment),Nt=s(" would be 2 and "),d(ae.$$.fragment),Ht=s(` would be 3 respectively. In the second hidden layer the numbers would be exactly
    the other way around.`),nt=p(),d(pe.$$.fragment),it=p(),oe=y("p"),At=s(`While the Xavier/Glorot initialization was studied in conjunction with the
    sigmoind and the tanh activation function, the Kaiming/He initialization was
    designed to work with the ReLU activation`),d($e.$$.fragment),Ct=s(". This is the standard initialization mode used in PyTorch."),at=p(),d(re.$$.fragment),ot=p(),G=y("p"),qt=s(`Implementing weight initialization in PyTorch is a piece of cake. PyTorch
    provides in `),He=y("code"),Bt=s("nn.init"),Ft=s(` different functions that can be used to
    initialize a tensor. You should have a look at the official
    `),se=y("a"),Kt=s("PyTorch documentation"),Ut=s(` if you would like to explore more initialization schemes. Below for example
    we use the Kaiming uniform initialization on an empty tensor. Notice that the
    initialization is done inplace.`),rt=p(),d(de.$$.fragment),st=p(),d(we.$$.fragment),lt=p(),le=y("p"),Gt=s("When we use the "),Ae=y("code"),Rt=s("nn.Linear"),Vt=s(` module, PyTorch automatically initializes
    weights and biases using the Kaiming He uniform initialization scheme. The sigmoid
    model from the last section was suffering from vanishing gradients, but we might
    remedy the problem, by changing the weight initialization. The Kaiming|He initialization
    was developed for the ReLU activation function, while the Glorot|Xavier initialization
    should be used with sigmoid activation functions. We once again create the same
    model that uses sigmoid activation functions. Only this time we loop over weights
    and biases and use the Xavier uniform initialization for weights and we set all
    biases to 0 at initialization.`),ft=p(),d(ce.$$.fragment),mt=p(),d(ge.$$.fragment),ut=p(),Se=y("p"),Xt=s("This time around the model performs much better."),ht=p(),d(_e.$$.fragment),pt=p(),d(ve.$$.fragment),$t=p(),d(ye.$$.fragment),dt=p(),be=y("img"),wt=p(),We=y("p"),Mt=s(`You should still use the ReLU activation function for deep neural networks,
    but be aware that weight initialization might have a significant impact on
    the performance of your model.`),this.h()},l(e){o=b(e,"P",{});var t=E(o);n=l(t,`Previously we had mentioned that weights can contribute to vanishing and
    exploding gradients. For the most part we adjust weights in a completely
    automated process by using backpropagation and applying gradient descent.
    For that reason we do not have a lot of influence on how the weights
    develop. The one place where we directly determine the distribution of
    weights is during the initialization process. This section is going to be
    dedicated to weight initialization: the pitfalls and best practices.`),t.forEach(i),a=$(e),h=b(e,"P",{});var Ce=E(h);q=l(Ce,`The first idea we might come up with is to initialize all weights equally,
    specifically to use 0 as the starting value for all weights and biases.`),Ce.forEach(i),I=$(e),P=b(e,"P",{});var qe=E(P);L=l(qe,`We will use this simple neural network to demonstrate the danger of such
    initialization. All we need to do is to work through a single forward and
    backward pass to realize the problem.`),qe.forEach(i),N=$(e),w(x.$$.fragment,e),H=$(e),f=b(e,"P",{});var Be=E(f);k=l(Be,`We will make the simplifying assumption, that there are no activation
    functions in the neural network and that we want to minimize the value of
    the output neuron and not some loss function. These assumptions do not have
    any effect on the results, but simplify notation and the depiction of the
    computational graph.`),Be.forEach(i),A=$(e),B=b(e,"P",{});var ze=E(B);Ne=l(ze,"If we have the same weight "),w(F.$$.fragment,ze),xe=l(ze,` for all nodes and layers, then in
    the very first forward pass all the neurons from the same layer will produce
    the same value.`),ze.forEach(i),ue=$(e),w(K.$$.fragment,e),Ie=$(e),U=b(e,"P",{});var Fe=E(U);gt=l(Fe,`When we apply backpropagation we will quickly notice that the gradients with
    respect to the weights of the same feature are identical in each node.`),Fe.forEach(i),Oe=$(e),w(V.$$.fragment,e),Ye=$(e),Pe=b(e,"P",{});var Ke=E(Pe);_t=l(Ke,`The same starting values and the same gradients can only mean that all nodes
    in a layer will always have the same value. This is no different than having
    a neural network with a single neuron per layer. The network will never be
    able to solve complex problems. And if you initialize all your weights with
    zero, the network will always have dead neurons, always staying at the 0
    value.`),Ke.forEach(i),je=$(e),w(X.$$.fragment,e),Je=$(e),Le=b(e,"P",{});var Ue=E(Le);vt=l(Ue,`Now let's use the same neural network and actually work though a dummy
    example. We assume feature values of 5 and 2 respectively and initialize all
    weights to 1.`),Ue.forEach(i),Ze=$(e),w(M.$$.fragment,e),Qe=$(e),De=b(e,"P",{});var Ge=E(De);yt=l(Ge,`Essentially you can observe two paths (left path and right path) in the
    computational graph above, representing the two neurons. But the paths are
    identical in their values and in their gradients. Even though there are 6
    weights in the neural network, half of them are basically clones.`),Ge.forEach(i),et=$(e),z=b(e,"P",{});var T=E(z);bt=l(T,`In order to break the symmetry researchers used to apply either a normal
    distribution (e.g. `),w(O.$$.fragment,T),zt=l(T," and "),w(Y.$$.fragment,T),kt=l(T,`)
    or a uniform distribution (e.g in the range `),w(j.$$.fragment,T),Tt=l(T,") to initialize weights. This might seem reasonable, but Glorot and Bengio"),w(he.$$.fragment,T),Et=l(T,` showed that it is much more preferable to initialize weights based on the
    number of neurons that are used as input into the layer and the number of neurons
    that are inside a layer. This initializiation technique makes sure, that during
    the forward pass the variance of neurons stays similar from layer to layer and
    during the backward pass the gradients keep a constant variance from layer to
    layer. That condition reduces the likelihood of vanishing or exploding gradients.
    The authors proposed to initialize weights either using a uniform distribution
    `),w(J.$$.fragment,T),xt=l(T," where "),w(Z.$$.fragment,T),It=l(T," or the normal distribution "),w(Q.$$.fragment,T),Pt=l(T,`, where
    `),w(ee.$$.fragment,T),Lt=l(T,"."),T.forEach(i),tt=$(e),D=b(e,"P",{});var W=E(D);Dt=l(W,"The words "),w(te.$$.fragment,W),St=l(W," and "),w(ne.$$.fragment,W),Wt=l(W,` stand for the number of neurons that go into the layer as input and the number
    of neurons that are in the layer respectively. In the below example in the first
    hidden layer `),w(ie.$$.fragment,W),Nt=l(W," would be 2 and "),w(ae.$$.fragment,W),Ht=l(W,` would be 3 respectively. In the second hidden layer the numbers would be exactly
    the other way around.`),W.forEach(i),nt=$(e),w(pe.$$.fragment,e),it=$(e),oe=b(e,"P",{});var ke=E(oe);At=l(ke,`While the Xavier/Glorot initialization was studied in conjunction with the
    sigmoind and the tanh activation function, the Kaiming/He initialization was
    designed to work with the ReLU activation`),w($e.$$.fragment,ke),Ct=l(ke,". This is the standard initialization mode used in PyTorch."),ke.forEach(i),at=$(e),w(re.$$.fragment,e),ot=$(e),G=b(e,"P",{});var R=E(G);qt=l(R,`Implementing weight initialization in PyTorch is a piece of cake. PyTorch
    provides in `),He=b(R,"CODE",{});var Re=E(He);Bt=l(Re,"nn.init"),Re.forEach(i),Ft=l(R,` different functions that can be used to
    initialize a tensor. You should have a look at the official
    `),se=b(R,"A",{href:!0,target:!0,rel:!0});var Ve=E(se);Kt=l(Ve,"PyTorch documentation"),Ve.forEach(i),Ut=l(R,` if you would like to explore more initialization schemes. Below for example
    we use the Kaiming uniform initialization on an empty tensor. Notice that the
    initialization is done inplace.`),R.forEach(i),rt=$(e),w(de.$$.fragment,e),st=$(e),w(we.$$.fragment,e),lt=$(e),le=b(e,"P",{});var Te=E(le);Gt=l(Te,"When we use the "),Ae=b(Te,"CODE",{});var Xe=E(Ae);Rt=l(Xe,"nn.Linear"),Xe.forEach(i),Vt=l(Te,` module, PyTorch automatically initializes
    weights and biases using the Kaiming He uniform initialization scheme. The sigmoid
    model from the last section was suffering from vanishing gradients, but we might
    remedy the problem, by changing the weight initialization. The Kaiming|He initialization
    was developed for the ReLU activation function, while the Glorot|Xavier initialization
    should be used with sigmoid activation functions. We once again create the same
    model that uses sigmoid activation functions. Only this time we loop over weights
    and biases and use the Xavier uniform initialization for weights and we set all
    biases to 0 at initialization.`),Te.forEach(i),ft=$(e),w(ce.$$.fragment,e),mt=$(e),w(ge.$$.fragment,e),ut=$(e),Se=b(e,"P",{});var Me=E(Se);Xt=l(Me,"This time around the model performs much better."),Me.forEach(i),ht=$(e),w(_e.$$.fragment,e),pt=$(e),w(ve.$$.fragment,e),$t=$(e),w(ye.$$.fragment,e),dt=$(e),be=b(e,"IMG",{src:!0,alt:!0}),wt=$(e),We=b(e,"P",{});var Yt=E(We);Mt=l(Yt,`You should still use the ReLU activation function for deep neural networks,
    but be aware that weight initialization might have a significant impact on
    the performance of your model.`),Yt.forEach(i),this.h()},h(){me(se,"href","https://pytorch.org/docs/stable/nn.init.html"),me(se,"target","_blank"),me(se,"rel","noreferrer"),an(be.src,Ot=ln)||me(be,"src",Ot),me(be,"alt","Performance metrics with Xavier initialization")},m(e,t){r(e,o,t),m(o,n),r(e,a,t),r(e,h,t),m(h,q),r(e,I,t),r(e,P,t),m(P,L),r(e,N,t),c(x,e,t),r(e,H,t),r(e,f,t),m(f,k),r(e,A,t),r(e,B,t),m(B,Ne),c(F,B,null),m(B,xe),r(e,ue,t),c(K,e,t),r(e,Ie,t),r(e,U,t),m(U,gt),r(e,Oe,t),c(V,e,t),r(e,Ye,t),r(e,Pe,t),m(Pe,_t),r(e,je,t),c(X,e,t),r(e,Je,t),r(e,Le,t),m(Le,vt),r(e,Ze,t),c(M,e,t),r(e,Qe,t),r(e,De,t),m(De,yt),r(e,et,t),r(e,z,t),m(z,bt),c(O,z,null),m(z,zt),c(Y,z,null),m(z,kt),c(j,z,null),m(z,Tt),c(he,z,null),m(z,Et),c(J,z,null),m(z,xt),c(Z,z,null),m(z,It),c(Q,z,null),m(z,Pt),c(ee,z,null),m(z,Lt),r(e,tt,t),r(e,D,t),m(D,Dt),c(te,D,null),m(D,St),c(ne,D,null),m(D,Wt),c(ie,D,null),m(D,Nt),c(ae,D,null),m(D,Ht),r(e,nt,t),c(pe,e,t),r(e,it,t),r(e,oe,t),m(oe,At),c($e,oe,null),m(oe,Ct),r(e,at,t),c(re,e,t),r(e,ot,t),r(e,G,t),m(G,qt),m(G,He),m(He,Bt),m(G,Ft),m(G,se),m(se,Kt),m(G,Ut),r(e,rt,t),c(de,e,t),r(e,st,t),c(we,e,t),r(e,lt,t),r(e,le,t),m(le,Gt),m(le,Ae),m(Ae,Rt),m(le,Vt),r(e,ft,t),c(ce,e,t),r(e,mt,t),c(ge,e,t),r(e,ut,t),r(e,Se,t),m(Se,Xt),r(e,ht,t),c(_e,e,t),r(e,pt,t),c(ve,e,t),r(e,$t,t),c(ye,e,t),r(e,dt,t),r(e,be,t),r(e,wt,t),r(e,We,t),m(We,Mt),ct=!0},p(e,t){const Ce={};t&1048576&&(Ce.$$scope={dirty:t,ctx:e}),F.$set(Ce);const qe={};t&1048576&&(qe.$$scope={dirty:t,ctx:e}),K.$set(qe);const Be={};t&1048576&&(Be.$$scope={dirty:t,ctx:e}),V.$set(Be);const ze={};t&1048576&&(ze.$$scope={dirty:t,ctx:e}),X.$set(ze);const Fe={};t&1&&(Fe.graph=e[0]),M.$set(Fe);const Ke={};t&1048576&&(Ke.$$scope={dirty:t,ctx:e}),O.$set(Ke);const Ue={};t&1048576&&(Ue.$$scope={dirty:t,ctx:e}),Y.$set(Ue);const Ge={};t&1048576&&(Ge.$$scope={dirty:t,ctx:e}),j.$set(Ge);const T={};t&1048576&&(T.$$scope={dirty:t,ctx:e}),J.$set(T);const W={};t&1048576&&(W.$$scope={dirty:t,ctx:e}),Z.$set(W);const ke={};t&1048576&&(ke.$$scope={dirty:t,ctx:e}),Q.$set(ke);const R={};t&1048576&&(R.$$scope={dirty:t,ctx:e}),ee.$set(R);const Re={};t&1048576&&(Re.$$scope={dirty:t,ctx:e}),te.$set(Re);const Ve={};t&1048576&&(Ve.$$scope={dirty:t,ctx:e}),ne.$set(Ve);const Te={};t&1048576&&(Te.$$scope={dirty:t,ctx:e}),ie.$set(Te);const Xe={};t&1048576&&(Xe.$$scope={dirty:t,ctx:e}),ae.$set(Xe);const Me={};t&1048576&&(Me.$$scope={dirty:t,ctx:e}),re.$set(Me)},i(e){ct||(g(x.$$.fragment,e),g(F.$$.fragment,e),g(K.$$.fragment,e),g(V.$$.fragment,e),g(X.$$.fragment,e),g(M.$$.fragment,e),g(O.$$.fragment,e),g(Y.$$.fragment,e),g(j.$$.fragment,e),g(he.$$.fragment,e),g(J.$$.fragment,e),g(Z.$$.fragment,e),g(Q.$$.fragment,e),g(ee.$$.fragment,e),g(te.$$.fragment,e),g(ne.$$.fragment,e),g(ie.$$.fragment,e),g(ae.$$.fragment,e),g(pe.$$.fragment,e),g($e.$$.fragment,e),g(re.$$.fragment,e),g(de.$$.fragment,e),g(we.$$.fragment,e),g(ce.$$.fragment,e),g(ge.$$.fragment,e),g(_e.$$.fragment,e),g(ve.$$.fragment,e),g(ye.$$.fragment,e),ct=!0)},o(e){_(x.$$.fragment,e),_(F.$$.fragment,e),_(K.$$.fragment,e),_(V.$$.fragment,e),_(X.$$.fragment,e),_(M.$$.fragment,e),_(O.$$.fragment,e),_(Y.$$.fragment,e),_(j.$$.fragment,e),_(he.$$.fragment,e),_(J.$$.fragment,e),_(Z.$$.fragment,e),_(Q.$$.fragment,e),_(ee.$$.fragment,e),_(te.$$.fragment,e),_(ne.$$.fragment,e),_(ie.$$.fragment,e),_(ae.$$.fragment,e),_(pe.$$.fragment,e),_($e.$$.fragment,e),_(re.$$.fragment,e),_(de.$$.fragment,e),_(we.$$.fragment,e),_(ce.$$.fragment,e),_(ge.$$.fragment,e),_(_e.$$.fragment,e),_(ve.$$.fragment,e),_(ye.$$.fragment,e),ct=!1},d(e){e&&i(o),e&&i(a),e&&i(h),e&&i(I),e&&i(P),e&&i(N),v(x,e),e&&i(H),e&&i(f),e&&i(A),e&&i(B),v(F),e&&i(ue),v(K,e),e&&i(Ie),e&&i(U),e&&i(Oe),v(V,e),e&&i(Ye),e&&i(Pe),e&&i(je),v(X,e),e&&i(Je),e&&i(Le),e&&i(Ze),v(M,e),e&&i(Qe),e&&i(De),e&&i(et),e&&i(z),v(O),v(Y),v(j),v(he),v(J),v(Z),v(Q),v(ee),e&&i(tt),e&&i(D),v(te),v(ne),v(ie),v(ae),e&&i(nt),v(pe,e),e&&i(it),e&&i(oe),v($e),e&&i(at),v(re,e),e&&i(ot),e&&i(G),e&&i(rt),v(de,e),e&&i(st),v(we,e),e&&i(lt),e&&i(le),e&&i(ft),v(ce,e),e&&i(mt),v(ge,e),e&&i(ut),e&&i(Se),e&&i(ht),v(_e,e),e&&i(pt),v(ve,e),e&&i($t),v(ye,e),e&&i(dt),e&&i(be),e&&i(wt),e&&i(We)}}}function En(u){let o,n,a,h,q,I,P,L,N,x,H;return L=new on({props:{$$slots:{default:[Tn]},$$scope:{ctx:u}}}),x=new rn({props:{references:u[3]}}),{c(){o=y("meta"),n=p(),a=y("h1"),h=s("Weight Initialization"),q=p(),I=y("div"),P=p(),d(L.$$.fragment),N=p(),d(x.$$.fragment),this.h()},l(f){const k=nn("svelte-18ilasu",document.head);o=b(k,"META",{name:!0,content:!0}),k.forEach(i),n=$(f),a=b(f,"H1",{});var A=E(a);h=l(A,"Weight Initialization"),A.forEach(i),q=$(f),I=b(f,"DIV",{class:!0}),E(I).forEach(i),P=$(f),w(L.$$.fragment,f),N=$(f),w(x.$$.fragment,f),this.h()},h(){document.title="Weight Initialization - World4AI",me(o,"name","description"),me(o,"content","Proper weight initialization techniques, like Xavier/Glorot or Kaiming/He initialization, can decrease the chances of exploding or vanishing gradients. Deep learning frameworks like PyTorch or Keras provide initializatin techniques out of the box."),me(I,"class","separator")},m(f,k){m(document.head,o),r(f,n,k),r(f,a,k),m(a,h),r(f,q,k),r(f,I,k),r(f,P,k),c(L,f,k),r(f,N,k),c(x,f,k),H=!0},p(f,[k]){const A={};k&1048577&&(A.$$scope={dirty:k,ctx:f}),L.$set(A)},i(f){H||(g(L.$$.fragment,f),g(x.$$.fragment,f),H=!0)},o(f){_(L.$$.fragment,f),_(x.$$.fragment,f),H=!1},d(f){i(o),f&&i(n),f&&i(a),f&&i(q),f&&i(I),f&&i(P),v(L,f),f&&i(N),v(x,f)}}}function xn(u,o,n){const a=[{title:"Input",nodes:[{value:"x_1",class:"fill-gray-300"},{value:"x_2",class:"fill-gray-300"}]},{title:"Hidden Layer",nodes:[{value:"a_1",class:"fill-gray-300"},{value:"a_2",class:"fill-gray-300"}]},{title:"Output",nodes:[{value:"o",class:"fill-gray-300"}]}],h=[{title:"Input",nodes:[{value:"x_1",class:"fill-gray-300"},{value:"x_2",class:"fill-gray-300"}]},{title:"Hidden 1",nodes:[{value:"a_1",class:"fill-gray-300"},{value:"a_2",class:"fill-gray-300"},{value:"a_3",class:"fill-gray-300"}]},{title:"Hidden 2",nodes:[{value:"a_1",class:"fill-gray-300"},{value:"a_2",class:"fill-gray-300"}]}];let q=[{author:"Glorot, Xavier and Bengio Yoshua",title:"Understanding the Difficulty of Training Deep Feedforward Neural Networks",journal:"Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, Journal of Machine Learning Research",year:"2010",pages:"249-256",volume:"9",issue:""},{author:"K. He, X. Zhang, S. Ren and J. Sun",title:" Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification",journal:"2015 IEEE International Conference on Computer Vision (ICCV)",year:"2015",pages:"1026-1024",volume:"",issue:""}];const I=new fe(5);I._name="Feature 1";const P=new fe(2);P._name="Feature 2";const L=new fe(1);L._name="Weight 1";const N=new fe(1);N._name="Weight 2";const x=new fe(1);x._name="Weight 3";const H=new fe(1);H._name="Weight 4";const f=new fe(1);f._name="Weight 5";const k=new fe(1);k._name="Weight 6";const A=L.mul(I),B=N.mul(P),Ne=x.mul(I),F=H.mul(P),xe=A.add(B);xe._name="a_1";const ue=Ne.add(F);ue._name="a_2";const K=xe.mul(f),Ie=ue.mul(k),U=K.add(Ie);return U._name="Output",U.backward(),[U,a,h,q]}class Cn extends Qt{constructor(o){super(),en(this,o,xn,En,tn,{})}}export{Cn as default};
