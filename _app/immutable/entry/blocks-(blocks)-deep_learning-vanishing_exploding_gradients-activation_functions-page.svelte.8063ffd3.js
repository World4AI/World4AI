import{S as ei,i as ti,s as ni,k as T,a as $,q as p,y,W as ii,l as k,h as n,c as m,m as U,r as u,z as _,n as M,N as h,b as a,A as b,g as x,d as E,B as L,L as Qn,C as H}from"../chunks/index.4d92b023.js";import{C as ai}from"../chunks/Container.b0705c7b.js";import{L as oe}from"../chunks/Latex.e0b308c0.js";import{H as si}from"../chunks/Highlight.b7c1de53.js";import{F as oi,I as ri}from"../chunks/InternalLink.7deb899c.js";import{A as Mn}from"../chunks/Alert.25a852b3.js";import{P as nt}from"../chunks/PythonCode.212ba7a6.js";import{P as it,T as at}from"../chunks/Ticks.45eca5c5.js";import{P as He}from"../chunks/Path.7e6df014.js";import{X as st,Y as ot}from"../chunks/YLabel.182e66a3.js";const fi=""+new URL("../assets/relu_metrics.88132013.webp",import.meta.url).href,li=""+new URL("../assets/sigmoid_metrics.8ff37c5b.webp",import.meta.url).href;function $i(l){let s,o,r,w,v,D,f,I;return s=new He({props:{data:l[1]}}),r=new st({props:{text:"z",type:"latex"}}),v=new ot({props:{text:"\\sigma(z)",type:"latex",x:0}}),f=new at({props:{xTicks:[-10,-8,-6,-4,-2,0,2,4,6,8,10],yTicks:[0,.2,.4,.6,.8,1],xOffset:-10,yOffset:23,fontSize:8}}),{c(){y(s.$$.fragment),o=$(),y(r.$$.fragment),w=$(),y(v.$$.fragment),D=$(),y(f.$$.fragment)},l(t){_(s.$$.fragment,t),o=m(t),_(r.$$.fragment,t),w=m(t),_(v.$$.fragment,t),D=m(t),_(f.$$.fragment,t)},m(t,g){b(s,t,g),a(t,o,g),b(r,t,g),a(t,w,g),b(v,t,g),a(t,D,g),b(f,t,g),I=!0},p:H,i(t){I||(x(s.$$.fragment,t),x(r.$$.fragment,t),x(v.$$.fragment,t),x(f.$$.fragment,t),I=!0)},o(t){E(s.$$.fragment,t),E(r.$$.fragment,t),E(v.$$.fragment,t),E(f.$$.fragment,t),I=!1},d(t){L(s,t),t&&n(o),L(r,t),t&&n(w),L(v,t),t&&n(D),L(f,t)}}}function mi(l){let s=String.raw`\dfrac{1}{1+e^{-z}}`+"",o;return{c(){o=p(s)},l(r){o=u(r,s)},m(r,w){a(r,o,w)},p:H,d(r){r&&n(o)}}}function pi(l){let s=String.raw`\dfrac{e^{z}}{\sum e^{z}}`+"",o;return{c(){o=p(s)},l(r){o=u(r,s)},m(r,w){a(r,o,w)},p:H,d(r){r&&n(o)}}}function ui(l){let s;return{c(){s=p(`Use the sigmoid and the softmax as activations if you need to scale values
    between 0 and 1.`)},l(o){s=u(o,`Use the sigmoid and the softmax as activations if you need to scale values
    between 0 and 1.`)},m(o,r){a(o,s,r)},d(o){o&&n(s)}}}function hi(l){let s=String.raw`\dfrac{e^{z} - e^{-z}}{e^{z} + e^{-z}}`+"",o;return{c(){o=p(s)},l(r){o=u(r,s)},m(r,w){a(r,o,w)},p:H,d(r){r&&n(o)}}}function ci(l){let s,o,r,w,v,D,f,I;return s=new He({props:{data:l[2]}}),r=new st({props:{text:"z",type:"latex"}}),v=new ot({props:{text:"\\tanh(z)",type:"latex",x:0}}),f=new at({props:{xTicks:[-10,-8,-6,-4,-2,0,2,4,6,8,10],yTicks:[-1,-.8,-.6,-.4,-.2,0,.2,.4,.6,.8,1],xOffset:-10,yOffset:23,fontSize:8}}),{c(){y(s.$$.fragment),o=$(),y(r.$$.fragment),w=$(),y(v.$$.fragment),D=$(),y(f.$$.fragment)},l(t){_(s.$$.fragment,t),o=m(t),_(r.$$.fragment,t),w=m(t),_(v.$$.fragment,t),D=m(t),_(f.$$.fragment,t)},m(t,g){b(s,t,g),a(t,o,g),b(r,t,g),a(t,w,g),b(v,t,g),a(t,D,g),b(f,t,g),I=!0},p:H,i(t){I||(x(s.$$.fragment,t),x(r.$$.fragment,t),x(v.$$.fragment,t),x(f.$$.fragment,t),I=!0)},o(t){E(s.$$.fragment,t),E(r.$$.fragment,t),E(v.$$.fragment,t),E(f.$$.fragment,t),I=!1},d(t){L(s,t),t&&n(o),L(r,t),t&&n(w),L(v,t),t&&n(D),L(f,t)}}}function vi(l){let s,o,r,w,v,D,f,I,t,g;return s=new He({props:{data:l[3][0],strokeDashArray:[2,4]}}),r=new He({props:{data:l[3][1]}}),v=new st({props:{text:"z",type:"latex",y:240}}),f=new ot({props:{text:"f(z)'",type:"latex",x:0}}),t=new at({props:{xTicks:[-10,-8,-6,-4,-2,0,2,4,6,8,10],yTicks:[0,.2,.4,.6,.8,1],xOffset:-10,yOffset:23,fontSize:8}}),{c(){y(s.$$.fragment),o=$(),y(r.$$.fragment),w=$(),y(v.$$.fragment),D=$(),y(f.$$.fragment),I=$(),y(t.$$.fragment)},l(d){_(s.$$.fragment,d),o=m(d),_(r.$$.fragment,d),w=m(d),_(v.$$.fragment,d),D=m(d),_(f.$$.fragment,d),I=m(d),_(t.$$.fragment,d)},m(d,c){b(s,d,c),a(d,o,c),b(r,d,c),a(d,w,c),b(v,d,c),a(d,D,c),b(f,d,c),a(d,I,c),b(t,d,c),g=!0},p:H,i(d){g||(x(s.$$.fragment,d),x(r.$$.fragment,d),x(v.$$.fragment,d),x(f.$$.fragment,d),x(t.$$.fragment,d),g=!0)},o(d){E(s.$$.fragment,d),E(r.$$.fragment,d),E(v.$$.fragment,d),E(f.$$.fragment,d),E(t.$$.fragment,d),g=!1},d(d){L(s,d),d&&n(o),L(r,d),d&&n(w),L(v,d),d&&n(D),L(f,d),d&&n(I),L(t,d)}}}function wi(l){let s;return{c(){s=p(`Use the tanh as your activation function if you need to scale values between
    -1 and 1.`)},l(o){s=u(o,`Use the tanh as your activation function if you need to scale values between
    -1 and 1.`)},m(o,r){a(o,s,r)},d(o){o&&n(s)}}}function di(l){let s;return{c(){s=p("z")},l(o){s=u(o,"z")},m(o,r){a(o,s,r)},d(o){o&&n(s)}}}function gi(l){let s=String.raw`
    \text{ReLU}(z) = 
        \begin{cases}
        z & \text{if } z > 0 \\
            0 & \text{ otherwise }
        \end{cases}
    `+"",o;return{c(){o=p(s)},l(r){o=u(r,s)},m(r,w){a(r,o,w)},p:H,d(r){r&&n(o)}}}function yi(l){let s,o,r,w,v,D,f,I;return s=new He({props:{data:l[4]}}),r=new st({props:{text:"z",type:"latex"}}),v=new ot({props:{text:"relu(z)",type:"latex",x:0}}),f=new at({props:{xTicks:[-10,-8,-6,-4,-2,0,2,4,6,8,10],yTicks:[0,1,2,3,4,5,6,7,8,9,10],xOffset:-10,yOffset:23,fontSize:8}}),{c(){y(s.$$.fragment),o=$(),y(r.$$.fragment),w=$(),y(v.$$.fragment),D=$(),y(f.$$.fragment)},l(t){_(s.$$.fragment,t),o=m(t),_(r.$$.fragment,t),w=m(t),_(v.$$.fragment,t),D=m(t),_(f.$$.fragment,t)},m(t,g){b(s,t,g),a(t,o,g),b(r,t,g),a(t,w,g),b(v,t,g),a(t,D,g),b(f,t,g),I=!0},p:H,i(t){I||(x(s.$$.fragment,t),x(r.$$.fragment,t),x(v.$$.fragment,t),x(f.$$.fragment,t),I=!0)},o(t){E(s.$$.fragment,t),E(r.$$.fragment,t),E(v.$$.fragment,t),E(f.$$.fragment,t),I=!1},d(t){L(s,t),t&&n(o),L(r,t),t&&n(w),L(v,t),t&&n(D),L(f,t)}}}function _i(l){let s;return{c(){s=p("z")},l(o){s=u(o,"z")},m(o,r){a(o,s,r)},d(o){o&&n(s)}}}function bi(l){let s=String.raw`
    \text{ReLU Derivative} = 
        \begin{cases}
        1 & \text{if } z > 0 \\
            0 & \text{ otherwise }
        \end{cases}
    `+"",o;return{c(){o=p(s)},l(r){o=u(r,s)},m(r,w){a(r,o,w)},p:H,d(r){r&&n(o)}}}function xi(l){let s;return{c(){s=p("dying relu problem")},l(o){s=u(o,"dying relu problem")},m(o,r){a(o,s,r)},d(o){o&&n(s)}}}function Ei(l){let s=String.raw`
    \text{ReLU} = 
        \begin{cases}
        z & \text{if } z > 0 \\
            \alpha * z & \text{ otherwise }
        \end{cases}
    `+"",o;return{c(){o=p(s)},l(r){o=u(r,s)},m(r,w){a(r,o,w)},p:H,d(r){r&&n(o)}}}function Li(l){let s,o,r,w,v,D,f,I;return s=new He({props:{data:l[5]}}),r=new st({props:{text:"z",type:"latex"}}),v=new ot({props:{text:"relu(z)",type:"latex",x:0}}),f=new at({props:{xTicks:[-10,-8,-6,-4,-2,0,2,4,6,8,10],yTicks:[0,1,2,3,4,5,6,7,8,9,10],xOffset:-10,yOffset:23,fontSize:8}}),{c(){y(s.$$.fragment),o=$(),y(r.$$.fragment),w=$(),y(v.$$.fragment),D=$(),y(f.$$.fragment)},l(t){_(s.$$.fragment,t),o=m(t),_(r.$$.fragment,t),w=m(t),_(v.$$.fragment,t),D=m(t),_(f.$$.fragment,t)},m(t,g){b(s,t,g),a(t,o,g),b(r,t,g),a(t,w,g),b(v,t,g),a(t,D,g),b(f,t,g),I=!0},p:H,i(t){I||(x(s.$$.fragment,t),x(r.$$.fragment,t),x(v.$$.fragment,t),x(f.$$.fragment,t),I=!0)},o(t){E(s.$$.fragment,t),E(r.$$.fragment,t),E(v.$$.fragment,t),E(f.$$.fragment,t),I=!1},d(t){L(s,t),t&&n(o),L(r,t),t&&n(w),L(v,t),t&&n(D),L(f,t)}}}function Di(l){let s;return{c(){s=p(`You should use the ReLU (or its relatives) as your main activation function.
    Deviate only from this activation, if you have any specific reason to do so.`)},l(o){s=u(o,`You should use the ReLU (or its relatives) as your main activation function.
    Deviate only from this activation, if you have any specific reason to do so.`)},m(o,r){a(o,s,r)},d(o){o&&n(s)}}}function Ti(l){let s,o,r,w,v,D,f,I,t,g,d,c,P,S,Vt,F,Jt,A,Kt,rt,q,ft,R,Qt,ze,Zt,en,Oe,tn,nn,Me,an,sn,We,on,rn,lt,re,$t,C,fn,Fe,ln,$n,mt,we,pt,de,mn,ut,j,pn,B,un,ht,ge,hn,ct,X,vt,ye,cn,wt,_e,G,dt,be,vn,gt,Y,yt,xe,wn,_t,fe,bt,Ee,xt,Le,dn,Et,z,gn,V,yn,le,_n,Lt,J,Dt,K,Tt,Q,bn,Z,xn,kt,ee,It,te,En,ne,Ln,Ut,De,Dn,Pt,$e,St,Te,Tn,Rt,ie,Nt,ke,kn,Ht,ae,zt,O,In,Ae,Un,Pn,qe,Sn,Rn,Ot,me,Mt,Ie,Nn,Wt,se,Ft,Ue,Hn,At,pe,qt,Pe,zn,Ct,ue,Fn,jt,Se,On,Bt,he,An,Xt;return c=new it({props:{width:500,height:250,maxWidth:800,domain:[-10,10],range:[0,1],padding:{top:5,right:40,bottom:30,left:40},$$slots:{default:[$i]},$$scope:{ctx:l}}}),F=new oe({props:{$$slots:{default:[mi]},$$scope:{ctx:l}}}),A=new oe({props:{$$slots:{default:[pi]},$$scope:{ctx:l}}}),q=new Mn({props:{type:"info",$$slots:{default:[ui]},$$scope:{ctx:l}}}),re=new nt({props:{code:`# functional way
sigmoid_output = torch.sigmoid(X)
# object-oriented way
sigmoid_layer = torch.nn.Sigmoid()
`}}),B=new oe({props:{$$slots:{default:[hi]},$$scope:{ctx:l}}}),X=new it({props:{width:500,height:250,maxWidth:800,domain:[-10,10],range:[-1,1],padding:{top:5,right:40,bottom:30,left:40},$$slots:{default:[ci]},$$scope:{ctx:l}}}),G=new it({props:{width:500,height:250,maxWidth:800,domain:[-10,10],range:[0,1],padding:{top:5,right:40,bottom:30,left:40},$$slots:{default:[vi]},$$scope:{ctx:l}}}),Y=new Mn({props:{type:"info",$$slots:{default:[wi]},$$scope:{ctx:l}}}),fe=new nt({props:{code:`tanh_output = torch.tanh(X)
tanh_layer = torch.nn.Tanh()`}}),V=new oe({props:{$$slots:{default:[di]},$$scope:{ctx:l}}}),le=new ri({props:{type:"reference",id:"1"}}),J=new oe({props:{$$slots:{default:[gi]},$$scope:{ctx:l}}}),K=new it({props:{width:500,height:250,maxWidth:800,domain:[-10,10],range:[0,10],padding:{top:15,right:40,bottom:30,left:40},$$slots:{default:[yi]},$$scope:{ctx:l}}}),Z=new oe({props:{$$slots:{default:[_i]},$$scope:{ctx:l}}}),ee=new oe({props:{$$slots:{default:[bi]},$$scope:{ctx:l}}}),ne=new si({props:{$$slots:{default:[xi]},$$scope:{ctx:l}}}),$e=new nt({props:{code:`relu_output = torch.relu(X)
relu_layer = torch.nn.ReLU()
`}}),ie=new oe({props:{$$slots:{default:[Ei]},$$scope:{ctx:l}}}),ae=new it({props:{width:500,height:250,maxWidth:800,domain:[-10,10],range:[0,10],padding:{top:15,right:40,bottom:30,left:40},$$slots:{default:[Li]},$$scope:{ctx:l}}}),me=new nt({props:{code:`lrelu_output = torch.nn.functional.leaky_relu(X, negative_slope=0.01)
lrelu_layer = torch.nn.LeakyReLU(negative_slope=0.01)
`}}),se=new Mn({props:{type:"info",$$slots:{default:[Di]},$$scope:{ctx:l}}}),pe=new nt({props:{code:`class SigmoidModel(nn.Module):
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
        
    def forward(self, features):
        return self.layers(features)
    
class ReluModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(NUM_FEATURES, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, NUM_LABELS)
        )
        
    def forward(self, features):
        return self.layers(features)`}}),{c(){s=T("p"),o=p(`The sigmoid activation function is one of the main causes of the vanishing
    gradients problem. Because of that researchers have tried to come up with
    activation functions with better properties. In this section we are going to
    compare and contrast some of the most popular activation functions, while
    emphasizing when each of the activations should be used.`),r=$(),w=T("div"),v=$(),D=T("h2"),f=p("Sigmoid and Softmax"),I=$(),t=T("p"),g=p(`From our previous discussion it might have seemed, that the sigmoid
    activation function (and by extension softmax) is the root cause of the
    vanishing gradient problem and should be avoided at all cost.`),d=$(),y(c.$$.fragment),P=$(),S=T("p"),Vt=p(`While this is somewhat true, the original argumentation that we used when we
    implemented logistic regression still applies. We can use the sigmoid and
    the softmax to turn logits into probabilities. Nowadays we primarily use the
    sigmoid `),y(F.$$.fragment),Jt=p(" and the softmax "),y(A.$$.fragment),Kt=p(` in the last layer of the neural network, to determine the probability to belong
    to a particular class.`),rt=$(),y(q.$$.fragment),ft=$(),R=T("p"),Qt=p(`There are generally two ways to implement activation functions. We can use
    PyTorch in a functional way and apply `),ze=T("code"),Zt=p("torch.sigmoid(X)"),en=p(` in the
    `),Oe=T("code"),tn=p("forward()"),nn=p(`
    function of the model or as we have done so far, we can use the object-oriented
    way and use the `),Me=T("code"),an=p("torch.nn.Sigmoid()"),sn=p(` as part of
    `),We=T("code"),on=p("nn.Sequential()"),rn=p("."),lt=$(),y(re.$$.fragment),$t=$(),C=T("p"),fn=p("If we can fit the whole logic of the model into "),Fe=T("code"),ln=p("nn.Sequential"),$n=p(`,
    we will generally do that and use the object-oriented way. Sometimes, when
    the code gets more complicated, this will not possible and we will resort to
    the functional approach. The choice is generally yours.`),mt=$(),we=T("div"),pt=$(),de=T("h2"),mn=p("Hyperbolic Tangent"),ut=$(),j=T("p"),pn=p("The tanh activation function "),y(B.$$.fragment),un=p(` (also called hypterbolic tangent) is similar in spirit to the sigmoid activation
    function. Looking from a distance you might confuse the two, but there are some
    subtle differences.`),ht=$(),ge=T("p"),hn=p(`While both functions saturate when we use very low and very high inputs, the
    sigmoid squishes values between 0 and 1, while the tanh squishes values
    between -1 and 1.`),ct=$(),y(X.$$.fragment),vt=$(),ye=T("p"),cn=p(`For a long time researchers used the tanh activation function instead of the
    sigmoid, because it worked better in practice. Generally tanh exhibits a
    more favourable derivative function. While the sigmoid can only have very
    low derivatives of up to 0.25, the tanh can exhibit a derivative of up to 1,
    thereby reducing the risk of vanishing gradients.`),wt=$(),_e=T("p"),y(G.$$.fragment),dt=$(),be=T("p"),vn=p(`Over time researchers found better activations functoions that they prefer
    over tanh, but in case you actually desire outputs between -1 and 1, you
    should use the tanh.`),gt=$(),y(Y.$$.fragment),yt=$(),xe=T("p"),wn=p(`Once again there are two broad approaches to activation functions: the
    functional and the object-oriented one.`),_t=$(),y(fe.$$.fragment),bt=$(),Ee=T("div"),xt=$(),Le=T("h2"),dn=p("ReLU"),Et=$(),z=T("p"),gn=p(`The ReLU (rectified linear unit) is at the same time extremely simple and
    extremely powerful. The function returns the unchanged input `),y(V.$$.fragment),yn=p(" as its output when the input value is positive and 0 otherwise "),y(le.$$.fragment),_n=p("."),Lt=$(),y(J.$$.fragment),Dt=$(),y(K.$$.fragment),Tt=$(),Q=T("p"),bn=p(`The calculation of the derivative is also extremely straightforward. It is
    exactly 1 when the net input `),y(Z.$$.fragment),xn=p(` is above 1 and 0 otherwise. While
    technically we can not differentiate the function at the knick, in practice this
    works very well.`),kt=$(),y(ee.$$.fragment),It=$(),te=T("p"),En=p(`Hopefully you will interject at this point and point out, that while the
    derivative of exactly 1 will help to fight the problem of vanishing
    gradients, a derivative of 0 will push the product in the chain rule to
    exactly 0. This is true and is known as the `),y(ne.$$.fragment),Ln=p(`, but in practice you will not encounter the problem too often. Given that
    you have a large amount of neurons in each layer, there should be enough
    paths to propagate the signal.`),Ut=$(),De=T("p"),Dn=p("PyTorch offers the two approaches for ReLU as well."),Pt=$(),y($e.$$.fragment),St=$(),Te=T("p"),Tn=p(`Over time researchers tried to come up with improvements for the ReLU
    activation. The leaky ReLU for example does not completely kill off the
    signal, but provides a small slope when the net input is negative.`),Rt=$(),y(ie.$$.fragment),Nt=$(),ke=T("p"),kn=p("In the example below alpha corresponds to 0.1."),Ht=$(),y(ae.$$.fragment),zt=$(),O=T("p"),In=p(`When activation functions start to get slighly more exotic, you will often
    not find the in the `),Ae=T("code"),Un=p("torch"),Pn=p(` namespace directly, but in the
    `),qe=T("code"),Sn=p("torch.nn.functional"),Rn=p(" namespace."),Ot=$(),y(me.$$.fragment),Mt=$(),Ie=T("p"),Nn=p(`There are many more activation functions out there, expecially those that
    try to improve the original ReLU. For the most part we will use the plain
    vanilla ReLU, because the mentioned improvements generally do not provide
    significant advantages.`),Wt=$(),y(se.$$.fragment),Ft=$(),Ue=T("p"),Hn=p(`Now let's have a peak at the difference in the performance between the
    sigmoid and the ReLU activation functions. Once again we are dealing with
    the MNIST dataset, but this time around we create two models, each with a
    different set of activation functions. Both models are larger, than they
    need to be, in order to demonstrate the vanishing gradient problem.`),At=$(),y(pe.$$.fragment),qt=$(),Pe=T("p"),zn=p(`The sigmoid model starts out very slowly and even after 30 iterations has
    not managed to decrease the training loss significantly. If you train the
    same sigmoid model several times, you will notice, that sometimes the loss
    does not decrease at all. It all depends on the starting weights.`),Ct=$(),ue=T("img"),jt=$(),Se=T("p"),On=p(`The loss of the ReLU model on the other hand decreases significantly, thus
    indicating that the gradients propagate much better with this type of
    activation function.`),Bt=$(),he=T("img"),this.h()},l(e){s=k(e,"P",{});var i=U(s);o=u(i,`The sigmoid activation function is one of the main causes of the vanishing
    gradients problem. Because of that researchers have tried to come up with
    activation functions with better properties. In this section we are going to
    compare and contrast some of the most popular activation functions, while
    emphasizing when each of the activations should be used.`),i.forEach(n),r=m(e),w=k(e,"DIV",{class:!0}),U(w).forEach(n),v=m(e),D=k(e,"H2",{});var Ce=U(D);f=u(Ce,"Sigmoid and Softmax"),Ce.forEach(n),I=m(e),t=k(e,"P",{});var je=U(t);g=u(je,`From our previous discussion it might have seemed, that the sigmoid
    activation function (and by extension softmax) is the root cause of the
    vanishing gradient problem and should be avoided at all cost.`),je.forEach(n),d=m(e),_(c.$$.fragment,e),P=m(e),S=k(e,"P",{});var W=U(S);Vt=u(W,`While this is somewhat true, the original argumentation that we used when we
    implemented logistic regression still applies. We can use the sigmoid and
    the softmax to turn logits into probabilities. Nowadays we primarily use the
    sigmoid `),_(F.$$.fragment,W),Jt=u(W," and the softmax "),_(A.$$.fragment,W),Kt=u(W,` in the last layer of the neural network, to determine the probability to belong
    to a particular class.`),W.forEach(n),rt=m(e),_(q.$$.fragment,e),ft=m(e),R=k(e,"P",{});var N=U(R);Qt=u(N,`There are generally two ways to implement activation functions. We can use
    PyTorch in a functional way and apply `),ze=k(N,"CODE",{});var Be=U(ze);Zt=u(Be,"torch.sigmoid(X)"),Be.forEach(n),en=u(N,` in the
    `),Oe=k(N,"CODE",{});var Xe=U(Oe);tn=u(Xe,"forward()"),Xe.forEach(n),nn=u(N,`
    function of the model or as we have done so far, we can use the object-oriented
    way and use the `),Me=k(N,"CODE",{});var Ge=U(Me);an=u(Ge,"torch.nn.Sigmoid()"),Ge.forEach(n),sn=u(N,` as part of
    `),We=k(N,"CODE",{});var Ye=U(We);on=u(Ye,"nn.Sequential()"),Ye.forEach(n),rn=u(N,"."),N.forEach(n),lt=m(e),_(re.$$.fragment,e),$t=m(e),C=k(e,"P",{});var ce=U(C);fn=u(ce,"If we can fit the whole logic of the model into "),Fe=k(ce,"CODE",{});var Ve=U(Fe);ln=u(Ve,"nn.Sequential"),Ve.forEach(n),$n=u(ce,`,
    we will generally do that and use the object-oriented way. Sometimes, when
    the code gets more complicated, this will not possible and we will resort to
    the functional approach. The choice is generally yours.`),ce.forEach(n),mt=m(e),we=k(e,"DIV",{class:!0}),U(we).forEach(n),pt=m(e),de=k(e,"H2",{});var Je=U(de);mn=u(Je,"Hyperbolic Tangent"),Je.forEach(n),ut=m(e),j=k(e,"P",{});var ve=U(j);pn=u(ve,"The tanh activation function "),_(B.$$.fragment,ve),un=u(ve,` (also called hypterbolic tangent) is similar in spirit to the sigmoid activation
    function. Looking from a distance you might confuse the two, but there are some
    subtle differences.`),ve.forEach(n),ht=m(e),ge=k(e,"P",{});var Ke=U(ge);hn=u(Ke,`While both functions saturate when we use very low and very high inputs, the
    sigmoid squishes values between 0 and 1, while the tanh squishes values
    between -1 and 1.`),Ke.forEach(n),ct=m(e),_(X.$$.fragment,e),vt=m(e),ye=k(e,"P",{});var Qe=U(ye);cn=u(Qe,`For a long time researchers used the tanh activation function instead of the
    sigmoid, because it worked better in practice. Generally tanh exhibits a
    more favourable derivative function. While the sigmoid can only have very
    low derivatives of up to 0.25, the tanh can exhibit a derivative of up to 1,
    thereby reducing the risk of vanishing gradients.`),Qe.forEach(n),wt=m(e),_e=k(e,"P",{});var Ze=U(_e);_(G.$$.fragment,Ze),Ze.forEach(n),dt=m(e),be=k(e,"P",{});var et=U(be);vn=u(et,`Over time researchers found better activations functoions that they prefer
    over tanh, but in case you actually desire outputs between -1 and 1, you
    should use the tanh.`),et.forEach(n),gt=m(e),_(Y.$$.fragment,e),yt=m(e),xe=k(e,"P",{});var tt=U(xe);wn=u(tt,`Once again there are two broad approaches to activation functions: the
    functional and the object-oriented one.`),tt.forEach(n),_t=m(e),_(fe.$$.fragment,e),bt=m(e),Ee=k(e,"DIV",{class:!0}),U(Ee).forEach(n),xt=m(e),Le=k(e,"H2",{});var qn=U(Le);dn=u(qn,"ReLU"),qn.forEach(n),Et=m(e),z=k(e,"P",{});var Re=U(z);gn=u(Re,`The ReLU (rectified linear unit) is at the same time extremely simple and
    extremely powerful. The function returns the unchanged input `),_(V.$$.fragment,Re),yn=u(Re," as its output when the input value is positive and 0 otherwise "),_(le.$$.fragment,Re),_n=u(Re,"."),Re.forEach(n),Lt=m(e),_(J.$$.fragment,e),Dt=m(e),_(K.$$.fragment,e),Tt=m(e),Q=k(e,"P",{});var Gt=U(Q);bn=u(Gt,`The calculation of the derivative is also extremely straightforward. It is
    exactly 1 when the net input `),_(Z.$$.fragment,Gt),xn=u(Gt,` is above 1 and 0 otherwise. While
    technically we can not differentiate the function at the knick, in practice this
    works very well.`),Gt.forEach(n),kt=m(e),_(ee.$$.fragment,e),It=m(e),te=k(e,"P",{});var Yt=U(te);En=u(Yt,`Hopefully you will interject at this point and point out, that while the
    derivative of exactly 1 will help to fight the problem of vanishing
    gradients, a derivative of 0 will push the product in the chain rule to
    exactly 0. This is true and is known as the `),_(ne.$$.fragment,Yt),Ln=u(Yt,`, but in practice you will not encounter the problem too often. Given that
    you have a large amount of neurons in each layer, there should be enough
    paths to propagate the signal.`),Yt.forEach(n),Ut=m(e),De=k(e,"P",{});var Cn=U(De);Dn=u(Cn,"PyTorch offers the two approaches for ReLU as well."),Cn.forEach(n),Pt=m(e),_($e.$$.fragment,e),St=m(e),Te=k(e,"P",{});var jn=U(Te);Tn=u(jn,`Over time researchers tried to come up with improvements for the ReLU
    activation. The leaky ReLU for example does not completely kill off the
    signal, but provides a small slope when the net input is negative.`),jn.forEach(n),Rt=m(e),_(ie.$$.fragment,e),Nt=m(e),ke=k(e,"P",{});var Bn=U(ke);kn=u(Bn,"In the example below alpha corresponds to 0.1."),Bn.forEach(n),Ht=m(e),_(ae.$$.fragment,e),zt=m(e),O=k(e,"P",{});var Ne=U(O);In=u(Ne,`When activation functions start to get slighly more exotic, you will often
    not find the in the `),Ae=k(Ne,"CODE",{});var Xn=U(Ae);Un=u(Xn,"torch"),Xn.forEach(n),Pn=u(Ne,` namespace directly, but in the
    `),qe=k(Ne,"CODE",{});var Gn=U(qe);Sn=u(Gn,"torch.nn.functional"),Gn.forEach(n),Rn=u(Ne," namespace."),Ne.forEach(n),Ot=m(e),_(me.$$.fragment,e),Mt=m(e),Ie=k(e,"P",{});var Yn=U(Ie);Nn=u(Yn,`There are many more activation functions out there, expecially those that
    try to improve the original ReLU. For the most part we will use the plain
    vanilla ReLU, because the mentioned improvements generally do not provide
    significant advantages.`),Yn.forEach(n),Wt=m(e),_(se.$$.fragment,e),Ft=m(e),Ue=k(e,"P",{});var Vn=U(Ue);Hn=u(Vn,`Now let's have a peak at the difference in the performance between the
    sigmoid and the ReLU activation functions. Once again we are dealing with
    the MNIST dataset, but this time around we create two models, each with a
    different set of activation functions. Both models are larger, than they
    need to be, in order to demonstrate the vanishing gradient problem.`),Vn.forEach(n),At=m(e),_(pe.$$.fragment,e),qt=m(e),Pe=k(e,"P",{});var Jn=U(Pe);zn=u(Jn,`The sigmoid model starts out very slowly and even after 30 iterations has
    not managed to decrease the training loss significantly. If you train the
    same sigmoid model several times, you will notice, that sometimes the loss
    does not decrease at all. It all depends on the starting weights.`),Jn.forEach(n),Ct=m(e),ue=k(e,"IMG",{src:!0,alt:!0}),jt=m(e),Se=k(e,"P",{});var Kn=U(Se);On=u(Kn,`The loss of the ReLU model on the other hand decreases significantly, thus
    indicating that the gradients propagate much better with this type of
    activation function.`),Kn.forEach(n),Bt=m(e),he=k(e,"IMG",{src:!0,alt:!0}),this.h()},h(){M(w,"class","separator"),M(we,"class","separator"),M(Ee,"class","separator"),Qn(ue.src,Fn=li)||M(ue,"src",Fn),M(ue,"alt","Metrics with sigmoid activation"),Qn(he.src,An=fi)||M(he,"src",An),M(he,"alt","Metrics with relu activation")},m(e,i){a(e,s,i),h(s,o),a(e,r,i),a(e,w,i),a(e,v,i),a(e,D,i),h(D,f),a(e,I,i),a(e,t,i),h(t,g),a(e,d,i),b(c,e,i),a(e,P,i),a(e,S,i),h(S,Vt),b(F,S,null),h(S,Jt),b(A,S,null),h(S,Kt),a(e,rt,i),b(q,e,i),a(e,ft,i),a(e,R,i),h(R,Qt),h(R,ze),h(ze,Zt),h(R,en),h(R,Oe),h(Oe,tn),h(R,nn),h(R,Me),h(Me,an),h(R,sn),h(R,We),h(We,on),h(R,rn),a(e,lt,i),b(re,e,i),a(e,$t,i),a(e,C,i),h(C,fn),h(C,Fe),h(Fe,ln),h(C,$n),a(e,mt,i),a(e,we,i),a(e,pt,i),a(e,de,i),h(de,mn),a(e,ut,i),a(e,j,i),h(j,pn),b(B,j,null),h(j,un),a(e,ht,i),a(e,ge,i),h(ge,hn),a(e,ct,i),b(X,e,i),a(e,vt,i),a(e,ye,i),h(ye,cn),a(e,wt,i),a(e,_e,i),b(G,_e,null),a(e,dt,i),a(e,be,i),h(be,vn),a(e,gt,i),b(Y,e,i),a(e,yt,i),a(e,xe,i),h(xe,wn),a(e,_t,i),b(fe,e,i),a(e,bt,i),a(e,Ee,i),a(e,xt,i),a(e,Le,i),h(Le,dn),a(e,Et,i),a(e,z,i),h(z,gn),b(V,z,null),h(z,yn),b(le,z,null),h(z,_n),a(e,Lt,i),b(J,e,i),a(e,Dt,i),b(K,e,i),a(e,Tt,i),a(e,Q,i),h(Q,bn),b(Z,Q,null),h(Q,xn),a(e,kt,i),b(ee,e,i),a(e,It,i),a(e,te,i),h(te,En),b(ne,te,null),h(te,Ln),a(e,Ut,i),a(e,De,i),h(De,Dn),a(e,Pt,i),b($e,e,i),a(e,St,i),a(e,Te,i),h(Te,Tn),a(e,Rt,i),b(ie,e,i),a(e,Nt,i),a(e,ke,i),h(ke,kn),a(e,Ht,i),b(ae,e,i),a(e,zt,i),a(e,O,i),h(O,In),h(O,Ae),h(Ae,Un),h(O,Pn),h(O,qe),h(qe,Sn),h(O,Rn),a(e,Ot,i),b(me,e,i),a(e,Mt,i),a(e,Ie,i),h(Ie,Nn),a(e,Wt,i),b(se,e,i),a(e,Ft,i),a(e,Ue,i),h(Ue,Hn),a(e,At,i),b(pe,e,i),a(e,qt,i),a(e,Pe,i),h(Pe,zn),a(e,Ct,i),a(e,ue,i),a(e,jt,i),a(e,Se,i),h(Se,On),a(e,Bt,i),a(e,he,i),Xt=!0},p(e,i){const Ce={};i&64&&(Ce.$$scope={dirty:i,ctx:e}),c.$set(Ce);const je={};i&64&&(je.$$scope={dirty:i,ctx:e}),F.$set(je);const W={};i&64&&(W.$$scope={dirty:i,ctx:e}),A.$set(W);const N={};i&64&&(N.$$scope={dirty:i,ctx:e}),q.$set(N);const Be={};i&64&&(Be.$$scope={dirty:i,ctx:e}),B.$set(Be);const Xe={};i&64&&(Xe.$$scope={dirty:i,ctx:e}),X.$set(Xe);const Ge={};i&64&&(Ge.$$scope={dirty:i,ctx:e}),G.$set(Ge);const Ye={};i&64&&(Ye.$$scope={dirty:i,ctx:e}),Y.$set(Ye);const ce={};i&64&&(ce.$$scope={dirty:i,ctx:e}),V.$set(ce);const Ve={};i&64&&(Ve.$$scope={dirty:i,ctx:e}),J.$set(Ve);const Je={};i&64&&(Je.$$scope={dirty:i,ctx:e}),K.$set(Je);const ve={};i&64&&(ve.$$scope={dirty:i,ctx:e}),Z.$set(ve);const Ke={};i&64&&(Ke.$$scope={dirty:i,ctx:e}),ee.$set(Ke);const Qe={};i&64&&(Qe.$$scope={dirty:i,ctx:e}),ne.$set(Qe);const Ze={};i&64&&(Ze.$$scope={dirty:i,ctx:e}),ie.$set(Ze);const et={};i&64&&(et.$$scope={dirty:i,ctx:e}),ae.$set(et);const tt={};i&64&&(tt.$$scope={dirty:i,ctx:e}),se.$set(tt)},i(e){Xt||(x(c.$$.fragment,e),x(F.$$.fragment,e),x(A.$$.fragment,e),x(q.$$.fragment,e),x(re.$$.fragment,e),x(B.$$.fragment,e),x(X.$$.fragment,e),x(G.$$.fragment,e),x(Y.$$.fragment,e),x(fe.$$.fragment,e),x(V.$$.fragment,e),x(le.$$.fragment,e),x(J.$$.fragment,e),x(K.$$.fragment,e),x(Z.$$.fragment,e),x(ee.$$.fragment,e),x(ne.$$.fragment,e),x($e.$$.fragment,e),x(ie.$$.fragment,e),x(ae.$$.fragment,e),x(me.$$.fragment,e),x(se.$$.fragment,e),x(pe.$$.fragment,e),Xt=!0)},o(e){E(c.$$.fragment,e),E(F.$$.fragment,e),E(A.$$.fragment,e),E(q.$$.fragment,e),E(re.$$.fragment,e),E(B.$$.fragment,e),E(X.$$.fragment,e),E(G.$$.fragment,e),E(Y.$$.fragment,e),E(fe.$$.fragment,e),E(V.$$.fragment,e),E(le.$$.fragment,e),E(J.$$.fragment,e),E(K.$$.fragment,e),E(Z.$$.fragment,e),E(ee.$$.fragment,e),E(ne.$$.fragment,e),E($e.$$.fragment,e),E(ie.$$.fragment,e),E(ae.$$.fragment,e),E(me.$$.fragment,e),E(se.$$.fragment,e),E(pe.$$.fragment,e),Xt=!1},d(e){e&&n(s),e&&n(r),e&&n(w),e&&n(v),e&&n(D),e&&n(I),e&&n(t),e&&n(d),L(c,e),e&&n(P),e&&n(S),L(F),L(A),e&&n(rt),L(q,e),e&&n(ft),e&&n(R),e&&n(lt),L(re,e),e&&n($t),e&&n(C),e&&n(mt),e&&n(we),e&&n(pt),e&&n(de),e&&n(ut),e&&n(j),L(B),e&&n(ht),e&&n(ge),e&&n(ct),L(X,e),e&&n(vt),e&&n(ye),e&&n(wt),e&&n(_e),L(G),e&&n(dt),e&&n(be),e&&n(gt),L(Y,e),e&&n(yt),e&&n(xe),e&&n(_t),L(fe,e),e&&n(bt),e&&n(Ee),e&&n(xt),e&&n(Le),e&&n(Et),e&&n(z),L(V),L(le),e&&n(Lt),L(J,e),e&&n(Dt),L(K,e),e&&n(Tt),e&&n(Q),L(Z),e&&n(kt),L(ee,e),e&&n(It),e&&n(te),L(ne),e&&n(Ut),e&&n(De),e&&n(Pt),L($e,e),e&&n(St),e&&n(Te),e&&n(Rt),L(ie,e),e&&n(Nt),e&&n(ke),e&&n(Ht),L(ae,e),e&&n(zt),e&&n(O),e&&n(Ot),L(me,e),e&&n(Mt),e&&n(Ie),e&&n(Wt),L(se,e),e&&n(Ft),e&&n(Ue),e&&n(At),L(pe,e),e&&n(qt),e&&n(Pe),e&&n(Ct),e&&n(ue),e&&n(jt),e&&n(Se),e&&n(Bt),e&&n(he)}}}function ki(l){let s,o,r,w,v,D,f,I,t,g,d;return I=new ai({props:{$$slots:{default:[Ti]},$$scope:{ctx:l}}}),g=new oi({props:{references:l[0]}}),{c(){s=T("meta"),o=$(),r=T("h1"),w=p("Activation Functions"),v=$(),D=T("div"),f=$(),y(I.$$.fragment),t=$(),y(g.$$.fragment),this.h()},l(c){const P=ii("svelte-yyuxtp",document.head);s=k(P,"META",{name:!0,content:!0}),P.forEach(n),o=m(c),r=k(c,"H1",{});var S=U(r);w=u(S,"Activation Functions"),S.forEach(n),v=m(c),D=k(c,"DIV",{class:!0}),U(D).forEach(n),f=m(c),_(I.$$.fragment,c),t=m(c),_(g.$$.fragment,c),this.h()},h(){document.title="Activation Functions - World4AI",M(s,"name","description"),M(s,"content","There are many different activation functions out there, but many encourage vanishing gradients. The sigmoid activation function is one of the main drivers of the vanishing gradients problem. The tanh is a slighly better option, but can still lead to vanishing gradients. ReLU (and its variants) is better suited for hidden units and is therefore the most popular activation function."),M(D,"class","separator")},m(c,P){h(document.head,s),a(c,o,P),a(c,r,P),h(r,w),a(c,v,P),a(c,D,P),a(c,f,P),b(I,c,P),a(c,t,P),b(g,c,P),d=!0},p(c,[P]){const S={};P&64&&(S.$$scope={dirty:P,ctx:c}),I.$set(S)},i(c){d||(x(I.$$.fragment,c),x(g.$$.fragment,c),d=!0)},o(c){E(I.$$.fragment,c),E(g.$$.fragment,c),d=!1},d(c){n(s),c&&n(o),c&&n(r),c&&n(v),c&&n(D),c&&n(f),L(I,c),c&&n(t),L(g,c)}}}function Wn(l){return 1/(1+Math.exp(-l))}function Ii(l){return Wn(l)*(1-Wn(l))}function Zn(l){return(Math.exp(l)-Math.exp(-l))/(Math.exp(l)+Math.exp(-l))}function Ui(l){return 1-Zn(l)**2}function Pi(l){return l<=0?0:l}function Si(l,s=.1){return l<=0?s*l:l}function Ri(l){let s=[{author:"Glorot, Xavier and Bordes, Antoine and Bengio, Yoshua",title:"Deep Sparse Rectifier Neural Networks, Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics",journal:"",year:"2011",pages:"315-323",volume:"15",issue:""}];const o=[];for(let f=-10;f<=10;f+=.1)o.push({x:f,y:Wn(f)});const r=[];for(let f=-10;f<=10;f+=.1)r.push({x:f,y:Zn(f)});const w=[[],[]];for(let f=-10;f<=10;f+=.1)w[0].push({x:f,y:Ii(f)}),w[1].push({x:f,y:Ui(f)});const v=[];for(let f=-10;f<=10;f+=.1)v.push({x:f,y:Pi(f)});const D=[];for(let f=-10;f<=10;f+=.1)D.push({x:f,y:Si(f)});return[s,o,r,w,v,D]}class ji extends ei{constructor(s){super(),ti(this,s,Ri,ki,ni,{})}}export{ji as default};
