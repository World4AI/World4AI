import{S as ce,i as ge,s as we,k as L,a as c,q as y,y as x,W as _e,l as O,h as i,c as g,m as q,r as b,z as P,n as Z,N as u,b as l,A as k,g as z,d as S,B as A,C as ve}from"../chunks/index.4d92b023.js";import{C as ye}from"../chunks/Container.b0705c7b.js";import{H as be}from"../chunks/Highlight.b7c1de53.js";import{P as le}from"../chunks/PythonCode.212ba7a6.js";import{P as Ee,T as Te}from"../chunks/Ticks.45eca5c5.js";import{P as fe}from"../chunks/Path.7e6df014.js";import{X as xe,Y as Pe}from"../chunks/YLabel.182e66a3.js";import{L as ue}from"../chunks/Legend.de38c007.js";function ke(T){let n;return{c(){n=y("early stopping")},l(p){n=b(p,"early stopping")},m(p,r){l(p,n,r)},d(p){p&&i(n)}}}function ze(T){let n,p,r,m,w,$,_,d,f,s,o,E,v,H,C,V;return n=new fe({props:{data:T[0],color:"var(--main-color-1)"}}),r=new fe({props:{data:T[1]}}),w=new fe({props:{data:T[2]}}),_=new Te({props:{xTicks:[0,10,20,30,40,50,60,70,80,90,100],yTicks:[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],xOffset:-19,yOffset:18,fontSize:10}}),f=new xe({props:{text:"Time",fontSize:15}}),o=new Pe({props:{text:"Loss",fontSize:15}}),v=new ue({props:{text:"Validation Loss",coordinates:{x:75,y:.92}}}),C=new ue({props:{text:"Training Loss",coordinates:{x:75,y:.85},legendColor:"var(--main-color-1)"}}),{c(){x(n.$$.fragment),p=c(),x(r.$$.fragment),m=c(),x(w.$$.fragment),$=c(),x(_.$$.fragment),d=c(),x(f.$$.fragment),s=c(),x(o.$$.fragment),E=c(),x(v.$$.fragment),H=c(),x(C.$$.fragment)},l(t){P(n.$$.fragment,t),p=g(t),P(r.$$.fragment,t),m=g(t),P(w.$$.fragment,t),$=g(t),P(_.$$.fragment,t),d=g(t),P(f.$$.fragment,t),s=g(t),P(o.$$.fragment,t),E=g(t),P(v.$$.fragment,t),H=g(t),P(C.$$.fragment,t)},m(t,h){k(n,t,h),l(t,p,h),k(r,t,h),l(t,m,h),k(w,t,h),l(t,$,h),k(_,t,h),l(t,d,h),k(f,t,h),l(t,s,h),k(o,t,h),l(t,E,h),k(v,t,h),l(t,H,h),k(C,t,h),V=!0},p:ve,i(t){V||(z(n.$$.fragment,t),z(r.$$.fragment,t),z(w.$$.fragment,t),z(_.$$.fragment,t),z(f.$$.fragment,t),z(o.$$.fragment,t),z(v.$$.fragment,t),z(C.$$.fragment,t),V=!0)},o(t){S(n.$$.fragment,t),S(r.$$.fragment,t),S(w.$$.fragment,t),S(_.$$.fragment,t),S(f.$$.fragment,t),S(o.$$.fragment,t),S(v.$$.fragment,t),S(C.$$.fragment,t),V=!1},d(t){A(n,t),t&&i(p),A(r,t),t&&i(m),A(w,t),t&&i($),A(_,t),t&&i(d),A(f,t),t&&i(s),A(o,t),t&&i(E),A(v,t),t&&i(H),A(C,t)}}}function Se(T){let n,p,r,m,w,$,_,d,f,s,o,E,v,H,C,V,t,h,ee,te,N,B,F,D,ie,X,se,ne,Y,oe,ae,J,I,K,M,re,Q,j,R;return r=new be({props:{$$slots:{default:[ke]},$$scope:{ctx:T}}}),$=new Ee({props:{width:500,height:300,maxWidth:700,domain:[0,100],range:[0,1],$$slots:{default:[ze]},$$scope:{ctx:T}}}),o=new le({props:{code:T[3]}}),B=new le({props:{code:T[4]}}),I=new le({props:{code:T[5]}}),{c(){n=L("p"),p=y(`A simple strategy to deal with overfitting is to interrupt training, once
    the validation loss has been increasing for a certain number of epochs. When
    the validation loss starts increasing, while the training loss keeps
    decreasing, it is reasonable to assume that the training process has entered
    the phase of overfitting. At that point we should not waste the time
    watching the divergence between the training and valuation loss increase.
    This strategy is called `),x(r.$$.fragment),m=y(`. After you
    stopped the training you go back to the weights that showed the lowest
    validation loss. The assumption is, that those weights will exhibit best
    generalization capabilities.`),w=c(),x($.$$.fragment),_=c(),d=L("p"),f=y(`PyTorch does not support early stopping out of the box, but if you know how
    to save and restore a model, you can easily implement this logic.`),s=c(),x(o.$$.fragment),E=c(),v=L("p"),H=y("The two functions that PyTorch provides are "),C=L("code"),V=y("torch.save()"),t=y(` and
    `),h=L("code"),ee=y("torch.load()"),te=y(`. Below for example we save and load a simple
    Tensor.`),N=c(),x(B.$$.fragment),F=c(),D=L("p"),ie=y(`Usually we are not interested in saving just tensors, but whole internal
    states. Modul states for example include all the weights and biases, but
    also layer specific parameters, like the dropout probability. Often we also
    need to save the state of the optimizer so that we can resume training at a
    later time. To retrieve a state, modules and optimizers provide a `),X=L("code"),se=y("state_dict()"),ne=y(`
    method. A state can be restored, by utilizing the
    `),Y=L("code"),oe=y("load_state_dict()"),ae=y(" method."),J=c(),x(I.$$.fragment),K=c(),M=L("p"),re=y(`We will not be using early stopping in the deep learning module, as this
    technique is generally considered a bad practice. Other techniques, like
    learning rate schedulers, that we will encounter in future chapters, will
    give us better options to decide if we found a good set of weights.`),Q=c(),j=L("div"),this.h()},l(e){n=O(e,"P",{});var a=q(n);p=b(a,`A simple strategy to deal with overfitting is to interrupt training, once
    the validation loss has been increasing for a certain number of epochs. When
    the validation loss starts increasing, while the training loss keeps
    decreasing, it is reasonable to assume that the training process has entered
    the phase of overfitting. At that point we should not waste the time
    watching the divergence between the training and valuation loss increase.
    This strategy is called `),P(r.$$.fragment,a),m=b(a,`. After you
    stopped the training you go back to the weights that showed the lowest
    validation loss. The assumption is, that those weights will exhibit best
    generalization capabilities.`),a.forEach(i),w=g(e),P($.$$.fragment,e),_=g(e),d=O(e,"P",{});var G=q(d);f=b(G,`PyTorch does not support early stopping out of the box, but if you know how
    to save and restore a model, you can easily implement this logic.`),G.forEach(i),s=g(e),P(o.$$.fragment,e),E=g(e),v=O(e,"P",{});var W=q(v);H=b(W,"The two functions that PyTorch provides are "),C=O(W,"CODE",{});var pe=q(C);V=b(pe,"torch.save()"),pe.forEach(i),t=b(W,` and
    `),h=O(W,"CODE",{});var me=q(h);ee=b(me,"torch.load()"),me.forEach(i),te=b(W,`. Below for example we save and load a simple
    Tensor.`),W.forEach(i),N=g(e),P(B.$$.fragment,e),F=g(e),D=O(e,"P",{});var U=q(D);ie=b(U,`Usually we are not interested in saving just tensors, but whole internal
    states. Modul states for example include all the weights and biases, but
    also layer specific parameters, like the dropout probability. Often we also
    need to save the state of the optimizer so that we can resume training at a
    later time. To retrieve a state, modules and optimizers provide a `),X=O(U,"CODE",{});var $e=q(X);se=b($e,"state_dict()"),$e.forEach(i),ne=b(U,`
    method. A state can be restored, by utilizing the
    `),Y=O(U,"CODE",{});var de=q(Y);oe=b(de,"load_state_dict()"),de.forEach(i),ae=b(U," method."),U.forEach(i),J=g(e),P(I.$$.fragment,e),K=g(e),M=O(e,"P",{});var he=q(M);re=b(he,`We will not be using early stopping in the deep learning module, as this
    technique is generally considered a bad practice. Other techniques, like
    learning rate schedulers, that we will encounter in future chapters, will
    give us better options to decide if we found a good set of weights.`),he.forEach(i),Q=g(e),j=O(e,"DIV",{class:!0}),q(j).forEach(i),this.h()},h(){Z(j,"class","separator")},m(e,a){l(e,n,a),u(n,p),k(r,n,null),u(n,m),l(e,w,a),k($,e,a),l(e,_,a),l(e,d,a),u(d,f),l(e,s,a),k(o,e,a),l(e,E,a),l(e,v,a),u(v,H),u(v,C),u(C,V),u(v,t),u(v,h),u(h,ee),u(v,te),l(e,N,a),k(B,e,a),l(e,F,a),l(e,D,a),u(D,ie),u(D,X),u(X,se),u(D,ne),u(D,Y),u(Y,oe),u(D,ae),l(e,J,a),k(I,e,a),l(e,K,a),l(e,M,a),u(M,re),l(e,Q,a),l(e,j,a),R=!0},p(e,a){const G={};a&256&&(G.$$scope={dirty:a,ctx:e}),r.$set(G);const W={};a&256&&(W.$$scope={dirty:a,ctx:e}),$.$set(W)},i(e){R||(z(r.$$.fragment,e),z($.$$.fragment,e),z(o.$$.fragment,e),z(B.$$.fragment,e),z(I.$$.fragment,e),R=!0)},o(e){S(r.$$.fragment,e),S($.$$.fragment,e),S(o.$$.fragment,e),S(B.$$.fragment,e),S(I.$$.fragment,e),R=!1},d(e){e&&i(n),A(r),e&&i(w),A($,e),e&&i(_),e&&i(d),e&&i(s),A(o,e),e&&i(E),e&&i(v),e&&i(N),A(B,e),e&&i(F),e&&i(D),e&&i(J),A(I,e),e&&i(K),e&&i(M),e&&i(Q),e&&i(j)}}}function Ae(T){let n,p,r,m,w,$,_,d,f;return d=new ye({props:{$$slots:{default:[Se]},$$scope:{ctx:T}}}),{c(){n=L("meta"),p=c(),r=L("h1"),m=y("Early Stopping"),w=c(),$=L("div"),_=c(),x(d.$$.fragment),this.h()},l(s){const o=_e("svelte-4x1b53",document.head);n=O(o,"META",{name:!0,content:!0}),o.forEach(i),p=g(s),r=O(s,"H1",{});var E=q(r);m=b(E,"Early Stopping"),E.forEach(i),w=g(s),$=O(s,"DIV",{class:!0}),q($).forEach(i),_=g(s),P(d.$$.fragment,s),this.h()},h(){document.title="Early Stopping - World4AI",Z(n,"name","description"),Z(n,"content","Early stopping is a simple technique to reduce overfitting by stopping the trainig process when the validation loss starts growing."),Z($,"class","separator")},m(s,o){u(document.head,n),l(s,p,o),l(s,r,o),u(r,m),l(s,w,o),l(s,$,o),l(s,_,o),k(d,s,o),f=!0},p(s,[o]){const E={};o&256&&(E.$$scope={dirty:o,ctx:s}),d.$set(E)},i(s){f||(z(d.$$.fragment,s),f=!0)},o(s){S(d.$$.fragment,s),f=!1},d(s){i(n),s&&i(p),s&&i(r),s&&i(w),s&&i($),s&&i(_),A(d,s)}}}function Ce(T){let n=[],p=[],r=1,m=1;for(let f=0;f<110;f++){let s=f,o=r;n.push({x:s,y:o}),o=m,p.push({x:s,y:o}),r*=.94,f<=30?m*=.95:f<=40?m*=.96:f<=50?m*=.97:f<=60?m*=.98:f<=65?m*=1:f<=70?m*=1.01:m*=1.03}return[n,p,[{x:65,y:0},{x:65,y:1}],`import torch
from torch import nn, optim`,`t = torch.ones(3, 3)
torch.save(t, f="tensor.pt")
loaded_t = torch.load(f="tensor.pt")`,`model = nn.Sequential(
    nn.Linear(10, 50), nn.Sigmoid(), nn.Linear(50, 10), nn.Sigmoid(), nn.Linear(10, 1)
)
optimizer = optim.SGD(model.parameters(), lr=0.01)

model_state = model.state_dict()
optim_state = optimizer.state_dict()

torch.save({"model": model_state, "optmim": optim_state}, f="state.py")
state = torch.load(f="state.py")

model.load_state_dict(state["model"])
optimizer.load_state_dict(state["optim"])`]}class Ie extends ce{constructor(n){super(),ge(this,n,Ce,Ae,we,{})}}export{Ie as default};
