import{S as lt,i as st,s as ft,Q as dt,q as u,R as gt,m as C,r as c,h as o,n as D,b as p,N as f,u as _t,C as ke,a8 as wt,k as R,a as T,y as w,W as yt,l as S,c as x,z as y,A as v,g as b,d as E,B as k}from"../chunks/index.4d92b023.js";import{C as vt}from"../chunks/Container.b0705c7b.js";import{L as Xe}from"../chunks/Latex.e0b308c0.js";import{H as Ze}from"../chunks/Highlight.b7c1de53.js";import{B as nt}from"../chunks/ButtonContainer.e9aac418.js";import{P as mt}from"../chunks/PlayButton.85103c5a.js";import{P as ot}from"../chunks/PythonCode.212ba7a6.js";import{P as it,T as ut}from"../chunks/Ticks.45eca5c5.js";import{P as ct}from"../chunks/Path.7e6df014.js";import{C as pt}from"../chunks/Circle.f281e92b.js";function bt(s){let t,a;return{c(){t=dt("text"),a=u(s[0]),this.h()},l(n){t=gt(n,"text",{"font-size":!0,class:!0,x:!0,y:!0});var m=C(t);a=c(m,s[0]),m.forEach(o),this.h()},h(){D(t,"font-size",s[1]),D(t,"class","title svelte-1nh78f2"),D(t,"x",s[2]),D(t,"y",s[3])},m(n,m){p(n,t,m),f(t,a)},p(n,[m]){m&1&&_t(a,n[0]),m&2&&D(t,"font-size",n[1]),m&4&&D(t,"x",n[2]),m&8&&D(t,"y",n[3])},i:ke,o:ke,d(n){n&&o(t)}}}function Et(s,t,a){let{text:n="Title"}=t,{fontSize:m=20}=t,g=wt("width"),{x:_=g/2}=t,{y:$=m}=t;return s.$$set=h=>{"text"in h&&a(0,n=h.text),"fontSize"in h&&a(1,m=h.fontSize),"x"in h&&a(2,_=h.x),"y"in h&&a(3,$=h.y)},[n,m,_,$]}class $t extends lt{constructor(t){super(),st(this,t,Et,bt,ft,{text:0,fontSize:1,x:2,y:3})}}function kt(s){let t;return{c(){t=u("\\alpha")},l(a){t=c(a,"\\alpha")},m(a,n){p(a,t,n)},d(a){a&&o(t)}}}function Tt(s){let t,a;return t=new mt({props:{f:s[6],delta:50}}),{c(){w(t.$$.fragment)},l(n){y(t.$$.fragment,n)},m(n,m){v(t,n,m),a=!0},p:ke,i(n){a||(b(t.$$.fragment,n),a=!0)},o(n){E(t.$$.fragment,n),a=!1},d(n){k(t,n)}}}function xt(s){let t,a,n,m,g,_,$,h;return t=new ut({props:{xTicks:[-10,-8,-6,-4,-2,0,2,4,6,8,10],yTicks:[0,10,20,30,40,50,60]}}),n=new ct({props:{data:s[5]}}),g=new pt({props:{data:[{x:s[0],y:s[4]}]}}),$=new $t({props:{text:`Constant Learning Rate ${ht.toFixed(2)}`}}),{c(){w(t.$$.fragment),a=T(),w(n.$$.fragment),m=T(),w(g.$$.fragment),_=T(),w($.$$.fragment)},l(r){y(t.$$.fragment,r),a=x(r),y(n.$$.fragment,r),m=x(r),y(g.$$.fragment,r),_=x(r),y($.$$.fragment,r)},m(r,i){v(t,r,i),p(r,a,i),v(n,r,i),p(r,m,i),v(g,r,i),p(r,_,i),v($,r,i),h=!0},p(r,i){const d={};i&17&&(d.data=[{x:r[0],y:r[4]}]),g.$set(d)},i(r){h||(b(t.$$.fragment,r),b(n.$$.fragment,r),b(g.$$.fragment,r),b($.$$.fragment,r),h=!0)},o(r){E(t.$$.fragment,r),E(n.$$.fragment,r),E(g.$$.fragment,r),E($.$$.fragment,r),h=!1},d(r){k(t,r),r&&o(a),k(n,r),r&&o(m),k(g,r),r&&o(_),k($,r)}}}function Pt(s){let t;return{c(){t=u("learning rate decay")},l(a){t=c(a,"learning rate decay")},m(a,n){p(a,t,n)},d(a){a&&o(t)}}}function Ct(s){let t;return{c(){t=u("learning rate scheduling")},l(a){t=c(a,"learning rate scheduling")},m(a,n){p(a,t,n)},d(a){a&&o(t)}}}function Rt(s){let t;return{c(){t=u("n")},l(a){t=c(a,"n")},m(a,n){p(a,t,n)},d(a){a&&o(t)}}}function St(s){let t;return{c(){t=u("0.9")},l(a){t=c(a,"0.9")},m(a,n){p(a,t,n)},d(a){a&&o(t)}}}function zt(s){let t;return{c(){t=u("reduce learning rate on plateau")},l(a){t=c(a,"reduce learning rate on plateau")},m(a,n){p(a,t,n)},d(a){a&&o(t)}}}function Lt(s){let t,a;return t=new mt({props:{f:s[7],delta:50}}),{c(){w(t.$$.fragment)},l(n){y(t.$$.fragment,n)},m(n,m){v(t,n,m),a=!0},p:ke,i(n){a||(b(t.$$.fragment,n),a=!0)},o(n){E(t.$$.fragment,n),a=!1},d(n){k(t,n)}}}function It(s){let t,a,n,m,g,_,$,h;return t=new ut({props:{xTicks:[-10,-8,-6,-4,-2,0,2,4,6,8,10],yTicks:[0,10,20,30,40,50,60]}}),n=new ct({props:{data:s[5]}}),g=new pt({props:{data:[{x:s[1],y:s[3]}]}}),$=new $t({props:{text:`Variable Learning Rate ${s[2].toFixed(3)}`}}),{c(){w(t.$$.fragment),a=T(),w(n.$$.fragment),m=T(),w(g.$$.fragment),_=T(),w($.$$.fragment)},l(r){y(t.$$.fragment,r),a=x(r),y(n.$$.fragment,r),m=x(r),y(g.$$.fragment,r),_=x(r),y($.$$.fragment,r)},m(r,i){v(t,r,i),p(r,a,i),v(n,r,i),p(r,m,i),v(g,r,i),p(r,_,i),v($,r,i),h=!0},p(r,i){const d={};i&10&&(d.data=[{x:r[1],y:r[3]}]),g.$set(d);const L={};i&4&&(L.text=`Variable Learning Rate ${r[2].toFixed(3)}`),$.$set(L)},i(r){h||(b(t.$$.fragment,r),b(n.$$.fragment,r),b(g.$$.fragment,r),b($.$$.fragment,r),h=!0)},o(r){E(t.$$.fragment,r),E(n.$$.fragment,r),E(g.$$.fragment,r),E($.$$.fragment,r),h=!1},d(r){k(t,r),r&&o(a),k(n,r),r&&o(m),k(g,r),r&&o(_),k($,r)}}}function Ot(s){let t,a,n,m,g,_,$,h,r,i,d,L,P,I,O,Te,U,xe,V,Pe,Ce,W,Re,q,Se,G,ze,$e,j,he,H,de,J,Le,ge,A,Ie,Z,Oe,Ae,ee,De,Fe,te,Me,Be,re,Ne,Ue,_e,K,we,M,Ve,ae,We,qe,ne,Ge,je,ye,Y,ve,B,He,oe,Ke,Ye,ie,Je,Qe,be,Q,Ee;return n=new Xe({props:{$$slots:{default:[kt]},$$scope:{ctx:s}}}),r=new nt({props:{$$slots:{default:[Tt]},$$scope:{ctx:s}}}),d=new it({props:{domain:[-8,8],range:[0,60],$$slots:{default:[xt]},$$scope:{ctx:s}}}),O=new Ze({props:{$$slots:{default:[Pt]},$$scope:{ctx:s}}}),U=new Ze({props:{$$slots:{default:[Ct]},$$scope:{ctx:s}}}),W=new Xe({props:{$$slots:{default:[Rt]},$$scope:{ctx:s}}}),q=new Xe({props:{$$slots:{default:[St]},$$scope:{ctx:s}}}),G=new Ze({props:{$$slots:{default:[zt]},$$scope:{ctx:s}}}),j=new nt({props:{$$slots:{default:[Lt]},$$scope:{ctx:s}}}),H=new it({props:{domain:[-8,8],range:[0,60],$$slots:{default:[It]},$$scope:{ctx:s}}}),K=new ot({props:{code:`model = Model().to(DEVICE)
criterion = nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.1)`}}),Y=new ot({props:{code:`def train(train_dataloader, val_dataloader, model, criterion, optimizer, scheduler):
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (features, labels) in enumerate(train_dataloader):
            # switch to training mode
            model.train()
            # move features and labels to GPU
            features = features.view(-1, NUM_FEATURES).to(DEVICE)
            labels = labels.to(DEVICE)

            # ------ FORWARD PASS --------
            probs = model(features)

            # ------CALCULATE LOSS --------
            loss = criterion(probs, labels)

            # ------BACKPROPAGATION --------
            loss.backward()

            # ------GRADIENT DESCENT --------
            optimizer.step()

            # ------CLEAR GRADIENTS --------
            optimizer.zero_grad()

        # ------TRACK LOSS --------
        train_loss, train_acc = track_performance(train_dataloader, model, criterion)
        val_loss, val_acc = track_performance(val_dataloader, model, criterion)
        
        # ------ADJUST LEARNING RATE --------
        scheduler.step(val_loss)`}}),{c(){t=R("p"),a=u(`There is probably no hyperparameter that is more important than the learning
    rate `),w(n.$$.fragment),m=u(`. If the learning rate is too high, we might
    overshood or oscilate. If the learning rate is too low, training might be
    too slow, or we might get stuck in some local minimum.`),g=T(),_=R("p"),$=u(`In the example below for example we pick a learning rate that is relatively
    large. The gradient descent algorithm (with momentum) overshoots and keeps
    oscilating for a while, before settling on the minimum.`),h=T(),w(r.$$.fragment),i=T(),w(d.$$.fragment),L=T(),P=R("p"),I=u(`It is possible, that a single constant rate is not the optimal solution.
    What if we start out with a relatively large learning rate to gain momentum
    at the beginning of trainig, but decrease the learning rate either over time
    or at specific events. In deep learning this is called `),w(O.$$.fragment),Te=u(" or "),w(U.$$.fragment),xe=u(`. There are dozens of
    different schedulers (see the
    `),V=R("a"),Pe=u("PyTorch documentation"),Ce=u(`
    for more info). You could for example decay the learing rate by subtracting a
    constant rate every `),w(W.$$.fragment),Re=u(` episodes. Or you could multiply the learning
    rate at the end of each epoch by a constant factor, for example
    `),w(q.$$.fragment),Se=u(`. Below we use a popular learning rate decay technique
    that is called `),w(G.$$.fragment),ze=u(`. Once
    a metric (like a loss) stops improving for certain amount of epochs we
    decrease the learning rate by a predetermined factor. Below we use this
    technique, which reduces the learning rate once the algorithm overshoots. It
    almost looks like the ball "glides" into the optimal value.`),$e=T(),w(j.$$.fragment),he=T(),w(H.$$.fragment),de=T(),J=R("p"),Le=u(`Deep learning frameworks like PyTorch or Keras make it extremely easy to
    create learning rate schedulers. Usually it involves no more than 2-3 lines
    of code.`),ge=T(),A=R("p"),Ie=u("Schedulers in PyTorch are located in "),Z=R("code"),Oe=u("otpim.lr_scheduler"),Ae=u(`, in our
    example we pick `),ee=R("code"),De=u("optim.lr_scheduler.ReduceLROnPlateau"),Fe=u(`. All
    schedulers take an optimizer as input. This is necessary because the
    learning rate is a part of an optimizer and the scheduler has to modify that
    paramter. The `),te=R("code"),Me=u("patience"),Be=u(` attribute is a ReduceLROnPlateau
    specific paramter that inidicates for how many epochs the performance metric
    (like cross-entropy) has to shrink in order to for the learning rate to be
    multiplied by the `),re=R("code"),Ne=u("factor"),Ue=u(" parameter of 0.1."),_e=T(),w(K.$$.fragment),we=T(),M=R("p"),Ve=u(`We have to adjust the train function to introduce the scheduler logic. This
    function might for example look as below. Similar to the `),ae=R("code"),We=u("optimizer.step()"),qe=u(`
    method there is a
    `),ne=R("code"),Ge=u("scheduler.step()"),je=u(` method. This function takes a performance measure
    lke the validation loss and adjusts the learning rate if necessary.`),ye=T(),w(Y.$$.fragment),ve=T(),B=R("p"),He=u(`There are no hard rules what scheduler you need to use in what situation,
    but when you use PyTorch you need to always call `),oe=R("code"),Ke=u("optimizer.step()"),Ye=u(`
    before you call `),ie=R("code"),Je=u("scheduler.step()"),Qe=u("."),be=T(),Q=R("div"),this.h()},l(e){t=S(e,"P",{});var l=C(t);a=c(l,`There is probably no hyperparameter that is more important than the learning
    rate `),y(n.$$.fragment,l),m=c(l,`. If the learning rate is too high, we might
    overshood or oscilate. If the learning rate is too low, training might be
    too slow, or we might get stuck in some local minimum.`),l.forEach(o),g=x(e),_=S(e,"P",{});var le=C(_);$=c(le,`In the example below for example we pick a learning rate that is relatively
    large. The gradient descent algorithm (with momentum) overshoots and keeps
    oscilating for a while, before settling on the minimum.`),le.forEach(o),h=x(e),y(r.$$.fragment,e),i=x(e),y(d.$$.fragment,e),L=x(e),P=S(e,"P",{});var z=C(P);I=c(z,`It is possible, that a single constant rate is not the optimal solution.
    What if we start out with a relatively large learning rate to gain momentum
    at the beginning of trainig, but decrease the learning rate either over time
    or at specific events. In deep learning this is called `),y(O.$$.fragment,z),Te=c(z," or "),y(U.$$.fragment,z),xe=c(z,`. There are dozens of
    different schedulers (see the
    `),V=S(z,"A",{href:!0,target:!0,rel:!0});var se=C(V);Pe=c(se,"PyTorch documentation"),se.forEach(o),Ce=c(z,`
    for more info). You could for example decay the learing rate by subtracting a
    constant rate every `),y(W.$$.fragment,z),Re=c(z,` episodes. Or you could multiply the learning
    rate at the end of each epoch by a constant factor, for example
    `),y(q.$$.fragment,z),Se=c(z,`. Below we use a popular learning rate decay technique
    that is called `),y(G.$$.fragment,z),ze=c(z,`. Once
    a metric (like a loss) stops improving for certain amount of epochs we
    decrease the learning rate by a predetermined factor. Below we use this
    technique, which reduces the learning rate once the algorithm overshoots. It
    almost looks like the ball "glides" into the optimal value.`),z.forEach(o),$e=x(e),y(j.$$.fragment,e),he=x(e),y(H.$$.fragment,e),de=x(e),J=S(e,"P",{});var fe=C(J);Le=c(fe,`Deep learning frameworks like PyTorch or Keras make it extremely easy to
    create learning rate schedulers. Usually it involves no more than 2-3 lines
    of code.`),fe.forEach(o),ge=x(e),A=S(e,"P",{});var F=C(A);Ie=c(F,"Schedulers in PyTorch are located in "),Z=S(F,"CODE",{});var me=C(Z);Oe=c(me,"otpim.lr_scheduler"),me.forEach(o),Ae=c(F,`, in our
    example we pick `),ee=S(F,"CODE",{});var ue=C(ee);De=c(ue,"optim.lr_scheduler.ReduceLROnPlateau"),ue.forEach(o),Fe=c(F,`. All
    schedulers take an optimizer as input. This is necessary because the
    learning rate is a part of an optimizer and the scheduler has to modify that
    paramter. The `),te=S(F,"CODE",{});var ce=C(te);Me=c(ce,"patience"),ce.forEach(o),Be=c(F,` attribute is a ReduceLROnPlateau
    specific paramter that inidicates for how many epochs the performance metric
    (like cross-entropy) has to shrink in order to for the learning rate to be
    multiplied by the `),re=S(F,"CODE",{});var pe=C(re);Ne=c(pe,"factor"),pe.forEach(o),Ue=c(F," parameter of 0.1."),F.forEach(o),_e=x(e),y(K.$$.fragment,e),we=x(e),M=S(e,"P",{});var N=C(M);Ve=c(N,`We have to adjust the train function to introduce the scheduler logic. This
    function might for example look as below. Similar to the `),ae=S(N,"CODE",{});var et=C(ae);We=c(et,"optimizer.step()"),et.forEach(o),qe=c(N,`
    method there is a
    `),ne=S(N,"CODE",{});var tt=C(ne);Ge=c(tt,"scheduler.step()"),tt.forEach(o),je=c(N,` method. This function takes a performance measure
    lke the validation loss and adjusts the learning rate if necessary.`),N.forEach(o),ye=x(e),y(Y.$$.fragment,e),ve=x(e),B=S(e,"P",{});var X=C(B);He=c(X,`There are no hard rules what scheduler you need to use in what situation,
    but when you use PyTorch you need to always call `),oe=S(X,"CODE",{});var rt=C(oe);Ke=c(rt,"optimizer.step()"),rt.forEach(o),Ye=c(X,`
    before you call `),ie=S(X,"CODE",{});var at=C(ie);Je=c(at,"scheduler.step()"),at.forEach(o),Qe=c(X,"."),X.forEach(o),be=x(e),Q=S(e,"DIV",{class:!0}),C(Q).forEach(o),this.h()},h(){D(V,"href","https://pytorch.org/docs/stable/optim.html"),D(V,"target","_blank"),D(V,"rel","noreferrer"),D(Q,"class","separator")},m(e,l){p(e,t,l),f(t,a),v(n,t,null),f(t,m),p(e,g,l),p(e,_,l),f(_,$),p(e,h,l),v(r,e,l),p(e,i,l),v(d,e,l),p(e,L,l),p(e,P,l),f(P,I),v(O,P,null),f(P,Te),v(U,P,null),f(P,xe),f(P,V),f(V,Pe),f(P,Ce),v(W,P,null),f(P,Re),v(q,P,null),f(P,Se),v(G,P,null),f(P,ze),p(e,$e,l),v(j,e,l),p(e,he,l),v(H,e,l),p(e,de,l),p(e,J,l),f(J,Le),p(e,ge,l),p(e,A,l),f(A,Ie),f(A,Z),f(Z,Oe),f(A,Ae),f(A,ee),f(ee,De),f(A,Fe),f(A,te),f(te,Me),f(A,Be),f(A,re),f(re,Ne),f(A,Ue),p(e,_e,l),v(K,e,l),p(e,we,l),p(e,M,l),f(M,Ve),f(M,ae),f(ae,We),f(M,qe),f(M,ne),f(ne,Ge),f(M,je),p(e,ye,l),v(Y,e,l),p(e,ve,l),p(e,B,l),f(B,He),f(B,oe),f(oe,Ke),f(B,Ye),f(B,ie),f(ie,Je),f(B,Qe),p(e,be,l),p(e,Q,l),Ee=!0},p(e,l){const le={};l&4096&&(le.$$scope={dirty:l,ctx:e}),n.$set(le);const z={};l&4096&&(z.$$scope={dirty:l,ctx:e}),r.$set(z);const se={};l&4113&&(se.$$scope={dirty:l,ctx:e}),d.$set(se);const fe={};l&4096&&(fe.$$scope={dirty:l,ctx:e}),O.$set(fe);const F={};l&4096&&(F.$$scope={dirty:l,ctx:e}),U.$set(F);const me={};l&4096&&(me.$$scope={dirty:l,ctx:e}),W.$set(me);const ue={};l&4096&&(ue.$$scope={dirty:l,ctx:e}),q.$set(ue);const ce={};l&4096&&(ce.$$scope={dirty:l,ctx:e}),G.$set(ce);const pe={};l&4096&&(pe.$$scope={dirty:l,ctx:e}),j.$set(pe);const N={};l&4110&&(N.$$scope={dirty:l,ctx:e}),H.$set(N)},i(e){Ee||(b(n.$$.fragment,e),b(r.$$.fragment,e),b(d.$$.fragment,e),b(O.$$.fragment,e),b(U.$$.fragment,e),b(W.$$.fragment,e),b(q.$$.fragment,e),b(G.$$.fragment,e),b(j.$$.fragment,e),b(H.$$.fragment,e),b(K.$$.fragment,e),b(Y.$$.fragment,e),Ee=!0)},o(e){E(n.$$.fragment,e),E(r.$$.fragment,e),E(d.$$.fragment,e),E(O.$$.fragment,e),E(U.$$.fragment,e),E(W.$$.fragment,e),E(q.$$.fragment,e),E(G.$$.fragment,e),E(j.$$.fragment,e),E(H.$$.fragment,e),E(K.$$.fragment,e),E(Y.$$.fragment,e),Ee=!1},d(e){e&&o(t),k(n),e&&o(g),e&&o(_),e&&o(h),k(r,e),e&&o(i),k(d,e),e&&o(L),e&&o(P),k(O),k(U),k(W),k(q),k(G),e&&o($e),k(j,e),e&&o(he),k(H,e),e&&o(de),e&&o(J),e&&o(ge),e&&o(A),e&&o(_e),k(K,e),e&&o(we),e&&o(M),e&&o(ye),k(Y,e),e&&o(ve),e&&o(B),e&&o(be),e&&o(Q)}}}function At(s){let t,a,n,m,g,_,$,h,r;return h=new vt({props:{$$slots:{default:[Ot]},$$scope:{ctx:s}}}),{c(){t=R("meta"),a=T(),n=R("h1"),m=u("Learning Rate Scheduling"),g=T(),_=R("div"),$=T(),w(h.$$.fragment),this.h()},l(i){const d=yt("svelte-gcga6n",document.head);t=S(d,"META",{name:!0,content:!0}),d.forEach(o),a=x(i),n=S(i,"H1",{});var L=C(n);m=c(L,"Learning Rate Scheduling"),L.forEach(o),g=x(i),_=S(i,"DIV",{class:!0}),C(_).forEach(o),$=x(i),y(h.$$.fragment,i),this.h()},h(){document.title="Learning Rate Scheduling - World4AI",D(t,"name","description"),D(t,"content","It is not always an easy task to finetune the learning rate. Learning rate schedulers are a common way to otpimize this hyperparameter, but changing the learning rate over time."),D(_,"class","separator")},m(i,d){f(document.head,t),p(i,a,d),p(i,n,d),f(n,m),p(i,g,d),p(i,_,d),p(i,$,d),v(h,i,d),r=!0},p(i,[d]){const L={};d&4127&&(L.$$scope={dirty:d,ctx:i}),h.$set(L)},i(i){r||(b(h.$$.fragment,i),r=!0)},o(i){E(h.$$.fragment,i),r=!1},d(i){o(t),i&&o(a),i&&o(n),i&&o(g),i&&o(_),i&&o($),k(h,i)}}}let ht=.1;function Dt(s,t,a){let n,m,g=[];for(let I=-8;I<=8;I+=.1)g.push({x:I,y:I**2});let _=8,$=0;function h(){let I=.95,O=2*_;$===0&&($=O),$=$*I+O*(1-I),a(0,_-=ht*$)}let r=8,i=.1,d=0,L=m;function P(){r**2>L&&a(2,i*=.88),L=m;let I=.95,O=2*r;d===0&&(d=O),d=d*I+O*(1-I),a(1,r-=i*d)}return s.$$.update=()=>{s.$$.dirty&1&&a(4,n=_**2),s.$$.dirty&2&&a(3,m=r**2)},[_,r,i,m,n,g,h,P]}class Ht extends lt{constructor(t){super(),st(this,t,Dt,At,ft,{})}}export{Ht as default};
