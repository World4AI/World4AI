import{S as Xt,i as Yt,s as Zt,k as c,a as l,q as s,y as w,W as er,l as m,h as t,c as n,m as p,r as a,z as _,n as D,N as o,b as i,A as E,g as y,d as v,B as g}from"../chunks/index.4d92b023.js";import{C as tr}from"../chunks/Container.b0705c7b.js";import{P as x}from"../chunks/PythonCode.212ba7a6.js";import{H as Qt}from"../chunks/Highlight.b7c1de53.js";import{A as rr}from"../chunks/Alert.25a852b3.js";function or(z){let f;return{c(){f=s("mixed precision traing")},l(h){f=a(h,"mixed precision traing")},m(h,u){i(h,f,u)},d(h){h&&t(f)}}}function ir(z){let f;return{c(){f=s(`Mixed precision training allows us to train a neural network utilizing
		different levels of precision for different layers.`)},l(h){f=a(h,`Mixed precision training allows us to train a neural network utilizing
		different levels of precision for different layers.`)},m(h,u){i(h,f,u)},d(h){h&&t(f)}}}function sr(z){let f;return{c(){f=s("automatic mixed precision")},l(h){f=a(h,"automatic mixed precision")},m(h,u){i(h,f,u)},d(h){h&&t(f)}}}function ar(z){let f,h,u,F,I,b,q,T,W,d,$,P,rt,Le,O,ke,ie,ot,Ce,M,N,it,de,st,at,lt,U,nt,he,ft,ct,ze,k,mt,A,pt,V,dt,ht,Ie,se,ut,Pe,H,Me,R,Se,j,De,G,qe,ae,$t,We,J,Oe,K,Ne,le,wt,Ae,Q,Ve,X,Be,ne,_t,Fe,Y,Ue,Z,He,L,Et,ue,yt,vt,$e,gt,bt,we,Tt,xt,Re,C,_e,Ee,Lt,kt,ye,ve,Ct,zt,ge,be,It,je,ee,Ge,te,Je,fe,Pt,Ke,re,Qe,oe,Xe,ce,Ye;return u=new Qt({props:{$$slots:{default:[or]},$$scope:{ctx:z}}}),O=new rr({props:{type:"info",$$slots:{default:[ir]},$$scope:{ctx:z}}}),A=new Qt({props:{$$slots:{default:[sr]},$$scope:{ctx:z}}}),H=new x({props:{code:`import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms as T

import time`}}),R=new x({props:{code:"assert torch.cuda.is_available()"}}),j=new x({props:{code:'train_dataset = MNIST(root="../datasets", train=True, download=True, transform=T.ToTensor())'}}),G=new x({props:{code:`train_dataloader=DataLoader(dataset=train_dataset, 
                            batch_size=256, 
                            shuffle=True, 
                            drop_last=True,
                            num_workers=2)`}}),J=new x({props:{code:`cfg = [[1, 32, 3, 1, 1],
       [32, 64, 3, 1, 1],
       [64, 64, 2, 2, 0],
       [64, 128, 3, 1, 1],
       [128, 128, 3, 1, 1],
       [128, 128, 3, 1, 1],
       [128, 128, 2, 2, 0],
       [128, 256, 3, 1, 1],
       [256, 256, 2, 1, 0],
       [256, 512, 3, 1, 1],
       [512, 512, 3, 1, 1],
       [512, 512, 3, 1, 1],
       [512, 512, 2, 2, 0],
       [512, 1024, 3, 1, 1],
]

class BasicBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(**kwargs),
            nn.BatchNorm2d(num_features=kwargs['out_channels']),
            nn.ReLU()
        )
  
    def forward(self, x):
        return self.block(x)

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.features = self._build_layers(cfg)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=10),
        )
  
    def _build_layers(self, cfg):
        layers = []
        for layer in cfg:
            layers += [BasicBlock(in_channels=layer[0],
                                   out_channels=layer[1],
                                   kernel_size=layer[2],
                                   stride=layer[3],
                                   padding=layer[4])]
        return nn.Sequential(*layers)
  
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x`}}),K=new x({props:{code:`NUM_EPOCHS=10
LR=0.0001
DEVICE = torch.device('cuda')`}}),Q=new x({props:{code:String.raw`def train(data_loader, model, optimizer, criterion):
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        losses = []
        for img, label in data_loader:
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            prediction = model(img)
            loss = criterion(prediction, label)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        end_time = time.time()
        s = f'Epoch: {epoch+1}, ' \
          f'Loss: {sum(losses)/len(losses):.4f}, ' \
          f'Elapsed Time: {end_time-start_time:.2f}sec'
        print(s)`}}),X=new x({props:{code:`model = Model(cfg)
model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)`}}),Y=new x({props:{code:"train(train_dataloader, model, optimizer, criterion)"}}),Z=new x({props:{isOutput:!0,code:`Epoch: 1, Loss: 0.2528, Elapsed Time: 22.82sec
Epoch: 2, Loss: 0.0316, Elapsed Time: 21.99sec
Epoch: 3, Loss: 0.0201, Elapsed Time: 22.11sec
Epoch: 4, Loss: 0.0155, Elapsed Time: 22.15sec
Epoch: 5, Loss: 0.0123, Elapsed Time: 22.14sec
Epoch: 6, Loss: 0.0106, Elapsed Time: 22.18sec
Epoch: 7, Loss: 0.0112, Elapsed Time: 22.11sec
Epoch: 8, Loss: 0.0084, Elapsed Time: 22.15sec
Epoch: 9, Loss: 0.0083, Elapsed Time: 22.17sec
Epoch: 10, Loss: 0.0078, Elapsed Time: 22.14sec`}}),ee=new x({props:{code:String.raw`def optimized_train(data_loader, model, optimizer, criterion):
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        losses = []
        for img, label in data_loader:
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(img)
                loss = criterion(prediction, label)
            losses.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        end_time = time.time()
        s = f'Epoch: {epoch+1}, ' \
          f'Loss: {sum(losses)/len(losses):.4f}, ' \
          f'Elapsed Time: {end_time-start_time:.2f}sec'
        print(s)`}}),te=new x({props:{code:`model = Model(cfg)
model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)`}}),re=new x({props:{code:"optimized_train(train_dataloader, model, optimizer, criterion)"}}),oe=new x({props:{isOutput:!0,code:`Epoch: 1, Loss: 0.2699, Elapsed Time: 13.00sec
Epoch: 2, Loss: 0.0319, Elapsed Time: 12.95sec
Epoch: 3, Loss: 0.0206, Elapsed Time: 12.93sec
Epoch: 4, Loss: 0.0144, Elapsed Time: 12.95sec
Epoch: 5, Loss: 0.0117, Elapsed Time: 12.95sec
Epoch: 6, Loss: 0.0104, Elapsed Time: 12.96sec
Epoch: 7, Loss: 0.0083, Elapsed Time: 12.95sec
Epoch: 8, Loss: 0.0095, Elapsed Time: 13.01sec
Epoch: 9, Loss: 0.0053, Elapsed Time: 12.97sec
Epoch: 10, Loss: 0.0091, Elapsed Time: 12.99sec`}}),{c(){f=c("p"),h=s(`In the next section we will begin looking at different CNN architectures.
		While the older architectures are relatively easy to train, more modern
		architectures require a lot of computational power. There are different ways
		to deal with those requirements, but in this section we will specifically
		focus on `),w(u.$$.fragment),F=s("."),I=l(),b=c("p"),q=s("So far when we trained neural networks, we utilized the "),T=c("code"),W=s("torch.float32"),d=s(`
		datatype. But there are layers, like linear layers and convolutions, that can
		be executed much faster using the lower
		`),$=c("code"),P=s("torch.float16"),rt=s(" precision."),Le=l(),w(O.$$.fragment),ke=l(),ie=c("p"),ot=s("Mixed precision training has at least two advantages."),Ce=l(),M=c("ol"),N=c("li"),it=s("Some layers are faster with "),de=c("code"),st=s("torch.float16"),at=s(` precision, therefore
			the whole training process will be significantly faster`),lt=l(),U=c("li"),nt=s("Operations using "),he=c("code"),ft=s("torch.float16"),ct=s(" require less memory than `torch.float32`\n			operations. That will reduce the necessary vram requirements and will allow\n			us to use a larger batch size."),ze=l(),k=c("p"),mt=s("PyTorch provides a so called "),w(A.$$.fragment),pt=s(` functionality, that automatically decides which of the operations will run
		with which precision. We do not have to make any of those decisions manually.
		The official PyTorch
		`),V=c("a"),dt=s("documentation"),ht=s(" provides more info on the topic."),Ie=l(),se=c("p"),ut=s(`We will demonstrate the performance boost from mixed precision training with
		the help of the MNIST dataset.`),Pe=l(),w(H.$$.fragment),Me=l(),w(R.$$.fragment),Se=l(),w(j.$$.fragment),De=l(),w(G.$$.fragment),qe=l(),ae=c("p"),$t=s(`We use a much larger network, than what is required to get a good
		performance for MINST in order to demonstrate the potential of mixed
		precision training.`),We=l(),w(J.$$.fragment),Oe=l(),w(K.$$.fragment),Ne=l(),le=c("p"),wt=s(`We start by training the neural network in a familiar manner, measuring the
		time an epoch takes. We can use those values as a benchmark.`),Ae=l(),w(Q.$$.fragment),Ve=l(),w(X.$$.fragment),Be=l(),ne=c("p"),_t=s("Each epoch takes slightly over 20 seconds to complete."),Fe=l(),w(Y.$$.fragment),Ue=l(),w(Z.$$.fragment),He=l(),L=c("p"),Et=s(`We repeat the training procedure, only this time we use mixed precision
		training. For that we will utilize the `),ue=c("code"),yt=s("torch.amp"),vt=s(` module. The
		`),$e=c("code"),gt=s("torch.amp.autocast"),bt=s(`
		context manager runs the region below the context manager in mixed precision.
		For our purposes the forward pass and the loss are calculated using mixed precision.
		We use
		`),we=c("code"),Tt=s("torch.cuda.amp.GradScalar"),xt=s(` object in order to scale the gradients
		of the loss. If the forward pass of a layer uses 16 bit precision, so will the
		backward pass. For some of the calculations the gradients will be relatively
		small and the precision of torch.float16 will not be sufficient to hold those
		small values. The values might therefore underflow. In order to remedy the problem,
		the loss is scaled and we let the scaler deal with backprop and gradient descent.
		At the end we reset the scaler object for the next batch. The three lines from
		below do exactly that.`),Re=l(),C=c("ul"),_e=c("li"),Ee=c("code"),Lt=s("scaler.scale(loss).backward()"),kt=l(),ye=c("li"),ve=c("code"),Ct=s("scaler.step(optimizer)"),zt=l(),ge=c("li"),be=c("code"),It=s("scaler.update()"),je=l(),w(ee.$$.fragment),Ge=l(),w(te.$$.fragment),Je=l(),fe=c("p"),Pt=s(`We improve the training speed significantly. The overhead to use automatic
		mixed precision is inconsequential when compared to the benefits.`),Ke=l(),w(re.$$.fragment),Qe=l(),w(oe.$$.fragment),Xe=l(),ce=c("div"),this.h()},l(e){f=m(e,"P",{});var r=p(f);h=a(r,`In the next section we will begin looking at different CNN architectures.
		While the older architectures are relatively easy to train, more modern
		architectures require a lot of computational power. There are different ways
		to deal with those requirements, but in this section we will specifically
		focus on `),_(u.$$.fragment,r),F=a(r,"."),r.forEach(t),I=n(e),b=m(e,"P",{});var S=p(b);q=a(S,"So far when we trained neural networks, we utilized the "),T=m(S,"CODE",{});var Te=p(T);W=a(Te,"torch.float32"),Te.forEach(t),d=a(S,`
		datatype. But there are layers, like linear layers and convolutions, that can
		be executed much faster using the lower
		`),$=m(S,"CODE",{});var xe=p($);P=a(xe,"torch.float16"),xe.forEach(t),rt=a(S," precision."),S.forEach(t),Le=n(e),_(O.$$.fragment,e),ke=n(e),ie=m(e,"P",{});var Mt=p(ie);ot=a(Mt,"Mixed precision training has at least two advantages."),Mt.forEach(t),Ce=n(e),M=m(e,"OL",{class:!0});var Ze=p(M);N=m(Ze,"LI",{class:!0});var et=p(N);it=a(et,"Some layers are faster with "),de=m(et,"CODE",{});var St=p(de);st=a(St,"torch.float16"),St.forEach(t),at=a(et,` precision, therefore
			the whole training process will be significantly faster`),et.forEach(t),lt=n(Ze),U=m(Ze,"LI",{});var tt=p(U);nt=a(tt,"Operations using "),he=m(tt,"CODE",{});var Dt=p(he);ft=a(Dt,"torch.float16"),Dt.forEach(t),ct=a(tt," require less memory than `torch.float32`\n			operations. That will reduce the necessary vram requirements and will allow\n			us to use a larger batch size."),tt.forEach(t),Ze.forEach(t),ze=n(e),k=m(e,"P",{});var me=p(k);mt=a(me,"PyTorch provides a so called "),_(A.$$.fragment,me),pt=a(me,` functionality, that automatically decides which of the operations will run
		with which precision. We do not have to make any of those decisions manually.
		The official PyTorch
		`),V=m(me,"A",{href:!0,target:!0,rel:!0});var qt=p(V);dt=a(qt,"documentation"),qt.forEach(t),ht=a(me," provides more info on the topic."),me.forEach(t),Ie=n(e),se=m(e,"P",{});var Wt=p(se);ut=a(Wt,`We will demonstrate the performance boost from mixed precision training with
		the help of the MNIST dataset.`),Wt.forEach(t),Pe=n(e),_(H.$$.fragment,e),Me=n(e),_(R.$$.fragment,e),Se=n(e),_(j.$$.fragment,e),De=n(e),_(G.$$.fragment,e),qe=n(e),ae=m(e,"P",{});var Ot=p(ae);$t=a(Ot,`We use a much larger network, than what is required to get a good
		performance for MINST in order to demonstrate the potential of mixed
		precision training.`),Ot.forEach(t),We=n(e),_(J.$$.fragment,e),Oe=n(e),_(K.$$.fragment,e),Ne=n(e),le=m(e,"P",{});var Nt=p(le);wt=a(Nt,`We start by training the neural network in a familiar manner, measuring the
		time an epoch takes. We can use those values as a benchmark.`),Nt.forEach(t),Ae=n(e),_(Q.$$.fragment,e),Ve=n(e),_(X.$$.fragment,e),Be=n(e),ne=m(e,"P",{});var At=p(ne);_t=a(At,"Each epoch takes slightly over 20 seconds to complete."),At.forEach(t),Fe=n(e),_(Y.$$.fragment,e),Ue=n(e),_(Z.$$.fragment,e),He=n(e),L=m(e,"P",{});var B=p(L);Et=a(B,`We repeat the training procedure, only this time we use mixed precision
		training. For that we will utilize the `),ue=m(B,"CODE",{});var Vt=p(ue);yt=a(Vt,"torch.amp"),Vt.forEach(t),vt=a(B,` module. The
		`),$e=m(B,"CODE",{});var Bt=p($e);gt=a(Bt,"torch.amp.autocast"),Bt.forEach(t),bt=a(B,`
		context manager runs the region below the context manager in mixed precision.
		For our purposes the forward pass and the loss are calculated using mixed precision.
		We use
		`),we=m(B,"CODE",{});var Ft=p(we);Tt=a(Ft,"torch.cuda.amp.GradScalar"),Ft.forEach(t),xt=a(B,` object in order to scale the gradients
		of the loss. If the forward pass of a layer uses 16 bit precision, so will the
		backward pass. For some of the calculations the gradients will be relatively
		small and the precision of torch.float16 will not be sufficient to hold those
		small values. The values might therefore underflow. In order to remedy the problem,
		the loss is scaled and we let the scaler deal with backprop and gradient descent.
		At the end we reset the scaler object for the next batch. The three lines from
		below do exactly that.`),B.forEach(t),Re=n(e),C=m(e,"UL",{});var pe=p(C);_e=m(pe,"LI",{});var Ut=p(_e);Ee=m(Ut,"CODE",{});var Ht=p(Ee);Lt=a(Ht,"scaler.scale(loss).backward()"),Ht.forEach(t),Ut.forEach(t),kt=n(pe),ye=m(pe,"LI",{});var Rt=p(ye);ve=m(Rt,"CODE",{});var jt=p(ve);Ct=a(jt,"scaler.step(optimizer)"),jt.forEach(t),Rt.forEach(t),zt=n(pe),ge=m(pe,"LI",{});var Gt=p(ge);be=m(Gt,"CODE",{});var Jt=p(be);It=a(Jt,"scaler.update()"),Jt.forEach(t),Gt.forEach(t),pe.forEach(t),je=n(e),_(ee.$$.fragment,e),Ge=n(e),_(te.$$.fragment,e),Je=n(e),fe=m(e,"P",{});var Kt=p(fe);Pt=a(Kt,`We improve the training speed significantly. The overhead to use automatic
		mixed precision is inconsequential when compared to the benefits.`),Kt.forEach(t),Ke=n(e),_(re.$$.fragment,e),Qe=n(e),_(oe.$$.fragment,e),Xe=n(e),ce=m(e,"DIV",{class:!0}),p(ce).forEach(t),this.h()},h(){D(N,"class","mb-2"),D(M,"class","list-decimal list-inside"),D(V,"href","https://pytorch.org/docs/stable/amp.html"),D(V,"target","_blank"),D(V,"rel","noreferrer"),D(ce,"class","separator")},m(e,r){i(e,f,r),o(f,h),E(u,f,null),o(f,F),i(e,I,r),i(e,b,r),o(b,q),o(b,T),o(T,W),o(b,d),o(b,$),o($,P),o(b,rt),i(e,Le,r),E(O,e,r),i(e,ke,r),i(e,ie,r),o(ie,ot),i(e,Ce,r),i(e,M,r),o(M,N),o(N,it),o(N,de),o(de,st),o(N,at),o(M,lt),o(M,U),o(U,nt),o(U,he),o(he,ft),o(U,ct),i(e,ze,r),i(e,k,r),o(k,mt),E(A,k,null),o(k,pt),o(k,V),o(V,dt),o(k,ht),i(e,Ie,r),i(e,se,r),o(se,ut),i(e,Pe,r),E(H,e,r),i(e,Me,r),E(R,e,r),i(e,Se,r),E(j,e,r),i(e,De,r),E(G,e,r),i(e,qe,r),i(e,ae,r),o(ae,$t),i(e,We,r),E(J,e,r),i(e,Oe,r),E(K,e,r),i(e,Ne,r),i(e,le,r),o(le,wt),i(e,Ae,r),E(Q,e,r),i(e,Ve,r),E(X,e,r),i(e,Be,r),i(e,ne,r),o(ne,_t),i(e,Fe,r),E(Y,e,r),i(e,Ue,r),E(Z,e,r),i(e,He,r),i(e,L,r),o(L,Et),o(L,ue),o(ue,yt),o(L,vt),o(L,$e),o($e,gt),o(L,bt),o(L,we),o(we,Tt),o(L,xt),i(e,Re,r),i(e,C,r),o(C,_e),o(_e,Ee),o(Ee,Lt),o(C,kt),o(C,ye),o(ye,ve),o(ve,Ct),o(C,zt),o(C,ge),o(ge,be),o(be,It),i(e,je,r),E(ee,e,r),i(e,Ge,r),E(te,e,r),i(e,Je,r),i(e,fe,r),o(fe,Pt),i(e,Ke,r),E(re,e,r),i(e,Qe,r),E(oe,e,r),i(e,Xe,r),i(e,ce,r),Ye=!0},p(e,r){const S={};r&1&&(S.$$scope={dirty:r,ctx:e}),u.$set(S);const Te={};r&1&&(Te.$$scope={dirty:r,ctx:e}),O.$set(Te);const xe={};r&1&&(xe.$$scope={dirty:r,ctx:e}),A.$set(xe)},i(e){Ye||(y(u.$$.fragment,e),y(O.$$.fragment,e),y(A.$$.fragment,e),y(H.$$.fragment,e),y(R.$$.fragment,e),y(j.$$.fragment,e),y(G.$$.fragment,e),y(J.$$.fragment,e),y(K.$$.fragment,e),y(Q.$$.fragment,e),y(X.$$.fragment,e),y(Y.$$.fragment,e),y(Z.$$.fragment,e),y(ee.$$.fragment,e),y(te.$$.fragment,e),y(re.$$.fragment,e),y(oe.$$.fragment,e),Ye=!0)},o(e){v(u.$$.fragment,e),v(O.$$.fragment,e),v(A.$$.fragment,e),v(H.$$.fragment,e),v(R.$$.fragment,e),v(j.$$.fragment,e),v(G.$$.fragment,e),v(J.$$.fragment,e),v(K.$$.fragment,e),v(Q.$$.fragment,e),v(X.$$.fragment,e),v(Y.$$.fragment,e),v(Z.$$.fragment,e),v(ee.$$.fragment,e),v(te.$$.fragment,e),v(re.$$.fragment,e),v(oe.$$.fragment,e),Ye=!1},d(e){e&&t(f),g(u),e&&t(I),e&&t(b),e&&t(Le),g(O,e),e&&t(ke),e&&t(ie),e&&t(Ce),e&&t(M),e&&t(ze),e&&t(k),g(A),e&&t(Ie),e&&t(se),e&&t(Pe),g(H,e),e&&t(Me),g(R,e),e&&t(Se),g(j,e),e&&t(De),g(G,e),e&&t(qe),e&&t(ae),e&&t(We),g(J,e),e&&t(Oe),g(K,e),e&&t(Ne),e&&t(le),e&&t(Ae),g(Q,e),e&&t(Ve),g(X,e),e&&t(Be),e&&t(ne),e&&t(Fe),g(Y,e),e&&t(Ue),g(Z,e),e&&t(He),e&&t(L),e&&t(Re),e&&t(C),e&&t(je),g(ee,e),e&&t(Ge),g(te,e),e&&t(Je),e&&t(fe),e&&t(Ke),g(re,e),e&&t(Qe),g(oe,e),e&&t(Xe),e&&t(ce)}}}function lr(z){let f,h,u,F,I,b,q,T,W;return T=new tr({props:{$$slots:{default:[ar]},$$scope:{ctx:z}}}),{c(){f=c("meta"),h=l(),u=c("h1"),F=s("Mixed Precision Training"),I=l(),b=c("div"),q=l(),w(T.$$.fragment),this.h()},l(d){const $=er("svelte-1qihqt7",document.head);f=m($,"META",{name:!0,content:!0}),$.forEach(t),h=n(d),u=m(d,"H1",{});var P=p(u);F=a(P,"Mixed Precision Training"),P.forEach(t),I=n(d),b=m(d,"DIV",{class:!0}),p(b).forEach(t),q=n(d),_(T.$$.fragment,d),this.h()},h(){document.title="Mixed Precision Training - World4AI",D(f,"name","description"),D(f,"content","Mixed precision training is a technique that allows deep learning researchers to train neural networks using either 32 or 16 bit precision. This allows some layers to train faster and to reduce the memory footprint of large neural networks. PyTorch allows us to use so called automatic mixed precision, which reduces the code overhead significantly."),D(b,"class","separator")},m(d,$){o(document.head,f),i(d,h,$),i(d,u,$),o(u,F),i(d,I,$),i(d,b,$),i(d,q,$),E(T,d,$),W=!0},p(d,[$]){const P={};$&1&&(P.$$scope={dirty:$,ctx:d}),T.$set(P)},i(d){W||(y(T.$$.fragment,d),W=!0)},o(d){v(T.$$.fragment,d),W=!1},d(d){t(f),d&&t(h),d&&t(u),d&&t(I),d&&t(b),d&&t(q),g(T,d)}}}class dr extends Xt{constructor(f){super(),Yt(this,f,null,lr,Zt,{})}}export{dr as default};
