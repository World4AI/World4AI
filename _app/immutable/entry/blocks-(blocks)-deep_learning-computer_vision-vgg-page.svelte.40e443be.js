import{S as mt,i as $t,s as ht,k,a as d,q as y,y as g,W as dt,l as A,h as c,c as _,m as L,r as G,z as x,n as ne,N as E,b as u,A as v,g as m,d as h,B as V,Q as _t,R as gt,C as me,e as se,v as Je,f as Xe,P as Ye}from"../chunks/index.4d92b023.js";import{C as xt}from"../chunks/Container.b0705c7b.js";import{F as vt,I as Vt}from"../chunks/InternalLink.7deb899c.js";import{S as wt}from"../chunks/SvgContainer.f70b5745.js";import{H as ot}from"../chunks/Highlight.b7c1de53.js";import{P as M}from"../chunks/PythonCode.212ba7a6.js";import{T as yt,a as Gt,b as bt,R as pt,H as Et,D as kt}from"../chunks/HeaderEntry.2b6e8f51.js";import{B as Ke}from"../chunks/Block.059eddcd.js";import{A as lt}from"../chunks/Arrow.ae91874c.js";function nt(p,a,n){const r=p.slice();return r[3]=a[n],r}function st(p,a,n){const r=p.slice();return r[6]=a[n],r}function it(p,a,n){const r=p.slice();return r[9]=a[n],r}function At(p){let a;return{c(){a=y("VGG")},l(n){a=G(n,"VGG")},m(n,r){u(n,a,r)},d(n){n&&c(a)}}}function Lt(p){let a,n,r,t,s,o,i;return n=new Ke({props:{x:I/2,y:le-z/2-Qe,width:Ze,height:z,text:"Conv2d: 3x3, S:1, P:1"}}),r=new Ke({props:{x:I/2,y:le/2,width:Ze,height:z,text:"BatchNorm2d"}}),t=new Ke({props:{x:I/2,y:z/2+Qe,width:Ze,height:z,text:"ReLU"}}),s=new lt({props:{data:[{x:I/2,y:le-z-Qe},{x:I/2,y:le/2+z/2+3}],dashed:!0,moving:!0}}),o=new lt({props:{data:[{x:I/2,y:le/2-z/2},{x:I/2,y:z+4}],dashed:!0,moving:!0}}),{c(){a=_t("svg"),g(n.$$.fragment),g(r.$$.fragment),g(t.$$.fragment),g(s.$$.fragment),g(o.$$.fragment),this.h()},l(l){a=gt(l,"svg",{viewBox:!0});var w=L(a);x(n.$$.fragment,w),x(r.$$.fragment,w),x(t.$$.fragment,w),x(s.$$.fragment,w),x(o.$$.fragment,w),w.forEach(c),this.h()},h(){ne(a,"viewBox","0 0 "+I+" "+le)},m(l,w){u(l,a,w),v(n,a,null),v(r,a,null),v(t,a,null),v(s,a,null),v(o,a,null),i=!0},p:me,i(l){i||(m(n.$$.fragment,l),m(r.$$.fragment,l),m(t.$$.fragment,l),m(s.$$.fragment,l),m(o.$$.fragment,l),i=!0)},o(l){h(n.$$.fragment,l),h(r.$$.fragment,l),h(t.$$.fragment,l),h(s.$$.fragment,l),h(o.$$.fragment,l),i=!1},d(l){l&&c(a),V(n),V(r),V(t),V(s),V(o)}}}function Dt(p){let a=p[9]+"",n;return{c(){n=y(a)},l(r){n=G(r,a)},m(r,t){u(r,n,t)},p:me,d(r){r&&c(n)}}}function ct(p){let a,n;return a=new Et({props:{$$slots:{default:[Dt]},$$scope:{ctx:p}}}),{c(){g(a.$$.fragment)},l(r){x(a.$$.fragment,r)},m(r,t){v(a,r,t),n=!0},p(r,t){const s={};t&4096&&(s.$$scope={dirty:t,ctx:r}),a.$set(s)},i(r){n||(m(a.$$.fragment,r),n=!0)},o(r){h(a.$$.fragment,r),n=!1},d(r){V(a,r)}}}function Mt(p){let a,n,r=p[0],t=[];for(let o=0;o<r.length;o+=1)t[o]=ct(it(p,r,o));const s=o=>h(t[o],1,1,()=>{t[o]=null});return{c(){for(let o=0;o<t.length;o+=1)t[o].c();a=se()},l(o){for(let i=0;i<t.length;i+=1)t[i].l(o);a=se()},m(o,i){for(let l=0;l<t.length;l+=1)t[l]&&t[l].m(o,i);u(o,a,i),n=!0},p(o,i){if(i&1){r=o[0];let l;for(l=0;l<r.length;l+=1){const w=it(o,r,l);t[l]?(t[l].p(w,i),m(t[l],1)):(t[l]=ct(w),t[l].c(),m(t[l],1),t[l].m(a.parentNode,a))}for(Je(),l=r.length;l<t.length;l+=1)s(l);Xe()}},i(o){if(!n){for(let i=0;i<r.length;i+=1)m(t[i]);n=!0}},o(o){t=t.filter(Boolean);for(let i=0;i<t.length;i+=1)h(t[i]);n=!1},d(o){Ye(t,o),o&&c(a)}}}function Tt(p){let a,n;return a=new pt({props:{$$slots:{default:[Mt]},$$scope:{ctx:p}}}),{c(){g(a.$$.fragment)},l(r){x(a.$$.fragment,r)},m(r,t){v(a,r,t),n=!0},p(r,t){const s={};t&4096&&(s.$$scope={dirty:t,ctx:r}),a.$set(s)},i(r){n||(m(a.$$.fragment,r),n=!0)},o(r){h(a.$$.fragment,r),n=!1},d(r){V(a,r)}}}function zt(p){let a=p[6]+"",n;return{c(){n=y(a)},l(r){n=G(r,a)},m(r,t){u(r,n,t)},p:me,d(r){r&&c(n)}}}function Pt(p){let a,n=p[6]+"",r;return{c(){a=k("span"),r=y(n),this.h()},l(t){a=A(t,"SPAN",{class:!0});var s=L(a);r=G(s,n),s.forEach(c),this.h()},h(){ne(a,"class","inline-block bg-slate-200 px-3 py-1 rounded-full")},m(t,s){u(t,a,s),E(a,r)},p:me,d(t){t&&c(a)}}}function Nt(p){let a,n=p[6]+"",r;return{c(){a=k("span"),r=y(n),this.h()},l(t){a=A(t,"SPAN",{class:!0});var s=L(a);r=G(s,n),s.forEach(c),this.h()},h(){ne(a,"class","inline-block bg-red-100 px-3 py-1 rounded-full")},m(t,s){u(t,a,s),E(a,r)},p:me,d(t){t&&c(a)}}}function Bt(p){let a;function n(s,o){return s[6]==="VGG Module"?Nt:s[6]==="Max Pooling"?Pt:zt}let t=n(p)(p);return{c(){t.c(),a=se()},l(s){t.l(s),a=se()},m(s,o){t.m(s,o),u(s,a,o)},p(s,o){t.p(s,o)},d(s){t.d(s),s&&c(a)}}}function ft(p){let a,n;return a=new kt({props:{$$slots:{default:[Bt]},$$scope:{ctx:p}}}),{c(){g(a.$$.fragment)},l(r){x(a.$$.fragment,r)},m(r,t){v(a,r,t),n=!0},p(r,t){const s={};t&4096&&(s.$$scope={dirty:t,ctx:r}),a.$set(s)},i(r){n||(m(a.$$.fragment,r),n=!0)},o(r){h(a.$$.fragment,r),n=!1},d(r){V(a,r)}}}function Ct(p){let a,n,r=p[3],t=[];for(let o=0;o<r.length;o+=1)t[o]=ft(st(p,r,o));const s=o=>h(t[o],1,1,()=>{t[o]=null});return{c(){for(let o=0;o<t.length;o+=1)t[o].c();a=d()},l(o){for(let i=0;i<t.length;i+=1)t[i].l(o);a=_(o)},m(o,i){for(let l=0;l<t.length;l+=1)t[l]&&t[l].m(o,i);u(o,a,i),n=!0},p(o,i){if(i&2){r=o[3];let l;for(l=0;l<r.length;l+=1){const w=st(o,r,l);t[l]?(t[l].p(w,i),m(t[l],1)):(t[l]=ft(w),t[l].c(),m(t[l],1),t[l].m(a.parentNode,a))}for(Je(),l=r.length;l<t.length;l+=1)s(l);Xe()}},i(o){if(!n){for(let i=0;i<r.length;i+=1)m(t[i]);n=!0}},o(o){t=t.filter(Boolean);for(let i=0;i<t.length;i+=1)h(t[i]);n=!1},d(o){Ye(t,o),o&&c(a)}}}function ut(p){let a,n;return a=new pt({props:{$$slots:{default:[Ct]},$$scope:{ctx:p}}}),{c(){g(a.$$.fragment)},l(r){x(a.$$.fragment,r)},m(r,t){v(a,r,t),n=!0},p(r,t){const s={};t&4096&&(s.$$scope={dirty:t,ctx:r}),a.$set(s)},i(r){n||(m(a.$$.fragment,r),n=!0)},o(r){h(a.$$.fragment,r),n=!1},d(r){V(a,r)}}}function Rt(p){let a,n,r=p[1],t=[];for(let o=0;o<r.length;o+=1)t[o]=ut(nt(p,r,o));const s=o=>h(t[o],1,1,()=>{t[o]=null});return{c(){for(let o=0;o<t.length;o+=1)t[o].c();a=se()},l(o){for(let i=0;i<t.length;i+=1)t[i].l(o);a=se()},m(o,i){for(let l=0;l<t.length;l+=1)t[l]&&t[l].m(o,i);u(o,a,i),n=!0},p(o,i){if(i&2){r=o[1];let l;for(l=0;l<r.length;l+=1){const w=nt(o,r,l);t[l]?(t[l].p(w,i),m(t[l],1)):(t[l]=ut(w),t[l].c(),m(t[l],1),t[l].m(a.parentNode,a))}for(Je(),l=r.length;l<t.length;l+=1)s(l);Xe()}},i(o){if(!n){for(let i=0;i<r.length;i+=1)m(t[i]);n=!0}},o(o){t=t.filter(Boolean);for(let i=0;i<t.length;i+=1)h(t[i]);n=!1},d(o){Ye(t,o),o&&c(a)}}}function St(p){let a,n,r,t;return a=new Gt({props:{$$slots:{default:[Tt]},$$scope:{ctx:p}}}),r=new bt({props:{$$slots:{default:[Rt]},$$scope:{ctx:p}}}),{c(){g(a.$$.fragment),n=d(),g(r.$$.fragment)},l(s){x(a.$$.fragment,s),n=_(s),x(r.$$.fragment,s)},m(s,o){v(a,s,o),u(s,n,o),v(r,s,o),t=!0},p(s,o){const i={};o&4096&&(i.$$scope={dirty:o,ctx:s}),a.$set(i);const l={};o&4096&&(l.$$scope={dirty:o,ctx:s}),r.$set(l)},i(s){t||(m(a.$$.fragment,s),m(r.$$.fragment,s),t=!0)},o(s){h(a.$$.fragment,s),h(r.$$.fragment,s),t=!1},d(s){V(a,s),s&&c(n),V(r,s)}}}function It(p){let a;return{c(){a=y("88%")},l(n){a=G(n,"88%")},m(n,r){u(n,a,r)},d(n){n&&c(a)}}}function Ft(p){let a,n,r,t,s,o,i,l,w,D,P,$,b,T,N,$e,ee,Re,he,te,Se,de,B,_e,ae,Ie,ge,F,xe,W,ve,U,Ve,H,we,O,ye,C,Fe,ie,We,Ue,Ge,j,be,re,He,Ee,q,ke,oe,Oe,Ae,K,Le,Q,De,Z,Me,J,Te,R,je,S,qe,ze,X,Pe,Y,Ne;return r=new ot({props:{$$slots:{default:[At]},$$scope:{ctx:p}}}),t=new Vt({props:{type:"reference",id:1}}),N=new wt({props:{maxWidth:Ut,$$slots:{default:[Lt]},$$scope:{ctx:p}}}),B=new yt({props:{$$slots:{default:[St]},$$scope:{ctx:p}}}),F=new M({props:{code:`import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets.cifar import CIFAR10
from torchvision import transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`}}),W=new M({props:{code:`train_transform = T.Compose([T.Resize((50, 50)), 
                             T.ToTensor(),
                             T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])`}}),U=new M({props:{code:"train_val_dataset = CIFAR10(root='../datasets', download=True, train=True, transform=train_transform)"}}),H=new M({props:{code:`# split dataset into train and validate
indices = list(range(len(train_val_dataset)))
train_idxs, val_idxs = train_test_split(
    indices, test_size=0.1, stratify=train_val_dataset.targets
)

train_dataset = Subset(train_val_dataset, train_idxs)
val_dataset = Subset(train_val_dataset, val_idxs)`}}),O=new M({props:{code:`# In the paper a batch size of 256 was used
batch_size=128
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)
val_dataloader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    drop_last=False,
)`}}),j=new M({props:{code:`class VGG_Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
          nn.BatchNorm2d(num_features=out_channels),
          nn.ReLU(inplace=True)
        )
  
    def forward(self, x):
        return self.layer(x)`}}),q=new M({props:{code:'cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]'}}),K=new M({props:{code:`class Model(nn.Module):

    def __init__(self, cfg, num_classes=1):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = self._make_feature_extractor()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10)
        )
        
    def _make_feature_extractor(self):
        layers = []
        in_channels = 3
        for element in self.cfg:
            if element == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [VGG_Block(in_channels, element)]
                in_channels = element
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x`}}),Q=new M({props:{code:`def track_performance(dataloader, model, criterion):
    # switch to evaluation mode
    model.eval()
    num_samples = 0
    num_correct = 0
    loss_sum = 0

    # no need to calculate gradients
    with torch.inference_mode():
        for _, (features, labels) in enumerate(dataloader):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                features = features.to(device)
                labels = labels.to(device)
                logits = model(features)

                predictions = logits.max(dim=1)[1]
                num_correct += (predictions == labels).sum().item()

                loss = criterion(logits, labels)
                loss_sum += loss.cpu().item()
                num_samples += len(features)

    # we return the average loss and the accuracy
    return loss_sum / num_samples, num_correct / num_samples`}}),Z=new M({props:{code:`def train(
    num_epochs,
    train_dataloader,
    val_dataloader,
    model,
    criterion,
    optimizer,
    scheduler=None,
):
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        start_time = time.time()
        for _, (features, labels) in enumerate(train_dataloader):
            model.train()
            features = features.to(device)
            labels = labels.to(device)

            # Empty the gradients
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # Forward Pass
                logits = model(features)
                # Calculate Loss
                loss = criterion(logits, labels)

            # Backward Pass
            scaler.scale(loss).backward()

            # Gradient Descent
            scaler.step(optimizer)
            scaler.update()

        val_loss, val_acc = track_performance(val_dataloader, model, criterion)
        end_time = time.time()

        s = (
            f"Epoch: {epoch+1:>2}/{num_epochs} | "
            f"Epoch Duration: {end_time - start_time:.3f} sec | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val Acc: {val_acc:.3f} |"
        )
        print(s)

        if scheduler:
            scheduler.step(val_loss)`}}),J=new M({props:{code:`model = Model(cfg)
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=2, verbose=True
)
criterion = nn.CrossEntropyLoss(reduction="sum")`}}),S=new ot({props:{$$slots:{default:[It]},$$scope:{ctx:p}}}),X=new M({props:{code:`train(
    num_epochs=30,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
)`}}),Y=new M({props:{isOutput:!0,code:`Epoch:  1/30 | Epoch Duration: 25.635 sec | Val Loss: 1.81580 | Val Acc: 0.296 |
Epoch:  2/30 | Epoch Duration: 24.916 sec | Val Loss: 1.38543 | Val Acc: 0.463 |
Epoch:  3/30 | Epoch Duration: 25.014 sec | Val Loss: 1.28278 | Val Acc: 0.547 |
Epoch:  4/30 | Epoch Duration: 25.074 sec | Val Loss: 1.19473 | Val Acc: 0.595 |
Epoch:  5/30 | Epoch Duration: 25.043 sec | Val Loss: 0.88059 | Val Acc: 0.689 |
Epoch:  6/30 | Epoch Duration: 25.063 sec | Val Loss: 0.71676 | Val Acc: 0.752 |
Epoch:  7/30 | Epoch Duration: 25.054 sec | Val Loss: 0.69538 | Val Acc: 0.760 |
Epoch:  8/30 | Epoch Duration: 25.065 sec | Val Loss: 0.77932 | Val Acc: 0.738 |
Epoch:  9/30 | Epoch Duration: 25.053 sec | Val Loss: 0.64442 | Val Acc: 0.792 |
Epoch: 10/30 | Epoch Duration: 25.080 sec | Val Loss: 0.55705 | Val Acc: 0.817 |
Epoch: 11/30 | Epoch Duration: 25.084 sec | Val Loss: 0.54697 | Val Acc: 0.821 |
Epoch: 12/30 | Epoch Duration: 25.086 sec | Val Loss: 0.51530 | Val Acc: 0.836 |
Epoch: 13/30 | Epoch Duration: 25.099 sec | Val Loss: 0.52571 | Val Acc: 0.832 |
Epoch: 14/30 | Epoch Duration: 25.081 sec | Val Loss: 0.52763 | Val Acc: 0.834 |
Epoch: 15/30 | Epoch Duration: 25.100 sec | Val Loss: 0.51354 | Val Acc: 0.852 |
Epoch: 16/30 | Epoch Duration: 25.063 sec | Val Loss: 0.49283 | Val Acc: 0.854 |
Epoch: 17/30 | Epoch Duration: 25.072 sec | Val Loss: 0.60646 | Val Acc: 0.839 |
Epoch: 18/30 | Epoch Duration: 25.110 sec | Val Loss: 0.68762 | Val Acc: 0.831 |
Epoch: 19/30 | Epoch Duration: 25.067 sec | Val Loss: 0.55200 | Val Acc: 0.852 |
Epoch 00019: reducing learning rate of group 0 to 1.0000e-04.
Epoch: 20/30 | Epoch Duration: 25.090 sec | Val Loss: 0.52681 | Val Acc: 0.877 |
Epoch: 21/30 | Epoch Duration: 25.084 sec | Val Loss: 0.54211 | Val Acc: 0.880 |
Epoch: 22/30 | Epoch Duration: 25.084 sec | Val Loss: 0.59634 | Val Acc: 0.878 |
Epoch 00022: reducing learning rate of group 0 to 1.0000e-05.
Epoch: 23/30 | Epoch Duration: 25.104 sec | Val Loss: 0.59584 | Val Acc: 0.881 |
Epoch: 24/30 | Epoch Duration: 25.052 sec | Val Loss: 0.60467 | Val Acc: 0.880 |
Epoch: 25/30 | Epoch Duration: 25.068 sec | Val Loss: 0.61155 | Val Acc: 0.880 |
Epoch 00025: reducing learning rate of group 0 to 1.0000e-06.
Epoch: 26/30 | Epoch Duration: 25.117 sec | Val Loss: 0.61680 | Val Acc: 0.879 |
Epoch: 27/30 | Epoch Duration: 25.059 sec | Val Loss: 0.62156 | Val Acc: 0.881 |
Epoch: 28/30 | Epoch Duration: 25.089 sec | Val Loss: 0.61393 | Val Acc: 0.878 |
Epoch 00028: reducing learning rate of group 0 to 1.0000e-07.
Epoch: 29/30 | Epoch Duration: 25.077 sec | Val Loss: 0.62117 | Val Acc: 0.880 |
Epoch: 30/30 | Epoch Duration: 25.075 sec | Val Loss: 0.61320 | Val Acc: 0.880 |`}}),{c(){a=k("p"),n=y("The "),g(r.$$.fragment),g(t.$$.fragment),s=y(` ConvNet
    architecture was developed by the Visual Geometry Group, a computer vision research
    lab at Oxford university. The neural network is similar in spirit to LeNet-5
    and AlexNet, but VGG is a much deeper neural network. Unlike AlexNet, VGG does
    not apply any large filters, but uses only small patches of 3x3. The authors
    attributed this design choice to the success of their neural network. VGG got
    second place for object classification and first place for object detection in
    the 2014 ImageNet challenge.`),o=d(),i=k("p"),l=y(`The VGG paper discussed networks of varying depth, from 11 layers to 19
    layers. We are going to discuss the 16 layer architecture, the so called
    VGG16 (architecture D in the paper).`),w=d(),D=k("p"),P=y(`As with many other deep learning architectures, VGG reuses the same module
    over and over again. The VGG module uses a convolutional layer with the
    kernel size of 3x3, stride of size 1 and padding of size 1, followed by
    batch normalization and the ReLU activation function. Be aware, that the
    BatchNorm2d layer was not used in the original VGG paper, but if you omit
    the normalization step, the network might suffer from vanishing gradients.`),$=d(),b=k("p"),T=d(),g(N.$$.fragment),$e=d(),ee=k("p"),Re=y(`After a couple of such modules, we apply a max pooling layer with a kernel
    of 2 and a stride of 2.`),he=d(),te=k("p"),Se=y("The full VGG16 implementation looks as follows."),de=d(),g(B.$$.fragment),_e=d(),ae=k("p"),Ie=y("Below we implement VGG16 to classify the images in the CIFAR-10 datset."),ge=d(),g(F.$$.fragment),xe=d(),g(W.$$.fragment),ve=d(),g(U.$$.fragment),Ve=d(),g(H.$$.fragment),we=d(),g(O.$$.fragment),ye=d(),C=k("p"),Fe=y("We create a "),ie=k("code"),We=y("VGG_Block"),Ue=y(" module, that we can reuse many times."),Ge=d(),g(j.$$.fragment),be=d(),re=k("p"),He=y(`VGG has a lot of repeatable blocks. It is common practice to store the
    configuration in a list and to construct the model from the config. The
    numbers represent the number of output filters in a convolutional layer. 'M'
    on the other hand indicates a maxpool layer.`),Ee=d(),g(q.$$.fragment),ke=d(),oe=k("p"),Oe=y(`Out model implementation is very close to the table above, but we have to
    account for the fact that our images are smaller, so we reduce the input in
    the first linear layer from 7x7x512 to 1x1x512.`),Ae=d(),g(K.$$.fragment),Le=d(),g(Q.$$.fragment),De=d(),g(Z.$$.fragment),Me=d(),g(J.$$.fragment),Te=d(),R=k("p"),je=y("When we train VGG16 on the CIFAR-10 dataset, we reach an accuracy of roughly "),g(S.$$.fragment),qe=y(", thereby beating the LeCun-5 and the AlexNet implementation."),ze=d(),g(X.$$.fragment),Pe=d(),g(Y.$$.fragment)},l(e){a=A(e,"P",{});var f=L(a);n=G(f,"The "),x(r.$$.fragment,f),x(t.$$.fragment,f),s=G(f,` ConvNet
    architecture was developed by the Visual Geometry Group, a computer vision research
    lab at Oxford university. The neural network is similar in spirit to LeNet-5
    and AlexNet, but VGG is a much deeper neural network. Unlike AlexNet, VGG does
    not apply any large filters, but uses only small patches of 3x3. The authors
    attributed this design choice to the success of their neural network. VGG got
    second place for object classification and first place for object detection in
    the 2014 ImageNet challenge.`),f.forEach(c),o=_(e),i=A(e,"P",{});var ce=L(i);l=G(ce,`The VGG paper discussed networks of varying depth, from 11 layers to 19
    layers. We are going to discuss the 16 layer architecture, the so called
    VGG16 (architecture D in the paper).`),ce.forEach(c),w=_(e),D=A(e,"P",{});var fe=L(D);P=G(fe,`As with many other deep learning architectures, VGG reuses the same module
    over and over again. The VGG module uses a convolutional layer with the
    kernel size of 3x3, stride of size 1 and padding of size 1, followed by
    batch normalization and the ReLU activation function. Be aware, that the
    BatchNorm2d layer was not used in the original VGG paper, but if you omit
    the normalization step, the network might suffer from vanishing gradients.`),fe.forEach(c),$=_(e),b=A(e,"P",{}),L(b).forEach(c),T=_(e),x(N.$$.fragment,e),$e=_(e),ee=A(e,"P",{});var ue=L(ee);Re=G(ue,`After a couple of such modules, we apply a max pooling layer with a kernel
    of 2 and a stride of 2.`),ue.forEach(c),he=_(e),te=A(e,"P",{});var pe=L(te);Se=G(pe,"The full VGG16 implementation looks as follows."),pe.forEach(c),de=_(e),x(B.$$.fragment,e),_e=_(e),ae=A(e,"P",{});var et=L(ae);Ie=G(et,"Below we implement VGG16 to classify the images in the CIFAR-10 datset."),et.forEach(c),ge=_(e),x(F.$$.fragment,e),xe=_(e),x(W.$$.fragment,e),ve=_(e),x(U.$$.fragment,e),Ve=_(e),x(H.$$.fragment,e),we=_(e),x(O.$$.fragment,e),ye=_(e),C=A(e,"P",{});var Be=L(C);Fe=G(Be,"We create a "),ie=A(Be,"CODE",{});var tt=L(ie);We=G(tt,"VGG_Block"),tt.forEach(c),Ue=G(Be," module, that we can reuse many times."),Be.forEach(c),Ge=_(e),x(j.$$.fragment,e),be=_(e),re=A(e,"P",{});var at=L(re);He=G(at,`VGG has a lot of repeatable blocks. It is common practice to store the
    configuration in a list and to construct the model from the config. The
    numbers represent the number of output filters in a convolutional layer. 'M'
    on the other hand indicates a maxpool layer.`),at.forEach(c),Ee=_(e),x(q.$$.fragment,e),ke=_(e),oe=A(e,"P",{});var rt=L(oe);Oe=G(rt,`Out model implementation is very close to the table above, but we have to
    account for the fact that our images are smaller, so we reduce the input in
    the first linear layer from 7x7x512 to 1x1x512.`),rt.forEach(c),Ae=_(e),x(K.$$.fragment,e),Le=_(e),x(Q.$$.fragment,e),De=_(e),x(Z.$$.fragment,e),Me=_(e),x(J.$$.fragment,e),Te=_(e),R=A(e,"P",{});var Ce=L(R);je=G(Ce,"When we train VGG16 on the CIFAR-10 dataset, we reach an accuracy of roughly "),x(S.$$.fragment,Ce),qe=G(Ce,", thereby beating the LeCun-5 and the AlexNet implementation."),Ce.forEach(c),ze=_(e),x(X.$$.fragment,e),Pe=_(e),x(Y.$$.fragment,e)},m(e,f){u(e,a,f),E(a,n),v(r,a,null),v(t,a,null),E(a,s),u(e,o,f),u(e,i,f),E(i,l),u(e,w,f),u(e,D,f),E(D,P),u(e,$,f),u(e,b,f),u(e,T,f),v(N,e,f),u(e,$e,f),u(e,ee,f),E(ee,Re),u(e,he,f),u(e,te,f),E(te,Se),u(e,de,f),v(B,e,f),u(e,_e,f),u(e,ae,f),E(ae,Ie),u(e,ge,f),v(F,e,f),u(e,xe,f),v(W,e,f),u(e,ve,f),v(U,e,f),u(e,Ve,f),v(H,e,f),u(e,we,f),v(O,e,f),u(e,ye,f),u(e,C,f),E(C,Fe),E(C,ie),E(ie,We),E(C,Ue),u(e,Ge,f),v(j,e,f),u(e,be,f),u(e,re,f),E(re,He),u(e,Ee,f),v(q,e,f),u(e,ke,f),u(e,oe,f),E(oe,Oe),u(e,Ae,f),v(K,e,f),u(e,Le,f),v(Q,e,f),u(e,De,f),v(Z,e,f),u(e,Me,f),v(J,e,f),u(e,Te,f),u(e,R,f),E(R,je),v(S,R,null),E(R,qe),u(e,ze,f),v(X,e,f),u(e,Pe,f),v(Y,e,f),Ne=!0},p(e,f){const ce={};f&4096&&(ce.$$scope={dirty:f,ctx:e}),r.$set(ce);const fe={};f&4096&&(fe.$$scope={dirty:f,ctx:e}),N.$set(fe);const ue={};f&4096&&(ue.$$scope={dirty:f,ctx:e}),B.$set(ue);const pe={};f&4096&&(pe.$$scope={dirty:f,ctx:e}),S.$set(pe)},i(e){Ne||(m(r.$$.fragment,e),m(t.$$.fragment,e),m(N.$$.fragment,e),m(B.$$.fragment,e),m(F.$$.fragment,e),m(W.$$.fragment,e),m(U.$$.fragment,e),m(H.$$.fragment,e),m(O.$$.fragment,e),m(j.$$.fragment,e),m(q.$$.fragment,e),m(K.$$.fragment,e),m(Q.$$.fragment,e),m(Z.$$.fragment,e),m(J.$$.fragment,e),m(S.$$.fragment,e),m(X.$$.fragment,e),m(Y.$$.fragment,e),Ne=!0)},o(e){h(r.$$.fragment,e),h(t.$$.fragment,e),h(N.$$.fragment,e),h(B.$$.fragment,e),h(F.$$.fragment,e),h(W.$$.fragment,e),h(U.$$.fragment,e),h(H.$$.fragment,e),h(O.$$.fragment,e),h(j.$$.fragment,e),h(q.$$.fragment,e),h(K.$$.fragment,e),h(Q.$$.fragment,e),h(Z.$$.fragment,e),h(J.$$.fragment,e),h(S.$$.fragment,e),h(X.$$.fragment,e),h(Y.$$.fragment,e),Ne=!1},d(e){e&&c(a),V(r),V(t),e&&c(o),e&&c(i),e&&c(w),e&&c(D),e&&c($),e&&c(b),e&&c(T),V(N,e),e&&c($e),e&&c(ee),e&&c(he),e&&c(te),e&&c(de),V(B,e),e&&c(_e),e&&c(ae),e&&c(ge),V(F,e),e&&c(xe),V(W,e),e&&c(ve),V(U,e),e&&c(Ve),V(H,e),e&&c(we),V(O,e),e&&c(ye),e&&c(C),e&&c(Ge),V(j,e),e&&c(be),e&&c(re),e&&c(Ee),V(q,e),e&&c(ke),e&&c(oe),e&&c(Ae),V(K,e),e&&c(Le),V(Q,e),e&&c(De),V(Z,e),e&&c(Me),V(J,e),e&&c(Te),e&&c(R),V(S),e&&c(ze),V(X,e),e&&c(Pe),V(Y,e)}}}function Wt(p){let a,n,r,t,s,o,i,l,w,D,P;return l=new xt({props:{$$slots:{default:[Ft]},$$scope:{ctx:p}}}),D=new vt({props:{references:p[2]}}),{c(){a=k("meta"),n=d(),r=k("h1"),t=y("VGG"),s=d(),o=k("div"),i=d(),g(l.$$.fragment),w=d(),g(D.$$.fragment),this.h()},l($){const b=dt("svelte-1crvhqj",document.head);a=A(b,"META",{name:!0,content:!0}),b.forEach(c),n=_($),r=A($,"H1",{});var T=L(r);t=G(T,"VGG"),T.forEach(c),s=_($),o=A($,"DIV",{class:!0}),L(o).forEach(c),i=_($),x(l.$$.fragment,$),w=_($),x(D.$$.fragment,$),this.h()},h(){document.title="VGG - World4AI",ne(a,"name","description"),ne(a,"content","VGG is at heart a very simple convolutional neural network architecture. It stacks layers of convolutions followed by max pooling. But compared to AlexNet or LeNet-5 this architecture showed that deeper and deeper networks might be necessary to achieve truly impressive results."),ne(o,"class","separator")},m($,b){E(document.head,a),u($,n,b),u($,r,b),E(r,t),u($,s,b),u($,o,b),u($,i,b),v(l,$,b),u($,w,b),v(D,$,b),P=!0},p($,[b]){const T={};b&4096&&(T.$$scope={dirty:b,ctx:$}),l.$set(T)},i($){P||(m(l.$$.fragment,$),m(D.$$.fragment,$),P=!0)},o($){h(l.$$.fragment,$),h(D.$$.fragment,$),P=!1},d($){c(a),$&&c(n),$&&c(r),$&&c(s),$&&c(o),$&&c(i),V(l,$),$&&c(w),V(D,$)}}}let Qe=1,I=90,le=100,Ze=80,z=20,Ut="200px";function Ht(p){return[["Type","Input Size","Output Size"],[["VGG Module","224x224x3","224x224x64"],["VGG Module","224x224x64","224x224x64"],["Max Pooling","224x224x64","112x112x64"],["VGG Module","112x112x64","112x112x128"],["VGG Module","112x112x128","112x112x128"],["Max Pooling","112x112x128","56x56x128"],["VGG Module","56x56x128","56x56x256"],["VGG Module","56x56x256","56x56x256"],["VGG Module","56x56x256","56x56x256"],["Max Pooling","56x56x256","28x28x256"],["VGG Module","28x28x256","28x28x512"],["VGG Module","28x28x512","28x28x512"],["VGG Module","28x28x512","28x28x512"],["Max Pooling","28x28x512","14x14x512"],["VGG Module","14x14x512","14x14x512"],["VGG Module","14x14x512","14x14x512"],["VGG Module","14x14x512","14x14x512"],["Max Pooling","14x14x512","7x7x512"],["Dropout","-","-"],["Fully Connected","25088","4096"],["ReLU","-","-"],["Dropout","-","-"],["Fully Connected","4096","4096"],["ReLU","-","-"],["Fully Connected","4096","1000"],["Softmax","-","-"]],[{author:"Simonyan, K., & Zisserman, A.",title:"Very deep convolutional networks for large-scale image recognition",journal:"",year:"2014",pages:"",volume:"",issue:""}]]}class ea extends mt{constructor(a){super(),$t(this,a,Ht,Wt,ht,{})}}export{ea as default};
