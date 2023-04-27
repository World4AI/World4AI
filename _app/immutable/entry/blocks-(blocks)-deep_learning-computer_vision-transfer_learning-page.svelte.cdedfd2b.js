import{S as _t,i as vt,s as yt,k as _,a as i,q as f,y as m,W as Et,l as v,h as t,c as n,m as y,r as c,z as h,n as I,N as p,b as r,A as d,g as $,d as g,B as w,L as bt}from"../chunks/index.4d92b023.js";import{C as Vt}from"../chunks/Container.b0705c7b.js";import{H as Ye}from"../chunks/Highlight.b7c1de53.js";import{A as At}from"../chunks/Alert.25a852b3.js";import{P as b}from"../chunks/PythonCode.212ba7a6.js";const Lt=""+new URL("../assets/pets.124dc457.webp",import.meta.url).href;function Tt(L){let s;return{c(){s=f(`You should utilize transfer learning when you do not have the necessary data
    or computational power at your disposal to train large models from scratch`)},l(o){s=c(o,`You should utilize transfer learning when you do not have the necessary data
    or computational power at your disposal to train large models from scratch`)},m(o,u){r(o,s,u)},d(o){o&&t(s)}}}function Dt(L){let s;return{c(){s=f("Transfer learning")},l(o){s=c(o,"Transfer learning")},m(o,u){r(o,s,u)},d(o){o&&t(s)}}}function zt(L){let s;return{c(){s=f("feature extraction")},l(o){s=c(o,"feature extraction")},m(o,u){r(o,s,u)},d(o){o&&t(s)}}}function It(L){let s;return{c(){s=f("fine-tuning")},l(o){s=c(o,"fine-tuning")},m(o,u){r(o,s,u)},d(o){o&&t(s)}}}function kt(L){let s;return{c(){s=f("90%")},l(o){s=c(o,"90%")},m(o,u){r(o,s,u)},d(o){o&&t(s)}}}function Wt(L){let s,o,u,D,k,V,T,z,W,l,E,A,Ke,P,Je,$e,ae,Qe,ge,re,Xe,we,se,Ze,_e,N,et,O,tt,at,ve,F,ye,S,Ee,j,be,G,ut,Ve,R,rt,ue,st,ot,Ae,H,Le,oe,lt,Te,M,De,B,ze,U,Ie,le,ke,Y,We,K,xe,ie,it,Pe,J,Ne,ne,nt,Oe,Q,Re,fe,ft,qe,X,Ce,Z,Fe,ee,Se,te,je,q,ct,C,pt,Ge,ce,He;return D=new At({props:{type:"info",$$slots:{default:[Tt]},$$scope:{ctx:L}}}),T=new Ye({props:{$$slots:{default:[Dt]},$$scope:{ctx:L}}}),A=new Ye({props:{$$slots:{default:[zt]},$$scope:{ctx:L}}}),P=new Ye({props:{$$slots:{default:[It]},$$scope:{ctx:L}}}),F=new b({props:{code:`import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import OxfordIIITPet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`}}),S=new b({props:{code:`dataset = OxfordIIITPet(root='../datasets', 
                              split='trainval', 
                              target_types='category', 
                              download=True)`}}),j=new b({props:{code:`fig = plt.figure(figsize=(8, 8))
columns = 4
rows = 5

for i in range(1, columns*rows +1):
    img, cls = dataset[i*50]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
    plt.title(f'Category {cls}')
    plt.axis('off')
plt.savefig('pets', bbox_inches='tight')
plt.show()`}}),H=new b({props:{code:"from torchvision.models import resnet34, ResNet34_Weights"}}),M=new b({props:{code:`weights = ResNet34_Weights.DEFAULT
preprocess = weights.transforms()`}}),B=new b({props:{code:`train_dataset = OxfordIIITPet(root='../datasets', 
                                  split='trainval', 
                                  target_types='category', 
                                  transform=preprocess, 
                                  download=True)

val_dataset = OxfordIIITPet(root='../datasets', 
                                  split='test', 
                                  target_types='category', 
                                  transform=preprocess, 
                                  download=True)`}}),U=new b({props:{code:`batch_size=128
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
)`}}),Y=new b({props:{code:`def track_performance(dataloader, model, criterion):
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
    return loss_sum / num_samples, num_correct / num_samples`}}),K=new b({props:{code:`def train(
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
            scheduler.step(val_loss)`}}),J=new b({props:{code:"model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)"}}),Q=new b({props:{code:`for param in model.parameters():
    param.requires_grad = False`}}),X=new b({props:{code:"model.fc = nn.Linear(in_features=512, out_features=37)"}}),Z=new b({props:{code:`optimizer = optim.Adam(params=model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=2, verbose=True
)
criterion = nn.CrossEntropyLoss(reduction="sum")`}}),ee=new b({props:{code:`train(
    num_epochs=30,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
)`}}),te=new b({props:{isOutput:!0,code:`Epoch:  1/30 | Epoch Duration: 11.154 sec | Val Loss: 1.81516 | Val Acc: 0.677 |
Epoch:  2/30 | Epoch Duration: 11.453 sec | Val Loss: 0.90655 | Val Acc: 0.854 |
Epoch:  3/30 | Epoch Duration: 11.348 sec | Val Loss: 0.63867 | Val Acc: 0.870 |
Epoch:  4/30 | Epoch Duration: 11.845 sec | Val Loss: 0.52753 | Val Acc: 0.883 |
Epoch:  5/30 | Epoch Duration: 12.005 sec | Val Loss: 0.46197 | Val Acc: 0.892 |
Epoch:  6/30 | Epoch Duration: 11.932 sec | Val Loss: 0.42866 | Val Acc: 0.894 |
Epoch:  7/30 | Epoch Duration: 12.047 sec | Val Loss: 0.40674 | Val Acc: 0.896 |
Epoch:  8/30 | Epoch Duration: 12.032 sec | Val Loss: 0.38285 | Val Acc: 0.899 |
Epoch:  9/30 | Epoch Duration: 12.055 sec | Val Loss: 0.37018 | Val Acc: 0.900 |
Epoch: 10/30 | Epoch Duration: 11.667 sec | Val Loss: 0.35984 | Val Acc: 0.901 |
Epoch: 11/30 | Epoch Duration: 12.243 sec | Val Loss: 0.34247 | Val Acc: 0.902 |
Epoch: 12/30 | Epoch Duration: 12.104 sec | Val Loss: 0.34527 | Val Acc: 0.900 |
Epoch: 13/30 | Epoch Duration: 12.275 sec | Val Loss: 0.34026 | Val Acc: 0.901 |
Epoch: 14/30 | Epoch Duration: 11.949 sec | Val Loss: 0.33695 | Val Acc: 0.896 |
Epoch: 15/30 | Epoch Duration: 12.117 sec | Val Loss: 0.32628 | Val Acc: 0.904 |
Epoch: 16/30 | Epoch Duration: 11.852 sec | Val Loss: 0.32397 | Val Acc: 0.901 |
Epoch: 17/30 | Epoch Duration: 12.116 sec | Val Loss: 0.32091 | Val Acc: 0.904 |
Epoch: 18/30 | Epoch Duration: 12.015 sec | Val Loss: 0.32093 | Val Acc: 0.904 |
Epoch: 19/30 | Epoch Duration: 12.026 sec | Val Loss: 0.31584 | Val Acc: 0.904 |
Epoch: 20/30 | Epoch Duration: 12.420 sec | Val Loss: 0.31596 | Val Acc: 0.905 |
Epoch: 21/30 | Epoch Duration: 12.367 sec | Val Loss: 0.32160 | Val Acc: 0.900 |
Epoch: 22/30 | Epoch Duration: 12.472 sec | Val Loss: 0.31340 | Val Acc: 0.904 |
Epoch: 23/30 | Epoch Duration: 12.134 sec | Val Loss: 0.31088 | Val Acc: 0.903 |
Epoch: 24/30 | Epoch Duration: 12.194 sec | Val Loss: 0.31267 | Val Acc: 0.903 |
Epoch: 25/30 | Epoch Duration: 12.317 sec | Val Loss: 0.31323 | Val Acc: 0.901 |
Epoch: 26/30 | Epoch Duration: 12.017 sec | Val Loss: 0.31473 | Val Acc: 0.900 |
Epoch 00026: reducing learning rate of group 0 to 1.0000e-04.
Epoch: 27/30 | Epoch Duration: 12.051 sec | Val Loss: 0.31021 | Val Acc: 0.905 |
Epoch: 28/30 | Epoch Duration: 12.122 sec | Val Loss: 0.30899 | Val Acc: 0.904 |
Epoch: 29/30 | Epoch Duration: 11.754 sec | Val Loss: 0.30938 | Val Acc: 0.904 |
Epoch: 30/30 | Epoch Duration: 12.284 sec | Val Loss: 0.30751 | Val Acc: 0.906 |
`}}),C=new Ye({props:{$$slots:{default:[kt]},$$scope:{ctx:L}}}),{c(){s=_("p"),o=f(`Often our datasets are extremely small and/or we do not have the compute to
    train a large model from scratch.`),u=i(),m(D.$$.fragment),k=i(),V=_("p"),m(T.$$.fragment),z=f(` allows you to take already existing
    pretrained models and to adjust them to your needs. The requirements towards
    computational resources and availability of data sinks dramatically once you
    start to you utilize transfer learning.`),W=i(),l=_("p"),E=f("There are generally two ways to utilize transfer learing: "),m(A.$$.fragment),Ke=f(" and "),m(P.$$.fragment),Je=f("."),$e=i(),ae=_("p"),Qe=f(`When we use the pretrained model as a feature extractor, we load the model,
    freeze all weights and replace the last couple of layers with the layers
    that suit our task. As this procedure only requires to train a few layers,
    it tends to be relatively fast.`),ge=i(),re=_("p"),Xe=f(`When we use fine-tuning, we load the weights, replace the last couple of
    layers, but fune-tune all available weights during the training process.
    There is a potential chance to get better results with fine-tuning, but this
    procedure obviously requires more compute.`),we=i(),se=_("p"),Ze=f(`The resoning behind the success of transfer learning is as follows. We have
    mentioned before that the convolutional layers are supposed to learn the
    features of the dataset. It can be argued that if the network has learned to
    recognize edges, colors and higher level features, that those features are
    also useful for other tasks. If the model has learned to classify cats and
    dogs, it should be a relative minor undertaking to adjust the model to
    recognize other animals. On the other hand it is going to be harder to
    fine-tune the same model on a car dataset. The closer the original datset is
    to your data, the more sense it makes to use the pretrained model.`),_e=i(),N=_("p"),et=f("For our presentation we have chosen the "),O=_("a"),tt=f("Oxford-IIIT Pet Dataset"),at=f(`. The daset consists of roughly 7400 samples of cats and dogs. There are 37
    categories of cat and dog breeds in the dataset with roughly 200 per
    category. As we will divide the dataset into the training and the validation
    dataset, there will be roughly 100 samples per category for training. All
    things considered, this is a relatively small dataset. We have chosen this
    dataset, because the original ImageNet contains cats and dogs and we expect
    transfer learning to work quite well.`),ve=i(),m(F.$$.fragment),ye=i(),m(S.$$.fragment),Ee=i(),m(j.$$.fragment),be=i(),G=_("img"),Ve=i(),R=_("p"),rt=f(`We will be using the ResNet34 architecture for transfer learning. We can get
    a lot of pretrained computer vision models, including ResNet, from the `),ue=_("code"),st=f("torchvision"),ot=f(`
    library.`),Ae=i(),m(H.$$.fragment),Le=i(),oe=_("p"),lt=f(`When we use transfer learning it is important to utilize the same
    preprocessing steps that were used for the training of the original model.`),Te=i(),m(M.$$.fragment),De=i(),m(B.$$.fragment),ze=i(),m(U.$$.fragment),Ie=i(),le=_("div"),ke=i(),m(Y.$$.fragment),We=i(),m(K.$$.fragment),xe=i(),ie=_("p"),it=f(`We create the ResNet34 model and download the weights that were pretrained
    on the ImageNet dataset.`),Pe=i(),m(J.$$.fragment),Ne=i(),ne=_("p"),nt=f(`We will utilize the model as a feature extractor, therefore we freeze all
    layer weights.`),Oe=i(),m(Q.$$.fragment),Re=i(),fe=_("p"),ft=f(`We replace the very last layer with a linear layer with 37 outputs. This is
    the only layer that is going to be trained.`),qe=i(),m(X.$$.fragment),Ce=i(),m(Z.$$.fragment),Fe=i(),m(ee.$$.fragment),Se=i(),m(te.$$.fragment),je=i(),q=_("p"),ct=f("Out of the box we get an accuracy of over "),m(C.$$.fragment),pt=f(`. Think
    about how amazing those results are. We had 37 different categories, limited
    data and limited computational resources and we have essentially trained a
    linear classifier based on the features from the ResNet model. Still we get
    an accuracy of over 90%. This is the power of transfer learning.`),Ge=i(),ce=_("div"),this.h()},l(e){s=v(e,"P",{});var a=y(s);o=c(a,`Often our datasets are extremely small and/or we do not have the compute to
    train a large model from scratch.`),a.forEach(t),u=n(e),h(D.$$.fragment,e),k=n(e),V=v(e,"P",{});var pe=y(V);h(T.$$.fragment,pe),z=c(pe,` allows you to take already existing
    pretrained models and to adjust them to your needs. The requirements towards
    computational resources and availability of data sinks dramatically once you
    start to you utilize transfer learning.`),pe.forEach(t),W=n(e),l=v(e,"P",{});var x=y(l);E=c(x,"There are generally two ways to utilize transfer learing: "),h(A.$$.fragment,x),Ke=c(x," and "),h(P.$$.fragment,x),Je=c(x,"."),x.forEach(t),$e=n(e),ae=v(e,"P",{});var me=y(ae);Qe=c(me,`When we use the pretrained model as a feature extractor, we load the model,
    freeze all weights and replace the last couple of layers with the layers
    that suit our task. As this procedure only requires to train a few layers,
    it tends to be relatively fast.`),me.forEach(t),ge=n(e),re=v(e,"P",{});var he=y(re);Xe=c(he,`When we use fine-tuning, we load the weights, replace the last couple of
    layers, but fune-tune all available weights during the training process.
    There is a potential chance to get better results with fine-tuning, but this
    procedure obviously requires more compute.`),he.forEach(t),we=n(e),se=v(e,"P",{});var de=y(se);Ze=c(de,`The resoning behind the success of transfer learning is as follows. We have
    mentioned before that the convolutional layers are supposed to learn the
    features of the dataset. It can be argued that if the network has learned to
    recognize edges, colors and higher level features, that those features are
    also useful for other tasks. If the model has learned to classify cats and
    dogs, it should be a relative minor undertaking to adjust the model to
    recognize other animals. On the other hand it is going to be harder to
    fine-tune the same model on a car dataset. The closer the original datset is
    to your data, the more sense it makes to use the pretrained model.`),de.forEach(t),_e=n(e),N=v(e,"P",{});var Me=y(N);et=c(Me,"For our presentation we have chosen the "),O=v(Me,"A",{href:!0,rel:!0,target:!0});var mt=y(O);tt=c(mt,"Oxford-IIIT Pet Dataset"),mt.forEach(t),at=c(Me,`. The daset consists of roughly 7400 samples of cats and dogs. There are 37
    categories of cat and dog breeds in the dataset with roughly 200 per
    category. As we will divide the dataset into the training and the validation
    dataset, there will be roughly 100 samples per category for training. All
    things considered, this is a relatively small dataset. We have chosen this
    dataset, because the original ImageNet contains cats and dogs and we expect
    transfer learning to work quite well.`),Me.forEach(t),ve=n(e),h(F.$$.fragment,e),ye=n(e),h(S.$$.fragment,e),Ee=n(e),h(j.$$.fragment,e),be=n(e),G=v(e,"IMG",{src:!0,alt:!0}),Ve=n(e),R=v(e,"P",{});var Be=y(R);rt=c(Be,`We will be using the ResNet34 architecture for transfer learning. We can get
    a lot of pretrained computer vision models, including ResNet, from the `),ue=v(Be,"CODE",{});var ht=y(ue);st=c(ht,"torchvision"),ht.forEach(t),ot=c(Be,`
    library.`),Be.forEach(t),Ae=n(e),h(H.$$.fragment,e),Le=n(e),oe=v(e,"P",{});var dt=y(oe);lt=c(dt,`When we use transfer learning it is important to utilize the same
    preprocessing steps that were used for the training of the original model.`),dt.forEach(t),Te=n(e),h(M.$$.fragment,e),De=n(e),h(B.$$.fragment,e),ze=n(e),h(U.$$.fragment,e),Ie=n(e),le=v(e,"DIV",{class:!0}),y(le).forEach(t),ke=n(e),h(Y.$$.fragment,e),We=n(e),h(K.$$.fragment,e),xe=n(e),ie=v(e,"P",{});var $t=y(ie);it=c($t,`We create the ResNet34 model and download the weights that were pretrained
    on the ImageNet dataset.`),$t.forEach(t),Pe=n(e),h(J.$$.fragment,e),Ne=n(e),ne=v(e,"P",{});var gt=y(ne);nt=c(gt,`We will utilize the model as a feature extractor, therefore we freeze all
    layer weights.`),gt.forEach(t),Oe=n(e),h(Q.$$.fragment,e),Re=n(e),fe=v(e,"P",{});var wt=y(fe);ft=c(wt,`We replace the very last layer with a linear layer with 37 outputs. This is
    the only layer that is going to be trained.`),wt.forEach(t),qe=n(e),h(X.$$.fragment,e),Ce=n(e),h(Z.$$.fragment,e),Fe=n(e),h(ee.$$.fragment,e),Se=n(e),h(te.$$.fragment,e),je=n(e),q=v(e,"P",{});var Ue=y(q);ct=c(Ue,"Out of the box we get an accuracy of over "),h(C.$$.fragment,Ue),pt=c(Ue,`. Think
    about how amazing those results are. We had 37 different categories, limited
    data and limited computational resources and we have essentially trained a
    linear classifier based on the features from the ResNet model. Still we get
    an accuracy of over 90%. This is the power of transfer learning.`),Ue.forEach(t),Ge=n(e),ce=v(e,"DIV",{class:!0}),y(ce).forEach(t),this.h()},h(){I(O,"href","https://www.robots.ox.ac.uk/~vgg/data/pets/"),I(O,"rel","noreferrer"),I(O,"target","_blank"),bt(G.src,ut=Lt)||I(G,"src",ut),I(G,"alt","Different breeds of cats and dogs"),I(le,"class","separator"),I(ce,"class","separator")},m(e,a){r(e,s,a),p(s,o),r(e,u,a),d(D,e,a),r(e,k,a),r(e,V,a),d(T,V,null),p(V,z),r(e,W,a),r(e,l,a),p(l,E),d(A,l,null),p(l,Ke),d(P,l,null),p(l,Je),r(e,$e,a),r(e,ae,a),p(ae,Qe),r(e,ge,a),r(e,re,a),p(re,Xe),r(e,we,a),r(e,se,a),p(se,Ze),r(e,_e,a),r(e,N,a),p(N,et),p(N,O),p(O,tt),p(N,at),r(e,ve,a),d(F,e,a),r(e,ye,a),d(S,e,a),r(e,Ee,a),d(j,e,a),r(e,be,a),r(e,G,a),r(e,Ve,a),r(e,R,a),p(R,rt),p(R,ue),p(ue,st),p(R,ot),r(e,Ae,a),d(H,e,a),r(e,Le,a),r(e,oe,a),p(oe,lt),r(e,Te,a),d(M,e,a),r(e,De,a),d(B,e,a),r(e,ze,a),d(U,e,a),r(e,Ie,a),r(e,le,a),r(e,ke,a),d(Y,e,a),r(e,We,a),d(K,e,a),r(e,xe,a),r(e,ie,a),p(ie,it),r(e,Pe,a),d(J,e,a),r(e,Ne,a),r(e,ne,a),p(ne,nt),r(e,Oe,a),d(Q,e,a),r(e,Re,a),r(e,fe,a),p(fe,ft),r(e,qe,a),d(X,e,a),r(e,Ce,a),d(Z,e,a),r(e,Fe,a),d(ee,e,a),r(e,Se,a),d(te,e,a),r(e,je,a),r(e,q,a),p(q,ct),d(C,q,null),p(q,pt),r(e,Ge,a),r(e,ce,a),He=!0},p(e,a){const pe={};a&1&&(pe.$$scope={dirty:a,ctx:e}),D.$set(pe);const x={};a&1&&(x.$$scope={dirty:a,ctx:e}),T.$set(x);const me={};a&1&&(me.$$scope={dirty:a,ctx:e}),A.$set(me);const he={};a&1&&(he.$$scope={dirty:a,ctx:e}),P.$set(he);const de={};a&1&&(de.$$scope={dirty:a,ctx:e}),C.$set(de)},i(e){He||($(D.$$.fragment,e),$(T.$$.fragment,e),$(A.$$.fragment,e),$(P.$$.fragment,e),$(F.$$.fragment,e),$(S.$$.fragment,e),$(j.$$.fragment,e),$(H.$$.fragment,e),$(M.$$.fragment,e),$(B.$$.fragment,e),$(U.$$.fragment,e),$(Y.$$.fragment,e),$(K.$$.fragment,e),$(J.$$.fragment,e),$(Q.$$.fragment,e),$(X.$$.fragment,e),$(Z.$$.fragment,e),$(ee.$$.fragment,e),$(te.$$.fragment,e),$(C.$$.fragment,e),He=!0)},o(e){g(D.$$.fragment,e),g(T.$$.fragment,e),g(A.$$.fragment,e),g(P.$$.fragment,e),g(F.$$.fragment,e),g(S.$$.fragment,e),g(j.$$.fragment,e),g(H.$$.fragment,e),g(M.$$.fragment,e),g(B.$$.fragment,e),g(U.$$.fragment,e),g(Y.$$.fragment,e),g(K.$$.fragment,e),g(J.$$.fragment,e),g(Q.$$.fragment,e),g(X.$$.fragment,e),g(Z.$$.fragment,e),g(ee.$$.fragment,e),g(te.$$.fragment,e),g(C.$$.fragment,e),He=!1},d(e){e&&t(s),e&&t(u),w(D,e),e&&t(k),e&&t(V),w(T),e&&t(W),e&&t(l),w(A),w(P),e&&t($e),e&&t(ae),e&&t(ge),e&&t(re),e&&t(we),e&&t(se),e&&t(_e),e&&t(N),e&&t(ve),w(F,e),e&&t(ye),w(S,e),e&&t(Ee),w(j,e),e&&t(be),e&&t(G),e&&t(Ve),e&&t(R),e&&t(Ae),w(H,e),e&&t(Le),e&&t(oe),e&&t(Te),w(M,e),e&&t(De),w(B,e),e&&t(ze),w(U,e),e&&t(Ie),e&&t(le),e&&t(ke),w(Y,e),e&&t(We),w(K,e),e&&t(xe),e&&t(ie),e&&t(Pe),w(J,e),e&&t(Ne),e&&t(ne),e&&t(Oe),w(Q,e),e&&t(Re),e&&t(fe),e&&t(qe),w(X,e),e&&t(Ce),w(Z,e),e&&t(Fe),w(ee,e),e&&t(Se),w(te,e),e&&t(je),e&&t(q),w(C),e&&t(Ge),e&&t(ce)}}}function xt(L){let s,o,u,D,k,V,T,z,W;return z=new Vt({props:{$$slots:{default:[Wt]},$$scope:{ctx:L}}}),{c(){s=_("meta"),o=i(),u=_("h1"),D=f("Transfer Learning"),k=i(),V=_("div"),T=i(),m(z.$$.fragment),this.h()},l(l){const E=Et("svelte-ulgsnm",document.head);s=v(E,"META",{name:!0,content:!0}),E.forEach(t),o=n(l),u=v(l,"H1",{});var A=y(u);D=c(A,"Transfer Learning"),A.forEach(t),k=n(l),V=v(l,"DIV",{class:!0}),y(V).forEach(t),T=n(l),h(z.$$.fragment,l),this.h()},h(){document.title="Transfer Learning - World4AI",I(s,"name","description"),I(s,"content","To train computer vision models on real life tasks requires a lot of computational power and data, whch might not be available. Transfer learning allows us to take a pretrained model and to tune it for our purposes. Transfer learning often workes even if we have a lower end computer and a few samles available."),I(V,"class","separator")},m(l,E){p(document.head,s),r(l,o,E),r(l,u,E),p(u,D),r(l,k,E),r(l,V,E),r(l,T,E),d(z,l,E),W=!0},p(l,[E]){const A={};E&1&&(A.$$scope={dirty:E,ctx:l}),z.$set(A)},i(l){W||($(z.$$.fragment,l),W=!0)},o(l){g(z.$$.fragment,l),W=!1},d(l){t(s),l&&t(o),l&&t(u),l&&t(k),l&&t(V),l&&t(T),w(z,l)}}}class Ct extends _t{constructor(s){super(),vt(this,s,null,xt,yt,{})}}export{Ct as default};
