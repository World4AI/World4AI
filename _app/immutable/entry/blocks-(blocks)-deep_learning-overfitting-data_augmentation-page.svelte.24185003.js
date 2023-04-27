import{S as Ct,i as Ft,s as qt,k as u,a as o,q as m,y,W as Gt,l as p,h as t,c as s,m as d,r as f,z as v,n as $,N as i,b as n,A as b,g as T,d as I,B as E,L as ye}from"../chunks/index.4d92b023.js";import{C as Ht}from"../chunks/Container.b0705c7b.js";import{H as Bt}from"../chunks/Highlight.b7c1de53.js";import{A as Ut}from"../chunks/Alert.25a852b3.js";import{P as z}from"../chunks/PythonCode.212ba7a6.js";const jt=""+new URL("../assets/overfitting.49abfd01.webp",import.meta.url).href,Kt=""+new URL("../assets/mnist_orig.478e94a6.webp",import.meta.url).href,Yt=""+new URL("../assets/mnist_blur.ad7c0010.webp",import.meta.url).href,Zt=""+new URL("../assets/mnist_flipped.c05a031f.webp",import.meta.url).href,Jt=""+new URL("../assets/mnist_rotated.a0fd671a.webp",import.meta.url).href;function Qt(h){let l;return{c(){l=m("data augmentation")},l(g){l=f(g,"data augmentation")},m(g,w){n(g,l,w)},d(g){g&&t(l)}}}function Xt(h){let l;return{c(){l=m(`Data augmentation is a techinque that applies transformations to the
    original dataset, thereby creating synthetic data, that can be used in
    training.`)},l(g){l=f(g,`Data augmentation is a techinque that applies transformations to the
    original dataset, thereby creating synthetic data, that can be used in
    training.`)},m(g,w){n(g,l,w)},d(g){g&&t(l)}}}function ea(h){let l;return{c(){l=m("transfer learning")},l(g){l=f(g,"transfer learning")},m(g,w){n(g,l,w)},d(g){g&&t(l)}}}function ta(h){let l,g,w,A,S,_,O,k,P,r,c,x,R,Je,Qe,ve,ne,Xe,be,ie,et,Te,oe,tt,Ie,W,Ee,se,at,ke,B,Pe,C,bt,Ae,D,nt,he,it,ot,Le,F,Me,q,Tt,ze,L,st,we,rt,lt,G,mt,re,It,Se,M,ft,$e,ut,pt,H,gt,le,Et,Oe,me,dt,xe,U,Re,fe,ct,De,j,Ne,ue,ht,Ve,K,We,pe,wt,Be,Y,Ce,Z,Fe,ge,$t,qe,J,Ge,Q,kt,He,de,_t,Ue,N,yt,V,vt,je,ce,Ke;return _=new Bt({props:{$$slots:{default:[Qt]},$$scope:{ctx:h}}}),P=new Ut({props:{type:"info",$$slots:{default:[Xt]},$$scope:{ctx:h}}}),W=new z({props:{code:h[0]}}),B=new z({props:{code:h[1]}}),F=new z({props:{code:h[2]}}),G=new z({props:{code:h[3]}}),H=new z({props:{code:h[4]}}),U=new z({props:{code:h[5]}}),j=new z({props:{code:h[6]}}),K=new z({props:{code:h[7]}}),Y=new z({props:{code:h[8]}}),Z=new z({props:{code:h[9],isOutput:!0}}),J=new z({props:{code:h[10]}}),V=new Bt({props:{$$slots:{default:[ea]},$$scope:{ctx:h}}}),{c(){l=u("p"),g=m(`One of the best ways to reduce the chances of overfitting is to gather more
    data. Let's assume that we are dealing with MNIST and want to teach a neural
    net to recognize handwritten digits. If we provide the neural network with
    just ten images for training, one for each category, there is a very little
    chance, that the network will generalize and actually learn to recognize the
    digits. Instead it will memorize the specific samples. If we provide the
    network with millions of images on the other hand, the network has a smaller
    chance to memorize all those images.`),w=o(),A=u("p"),S=m(`MNIST provides 60,000 training images and 10,000 test images. This data is
    sufficient to train a good performing neral network, because the task is
    comparatively easy. In modern day deep learning this amount of data would be
    insufficient and we would be required to collect more data. Oftentimes
    collection of additional samples is not feasable and we will resort to `),y(_.$$.fragment),O=m("."),k=o(),y(P.$$.fragment),r=o(),c=u("p"),x=m(`We can for example rotate, blur or flip the images, but there are many more
    options available. You can have a look at the `),R=u("a"),Je=m("PyTorch documentation"),Qe=m(" to study the available options."),ve=o(),ne=u("p"),Xe=m(`It is not always the case that we would take the 60,000 MNIST training
    samples, apply let's say 140,000 transformations and end up with 200,000
    images for training. Often we apply random transformations to each batch of
    traning that we encounter. For example we could slightly rotate and blur
    each of the 32 images in our batch using some random parameters. That way
    our neural network never encounters the exact same image twice and has to
    learn to generalize. This the approach we are going to take with PyTorch.`),be=o(),ie=u("p"),et=m(`We are going to use the exact same model and training loop, that we used in
    the previous section, so let us focus on the parts that acutally change.`),Te=o(),oe=u("p"),tt=m("We create a simple function, that saves and displays MNIST images."),Ie=o(),y(W.$$.fragment),Ee=o(),se=u("p"),at=m("First we generate 6 non-augmented images from the training dataset."),ke=o(),y(B.$$.fragment),Pe=o(),C=u("img"),Ae=o(),D=u("p"),nt=m("We can rotate the images by using "),he=u("code"),it=m("T.RandomRotation"),ot=m(`. We use an
    angle between -30 and 30 degrees to get the following results.`),Le=o(),y(F.$$.fragment),Me=o(),q=u("img"),ze=o(),L=u("p"),st=m("We can blur the images by using "),we=u("code"),rt=m("T.GaussianBlur"),lt=m(`.
    `),y(G.$$.fragment),mt=o(),re=u("img"),Se=o(),M=u("p"),ft=m("Or we can randomly flip the images by using "),$e=u("code"),ut=m("T.RandomHorizontalFlip"),pt=m(`.
    `),y(H.$$.fragment),gt=o(),le=u("img"),Oe=o(),me=u("p"),dt=m(`There are many more different augmentation transforms available, but in this
    example we will only apply one. First apply gaussian blur to the PIL image
    and then we transform the result into a PyTorch tensor.`),xe=o(),y(U.$$.fragment),Re=o(),fe=u("p"),ct=m(`As we have created new transforms, we have to to create a new training
    dataset and dataloader.`),De=o(),y(j.$$.fragment),Ne=o(),ue=u("p"),ht=m(`It turns out that the learning rate that we used before is too large if we
    apply augmentations, so we use a reduced learning rate.`),Ve=o(),y(K.$$.fragment),We=o(),pe=u("p"),wt=m("By using augmentation we reduce overfitting significantly."),Be=o(),y(Y.$$.fragment),Ce=o(),y(Z.$$.fragment),Fe=o(),ge=u("p"),$t=m("The validation plot follows the trainig plot very closely."),qe=o(),y(J.$$.fragment),Ge=o(),Q=u("img"),He=o(),de=u("p"),_t=m(`It is relatively easy to augment image data, but it is not always easy to
    augment text or time series data. To augment text data on Kaggle for
    example, in some competitions people used google translate to translate a
    sentence into a foreign language first and then translate the sentence back
    into english. The sentence changes slightly, but is similar enough to be
    used in the training process. Sometimes you might need to get creative to
    find a good data augmentation approach.`),Ue=o(),N=u("p"),yt=m(`Before we move on to the next section let us mention that there is a
    significantly more powerful technique to deal with limited data: `),y(V.$$.fragment),vt=m(`. Tranfer learning allows you to use a model, that was pretrained on
    millions of images or millions of texts, thereby allowing you to finetune
    the model to your needs. Those types of models need significantly less data
    to learn a particular task. It makes little sense to cover transfer learning
    in detail, before we have learned convolutional neural networks or
    transformers. Once we encounter those types of networks we will discuss this
    topic in more detail.`),je=o(),ce=u("div"),this.h()},l(e){l=p(e,"P",{});var a=d(l);g=f(a,`One of the best ways to reduce the chances of overfitting is to gather more
    data. Let's assume that we are dealing with MNIST and want to teach a neural
    net to recognize handwritten digits. If we provide the neural network with
    just ten images for training, one for each category, there is a very little
    chance, that the network will generalize and actually learn to recognize the
    digits. Instead it will memorize the specific samples. If we provide the
    network with millions of images on the other hand, the network has a smaller
    chance to memorize all those images.`),a.forEach(t),w=s(e),A=p(e,"P",{});var X=d(A);S=f(X,`MNIST provides 60,000 training images and 10,000 test images. This data is
    sufficient to train a good performing neral network, because the task is
    comparatively easy. In modern day deep learning this amount of data would be
    insufficient and we would be required to collect more data. Oftentimes
    collection of additional samples is not feasable and we will resort to `),v(_.$$.fragment,X),O=f(X,"."),X.forEach(t),k=s(e),v(P.$$.fragment,e),r=s(e),c=p(e,"P",{});var ee=d(c);x=f(ee,`We can for example rotate, blur or flip the images, but there are many more
    options available. You can have a look at the `),R=p(ee,"A",{href:!0,rel:!0,target:!0});var _e=d(R);Je=f(_e,"PyTorch documentation"),_e.forEach(t),Qe=f(ee," to study the available options."),ee.forEach(t),ve=s(e),ne=p(e,"P",{});var Pt=d(ne);Xe=f(Pt,`It is not always the case that we would take the 60,000 MNIST training
    samples, apply let's say 140,000 transformations and end up with 200,000
    images for training. Often we apply random transformations to each batch of
    traning that we encounter. For example we could slightly rotate and blur
    each of the 32 images in our batch using some random parameters. That way
    our neural network never encounters the exact same image twice and has to
    learn to generalize. This the approach we are going to take with PyTorch.`),Pt.forEach(t),be=s(e),ie=p(e,"P",{});var At=d(ie);et=f(At,`We are going to use the exact same model and training loop, that we used in
    the previous section, so let us focus on the parts that acutally change.`),At.forEach(t),Te=s(e),oe=p(e,"P",{});var Lt=d(oe);tt=f(Lt,"We create a simple function, that saves and displays MNIST images."),Lt.forEach(t),Ie=s(e),v(W.$$.fragment,e),Ee=s(e),se=p(e,"P",{});var Mt=d(se);at=f(Mt,"First we generate 6 non-augmented images from the training dataset."),Mt.forEach(t),ke=s(e),v(B.$$.fragment,e),Pe=s(e),C=p(e,"IMG",{src:!0,alt:!0}),Ae=s(e),D=p(e,"P",{});var Ye=d(D);nt=f(Ye,"We can rotate the images by using "),he=p(Ye,"CODE",{});var zt=d(he);it=f(zt,"T.RandomRotation"),zt.forEach(t),ot=f(Ye,`. We use an
    angle between -30 and 30 degrees to get the following results.`),Ye.forEach(t),Le=s(e),v(F.$$.fragment,e),Me=s(e),q=p(e,"IMG",{src:!0,alt:!0}),ze=s(e),L=p(e,"P",{});var te=d(L);st=f(te,"We can blur the images by using "),we=p(te,"CODE",{});var St=d(we);rt=f(St,"T.GaussianBlur"),St.forEach(t),lt=f(te,`.
    `),v(G.$$.fragment,te),mt=s(te),re=p(te,"IMG",{src:!0,alt:!0}),te.forEach(t),Se=s(e),M=p(e,"P",{});var ae=d(M);ft=f(ae,"Or we can randomly flip the images by using "),$e=p(ae,"CODE",{});var Ot=d($e);ut=f(Ot,"T.RandomHorizontalFlip"),Ot.forEach(t),pt=f(ae,`.
    `),v(H.$$.fragment,ae),gt=s(ae),le=p(ae,"IMG",{src:!0,alt:!0}),ae.forEach(t),Oe=s(e),me=p(e,"P",{});var xt=d(me);dt=f(xt,`There are many more different augmentation transforms available, but in this
    example we will only apply one. First apply gaussian blur to the PIL image
    and then we transform the result into a PyTorch tensor.`),xt.forEach(t),xe=s(e),v(U.$$.fragment,e),Re=s(e),fe=p(e,"P",{});var Rt=d(fe);ct=f(Rt,`As we have created new transforms, we have to to create a new training
    dataset and dataloader.`),Rt.forEach(t),De=s(e),v(j.$$.fragment,e),Ne=s(e),ue=p(e,"P",{});var Dt=d(ue);ht=f(Dt,`It turns out that the learning rate that we used before is too large if we
    apply augmentations, so we use a reduced learning rate.`),Dt.forEach(t),Ve=s(e),v(K.$$.fragment,e),We=s(e),pe=p(e,"P",{});var Nt=d(pe);wt=f(Nt,"By using augmentation we reduce overfitting significantly."),Nt.forEach(t),Be=s(e),v(Y.$$.fragment,e),Ce=s(e),v(Z.$$.fragment,e),Fe=s(e),ge=p(e,"P",{});var Vt=d(ge);$t=f(Vt,"The validation plot follows the trainig plot very closely."),Vt.forEach(t),qe=s(e),v(J.$$.fragment,e),Ge=s(e),Q=p(e,"IMG",{src:!0,alt:!0}),He=s(e),de=p(e,"P",{});var Wt=d(de);_t=f(Wt,`It is relatively easy to augment image data, but it is not always easy to
    augment text or time series data. To augment text data on Kaggle for
    example, in some competitions people used google translate to translate a
    sentence into a foreign language first and then translate the sentence back
    into english. The sentence changes slightly, but is similar enough to be
    used in the training process. Sometimes you might need to get creative to
    find a good data augmentation approach.`),Wt.forEach(t),Ue=s(e),N=p(e,"P",{});var Ze=d(N);yt=f(Ze,`Before we move on to the next section let us mention that there is a
    significantly more powerful technique to deal with limited data: `),v(V.$$.fragment,Ze),vt=f(Ze,`. Tranfer learning allows you to use a model, that was pretrained on
    millions of images or millions of texts, thereby allowing you to finetune
    the model to your needs. Those types of models need significantly less data
    to learn a particular task. It makes little sense to cover transfer learning
    in detail, before we have learned convolutional neural networks or
    transformers. Once we encounter those types of networks we will discuss this
    topic in more detail.`),Ze.forEach(t),je=s(e),ce=p(e,"DIV",{class:!0}),d(ce).forEach(t),this.h()},h(){$(R,"href","https://pytorch.org/vision/stable/transforms.html"),$(R,"rel","noreferrer"),$(R,"target","_blank"),ye(C.src,bt=Kt)||$(C,"src",bt),$(C,"alt","Original MMNIST images"),ye(q.src,Tt=Jt)||$(q,"src",Tt),$(q,"alt","Rotated MMNIST images"),ye(re.src,It=Yt)||$(re,"src",It),$(re,"alt","Blurred MMNIST images"),ye(le.src,Et=Zt)||$(le,"src",Et),$(le,"alt","Flipped MMNIST images"),ye(Q.src,kt=jt)||$(Q,"src",kt),$(Q,"alt","Overfitting after augmentation"),$(ce,"class","separator")},m(e,a){n(e,l,a),i(l,g),n(e,w,a),n(e,A,a),i(A,S),b(_,A,null),i(A,O),n(e,k,a),b(P,e,a),n(e,r,a),n(e,c,a),i(c,x),i(c,R),i(R,Je),i(c,Qe),n(e,ve,a),n(e,ne,a),i(ne,Xe),n(e,be,a),n(e,ie,a),i(ie,et),n(e,Te,a),n(e,oe,a),i(oe,tt),n(e,Ie,a),b(W,e,a),n(e,Ee,a),n(e,se,a),i(se,at),n(e,ke,a),b(B,e,a),n(e,Pe,a),n(e,C,a),n(e,Ae,a),n(e,D,a),i(D,nt),i(D,he),i(he,it),i(D,ot),n(e,Le,a),b(F,e,a),n(e,Me,a),n(e,q,a),n(e,ze,a),n(e,L,a),i(L,st),i(L,we),i(we,rt),i(L,lt),b(G,L,null),i(L,mt),i(L,re),n(e,Se,a),n(e,M,a),i(M,ft),i(M,$e),i($e,ut),i(M,pt),b(H,M,null),i(M,gt),i(M,le),n(e,Oe,a),n(e,me,a),i(me,dt),n(e,xe,a),b(U,e,a),n(e,Re,a),n(e,fe,a),i(fe,ct),n(e,De,a),b(j,e,a),n(e,Ne,a),n(e,ue,a),i(ue,ht),n(e,Ve,a),b(K,e,a),n(e,We,a),n(e,pe,a),i(pe,wt),n(e,Be,a),b(Y,e,a),n(e,Ce,a),b(Z,e,a),n(e,Fe,a),n(e,ge,a),i(ge,$t),n(e,qe,a),b(J,e,a),n(e,Ge,a),n(e,Q,a),n(e,He,a),n(e,de,a),i(de,_t),n(e,Ue,a),n(e,N,a),i(N,yt),b(V,N,null),i(N,vt),n(e,je,a),n(e,ce,a),Ke=!0},p(e,a){const X={};a&2048&&(X.$$scope={dirty:a,ctx:e}),_.$set(X);const ee={};a&2048&&(ee.$$scope={dirty:a,ctx:e}),P.$set(ee);const _e={};a&2048&&(_e.$$scope={dirty:a,ctx:e}),V.$set(_e)},i(e){Ke||(T(_.$$.fragment,e),T(P.$$.fragment,e),T(W.$$.fragment,e),T(B.$$.fragment,e),T(F.$$.fragment,e),T(G.$$.fragment,e),T(H.$$.fragment,e),T(U.$$.fragment,e),T(j.$$.fragment,e),T(K.$$.fragment,e),T(Y.$$.fragment,e),T(Z.$$.fragment,e),T(J.$$.fragment,e),T(V.$$.fragment,e),Ke=!0)},o(e){I(_.$$.fragment,e),I(P.$$.fragment,e),I(W.$$.fragment,e),I(B.$$.fragment,e),I(F.$$.fragment,e),I(G.$$.fragment,e),I(H.$$.fragment,e),I(U.$$.fragment,e),I(j.$$.fragment,e),I(K.$$.fragment,e),I(Y.$$.fragment,e),I(Z.$$.fragment,e),I(J.$$.fragment,e),I(V.$$.fragment,e),Ke=!1},d(e){e&&t(l),e&&t(w),e&&t(A),E(_),e&&t(k),E(P,e),e&&t(r),e&&t(c),e&&t(ve),e&&t(ne),e&&t(be),e&&t(ie),e&&t(Te),e&&t(oe),e&&t(Ie),E(W,e),e&&t(Ee),e&&t(se),e&&t(ke),E(B,e),e&&t(Pe),e&&t(C),e&&t(Ae),e&&t(D),e&&t(Le),E(F,e),e&&t(Me),e&&t(q),e&&t(ze),e&&t(L),E(G),e&&t(Se),e&&t(M),E(H),e&&t(Oe),e&&t(me),e&&t(xe),E(U,e),e&&t(Re),e&&t(fe),e&&t(De),E(j,e),e&&t(Ne),e&&t(ue),e&&t(Ve),E(K,e),e&&t(We),e&&t(pe),e&&t(Be),E(Y,e),e&&t(Ce),E(Z,e),e&&t(Fe),e&&t(ge),e&&t(qe),E(J,e),e&&t(Ge),e&&t(Q),e&&t(He),e&&t(de),e&&t(Ue),e&&t(N),E(V),e&&t(je),e&&t(ce)}}}function aa(h){let l,g,w,A,S,_,O,k,P;return k=new Ht({props:{$$slots:{default:[ta]},$$scope:{ctx:h}}}),{c(){l=u("meta"),g=o(),w=u("h1"),A=m("Data Augmentation"),S=o(),_=u("div"),O=o(),y(k.$$.fragment),this.h()},l(r){const c=Gt("svelte-17d51sp",document.head);l=p(c,"META",{name:!0,content:!0}),c.forEach(t),g=s(r),w=p(r,"H1",{});var x=d(w);A=f(x,"Data Augmentation"),x.forEach(t),S=s(r),_=p(r,"DIV",{class:!0}),d(_).forEach(t),O=s(r),v(k.$$.fragment,r),this.h()},h(){document.title="Data Augmentation - World4AI",$(l,"name","description"),$(l,"content","One of the best ways to fight overfitting is to use more data for training, we do not always posess sufficient amounts of data. Data augmentation is a simple technique to produce synthetic data that can be used to train a neural network."),$(_,"class","separator")},m(r,c){i(document.head,l),n(r,g,c),n(r,w,c),i(w,A),n(r,S,c),n(r,_,c),n(r,O,c),b(k,r,c),P=!0},p(r,[c]){const x={};c&2048&&(x.$$scope={dirty:c,ctx:r}),k.$set(x)},i(r){P||(T(k.$$.fragment,r),P=!0)},o(r){I(k.$$.fragment,r),P=!1},d(r){t(l),r&&t(g),r&&t(w),r&&t(S),r&&t(_),r&&t(O),E(k,r)}}}function na(h){return[`# function to loop over a list of images and to draw them using matplotlib
def draw_images(images, name):
    fig = plt.figure(figsize=(10, 10))
    for i, img in enumerate(images):
        fig.add_subplot(1, len(images), i+1)
        img = img.squeeze()
        plt.imshow(img, cmap="gray")
        plt.axis('off')
    plt.savefig(f'{name}.png', bbox_inches='tight')
    plt.show()`,`# original images
images = [train_validation_dataset[i][0] for i in range(6)]
draw_images(images, 'minst_orig')`,`# rotate
transform = T.RandomRotation(degrees=(-30, 30))
transformed_images = [transform(img) for img in images]
draw_images(transformed_images, 'mnist_rotated')`,`# gaussian blur
transform = T.GaussianBlur(kernel_size=(5,5))
transformed_images = [transform(img) for img in images]
draw_images(transformed_images, 'mnist_blur')`,`# flip
transform = T.RandomHorizontalFlip(p=1)
transformed_images = [transform(img) for img in images]
draw_images(transformed_images, 'mnist_flipped')`,`transform = T.Compose([
    T.GaussianBlur(kernel_size=(5,5)),
    T.ToTensor(),
])`,`train_validation_dataset_aug = MNIST(root="../datasets/", train=True, download=True, transform=transform)
train_dataset_aug = Subset(train_validation_dataset_aug, train_idxs)
val_dataset_aug = Subset(train_validation_dataset_aug, val_idxs)


train_dataloader_aug = DataLoader(dataset=train_dataset_aug, 
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              drop_last=True,
                              num_workers=4)
train_dataloader_aug = DataLoader(dataset=val_dataset_aug, 
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              drop_last=False,
                              num_workers=4)`,`model = Model().to(DEVICE)
criterion = nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.SGD(model.parameters(), lr=0.005)`,"history = train(NUM_EPOCHS, train_dataloader_aug, train_dataloader_aug, model, criterion, optimizer)",`Epoch: 1/50|Train Loss: 0.4877 |Val Loss: 0.4859 |Train Acc: 0.8565 |Val Acc: 0.8580
Epoch: 10/50|Train Loss: 0.1616 |Val Loss: 0.1652 |Train Acc: 0.9507 |Val Acc: 0.9470
Epoch: 20/50|Train Loss: 0.1158 |Val Loss: 0.1149 |Train Acc: 0.9657 |Val Acc: 0.9633
Epoch: 30/50|Train Loss: 0.1366 |Val Loss: 0.1377 |Train Acc: 0.9578 |Val Acc: 0.9590
Epoch: 40/50|Train Loss: 0.1215 |Val Loss: 0.1187 |Train Acc: 0.9652 |Val Acc: 0.9638
Epoch: 50/50|Train Loss: 0.1265 |Val Loss: 0.1209 |Train Acc: 0.9635 |Val Acc: 0.9648
`,"plot_history(history)"]}class ma extends Ct{constructor(l){super(),Ft(this,l,na,aa,qt,{})}}export{ma as default};
