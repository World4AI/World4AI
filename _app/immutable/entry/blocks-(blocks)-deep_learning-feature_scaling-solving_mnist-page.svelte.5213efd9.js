import{S as xa,i as za,s as Aa,k as i,a as h,q as n,y as u,W as Oa,l,h as t,c as m,m as f,r,z as w,n as b,N as o,b as s,A as $,g as _,d as y,B as v,L as Wa}from"../chunks/index.4d92b023.js";import{C as Ba}from"../chunks/Container.b0705c7b.js";import{P as g}from"../chunks/PythonCode.212ba7a6.js";import{A as Fa}from"../chunks/Alert.25a852b3.js";const Ha=""+new URL("../assets/mnist.a688ff21.webp",import.meta.url).href;function Ra(c){let p;return{c(){p=n(`Not scaling input features is one of the many pitfalls you will encounter
    when you will work on your own projects. If you do not observe any progress,
    check if you have correctly scaled your input features.`)},l(I){p=r(I,`Not scaling input features is one of the many pitfalls you will encounter
    when you will work on your own projects. If you do not observe any progress,
    check if you have correctly scaled your input features.`)},m(I,E){s(I,p,E)},d(I){I&&t(p)}}}function ja(c){let p,I,E,B,F,k,S,D,P,d,L,x,H,De,z,Qe,we,Xe,et,ke,qt,Ut,tt,V,ot,N,Zt,Ne,Vt,Yt,Me,Kt,Jt,xe,Qt,Xt,at,$e,eo,st,Y,nt,K,rt,_e,to,it,J,lt,Q,Xo,ft,ye,oo,ht,X,mt,ee,ao,dt,ve,so,ct,R,no,ze,ro,io,pt,te,ut,T,lo,Ae,fo,ho,Oe,mo,co,We,po,uo,j,wo,$o,Be,_o,yo,Fe,vo,Eo,wt,Ee,go,$t,oe,_t,ge,To,yt,ae,vt,Te,bo,Et,se,gt,A,Lo,He,Io,So,Re,Po,Co,Tt,ne,bt,re,Lt,ie,It,O,Do,je,ko,No,Ge,Mo,xo,St,le,Pt,fe,Ct,G,zo,qe,Ao,Oo,Dt,he,kt,W,Wo,Ue,Bo,Fo,Ze,Ho,Ro,Nt,me,Mt,M,jo,Ve,Go,qo,Ye,Uo,Zo,Ke,Vo,Yo,xt,de,zt,ce,Ko,At,pe,Ot,ue,Jo,Wt,be,Qo,Bt,q,Ft,Le,Ht;return H=new g({props:{code:c[0]}}),V=new g({props:{code:c[1]}}),Y=new g({props:{code:c[2]}}),K=new g({props:{code:c[3]}}),J=new g({props:{code:c[4]}}),X=new g({props:{code:c[5]}}),te=new g({props:{code:c[6]}}),oe=new g({props:{code:c[7]}}),ae=new g({props:{code:c[8]}}),se=new g({props:{code:c[9]}}),ne=new g({props:{code:c[10]}}),re=new g({props:{code:c[11]}}),ie=new g({props:{code:c[12]}}),le=new g({props:{code:c[13]}}),fe=new g({props:{code:c[14]}}),he=new g({props:{code:c[15]}}),me=new g({props:{code:c[16]}}),de=new g({props:{code:c[17]}}),pe=new g({props:{code:c[18]}}),q=new Fa({props:{type:"warning",$$slots:{default:[Ra]},$$scope:{ctx:c}}}),{c(){p=i("p"),I=n(`It is tradition in the deep learning community to kick off the deep learning
    journey, by classifying handwritten digits using the `),E=i("a"),B=n("MNIST"),F=n(" dataset."),k=h(),S=i("p"),D=n("Additionally to the libraries, that we have used before we will utilize the "),P=i("a"),d=n("torchvision"),L=n(` library. Torchvision is a part of the PyTorch stack that has a lot of useful
    functions for computer vision. Especially useful are the datasets, like MNIST,
    that we can utilize out of the box without spending time collecting data on our
    own.`),x=h(),u(H.$$.fragment),De=h(),z=i("p"),Qe=n("The torchvision "),we=i("code"),Xe=n("MNIST"),et=n(` class downloads the data and returns a
    `),ke=i("code"),qt=n("Dataset"),Ut=n(" object."),tt=h(),u(V.$$.fragment),ot=h(),N=i("p"),Zt=n("The "),Ne=i("code"),Vt=n("root"),Yt=n(` attribute designates the folder where the data will be
    kept. The `),Me=i("code"),Kt=n("train"),Jt=n(` property is a boolean value. If True the object
    returns the train dataset, if False the object returns the test dataset. The
    `),xe=i("code"),Qt=n("download"),Xt=n(` property is a boolean, which designates whether the data
    should be downloaded or not. You usually need to download the data only once,
    after that it will be cached in your root folder.`),at=h(),$e=i("p"),eo=n(`Each datapoint is a tuple, consisting a PIL image and the class label.
    Labels range from 0 to 9, representing the correspoinding number of a
    handwritten digit. Images are black and white, of size 28x28 pixels.
    Alltogether there are 70,000 images, 60,000 training and 10,000 testing
    images. While this might look like a lot, modern deep learning architectures
    deal with millions of images. For the purpose of designing our first useful
    first neural network on the other hand, MNIST is the perfect dataset.`),st=h(),u(Y.$$.fragment),nt=h(),u(K.$$.fragment),rt=h(),_e=i("p"),to=n("Let's display some of the images to get a feel for what we are dealing with."),it=h(),u(J.$$.fragment),lt=h(),Q=i("img"),ft=h(),ye=i("p"),oo=n(`When we look at the minimum and maximum pixel values, we will notice that
    they range from 0 to 255.`),ht=h(),u(X.$$.fragment),mt=h(),ee=i("pre"),ao=n(`Minimum pixel value: 0
Maximum pixel value: 255
`),dt=h(),ve=i("p"),so=n(`This is the usual range that all images have. The higher the value, the
    higher the intensity. For black and white images 0 represents black value,
    256 represents white values and all the values inbetween are shades of grey.
    When we start encountering colored images, we will deal with the RGB (red
    green blue) format. Each of the 3 so called channels (red channel, green
    channel and blue channel) can have values from 0 to 255. In our case we are
    only dealing with a single channel, because we are dealing with black and
    white images. So essentially an MNIST image has the format (1, 28, 28) and
    the batch of MNIST images, given a batch size of 32, will have a shape of
    (32, 1, 28, 28). This format is often abbreviated as (B, C, H, W), which
    stands for batch size, channels, hight, width.`),ct=h(),R=i("p"),no=n(`When it comes to computer vision, PyTorch provides scaling capabilities out
    of the box in `),ze=i("code"),ro=n("torchvision.transforms"),io=n("."),pt=h(),u(te.$$.fragment),ut=h(),T=i("p"),lo=n("When we create a dataset using the "),Ae=i("code"),fo=n("MNIST"),ho=n(` class, we can pass a
    `),Oe=i("code"),mo=n("transform"),co=n(`
    argument. As the name suggests we can apply a transform to images, before
    using those values for training. For example if we use the
    `),We=i("code"),po=n("PILToTensor"),uo=n(`
    transform, we transform the data from PIL format to a tensor format.
    Torchvision provides a great number of transforms, see
    `),j=i("a"),wo=n("Torchvision Docs"),$o=n(`, but sometimes you might want more control. For that purpose you can use
    `),Be=i("code"),_o=n("transforms.Lambda()"),yo=n(`, which takes a Python lambda function, in
    which you can process images as you desire. Often you will need to apply
    more than one transform. For that you can concatenate transforms using
    `),Fe=i("code"),vo=n("transform.Compose([transform1,transform2,...])"),Eo=n(`. Below we
    prepare two sets of transforms. One set contains feature scaling, the other
    does not. We will both apply to MNIST and compare the results.`),wt=h(),Ee=i("p"),go=n(`The first set of transforms first transforms the PIL image into a Tensor and
    then turns the Tensor into a float32 data format. Both steps are important,
    because PyTorch can only work with tensors and as we intend to use the GPU,
    float32 is required.`),$t=h(),u(oe.$$.fragment),_t=h(),ge=i("p"),To=n(`Those transforms do not include any form of scaling, therefore we expect the
    training to be relatively slow.`),yt=h(),u(ae.$$.fragment),vt=h(),Te=i("p"),bo=n(`Below we calculate the mean and the standard deviation of the images pixel
    values. You will notice that there is only one mean and std and not 784
    (28*28 pixels). That is because in computer vision the scaling is done per
    channel and not per pixel. If we were dealing with color images, we would
    have 3 channels and would therefore require 3 mean and std calculations.`),Et=h(),u(se.$$.fragment),gt=h(),A=i("p"),Lo=n("The second set of transforms first applies "),He=i("code"),Io=n("transforms.ToTensor"),So=n(`
    which turns the PIL image into a float32 Tensor and scales the image into a
    0-1 range. The `),Re=i("code"),Po=n("transforms.Normalize"),Co=n(` transform conducts what we call
    standardization or z-score normalization. The procedure essentially subracts
    the mean and divides by the standard deviation. If you have a color image with
    3 channels, you need to provide a tuple of mean and std values, 1 for each channel.`),Tt=h(),u(ne.$$.fragment),bt=h(),u(re.$$.fragment),Lt=h(),u(ie.$$.fragment),It=h(),O=i("p"),Do=n("Based on the datasets we create two dataloaders: "),je=i("code"),ko=n("dataloader_orig"),No=n(`
    without scaling and `),Ge=i("code"),Mo=n("dataloader_normalized"),xo=n(" with scaling."),St=h(),u(le.$$.fragment),Pt=h(),u(fe.$$.fragment),Ct=h(),G=i("p"),zo=n("The "),qe=i("code"),Ao=n("train"),Oo=n(` function is the same generic function that we used in
    the previous PyTorch tutorials.`),Dt=h(),u(he.$$.fragment),kt=h(),W=i("p"),Wo=n("The "),Ue=i("code"),Bo=n("Model"),Fo=n(` class is slighly different. Our batch has the shape
    (32, 1, 28, 28), but fully connected neural networks need a flat tensor of
    shape (31, 784). We essentially need to create a large vector out of all
    rows of the image. The layer `),Ze=i("code"),Ho=n("nn.Flatten()"),Ro=n(` does just that. Our output
    layer consists of 10 neurons this time. This is due to the fact, that we have
    10 labels and we need ten neurons which are used as input into the softmax activation
    function. We do not explicitly define the softmax layer as part of the model,
    because our loss function will combine the softmax with the cross-entropy loss.`),Nt=h(),u(me.$$.fragment),Mt=h(),M=i("p"),jo=n(`Below we train the same model with and without feature scaling and compare
    the results. The `),Ve=i("code"),Go=n("CrossEntropyLoss"),qo=n(` criterion stacks the log
    softmax activation function and the cross-entropy loss. This log version of
    the softmax activation and the combination of the activation with the loss
    is useful for numerical stability. Theoretically you can explicitly add the
    `),Ye=i("code"),Uo=n("nn.LogSoftmax"),Zo=n(`
    activation to your model and use the `),Ke=i("code"),Vo=n("nn.NLLLoss"),Yo=n(`, but that is
    not recommended.`),xt=h(),u(de.$$.fragment),zt=h(),ce=i("pre"),Ko=n(`Epoch: 1 Loss: 0.97742760181427
Epoch: 2 Loss: 0.7255294919013977
Epoch: 3 Loss: 0.7582691311836243
Epoch: 4 Loss: 0.6830052733421326
Epoch: 5 Loss: 0.6659824252128601
Epoch: 6 Loss: 0.6156877875328064
Epoch: 7 Loss: 0.6003748178482056
Epoch: 8 Loss: 0.5670294165611267
Epoch: 9 Loss: 0.6026986837387085
Epoch: 10 Loss: 0.5925905108451843
  `),At=h(),u(pe.$$.fragment),Ot=h(),ue=i("pre"),Jo=n(`Epoch: 1 Loss: 0.7985861897468567
Epoch: 2 Loss: 0.2571895718574524
Epoch: 3 Loss: 0.17698505520820618
Epoch: 4 Loss: 0.1328950673341751
Epoch: 5 Loss: 0.1063883826136589
Epoch: 6 Loss: 0.08727587759494781
Epoch: 7 Loss: 0.0743139460682869
Epoch: 8 Loss: 0.06442411243915558
Epoch: 9 Loss: 0.05526750162243843
Epoch: 10 Loss: 0.047709111124277115
  `),Wt=h(),be=i("p"),Qo=n(`The difference is huge. Without feature scaling training is slow and the
    loss oscilates from time to time. Training with feature scaling on the other
    hand decreases the loss dramatically.`),Bt=h(),u(q.$$.fragment),Ft=h(),Le=i("div"),this.h()},l(e){p=l(e,"P",{});var a=f(p);I=r(a,`It is tradition in the deep learning community to kick off the deep learning
    journey, by classifying handwritten digits using the `),E=l(a,"A",{href:!0,rel:!0,target:!0});var Je=f(E);B=r(Je,"MNIST"),Je.forEach(t),F=r(a," dataset."),a.forEach(t),k=m(e),S=l(e,"P",{});var Rt=f(S);D=r(Rt,"Additionally to the libraries, that we have used before we will utilize the "),P=l(Rt,"A",{href:!0,target:!0,rel:!0});var ea=f(P);d=r(ea,"torchvision"),ea.forEach(t),L=r(Rt,` library. Torchvision is a part of the PyTorch stack that has a lot of useful
    functions for computer vision. Especially useful are the datasets, like MNIST,
    that we can utilize out of the box without spending time collecting data on our
    own.`),Rt.forEach(t),x=m(e),w(H.$$.fragment,e),De=m(e),z=l(e,"P",{});var Ie=f(z);Qe=r(Ie,"The torchvision "),we=l(Ie,"CODE",{});var ta=f(we);Xe=r(ta,"MNIST"),ta.forEach(t),et=r(Ie,` class downloads the data and returns a
    `),ke=l(Ie,"CODE",{});var oa=f(ke);qt=r(oa,"Dataset"),oa.forEach(t),Ut=r(Ie," object."),Ie.forEach(t),tt=m(e),w(V.$$.fragment,e),ot=m(e),N=l(e,"P",{});var U=f(N);Zt=r(U,"The "),Ne=l(U,"CODE",{});var aa=f(Ne);Vt=r(aa,"root"),aa.forEach(t),Yt=r(U,` attribute designates the folder where the data will be
    kept. The `),Me=l(U,"CODE",{});var sa=f(Me);Kt=r(sa,"train"),sa.forEach(t),Jt=r(U,` property is a boolean value. If True the object
    returns the train dataset, if False the object returns the test dataset. The
    `),xe=l(U,"CODE",{});var na=f(xe);Qt=r(na,"download"),na.forEach(t),Xt=r(U,` property is a boolean, which designates whether the data
    should be downloaded or not. You usually need to download the data only once,
    after that it will be cached in your root folder.`),U.forEach(t),at=m(e),$e=l(e,"P",{});var ra=f($e);eo=r(ra,`Each datapoint is a tuple, consisting a PIL image and the class label.
    Labels range from 0 to 9, representing the correspoinding number of a
    handwritten digit. Images are black and white, of size 28x28 pixels.
    Alltogether there are 70,000 images, 60,000 training and 10,000 testing
    images. While this might look like a lot, modern deep learning architectures
    deal with millions of images. For the purpose of designing our first useful
    first neural network on the other hand, MNIST is the perfect dataset.`),ra.forEach(t),st=m(e),w(Y.$$.fragment,e),nt=m(e),w(K.$$.fragment,e),rt=m(e),_e=l(e,"P",{});var ia=f(_e);to=r(ia,"Let's display some of the images to get a feel for what we are dealing with."),ia.forEach(t),it=m(e),w(J.$$.fragment,e),lt=m(e),Q=l(e,"IMG",{alt:!0,src:!0}),ft=m(e),ye=l(e,"P",{});var la=f(ye);oo=r(la,`When we look at the minimum and maximum pixel values, we will notice that
    they range from 0 to 255.`),la.forEach(t),ht=m(e),w(X.$$.fragment,e),mt=m(e),ee=l(e,"PRE",{class:!0});var fa=f(ee);ao=r(fa,`Minimum pixel value: 0
Maximum pixel value: 255
`),fa.forEach(t),dt=m(e),ve=l(e,"P",{});var ha=f(ve);so=r(ha,`This is the usual range that all images have. The higher the value, the
    higher the intensity. For black and white images 0 represents black value,
    256 represents white values and all the values inbetween are shades of grey.
    When we start encountering colored images, we will deal with the RGB (red
    green blue) format. Each of the 3 so called channels (red channel, green
    channel and blue channel) can have values from 0 to 255. In our case we are
    only dealing with a single channel, because we are dealing with black and
    white images. So essentially an MNIST image has the format (1, 28, 28) and
    the batch of MNIST images, given a batch size of 32, will have a shape of
    (32, 1, 28, 28). This format is often abbreviated as (B, C, H, W), which
    stands for batch size, channels, hight, width.`),ha.forEach(t),ct=m(e),R=l(e,"P",{});var jt=f(R);no=r(jt,`When it comes to computer vision, PyTorch provides scaling capabilities out
    of the box in `),ze=l(jt,"CODE",{});var ma=f(ze);ro=r(ma,"torchvision.transforms"),ma.forEach(t),io=r(jt,"."),jt.forEach(t),pt=m(e),w(te.$$.fragment,e),ut=m(e),T=l(e,"P",{});var C=f(T);lo=r(C,"When we create a dataset using the "),Ae=l(C,"CODE",{});var da=f(Ae);fo=r(da,"MNIST"),da.forEach(t),ho=r(C,` class, we can pass a
    `),Oe=l(C,"CODE",{});var ca=f(Oe);mo=r(ca,"transform"),ca.forEach(t),co=r(C,`
    argument. As the name suggests we can apply a transform to images, before
    using those values for training. For example if we use the
    `),We=l(C,"CODE",{});var pa=f(We);po=r(pa,"PILToTensor"),pa.forEach(t),uo=r(C,`
    transform, we transform the data from PIL format to a tensor format.
    Torchvision provides a great number of transforms, see
    `),j=l(C,"A",{href:!0,target:!0,rel:!0});var ua=f(j);wo=r(ua,"Torchvision Docs"),ua.forEach(t),$o=r(C,`, but sometimes you might want more control. For that purpose you can use
    `),Be=l(C,"CODE",{});var wa=f(Be);_o=r(wa,"transforms.Lambda()"),wa.forEach(t),yo=r(C,`, which takes a Python lambda function, in
    which you can process images as you desire. Often you will need to apply
    more than one transform. For that you can concatenate transforms using
    `),Fe=l(C,"CODE",{});var $a=f(Fe);vo=r($a,"transform.Compose([transform1,transform2,...])"),$a.forEach(t),Eo=r(C,`. Below we
    prepare two sets of transforms. One set contains feature scaling, the other
    does not. We will both apply to MNIST and compare the results.`),C.forEach(t),wt=m(e),Ee=l(e,"P",{});var _a=f(Ee);go=r(_a,`The first set of transforms first transforms the PIL image into a Tensor and
    then turns the Tensor into a float32 data format. Both steps are important,
    because PyTorch can only work with tensors and as we intend to use the GPU,
    float32 is required.`),_a.forEach(t),$t=m(e),w(oe.$$.fragment,e),_t=m(e),ge=l(e,"P",{});var ya=f(ge);To=r(ya,`Those transforms do not include any form of scaling, therefore we expect the
    training to be relatively slow.`),ya.forEach(t),yt=m(e),w(ae.$$.fragment,e),vt=m(e),Te=l(e,"P",{});var va=f(Te);bo=r(va,`Below we calculate the mean and the standard deviation of the images pixel
    values. You will notice that there is only one mean and std and not 784
    (28*28 pixels). That is because in computer vision the scaling is done per
    channel and not per pixel. If we were dealing with color images, we would
    have 3 channels and would therefore require 3 mean and std calculations.`),va.forEach(t),Et=m(e),w(se.$$.fragment,e),gt=m(e),A=l(e,"P",{});var Se=f(A);Lo=r(Se,"The second set of transforms first applies "),He=l(Se,"CODE",{});var Ea=f(He);Io=r(Ea,"transforms.ToTensor"),Ea.forEach(t),So=r(Se,`
    which turns the PIL image into a float32 Tensor and scales the image into a
    0-1 range. The `),Re=l(Se,"CODE",{});var ga=f(Re);Po=r(ga,"transforms.Normalize"),ga.forEach(t),Co=r(Se,` transform conducts what we call
    standardization or z-score normalization. The procedure essentially subracts
    the mean and divides by the standard deviation. If you have a color image with
    3 channels, you need to provide a tuple of mean and std values, 1 for each channel.`),Se.forEach(t),Tt=m(e),w(ne.$$.fragment,e),bt=m(e),w(re.$$.fragment,e),Lt=m(e),w(ie.$$.fragment,e),It=m(e),O=l(e,"P",{});var Pe=f(O);Do=r(Pe,"Based on the datasets we create two dataloaders: "),je=l(Pe,"CODE",{});var Ta=f(je);ko=r(Ta,"dataloader_orig"),Ta.forEach(t),No=r(Pe,`
    without scaling and `),Ge=l(Pe,"CODE",{});var ba=f(Ge);Mo=r(ba,"dataloader_normalized"),ba.forEach(t),xo=r(Pe," with scaling."),Pe.forEach(t),St=m(e),w(le.$$.fragment,e),Pt=m(e),w(fe.$$.fragment,e),Ct=m(e),G=l(e,"P",{});var Gt=f(G);zo=r(Gt,"The "),qe=l(Gt,"CODE",{});var La=f(qe);Ao=r(La,"train"),La.forEach(t),Oo=r(Gt,` function is the same generic function that we used in
    the previous PyTorch tutorials.`),Gt.forEach(t),Dt=m(e),w(he.$$.fragment,e),kt=m(e),W=l(e,"P",{});var Ce=f(W);Wo=r(Ce,"The "),Ue=l(Ce,"CODE",{});var Ia=f(Ue);Bo=r(Ia,"Model"),Ia.forEach(t),Fo=r(Ce,` class is slighly different. Our batch has the shape
    (32, 1, 28, 28), but fully connected neural networks need a flat tensor of
    shape (31, 784). We essentially need to create a large vector out of all
    rows of the image. The layer `),Ze=l(Ce,"CODE",{});var Sa=f(Ze);Ho=r(Sa,"nn.Flatten()"),Sa.forEach(t),Ro=r(Ce,` does just that. Our output
    layer consists of 10 neurons this time. This is due to the fact, that we have
    10 labels and we need ten neurons which are used as input into the softmax activation
    function. We do not explicitly define the softmax layer as part of the model,
    because our loss function will combine the softmax with the cross-entropy loss.`),Ce.forEach(t),Nt=m(e),w(me.$$.fragment,e),Mt=m(e),M=l(e,"P",{});var Z=f(M);jo=r(Z,`Below we train the same model with and without feature scaling and compare
    the results. The `),Ve=l(Z,"CODE",{});var Pa=f(Ve);Go=r(Pa,"CrossEntropyLoss"),Pa.forEach(t),qo=r(Z,` criterion stacks the log
    softmax activation function and the cross-entropy loss. This log version of
    the softmax activation and the combination of the activation with the loss
    is useful for numerical stability. Theoretically you can explicitly add the
    `),Ye=l(Z,"CODE",{});var Ca=f(Ye);Uo=r(Ca,"nn.LogSoftmax"),Ca.forEach(t),Zo=r(Z,`
    activation to your model and use the `),Ke=l(Z,"CODE",{});var Da=f(Ke);Vo=r(Da,"nn.NLLLoss"),Da.forEach(t),Yo=r(Z,`, but that is
    not recommended.`),Z.forEach(t),xt=m(e),w(de.$$.fragment,e),zt=m(e),ce=l(e,"PRE",{class:!0});var ka=f(ce);Ko=r(ka,`Epoch: 1 Loss: 0.97742760181427
Epoch: 2 Loss: 0.7255294919013977
Epoch: 3 Loss: 0.7582691311836243
Epoch: 4 Loss: 0.6830052733421326
Epoch: 5 Loss: 0.6659824252128601
Epoch: 6 Loss: 0.6156877875328064
Epoch: 7 Loss: 0.6003748178482056
Epoch: 8 Loss: 0.5670294165611267
Epoch: 9 Loss: 0.6026986837387085
Epoch: 10 Loss: 0.5925905108451843
  `),ka.forEach(t),At=m(e),w(pe.$$.fragment,e),Ot=m(e),ue=l(e,"PRE",{class:!0});var Na=f(ue);Jo=r(Na,`Epoch: 1 Loss: 0.7985861897468567
Epoch: 2 Loss: 0.2571895718574524
Epoch: 3 Loss: 0.17698505520820618
Epoch: 4 Loss: 0.1328950673341751
Epoch: 5 Loss: 0.1063883826136589
Epoch: 6 Loss: 0.08727587759494781
Epoch: 7 Loss: 0.0743139460682869
Epoch: 8 Loss: 0.06442411243915558
Epoch: 9 Loss: 0.05526750162243843
Epoch: 10 Loss: 0.047709111124277115
  `),Na.forEach(t),Wt=m(e),be=l(e,"P",{});var Ma=f(be);Qo=r(Ma,`The difference is huge. Without feature scaling training is slow and the
    loss oscilates from time to time. Training with feature scaling on the other
    hand decreases the loss dramatically.`),Ma.forEach(t),Bt=m(e),w(q.$$.fragment,e),Ft=m(e),Le=l(e,"DIV",{class:!0}),f(Le).forEach(t),this.h()},h(){b(E,"href","http://yann.lecun.com/exdb/mnist/"),b(E,"rel","noreferrer"),b(E,"target","_blank"),b(P,"href","https://pytorch.org/vision/stable/index.html"),b(P,"target","_blank"),b(P,"rel","noreferrer"),b(Q,"alt","5 MNIST images"),Wa(Q.src,Xo=Ha)||b(Q,"src",Xo),b(ee,"class","text-sm"),b(j,"href","https://pytorch.org/vision/stable/transforms.html#"),b(j,"target","_blank"),b(j,"rel","noreferrer"),b(ce,"class","text-sm"),b(ue,"class","text-sm"),b(Le,"class","separator")},m(e,a){s(e,p,a),o(p,I),o(p,E),o(E,B),o(p,F),s(e,k,a),s(e,S,a),o(S,D),o(S,P),o(P,d),o(S,L),s(e,x,a),$(H,e,a),s(e,De,a),s(e,z,a),o(z,Qe),o(z,we),o(we,Xe),o(z,et),o(z,ke),o(ke,qt),o(z,Ut),s(e,tt,a),$(V,e,a),s(e,ot,a),s(e,N,a),o(N,Zt),o(N,Ne),o(Ne,Vt),o(N,Yt),o(N,Me),o(Me,Kt),o(N,Jt),o(N,xe),o(xe,Qt),o(N,Xt),s(e,at,a),s(e,$e,a),o($e,eo),s(e,st,a),$(Y,e,a),s(e,nt,a),$(K,e,a),s(e,rt,a),s(e,_e,a),o(_e,to),s(e,it,a),$(J,e,a),s(e,lt,a),s(e,Q,a),s(e,ft,a),s(e,ye,a),o(ye,oo),s(e,ht,a),$(X,e,a),s(e,mt,a),s(e,ee,a),o(ee,ao),s(e,dt,a),s(e,ve,a),o(ve,so),s(e,ct,a),s(e,R,a),o(R,no),o(R,ze),o(ze,ro),o(R,io),s(e,pt,a),$(te,e,a),s(e,ut,a),s(e,T,a),o(T,lo),o(T,Ae),o(Ae,fo),o(T,ho),o(T,Oe),o(Oe,mo),o(T,co),o(T,We),o(We,po),o(T,uo),o(T,j),o(j,wo),o(T,$o),o(T,Be),o(Be,_o),o(T,yo),o(T,Fe),o(Fe,vo),o(T,Eo),s(e,wt,a),s(e,Ee,a),o(Ee,go),s(e,$t,a),$(oe,e,a),s(e,_t,a),s(e,ge,a),o(ge,To),s(e,yt,a),$(ae,e,a),s(e,vt,a),s(e,Te,a),o(Te,bo),s(e,Et,a),$(se,e,a),s(e,gt,a),s(e,A,a),o(A,Lo),o(A,He),o(He,Io),o(A,So),o(A,Re),o(Re,Po),o(A,Co),s(e,Tt,a),$(ne,e,a),s(e,bt,a),$(re,e,a),s(e,Lt,a),$(ie,e,a),s(e,It,a),s(e,O,a),o(O,Do),o(O,je),o(je,ko),o(O,No),o(O,Ge),o(Ge,Mo),o(O,xo),s(e,St,a),$(le,e,a),s(e,Pt,a),$(fe,e,a),s(e,Ct,a),s(e,G,a),o(G,zo),o(G,qe),o(qe,Ao),o(G,Oo),s(e,Dt,a),$(he,e,a),s(e,kt,a),s(e,W,a),o(W,Wo),o(W,Ue),o(Ue,Bo),o(W,Fo),o(W,Ze),o(Ze,Ho),o(W,Ro),s(e,Nt,a),$(me,e,a),s(e,Mt,a),s(e,M,a),o(M,jo),o(M,Ve),o(Ve,Go),o(M,qo),o(M,Ye),o(Ye,Uo),o(M,Zo),o(M,Ke),o(Ke,Vo),o(M,Yo),s(e,xt,a),$(de,e,a),s(e,zt,a),s(e,ce,a),o(ce,Ko),s(e,At,a),$(pe,e,a),s(e,Ot,a),s(e,ue,a),o(ue,Jo),s(e,Wt,a),s(e,be,a),o(be,Qo),s(e,Bt,a),$(q,e,a),s(e,Ft,a),s(e,Le,a),Ht=!0},p(e,a){const Je={};a&524288&&(Je.$$scope={dirty:a,ctx:e}),q.$set(Je)},i(e){Ht||(_(H.$$.fragment,e),_(V.$$.fragment,e),_(Y.$$.fragment,e),_(K.$$.fragment,e),_(J.$$.fragment,e),_(X.$$.fragment,e),_(te.$$.fragment,e),_(oe.$$.fragment,e),_(ae.$$.fragment,e),_(se.$$.fragment,e),_(ne.$$.fragment,e),_(re.$$.fragment,e),_(ie.$$.fragment,e),_(le.$$.fragment,e),_(fe.$$.fragment,e),_(he.$$.fragment,e),_(me.$$.fragment,e),_(de.$$.fragment,e),_(pe.$$.fragment,e),_(q.$$.fragment,e),Ht=!0)},o(e){y(H.$$.fragment,e),y(V.$$.fragment,e),y(Y.$$.fragment,e),y(K.$$.fragment,e),y(J.$$.fragment,e),y(X.$$.fragment,e),y(te.$$.fragment,e),y(oe.$$.fragment,e),y(ae.$$.fragment,e),y(se.$$.fragment,e),y(ne.$$.fragment,e),y(re.$$.fragment,e),y(ie.$$.fragment,e),y(le.$$.fragment,e),y(fe.$$.fragment,e),y(he.$$.fragment,e),y(me.$$.fragment,e),y(de.$$.fragment,e),y(pe.$$.fragment,e),y(q.$$.fragment,e),Ht=!1},d(e){e&&t(p),e&&t(k),e&&t(S),e&&t(x),v(H,e),e&&t(De),e&&t(z),e&&t(tt),v(V,e),e&&t(ot),e&&t(N),e&&t(at),e&&t($e),e&&t(st),v(Y,e),e&&t(nt),v(K,e),e&&t(rt),e&&t(_e),e&&t(it),v(J,e),e&&t(lt),e&&t(Q),e&&t(ft),e&&t(ye),e&&t(ht),v(X,e),e&&t(mt),e&&t(ee),e&&t(dt),e&&t(ve),e&&t(ct),e&&t(R),e&&t(pt),v(te,e),e&&t(ut),e&&t(T),e&&t(wt),e&&t(Ee),e&&t($t),v(oe,e),e&&t(_t),e&&t(ge),e&&t(yt),v(ae,e),e&&t(vt),e&&t(Te),e&&t(Et),v(se,e),e&&t(gt),e&&t(A),e&&t(Tt),v(ne,e),e&&t(bt),v(re,e),e&&t(Lt),v(ie,e),e&&t(It),e&&t(O),e&&t(St),v(le,e),e&&t(Pt),v(fe,e),e&&t(Ct),e&&t(G),e&&t(Dt),v(he,e),e&&t(kt),e&&t(W),e&&t(Nt),v(me,e),e&&t(Mt),e&&t(M),e&&t(xt),v(de,e),e&&t(zt),e&&t(ce),e&&t(At),v(pe,e),e&&t(Ot),e&&t(ue),e&&t(Wt),e&&t(be),e&&t(Bt),v(q,e),e&&t(Ft),e&&t(Le)}}}function Ga(c){let p,I,E,B,F,k,S,D,P;return D=new Ba({props:{$$slots:{default:[ja]},$$scope:{ctx:c}}}),{c(){p=i("meta"),I=h(),E=i("h1"),B=n("Solving MNIST"),F=h(),k=i("div"),S=h(),u(D.$$.fragment),this.h()},l(d){const L=Oa("svelte-1gyhvpq",document.head);p=l(L,"META",{name:!0,content:!0}),L.forEach(t),I=m(d),E=l(d,"H1",{});var x=f(E);B=r(x,"Solving MNIST"),x.forEach(t),F=m(d),k=l(d,"DIV",{class:!0}),f(k).forEach(t),S=m(d),w(D.$$.fragment,d),this.h()},h(){document.title="MNIST Feature Scaling - World4AI",b(p,"name","description"),b(p,"content","The MNIST handwritten classification problem is considered to be one of the introductory problems of deep learning. Yet even for such a simple task, feature scaling is a must. Luckily PyTorch makes it easy for us to apply feature scaling to computer vision tasks."),b(k,"class","separator")},m(d,L){o(document.head,p),s(d,I,L),s(d,E,L),o(E,B),s(d,F,L),s(d,k,L),s(d,S,L),$(D,d,L),P=!0},p(d,[L]){const x={};L&524288&&(x.$$scope={dirty:L,ctx:d}),D.$set(x)},i(d){P||(_(D.$$.fragment,d),P=!0)},o(d){y(D.$$.fragment,d),P=!1},d(d){t(p),d&&t(I),d&&t(E),d&&t(F),d&&t(k),d&&t(S),v(D,d)}}}function qa(c){const p=`import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader`,I=`train_dataset = MNIST(root="../datasets", train=True, download=True)
test_dataset = MNIST(root="../datasets", train=False, download=False)`,E=String.raw`print(f'A training sample is a tuple: \n{train_dataset[0]}')
print(f'There are {len(train_dataset)} training samples.')
print(f'There are {len(test_dataset)} testing samples.')
img = np.array(train_dataset[0][0])
print(f'The shape of images is: {img.shape}')`,B=String.raw`A training sample is a tuple: 
(<PIL.Image.Image image mode=L size=28x28 at 0x7FCB9FA9E980>, 5) There are 60000 training samples.
There are 10000 testing samples.
The shape of images is: (28, 28)
`;return[p,I,E,B,`fig = plt.figure(figsize=(10, 10))
for i in range(6):
    fig.add_subplot(1, 6, i+1)
    img = np.array(train_dataset[i][0])
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.title(f'Class Nr. {train_dataset[i][1]}')
plt.show()`,`print(f'Minimum pixel value: {img.min()}')
print(f'Maximum pixel value: {img.max()}')`,"import torchvision.transforms as T",`transform = T.Compose([T.PILToTensor(), 
                       T.Lambda(lambda tensor : tensor.to(torch.float32))
])`,'dataset_orig = MNIST(root="../datasets/", train=True, download=True, transform=transform)',`# calculate mean and std
# we will need this part later for normalization
# we divide by 255.0, because the images will be transformed into the 0-1 range automatically
mean = (dataset_orig.data.float() / 255.0).mean()
std = (dataset_orig.data.float() / 255.0).std()`,`transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=mean, std=std)
])`,'dataset_normalized = MNIST(root="../datasets/", train=True, download=True, transform=transform)',`# parameters
DEVICE = ("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS=10
BATCH_SIZE=32

#number of hidden units in the first and second hidden layer
HIDDEN_SIZE_1 = 100
HIDDEN_SIZE_2 = 50
NUM_LABELS = 10
ALPHA = 0.1`,`dataloader_orig = DataLoader(dataset=dataset_orig, 
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              drop_last=True,
                              num_workers=4)`,`dataloader_normalized = DataLoader(dataset=dataset_normalized, 
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              drop_last=True,
                              num_workers=4)`,`def train(dataloader, model, criterion, optimizer):
    for epoch in range(NUM_EPOCHS):
        loss_sum = 0
        batch_nums = 0
        for batch_idx, (features, labels) in enumerate(dataloader):
            # move features and labels to GPU
            features = features.to(DEVICE)
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
            batch_nums += 1
            loss_sum += loss.detach().cpu()

        print(f'Epoch: {epoch+1} Loss: {loss_sum / batch_nums}')`,`class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(28*28, HIDDEN_SIZE_1),
                nn.Sigmoid(),
                nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
                nn.Sigmoid(),
                nn.Linear(HIDDEN_SIZE_2, NUM_LABELS),
            )
    
    def forward(self, features):
        return self.layers(features)`,`model = Model().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=ALPHA)
train(dataloader_orig, model, criterion, optimizer)`,`model = Model().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=ALPHA)
train(dataloader_normalized, model, criterion, optimizer)`]}class Ka extends xa{constructor(p){super(),za(this,p,qa,Ga,Aa,{})}}export{Ka as default};
