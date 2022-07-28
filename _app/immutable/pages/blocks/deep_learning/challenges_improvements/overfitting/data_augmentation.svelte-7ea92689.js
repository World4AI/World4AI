import{S as ee,i as te,s as ae,l as w,a as $,r as d,w as V,T as ne,m as y,h as t,c as b,n as v,u as p,x as G,p as P,G as g,b as o,y as j,f as J,t as Q,B as R}from"../../../../../chunks/index-caa95cd4.js";import{C as ie}from"../../../../../chunks/Container-5c6b7f6d.js";import{H as Z}from"../../../../../chunks/Highlight-284f0e69.js";/* empty css                                                                    */function oe(T){let n;return{c(){n=d("data augmentation")},l(s){n=p(s,"data augmentation")},m(s,l){o(s,n,l)},d(s){s&&t(n)}}}function se(T){let n;return{c(){n=d("transfer learning")},l(s){n=p(s,"transfer learning")},m(s,l){o(s,n,l)},d(s){s&&t(n)}}}function re(T){let n,s,l,u,_,m,k,f,c,a,r,h,C,O,z,H,M,D,L,N,x,F,I,K,A,S,W;return m=new Z({props:{$$slots:{default:[oe]},$$scope:{ctx:T}}}),I=new Z({props:{$$slots:{default:[se]},$$scope:{ctx:T}}}),{c(){n=w("p"),s=d(`One of the best ways to reduce the chances of overfitting is to gather more
    data. Lets assume we are dealing with MNIST and want to teach a neural net
    to recognize hand written digits. If we provide the neural network with a
    limited amount of data, there is a very little chance, that the network will
    learn to recognize the digits. Instead it will memorize the specific
    samples. If we provide the network with millions of images, the network has
    a smaller chance to memorize all those images.`),l=$(),u=w("p"),_=d(`MNIST provides 60,000 training images and 10,000 test images. This data is
    sufficient to train a good performing neral network, because the task is
    comparatively easy. In modern day deep learning this amount of data would be
    insufficient and we would be required to collect more data. Oftentimes
    collection of additional samples is not feasable and we will resort to `),V(m.$$.fragment),k=d("."),f=$(),c=w("p"),a=d(`Data augmentation is a techinque that applies transformations to the
    original dataset, thereby creating synthetic data, that can be used in
    training.`),r=$(),h=w("p"),C=d(`We could for example rotate, blur or flip the images, but there are many
    more options available, that we will cover in our practical PyTorch
    excercises.`),O=$(),z=w("p"),H=d(`It is not always the case that we would take the 60,000 MNIST training
    samples, apply let's say 140,000 transformations and end up with 200,000
    images for training. Often we apply random transformations to each batch of
    traning that we encounter. For example we could slightly rotate and blur
    each of the 32 images in our batch using some random parameters. That way
    our neural network never encounters the exact same image twice and has to
    learn to generalize.`),M=$(),D=w("p"),L=d(`It is relatively easy to augment image data, but it is not always easy to
    augment text or time series data. To augment text data on Kaggle for
    example, in some competitions people used google translate to translate a
    sentence into a foreign language first and then translate the sentence back
    into english. The sentence changes slightly, but is similar enough to be
    used in the training process. Sometimes you might need to get creative to
    find a good data augmentation approach.`),N=$(),x=w("p"),F=d(`Before we move on to the next section let us mention that there is a
    significantly more powerful technique to deal with limited data: `),V(I.$$.fragment),K=d(`. Tranfer learning allows you to use a model, that was pretrained on
    millions of images or millions of texts, thereby allowing you to finetune
    the model to your needs. Those types of models need significantly less data
    to learn a particular task. It makes little sense to cover transfer learning
    in detail, before we have learned convolutional neural networks or
    transformers. Once we encounter those types of networks we will discuss this
    topic in more detail.`),A=$(),S=w("div"),this.h()},l(e){n=y(e,"P",{});var i=v(n);s=p(i,`One of the best ways to reduce the chances of overfitting is to gather more
    data. Lets assume we are dealing with MNIST and want to teach a neural net
    to recognize hand written digits. If we provide the neural network with a
    limited amount of data, there is a very little chance, that the network will
    learn to recognize the digits. Instead it will memorize the specific
    samples. If we provide the network with millions of images, the network has
    a smaller chance to memorize all those images.`),i.forEach(t),l=b(e),u=y(e,"P",{});var E=v(u);_=p(E,`MNIST provides 60,000 training images and 10,000 test images. This data is
    sufficient to train a good performing neral network, because the task is
    comparatively easy. In modern day deep learning this amount of data would be
    insufficient and we would be required to collect more data. Oftentimes
    collection of additional samples is not feasable and we will resort to `),G(m.$$.fragment,E),k=p(E,"."),E.forEach(t),f=b(e),c=y(e,"P",{class:!0});var q=v(c);a=p(q,`Data augmentation is a techinque that applies transformations to the
    original dataset, thereby creating synthetic data, that can be used in
    training.`),q.forEach(t),r=b(e),h=y(e,"P",{});var U=v(h);C=p(U,`We could for example rotate, blur or flip the images, but there are many
    more options available, that we will cover in our practical PyTorch
    excercises.`),U.forEach(t),O=b(e),z=y(e,"P",{});var X=v(z);H=p(X,`It is not always the case that we would take the 60,000 MNIST training
    samples, apply let's say 140,000 transformations and end up with 200,000
    images for training. Often we apply random transformations to each batch of
    traning that we encounter. For example we could slightly rotate and blur
    each of the 32 images in our batch using some random parameters. That way
    our neural network never encounters the exact same image twice and has to
    learn to generalize.`),X.forEach(t),M=b(e),D=y(e,"P",{});var Y=v(D);L=p(Y,`It is relatively easy to augment image data, but it is not always easy to
    augment text or time series data. To augment text data on Kaggle for
    example, in some competitions people used google translate to translate a
    sentence into a foreign language first and then translate the sentence back
    into english. The sentence changes slightly, but is similar enough to be
    used in the training process. Sometimes you might need to get creative to
    find a good data augmentation approach.`),Y.forEach(t),N=b(e),x=y(e,"P",{});var B=v(x);F=p(B,`Before we move on to the next section let us mention that there is a
    significantly more powerful technique to deal with limited data: `),G(I.$$.fragment,B),K=p(B,`. Tranfer learning allows you to use a model, that was pretrained on
    millions of images or millions of texts, thereby allowing you to finetune
    the model to your needs. Those types of models need significantly less data
    to learn a particular task. It makes little sense to cover transfer learning
    in detail, before we have learned convolutional neural networks or
    transformers. Once we encounter those types of networks we will discuss this
    topic in more detail.`),B.forEach(t),A=b(e),S=y(e,"DIV",{class:!0}),v(S).forEach(t),this.h()},h(){P(c,"class","info"),P(S,"class","separator")},m(e,i){o(e,n,i),g(n,s),o(e,l,i),o(e,u,i),g(u,_),j(m,u,null),g(u,k),o(e,f,i),o(e,c,i),g(c,a),o(e,r,i),o(e,h,i),g(h,C),o(e,O,i),o(e,z,i),g(z,H),o(e,M,i),o(e,D,i),g(D,L),o(e,N,i),o(e,x,i),g(x,F),j(I,x,null),g(x,K),o(e,A,i),o(e,S,i),W=!0},p(e,i){const E={};i&1&&(E.$$scope={dirty:i,ctx:e}),m.$set(E);const q={};i&1&&(q.$$scope={dirty:i,ctx:e}),I.$set(q)},i(e){W||(J(m.$$.fragment,e),J(I.$$.fragment,e),W=!0)},o(e){Q(m.$$.fragment,e),Q(I.$$.fragment,e),W=!1},d(e){e&&t(n),e&&t(l),e&&t(u),R(m),e&&t(f),e&&t(c),e&&t(r),e&&t(h),e&&t(O),e&&t(z),e&&t(M),e&&t(D),e&&t(N),e&&t(x),R(I),e&&t(A),e&&t(S)}}}function le(T){let n,s,l,u,_,m,k,f,c;return f=new ie({props:{$$slots:{default:[re]},$$scope:{ctx:T}}}),{c(){n=w("meta"),s=$(),l=w("h1"),u=d("Data Augmentation"),_=$(),m=w("div"),k=$(),V(f.$$.fragment),this.h()},l(a){const r=ne('[data-svelte="svelte-6o5c0p"]',document.head);n=y(r,"META",{name:!0,content:!0}),r.forEach(t),s=b(a),l=y(a,"H1",{});var h=v(l);u=p(h,"Data Augmentation"),h.forEach(t),_=b(a),m=y(a,"DIV",{class:!0}),v(m).forEach(t),k=b(a),G(f.$$.fragment,a),this.h()},h(){document.title="World4AI | Deep Learning | Data Augmentation",P(n,"name","description"),P(n,"content","We do not always posess sufficient amounts of data to avoid overfitting. Data augmentation is a simple technique to produce synthetic data that can be used to train a neural network."),P(m,"class","separator")},m(a,r){g(document.head,n),o(a,s,r),o(a,l,r),g(l,u),o(a,_,r),o(a,m,r),o(a,k,r),j(f,a,r),c=!0},p(a,[r]){const h={};r&1&&(h.$$scope={dirty:r,ctx:a}),f.$set(h)},i(a){c||(J(f.$$.fragment,a),c=!0)},o(a){Q(f.$$.fragment,a),c=!1},d(a){t(n),a&&t(s),a&&t(l),a&&t(_),a&&t(m),a&&t(k),R(f,a)}}}class he extends ee{constructor(n){super(),te(this,n,null,le,ae,{})}}export{he as default};
