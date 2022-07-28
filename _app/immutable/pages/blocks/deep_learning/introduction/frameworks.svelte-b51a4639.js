import{S as Ut,i as Ht,s as Bt,l as f,a as g,r as u,w as Ot,T as qt,m as c,h as t,c as y,n as l,u as d,x as Rt,p as e,G as a,b as s,y as Yt,f as jt,t as Wt,B as Nt,L as o,M as n,a5 as Xt,q as Qt,E as Zt}from"../../../../chunks/index-caa95cd4.js";import{C as er}from"../../../../chunks/Container-5c6b7f6d.js";function tr(je){let _,M,x,I,z,F,T,E,D,i,m,w,We,P,xe,L,Fe,j,W,N,J,b,Q,Z,ee,Te,$,Pe,te,re,ae,X,k,le,oe,ne,Ne,p,se,Qe,Ve,ie,Ie,he,De,fe,Ke,G,U,Ze,et,H,tt,rt,Le,ce,Je,C,at,v,pe,ue,S,Ge,V,de,me,ve,ge,ye,we,_e,$e,A,B,lt,ot,Ae,nt,st,Xe,be,Ue,O,it,He,q,R,ht,ft,Be,ke,Oe,Y,ct,qe,Ee;return{c(){_=f("p"),M=u(`Deep learning algorithms are usually implemented with specialized deep
    learning frameworks. These frameworks can run on the GPU to improve the
    speed of training and execution, have build in functionality that is
    commonly used in deep learning research and are battle tested. Implementing
    all those details by hand is usually a bad idea, unless you do it for
    learning purposes.`),x=g(),I=f("p"),z=u(`Deep learning frameworks have a rich history. If we wrote this chapter two
    years ago the content would look different. Five years ago the contents
    would be different again. That says a lot about the speed with which the
    development of those frameworks progresses.`),F=g(),T=f("p"),E=u(`All of the below mentioned frameworks use Python as the interface language
    between the programmer and the underlying execution. The frameworks
    themselves on the other hand are written in C++ and CUDA to optimize
    performance. There are fromeworks that use other programming languages
    instead of Python like R, Julia or even JavaScript, but mostly research and
    education is done in Python so you are out of luck if you prefer any other
    programming language. Python is still king in the deep learning community.`),D=g(),i=f("div"),m=g(),w=f("h2"),We=u(`TensorFlow
    `),P=o("svg"),xe=o("g"),L=o("g"),Fe=o("defs"),j=o("path"),W=o("clipPath"),N=o("use"),J=o("g"),b=o("linearGradient"),Q=o("stop"),Z=o("stop"),ee=o("path"),Te=o("g"),$=o("g"),Pe=o("defs"),te=o("path"),re=o("clipPath"),ae=o("use"),X=o("g"),k=o("linearGradient"),le=o("stop"),oe=o("stop"),ne=o("path"),Ne=u(`
    and Keras
    
    `),p=o("svg"),se=o("style"),Qe=u(`.st0 {
          fill: #ffffff;
        }
        .st1 {
          fill: #d00000;
        }
      `),Ve=o("g"),ie=o("path"),Ie=o("g"),he=o("path"),De=o("g"),fe=o("path"),Ke=g(),G=f("p"),U=f("a"),Ze=u("TensorFlow"),et=u(` is
    Google's flagship deep learning framework. The framework was first released
    in the year 2015, which makes it the oldest in the list.
    `),H=f("a"),tt=u("Keras"),rt=u(` is a higher level api that
    sits on top of TensorFlow and provides a lot of convenience for developers by
    reducing a lot of overhead. Keras was actually developed independently by Fran\xE7ois
    Chollet in order to facilitate fast and easy deep learning development. Originally
    Keras supported multiple deep learning libraries (for example Theano), but became
    an integral part of TensorFlow 2 in 2019 and nowadays only supports TensorFlow.
    You can say that TensorFlow and Keras go hand in hand. Keras is an incredible
    great choice for beginners. It has a vibrant community and you will have no difficulty
    to find tutorials and books to learn from.`),Le=g(),ce=f("div"),Je=g(),C=f("h2"),at=u(`PyTorch
    `),v=o("svg"),pe=o("g"),ue=o("path"),S=o("circle"),Ge=o("g"),V=o("g"),de=o("path"),me=o("path"),ve=o("path"),ge=o("path"),ye=o("path"),we=o("path"),_e=o("path"),$e=g(),A=f("p"),B=f("a"),lt=u("PyTorch"),ot=u(` is Meta's
    flagship deep learning framework, which was originally inspired by Torch, a
    Lua based deep learning library. In the years after the initial release in
    2016, PyTorch has become a favorite among deep learning researchers and has
    overtaken TensorFlow. PyTorch is often described as being more intuitive to
    Python developers than TensorFlow, you can say that it is more
    `),Ae=f("em"),nt=u("pythonic"),st=u(" in some regards."),Xe=g(),be=f("div"),Ue=g(),O=f("h2"),it=u("JAX"),He=g(),q=f("p"),R=f("a"),ht=u("JAX"),ft=u(` is the new kid
    on the block. Like Keras and TensorFlow, JAX is developed by Google. The first
    release appeard in the year 2020 on github. JAX has been gaining a lot of traction
    over the last couple of years and there are even rumors that over time TensorFlow
    is going to be replaced by JAX.`),Be=g(),ke=f("div"),Oe=g(),Y=f("p"),ct=u(`It is up to you which framework you choose. Especially when it comes to
    studying deep learning, all of the above frameworks are fine choices and are
    highly recommended. Keras and PyTorch have a much larger community and
    provide many more books and tutorials, but this might change within a couple
    of years. At the moment of writing most of our code examples are implemented
    in PyTorch.`),qe=g(),Ee=f("div"),this.h()},l(r){_=c(r,"P",{});var h=l(_);M=d(h,`Deep learning algorithms are usually implemented with specialized deep
    learning frameworks. These frameworks can run on the GPU to improve the
    speed of training and execution, have build in functionality that is
    commonly used in deep learning research and are battle tested. Implementing
    all those details by hand is usually a bad idea, unless you do it for
    learning purposes.`),h.forEach(t),x=y(r),I=c(r,"P",{});var bt=l(I);z=d(bt,`Deep learning frameworks have a rich history. If we wrote this chapter two
    years ago the content would look different. Five years ago the contents
    would be different again. That says a lot about the speed with which the
    development of those frameworks progresses.`),bt.forEach(t),F=y(r),T=c(r,"P",{});var kt=l(T);E=d(kt,`All of the below mentioned frameworks use Python as the interface language
    between the programmer and the underlying execution. The frameworks
    themselves on the other hand are written in C++ and CUDA to optimize
    performance. There are fromeworks that use other programming languages
    instead of Python like R, Julia or even JavaScript, but mostly research and
    education is done in Python so you are out of luck if you prefer any other
    programming language. Python is still king in the deep learning community.`),kt.forEach(t),D=y(r),i=c(r,"DIV",{class:!0}),l(i).forEach(t),m=y(r),w=c(r,"H2",{class:!0});var Me=l(w);We=d(Me,`TensorFlow
    `),P=n(Me,"svg",{xmlns:!0,"xmlns:xlink":!0,width:!0,viewBox:!0,class:!0});var pt=l(P);xe=n(pt,"g",{});var Et=l(xe);L=n(Et,"g",{});var Re=l(L);Fe=n(Re,"defs",{});var xt=l(Fe);j=n(xt,"path",{id:!0,d:!0}),l(j).forEach(t),xt.forEach(t),W=n(Re,"clipPath",{id:!0});var Ft=l(W);N=n(Ft,"use",{"xlink:href":!0,overflow:!0}),l(N).forEach(t),Ft.forEach(t),J=n(Re,"g",{"clip-path":!0});var ut=l(J);b=n(ut,"linearGradient",{id:!0,gradientUnits:!0,x1:!0,y1:!0,x2:!0,y2:!0,gradientTransform:!0});var dt=l(b);Q=n(dt,"stop",{offset:!0,"stop-color":!0}),l(Q).forEach(t),Z=n(dt,"stop",{offset:!0,"stop-color":!0}),l(Z).forEach(t),dt.forEach(t),ee=n(ut,"path",{d:!0,fill:!0}),l(ee).forEach(t),ut.forEach(t),Re.forEach(t),Et.forEach(t),Te=n(pt,"g",{});var Tt=l(Te);$=n(Tt,"g",{});var Ye=l($);Pe=n(Ye,"defs",{});var Pt=l(Pe);te=n(Pt,"path",{id:!0,d:!0}),l(te).forEach(t),Pt.forEach(t),re=n(Ye,"clipPath",{id:!0});var Vt=l(re);ae=n(Vt,"use",{"xlink:href":!0,overflow:!0}),l(ae).forEach(t),Vt.forEach(t),X=n(Ye,"g",{"clip-path":!0});var mt=l(X);k=n(mt,"linearGradient",{id:!0,gradientUnits:!0,x1:!0,y1:!0,x2:!0,y2:!0,gradientTransform:!0});var vt=l(k);le=n(vt,"stop",{offset:!0,"stop-color":!0}),l(le).forEach(t),oe=n(vt,"stop",{offset:!0,"stop-color":!0}),l(oe).forEach(t),vt.forEach(t),ne=n(mt,"path",{d:!0,fill:!0}),l(ne).forEach(t),mt.forEach(t),Ye.forEach(t),Tt.forEach(t),pt.forEach(t),Ne=d(Me,`
    and Keras
    
    `),p=n(Me,"svg",{version:!0,id:!0,xmlns:!0,"xmlns:xlink":!0,width:!0,height:!0,x:!0,y:!0,viewBox:!0,style:!0,"xml:space":!0,class:!0});var ze=l(p);se=n(ze,"style",{type:!0});var It=l(se);Qe=d(It,`.st0 {
          fill: #ffffff;
        }
        .st1 {
          fill: #d00000;
        }
      `),It.forEach(t),Ve=n(ze,"g",{});var Dt=l(Ve);ie=n(Dt,"path",{class:!0,d:!0}),l(ie).forEach(t),Dt.forEach(t),Ie=n(ze,"g",{});var Gt=l(Ie);he=n(Gt,"path",{class:!0,d:!0}),l(he).forEach(t),Gt.forEach(t),De=n(ze,"g",{});var At=l(De);fe=n(At,"path",{class:!0,d:!0}),l(fe).forEach(t),At.forEach(t),ze.forEach(t),Me.forEach(t),Ke=y(r),G=c(r,"P",{});var Ce=l(G);U=c(Ce,"A",{href:!0,target:!0});var Mt=l(U);Ze=d(Mt,"TensorFlow"),Mt.forEach(t),et=d(Ce,` is
    Google's flagship deep learning framework. The framework was first released
    in the year 2015, which makes it the oldest in the list.
    `),H=c(Ce,"A",{href:!0,target:!0});var zt=l(H);tt=d(zt,"Keras"),zt.forEach(t),rt=d(Ce,` is a higher level api that
    sits on top of TensorFlow and provides a lot of convenience for developers by
    reducing a lot of overhead. Keras was actually developed independently by Fran\xE7ois
    Chollet in order to facilitate fast and easy deep learning development. Originally
    Keras supported multiple deep learning libraries (for example Theano), but became
    an integral part of TensorFlow 2 in 2019 and nowadays only supports TensorFlow.
    You can say that TensorFlow and Keras go hand in hand. Keras is an incredible
    great choice for beginners. It has a vibrant community and you will have no difficulty
    to find tutorials and books to learn from.`),Ce.forEach(t),Le=y(r),ce=c(r,"DIV",{class:!0}),l(ce).forEach(t),Je=y(r),C=c(r,"H2",{class:!0});var gt=l(C);at=d(gt,`PyTorch
    `),v=n(gt,"svg",{version:!0,id:!0,xmlns:!0,"xmlns:xlink":!0,x:!0,y:!0,width:!0,viewBox:!0,"enable-background":!0,"xml:space":!0,class:!0});var yt=l(v);pe=n(yt,"g",{});var wt=l(pe);ue=n(wt,"path",{fill:!0,d:!0}),l(ue).forEach(t),S=n(wt,"circle",{fill:!0,cx:!0,cy:!0,r:!0}),l(S).forEach(t),wt.forEach(t),Ge=n(yt,"g",{});var Ct=l(Ge);V=n(Ct,"g",{});var K=l(V);de=n(K,"path",{fill:!0,d:!0}),l(de).forEach(t),me=n(K,"path",{fill:!0,d:!0}),l(me).forEach(t),ve=n(K,"path",{fill:!0,d:!0}),l(ve).forEach(t),ge=n(K,"path",{fill:!0,d:!0}),l(ge).forEach(t),ye=n(K,"path",{fill:!0,d:!0}),l(ye).forEach(t),we=n(K,"path",{fill:!0,d:!0}),l(we).forEach(t),_e=n(K,"path",{fill:!0,d:!0}),l(_e).forEach(t),K.forEach(t),Ct.forEach(t),yt.forEach(t),gt.forEach(t),$e=y(r),A=c(r,"P",{});var Se=l(A);B=c(Se,"A",{href:!0,target:!0});var St=l(B);lt=d(St,"PyTorch"),St.forEach(t),ot=d(Se,` is Meta's
    flagship deep learning framework, which was originally inspired by Torch, a
    Lua based deep learning library. In the years after the initial release in
    2016, PyTorch has become a favorite among deep learning researchers and has
    overtaken TensorFlow. PyTorch is often described as being more intuitive to
    Python developers than TensorFlow, you can say that it is more
    `),Ae=c(Se,"EM",{});var Kt=l(Ae);nt=d(Kt,"pythonic"),Kt.forEach(t),st=d(Se," in some regards."),Se.forEach(t),Xe=y(r),be=c(r,"DIV",{class:!0}),l(be).forEach(t),Ue=y(r),O=c(r,"H2",{class:!0});var Lt=l(O);it=d(Lt,"JAX"),Lt.forEach(t),He=y(r),q=c(r,"P",{});var _t=l(q);R=c(_t,"A",{href:!0,target:!0});var Jt=l(R);ht=d(Jt,"JAX"),Jt.forEach(t),ft=d(_t,` is the new kid
    on the block. Like Keras and TensorFlow, JAX is developed by Google. The first
    release appeard in the year 2020 on github. JAX has been gaining a lot of traction
    over the last couple of years and there are even rumors that over time TensorFlow
    is going to be replaced by JAX.`),_t.forEach(t),Be=y(r),ke=c(r,"DIV",{class:!0}),l(ke).forEach(t),Oe=y(r),Y=c(r,"P",{class:!0});var $t=l(Y);ct=d($t,`It is up to you which framework you choose. Especially when it comes to
    studying deep learning, all of the above frameworks are fine choices and are
    highly recommended. Keras and PyTorch have a much larger community and
    provide many more books and tutorials, but this might change within a couple
    of years. At the moment of writing most of our code examples are implemented
    in PyTorch.`),$t.forEach(t),qe=y(r),Ee=c(r,"DIV",{class:!0}),l(Ee).forEach(t),this.h()},h(){e(i,"class","separator"),e(j,"id","SVGID_1_"),e(j,"d","M47.5 17.6L25 4.8v52.6l9-5.2V37.4l6.8 3.9-.1-10.1-6.7-3.9v-5.9l13.5 7.9z"),Xt(N,"xlink:href","#SVGID_1_"),e(N,"overflow","visible"),e(W,"id","SVGID_2_"),e(Q,"offset","0"),e(Q,"stop-color","#ff6f00"),e(Z,"offset","1"),e(Z,"stop-color","#ffa800"),e(b,"id","SVGID_3_"),e(b,"gradientUnits","userSpaceOnUse"),e(b,"x1","-1.6"),e(b,"y1","335.05"),e(b,"x2","53.6"),e(b,"y2","335.05"),e(b,"gradientTransform","translate(0 -304)"),e(ee,"d","M-1.6 4.6h55.2v52.9H-1.6V4.6z"),e(ee,"fill","url(#SVGID_3_)"),e(J,"clip-path","url(#SVGID_2_)"),e(te,"id","SVGID_4_"),e(te,"d","M.5 17.6L23 4.8v52.6l-9-5.2V21.4L.5 29.3z"),Xt(ae,"xlink:href","#SVGID_4_"),e(ae,"overflow","visible"),e(re,"id","SVGID_5_"),e(le,"offset","0"),e(le,"stop-color","#ff6f00"),e(oe,"offset","1"),e(oe,"stop-color","#ffa800"),e(k,"id","SVGID_6_"),e(k,"gradientUnits","userSpaceOnUse"),e(k,"x1","-1.9"),e(k,"y1","335.05"),e(k,"x2","53.3"),e(k,"y2","335.05"),e(k,"gradientTransform","translate(0 -304)"),e(ne,"d","M-1.9 4.6h55.2v52.9H-1.9V4.6z"),e(ne,"fill","url(#SVGID_6_)"),e(X,"clip-path","url(#SVGID_5_)"),e(P,"xmlns","http://www.w3.org/2000/svg"),e(P,"xmlns:xlink","http://www.w3.org/1999/xlink"),e(P,"width","32px"),e(P,"viewBox","0 0 54 64"),e(P,"class","svelte-l1art"),e(se,"type","text/css"),e(ie,"class","st0"),e(ie,"d",`M1080,1079.96c0,0.02-0.02,0.04-0.04,0.04H0.04c-0.02,0-0.04-0.02-0.04-0.04V0.04C0,0.02,0.02,0,0.04,0
		h1079.93c0.02,0,0.04,0.02,0.04,0.04V1079.96z`),e(he,"class","st1"),e(he,"d",`M1062,1061.96c0,0.02-0.02,0.04-0.04,0.04H18.04c-0.02,0-0.04-0.02-0.04-0.04V18.04
		c0-0.02,0.02-0.04,0.04-0.04h1043.93c0.02,0,0.04,0.02,0.04,0.04V1061.96z`),e(fe,"class","st0"),e(fe,"d",`M303,823.67c0,0.79,0.46,1.89,1.01,2.44l17.87,17.87c0.56,0.56,1.66,1.01,2.44,1.01h61.15
		c0.79,0,1.89-0.46,2.44-1.01l17.87-17.87c0.56-0.56,1.01-1.66,1.01-2.44V629.64c0-0.79,0.47-1.88,1.04-2.42l77.69-74.2
		c0.57-0.54,1.4-0.46,1.84,0.2l196.29,290.6c0.44,0.65,1.45,1.19,2.23,1.19h86.63c0.79,0,1.73-0.57,2.09-1.27l15.72-30.46
		c0.36-0.7,0.29-1.8-0.16-2.45L560.56,478.03c-0.45-0.65-0.36-1.63,0.2-2.19l211.18-210.19c0.56-0.56,1.01-1.65,1.01-2.44v-3.88
		c0-0.79-0.26-2.02-0.57-2.75l-12.18-28.01c-0.31-0.72-1.22-1.31-2-1.31h-85.63c-0.79,0-1.89,0.46-2.44,1.01l-262.31,263.3
		c-0.56,0.56-1.01,0.37-1.01-0.42V249.6c0-0.79-0.44-1.9-0.98-2.48l-17.53-18.8c-0.54-0.58-1.62-1.05-2.41-1.05h-61.57
		c-0.79,0-1.87,0.47-2.41,1.05l-17.95,19.38c-0.54,0.58-0.97,1.69-0.97,2.48V823.67z`),e(p,"version","1.1"),e(p,"id","Layer_1"),e(p,"xmlns","http://www.w3.org/2000/svg"),e(p,"xmlns:xlink","http://www.w3.org/1999/xlink"),e(p,"width","32px"),e(p,"height","32px"),e(p,"x","0px"),e(p,"y","0px"),e(p,"viewBox","0 0 1080 1080"),Qt(p,"enable-background","new 0 0 1080 1080"),e(p,"xml:space","preserve"),e(p,"class","svelte-l1art"),e(w,"class","svelte-l1art"),e(U,"href","https://www.tensorflow.org/"),e(U,"target","_blank"),e(H,"href","https://keras.io/"),e(H,"target","_blank"),e(ce,"class","separator"),e(ue,"fill","#EE4C2C"),e(ue,"d",`M63.1,567.3l-6.6,6.6c10.8,10.8,10.8,28.2,0,38.8c-10.8,10.8-28.2,10.8-38.8,0c-10.8-10.8-10.8-28.2,0-38.8
		l0,0l17.1-17.1l2.4-2.4l0,0v-12.9l-25.8,25.8c-14.4,14.4-14.4,37.6,0,52s37.6,14.4,51.7,0C77.5,604.8,77.5,581.7,63.1,567.3z`),e(S,"fill","#EE4C2C"),e(S,"cx","50.2"),e(S,"cy","560.9"),e(S,"r","4.8"),e(de,"fill","#FFFFFF"),e(de,"d",`M129.8,600.3h-11.1v28.5h-8.4v-81.1c0,0,19.2,0,20.4,0c21.3,0,31.5,10.5,31.5,25.2
			C162.5,591,149.9,600.3,129.8,600.3z M130.7,555.8c-0.9,0-11.7,0-11.7,0v37.3l11.4-0.3c15.3-0.3,23.7-6.3,23.7-18.9
			C154.1,562.1,145.7,555.8,130.7,555.8z`),e(me,"fill","#FFFFFF"),e(me,"d",`M199.8,628.5l-4.8,12.9c-5.4,14.4-11.1,18.6-19.2,18.6c-4.5,0-7.8-1.2-11.4-2.7l2.4-7.5
			c2.7,1.5,5.7,2.7,9,2.7c4.5,0,7.8-2.4,12.3-13.8l3.9-10.5l-23.1-58.6h8.7l18.6,49l18.3-49h8.4L199.8,628.5z`),e(ve,"fill","#FFFFFF"),e(ve,"d","M250.3,555.8v73.3h-8.4v-73.3h-28.5V548h65.2v7.8C278.5,555.8,250.3,555.8,250.3,555.8z"),e(ge,"fill","#FFFFFF"),e(ge,"d",`M302.3,630.6c-16.5,0-28.5-12.3-28.5-31.2c0-18.9,12.6-31.5,29.4-31.5s28.5,12.3,28.5,31.2
			C331.4,618,318.8,630.6,302.3,630.6z M302.6,575.4c-12.6,0-20.7,9.9-20.7,24c0,14.4,8.4,24.3,21,24.3s20.7-9.9,20.7-24
			C323.6,585,315.2,575.4,302.6,575.4z`),e(ye,"fill","#FFFFFF"),e(ye,"d",`M351.8,629.1h-8.1v-59.5l8.1-1.8v12.6c3.9-7.5,9.6-12.6,17.4-12.6c3.9,0,7.5,1.2,10.5,2.7l-2.1,7.5
			c-2.7-1.5-5.7-2.7-9-2.7c-6.3,0-12,4.8-16.8,15.3V629.1L351.8,629.1z`),e(we,"fill","#FFFFFF"),e(we,"d",`M411.3,630.6c-18,0-29.1-12.9-29.1-31.2c0-18.6,12.3-31.5,29.1-31.5c7.2,0,13.5,1.8,18.6,5.1l-2.1,7.2
			c-4.5-3-10.2-4.8-16.5-4.8c-12.9,0-20.7,9.6-20.7,23.7c0,14.4,8.4,24,21,24c6,0,12-1.8,16.5-4.8l1.8,7.5
			C424.5,628.8,418.2,630.6,411.3,630.6z`),e(_e,"fill","#FFFFFF"),e(_e,"d",`M479.5,629.1v-38.5c0-10.5-4.2-15-12.6-15c-6.9,0-13.5,3.6-18.3,8.4v45.1h-8.1v-87.4l8.1-1.8
			c0,0,0,37.3,0,37.6c6.3-6.3,14.1-9.3,20.7-9.3c11.4,0,18.6,7.5,18.6,20.4v40.6H479.5z`),e(v,"version","1.1"),e(v,"id","Layer_1"),e(v,"xmlns","http://www.w3.org/2000/svg"),e(v,"xmlns:xlink","http://www.w3.org/1999/xlink"),e(v,"x","0px"),e(v,"y","0px"),e(v,"width","160px"),e(v,"viewBox","0.6 539.9 487.3 120.2"),e(v,"enable-background","new 0.6 539.9 487.3 120.2"),e(v,"xml:space","preserve"),e(v,"class","svelte-l1art"),e(C,"class","svelte-l1art"),e(B,"href","https://pytorch.org/"),e(B,"target","_blank"),e(be,"class","separator"),e(O,"class","svelte-l1art"),e(R,"href","https://github.com/google/jax"),e(R,"target","_blank"),e(ke,"class","separator"),e(Y,"class","warning"),e(Ee,"class","separator")},m(r,h){s(r,_,h),a(_,M),s(r,x,h),s(r,I,h),a(I,z),s(r,F,h),s(r,T,h),a(T,E),s(r,D,h),s(r,i,h),s(r,m,h),s(r,w,h),a(w,We),a(w,P),a(P,xe),a(xe,L),a(L,Fe),a(Fe,j),a(L,W),a(W,N),a(L,J),a(J,b),a(b,Q),a(b,Z),a(J,ee),a(P,Te),a(Te,$),a($,Pe),a(Pe,te),a($,re),a(re,ae),a($,X),a(X,k),a(k,le),a(k,oe),a(X,ne),a(w,Ne),a(w,p),a(p,se),a(se,Qe),a(p,Ve),a(Ve,ie),a(p,Ie),a(Ie,he),a(p,De),a(De,fe),s(r,Ke,h),s(r,G,h),a(G,U),a(U,Ze),a(G,et),a(G,H),a(H,tt),a(G,rt),s(r,Le,h),s(r,ce,h),s(r,Je,h),s(r,C,h),a(C,at),a(C,v),a(v,pe),a(pe,ue),a(pe,S),a(v,Ge),a(Ge,V),a(V,de),a(V,me),a(V,ve),a(V,ge),a(V,ye),a(V,we),a(V,_e),s(r,$e,h),s(r,A,h),a(A,B),a(B,lt),a(A,ot),a(A,Ae),a(Ae,nt),a(A,st),s(r,Xe,h),s(r,be,h),s(r,Ue,h),s(r,O,h),a(O,it),s(r,He,h),s(r,q,h),a(q,R),a(R,ht),a(q,ft),s(r,Be,h),s(r,ke,h),s(r,Oe,h),s(r,Y,h),a(Y,ct),s(r,qe,h),s(r,Ee,h)},p:Zt,d(r){r&&t(_),r&&t(x),r&&t(I),r&&t(F),r&&t(T),r&&t(D),r&&t(i),r&&t(m),r&&t(w),r&&t(Ke),r&&t(G),r&&t(Le),r&&t(ce),r&&t(Je),r&&t(C),r&&t($e),r&&t(A),r&&t(Xe),r&&t(be),r&&t(Ue),r&&t(O),r&&t(He),r&&t(q),r&&t(Be),r&&t(ke),r&&t(Oe),r&&t(Y),r&&t(qe),r&&t(Ee)}}}function rr(je){let _,M,x,I,z,F,T,E,D;return E=new er({props:{$$slots:{default:[tr]},$$scope:{ctx:je}}}),{c(){_=f("meta"),M=g(),x=f("h1"),I=u("Deep Learning Frameworks"),z=g(),F=f("div"),T=g(),Ot(E.$$.fragment),this.h()},l(i){const m=qt('[data-svelte="svelte-74lh5e"]',document.head);_=c(m,"META",{name:!0,content:!0}),m.forEach(t),M=y(i),x=c(i,"H1",{});var w=l(x);I=d(w,"Deep Learning Frameworks"),w.forEach(t),z=y(i),F=c(i,"DIV",{class:!0}),l(F).forEach(t),T=y(i),Rt(E.$$.fragment,i),this.h()},h(){document.title="World4AI | Deep Learning | Frameworks",e(_,"name","description"),e(_,"content","Deep learning frameworks are necessary for an efficient workflow. At the moment TensorFlow, PyTorch and JAX dominate the market."),e(F,"class","separator")},m(i,m){a(document.head,_),s(i,M,m),s(i,x,m),a(x,I),s(i,z,m),s(i,F,m),s(i,T,m),Yt(E,i,m),D=!0},p(i,[m]){const w={};m&1&&(w.$$scope={dirty:m,ctx:i}),E.$set(w)},i(i){D||(jt(E.$$.fragment,i),D=!0)},o(i){Wt(E.$$.fragment,i),D=!1},d(i){t(_),i&&t(M),i&&t(x),i&&t(z),i&&t(F),i&&t(T),Nt(E,i)}}}class or extends Ut{constructor(_){super(),Ht(this,_,null,rr,Bt,{})}}export{or as default};
