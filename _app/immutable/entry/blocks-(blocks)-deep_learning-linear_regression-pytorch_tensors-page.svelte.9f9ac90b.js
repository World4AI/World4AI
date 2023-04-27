import{S as Yn,i as Zn,s as es,k as l,a as d,q as r,y as v,W as ts,l as i,h as t,c as f,m as c,r as n,z as $,n as z,N as o,b as a,A as w,g as y,d as _,B as E,C as ln}from"../chunks/index.4d92b023.js";import{C as os}from"../chunks/Container.b0705c7b.js";import{P as C}from"../chunks/PythonCode.212ba7a6.js";import{H as rs}from"../chunks/Highlight.b7c1de53.js";import{L as an}from"../chunks/Latex.e0b308c0.js";function ns(S){let p;return{c(){p=r("tensor")},l(h){p=n(h,"tensor")},m(h,m){a(h,p,m)},d(h){h&&t(p)}}}function ss(S){let p=String.raw`\mathbf{A}`+"",h;return{c(){h=r(p)},l(m){h=n(m,p)},m(m,O){a(m,h,O)},p:ln,d(m){m&&t(h)}}}function as(S){let p=String.raw`\mathbf{B}`+"",h;return{c(){h=r(p)},l(m){h=n(m,p)},m(m,O){a(m,h,O)},p:ln,d(m){m&&t(h)}}}function ls(S){let p=String.raw`\mathbf{A \cdot B}`+"",h;return{c(){h=r(p)},l(m){h=n(m,p)},m(m,O){a(m,h,O)},p:ln,d(m){m&&t(h)}}}function is(S){let p,h,m,O,D,B,P,A,F,u,b,I,fo,mo,yt,ne,_t,se,po,Et,J,uo,Se,ho,vo,bt,K,$o,Be,wo,yo,Tt,ae,gt,T,_o,Ie,Eo,bo,ze,To,go,Fe,Po,xo,Ne,Co,Oo,Re,Do,Ao,Pt,le,xt,ie,ko,Ct,k,qo,Le,jo,Wo,He,So,Bo,Me,Io,zo,Ot,ce,Dt,Q,Fo,Ge,No,Ro,At,q,Lo,Ue,Ho,Mo,Ve,Go,Uo,Je,Vo,Jo,kt,N,Ko,Ke,Qo,Xo,Qe,Yo,Zo,qt,de,jt,fe,er,Wt,R,tr,Xe,or,rr,Ye,nr,sr,St,me,Bt,pe,ar,It,X,lr,Ze,ir,cr,zt,L,dr,Y,fr,Z,mr,Ft,ue,Nt,x,pr,et,ur,hr,tt,vr,$r,ot,wr,yr,rt,_r,Er,Rt,he,Lt,ve,br,Ht,$e,Tr,xe,gr,H,Pr,nt,xr,Cr,st,Or,Dr,at,Ar,Mt,we,Gt,g,kr,lt,qr,jr,it,Wr,Sr,ct,Br,Ir,dt,zr,Fr,ft,Nr,Rr,Ut,ye,Vt,_e,Lr,Jt,M,Hr,ee,Mr,mt,Gr,Ur,Kt,Ee,Qt,be,Vr,Xt,G,Jr,pt,Kr,Qr,ut,Xr,Yr,Yt,Te,Zt,te,Zr,ht,en,tn,eo,ge,to,Ce,on,oo,Oe,rn,ro,De,no;return m=new rs({props:{$$slots:{default:[ns]},$$scope:{ctx:S}}}),D=new C({props:{code:"import torch"}}),ne=new C({props:{code:String.raw`tensor = torch.tensor([[0, 1, 2], [3, 4, 5]]) 
print(tensor)`}}),ae=new C({props:{code:String.raw`tensor = torch.tensor(data=[[0, 1, 2], [3, 4, 5]])`}}),le=new C({props:{code:String.raw`tensor = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.float32) 
print(tensor.dtype)`}}),ce=new C({props:{code:String.raw`# cuda:0 represents the first nvidia device
# theoretically you could have several graphics cards
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tensor = torch.tensor([[0, 1, 2], [3, 4, 5]], device=device)`}}),de=new C({props:{code:String.raw`tensor = torch.tensor([[0, 1, 2], [3, 4, 5]])
print(f'Original Tensor: dtype={tensor.dtype}, device={tensor.device}, requires_grad={tensor.requires_grad}')
tensor = tensor.to(torch.float32)
print(f'Adjusted dtype: dtype={tensor.dtype}, device={tensor.device}, requires_grad={tensor.requires_grad}')
tensor = tensor.to(device)
print(f'Adjusted device: dtype={tensor.dtype}, device={tensor.device}, requires_grad={tensor.requires_grad}')
tensor.requires_grad = True
print(f'Adjusted requres_grad: dtype={tensor.dtype}, device={tensor.device}, requires_grad={tensor.requires_grad}')
`}}),me=new C({props:{code:String.raw`print(tensor.size())
print(tensor.shape)
`}}),Y=new an({props:{$$slots:{default:[ss]},$$scope:{ctx:S}}}),Z=new an({props:{$$slots:{default:[as]},$$scope:{ctx:S}}}),ue=new C({props:{code:String.raw`A = torch.ones(size=(2, 2), dtype=torch.float32)
B = torch.tensor([[1, 2],[3, 4]], dtype=torch.float32)
`}}),he=new C({props:{code:String.raw`print(A + B)
print(A - B)
print(A * B)
print(A / B)
`}}),we=new C({props:{code:String.raw`print(A.add(B))
print(A.subtract(B))
print(A.multiply(B))
print(A.divide(B))
`}}),ye=new C({props:{code:String.raw`test = torch.tensor([[1, 2], [4, 4]], dtype=torch.float32)
test.add_(A)
# the test tensor was changed
print(test)
`}}),ee=new an({props:{$$slots:{default:[ls]},$$scope:{ctx:S}}}),Ee=new C({props:{code:String.raw`# Equivalent to torch.matmul(A, B)
A.matmul(B)
`}}),Te=new C({props:{code:String.raw`# Equivalent to torch.matmul(A, B)
A @ B
`}}),ge=new C({props:{code:String.raw`t = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t.sum())
print(t.sum(dim=0))
print(t.sum(dim=1))
`}}),{c(){p=l("p"),h=r(`Literally all modern deep learning libraries are based on a fundamental
    mathematical object called `),v(m.$$.fragment),O=r(`. We will use this
    object throughout all remaining chapters of this block, no matter if we
    implement something as trivial as linear regresssion or a state of the art
    deep learning architecture.
    `),v(D.$$.fragment),B=d(),P=l("p"),A=r("According to the PyTorch documentation, "),F=l("code"),u=r("torch.Tensor"),b=r(` is a
    multi-dimensional matrix containing elements of a single data type. The
    method `),I=l("code"),fo=r("torch.tensor()"),mo=r(` is the most straightforward way to create
    a tensor. Below for example we create a tensor object with 2 rows and 3 columns.`),yt=d(),v(ne.$$.fragment),_t=d(),se=l("pre"),po=r(`tensor([[0, 1, 2],
        [3, 4, 5]])
  `),Et=d(),J=l("p"),uo=r(`The method has some arguments, that allow us to control the properties of
    the tensor: `),Se=l("code"),ho=r("torch.tensor(data, dtype=None, device=None, requires_grad=False)"),vo=r("."),bt=d(),K=l("p"),$o=r("The "),Be=l("code"),wo=r("data"),yo=r(` argument is the only required parameter. With this argument
    we provide an arraylike structure, like a list, a tuple or a NumPy ndarray, to
    construct a tensor.`),Tt=d(),v(ae.$$.fragment),gt=d(),T=l("p"),_o=r("The "),Ie=l("code"),Eo=r("dtype"),bo=r(` argument determines the type of the tensor. This
    essentially means, that we have to think about in advance what type of data
    a tensor is supposed to contain. If we do not specify the type explicitly,
    `),ze=l("code"),To=r("dtype"),go=r(`
    is going to be `),Fe=l("code"),Po=r("torch.int64"),xo=r(`, if all of inputs are integers and
    it is going to be `),Ne=l("code"),Co=r("torch.float32"),Oo=r(` if even one of the inputs is a
    float. Most neural network weights and biases are going to be
    `),Re=l("code"),Do=r("torch.float32"),Ao=r(`, so for the time being those two datatypes are
    actually sufficient to get us started. When the need arises, we will cover
    more datatypes.`),Pt=d(),v(le.$$.fragment),xt=d(),ie=l("pre"),ko=r(`torch.float32
  `),Ct=d(),k=l("p"),qo=r("Tensors can live on different devices, like the cpu, the gpu or tpu and the "),Le=l("code"),jo=r("device"),Wo=r(`
    argument allows us to create a tensor on a particular device. If we do not specify
    a device, we will use the cpu as the default. For the most part we will be interested
    in moving a tensor to the gpu to get better parallelisation. For that we need
    to have an Nvidia graphics card. We can test if we have a valid graphics card,
    by running
    `),He=l("code"),So=r("torch.cuda.is_available()"),Bo=r(`. If the method returns
    `),Me=l("code"),Io=r("True"),zo=r(", we are good to go."),Ot=d(),v(ce.$$.fragment),Dt=d(),Q=l("p"),Fo=r("The last argument, "),Ge=l("code"),No=r("requires_grad"),Ro=r(` determines whether the tensor needs
    to be included in gradient descent calculations. This will be covered in more
    detail in future tutorials.`),At=d(),q=l("p"),Lo=r("There are many more methods to create a Tensor. The method "),Ue=l("code"),Ho=r("torch.from_numpy()"),Mo=r(`
    turns a numpy ndarray into a PyTorch tensor, `),Ve=l("code"),Go=r("torch.zeros()"),Uo=r(`
    returns a Tensor with all zeros and `),Je=l("code"),Vo=r("torch.ones()"),Jo=r(` returns a Tensor
    with all ones. We will see more of those methods as we go along. It makes no
    sense to cover all of them without any context.`),kt=d(),N=l("p"),Ko=r(`If we need to change the parameters of an already initialized Tensor, we can
    do the adjustments in a later step, primarily using the `),Ke=l("code"),Qo=r("to"),Xo=r(`
    method of the Tensor class. The `),Qe=l("code"),Yo=r("to"),Zo=r(` method does not overwrite the
    original Tensor, but returns an adjusted one.`),qt=d(),v(de.$$.fragment),jt=d(),fe=l("pre"),er=r(`Original Tensor: dtype=torch.int64, device=cpu, requires_grad=False
Adjusted dtype: dtype=torch.float32, device=cpu, requires_grad=False
Adjusted device: dtype=torch.float32, device=cuda:0, requires_grad=False
Adjusted requres_grad: dtype=torch.float32, device=cuda:0, requires_grad=True
  `),Wt=d(),R=l("p"),tr=r(`In practice we are often interested in the shape of a particular tensor. We
    can use use my_`),Xe=l("code"),or=r("tensor.size()"),rr=r(" or "),Ye=l("code"),nr=r("my_tensor.shape"),sr=r(` to
    find out the dimensions of the tensor.`),St=d(),v(me.$$.fragment),Bt=d(),pe=l("pre"),ar=r(`torch.Size([2, 3])
torch.Size([2, 3])
  `),It=d(),X=l("p"),lr=r(`PyTorch, like other frameworks that work with tensors, is extremely
    efficient when it comes to matrix operations. These operations are done in
    parallel and can be transfered to the GPU if you have a cuda compatibale
    graphics card. Essentially all of deep learning is based on matrix
    operations, so let"s spend some time to learn how we can invoke matrix
    operations using `),Ze=l("code"),ir=r("Tensor"),cr=r(" objects."),zt=d(),L=l("p"),dr=r("We will use two tensors, "),v(Y.$$.fragment),fr=r(" and "),v(Z.$$.fragment),mr=r(" to demonstrate basic mathematical operations."),Ft=d(),v(ue.$$.fragment),Nt=d(),x=l("p"),pr=r(`We can add, subtract, multiply and divide those matrices using basic
    mathematic operators like `),et=l("code"),ur=r("+"),hr=r(", "),tt=l("code"),vr=r("-"),$r=r(", "),ot=l("code"),wr=r("*"),yr=r(`,
    `),rt=l("code"),_r=r("/"),Er=r(`. All those operations work elementwise, so when you multiply
    two matrices you won't actually use matrix multiplication that involves dot
    products but elementwise multiplication.`),Rt=d(),v(he.$$.fragment),Lt=d(),ve=l("pre"),br=r(`  tensor([[2., 3.],
          [4., 5.]])
  tensor([[ 0., -1.],
          [-2., -3.]])
  tensor([[1., 2.],
          [3., 4.]])
  tensor([[1.0000, 0.5000],
          [0.3333, 0.2500]])
  `),Ht=d(),$e=l("p"),Tr=r("We can achieve the same results using the explicit methods: "),xe=l("code"),gr=r("Tensor.add()"),H=l("code"),Pr=r(", "),nt=l("code"),xr=r("Tensor.subtract()"),Cr=r(", "),st=l("code"),Or=r("Tensor.multiply()"),Dr=r(`,
        `),at=l("code"),Ar=r("Tensor.divide()."),Mt=d(),v(we.$$.fragment),Gt=d(),g=l("p"),kr=r(`While the above methods do not change the original tensors, each of the
    methods has a corresponding method that changes the tensor in place. These
    methods always end with a `),lt=l("code"),qr=r("_"),jr=r(": "),it=l("code"),Wr=r("add_()"),Sr=r(`,
    `),ct=l("code"),Br=r("subtract_()"),Ir=r(", "),dt=l("code"),zr=r("multiply_()"),Fr=r(", "),ft=l("code"),Nr=r("divide_()"),Rr=r("."),Ut=d(),v(ye.$$.fragment),Vt=d(),_e=l("pre"),Lr=r(`tensor([[2., 3.],
        [5., 5.]])
  `),Jt=d(),M=l("p"),Hr=r(`Probaly one of the most important matrix operations in all of deep learning
    is product of two matrices, `),v(ee.$$.fragment),Mr=r(`.
    For that purpose we can use the `),mt=l("code"),Gr=r("matmul"),Ur=r(" method."),Kt=d(),v(Ee.$$.fragment),Qt=d(),be=l("pre"),Vr=r(`tensor([[4., 6.],
        [4., 6.]])
  `),Xt=d(),G=l("p"),Jr=r("Alternatively we can use "),pt=l("code"),Kr=r("@"),Qr=r(` as a convenient way to use matrix
    multiplication. This is essentially just a shorthand notation for
    `),ut=l("code"),Xr=r("torch.matmul"),Yr=r("."),Yt=d(),v(Te.$$.fragment),Zt=d(),te=l("p"),Zr=r(`A final concept that we would like to mention is the concept of dimensions
    in PyTorch. Often we would like to calculate some summary statistics (like a
    sum or a mean) for a Tensor object. But we would like those to be calculated
    for a particular dimension. We can explicitly set the dimension by defining
    the `),ht=l("code"),en=r("dim"),tn=r(" parameter."),eo=d(),v(ge.$$.fragment),to=d(),Ce=l("pre"),on=r(`tensor(21)
tensor([5, 7, 9])
tensor([ 6, 15])
  `),oo=d(),Oe=l("p"),rn=r(`The very first sum that we calculate in the example below, does not take any
    dimensions into consideration and just calculates the sum over the whole
    tensor. In the second example we calculate the sum over the 0th, the row,
    dimension. That means that for each of the available columns we calculate
    the sum by moving down the rows. When we calculate the sum for the 1st, the
    column dimension, we go over each row and calculate the sum by moving
    through the columns.`),ro=d(),De=l("div"),this.h()},l(e){p=i(e,"P",{});var s=c(p);h=n(s,`Literally all modern deep learning libraries are based on a fundamental
    mathematical object called `),$(m.$$.fragment,s),O=n(s,`. We will use this
    object throughout all remaining chapters of this block, no matter if we
    implement something as trivial as linear regresssion or a state of the art
    deep learning architecture.
    `),$(D.$$.fragment,s),s.forEach(t),B=f(e),P=i(e,"P",{});var V=c(P);A=n(V,"According to the PyTorch documentation, "),F=i(V,"CODE",{});var vt=c(F);u=n(vt,"torch.Tensor"),vt.forEach(t),b=n(V,` is a
    multi-dimensional matrix containing elements of a single data type. The
    method `),I=i(V,"CODE",{});var $t=c(I);fo=n($t,"torch.tensor()"),$t.forEach(t),mo=n(V,` is the most straightforward way to create
    a tensor. Below for example we create a tensor object with 2 rows and 3 columns.`),V.forEach(t),yt=f(e),$(ne.$$.fragment,e),_t=f(e),se=i(e,"PRE",{class:!0});var wt=c(se);po=n(wt,`tensor([[0, 1, 2],
        [3, 4, 5]])
  `),wt.forEach(t),Et=f(e),J=i(e,"P",{});var so=c(J);uo=n(so,`The method has some arguments, that allow us to control the properties of
    the tensor: `),Se=i(so,"CODE",{});var cn=c(Se);ho=n(cn,"torch.tensor(data, dtype=None, device=None, requires_grad=False)"),cn.forEach(t),vo=n(so,"."),so.forEach(t),bt=f(e),K=i(e,"P",{});var ao=c(K);$o=n(ao,"The "),Be=i(ao,"CODE",{});var dn=c(Be);wo=n(dn,"data"),dn.forEach(t),yo=n(ao,` argument is the only required parameter. With this argument
    we provide an arraylike structure, like a list, a tuple or a NumPy ndarray, to
    construct a tensor.`),ao.forEach(t),Tt=f(e),$(ae.$$.fragment,e),gt=f(e),T=i(e,"P",{});var j=c(T);_o=n(j,"The "),Ie=i(j,"CODE",{});var fn=c(Ie);Eo=n(fn,"dtype"),fn.forEach(t),bo=n(j,` argument determines the type of the tensor. This
    essentially means, that we have to think about in advance what type of data
    a tensor is supposed to contain. If we do not specify the type explicitly,
    `),ze=i(j,"CODE",{});var mn=c(ze);To=n(mn,"dtype"),mn.forEach(t),go=n(j,`
    is going to be `),Fe=i(j,"CODE",{});var pn=c(Fe);Po=n(pn,"torch.int64"),pn.forEach(t),xo=n(j,`, if all of inputs are integers and
    it is going to be `),Ne=i(j,"CODE",{});var un=c(Ne);Co=n(un,"torch.float32"),un.forEach(t),Oo=n(j,` if even one of the inputs is a
    float. Most neural network weights and biases are going to be
    `),Re=i(j,"CODE",{});var hn=c(Re);Do=n(hn,"torch.float32"),hn.forEach(t),Ao=n(j,`, so for the time being those two datatypes are
    actually sufficient to get us started. When the need arises, we will cover
    more datatypes.`),j.forEach(t),Pt=f(e),$(le.$$.fragment,e),xt=f(e),ie=i(e,"PRE",{class:!0});var vn=c(ie);ko=n(vn,`torch.float32
  `),vn.forEach(t),Ct=f(e),k=i(e,"P",{});var oe=c(k);qo=n(oe,"Tensors can live on different devices, like the cpu, the gpu or tpu and the "),Le=i(oe,"CODE",{});var $n=c(Le);jo=n($n,"device"),$n.forEach(t),Wo=n(oe,`
    argument allows us to create a tensor on a particular device. If we do not specify
    a device, we will use the cpu as the default. For the most part we will be interested
    in moving a tensor to the gpu to get better parallelisation. For that we need
    to have an Nvidia graphics card. We can test if we have a valid graphics card,
    by running
    `),He=i(oe,"CODE",{});var wn=c(He);So=n(wn,"torch.cuda.is_available()"),wn.forEach(t),Bo=n(oe,`. If the method returns
    `),Me=i(oe,"CODE",{});var yn=c(Me);Io=n(yn,"True"),yn.forEach(t),zo=n(oe,", we are good to go."),oe.forEach(t),Ot=f(e),$(ce.$$.fragment,e),Dt=f(e),Q=i(e,"P",{});var lo=c(Q);Fo=n(lo,"The last argument, "),Ge=i(lo,"CODE",{});var _n=c(Ge);No=n(_n,"requires_grad"),_n.forEach(t),Ro=n(lo,` determines whether the tensor needs
    to be included in gradient descent calculations. This will be covered in more
    detail in future tutorials.`),lo.forEach(t),At=f(e),q=i(e,"P",{});var re=c(q);Lo=n(re,"There are many more methods to create a Tensor. The method "),Ue=i(re,"CODE",{});var En=c(Ue);Ho=n(En,"torch.from_numpy()"),En.forEach(t),Mo=n(re,`
    turns a numpy ndarray into a PyTorch tensor, `),Ve=i(re,"CODE",{});var bn=c(Ve);Go=n(bn,"torch.zeros()"),bn.forEach(t),Uo=n(re,`
    returns a Tensor with all zeros and `),Je=i(re,"CODE",{});var Tn=c(Je);Vo=n(Tn,"torch.ones()"),Tn.forEach(t),Jo=n(re,` returns a Tensor
    with all ones. We will see more of those methods as we go along. It makes no
    sense to cover all of them without any context.`),re.forEach(t),kt=f(e),N=i(e,"P",{});var Ae=c(N);Ko=n(Ae,`If we need to change the parameters of an already initialized Tensor, we can
    do the adjustments in a later step, primarily using the `),Ke=i(Ae,"CODE",{});var gn=c(Ke);Qo=n(gn,"to"),gn.forEach(t),Xo=n(Ae,`
    method of the Tensor class. The `),Qe=i(Ae,"CODE",{});var Pn=c(Qe);Yo=n(Pn,"to"),Pn.forEach(t),Zo=n(Ae,` method does not overwrite the
    original Tensor, but returns an adjusted one.`),Ae.forEach(t),qt=f(e),$(de.$$.fragment,e),jt=f(e),fe=i(e,"PRE",{class:!0});var xn=c(fe);er=n(xn,`Original Tensor: dtype=torch.int64, device=cpu, requires_grad=False
Adjusted dtype: dtype=torch.float32, device=cpu, requires_grad=False
Adjusted device: dtype=torch.float32, device=cuda:0, requires_grad=False
Adjusted requres_grad: dtype=torch.float32, device=cuda:0, requires_grad=True
  `),xn.forEach(t),Wt=f(e),R=i(e,"P",{});var ke=c(R);tr=n(ke,`In practice we are often interested in the shape of a particular tensor. We
    can use use my_`),Xe=i(ke,"CODE",{});var Cn=c(Xe);or=n(Cn,"tensor.size()"),Cn.forEach(t),rr=n(ke," or "),Ye=i(ke,"CODE",{});var On=c(Ye);nr=n(On,"my_tensor.shape"),On.forEach(t),sr=n(ke,` to
    find out the dimensions of the tensor.`),ke.forEach(t),St=f(e),$(me.$$.fragment,e),Bt=f(e),pe=i(e,"PRE",{class:!0});var Dn=c(pe);ar=n(Dn,`torch.Size([2, 3])
torch.Size([2, 3])
  `),Dn.forEach(t),It=f(e),X=i(e,"P",{});var io=c(X);lr=n(io,`PyTorch, like other frameworks that work with tensors, is extremely
    efficient when it comes to matrix operations. These operations are done in
    parallel and can be transfered to the GPU if you have a cuda compatibale
    graphics card. Essentially all of deep learning is based on matrix
    operations, so let"s spend some time to learn how we can invoke matrix
    operations using `),Ze=i(io,"CODE",{});var An=c(Ze);ir=n(An,"Tensor"),An.forEach(t),cr=n(io," objects."),io.forEach(t),zt=f(e),L=i(e,"P",{});var qe=c(L);dr=n(qe,"We will use two tensors, "),$(Y.$$.fragment,qe),fr=n(qe," and "),$(Z.$$.fragment,qe),mr=n(qe," to demonstrate basic mathematical operations."),qe.forEach(t),Ft=f(e),$(ue.$$.fragment,e),Nt=f(e),x=i(e,"P",{});var U=c(x);pr=n(U,`We can add, subtract, multiply and divide those matrices using basic
    mathematic operators like `),et=i(U,"CODE",{});var kn=c(et);ur=n(kn,"+"),kn.forEach(t),hr=n(U,", "),tt=i(U,"CODE",{});var qn=c(tt);vr=n(qn,"-"),qn.forEach(t),$r=n(U,", "),ot=i(U,"CODE",{});var jn=c(ot);wr=n(jn,"*"),jn.forEach(t),yr=n(U,`,
    `),rt=i(U,"CODE",{});var Wn=c(rt);_r=n(Wn,"/"),Wn.forEach(t),Er=n(U,`. All those operations work elementwise, so when you multiply
    two matrices you won't actually use matrix multiplication that involves dot
    products but elementwise multiplication.`),U.forEach(t),Rt=f(e),$(he.$$.fragment,e),Lt=f(e),ve=i(e,"PRE",{class:!0});var Sn=c(ve);br=n(Sn,`  tensor([[2., 3.],
          [4., 5.]])
  tensor([[ 0., -1.],
          [-2., -3.]])
  tensor([[1., 2.],
          [3., 4.]])
  tensor([[1.0000, 0.5000],
          [0.3333, 0.2500]])
  `),Sn.forEach(t),Ht=f(e),$e=i(e,"P",{});var nn=c($e);Tr=n(nn,"We can achieve the same results using the explicit methods: "),xe=i(nn,"CODE",{});var sn=c(xe);gr=n(sn,"Tensor.add()"),H=i(sn,"CODE",{});var Pe=c(H);Pr=n(Pe,", "),nt=i(Pe,"CODE",{});var Bn=c(nt);xr=n(Bn,"Tensor.subtract()"),Bn.forEach(t),Cr=n(Pe,", "),st=i(Pe,"CODE",{});var In=c(st);Or=n(In,"Tensor.multiply()"),In.forEach(t),Dr=n(Pe,`,
        `),at=i(Pe,"CODE",{});var zn=c(at);Ar=n(zn,"Tensor.divide()."),zn.forEach(t),Pe.forEach(t),sn.forEach(t),nn.forEach(t),Mt=f(e),$(we.$$.fragment,e),Gt=f(e),g=i(e,"P",{});var W=c(g);kr=n(W,`While the above methods do not change the original tensors, each of the
    methods has a corresponding method that changes the tensor in place. These
    methods always end with a `),lt=i(W,"CODE",{});var Fn=c(lt);qr=n(Fn,"_"),Fn.forEach(t),jr=n(W,": "),it=i(W,"CODE",{});var Nn=c(it);Wr=n(Nn,"add_()"),Nn.forEach(t),Sr=n(W,`,
    `),ct=i(W,"CODE",{});var Rn=c(ct);Br=n(Rn,"subtract_()"),Rn.forEach(t),Ir=n(W,", "),dt=i(W,"CODE",{});var Ln=c(dt);zr=n(Ln,"multiply_()"),Ln.forEach(t),Fr=n(W,", "),ft=i(W,"CODE",{});var Hn=c(ft);Nr=n(Hn,"divide_()"),Hn.forEach(t),Rr=n(W,"."),W.forEach(t),Ut=f(e),$(ye.$$.fragment,e),Vt=f(e),_e=i(e,"PRE",{class:!0});var Mn=c(_e);Lr=n(Mn,`tensor([[2., 3.],
        [5., 5.]])
  `),Mn.forEach(t),Jt=f(e),M=i(e,"P",{});var je=c(M);Hr=n(je,`Probaly one of the most important matrix operations in all of deep learning
    is product of two matrices, `),$(ee.$$.fragment,je),Mr=n(je,`.
    For that purpose we can use the `),mt=i(je,"CODE",{});var Gn=c(mt);Gr=n(Gn,"matmul"),Gn.forEach(t),Ur=n(je," method."),je.forEach(t),Kt=f(e),$(Ee.$$.fragment,e),Qt=f(e),be=i(e,"PRE",{class:!0});var Un=c(be);Vr=n(Un,`tensor([[4., 6.],
        [4., 6.]])
  `),Un.forEach(t),Xt=f(e),G=i(e,"P",{});var We=c(G);Jr=n(We,"Alternatively we can use "),pt=i(We,"CODE",{});var Vn=c(pt);Kr=n(Vn,"@"),Vn.forEach(t),Qr=n(We,` as a convenient way to use matrix
    multiplication. This is essentially just a shorthand notation for
    `),ut=i(We,"CODE",{});var Jn=c(ut);Xr=n(Jn,"torch.matmul"),Jn.forEach(t),Yr=n(We,"."),We.forEach(t),Yt=f(e),$(Te.$$.fragment,e),Zt=f(e),te=i(e,"P",{});var co=c(te);Zr=n(co,`A final concept that we would like to mention is the concept of dimensions
    in PyTorch. Often we would like to calculate some summary statistics (like a
    sum or a mean) for a Tensor object. But we would like those to be calculated
    for a particular dimension. We can explicitly set the dimension by defining
    the `),ht=i(co,"CODE",{});var Kn=c(ht);en=n(Kn,"dim"),Kn.forEach(t),tn=n(co," parameter."),co.forEach(t),eo=f(e),$(ge.$$.fragment,e),to=f(e),Ce=i(e,"PRE",{});var Qn=c(Ce);on=n(Qn,`tensor(21)
tensor([5, 7, 9])
tensor([ 6, 15])
  `),Qn.forEach(t),oo=f(e),Oe=i(e,"P",{});var Xn=c(Oe);rn=n(Xn,`The very first sum that we calculate in the example below, does not take any
    dimensions into consideration and just calculates the sum over the whole
    tensor. In the second example we calculate the sum over the 0th, the row,
    dimension. That means that for each of the available columns we calculate
    the sum by moving down the rows. When we calculate the sum for the 1st, the
    column dimension, we go over each row and calculate the sum by moving
    through the columns.`),Xn.forEach(t),ro=f(e),De=i(e,"DIV",{class:!0}),c(De).forEach(t),this.h()},h(){z(se,"class","text-sm"),z(ie,"class","text-sm"),z(fe,"class","text-sm"),z(pe,"class","text-sm"),z(ve,"class","text-sm"),z(_e,"class","text-sm"),z(be,"class","text-sm"),z(De,"class","separator")},m(e,s){a(e,p,s),o(p,h),w(m,p,null),o(p,O),w(D,p,null),a(e,B,s),a(e,P,s),o(P,A),o(P,F),o(F,u),o(P,b),o(P,I),o(I,fo),o(P,mo),a(e,yt,s),w(ne,e,s),a(e,_t,s),a(e,se,s),o(se,po),a(e,Et,s),a(e,J,s),o(J,uo),o(J,Se),o(Se,ho),o(J,vo),a(e,bt,s),a(e,K,s),o(K,$o),o(K,Be),o(Be,wo),o(K,yo),a(e,Tt,s),w(ae,e,s),a(e,gt,s),a(e,T,s),o(T,_o),o(T,Ie),o(Ie,Eo),o(T,bo),o(T,ze),o(ze,To),o(T,go),o(T,Fe),o(Fe,Po),o(T,xo),o(T,Ne),o(Ne,Co),o(T,Oo),o(T,Re),o(Re,Do),o(T,Ao),a(e,Pt,s),w(le,e,s),a(e,xt,s),a(e,ie,s),o(ie,ko),a(e,Ct,s),a(e,k,s),o(k,qo),o(k,Le),o(Le,jo),o(k,Wo),o(k,He),o(He,So),o(k,Bo),o(k,Me),o(Me,Io),o(k,zo),a(e,Ot,s),w(ce,e,s),a(e,Dt,s),a(e,Q,s),o(Q,Fo),o(Q,Ge),o(Ge,No),o(Q,Ro),a(e,At,s),a(e,q,s),o(q,Lo),o(q,Ue),o(Ue,Ho),o(q,Mo),o(q,Ve),o(Ve,Go),o(q,Uo),o(q,Je),o(Je,Vo),o(q,Jo),a(e,kt,s),a(e,N,s),o(N,Ko),o(N,Ke),o(Ke,Qo),o(N,Xo),o(N,Qe),o(Qe,Yo),o(N,Zo),a(e,qt,s),w(de,e,s),a(e,jt,s),a(e,fe,s),o(fe,er),a(e,Wt,s),a(e,R,s),o(R,tr),o(R,Xe),o(Xe,or),o(R,rr),o(R,Ye),o(Ye,nr),o(R,sr),a(e,St,s),w(me,e,s),a(e,Bt,s),a(e,pe,s),o(pe,ar),a(e,It,s),a(e,X,s),o(X,lr),o(X,Ze),o(Ze,ir),o(X,cr),a(e,zt,s),a(e,L,s),o(L,dr),w(Y,L,null),o(L,fr),w(Z,L,null),o(L,mr),a(e,Ft,s),w(ue,e,s),a(e,Nt,s),a(e,x,s),o(x,pr),o(x,et),o(et,ur),o(x,hr),o(x,tt),o(tt,vr),o(x,$r),o(x,ot),o(ot,wr),o(x,yr),o(x,rt),o(rt,_r),o(x,Er),a(e,Rt,s),w(he,e,s),a(e,Lt,s),a(e,ve,s),o(ve,br),a(e,Ht,s),a(e,$e,s),o($e,Tr),o($e,xe),o(xe,gr),o(xe,H),o(H,Pr),o(H,nt),o(nt,xr),o(H,Cr),o(H,st),o(st,Or),o(H,Dr),o(H,at),o(at,Ar),a(e,Mt,s),w(we,e,s),a(e,Gt,s),a(e,g,s),o(g,kr),o(g,lt),o(lt,qr),o(g,jr),o(g,it),o(it,Wr),o(g,Sr),o(g,ct),o(ct,Br),o(g,Ir),o(g,dt),o(dt,zr),o(g,Fr),o(g,ft),o(ft,Nr),o(g,Rr),a(e,Ut,s),w(ye,e,s),a(e,Vt,s),a(e,_e,s),o(_e,Lr),a(e,Jt,s),a(e,M,s),o(M,Hr),w(ee,M,null),o(M,Mr),o(M,mt),o(mt,Gr),o(M,Ur),a(e,Kt,s),w(Ee,e,s),a(e,Qt,s),a(e,be,s),o(be,Vr),a(e,Xt,s),a(e,G,s),o(G,Jr),o(G,pt),o(pt,Kr),o(G,Qr),o(G,ut),o(ut,Xr),o(G,Yr),a(e,Yt,s),w(Te,e,s),a(e,Zt,s),a(e,te,s),o(te,Zr),o(te,ht),o(ht,en),o(te,tn),a(e,eo,s),w(ge,e,s),a(e,to,s),a(e,Ce,s),o(Ce,on),a(e,oo,s),a(e,Oe,s),o(Oe,rn),a(e,ro,s),a(e,De,s),no=!0},p(e,s){const V={};s&1&&(V.$$scope={dirty:s,ctx:e}),m.$set(V);const vt={};s&1&&(vt.$$scope={dirty:s,ctx:e}),Y.$set(vt);const $t={};s&1&&($t.$$scope={dirty:s,ctx:e}),Z.$set($t);const wt={};s&1&&(wt.$$scope={dirty:s,ctx:e}),ee.$set(wt)},i(e){no||(y(m.$$.fragment,e),y(D.$$.fragment,e),y(ne.$$.fragment,e),y(ae.$$.fragment,e),y(le.$$.fragment,e),y(ce.$$.fragment,e),y(de.$$.fragment,e),y(me.$$.fragment,e),y(Y.$$.fragment,e),y(Z.$$.fragment,e),y(ue.$$.fragment,e),y(he.$$.fragment,e),y(we.$$.fragment,e),y(ye.$$.fragment,e),y(ee.$$.fragment,e),y(Ee.$$.fragment,e),y(Te.$$.fragment,e),y(ge.$$.fragment,e),no=!0)},o(e){_(m.$$.fragment,e),_(D.$$.fragment,e),_(ne.$$.fragment,e),_(ae.$$.fragment,e),_(le.$$.fragment,e),_(ce.$$.fragment,e),_(de.$$.fragment,e),_(me.$$.fragment,e),_(Y.$$.fragment,e),_(Z.$$.fragment,e),_(ue.$$.fragment,e),_(he.$$.fragment,e),_(we.$$.fragment,e),_(ye.$$.fragment,e),_(ee.$$.fragment,e),_(Ee.$$.fragment,e),_(Te.$$.fragment,e),_(ge.$$.fragment,e),no=!1},d(e){e&&t(p),E(m),E(D),e&&t(B),e&&t(P),e&&t(yt),E(ne,e),e&&t(_t),e&&t(se),e&&t(Et),e&&t(J),e&&t(bt),e&&t(K),e&&t(Tt),E(ae,e),e&&t(gt),e&&t(T),e&&t(Pt),E(le,e),e&&t(xt),e&&t(ie),e&&t(Ct),e&&t(k),e&&t(Ot),E(ce,e),e&&t(Dt),e&&t(Q),e&&t(At),e&&t(q),e&&t(kt),e&&t(N),e&&t(qt),E(de,e),e&&t(jt),e&&t(fe),e&&t(Wt),e&&t(R),e&&t(St),E(me,e),e&&t(Bt),e&&t(pe),e&&t(It),e&&t(X),e&&t(zt),e&&t(L),E(Y),E(Z),e&&t(Ft),E(ue,e),e&&t(Nt),e&&t(x),e&&t(Rt),E(he,e),e&&t(Lt),e&&t(ve),e&&t(Ht),e&&t($e),e&&t(Mt),E(we,e),e&&t(Gt),e&&t(g),e&&t(Ut),E(ye,e),e&&t(Vt),e&&t(_e),e&&t(Jt),e&&t(M),E(ee),e&&t(Kt),E(Ee,e),e&&t(Qt),e&&t(be),e&&t(Xt),e&&t(G),e&&t(Yt),E(Te,e),e&&t(Zt),e&&t(te),e&&t(eo),E(ge,e),e&&t(to),e&&t(Ce),e&&t(oo),e&&t(Oe),e&&t(ro),e&&t(De)}}}function cs(S){let p,h,m,O,D,B,P,A,F;return A=new os({props:{$$slots:{default:[is]},$$scope:{ctx:S}}}),{c(){p=l("meta"),h=d(),m=l("h1"),O=r("PyTorch Tensors"),D=d(),B=l("div"),P=d(),v(A.$$.fragment),this.h()},l(u){const b=ts("svelte-3livbz",document.head);p=i(b,"META",{name:!0,content:!0}),b.forEach(t),h=f(u),m=i(u,"H1",{});var I=c(m);O=n(I,"PyTorch Tensors"),I.forEach(t),D=f(u),B=i(u,"DIV",{class:!0}),c(B).forEach(t),P=f(u),$(A.$$.fragment,u),this.h()},h(){document.title="PyTorch Tensors - World4AI",z(p,"name","description"),z(p,"content","All deep learning libraries are build around a tensor object. A tensor is primarily used to store matrices of different dimensions and to apply rules of linear algebra."),z(B,"class","separator")},m(u,b){o(document.head,p),a(u,h,b),a(u,m,b),o(m,O),a(u,D,b),a(u,B,b),a(u,P,b),w(A,u,b),F=!0},p(u,[b]){const I={};b&1&&(I.$$scope={dirty:b,ctx:u}),A.$set(I)},i(u){F||(y(A.$$.fragment,u),F=!0)},o(u){_(A.$$.fragment,u),F=!1},d(u){t(p),u&&t(h),u&&t(m),u&&t(D),u&&t(B),u&&t(P),E(A,u)}}}class hs extends Yn{constructor(p){super(),Zn(this,p,null,cs,es,{})}}export{hs as default};
