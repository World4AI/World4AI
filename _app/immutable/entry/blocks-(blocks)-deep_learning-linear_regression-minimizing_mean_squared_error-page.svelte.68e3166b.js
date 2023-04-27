import{S as Ql,i as Zl,s as ei,k as d,a as h,q as i,y as w,W as ti,l as c,h as a,c as m,m as E,r as o,z as g,n as D,N as p,b as l,A as _,g as v,d as b,B as y,C as O}from"../chunks/index.4d92b023.js";import{C as ai}from"../chunks/Container.b0705c7b.js";import{H as bt}from"../chunks/Highlight.b7c1de53.js";import{L as P}from"../chunks/Latex.e0b308c0.js";import{B as Ul}from"../chunks/ButtonContainer.e9aac418.js";import{P as Kl}from"../chunks/PlayButton.85103c5a.js";import{A as yn}from"../chunks/Alert.25a852b3.js";import{P as le}from"../chunks/PythonCode.212ba7a6.js";import{M as ni}from"../chunks/Mse.e3b58c1f.js";import{B as Ee}from"../chunks/BackpropGraph.6a7a3666.js";import{V as L}from"../chunks/Network.03de8e4c.js";function ri(f){let r=String.raw`\mathbf{w}`+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function si(f){let r;return{c(){r=i("b")},l(t){r=o(t,"b")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function li(f){let r=String.raw`
        MSE=\dfrac{1}{n}\sum_i^n (y^{(i)} - \hat{y}^{(i)})^2 \\

        `+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function ii(f){let r=String.raw`
             \hat{y}^{(i)} = \mathbf{x}^{(i)} \mathbf{w}^T + b 
     `+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function oi(f){let r,t,s,u,W,q;return r=new P({props:{$$slots:{default:[li]},$$scope:{ctx:f}}}),W=new P({props:{$$slots:{default:[ii]},$$scope:{ctx:f}}}),{c(){w(r.$$.fragment),t=h(),s=d("div"),u=h(),w(W.$$.fragment),this.h()},l($){g(r.$$.fragment,$),t=m($),s=c($,"DIV",{class:!0}),E(s).forEach(a),u=m($),g(W.$$.fragment,$),this.h()},h(){D(s,"class","my-2")},m($,T){_(r,$,T),l($,t,T),l($,s,T),l($,u,T),_(W,$,T),q=!0},p($,T){const S={};T&536870912&&(S.$$scope={dirty:T,ctx:$}),r.$set(S);const k={};T&536870912&&(k.$$scope={dirty:T,ctx:$}),W.$set(k)},i($){q||(v(r.$$.fragment,$),v(W.$$.fragment,$),q=!0)},o($){b(r.$$.fragment,$),b(W.$$.fragment,$),q=!1},d($){y(r,$),$&&a(t),$&&a(s),$&&a(u),y(W,$)}}}function fi(f){let r;return{c(){r=i("x")},l(t){r=o(t,"x")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function pi(f){let r=String.raw` MSE=(y - [xw + b])^2`+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function ui(f){let r;return{c(){r=i("chain rule")},l(t){r=o(t,"chain rule")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function $i(f){let r;return{c(){r=i("z(y(x))")},l(t){r=o(t,"z(y(x))")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function hi(f){let r;return{c(){r=i("z")},l(t){r=o(t,"z")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function mi(f){let r;return{c(){r=i("x")},l(t){r=o(t,"x")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function di(f){let r=String.raw`
        \dfrac{dz}{dx} = \dfrac{dz}{dy} \dfrac{dy}{dx} = \dfrac{dz}{\xcancel{dy}} \dfrac{\xcancel{dy}}{dx} = \dfrac{dz}{dx} 
    `+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function ci(f){let r,t,s,u,W,q,$,T,S,k,z;return t=new P({props:{$$slots:{default:[$i]},$$scope:{ctx:f}}}),u=new P({props:{$$slots:{default:[hi]},$$scope:{ctx:f}}}),q=new P({props:{$$slots:{default:[mi]},$$scope:{ctx:f}}}),k=new P({props:{$$slots:{default:[di]},$$scope:{ctx:f}}}),{c(){r=i("If we have a composite function "),w(t.$$.fragment),s=i(`, we can calculate the
    derivative of `),w(u.$$.fragment),W=i(" with respect of "),w(q.$$.fragment),$=i(` by applying the
    chain rule.
    `),T=d("div"),S=h(),w(k.$$.fragment),this.h()},l(x){r=o(x,"If we have a composite function "),g(t.$$.fragment,x),s=o(x,`, we can calculate the
    derivative of `),g(u.$$.fragment,x),W=o(x," with respect of "),g(q.$$.fragment,x),$=o(x,` by applying the
    chain rule.
    `),T=c(x,"DIV",{class:!0}),E(T).forEach(a),S=m(x),g(k.$$.fragment,x),this.h()},h(){D(T,"class","mb-2")},m(x,I){l(x,r,I),_(t,x,I),l(x,s,I),_(u,x,I),l(x,W,I),_(q,x,I),l(x,$,I),l(x,T,I),l(x,S,I),_(k,x,I),z=!0},p(x,I){const J={};I&536870912&&(J.$$scope={dirty:I,ctx:x}),t.$set(J);const ue={};I&536870912&&(ue.$$scope={dirty:I,ctx:x}),u.$set(ue);const $e={};I&536870912&&($e.$$scope={dirty:I,ctx:x}),q.$set($e);const R={};I&536870912&&(R.$$scope={dirty:I,ctx:x}),k.$set(R)},i(x){z||(v(t.$$.fragment,x),v(u.$$.fragment,x),v(q.$$.fragment,x),v(k.$$.fragment,x),z=!0)},o(x){b(t.$$.fragment,x),b(u.$$.fragment,x),b(q.$$.fragment,x),b(k.$$.fragment,x),z=!1},d(x){x&&a(r),y(t,x),x&&a(s),y(u,x),x&&a(W),y(q,x),x&&a($),x&&a(T),x&&a(S),y(k,x)}}}function wi(f){let r=String.raw`(y - [xw + b])^2`+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function gi(f){let r;return{c(){r=i("s")},l(t){r=o(t,"s")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function _i(f){let r;return{c(){r=i("s = w\\times x")},l(t){r=o(t,"s = w\\times x")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function vi(f){let r=String.raw`\hat{y} = s + b`+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function bi(f){let r=String.raw`\hat{y}`+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function yi(f){let r;return{c(){r=i("e")},l(t){r=o(t,"e")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function Ei(f){let r;return{c(){r=i("y")},l(t){r=o(t,"y")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function ki(f){let r=String.raw`e = y - \hat{y}`+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function xi(f){let r;return{c(){r=i("mse = e^2")},l(t){r=o(t,"mse = e^2")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function Ti(f){let r;return{c(){r=i("b")},l(t){r=o(t,"b")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function Si(f){let r;return{c(){r=i("w")},l(t){r=o(t,"w")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function Wi(f){let r=String.raw`\dfrac{\partial mse}{\partial b} = \dfrac{\partial mse}{\partial e} \times \dfrac{\partial e}{\partial \hat{y}} \times \dfrac{\partial \hat{y}}{\partial b}`+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function qi(f){let r=String.raw`\dfrac{\partial mse}{dw} = \dfrac{\partial mse}{\partial e} \times \dfrac{\partial e}{\partial \hat{y}} \times \dfrac{\partial \hat{y}}{\partial s} \times \dfrac{\partial s}{\partial w}`+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function Pi(f){let r=String.raw`\dfrac{\partial mse}{\partial w_j} = \dfrac{\partial mse}{\partial e} \times \dfrac{\partial e}{\partial \hat{y}} \times \dfrac{\partial \hat{y}}{\partial s} \times \dfrac{\partial s}{\partial w_j}`+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function Mi(f){let r,t,s,u,W,q,$,T;return u=new P({props:{$$slots:{default:[Pi]},$$scope:{ctx:f}}}),{c(){r=i(`For didactic reasons we are focusing on linear regression with a single
    feature, but for the most part the calcultion with more than one feature
    would be the same.
    `),t=d("div"),s=h(),w(u.$$.fragment),W=h(),q=d("div"),$=i(`
    In that case we have to calculate as many partial derivatives, as there are features.`),this.h()},l(S){r=o(S,`For didactic reasons we are focusing on linear regression with a single
    feature, but for the most part the calcultion with more than one feature
    would be the same.
    `),t=c(S,"DIV",{class:!0}),E(t).forEach(a),s=m(S),g(u.$$.fragment,S),W=m(S),q=c(S,"DIV",{class:!0}),E(q).forEach(a),$=o(S,`
    In that case we have to calculate as many partial derivatives, as there are features.`),this.h()},h(){D(t,"class","mb-2"),D(q,"class","mb-2")},m(S,k){l(S,r,k),l(S,t,k),l(S,s,k),_(u,S,k),l(S,W,k),l(S,q,k),l(S,$,k),T=!0},p(S,k){const z={};k&536870912&&(z.$$scope={dirty:k,ctx:S}),u.$set(z)},i(S){T||(v(u.$$.fragment,S),T=!0)},o(S){b(u.$$.fragment,S),T=!1},d(S){S&&a(r),S&&a(t),S&&a(s),y(u,S),S&&a(W),S&&a(q),S&&a($)}}}function zi(f){let r=String.raw`\dfrac{\partial mse}{\partial e} = 2e`+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function Ii(f){let r=String.raw`\dfrac{\partial e}{\partial \hat{y}} = -1`+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function ji(f){let r=String.raw`\dfrac{\partial \hat{y}}{\partial \hat{b}} = 1`+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function Di(f){let r=String.raw`\dfrac{\partial \hat{y}}{\partial \hat{s}} = 1`+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function Oi(f){let r=String.raw`\dfrac{\partial s}{\partial w} = x`+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function Bi(f){let r=String.raw`
  \begin{aligned} 
  \dfrac{\partial mse}{\partial b} &= 2e * (-1) * 1 \\
  &= -2(y - \hat{y})  \\
  &= -2(y - (wx + b)) 
  \end{aligned}
  `+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function Ni(f){let r=String.raw`
  \begin{aligned} 
  \dfrac{\partial mse}{\partial b} &= 2e * (-1) * 1 * x \\
  &= -2x(y - \hat{y})  \\
  &= -2x(y - (wx + b)) 
  \end{aligned}
  `+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function Ai(f){let r=String.raw`\mathbf{w}_{t+1} \coloneqq \mathbf{w}_t - \alpha \mathbf{\nabla}_w `+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function Ci(f){let r=String.raw`b_{t+1} \coloneqq b_t - \alpha \dfrac{\partial}{\partial b} `+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function Gi(f){let r,t,s,u,W,q;return r=new P({props:{$$slots:{default:[Ai]},$$scope:{ctx:f}}}),W=new P({props:{$$slots:{default:[Ci]},$$scope:{ctx:f}}}),{c(){w(r.$$.fragment),t=h(),s=d("div"),u=h(),w(W.$$.fragment),this.h()},l($){g(r.$$.fragment,$),t=m($),s=c($,"DIV",{class:!0}),E(s).forEach(a),u=m($),g(W.$$.fragment,$),this.h()},h(){D(s,"class","mb-2")},m($,T){_(r,$,T),l($,t,T),l($,s,T),l($,u,T),_(W,$,T),q=!0},p($,T){const S={};T&536870912&&(S.$$scope={dirty:T,ctx:$}),r.$set(S);const k={};T&536870912&&(k.$$scope={dirty:T,ctx:$}),W.$set(k)},i($){q||(v(r.$$.fragment,$),v(W.$$.fragment,$),q=!0)},o($){b(r.$$.fragment,$),b(W.$$.fragment,$),q=!1},d($){y(r,$),$&&a(t),$&&a(s),$&&a(u),y(W,$)}}}function Vi(f){let r;return{c(){r=i("computational graph")},l(t){r=o(t,"computational graph")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function Fi(f){let r;return{c(){r=i("automatic differentiation")},l(t){r=o(t,"automatic differentiation")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function Ji(f){let r,t,s,u,W,q;return t=new bt({props:{$$slots:{default:[Vi]},$$scope:{ctx:f}}}),u=new bt({props:{$$slots:{default:[Fi]},$$scope:{ctx:f}}}),{c(){r=i("We can construct a so called "),w(t.$$.fragment),s=i(` and use
    the chain rule for `),w(u.$$.fragment),W=i(".")},l($){r=o($,"We can construct a so called "),g(t.$$.fragment,$),s=o($,` and use
    the chain rule for `),g(u.$$.fragment,$),W=o($,".")},m($,T){l($,r,T),_(t,$,T),l($,s,T),_(u,$,T),l($,W,T),q=!0},p($,T){const S={};T&536870912&&(S.$$scope={dirty:T,ctx:$}),t.$set(S);const k={};T&536870912&&(k.$$scope={dirty:T,ctx:$}),u.$set(k)},i($){q||(v(t.$$.fragment,$),v(u.$$.fragment,$),q=!0)},o($){b(t.$$.fragment,$),b(u.$$.fragment,$),q=!1},d($){$&&a(r),y(t,$),$&&a(s),y(u,$),$&&a(W)}}}function Ri(f){let r;return{c(){r=i("w")},l(t){r=o(t,"w")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function Hi(f){let r;return{c(){r=i("x")},l(t){r=o(t,"x")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function Li(f){let r;return{c(){r=i("w*x")},l(t){r=o(t,"w*x")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function Xi(f){let r;return{c(){r=i("b")},l(t){r=o(t,"b")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function Ui(f){let r;return{c(){r=i("e^2")},l(t){r=o(t,"e^2")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function Yi(f){let r;return{c(){r=i("2 \\times error")},l(t){r=o(t,"2 \\times error")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function Ki(f){let r;return{c(){r=i("Error = Target - Prediction")},l(t){r=o(t,"Error = Target - Prediction")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function Qi(f){let r,t;return r=new Kl({props:{f:f[10],delta:200}}),{c(){w(r.$$.fragment)},l(s){g(r.$$.fragment,s)},m(s,u){_(r,s,u),t=!0},p:O,i(s){t||(v(r.$$.fragment,s),t=!0)},o(s){b(r.$$.fragment,s),t=!1},d(s){y(r,s)}}}function Zi(f){let r;return{c(){r=i("forward pass")},l(t){r=o(t,"forward pass")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function eo(f){let r;return{c(){r=i("backward pass")},l(t){r=o(t,"backward pass")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function to(f){let r;return{c(){r=i("backpropagation")},l(t){r=o(t,"backpropagation")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function ao(f){let r,t;return r=new Kl({props:{f:f[9],delta:1}}),{c(){w(r.$$.fragment)},l(s){g(r.$$.fragment,s)},m(s,u){_(r,s,u),t=!0},p:O,i(s){t||(v(r.$$.fragment,s),t=!0)},o(s){b(r.$$.fragment,s),t=!1},d(s){y(r,s)}}}function no(f){let r;return{c(){r=i("autograd")},l(t){r=o(t,"autograd")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function ro(f){let r=String.raw`\mathbf{X}`+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function so(f){let r=String.raw`\mathbf{y}`+"",t;return{c(){t=i(r)},l(s){t=o(s,r)},m(s,u){l(s,t,u)},p:O,d(s){s&&a(t)}}}function lo(f){let r;return{c(){r=i("training loop")},l(t){r=o(t,"training loop")},m(t,s){l(t,r,s)},d(t){t&&a(r)}}}function io(f){let r,t,s,u,W,q,$,T,S,k,z,x,I,J,ue,$e,R,Xt,Q,yt,Ea,he,ka,j,En,me,kn,de,xn,ce,Tn,M,G,B,ke,V,ie,H,we,Z,as,xe,ns,Sn,oe,rs,Te,ss,Se,ls,Wn,ee,We,is,qn,os,qe,Pn,Pe,Mn,Ut,fs,zn,N,Me,ps,xa,us,ze,$s,Ta,hs,Ie,ms,Sa,ds,je,cs,Wa,ws,De,In,Yt,gs,jn,te,Oe,_s,Dn,vs,Be,On,Kt,bs,Bn,Ne,Nn,Qt,An,Zt,ys,Cn,ea,Es,Gn,Ae,Vn,ta,ks,Fn,X,xs,Ce,Ts,Ge,Ss,Ve,Ws,Jn,Fe,Rn,aa,qs,Hn,Je,Ps,Re,Ms,Ln,He,Xn,na,zs,Un,Le,Yn,ra,Is,Kn,sa,js,Qn,Xe,Zn,la,Ds,er,fe,Os,Ue,Bs,Ye,Ns,tr,Ke,ar,Qe,As,Ze,Cs,nr,et,rr,ia,Gs,sr,tt,lr,oa,Vs,ir,at,or,nt,fr,fa,Fs,pr,pa,Js,ur,U,Rs,rt,Hs,st,Ls,lt,Xs,$r,ua,hr,$a,Us,mr,ha,Ys,dr,it,cr,ma,Ks,wr,da,Qs,gr,ca,Zs,_r,ot,vr,ft,br,wa,yr,ga,el,Er,pt,tl,ut,al,kr,Et,xr,pe,nl,$t,rl,ht,sl,Tr,kt,Sr,mt,ll,qa,il,ol,Wr,xt,qr,Y,fl,Pa,pl,ul,Ma,$l,hl,za,ml,dl,Pr,Tt,Mr,K,cl,Ia,wl,gl,ja,_l,vl,Da,bl,yl,zr,St,Ir,Wt,El,jr,dt,kl,Oa,xl,Tl,Dr,qt,Or,_a,Sl,Br,Pt,Nr,Mt,Wl,Ar,ct,ql,Ba,Pl,Ml,Cr,zt,Gr,It,zl,Vr,va,Il,Fr,jt,Jr,Dt,jl,Rr,wt,Dl,Na,Ol,Bl,Hr,Ot,Lr,Bt,Nl,Xr,gt,Al,Aa,Cl,Gl,Ur,Nt,Yr,At,Vl,Kr,_t,Fl,vt,Jl,Qr,ba,Zr;return q=new P({props:{$$slots:{default:[ri]},$$scope:{ctx:f}}}),T=new P({props:{$$slots:{default:[si]},$$scope:{ctx:f}}}),k=new yn({props:{type:"info",$$slots:{default:[oi]},$$scope:{ctx:f}}}),x=new P({props:{$$slots:{default:[fi]},$$scope:{ctx:f}}}),J=new P({props:{$$slots:{default:[pi]},$$scope:{ctx:f}}}),Q=new bt({props:{$$slots:{default:[ui]},$$scope:{ctx:f}}}),he=new yn({props:{type:"info",$$slots:{default:[ci]},$$scope:{ctx:f}}}),me=new P({props:{$$slots:{default:[wi]},$$scope:{ctx:f}}}),de=new P({props:{$$slots:{default:[gi]},$$scope:{ctx:f}}}),ce=new P({props:{$$slots:{default:[_i]},$$scope:{ctx:f}}}),M=new P({props:{$$slots:{default:[vi]},$$scope:{ctx:f}}}),B=new P({props:{$$slots:{default:[bi]},$$scope:{ctx:f}}}),V=new P({props:{$$slots:{default:[yi]},$$scope:{ctx:f}}}),H=new P({props:{$$slots:{default:[Ei]},$$scope:{ctx:f}}}),Z=new P({props:{$$slots:{default:[ki]},$$scope:{ctx:f}}}),xe=new P({props:{$$slots:{default:[xi]},$$scope:{ctx:f}}}),Te=new P({props:{$$slots:{default:[Ti]},$$scope:{ctx:f}}}),Se=new P({props:{$$slots:{default:[Si]},$$scope:{ctx:f}}}),We=new P({props:{$$slots:{default:[Wi]},$$scope:{ctx:f}}}),qe=new P({props:{$$slots:{default:[qi]},$$scope:{ctx:f}}}),Pe=new yn({props:{type:"info",$$slots:{default:[Mi]},$$scope:{ctx:f}}}),Me=new P({props:{$$slots:{default:[zi]},$$scope:{ctx:f}}}),ze=new P({props:{$$slots:{default:[Ii]},$$scope:{ctx:f}}}),Ie=new P({props:{$$slots:{default:[ji]},$$scope:{ctx:f}}}),je=new P({props:{$$slots:{default:[Di]},$$scope:{ctx:f}}}),De=new P({props:{$$slots:{default:[Oi]},$$scope:{ctx:f}}}),Oe=new P({props:{$$slots:{default:[Bi]},$$scope:{ctx:f}}}),Be=new P({props:{$$slots:{default:[Ni]},$$scope:{ctx:f}}}),Ne=new yn({props:{type:"info",$$slots:{default:[Gi]},$$scope:{ctx:f}}}),Ae=new yn({props:{type:"info",$$slots:{default:[Ji]},$$scope:{ctx:f}}}),Ce=new P({props:{$$slots:{default:[Ri]},$$scope:{ctx:f}}}),Ge=new P({props:{$$slots:{default:[Hi]},$$scope:{ctx:f}}}),Ve=new P({props:{$$slots:{default:[Li]},$$scope:{ctx:f}}}),Fe=new Ee({props:{graph:f[3],maxWidth:300,height:300,width:280}}),Re=new P({props:{$$slots:{default:[Xi]},$$scope:{ctx:f}}}),He=new Ee({props:{graph:f[4]}}),Le=new Ee({props:{graph:f[5],maxWidth:350,width:450,height:910}}),Xe=new Ee({props:{graph:f[6],maxWidth:150,width:150,height:100}}),Ue=new P({props:{$$slots:{default:[Ui]},$$scope:{ctx:f}}}),Ye=new P({props:{$$slots:{default:[Yi]},$$scope:{ctx:f}}}),Ke=new Ee({props:{graph:f[6],maxWidth:150,width:150,height:300}}),Ze=new P({props:{$$slots:{default:[Ki]},$$scope:{ctx:f}}}),et=new Ee({props:{graph:f[6],maxWidth:300,width:300,height:500}}),tt=new Ee({props:{graph:f[6],maxWidth:350,width:450,height:910}}),at=new Ul({props:{$$slots:{default:[Qi]},$$scope:{ctx:f}}}),nt=new Ee({props:{graph:f[2],maxWidth:350,width:450,height:910}}),rt=new bt({props:{$$slots:{default:[Zi]},$$scope:{ctx:f}}}),st=new bt({props:{$$slots:{default:[eo]},$$scope:{ctx:f}}}),lt=new bt({props:{$$slots:{default:[to]},$$scope:{ctx:f}}}),it=new Ee({props:{graph:f[7],maxWidth:350,height:500,width:450}}),ot=new Ul({props:{$$slots:{default:[ao]},$$scope:{ctx:f}}}),ft=new ni({props:{data:f[8],w:f[0].data,b:f[1].data}}),ut=new bt({props:{$$slots:{default:[no]},$$scope:{ctx:f}}}),Et=new le({props:{code:f[11]}}),$t=new P({props:{$$slots:{default:[ro]},$$scope:{ctx:f}}}),ht=new P({props:{$$slots:{default:[so]},$$scope:{ctx:f}}}),kt=new le({props:{code:f[12]}}),xt=new le({props:{code:f[13]}}),Tt=new le({props:{code:f[14]}}),St=new le({props:{code:f[15]}}),qt=new le({props:{code:f[16]}}),Pt=new le({props:{code:f[17]}}),zt=new le({props:{code:f[18]}}),jt=new le({props:{code:f[19]}}),Ot=new le({props:{code:f[20]}}),Nt=new le({props:{code:f[21]}}),vt=new bt({props:{$$slots:{default:[lo]},$$scope:{ctx:f}}}),{c(){r=d("h2"),t=i("Single Training Sample"),s=h(),u=d("p"),W=i(`Let us remind ourselves that our goal is to minimize the mean squared error,
    by tweaking the weight vector `),w(q.$$.fragment),$=i(` and the
    bias scalar `),w(T.$$.fragment),S=i(` using gradient descent.
    `),w(k.$$.fragment),z=i(`
    To make our journey easier, let us for now assume that we have one single feature
    `),w(x.$$.fragment),I=i(` and one single training sample. That reduces the mean squared
    error to a much simpler form:
    `),w(J.$$.fragment),ue=i(`. We will extend our
    calculations once we have covered the basics.`),$e=h(),R=d("p"),Xt=i(`The computation of gradients in the deep learning world relies heavily on
    the `),w(Q.$$.fragment),yt=i("."),Ea=h(),w(he.$$.fragment),ka=h(),j=d("p"),En=i("The mean squared error "),w(me.$$.fragment),kn=i(` is a great
    example of a composite function. We start by calculating the scaled feature
    `),w(de.$$.fragment),xn=i(", where "),w(ce.$$.fragment),Tn=i(`. We use s as an input
    and calculate the linear regression prediction, `),w(M.$$.fragment),G=i(". Using "),w(B.$$.fragment),ke=i(` as input, we can calculate the
    error `),w(V.$$.fragment),ie=i(` as the difference between the prediction and the true
    target `),w(H.$$.fragment),we=i(", "),w(Z.$$.fragment),as=i(`.
    Finally we calculate the mean squared error for one single training example, `),w(xe.$$.fragment),ns=i("."),Sn=h(),oe=d("p"),rs=i(`If we utilize the chain rule, we can calculate the derivative of the mean
    squared error with respect to the bias `),w(Te.$$.fragment),ss=i(" and the weight "),w(Se.$$.fragment),ls=i(" by multiplying the intermediary derivatives."),Wn=h(),ee=d("div"),w(We.$$.fragment),is=h(),qn=d("div"),os=h(),w(qe.$$.fragment),Pn=h(),w(Pe.$$.fragment),Mn=h(),Ut=d("p"),fs=i(`Calculating those intermediary derivatives is relatively straightforward,
    using basic rules of calculus.`),zn=h(),N=d("div"),w(Me.$$.fragment),ps=h(),xa=d("div"),us=h(),w(ze.$$.fragment),$s=h(),Ta=d("div"),hs=h(),w(Ie.$$.fragment),ms=h(),Sa=d("div"),ds=h(),w(je.$$.fragment),cs=h(),Wa=d("div"),ws=h(),w(De.$$.fragment),In=h(),Yt=d("p"),gs=i(`Using the chain rule we can easily calculate the derivatives with respect to
    the weight and bias.`),jn=h(),te=d("div"),w(Oe.$$.fragment),_s=h(),Dn=d("div"),vs=h(),w(Be.$$.fragment),On=h(),Kt=d("p"),bs=i(`Once we have the gradients, the gradient descent algorithm works as
    expected.`),Bn=h(),w(Ne.$$.fragment),Nn=h(),Qt=d("div"),An=h(),Zt=d("h2"),ys=i("Computationial Graph"),Cn=h(),ea=d("p"),Es=i(`We have mentioned before, that the chain rule plays an integral role in deep
    learning, but what we have covered so far has probably not made it clear why
    it is so essential.`),Gn=h(),w(Ae.$$.fragment),Vn=h(),ta=d("p"),ks=i(`So let's construct the computational graph for the mean squared error step
    by step and see how automatic differentiation looks like.`),Fn=h(),X=d("p"),xs=i(`A computational graph basically tracks all atomic calculatios and their
    results in a tree like structure. We start out the calculation of MSE by
    creating the weight node `),w(Ce.$$.fragment),Ts=i(" and the feature node "),w(Ge.$$.fragment),Ss=i(`, but at that point those two variables are not connected to each other
    yet. Once we calculate `),w(Ve.$$.fragment),Ws=i(`, we connect the two and the result
    is a new node, that lies on a higher level.`),Jn=h(),w(Fe.$$.fragment),Rn=h(),aa=d("p"),qs=i(`The yellow box represents the operations (like additions and
    multiplications), blue boxes contains the values of the node, while the red
    boxes contain the derivatives. We assume a feature of 5 and the weight of 1
    at the beginning, so the resulting node has the value of 5. At this point in
    time we have not calculated any derivatives yet, so all derivatives will
    amount to 0 for now.`),Hn=h(),Je=d("p"),Ps=i("Next we create the bias "),w(Re.$$.fragment),Ms=i(` node with the initial value of 1 and
    add the node to the previously calculated node. Once again those nodes are reflected
    in the computational graph.`),Ln=h(),w(He.$$.fragment),Xn=h(),na=d("p"),zs=i(`We keep creating new nodes, and connect them to previous generated nodes and
    the graph keeps growing. We do that until we reach the final node, in our
    case the mean squared error.`),Un=h(),w(Le.$$.fragment),Yn=h(),ra=d("p"),Is=i(`Once we have created our comutational graph, we can start calculating the
    gradients and applying the chain rule. This time around we start with the
    last node and go all the way to the nodes we would like to adjust: the
    weigth and the bias.`),Kn=h(),sa=d("p"),js=i(`Remember that we want to calculate the derivatives of MSE with respect to
    the weight and the bias. To achieve that we need to calculate the
    intermediary derivatives and apply the chain rule along the way. First we
    calculate the derivative of MSE with respect to itself. Basically we are
    asking ourselves, how much does the MSE change when we increase the MSE by
    1. As expected the result is just 1.`),Qn=h(),w(Xe.$$.fragment),Zn=h(),la=d("p"),Ds=i(`This might seem to be an unnecessary step, but the chain rule in the
    computational graph relies on multiplying intermediary derivatives by the
    derivative of the above node and if we left the value of 0, the algorithm
    would not have worked.`),er=h(),fe=d("p"),Os=i(`Next we go a level below and calculate the local derivative of the error
    with respect to MSE. The derivative of `),w(Ue.$$.fragment),Bs=i(" is just "),w(Ye.$$.fragment),Ns=i(`, wich is 28. We apply the chain rule and multiply 28 by the derivative
    value of the above node and end up with 28.00.`),tr=h(),w(Ke.$$.fragment),ar=h(),Qe=d("p"),As=i(`We keep going down and calculate and calculate the next intermediary
    derivatives. This time around we face two nodes. The target node and the
    prediction node, where `),w(Ze.$$.fragment),Cs=i(`. The
    intermediary derivative of the error node with respect to the target is just
    1. The intermediary derivative of the error with respect to the prediction
    node is -1. Multiplying the intermediary derivatives with the derivative
    from the above node yields 28 and -28 respectively. These are the
    derivatives of the mean squared error with respect to the target and the
    prediction.`),nr=h(),w(et.$$.fragment),rr=h(),ia=d("p"),Gs=i(`If we continue doing these calculations we will eventually end up with the
    below graph.`),sr=h(),w(tt.$$.fragment),lr=h(),oa=d("p"),Vs=i(`Once we have the gradients, we can apply the gradient descent algorithm. The
    below example iterates between constructing the graph, calculating the
    gradients and applying gradient descent, eventually leading to a mean
    squared error of 0.`),ir=h(),w(at.$$.fragment),or=h(),w(nt.$$.fragment),fr=h(),fa=d("p"),Fs=i(`The best part about the calculations above is that we do not have to track
    the nodes manually or to calculate the gradients ourselves. The connections
    between the nodes happen automatically. When we do any operations on these
    node objects, the computational graph is adjusted automatically to reflect
    the operations. The same goes for the calculation of the gradients, thus the
    name autodiff. Because the local nodes operations are usually very common
    operations like additions, multiplications and other common operations that
    are known to the deep learning community, we know how to calculate the
    intermediary derivatives. This behaviour is implemented in all deep learning
    packages and we do need to think about these things explicitly.`),pr=h(),pa=d("p"),Js=i(`There are several more advantages to utilizing a computation graph. For
    once, this allows us to easily calculate the gradients of arbitrary
    expressions, even if the expressions are extremely complex. As you can
    imagine, neural networks are such expressions, but if you only think in
    terms of local node gradients and the gradients that come from the node
    above, the calculations are rather simple to think about. Secondly, often we
    can reuse many gradient calculations, as these are used as inputs in
    multiple locations. For example the prediction node derivative is moved down
    to the bias and the scaled value and we only need to calculate the above
    derivative once. This is especially important for neural networks, as this
    algorithm allows us to efficiently distribute gradients, thus saving a lot
    of computational time.`),ur=h(),U=d("p"),Rs=i(`At this point we should mention, that in deep learning the graph
    construction phase and the gradient calculation phase have distinct names,
    that you will hear over and over again. The so called `),w(rt.$$.fragment),Hs=i(" is essentially the graph constructon phase. In the "),w(st.$$.fragment),Ls=i(` we start to propagate the gradients from the the mean squared error to the
    weights and bias using automatic differentiation. This algorithm of finding the
    gradients of individual nodes, by utilizing the chain rule and a computational
    graph is also called `),w(lt.$$.fragment),Xs=i(`. Backpropagation
    is the bread and butter of modern deep learning.`),$r=h(),ua=d("div"),hr=h(),$a=d("h2"),Us=i("Multiple Training Samples"),mr=h(),ha=d("p"),Ys=i(`As it turns out making a jump from one to several samples is not that
    complicated. Let's for example assume that we have two samples. This creates
    two branches in our computational graph.`),dr=h(),w(it.$$.fragment),cr=h(),ma=d("p"),Ks=i(`The branch nr. 2 that amounts to 196.0 is essentially the same path that we
    calulated above. For the other branch we would have to do the same
    calculations that we did above, but using the features from the other
    sample. To finish the calculations of the mean squared error, we would have
    to sum up the two branches and calculate the average. When we initiate the
    backward pass, the gradients are propagated into the individual branches.
    You should remember the the same weights and the same bias exist in the
    different branches. That means that those parameters receive different
    gradient signals. In that case the gradients are accumulated through a sum.
    This is the same as saying: "the derivative of a sum is the sum of
    derivatives". You should also observe, that the mean squared error scales
    each of the gradient signals by the number of samples that we use for
    training. If we have two samples, each of the gradients is divided by 2 (or
    multiplied by 0.5).`),wr=h(),da=d("p"),Qs=i(`So the calculations remain very similar: construct the graph and distribute
    the gradients to the weight and bias using automatic differentiation. When
    we use the procedures described above it does not make a huge difference
    whether we use 1, 4 or 100 samples.`),gr=h(),ca=d("p"),Zs=i(`To convince ourselves that automatic differentiation actually works, below
    we present the example from the last section, that we solve by implementing
    a custom autodiff package in JavaScript.`),_r=h(),w(ot.$$.fragment),vr=h(),w(ft.$$.fragment),br=h(),wa=d("div"),yr=h(),ga=d("h2"),el=i("Autograd"),Er=h(),pt=d("p"),tl=i(`As mentioned before, in modern deep learning we do not iterate over
    individual samples to construct a graph, but work with tensors to
    parallelize the computations. When we utilize any of the modern deep
    learning packages and use the provided tensor objects, we get
    parallelisation and automatic differentiation out of the box. We do not need
    to explicitly construct a graph and make sure that all nodes are connected.
    PyTorch for example has a built in automatic differentiation library, called `),w(ut.$$.fragment),al=i(", so let's see how we can utilize the package."),kr=h(),w(Et.$$.fragment),xr=h(),pe=d("p"),nl=i("We start by creating the features "),w($t.$$.fragment),rl=i(` and
    the labels `),w(ht.$$.fragment),sl=i("."),Tr=h(),w(kt.$$.fragment),Sr=h(),mt=d("p"),ll=i(`We transorm the generated numpy arrays into PyTorch Tensors. For the labels
    Tensor we use the `),qa=d("code"),il=i("unsqueeze(dim)"),ol=i(` method. This adds an additional
    dimension, transforming the labels from a (100,) into a (100, 1) dimensional
    Tensor. This makes sure that the predictions that are generated in the the forward
    pass and the actual labels have identical dimensions.`),Wr=h(),w(xt.$$.fragment),qr=h(),Y=d("p"),fl=i("We generate the "),Pa=d("code"),pl=i("init_weights()"),ul=i(` function, which initializes the
    weights and the biases randomly using the standard normal distribution. This
    time around we set the `),Ma=d("code"),$l=i("requires_grad"),hl=i(` property to
    `),za=d("code"),ml=i("True"),dl=i(` in order to track the gradients. We didn't do that for the
    features and the label tensors, as those are fixed and should not be adjusted.`),Pr=h(),w(Tt.$$.fragment),Mr=h(),K=d("p"),cl=i("For the sake of making our explanations easier let us introduce the "),Ia=d("code"),wl=i("print_all()"),gl=i(`
    function. This function makes use of the two imortant properties that each
    Tensor object posesses. The `),ja=d("code"),_l=i("data"),vl=i(" and the "),Da=d("code"),bl=i("grad"),yl=i(` property.
    Those properties are probaly self explanatory: data contains the actual values
    of a tensor, while grad contains the gradiens with respect to each value in the
    data list.`),zr=h(),w(St.$$.fragment),Ir=h(),Wt=d("pre"),El=i(`    Weight: tensor([[-0.6779,  0.4228]]), Grad: None
    Bias: tensor([[0.2107]]), Grad: None
  `),jr=h(),dt=d("p"),kl=i(`When we print the data and the grad right after initializing the tensors,
    the objects posess a randomized value, but gradients amount to `),Oa=d("code"),xl=i("None"),Tl=i("."),Dr=h(),w(qt.$$.fragment),Or=h(),_a=d("p"),Sl=i(`Even when we calculate the mean squared error, by running through the
    forward pass, the gradients remain empty.`),Br=h(),w(Pt.$$.fragment),Nr=h(),Mt=d("pre"),Wl=i(`    Weight: tensor([[-0.6779,  0.4228]]), Grad: None
    Bias: tensor([[0.2107]]), Grad: None
  `),Ar=h(),ct=d("p"),ql=i("To actually run the backward pass, we have to call the "),Ba=d("code"),Pl=i("backward()"),Ml=i(` method on the loss function. The gradients are always based on the tensor that
    initiated the backward pass. So if we run the backward pass on the mean squared
    error tensor, the gradients tell us how we should shift the weights and the bias
    to reduce the loss. This is exactly what we are looking for.`),Cr=h(),w(zt.$$.fragment),Gr=h(),It=d("pre"),zl=i(`    Weight: tensor([[-0.6779,  0.4228]]), Grad: tensor([[-102.5140,  -98.1595]])
    Bias: tensor([[0.2107]]), Grad: tensor([[4.4512]])
  `),Vr=h(),va=d("p"),Il=i(`If we run the forward and the backward passes again, you will notice, that
    the weights and the bias gradients are twice as large. Each time we
    calculate the gradients, the gradients are accumulated. The old gradient
    values are not erased, as one might assume.`),Fr=h(),w(jt.$$.fragment),Jr=h(),Dt=d("pre"),jl=i(`    Weight: tensor([[-0.6779,  0.4228]]), Grad: tensor([[-205.0279, -196.3191]])
    Bias: tensor([[0.2107]]), Grad: tensor([[8.9024]])
  `),Rr=h(),wt=d("p"),Dl=i(`Each time we are done with a gradient descent step, we should clear the
    gradients. We can do that by using the `),Na=d("code"),Ol=i("zero_()"),Bl=i(` method, which zeroes
    out the gradients inplace.`),Hr=h(),w(Ot.$$.fragment),Lr=h(),Bt=d("pre"),Nl=i(`    tensor([[0., 0.]])
  `),Xr=h(),gt=d("p"),Al=i(`Below we show the full implementation of gradiet descent. Most of the
    implementation was already discussed before, but the context manager `),Aa=d("code"),Cl=i("torch.inference_mode()"),Gl=i(` might be new to you. This part tells PyTorch to not include the following parts
    in the computational graph. The actual gradient descent step is not part of the
    forward pass and should therefore not be tracked.`),Ur=h(),w(Nt.$$.fragment),Yr=h(),At=d("pre"),Vl=i(`Mean squared error: 6125.7783203125
Mean squared error: 4322.3662109375
Mean squared error: 3054.9150390625
Mean squared error: 2162.31787109375
Mean squared error: 1532.5537109375
Mean squared error: 1087.494873046875
Mean squared error: 772.5029907226562
Mean squared error: 549.2703857421875
Mean squared error: 390.8788757324219
Mean squared error: 278.37469482421875
  `),Kr=h(),_t=d("p"),Fl=i(`We iterate over the forward pass, the backward pass and the gradient descent
    step for 10 iterations and the mean squared error decreases dramatically.
    This iteration process is called the `),w(vt.$$.fragment),Jl=i(` in
    deep learning lingo. We will encounter those loops over and over again.`),Qr=h(),ba=d("div"),this.h()},l(e){r=c(e,"H2",{});var n=E(r);t=o(n,"Single Training Sample"),n.forEach(a),s=m(e),u=c(e,"P",{});var F=E(u);W=o(F,`Let us remind ourselves that our goal is to minimize the mean squared error,
    by tweaking the weight vector `),g(q.$$.fragment,F),$=o(F,` and the
    bias scalar `),g(T.$$.fragment,F),S=o(F,` using gradient descent.
    `),g(k.$$.fragment,F),z=o(F,`
    To make our journey easier, let us for now assume that we have one single feature
    `),g(x.$$.fragment,F),I=o(F,` and one single training sample. That reduces the mean squared
    error to a much simpler form:
    `),g(J.$$.fragment,F),ue=o(F,`. We will extend our
    calculations once we have covered the basics.`),F.forEach(a),$e=m(e),R=c(e,"P",{});var Ct=E(R);Xt=o(Ct,`The computation of gradients in the deep learning world relies heavily on
    the `),g(Q.$$.fragment,Ct),yt=o(Ct,"."),Ct.forEach(a),Ea=m(e),g(he.$$.fragment,e),ka=m(e),j=c(e,"P",{});var A=E(j);En=o(A,"The mean squared error "),g(me.$$.fragment,A),kn=o(A,` is a great
    example of a composite function. We start by calculating the scaled feature
    `),g(de.$$.fragment,A),xn=o(A,", where "),g(ce.$$.fragment,A),Tn=o(A,`. We use s as an input
    and calculate the linear regression prediction, `),g(M.$$.fragment,A),G=o(A,". Using "),g(B.$$.fragment,A),ke=o(A,` as input, we can calculate the
    error `),g(V.$$.fragment,A),ie=o(A,` as the difference between the prediction and the true
    target `),g(H.$$.fragment,A),we=o(A,", "),g(Z.$$.fragment,A),as=o(A,`.
    Finally we calculate the mean squared error for one single training example, `),g(xe.$$.fragment,A),ns=o(A,"."),A.forEach(a),Sn=m(e),oe=c(e,"P",{});var ge=E(oe);rs=o(ge,`If we utilize the chain rule, we can calculate the derivative of the mean
    squared error with respect to the bias `),g(Te.$$.fragment,ge),ss=o(ge," and the weight "),g(Se.$$.fragment,ge),ls=o(ge," by multiplying the intermediary derivatives."),ge.forEach(a),Wn=m(e),ee=c(e,"DIV",{class:!0});var _e=E(ee);g(We.$$.fragment,_e),is=m(_e),qn=c(_e,"DIV",{}),E(qn).forEach(a),os=m(_e),g(qe.$$.fragment,_e),_e.forEach(a),Pn=m(e),g(Pe.$$.fragment,e),Mn=m(e),Ut=c(e,"P",{});var Ca=E(Ut);fs=o(Ca,`Calculating those intermediary derivatives is relatively straightforward,
    using basic rules of calculus.`),Ca.forEach(a),zn=m(e),N=c(e,"DIV",{class:!0});var C=E(N);g(Me.$$.fragment,C),ps=m(C),xa=c(C,"DIV",{class:!0}),E(xa).forEach(a),us=m(C),g(ze.$$.fragment,C),$s=m(C),Ta=c(C,"DIV",{class:!0}),E(Ta).forEach(a),hs=m(C),g(Ie.$$.fragment,C),ms=m(C),Sa=c(C,"DIV",{class:!0}),E(Sa).forEach(a),ds=m(C),g(je.$$.fragment,C),cs=m(C),Wa=c(C,"DIV",{class:!0}),E(Wa).forEach(a),ws=m(C),g(De.$$.fragment,C),C.forEach(a),In=m(e),Yt=c(e,"P",{});var Ga=E(Yt);gs=o(Ga,`Using the chain rule we can easily calculate the derivatives with respect to
    the weight and bias.`),Ga.forEach(a),jn=m(e),te=c(e,"DIV",{class:!0});var ve=E(te);g(Oe.$$.fragment,ve),_s=m(ve),Dn=c(ve,"DIV",{}),E(Dn).forEach(a),vs=m(ve),g(Be.$$.fragment,ve),ve.forEach(a),On=m(e),Kt=c(e,"P",{});var Va=E(Kt);bs=o(Va,`Once we have the gradients, the gradient descent algorithm works as
    expected.`),Va.forEach(a),Bn=m(e),g(Ne.$$.fragment,e),Nn=m(e),Qt=c(e,"DIV",{class:!0}),E(Qt).forEach(a),An=m(e),Zt=c(e,"H2",{});var Fa=E(Zt);ys=o(Fa,"Computationial Graph"),Fa.forEach(a),Cn=m(e),ea=c(e,"P",{});var Ja=E(ea);Es=o(Ja,`We have mentioned before, that the chain rule plays an integral role in deep
    learning, but what we have covered so far has probably not made it clear why
    it is so essential.`),Ja.forEach(a),Gn=m(e),g(Ae.$$.fragment,e),Vn=m(e),ta=c(e,"P",{});var Ra=E(ta);ks=o(Ra,`So let's construct the computational graph for the mean squared error step
    by step and see how automatic differentiation looks like.`),Ra.forEach(a),Fn=m(e),X=c(e,"P",{});var ae=E(X);xs=o(ae,`A computational graph basically tracks all atomic calculatios and their
    results in a tree like structure. We start out the calculation of MSE by
    creating the weight node `),g(Ce.$$.fragment,ae),Ts=o(ae," and the feature node "),g(Ge.$$.fragment,ae),Ss=o(ae,`, but at that point those two variables are not connected to each other
    yet. Once we calculate `),g(Ve.$$.fragment,ae),Ws=o(ae,`, we connect the two and the result
    is a new node, that lies on a higher level.`),ae.forEach(a),Jn=m(e),g(Fe.$$.fragment,e),Rn=m(e),aa=c(e,"P",{});var Ha=E(aa);qs=o(Ha,`The yellow box represents the operations (like additions and
    multiplications), blue boxes contains the values of the node, while the red
    boxes contain the derivatives. We assume a feature of 5 and the weight of 1
    at the beginning, so the resulting node has the value of 5. At this point in
    time we have not calculated any derivatives yet, so all derivatives will
    amount to 0 for now.`),Ha.forEach(a),Hn=m(e),Je=c(e,"P",{});var Gt=E(Je);Ps=o(Gt,"Next we create the bias "),g(Re.$$.fragment,Gt),Ms=o(Gt,` node with the initial value of 1 and
    add the node to the previously calculated node. Once again those nodes are reflected
    in the computational graph.`),Gt.forEach(a),Ln=m(e),g(He.$$.fragment,e),Xn=m(e),na=c(e,"P",{});var La=E(na);zs=o(La,`We keep creating new nodes, and connect them to previous generated nodes and
    the graph keeps growing. We do that until we reach the final node, in our
    case the mean squared error.`),La.forEach(a),Un=m(e),g(Le.$$.fragment,e),Yn=m(e),ra=c(e,"P",{});var Xa=E(ra);Is=o(Xa,`Once we have created our comutational graph, we can start calculating the
    gradients and applying the chain rule. This time around we start with the
    last node and go all the way to the nodes we would like to adjust: the
    weigth and the bias.`),Xa.forEach(a),Kn=m(e),sa=c(e,"P",{});var Ua=E(sa);js=o(Ua,`Remember that we want to calculate the derivatives of MSE with respect to
    the weight and the bias. To achieve that we need to calculate the
    intermediary derivatives and apply the chain rule along the way. First we
    calculate the derivative of MSE with respect to itself. Basically we are
    asking ourselves, how much does the MSE change when we increase the MSE by
    1. As expected the result is just 1.`),Ua.forEach(a),Qn=m(e),g(Xe.$$.fragment,e),Zn=m(e),la=c(e,"P",{});var Ya=E(la);Ds=o(Ya,`This might seem to be an unnecessary step, but the chain rule in the
    computational graph relies on multiplying intermediary derivatives by the
    derivative of the above node and if we left the value of 0, the algorithm
    would not have worked.`),Ya.forEach(a),er=m(e),fe=c(e,"P",{});var be=E(fe);Os=o(be,`Next we go a level below and calculate the local derivative of the error
    with respect to MSE. The derivative of `),g(Ue.$$.fragment,be),Bs=o(be," is just "),g(Ye.$$.fragment,be),Ns=o(be,`, wich is 28. We apply the chain rule and multiply 28 by the derivative
    value of the above node and end up with 28.00.`),be.forEach(a),tr=m(e),g(Ke.$$.fragment,e),ar=m(e),Qe=c(e,"P",{});var Vt=E(Qe);As=o(Vt,`We keep going down and calculate and calculate the next intermediary
    derivatives. This time around we face two nodes. The target node and the
    prediction node, where `),g(Ze.$$.fragment,Vt),Cs=o(Vt,`. The
    intermediary derivative of the error node with respect to the target is just
    1. The intermediary derivative of the error with respect to the prediction
    node is -1. Multiplying the intermediary derivatives with the derivative
    from the above node yields 28 and -28 respectively. These are the
    derivatives of the mean squared error with respect to the target and the
    prediction.`),Vt.forEach(a),nr=m(e),g(et.$$.fragment,e),rr=m(e),ia=c(e,"P",{});var Ka=E(ia);Gs=o(Ka,`If we continue doing these calculations we will eventually end up with the
    below graph.`),Ka.forEach(a),sr=m(e),g(tt.$$.fragment,e),lr=m(e),oa=c(e,"P",{});var Qa=E(oa);Vs=o(Qa,`Once we have the gradients, we can apply the gradient descent algorithm. The
    below example iterates between constructing the graph, calculating the
    gradients and applying gradient descent, eventually leading to a mean
    squared error of 0.`),Qa.forEach(a),ir=m(e),g(at.$$.fragment,e),or=m(e),g(nt.$$.fragment,e),fr=m(e),fa=c(e,"P",{});var Za=E(fa);Fs=o(Za,`The best part about the calculations above is that we do not have to track
    the nodes manually or to calculate the gradients ourselves. The connections
    between the nodes happen automatically. When we do any operations on these
    node objects, the computational graph is adjusted automatically to reflect
    the operations. The same goes for the calculation of the gradients, thus the
    name autodiff. Because the local nodes operations are usually very common
    operations like additions, multiplications and other common operations that
    are known to the deep learning community, we know how to calculate the
    intermediary derivatives. This behaviour is implemented in all deep learning
    packages and we do need to think about these things explicitly.`),Za.forEach(a),pr=m(e),pa=c(e,"P",{});var en=E(pa);Js=o(en,`There are several more advantages to utilizing a computation graph. For
    once, this allows us to easily calculate the gradients of arbitrary
    expressions, even if the expressions are extremely complex. As you can
    imagine, neural networks are such expressions, but if you only think in
    terms of local node gradients and the gradients that come from the node
    above, the calculations are rather simple to think about. Secondly, often we
    can reuse many gradient calculations, as these are used as inputs in
    multiple locations. For example the prediction node derivative is moved down
    to the bias and the scaled value and we only need to calculate the above
    derivative once. This is especially important for neural networks, as this
    algorithm allows us to efficiently distribute gradients, thus saving a lot
    of computational time.`),en.forEach(a),ur=m(e),U=c(e,"P",{});var ne=E(U);Rs=o(ne,`At this point we should mention, that in deep learning the graph
    construction phase and the gradient calculation phase have distinct names,
    that you will hear over and over again. The so called `),g(rt.$$.fragment,ne),Hs=o(ne," is essentially the graph constructon phase. In the "),g(st.$$.fragment,ne),Ls=o(ne,` we start to propagate the gradients from the the mean squared error to the
    weights and bias using automatic differentiation. This algorithm of finding the
    gradients of individual nodes, by utilizing the chain rule and a computational
    graph is also called `),g(lt.$$.fragment,ne),Xs=o(ne,`. Backpropagation
    is the bread and butter of modern deep learning.`),ne.forEach(a),$r=m(e),ua=c(e,"DIV",{class:!0}),E(ua).forEach(a),hr=m(e),$a=c(e,"H2",{});var tn=E($a);Us=o(tn,"Multiple Training Samples"),tn.forEach(a),mr=m(e),ha=c(e,"P",{});var an=E(ha);Ys=o(an,`As it turns out making a jump from one to several samples is not that
    complicated. Let's for example assume that we have two samples. This creates
    two branches in our computational graph.`),an.forEach(a),dr=m(e),g(it.$$.fragment,e),cr=m(e),ma=c(e,"P",{});var nn=E(ma);Ks=o(nn,`The branch nr. 2 that amounts to 196.0 is essentially the same path that we
    calulated above. For the other branch we would have to do the same
    calculations that we did above, but using the features from the other
    sample. To finish the calculations of the mean squared error, we would have
    to sum up the two branches and calculate the average. When we initiate the
    backward pass, the gradients are propagated into the individual branches.
    You should remember the the same weights and the same bias exist in the
    different branches. That means that those parameters receive different
    gradient signals. In that case the gradients are accumulated through a sum.
    This is the same as saying: "the derivative of a sum is the sum of
    derivatives". You should also observe, that the mean squared error scales
    each of the gradient signals by the number of samples that we use for
    training. If we have two samples, each of the gradients is divided by 2 (or
    multiplied by 0.5).`),nn.forEach(a),wr=m(e),da=c(e,"P",{});var rn=E(da);Qs=o(rn,`So the calculations remain very similar: construct the graph and distribute
    the gradients to the weight and bias using automatic differentiation. When
    we use the procedures described above it does not make a huge difference
    whether we use 1, 4 or 100 samples.`),rn.forEach(a),gr=m(e),ca=c(e,"P",{});var sn=E(ca);Zs=o(sn,`To convince ourselves that automatic differentiation actually works, below
    we present the example from the last section, that we solve by implementing
    a custom autodiff package in JavaScript.`),sn.forEach(a),_r=m(e),g(ot.$$.fragment,e),vr=m(e),g(ft.$$.fragment,e),br=m(e),wa=c(e,"DIV",{class:!0}),E(wa).forEach(a),yr=m(e),ga=c(e,"H2",{});var ln=E(ga);el=o(ln,"Autograd"),ln.forEach(a),Er=m(e),pt=c(e,"P",{});var Ft=E(pt);tl=o(Ft,`As mentioned before, in modern deep learning we do not iterate over
    individual samples to construct a graph, but work with tensors to
    parallelize the computations. When we utilize any of the modern deep
    learning packages and use the provided tensor objects, we get
    parallelisation and automatic differentiation out of the box. We do not need
    to explicitly construct a graph and make sure that all nodes are connected.
    PyTorch for example has a built in automatic differentiation library, called `),g(ut.$$.fragment,Ft),al=o(Ft,", so let's see how we can utilize the package."),Ft.forEach(a),kr=m(e),g(Et.$$.fragment,e),xr=m(e),pe=c(e,"P",{});var ye=E(pe);nl=o(ye,"We start by creating the features "),g($t.$$.fragment,ye),rl=o(ye,` and
    the labels `),g(ht.$$.fragment,ye),sl=o(ye,"."),ye.forEach(a),Tr=m(e),g(kt.$$.fragment,e),Sr=m(e),mt=c(e,"P",{});var Jt=E(mt);ll=o(Jt,`We transorm the generated numpy arrays into PyTorch Tensors. For the labels
    Tensor we use the `),qa=c(Jt,"CODE",{});var on=E(qa);il=o(on,"unsqueeze(dim)"),on.forEach(a),ol=o(Jt,` method. This adds an additional
    dimension, transforming the labels from a (100,) into a (100, 1) dimensional
    Tensor. This makes sure that the predictions that are generated in the the forward
    pass and the actual labels have identical dimensions.`),Jt.forEach(a),Wr=m(e),g(xt.$$.fragment,e),qr=m(e),Y=c(e,"P",{});var re=E(Y);fl=o(re,"We generate the "),Pa=c(re,"CODE",{});var fn=E(Pa);pl=o(fn,"init_weights()"),fn.forEach(a),ul=o(re,` function, which initializes the
    weights and the biases randomly using the standard normal distribution. This
    time around we set the `),Ma=c(re,"CODE",{});var pn=E(Ma);$l=o(pn,"requires_grad"),pn.forEach(a),hl=o(re,` property to
    `),za=c(re,"CODE",{});var un=E(za);ml=o(un,"True"),un.forEach(a),dl=o(re,` in order to track the gradients. We didn't do that for the
    features and the label tensors, as those are fixed and should not be adjusted.`),re.forEach(a),Pr=m(e),g(Tt.$$.fragment,e),Mr=m(e),K=c(e,"P",{});var se=E(K);cl=o(se,"For the sake of making our explanations easier let us introduce the "),Ia=c(se,"CODE",{});var $n=E(Ia);wl=o($n,"print_all()"),$n.forEach(a),gl=o(se,`
    function. This function makes use of the two imortant properties that each
    Tensor object posesses. The `),ja=c(se,"CODE",{});var hn=E(ja);_l=o(hn,"data"),hn.forEach(a),vl=o(se," and the "),Da=c(se,"CODE",{});var mn=E(Da);bl=o(mn,"grad"),mn.forEach(a),yl=o(se,` property.
    Those properties are probaly self explanatory: data contains the actual values
    of a tensor, while grad contains the gradiens with respect to each value in the
    data list.`),se.forEach(a),zr=m(e),g(St.$$.fragment,e),Ir=m(e),Wt=c(e,"PRE",{class:!0});var dn=E(Wt);El=o(dn,`    Weight: tensor([[-0.6779,  0.4228]]), Grad: None
    Bias: tensor([[0.2107]]), Grad: None
  `),dn.forEach(a),jr=m(e),dt=c(e,"P",{});var Rt=E(dt);kl=o(Rt,`When we print the data and the grad right after initializing the tensors,
    the objects posess a randomized value, but gradients amount to `),Oa=c(Rt,"CODE",{});var cn=E(Oa);xl=o(cn,"None"),cn.forEach(a),Tl=o(Rt,"."),Rt.forEach(a),Dr=m(e),g(qt.$$.fragment,e),Or=m(e),_a=c(e,"P",{});var wn=E(_a);Sl=o(wn,`Even when we calculate the mean squared error, by running through the
    forward pass, the gradients remain empty.`),wn.forEach(a),Br=m(e),g(Pt.$$.fragment,e),Nr=m(e),Mt=c(e,"PRE",{class:!0});var gn=E(Mt);Wl=o(gn,`    Weight: tensor([[-0.6779,  0.4228]]), Grad: None
    Bias: tensor([[0.2107]]), Grad: None
  `),gn.forEach(a),Ar=m(e),ct=c(e,"P",{});var Ht=E(ct);ql=o(Ht,"To actually run the backward pass, we have to call the "),Ba=c(Ht,"CODE",{});var ya=E(Ba);Pl=o(ya,"backward()"),ya.forEach(a),Ml=o(Ht,` method on the loss function. The gradients are always based on the tensor that
    initiated the backward pass. So if we run the backward pass on the mean squared
    error tensor, the gradients tell us how we should shift the weights and the bias
    to reduce the loss. This is exactly what we are looking for.`),Ht.forEach(a),Cr=m(e),g(zt.$$.fragment,e),Gr=m(e),It=c(e,"PRE",{class:!0});var _n=E(It);zl=o(_n,`    Weight: tensor([[-0.6779,  0.4228]]), Grad: tensor([[-102.5140,  -98.1595]])
    Bias: tensor([[0.2107]]), Grad: tensor([[4.4512]])
  `),_n.forEach(a),Vr=m(e),va=c(e,"P",{});var vn=E(va);Il=o(vn,`If we run the forward and the backward passes again, you will notice, that
    the weights and the bias gradients are twice as large. Each time we
    calculate the gradients, the gradients are accumulated. The old gradient
    values are not erased, as one might assume.`),vn.forEach(a),Fr=m(e),g(jt.$$.fragment,e),Jr=m(e),Dt=c(e,"PRE",{class:!0});var bn=E(Dt);jl=o(bn,`    Weight: tensor([[-0.6779,  0.4228]]), Grad: tensor([[-205.0279, -196.3191]])
    Bias: tensor([[0.2107]]), Grad: tensor([[8.9024]])
  `),bn.forEach(a),Rr=m(e),wt=c(e,"P",{});var Lt=E(wt);Dl=o(Lt,`Each time we are done with a gradient descent step, we should clear the
    gradients. We can do that by using the `),Na=c(Lt,"CODE",{});var Rl=E(Na);Ol=o(Rl,"zero_()"),Rl.forEach(a),Bl=o(Lt,` method, which zeroes
    out the gradients inplace.`),Lt.forEach(a),Hr=m(e),g(Ot.$$.fragment,e),Lr=m(e),Bt=c(e,"PRE",{class:!0});var Hl=E(Bt);Nl=o(Hl,`    tensor([[0., 0.]])
  `),Hl.forEach(a),Xr=m(e),gt=c(e,"P",{});var es=E(gt);Al=o(es,`Below we show the full implementation of gradiet descent. Most of the
    implementation was already discussed before, but the context manager `),Aa=c(es,"CODE",{});var Ll=E(Aa);Cl=o(Ll,"torch.inference_mode()"),Ll.forEach(a),Gl=o(es,` might be new to you. This part tells PyTorch to not include the following parts
    in the computational graph. The actual gradient descent step is not part of the
    forward pass and should therefore not be tracked.`),es.forEach(a),Ur=m(e),g(Nt.$$.fragment,e),Yr=m(e),At=c(e,"PRE",{class:!0});var Xl=E(At);Vl=o(Xl,`Mean squared error: 6125.7783203125
Mean squared error: 4322.3662109375
Mean squared error: 3054.9150390625
Mean squared error: 2162.31787109375
Mean squared error: 1532.5537109375
Mean squared error: 1087.494873046875
Mean squared error: 772.5029907226562
Mean squared error: 549.2703857421875
Mean squared error: 390.8788757324219
Mean squared error: 278.37469482421875
  `),Xl.forEach(a),Kr=m(e),_t=c(e,"P",{});var ts=E(_t);Fl=o(ts,`We iterate over the forward pass, the backward pass and the gradient descent
    step for 10 iterations and the mean squared error decreases dramatically.
    This iteration process is called the `),g(vt.$$.fragment,ts),Jl=o(ts,` in
    deep learning lingo. We will encounter those loops over and over again.`),ts.forEach(a),Qr=m(e),ba=c(e,"DIV",{class:!0}),E(ba).forEach(a),this.h()},h(){D(ee,"class","flex justify-center items-center flex-col"),D(xa,"class","mb-1"),D(Ta,"class","mb-1"),D(Sa,"class","mb-1"),D(Wa,"class","mb-1"),D(N,"class","flex justify-center items-center flex-col"),D(te,"class","flex justify-center items-center flex-col"),D(Qt,"class","separator"),D(ua,"class","separator"),D(wa,"class","separator"),D(Wt,"class","text-sm"),D(Mt,"class","text-sm"),D(It,"class","text-sm"),D(Dt,"class","text-sm"),D(Bt,"class","text-sm"),D(At,"class","flex justify-center text-sm"),D(ba,"class","separator")},m(e,n){l(e,r,n),p(r,t),l(e,s,n),l(e,u,n),p(u,W),_(q,u,null),p(u,$),_(T,u,null),p(u,S),_(k,u,null),p(u,z),_(x,u,null),p(u,I),_(J,u,null),p(u,ue),l(e,$e,n),l(e,R,n),p(R,Xt),_(Q,R,null),p(R,yt),l(e,Ea,n),_(he,e,n),l(e,ka,n),l(e,j,n),p(j,En),_(me,j,null),p(j,kn),_(de,j,null),p(j,xn),_(ce,j,null),p(j,Tn),_(M,j,null),p(j,G),_(B,j,null),p(j,ke),_(V,j,null),p(j,ie),_(H,j,null),p(j,we),_(Z,j,null),p(j,as),_(xe,j,null),p(j,ns),l(e,Sn,n),l(e,oe,n),p(oe,rs),_(Te,oe,null),p(oe,ss),_(Se,oe,null),p(oe,ls),l(e,Wn,n),l(e,ee,n),_(We,ee,null),p(ee,is),p(ee,qn),p(ee,os),_(qe,ee,null),l(e,Pn,n),_(Pe,e,n),l(e,Mn,n),l(e,Ut,n),p(Ut,fs),l(e,zn,n),l(e,N,n),_(Me,N,null),p(N,ps),p(N,xa),p(N,us),_(ze,N,null),p(N,$s),p(N,Ta),p(N,hs),_(Ie,N,null),p(N,ms),p(N,Sa),p(N,ds),_(je,N,null),p(N,cs),p(N,Wa),p(N,ws),_(De,N,null),l(e,In,n),l(e,Yt,n),p(Yt,gs),l(e,jn,n),l(e,te,n),_(Oe,te,null),p(te,_s),p(te,Dn),p(te,vs),_(Be,te,null),l(e,On,n),l(e,Kt,n),p(Kt,bs),l(e,Bn,n),_(Ne,e,n),l(e,Nn,n),l(e,Qt,n),l(e,An,n),l(e,Zt,n),p(Zt,ys),l(e,Cn,n),l(e,ea,n),p(ea,Es),l(e,Gn,n),_(Ae,e,n),l(e,Vn,n),l(e,ta,n),p(ta,ks),l(e,Fn,n),l(e,X,n),p(X,xs),_(Ce,X,null),p(X,Ts),_(Ge,X,null),p(X,Ss),_(Ve,X,null),p(X,Ws),l(e,Jn,n),_(Fe,e,n),l(e,Rn,n),l(e,aa,n),p(aa,qs),l(e,Hn,n),l(e,Je,n),p(Je,Ps),_(Re,Je,null),p(Je,Ms),l(e,Ln,n),_(He,e,n),l(e,Xn,n),l(e,na,n),p(na,zs),l(e,Un,n),_(Le,e,n),l(e,Yn,n),l(e,ra,n),p(ra,Is),l(e,Kn,n),l(e,sa,n),p(sa,js),l(e,Qn,n),_(Xe,e,n),l(e,Zn,n),l(e,la,n),p(la,Ds),l(e,er,n),l(e,fe,n),p(fe,Os),_(Ue,fe,null),p(fe,Bs),_(Ye,fe,null),p(fe,Ns),l(e,tr,n),_(Ke,e,n),l(e,ar,n),l(e,Qe,n),p(Qe,As),_(Ze,Qe,null),p(Qe,Cs),l(e,nr,n),_(et,e,n),l(e,rr,n),l(e,ia,n),p(ia,Gs),l(e,sr,n),_(tt,e,n),l(e,lr,n),l(e,oa,n),p(oa,Vs),l(e,ir,n),_(at,e,n),l(e,or,n),_(nt,e,n),l(e,fr,n),l(e,fa,n),p(fa,Fs),l(e,pr,n),l(e,pa,n),p(pa,Js),l(e,ur,n),l(e,U,n),p(U,Rs),_(rt,U,null),p(U,Hs),_(st,U,null),p(U,Ls),_(lt,U,null),p(U,Xs),l(e,$r,n),l(e,ua,n),l(e,hr,n),l(e,$a,n),p($a,Us),l(e,mr,n),l(e,ha,n),p(ha,Ys),l(e,dr,n),_(it,e,n),l(e,cr,n),l(e,ma,n),p(ma,Ks),l(e,wr,n),l(e,da,n),p(da,Qs),l(e,gr,n),l(e,ca,n),p(ca,Zs),l(e,_r,n),_(ot,e,n),l(e,vr,n),_(ft,e,n),l(e,br,n),l(e,wa,n),l(e,yr,n),l(e,ga,n),p(ga,el),l(e,Er,n),l(e,pt,n),p(pt,tl),_(ut,pt,null),p(pt,al),l(e,kr,n),_(Et,e,n),l(e,xr,n),l(e,pe,n),p(pe,nl),_($t,pe,null),p(pe,rl),_(ht,pe,null),p(pe,sl),l(e,Tr,n),_(kt,e,n),l(e,Sr,n),l(e,mt,n),p(mt,ll),p(mt,qa),p(qa,il),p(mt,ol),l(e,Wr,n),_(xt,e,n),l(e,qr,n),l(e,Y,n),p(Y,fl),p(Y,Pa),p(Pa,pl),p(Y,ul),p(Y,Ma),p(Ma,$l),p(Y,hl),p(Y,za),p(za,ml),p(Y,dl),l(e,Pr,n),_(Tt,e,n),l(e,Mr,n),l(e,K,n),p(K,cl),p(K,Ia),p(Ia,wl),p(K,gl),p(K,ja),p(ja,_l),p(K,vl),p(K,Da),p(Da,bl),p(K,yl),l(e,zr,n),_(St,e,n),l(e,Ir,n),l(e,Wt,n),p(Wt,El),l(e,jr,n),l(e,dt,n),p(dt,kl),p(dt,Oa),p(Oa,xl),p(dt,Tl),l(e,Dr,n),_(qt,e,n),l(e,Or,n),l(e,_a,n),p(_a,Sl),l(e,Br,n),_(Pt,e,n),l(e,Nr,n),l(e,Mt,n),p(Mt,Wl),l(e,Ar,n),l(e,ct,n),p(ct,ql),p(ct,Ba),p(Ba,Pl),p(ct,Ml),l(e,Cr,n),_(zt,e,n),l(e,Gr,n),l(e,It,n),p(It,zl),l(e,Vr,n),l(e,va,n),p(va,Il),l(e,Fr,n),_(jt,e,n),l(e,Jr,n),l(e,Dt,n),p(Dt,jl),l(e,Rr,n),l(e,wt,n),p(wt,Dl),p(wt,Na),p(Na,Ol),p(wt,Bl),l(e,Hr,n),_(Ot,e,n),l(e,Lr,n),l(e,Bt,n),p(Bt,Nl),l(e,Xr,n),l(e,gt,n),p(gt,Al),p(gt,Aa),p(Aa,Cl),p(gt,Gl),l(e,Ur,n),_(Nt,e,n),l(e,Yr,n),l(e,At,n),p(At,Vl),l(e,Kr,n),l(e,_t,n),p(_t,Fl),_(vt,_t,null),p(_t,Jl),l(e,Qr,n),l(e,ba,n),Zr=!0},p(e,n){const F={};n&536870912&&(F.$$scope={dirty:n,ctx:e}),q.$set(F);const Ct={};n&536870912&&(Ct.$$scope={dirty:n,ctx:e}),T.$set(Ct);const A={};n&536870912&&(A.$$scope={dirty:n,ctx:e}),k.$set(A);const ge={};n&536870912&&(ge.$$scope={dirty:n,ctx:e}),x.$set(ge);const _e={};n&536870912&&(_e.$$scope={dirty:n,ctx:e}),J.$set(_e);const Ca={};n&536870912&&(Ca.$$scope={dirty:n,ctx:e}),Q.$set(Ca);const C={};n&536870912&&(C.$$scope={dirty:n,ctx:e}),he.$set(C);const Ga={};n&536870912&&(Ga.$$scope={dirty:n,ctx:e}),me.$set(Ga);const ve={};n&536870912&&(ve.$$scope={dirty:n,ctx:e}),de.$set(ve);const Va={};n&536870912&&(Va.$$scope={dirty:n,ctx:e}),ce.$set(Va);const Fa={};n&536870912&&(Fa.$$scope={dirty:n,ctx:e}),M.$set(Fa);const Ja={};n&536870912&&(Ja.$$scope={dirty:n,ctx:e}),B.$set(Ja);const Ra={};n&536870912&&(Ra.$$scope={dirty:n,ctx:e}),V.$set(Ra);const ae={};n&536870912&&(ae.$$scope={dirty:n,ctx:e}),H.$set(ae);const Ha={};n&536870912&&(Ha.$$scope={dirty:n,ctx:e}),Z.$set(Ha);const Gt={};n&536870912&&(Gt.$$scope={dirty:n,ctx:e}),xe.$set(Gt);const La={};n&536870912&&(La.$$scope={dirty:n,ctx:e}),Te.$set(La);const Xa={};n&536870912&&(Xa.$$scope={dirty:n,ctx:e}),Se.$set(Xa);const Ua={};n&536870912&&(Ua.$$scope={dirty:n,ctx:e}),We.$set(Ua);const Ya={};n&536870912&&(Ya.$$scope={dirty:n,ctx:e}),qe.$set(Ya);const be={};n&536870912&&(be.$$scope={dirty:n,ctx:e}),Pe.$set(be);const Vt={};n&536870912&&(Vt.$$scope={dirty:n,ctx:e}),Me.$set(Vt);const Ka={};n&536870912&&(Ka.$$scope={dirty:n,ctx:e}),ze.$set(Ka);const Qa={};n&536870912&&(Qa.$$scope={dirty:n,ctx:e}),Ie.$set(Qa);const Za={};n&536870912&&(Za.$$scope={dirty:n,ctx:e}),je.$set(Za);const en={};n&536870912&&(en.$$scope={dirty:n,ctx:e}),De.$set(en);const ne={};n&536870912&&(ne.$$scope={dirty:n,ctx:e}),Oe.$set(ne);const tn={};n&536870912&&(tn.$$scope={dirty:n,ctx:e}),Be.$set(tn);const an={};n&536870912&&(an.$$scope={dirty:n,ctx:e}),Ne.$set(an);const nn={};n&536870912&&(nn.$$scope={dirty:n,ctx:e}),Ae.$set(nn);const rn={};n&536870912&&(rn.$$scope={dirty:n,ctx:e}),Ce.$set(rn);const sn={};n&536870912&&(sn.$$scope={dirty:n,ctx:e}),Ge.$set(sn);const ln={};n&536870912&&(ln.$$scope={dirty:n,ctx:e}),Ve.$set(ln);const Ft={};n&8&&(Ft.graph=e[3]),Fe.$set(Ft);const ye={};n&536870912&&(ye.$$scope={dirty:n,ctx:e}),Re.$set(ye);const Jt={};n&16&&(Jt.graph=e[4]),He.$set(Jt);const on={};n&32&&(on.graph=e[5]),Le.$set(on);const re={};n&64&&(re.graph=e[6]),Xe.$set(re);const fn={};n&536870912&&(fn.$$scope={dirty:n,ctx:e}),Ue.$set(fn);const pn={};n&536870912&&(pn.$$scope={dirty:n,ctx:e}),Ye.$set(pn);const un={};n&64&&(un.graph=e[6]),Ke.$set(un);const se={};n&536870912&&(se.$$scope={dirty:n,ctx:e}),Ze.$set(se);const $n={};n&64&&($n.graph=e[6]),et.$set($n);const hn={};n&64&&(hn.graph=e[6]),tt.$set(hn);const mn={};n&536870912&&(mn.$$scope={dirty:n,ctx:e}),at.$set(mn);const dn={};n&4&&(dn.graph=e[2]),nt.$set(dn);const Rt={};n&536870912&&(Rt.$$scope={dirty:n,ctx:e}),rt.$set(Rt);const cn={};n&536870912&&(cn.$$scope={dirty:n,ctx:e}),st.$set(cn);const wn={};n&536870912&&(wn.$$scope={dirty:n,ctx:e}),lt.$set(wn);const gn={};n&128&&(gn.graph=e[7]),it.$set(gn);const Ht={};n&536870912&&(Ht.$$scope={dirty:n,ctx:e}),ot.$set(Ht);const ya={};n&1&&(ya.w=e[0].data),n&2&&(ya.b=e[1].data),ft.$set(ya);const _n={};n&536870912&&(_n.$$scope={dirty:n,ctx:e}),ut.$set(_n);const vn={};n&536870912&&(vn.$$scope={dirty:n,ctx:e}),$t.$set(vn);const bn={};n&536870912&&(bn.$$scope={dirty:n,ctx:e}),ht.$set(bn);const Lt={};n&536870912&&(Lt.$$scope={dirty:n,ctx:e}),vt.$set(Lt)},i(e){Zr||(v(q.$$.fragment,e),v(T.$$.fragment,e),v(k.$$.fragment,e),v(x.$$.fragment,e),v(J.$$.fragment,e),v(Q.$$.fragment,e),v(he.$$.fragment,e),v(me.$$.fragment,e),v(de.$$.fragment,e),v(ce.$$.fragment,e),v(M.$$.fragment,e),v(B.$$.fragment,e),v(V.$$.fragment,e),v(H.$$.fragment,e),v(Z.$$.fragment,e),v(xe.$$.fragment,e),v(Te.$$.fragment,e),v(Se.$$.fragment,e),v(We.$$.fragment,e),v(qe.$$.fragment,e),v(Pe.$$.fragment,e),v(Me.$$.fragment,e),v(ze.$$.fragment,e),v(Ie.$$.fragment,e),v(je.$$.fragment,e),v(De.$$.fragment,e),v(Oe.$$.fragment,e),v(Be.$$.fragment,e),v(Ne.$$.fragment,e),v(Ae.$$.fragment,e),v(Ce.$$.fragment,e),v(Ge.$$.fragment,e),v(Ve.$$.fragment,e),v(Fe.$$.fragment,e),v(Re.$$.fragment,e),v(He.$$.fragment,e),v(Le.$$.fragment,e),v(Xe.$$.fragment,e),v(Ue.$$.fragment,e),v(Ye.$$.fragment,e),v(Ke.$$.fragment,e),v(Ze.$$.fragment,e),v(et.$$.fragment,e),v(tt.$$.fragment,e),v(at.$$.fragment,e),v(nt.$$.fragment,e),v(rt.$$.fragment,e),v(st.$$.fragment,e),v(lt.$$.fragment,e),v(it.$$.fragment,e),v(ot.$$.fragment,e),v(ft.$$.fragment,e),v(ut.$$.fragment,e),v(Et.$$.fragment,e),v($t.$$.fragment,e),v(ht.$$.fragment,e),v(kt.$$.fragment,e),v(xt.$$.fragment,e),v(Tt.$$.fragment,e),v(St.$$.fragment,e),v(qt.$$.fragment,e),v(Pt.$$.fragment,e),v(zt.$$.fragment,e),v(jt.$$.fragment,e),v(Ot.$$.fragment,e),v(Nt.$$.fragment,e),v(vt.$$.fragment,e),Zr=!0)},o(e){b(q.$$.fragment,e),b(T.$$.fragment,e),b(k.$$.fragment,e),b(x.$$.fragment,e),b(J.$$.fragment,e),b(Q.$$.fragment,e),b(he.$$.fragment,e),b(me.$$.fragment,e),b(de.$$.fragment,e),b(ce.$$.fragment,e),b(M.$$.fragment,e),b(B.$$.fragment,e),b(V.$$.fragment,e),b(H.$$.fragment,e),b(Z.$$.fragment,e),b(xe.$$.fragment,e),b(Te.$$.fragment,e),b(Se.$$.fragment,e),b(We.$$.fragment,e),b(qe.$$.fragment,e),b(Pe.$$.fragment,e),b(Me.$$.fragment,e),b(ze.$$.fragment,e),b(Ie.$$.fragment,e),b(je.$$.fragment,e),b(De.$$.fragment,e),b(Oe.$$.fragment,e),b(Be.$$.fragment,e),b(Ne.$$.fragment,e),b(Ae.$$.fragment,e),b(Ce.$$.fragment,e),b(Ge.$$.fragment,e),b(Ve.$$.fragment,e),b(Fe.$$.fragment,e),b(Re.$$.fragment,e),b(He.$$.fragment,e),b(Le.$$.fragment,e),b(Xe.$$.fragment,e),b(Ue.$$.fragment,e),b(Ye.$$.fragment,e),b(Ke.$$.fragment,e),b(Ze.$$.fragment,e),b(et.$$.fragment,e),b(tt.$$.fragment,e),b(at.$$.fragment,e),b(nt.$$.fragment,e),b(rt.$$.fragment,e),b(st.$$.fragment,e),b(lt.$$.fragment,e),b(it.$$.fragment,e),b(ot.$$.fragment,e),b(ft.$$.fragment,e),b(ut.$$.fragment,e),b(Et.$$.fragment,e),b($t.$$.fragment,e),b(ht.$$.fragment,e),b(kt.$$.fragment,e),b(xt.$$.fragment,e),b(Tt.$$.fragment,e),b(St.$$.fragment,e),b(qt.$$.fragment,e),b(Pt.$$.fragment,e),b(zt.$$.fragment,e),b(jt.$$.fragment,e),b(Ot.$$.fragment,e),b(Nt.$$.fragment,e),b(vt.$$.fragment,e),Zr=!1},d(e){e&&a(r),e&&a(s),e&&a(u),y(q),y(T),y(k),y(x),y(J),e&&a($e),e&&a(R),y(Q),e&&a(Ea),y(he,e),e&&a(ka),e&&a(j),y(me),y(de),y(ce),y(M),y(B),y(V),y(H),y(Z),y(xe),e&&a(Sn),e&&a(oe),y(Te),y(Se),e&&a(Wn),e&&a(ee),y(We),y(qe),e&&a(Pn),y(Pe,e),e&&a(Mn),e&&a(Ut),e&&a(zn),e&&a(N),y(Me),y(ze),y(Ie),y(je),y(De),e&&a(In),e&&a(Yt),e&&a(jn),e&&a(te),y(Oe),y(Be),e&&a(On),e&&a(Kt),e&&a(Bn),y(Ne,e),e&&a(Nn),e&&a(Qt),e&&a(An),e&&a(Zt),e&&a(Cn),e&&a(ea),e&&a(Gn),y(Ae,e),e&&a(Vn),e&&a(ta),e&&a(Fn),e&&a(X),y(Ce),y(Ge),y(Ve),e&&a(Jn),y(Fe,e),e&&a(Rn),e&&a(aa),e&&a(Hn),e&&a(Je),y(Re),e&&a(Ln),y(He,e),e&&a(Xn),e&&a(na),e&&a(Un),y(Le,e),e&&a(Yn),e&&a(ra),e&&a(Kn),e&&a(sa),e&&a(Qn),y(Xe,e),e&&a(Zn),e&&a(la),e&&a(er),e&&a(fe),y(Ue),y(Ye),e&&a(tr),y(Ke,e),e&&a(ar),e&&a(Qe),y(Ze),e&&a(nr),y(et,e),e&&a(rr),e&&a(ia),e&&a(sr),y(tt,e),e&&a(lr),e&&a(oa),e&&a(ir),y(at,e),e&&a(or),y(nt,e),e&&a(fr),e&&a(fa),e&&a(pr),e&&a(pa),e&&a(ur),e&&a(U),y(rt),y(st),y(lt),e&&a($r),e&&a(ua),e&&a(hr),e&&a($a),e&&a(mr),e&&a(ha),e&&a(dr),y(it,e),e&&a(cr),e&&a(ma),e&&a(wr),e&&a(da),e&&a(gr),e&&a(ca),e&&a(_r),y(ot,e),e&&a(vr),y(ft,e),e&&a(br),e&&a(wa),e&&a(yr),e&&a(ga),e&&a(Er),e&&a(pt),y(ut),e&&a(kr),y(Et,e),e&&a(xr),e&&a(pe),y($t),y(ht),e&&a(Tr),y(kt,e),e&&a(Sr),e&&a(mt),e&&a(Wr),y(xt,e),e&&a(qr),e&&a(Y),e&&a(Pr),y(Tt,e),e&&a(Mr),e&&a(K),e&&a(zr),y(St,e),e&&a(Ir),e&&a(Wt),e&&a(jr),e&&a(dt),e&&a(Dr),y(qt,e),e&&a(Or),e&&a(_a),e&&a(Br),y(Pt,e),e&&a(Nr),e&&a(Mt),e&&a(Ar),e&&a(ct),e&&a(Cr),y(zt,e),e&&a(Gr),e&&a(It),e&&a(Vr),e&&a(va),e&&a(Fr),y(jt,e),e&&a(Jr),e&&a(Dt),e&&a(Rr),e&&a(wt),e&&a(Hr),y(Ot,e),e&&a(Lr),e&&a(Bt),e&&a(Xr),e&&a(gt),e&&a(Ur),y(Nt,e),e&&a(Yr),e&&a(At),e&&a(Kr),e&&a(_t),y(vt),e&&a(Qr),e&&a(ba)}}}function oo(f){let r,t,s,u,W,q,$,T,S;return T=new ai({props:{$$slots:{default:[io]},$$scope:{ctx:f}}}),{c(){r=d("meta"),t=h(),s=d("h1"),u=i("Minimizing MSE"),W=h(),q=d("div"),$=h(),w(T.$$.fragment),this.h()},l(k){const z=ti("svelte-116j2b7",document.head);r=c(z,"META",{name:!0,content:!0}),z.forEach(a),t=m(k),s=c(k,"H1",{});var x=E(s);u=o(x,"Minimizing MSE"),x.forEach(a),W=m(k),q=c(k,"DIV",{class:!0}),E(q).forEach(a),$=m(k),g(T.$$.fragment,k),this.h()},h(){document.title="Minimizing Mean Squared Error- World4AI",D(r,"name","description"),D(r,"content","In linear regression we can calculate the gradients of the weights and bias by construction a computational graph and applying automatic differentiation. Those gradients can be used in the gradient descent procedure to find optimal weights and biases."),D(q,"class","separator")},m(k,z){p(document.head,r),l(k,t,z),l(k,s,z),p(s,u),l(k,W,z),l(k,q,z),l(k,$,z),_(T,k,z),S=!0},p(k,[z]){const x={};z&536871167&&(x.$$scope={dirty:z,ctx:k}),T.$set(x)},i(k){S||(v(T.$$.fragment,k),S=!0)},o(k){b(T.$$.fragment,k),S=!1},d(k){a(r),k&&a(t),k&&a(s),k&&a(W),k&&a(q),k&&a($),y(T,k)}}}let Yl=.001;function fo(f,r,t){const s=[{x:5,y:20},{x:10,y:40},{x:35,y:15},{x:45,y:59}];let u=new L(1),W=new L(1);function q(){let M=new L(0);s.forEach(G=>{let B=u.mul(G.x).add(W);M=M.add(B.sub(new L(G.y)).pow(2))}),M=M.div(4),M.backward(),t(0,u.data-=Yl*u.grad,u),t(1,W.data-=Yl*W.grad,W),t(0,u.grad=0,u),t(1,W.grad=0,W)}let $;function T(){let M=new L(1);M._name="Weight: w";let G=new L(1);G._name="Bias b";let B;function ke(){M.grad=0,G.grad=0,B=null;let V=new L(s[0].x);V._name="Feature: x";let ie=M.mul(V);ie._name="w * x";let H=ie.add(G);H._name="Prediction";let we=new L(s[0].y);we._name="Target";let Z=we.sub(H);return Z._name="Error",B=Z.pow(2),B._name="MSE",B.backward(),{mse:B,w:M,b:G}}return ke}let S=T();function k(){let M,G,B=S();t(2,{mse:$,w:M,b:G}=B,$),M.data-=.001*M.grad,G.data-=.001*G.grad}let z;({mse:$,...z}=S());let x,I,J,ue;function $e(){let M=new L(1);M._name="Weight: w";let G=new L(1);G._name="Bias b";let B;M.grad=0,G.grad=0;let ke=new L(s[0].x);ke._name="Feature: x";let V=M.mul(ke);V._name="w * x",t(3,x=JSON.parse(JSON.stringify(V)));let ie=V.add(G);ie._name="Prediction",t(4,I=JSON.parse(JSON.stringify(ie)));let H=new L(s[0].y);H._name="Target";let we=H.sub(ie);we._name="Error",B=we.pow(2),B._name="MSE",t(5,J=JSON.parse(JSON.stringify(B))),B.backward(),t(6,ue=JSON.parse(JSON.stringify(B)))}$e();let R=new L(500);R._name="Branch 1";let Xt=new L(196);Xt._name="Branch 2";let Q=R.add(Xt);Q._name="Sum";let yt=Q.mul(.5);return yt._name="MSE",yt.backward(),[u,W,$,x,I,J,ue,yt,s,q,k,`import torch
import sklearn.datasets as datasets`,"X, y = datasets.make_regression(n_samples=100, n_features=2, n_informative=2, noise=0.01)",`X = torch.from_numpy(X).to(torch.float32);
y = torch.from_numpy(y).to(torch.float32).unsqueeze(1)`,`def init_weights():
    w = torch.randn(1, 2, requires_grad=True)
    b = torch.randn(1, 1, requires_grad=True)
    return w, b
w, b = init_weights()`,`def print_all():
    print(f'Weight: {w.data}, Grad: {w.grad}')
    print(f'Bias: {b.data}, Grad: {b.grad}')
print_all()`,`def forward(w, b):
    y_hat = X @ w.T + b
    return ((y - y_hat)**2).sum() / 100.0
mse = forward(w, b)`,"print_all()",`mse.backward()
print_all()`,`mse = forward(w, b)
mse.backward()
print_all()`,`w.grad.zero_()
print(w.grad)`,`lr = 0.1
w, b = init_weights()
for _ in range(10):
    # forward pass
    mse = forward(w, b)
    
    print(f'Mean squared error: {mse.data}')
    
    # backward pass
    mse.backward()
    
    # gradient descent
    with torch.inference_mode():
        w.data.sub_(w.grad * lr)
        b.data.sub_(b.grad * lr)
        w.grad.zero_()
        b.grad.zero_()`]}class yo extends Ql{constructor(r){super(),Zl(this,r,fo,oo,ei,{})}}export{yo as default};
