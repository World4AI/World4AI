import{S as qo,i as Uo,s as Ko,y as S,a as y,k as T,z as I,c as E,l as A,A as P,b as i,g as D,d as C,B as W,h as a,C as _e,Q as M,q as v,e as ae,R,m as g,r as $,n as u,N as p,P as me,u as qs,W as Us,L as Ks}from"../chunks/index.4d92b023.js";import{C as Zs}from"../chunks/Container.b0705c7b.js";import{H as Lt}from"../chunks/Highlight.b7c1de53.js";import{S as js}from"../chunks/SvgContainer.f70b5745.js";import{S as xs}from"../chunks/StepButton.2fb0289b.js";import{B as Gs}from"../chunks/ButtonContainer.e9aac418.js";import{P as x}from"../chunks/PythonCode.212ba7a6.js";import{P as Ys,T as Xs}from"../chunks/Ticks.45eca5c5.js";import{P as Es}from"../chunks/Path.7e6df014.js";import{X as Qs,Y as Js}from"../chunks/YLabel.182e66a3.js";import{L as ks}from"../chunks/Legend.de38c007.js";function Ts(o,t,l){const s=o.slice();return s[6]=t[l],s[8]=l,s}function As(o,t,l){const s=o.slice();return s[6]=t[l],s[8]=l,s}function Vs(o,t,l){const s=o.slice();return s[6]=t[l],s[8]=l,s}function Ls(o,t,l){const s=o.slice();return s[6]=t[l],s[8]=l,s}function el(o){let t,l;return t=new xs({}),t.$on("click",o[5]),{c(){S(t.$$.fragment)},l(s){I(t.$$.fragment,s)},m(s,n){P(t,s,n),l=!0},p:_e,i(s){l||(D(t.$$.fragment,s),l=!0)},o(s){C(t.$$.fragment,s),l=!1},d(s){W(t,s)}}}function Ss(o){let t,l;return{c(){t=M("line"),this.h()},l(s){t=R(s,"line",{x1:!0,x2:!0,y1:!0,y2:!0,stroke:!0,"stroke-width":!0,"stroke-dasharray":!0}),g(t).forEach(a),this.h()},h(){u(t,"x1",O+oe),u(t,"x2",Oa),u(t,"y1",l=o[8]*2+(o[2]*o[4]+(o[2]-2)*O)/2),u(t,"y2",o[8]*O+o[8]*o[4]+o[4]/2),u(t,"stroke","black"),u(t,"stroke-width","0.5px"),u(t,"stroke-dasharray","5 5")},m(s,n){i(s,t,n)},p(s,n){n&4&&l!==(l=s[8]*2+(s[2]*s[4]+(s[2]-2)*O)/2)&&u(t,"y1",l)},d(s){s&&a(t)}}}function Is(o){let t,l,s,n=o[8]+1+"",d;return{c(){t=M("rect"),l=M("text"),s=v("Fold Nr. "),d=v(n),this.h()},l(h){t=R(h,"rect",{x:!0,y:!0,width:!0,height:!0,fill:!0,class:!0}),g(t).forEach(a),l=R(h,"text",{x:!0,y:!0,class:!0});var f=g(l);s=$(f,"Fold Nr. "),d=$(f,n),f.forEach(a),this.h()},h(){u(t,"x",Oa),u(t,"y",o[8]*O+o[8]*o[4]),u(t,"width",oe),u(t,"height",o[4]),u(t,"fill","var(--main-color-4)"),u(t,"class","svelte-9ss2l3"),u(l,"x",Oa+oe/2),u(l,"y",o[8]*O+o[8]*o[4]+o[4]/2),u(l,"class","svelte-9ss2l3")},m(h,f){i(h,t,f),i(h,l,f),p(l,s),p(l,d)},p:_e,d(h){h&&a(t),h&&a(l)}}}function Ps(o){let t,l,s,n=o[8]===o[3]?"Validate":"Train",d;return{c(){t=M("rect"),s=M("text"),d=v(n),this.h()},l(h){t=R(h,"rect",{x:!0,y:!0,width:!0,height:!0,fill:!0,class:!0}),g(t).forEach(a),s=R(h,"text",{x:!0,y:!0,class:!0});var f=g(s);d=$(f,n),f.forEach(a),this.h()},h(){u(t,"x",jo),u(t,"y",o[8]*O+o[8]*o[4]),u(t,"width",oe),u(t,"height",o[4]),u(t,"fill",l=o[8]===o[3]?"var(--main-color-1)":"var(--main-color-2)"),u(t,"class","svelte-9ss2l3"),u(s,"x",jo+oe/2),u(s,"y",o[8]*O+o[8]*o[4]+o[4]/2),u(s,"class","svelte-9ss2l3")},m(h,f){i(h,t,f),i(h,s,f),p(s,d)},p(h,f){f&8&&l!==(l=h[8]===h[3]?"var(--main-color-1)":"var(--main-color-2)")&&u(t,"fill",l),f&8&&n!==(n=h[8]===h[3]?"Validate":"Train")&&qs(d,n)},d(h){h&&a(t),h&&a(s)}}}function Ds(o){let t;return{c(){t=M("line"),this.h()},l(l){t=R(l,"line",{x1:!0,x2:!0,y1:!0,y2:!0,stroke:!0,"stroke-width":!0,"stroke-dasharray":!0}),g(t).forEach(a),this.h()},h(){u(t,"x1",Oa+oe),u(t,"x2",jo),u(t,"y1",o[8]*O+o[8]*o[4]+o[4]/2),u(t,"y2",o[8]*O+o[8]*o[4]+o[4]/2),u(t,"stroke","black"),u(t,"stroke-width","0.5px"),u(t,"stroke-dasharray","2 2")},m(l,s){i(l,t,s)},p:_e,d(l){l&&a(t)}}}function tl(o){let t,l,s,n,d,h,f,m,k,c,V,L,Z,j,b=Array(o[2]),F=[];for(let w=0;w<b.length;w+=1)F[w]=Ss(Ls(o,b,w));let G=Array(o[2]),q=[];for(let w=0;w<G.length;w+=1)q[w]=Is(Vs(o,G,w));let Y=Array(o[2]),U=[];for(let w=0;w<Y.length;w+=1)U[w]=Ps(As(o,Y,w));let Q=Array(o[2]),B=[];for(let w=0;w<Q.length;w+=1)B[w]=Ds(Ts(o,Q,w));return{c(){t=M("svg"),l=M("rect"),n=M("rect");for(let w=0;w<F.length;w+=1)F[w].c();h=M("text"),f=v("Train + Validate"),k=M("text"),c=v("Test");for(let w=0;w<q.length;w+=1)q[w].c();L=ae();for(let w=0;w<U.length;w+=1)U[w].c();Z=ae();for(let w=0;w<B.length;w+=1)B[w].c();this.h()},l(w){t=R(w,"svg",{viewBox:!0});var N=g(t);l=R(N,"rect",{x:!0,y:!0,width:!0,height:!0,fill:!0,class:!0}),g(l).forEach(a),n=R(N,"rect",{x:!0,y:!0,width:!0,height:!0,fill:!0,class:!0}),g(n).forEach(a);for(let H=0;H<F.length;H+=1)F[H].l(N);h=R(N,"text",{x:!0,y:!0,class:!0});var _=g(h);f=$(_,"Train + Validate"),_.forEach(a),k=R(N,"text",{x:!0,y:!0,class:!0});var z=g(k);c=$(z,"Test"),z.forEach(a);for(let H=0;H<q.length;H+=1)q[H].l(N);L=ae();for(let H=0;H<U.length;H+=1)U[H].l(N);Z=ae();for(let H=0;H<B.length;H+=1)B[H].l(N);N.forEach(a),this.h()},h(){u(l,"x",O),u(l,"y",O),u(l,"width",oe),u(l,"height",s=o[2]*o[4]+(o[2]-2)*O),u(l,"fill","var(--main-color-3)"),u(l,"class","svelte-9ss2l3"),u(n,"x",O),u(n,"y",d=O+o[2]*o[4]+(o[2]-2)*O),u(n,"width",oe),u(n,"height",o[4]),u(n,"fill","var(--main-color-3)"),u(n,"class","svelte-9ss2l3"),u(h,"x",oe/2),u(h,"y",m=(o[2]*o[4]+(o[2]-2)*O)/2),u(h,"class","svelte-9ss2l3"),u(k,"x",oe/2),u(k,"y",V=O+o[2]*o[4]+o[4]/2+(o[2]-2)*O),u(k,"class","svelte-9ss2l3"),u(t,"viewBox",j="0 0 "+o[0]+" "+(o[1]+O))},m(w,N){i(w,t,N),p(t,l),p(t,n);for(let _=0;_<F.length;_+=1)F[_]&&F[_].m(t,null);p(t,h),p(h,f),p(t,k),p(k,c);for(let _=0;_<q.length;_+=1)q[_]&&q[_].m(t,null);p(t,L);for(let _=0;_<U.length;_+=1)U[_]&&U[_].m(t,null);p(t,Z);for(let _=0;_<B.length;_+=1)B[_]&&B[_].m(t,null)},p(w,N){if(N&4&&s!==(s=w[2]*w[4]+(w[2]-2)*O)&&u(l,"height",s),N&4&&d!==(d=O+w[2]*w[4]+(w[2]-2)*O)&&u(n,"y",d),N&20){b=Array(w[2]);let _;for(_=0;_<b.length;_+=1){const z=Ls(w,b,_);F[_]?F[_].p(z,N):(F[_]=Ss(z),F[_].c(),F[_].m(t,h))}for(;_<F.length;_+=1)F[_].d(1);F.length=b.length}if(N&4&&m!==(m=(w[2]*w[4]+(w[2]-2)*O)/2)&&u(h,"y",m),N&4&&V!==(V=O+w[2]*w[4]+w[4]/2+(w[2]-2)*O)&&u(k,"y",V),N&20){G=Array(w[2]);let _;for(_=0;_<G.length;_+=1){const z=Vs(w,G,_);q[_]?q[_].p(z,N):(q[_]=Is(z),q[_].c(),q[_].m(t,L))}for(;_<q.length;_+=1)q[_].d(1);q.length=G.length}if(N&28){Y=Array(w[2]);let _;for(_=0;_<Y.length;_+=1){const z=As(w,Y,_);U[_]?U[_].p(z,N):(U[_]=Ps(z),U[_].c(),U[_].m(t,Z))}for(;_<U.length;_+=1)U[_].d(1);U.length=Y.length}if(N&20){Q=Array(w[2]);let _;for(_=0;_<Q.length;_+=1){const z=Ts(w,Q,_);B[_]?B[_].p(z,N):(B[_]=Ds(z),B[_].c(),B[_].m(t,null))}for(;_<B.length;_+=1)B[_].d(1);B.length=Q.length}N&3&&j!==(j="0 0 "+w[0]+" "+(w[1]+O))&&u(t,"viewBox",j)},d(w){w&&a(t),me(F,w),me(q,w),me(U,w),me(B,w)}}}function al(o){let t,l,s,n,d,h;return t=new Gs({props:{$$slots:{default:[el]},$$scope:{ctx:o}}}),d=new js({props:{maxWidth:"800px",$$slots:{default:[tl]},$$scope:{ctx:o}}}),{c(){S(t.$$.fragment),l=y(),s=T("br"),n=y(),S(d.$$.fragment)},l(f){I(t.$$.fragment,f),l=E(f),s=A(f,"BR",{}),n=E(f),I(d.$$.fragment,f)},m(f,m){P(t,f,m),i(f,l,m),i(f,s,m),i(f,n,m),P(d,f,m),h=!0},p(f,[m]){const k={};m&4096&&(k.$$scope={dirty:m,ctx:f}),t.$set(k);const c={};m&4111&&(c.$$scope={dirty:m,ctx:f}),d.$set(c)},i(f){h||(D(t.$$.fragment,f),D(d.$$.fragment,f),h=!0)},o(f){C(t.$$.fragment,f),C(d.$$.fragment,f),h=!1},d(f){W(t,f),f&&a(l),f&&a(s),f&&a(n),W(d,f)}}}let O=2,oe=100,Oa=200,jo=400;function ol(o,t,l){let{width:s=500}=t,{height:n=250}=t,{numFolds:d=10}=t,h=n/(d+1)-O,f=0;function m(){l(3,f+=1),l(3,f%=d)}return o.$$set=k=>{"width"in k&&l(0,s=k.width),"height"in k&&l(1,n=k.height),"numFolds"in k&&l(2,d=k.numFolds)},[s,n,d,f,h,m]}class sl extends qo{constructor(t){super(),Uo(this,t,ol,al,Ko,{width:0,height:1,numFolds:2})}}function Cs(o,t,l){const s=o.slice();return s[5]=t[l],s[7]=l,s}function Ws(o,t,l){const s=o.slice();return s[8]=t[l],s[10]=l,s}function Fs(o,t,l){const s=o.slice();return s[5]=t[l],s[7]=l,s}function Ns(o,t,l){const s=o.slice();return s[8]=t[l],s[10]=l,s}function Os(o){let t,l,s=o[8].value+"",n;return{c(){t=M("rect"),l=M("text"),n=v(s),this.h()},l(d){t=R(d,"rect",{x:!0,y:!0,class:!0,width:!0,height:!0}),g(t).forEach(a),l=R(d,"text",{x:!0,y:!0,class:!0});var h=g(l);n=$(h,s),h.forEach(a),this.h()},h(){u(t,"x",o[10]*o[2]+o[10]*K+K),u(t,"y",o[7]*o[2]+o[7]*K+K),u(t,"class","fill-slate-600 stroke-black"),u(t,"width",o[2]),u(t,"height",o[2]),u(l,"x",o[10]*o[2]+o[10]*K+o[2]/2+K),u(l,"y",o[7]*o[2]+o[7]*K+o[2]/2+K+1),u(l,"class","fill-white svelte-uhpxmv")},m(d,h){i(d,t,h),i(d,l,h),p(l,n)},p:_e,d(d){d&&a(t),d&&a(l)}}}function Bs(o){let t,l=o[5],s=[];for(let n=0;n<l.length;n+=1)s[n]=Os(Ns(o,l,n));return{c(){for(let n=0;n<s.length;n+=1)s[n].c();t=ae()},l(n){for(let d=0;d<s.length;d+=1)s[d].l(n);t=ae()},m(n,d){for(let h=0;h<s.length;h+=1)s[h]&&s[h].m(n,d);i(n,t,d)},p(n,d){if(d&12){l=n[5];let h;for(h=0;h<l.length;h+=1){const f=Ns(n,l,h);s[h]?s[h].p(f,d):(s[h]=Os(f),s[h].c(),s[h].m(t.parentNode,t))}for(;h<s.length;h+=1)s[h].d(1);s.length=l.length}},d(n){me(s,n),n&&a(t)}}}function zs(o){let t,l,s,n=o[8].value+"",d,h;return{c(){t=M("rect"),s=M("text"),d=v(n),this.h()},l(f){t=R(f,"rect",{x:!0,y:!0,class:!0,width:!0,height:!0}),g(t).forEach(a),s=R(f,"text",{x:!0,y:!0,class:!0});var m=g(s);d=$(m,n),m.forEach(a),this.h()},h(){u(t,"x",l=o[0]-o[2]-(o[10]*o[2]+o[10]*K+K)),u(t,"y",o[7]*o[2]+o[7]*K+K),u(t,"class",`stroke-black ${o[8].splitIndex===0?"fill-red-300":"fill-blue-300"}`),u(t,"width",o[2]),u(t,"height",o[2]),u(s,"x",h=o[0]-o[2]/2-(o[10]*o[2]+o[10]*K+K)),u(s,"y",o[7]*o[2]+o[7]*K+o[2]/2+K+1),u(s,"class","svelte-uhpxmv")},m(f,m){i(f,t,m),i(f,s,m),p(s,d)},p(f,m){m&1&&l!==(l=f[0]-f[2]-(f[10]*f[2]+f[10]*K+K))&&u(t,"x",l),m&1&&h!==(h=f[0]-f[2]/2-(f[10]*f[2]+f[10]*K+K))&&u(s,"x",h)},d(f){f&&a(t),f&&a(s)}}}function Hs(o){let t,l=o[5],s=[];for(let n=0;n<l.length;n+=1)s[n]=zs(Ws(o,l,n));return{c(){for(let n=0;n<s.length;n+=1)s[n].c();t=ae()},l(n){for(let d=0;d<s.length;d+=1)s[d].l(n);t=ae()},m(n,d){for(let h=0;h<s.length;h+=1)s[h]&&s[h].m(n,d);i(n,t,d)},p(n,d){if(d&13){l=n[5];let h;for(h=0;h<l.length;h+=1){const f=Ws(n,l,h);s[h]?s[h].p(f,d):(s[h]=zs(f),s[h].c(),s[h].m(t.parentNode,t))}for(;h<s.length;h+=1)s[h].d(1);s.length=l.length}},d(n){me(s,n),n&&a(t)}}}function ll(o){let t,l,s,n=o[3],d=[];for(let m=0;m<n.length;m+=1)d[m]=Bs(Fs(o,n,m));let h=o[3],f=[];for(let m=0;m<h.length;m+=1)f[m]=Hs(Cs(o,h,m));return{c(){t=M("svg");for(let m=0;m<d.length;m+=1)d[m].c();l=ae();for(let m=0;m<f.length;m+=1)f[m].c();this.h()},l(m){t=R(m,"svg",{viewBox:!0});var k=g(t);for(let c=0;c<d.length;c+=1)d[c].l(k);l=ae();for(let c=0;c<f.length;c+=1)f[c].l(k);k.forEach(a),this.h()},h(){u(t,"viewBox",s="0 0 "+o[0]+" "+(o[1]+K*Ba))},m(m,k){i(m,t,k);for(let c=0;c<d.length;c+=1)d[c]&&d[c].m(t,null);p(t,l);for(let c=0;c<f.length;c+=1)f[c]&&f[c].m(t,null)},p(m,[k]){if(k&12){n=m[3];let c;for(c=0;c<n.length;c+=1){const V=Fs(m,n,c);d[c]?d[c].p(V,k):(d[c]=Bs(V),d[c].c(),d[c].m(t,l))}for(;c<d.length;c+=1)d[c].d(1);d.length=n.length}if(k&13){h=m[3];let c;for(c=0;c<h.length;c+=1){const V=Cs(m,h,c);f[c]?f[c].p(V,k):(f[c]=Hs(V),f[c].c(),f[c].m(t,null))}for(;c<f.length;c+=1)f[c].d(1);f.length=h.length}k&3&&s!==(s="0 0 "+m[0]+" "+(m[1]+K*Ba))&&u(t,"viewBox",s)},i:_e,o:_e,d(m){m&&a(t),me(d,m),me(f,m)}}}let Ba=10,Ms=10,K=2;function rl(o,t,l){let{width:s=500}=t,{height:n=200}=t,{type:d="random"}=t,h=n/Ba,f=[];for(let m=0;m<Ba;m++){let k=[];for(let c=0;c<Ms;c++){let V=m,L;d==="random"?Math.random()>=.5?L=0:L=1:d==="stratified"&&(c<Ms/2?L=0:L=1),k.push({value:V,splitIndex:L})}f.push(k)}return o.$$set=m=>{"width"in m&&l(0,s=m.width),"height"in m&&l(1,n=m.height),"type"in m&&l(4,d=m.type)},[s,n,h,f,d]}class Rs extends qo{constructor(t){super(),Uo(this,t,rl,ll,Ko,{width:0,height:1,type:4})}}const il=""+new URL("../assets/overfitting.311ef762.webp",import.meta.url).href;function nl(o){let t;return{c(){t=v("training")},l(l){t=$(l,"training")},m(l,s){i(l,t,s)},d(l){l&&a(t)}}}function fl(o){let t;return{c(){t=v("validation")},l(l){t=$(l,"validation")},m(l,s){i(l,t,s)},d(l){l&&a(t)}}}function hl(o){let t,l,s,n,d,h,f,m,k,c,V,L,Z,j;return t=new Es({props:{data:o[0],color:"var(--main-color-1)"}}),s=new Es({props:{data:o[1]}}),d=new Xs({props:{xTicks:[0,10,20,30,40,50,60,70,80,90,100],yTicks:[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],xOffset:-19,yOffset:18,fontSize:10}}),f=new Qs({props:{text:"Time",fontSize:15}}),k=new Js({props:{text:"Loss",fontSize:15}}),V=new ks({props:{text:"Validation Loss",coordinates:{x:75,y:.92}}}),Z=new ks({props:{text:"Training Loss",coordinates:{x:75,y:.85},legendColor:"var(--main-color-1)"}}),{c(){S(t.$$.fragment),l=y(),S(s.$$.fragment),n=y(),S(d.$$.fragment),h=y(),S(f.$$.fragment),m=y(),S(k.$$.fragment),c=y(),S(V.$$.fragment),L=y(),S(Z.$$.fragment)},l(b){I(t.$$.fragment,b),l=E(b),I(s.$$.fragment,b),n=E(b),I(d.$$.fragment,b),h=E(b),I(f.$$.fragment,b),m=E(b),I(k.$$.fragment,b),c=E(b),I(V.$$.fragment,b),L=E(b),I(Z.$$.fragment,b)},m(b,F){P(t,b,F),i(b,l,F),P(s,b,F),i(b,n,F),P(d,b,F),i(b,h,F),P(f,b,F),i(b,m,F),P(k,b,F),i(b,c,F),P(V,b,F),i(b,L,F),P(Z,b,F),j=!0},p:_e,i(b){j||(D(t.$$.fragment,b),D(s.$$.fragment,b),D(d.$$.fragment,b),D(f.$$.fragment,b),D(k.$$.fragment,b),D(V.$$.fragment,b),D(Z.$$.fragment,b),j=!0)},o(b){C(t.$$.fragment,b),C(s.$$.fragment,b),C(d.$$.fragment,b),C(f.$$.fragment,b),C(k.$$.fragment,b),C(V.$$.fragment,b),C(Z.$$.fragment,b),j=!1},d(b){W(t,b),b&&a(l),W(s,b),b&&a(n),W(d,b),b&&a(h),W(f,b),b&&a(m),W(k,b),b&&a(c),W(V,b),b&&a(L),W(Z,b)}}}function ul(o){let t;return{c(){t=v("test")},l(l){t=$(l,"test")},m(l,s){i(l,t,s)},d(l){l&&a(t)}}}function dl(o){let t,l,s,n,d,h,f,m,k,c;return{c(){t=M("svg"),l=M("rect"),s=M("rect"),n=M("rect"),d=M("text"),h=v("Test"),f=M("text"),m=v("Validate"),k=M("text"),c=v("Train"),this.h()},l(V){t=R(V,"svg",{class:!0,viewBox:!0});var L=g(t);l=R(L,"rect",{x:!0,y:!0,width:!0,height:!0,class:!0}),g(l).forEach(a),s=R(L,"rect",{x:!0,y:!0,width:!0,height:!0,class:!0}),g(s).forEach(a),n=R(L,"rect",{x:!0,y:!0,width:!0,height:!0,class:!0}),g(n).forEach(a),d=R(L,"text",{x:!0,y:!0,class:!0});var Z=g(d);h=$(Z,"Test"),Z.forEach(a),f=R(L,"text",{x:!0,y:!0,class:!0});var j=g(f);m=$(j,"Validate"),j.forEach(a),k=R(L,"text",{x:!0,y:!0,class:!0});var b=g(k);c=$(b,"Train"),b.forEach(a),L.forEach(a),this.h()},h(){u(l,"x","0"),u(l,"y","0"),u(l,"width","50"),u(l,"height","40"),u(l,"class","fill-blue-200"),u(s,"x","51"),u(s,"y","0"),u(s,"width","50"),u(s,"height","40"),u(s,"class","fill-red-200"),u(n,"x","102"),u(n,"y","0"),u(n,"width","400"),u(n,"height","40"),u(n,"class","fill-gray-200"),u(d,"x","25"),u(d,"y","20"),u(d,"class","svelte-142ygny"),u(f,"x","75"),u(f,"y","20"),u(f,"class","svelte-142ygny"),u(k,"x","300"),u(k,"y","20"),u(k,"class","svelte-142ygny"),u(t,"class","split"),u(t,"viewBox","0 0 500 40")},m(V,L){i(V,t,L),p(t,l),p(t,s),p(t,n),p(t,d),p(d,h),p(t,f),p(f,m),p(t,k),p(k,c)},p:_e,d(V){V&&a(t)}}}function pl(o){let t;return{c(){t=v("stratified")},l(l){t=$(l,"stratified")},m(l,s){i(l,t,s)},d(l){l&&a(t)}}}function cl(o){let t;return{c(){t=v("k-fold cross-validation")},l(l){t=$(l,"k-fold cross-validation")},m(l,s){i(l,t,s)},d(l){l&&a(t)}}}function ml(o){let t,l,s,n,d,h,f,m,k,c,V,L,Z,j,b,F,G,q,Y,U,Q,B,w,N,_,z,H,St,qe,za,It,se,Pt,Ue,Ha,Dt,Ke,Ma,Ct,ve,Wt,Ze,Ra,Ft,le,ja,re,qa,Nt,$e,Ot,xe,Ua,Bt,we,zt,ie,Ka,ht,Za,xa,Ht,ge,Mt,ne,Ga,ut,Ya,Xa,Rt,be,jt,fe,Qa,dt,Ja,eo,qt,ye,Ut,Ge,to,Kt,Ee,Zt,Ye,ao,xt,ke,Gt,he,oo,pt,so,lo,Yt,Te,Xt,ue,ro,ct,io,no,Qt,Ae,Jt,ee,fo,mt,ho,uo,_t,po,co,ea,Ve,ta,Xe,mo,aa,Le,oa,Se,sa,Qe,_o,la,Ie,ra,Pe,ia,Je,vo,na,De,fa,et,$o,ha,Ce,ua,We,Zo,da,tt,wo,pa,Fe,ca,Ne,ma,at,_a,ot,go,va,de,bo,pe,yo,$a,st,Eo,wa,Oe,ga,lt,ko,ba,rt,To,ya,J,Ao,vt,Vo,Lo,$t,So,Io,wt,Po,Do,Ea,Be,ka,X,Co,gt,Wo,Fo,bt,No,Oo,yt,Bo,zo,Et,Ho,Mo,Ta,ze,Aa,He,Va,it,Ro,La,nt,Sa;return j=new Lt({props:{$$slots:{default:[nl]},$$scope:{ctx:o}}}),Y=new Lt({props:{$$slots:{default:[fl]},$$scope:{ctx:o}}}),B=new Ys({props:{width:500,height:300,maxWidth:700,domain:[0,100],range:[0,1],$$slots:{default:[hl]},$$scope:{ctx:o}}}),z=new Lt({props:{$$slots:{default:[ul]},$$scope:{ctx:o}}}),se=new js({props:{maxWidth:"700px",$$slots:{default:[dl]},$$scope:{ctx:o}}}),ve=new Rs({props:{type:"random"}}),re=new Lt({props:{$$slots:{default:[pl]},$$scope:{ctx:o}}}),$e=new Rs({props:{type:"stratified"}}),we=new x({props:{code:o[2]}}),ge=new x({props:{code:o[3]}}),be=new x({props:{code:o[4]}}),ye=new x({props:{code:o[5]}}),Ee=new x({props:{code:o[6]}}),ke=new x({props:{code:o[7]}}),Te=new x({props:{code:o[8]}}),Ae=new x({props:{code:o[9]}}),Ve=new x({props:{code:o[10]}}),Le=new x({props:{code:o[11]}}),Se=new x({props:{code:o[12]}}),Ie=new x({props:{code:o[13]}}),Pe=new x({props:{code:o[14],isOutput:!0}}),De=new x({props:{code:o[15]}}),Ce=new x({props:{code:o[16]}}),Fe=new x({props:{code:o[17]}}),Ne=new x({props:{code:o[18],isOutput:!0}}),pe=new Lt({props:{$$slots:{default:[cl]},$$scope:{ctx:o}}}),Oe=new sl({}),Be=new x({props:{code:o[19]}}),ze=new x({props:{code:o[20]}}),He=new x({props:{code:o[21],isOutput:!0}}),{c(){t=T("p"),l=v(`Intuitively we understand that overfitting leads to a model that does not
    generalize well to new unforseen data, but we would also like to have some
    tools that would allow us to measure the level of overfitting during the
    training process. It turns out that splitting the dataset into different
    buckets, called sets is essential to achieve the goal of measuring
    overfitting.`),s=y(),n=T("div"),d=y(),h=T("h2"),f=v("Data Splitting"),m=y(),k=T("p"),c=v(`All examples that we covered so far assumed that we have a training dataset.
    In practice we split the dataset into preferably 3 sets. The training set,
    the validation set and the test set.`),V=y(),L=T("p"),Z=v("The "),S(j.$$.fragment),b=v(` set contians the vast majority of available
    data. It is the part of the data that is actually used to train a neural network.
    The other sets are never used to directly adjust the weights and biases of a
    neural network.`),F=y(),G=T("p"),q=v("The "),S(Y.$$.fragment),U=v(` set is also used in the training process,
    but only during the performance measurement step. The validation set allows us
    to simulate a situation, where the neural network encounters new data. After
    each epoch (or batch) we use the training and the validation sets separately
    to measure the loss. At first both losses will decline, but after a while the
    validation loss might start to increase again, while the training loss keeps
    decreasing. This is a strong indication that our model overfits to the training
    data. The larger the divergence, the larger the level of overfitting.`),Q=y(),S(B.$$.fragment),w=y(),N=T("p"),_=v(`When we encounter overfitting we will most likely change several
    hyperparameters of the neural network and apply some techniques in order to
    reduce overfitting. It is not unlikely that we will continue doing that
    until we are satisfied with the performance. While we are not using the
    validation dataset directly in training, we are still observing the
    performance of the validation data and adjust accordingly, thus injecting
    our knowledge about the validation dataset into the training of the weights
    and biases. At some point it becomes hard to argue that the validation
    dataset represents completely unforseen data. The `),S(z.$$.fragment),H=v(` set on the other hand is neither touched nor seen during the training process
    at all. The intention of having this additional dataset is to provide a method
    to test the performance of our model when it encounters truly never before seen
    data. We only use the data once. If we find out that we overfitted to the training
    and the validation dataset, we can not go back to tweak the parameters, because
    we would require a completely new test dataset, which we might not posess.`),St=y(),qe=T("p"),za=v(`While there are no hard rules when it comes to the proportions of your
    splits, there are some rules of thumb. A 10-10-80 split is for example
    relatively common.`),It=y(),S(se.$$.fragment),Pt=y(),Ue=T("p"),Ha=v(`In that case we want to keep the biggest chunk of our data for training and
    keep roughly 10 percent for validation and 10 percent for testing.`),Dt=y(),Ke=T("p"),Ma=v(`We have a couple of options, when we split the dataset. The simplest
    approach would be to separate the data randomly. While this type of split is
    easy to implement, it might pose some problems. In the example below we are
    faced with a dataset consisting of 10 classes (numbers 0 to 9) with 10
    samples each. In the random procedure that we use below we generate a random
    number between 0 and 1. If the number is below 0.5 we assign the number to
    the blue split, otherwise the number is assigned to the red split.`),Ct=y(),S(ve.$$.fragment),Wt=y(),Ze=T("p"),Ra=v(`If you observe the splits, you will most likely notice that some splits have
    more numbers of a certain category. That means that the proportions of some
    categories in the two splits are different. This is especially a problem,
    when some of the categories have a limited number of samples. We could end
    up creating a split that doesn't include a particular category at all.`),Ft=y(),le=T("p"),ja=v("A "),S(re.$$.fragment),qa=v(` split on the other hand tries to keep the
    proportions of the different classes consistent.`),Nt=y(),S($e.$$.fragment),Ot=y(),xe=T("p"),Ua=v(`Now let's see how we can create and utilize the three datasets in PyTorch
    and whether theory matches the practice.`),Bt=y(),S(we.$$.fragment),zt=y(),ie=T("p"),Ka=v("The "),ht=T("code"),Za=v("MNIST"),xa=v(` object does not provide any functionality to get a validation
    dataset out of the box. We will download the training and testing datasets first
    and divide the training dataset into two parts: one for training one for validation.`),Ht=y(),S(ge.$$.fragment),Mt=y(),ne=T("p"),Ga=v("We use the "),ut=T("code"),Ya=v("train_test_split"),Xa=v(` function from sklearn to generate indices.
    Those indices indicate if a particular sample is going to be used for training
    or testing. We conduct a stratified split to keep the distribution of labels
    consistent. 90% of the data is going to be used for training and 10% for validation.`),Rt=y(),S(be.$$.fragment),jt=y(),fe=T("p"),Qa=v("To separate the dataset we use the "),dt=T("code"),Ja=v("Subset"),eo=v(` class which takes the
    original dataset and the indices and returns the modified dataset, where the
    samples that are not contained in the index list have been filtered out.`),qt=y(),S(ye.$$.fragment),Ut=y(),Ge=T("p"),to=v(`We keep the parameters similar to those in the previous section, but
    increase the number of epochs to show the effect of overfitting.`),Kt=y(),S(Ee.$$.fragment),Zt=y(),Ye=T("p"),ao=v(`Now we have everything that we require to create the three dataloaders: one
    for training, one for validating, one for testing.`),xt=y(),S(ke.$$.fragment),Gt=y(),he=T("p"),oo=v("We create the "),pt=T("code"),so=v("track_performance()"),lo=v(` function to calculate the average
    loss and the accuracy of the model.`),Yt=y(),S(Te.$$.fragment),Xt=y(),ue=T("p"),ro=v("The "),ct=T("code"),io=v("train_epoch()"),no=v(" function trains the model for a single epoch."),Qt=y(),S(Ae.$$.fragment),Jt=y(),ee=T("p"),fo=v("The "),mt=T("code"),ho=v("train()"),uo=v(` function simply loops over the number of epochs and
    puts measures the performance after each iteration. The results are saved in
    the `),_t=T("code"),po=v("history"),co=v(" dictionary."),ea=y(),S(Ve.$$.fragment),ta=y(),Xe=T("p"),mo=v("The model is identical to the one we have used over the last sections."),aa=y(),S(Le.$$.fragment),oa=y(),S(Se.$$.fragment),sa=y(),Qe=T("p"),_o=v(`We print our metrics for the training and the valdiation dataset. You can
    see that the model starts to overfit relatively fast.`),la=y(),S(Ie.$$.fragment),ra=y(),S(Pe.$$.fragment),ia=y(),Je=T("p"),vo=v("To reinforce our results, we draw the progression for the two datasets."),na=y(),S(De.$$.fragment),fa=y(),et=T("p"),$o=v("The divergence due to overfitting is obvious."),ha=y(),S(Ce.$$.fragment),ua=y(),We=T("img"),da=y(),tt=T("p"),wo=v(`The metrics for the test dataset are relatively close to those based on the
    validation dataset.`),pa=y(),S(Fe.$$.fragment),ca=y(),S(Ne.$$.fragment),ma=y(),at=T("div"),_a=y(),ot=T("h2"),go=v("K-Fold Cross-Validation"),va=y(),de=T("p"),bo=v(`In the approach above we divided the dataset into three distinct buckets and
    kept them constant during the whole training process, but ideally we would
    like to somehow use all available data in training and testing
    simultaneously. This is especially important if our dataset is relatively
    small. While we need to keep the test data separate, untouched by training,
    we can use the rest of the data simultaneously for training and validation
    by using `),S(pe.$$.fragment),yo=v("."),$a=y(),st=T("p"),Eo=v(`We divide the data (excluding the test set) into k equal sets, called folds.
    k is a hyperparameter, but usually we construct 5 or 10 folds. Each fold is
    basically a bucket of data that can be used either for trainig or
    validation. We use one of the folds for validation and the rest (k-1 folds)
    for training and we repeat the trainig process k times, switching the fold
    that is used for validation each time. After the k iterations we are left
    with k models and k measures of overfitting.`),wa=y(),S(Oe.$$.fragment),ga=y(),lt=T("p"),ko=v(`K-Fold cross-validation provides a much more robust measure of performance.
    At the end of the trainig process we average over the results of the k-folds
    to get a more accurate estimate of how our model performs. Once we are
    satisfied with the choise of our hyperparameters, we could retrain the model
    on the full k folds.`),ba=y(),rt=T("p"),To=v(`There is obviously also a downside to using k models. Training a neural
    network just once requires a lot of computaional resources. By using k folds
    we will more or less increase the training time by a factor of k-1.`),ya=y(),J=T("p"),Ao=v(`To implement k-fold cross-validation with PyTorch we will mostly reuse the
    code from above, but we still require a couple more components. PyTorch does
    not offer k-fold out of the box, but once again sklearn is the perfect
    companion for that purpose. Additionally we import the `),vt=T("code"),Vo=v("SubsetRandomSampler"),Lo=v(". A sampler can be used as an input into the "),$t=T("code"),So=v("DataLoader"),Io=v(`
    object, in order to determine how the samples in a dataset are going to be
    drawn. A random subset sampler specifically allows us to determine a subset,
    like a fold and the data is going to be sampled in a random fashion. We
    could have used the `),wt=T("code"),Po=v("Subset"),Do=v(` object from above to accomplish the same,
    but we wanted to teach you different approaches.`),Ea=y(),S(Be.$$.fragment),ka=y(),X=T("p"),Co=v("We use the "),gt=T("code"),Wo=v("seed"),Fo=v(" variable as input into "),bt=T("code"),No=v("KFold"),Oo=v(` and
    `),yt=T("code"),Bo=v("torch.maual_seed"),zo=v(`. A seed is variable that is used as input into
    the random number generator. The initial weights and biases of a neural
    network are generated randomly. By providing a seed into the function
    `),Et=T("code"),Ho=v("torch.manual_seed()"),Mo=v(` we make the parameters of the neural network
    identical for each of the folds and make our results reproduceble.`),Ta=y(),S(ze.$$.fragment),Aa=y(),S(He.$$.fragment),Va=y(),it=T("p"),Ro=v(`There is some variability in the folds, but that is not too bad for our
    first attempt.`),La=y(),nt=T("div"),this.h()},l(e){t=A(e,"P",{});var r=g(t);l=$(r,`Intuitively we understand that overfitting leads to a model that does not
    generalize well to new unforseen data, but we would also like to have some
    tools that would allow us to measure the level of overfitting during the
    training process. It turns out that splitting the dataset into different
    buckets, called sets is essential to achieve the goal of measuring
    overfitting.`),r.forEach(a),s=E(e),n=A(e,"DIV",{class:!0}),g(n).forEach(a),d=E(e),h=A(e,"H2",{});var kt=g(h);f=$(kt,"Data Splitting"),kt.forEach(a),m=E(e),k=A(e,"P",{});var Tt=g(k);c=$(Tt,`All examples that we covered so far assumed that we have a training dataset.
    In practice we split the dataset into preferably 3 sets. The training set,
    the validation set and the test set.`),Tt.forEach(a),V=E(e),L=A(e,"P",{});var Me=g(L);Z=$(Me,"The "),I(j.$$.fragment,Me),b=$(Me,` set contians the vast majority of available
    data. It is the part of the data that is actually used to train a neural network.
    The other sets are never used to directly adjust the weights and biases of a
    neural network.`),Me.forEach(a),F=E(e),G=A(e,"P",{});var Re=g(G);q=$(Re,"The "),I(Y.$$.fragment,Re),U=$(Re,` set is also used in the training process,
    but only during the performance measurement step. The validation set allows us
    to simulate a situation, where the neural network encounters new data. After
    each epoch (or batch) we use the training and the validation sets separately
    to measure the loss. At first both losses will decline, but after a while the
    validation loss might start to increase again, while the training loss keeps
    decreasing. This is a strong indication that our model overfits to the training
    data. The larger the divergence, the larger the level of overfitting.`),Re.forEach(a),Q=E(e),I(B.$$.fragment,e),w=E(e),N=A(e,"P",{});var je=g(N);_=$(je,`When we encounter overfitting we will most likely change several
    hyperparameters of the neural network and apply some techniques in order to
    reduce overfitting. It is not unlikely that we will continue doing that
    until we are satisfied with the performance. While we are not using the
    validation dataset directly in training, we are still observing the
    performance of the validation data and adjust accordingly, thus injecting
    our knowledge about the validation dataset into the training of the weights
    and biases. At some point it becomes hard to argue that the validation
    dataset represents completely unforseen data. The `),I(z.$$.fragment,je),H=$(je,` set on the other hand is neither touched nor seen during the training process
    at all. The intention of having this additional dataset is to provide a method
    to test the performance of our model when it encounters truly never before seen
    data. We only use the data once. If we find out that we overfitted to the training
    and the validation dataset, we can not go back to tweak the parameters, because
    we would require a completely new test dataset, which we might not posess.`),je.forEach(a),St=E(e),qe=A(e,"P",{});var At=g(qe);za=$(At,`While there are no hard rules when it comes to the proportions of your
    splits, there are some rules of thumb. A 10-10-80 split is for example
    relatively common.`),At.forEach(a),It=E(e),I(se.$$.fragment,e),Pt=E(e),Ue=A(e,"P",{});var Vt=g(Ue);Ha=$(Vt,`In that case we want to keep the biggest chunk of our data for training and
    keep roughly 10 percent for validation and 10 percent for testing.`),Vt.forEach(a),Dt=E(e),Ke=A(e,"P",{});var xo=g(Ke);Ma=$(xo,`We have a couple of options, when we split the dataset. The simplest
    approach would be to separate the data randomly. While this type of split is
    easy to implement, it might pose some problems. In the example below we are
    faced with a dataset consisting of 10 classes (numbers 0 to 9) with 10
    samples each. In the random procedure that we use below we generate a random
    number between 0 and 1. If the number is below 0.5 we assign the number to
    the blue split, otherwise the number is assigned to the red split.`),xo.forEach(a),Ct=E(e),I(ve.$$.fragment,e),Wt=E(e),Ze=A(e,"P",{});var Go=g(Ze);Ra=$(Go,`If you observe the splits, you will most likely notice that some splits have
    more numbers of a certain category. That means that the proportions of some
    categories in the two splits are different. This is especially a problem,
    when some of the categories have a limited number of samples. We could end
    up creating a split that doesn't include a particular category at all.`),Go.forEach(a),Ft=E(e),le=A(e,"P",{});var Ia=g(le);ja=$(Ia,"A "),I(re.$$.fragment,Ia),qa=$(Ia,` split on the other hand tries to keep the
    proportions of the different classes consistent.`),Ia.forEach(a),Nt=E(e),I($e.$$.fragment,e),Ot=E(e),xe=A(e,"P",{});var Yo=g(xe);Ua=$(Yo,`Now let's see how we can create and utilize the three datasets in PyTorch
    and whether theory matches the practice.`),Yo.forEach(a),Bt=E(e),I(we.$$.fragment,e),zt=E(e),ie=A(e,"P",{});var Pa=g(ie);Ka=$(Pa,"The "),ht=A(Pa,"CODE",{});var Xo=g(ht);Za=$(Xo,"MNIST"),Xo.forEach(a),xa=$(Pa,` object does not provide any functionality to get a validation
    dataset out of the box. We will download the training and testing datasets first
    and divide the training dataset into two parts: one for training one for validation.`),Pa.forEach(a),Ht=E(e),I(ge.$$.fragment,e),Mt=E(e),ne=A(e,"P",{});var Da=g(ne);Ga=$(Da,"We use the "),ut=A(Da,"CODE",{});var Qo=g(ut);Ya=$(Qo,"train_test_split"),Qo.forEach(a),Xa=$(Da,` function from sklearn to generate indices.
    Those indices indicate if a particular sample is going to be used for training
    or testing. We conduct a stratified split to keep the distribution of labels
    consistent. 90% of the data is going to be used for training and 10% for validation.`),Da.forEach(a),Rt=E(e),I(be.$$.fragment,e),jt=E(e),fe=A(e,"P",{});var Ca=g(fe);Qa=$(Ca,"To separate the dataset we use the "),dt=A(Ca,"CODE",{});var Jo=g(dt);Ja=$(Jo,"Subset"),Jo.forEach(a),eo=$(Ca,` class which takes the
    original dataset and the indices and returns the modified dataset, where the
    samples that are not contained in the index list have been filtered out.`),Ca.forEach(a),qt=E(e),I(ye.$$.fragment,e),Ut=E(e),Ge=A(e,"P",{});var es=g(Ge);to=$(es,`We keep the parameters similar to those in the previous section, but
    increase the number of epochs to show the effect of overfitting.`),es.forEach(a),Kt=E(e),I(Ee.$$.fragment,e),Zt=E(e),Ye=A(e,"P",{});var ts=g(Ye);ao=$(ts,`Now we have everything that we require to create the three dataloaders: one
    for training, one for validating, one for testing.`),ts.forEach(a),xt=E(e),I(ke.$$.fragment,e),Gt=E(e),he=A(e,"P",{});var Wa=g(he);oo=$(Wa,"We create the "),pt=A(Wa,"CODE",{});var as=g(pt);so=$(as,"track_performance()"),as.forEach(a),lo=$(Wa,` function to calculate the average
    loss and the accuracy of the model.`),Wa.forEach(a),Yt=E(e),I(Te.$$.fragment,e),Xt=E(e),ue=A(e,"P",{});var Fa=g(ue);ro=$(Fa,"The "),ct=A(Fa,"CODE",{});var os=g(ct);io=$(os,"train_epoch()"),os.forEach(a),no=$(Fa," function trains the model for a single epoch."),Fa.forEach(a),Qt=E(e),I(Ae.$$.fragment,e),Jt=E(e),ee=A(e,"P",{});var ft=g(ee);fo=$(ft,"The "),mt=A(ft,"CODE",{});var ss=g(mt);ho=$(ss,"train()"),ss.forEach(a),uo=$(ft,` function simply loops over the number of epochs and
    puts measures the performance after each iteration. The results are saved in
    the `),_t=A(ft,"CODE",{});var ls=g(_t);po=$(ls,"history"),ls.forEach(a),co=$(ft," dictionary."),ft.forEach(a),ea=E(e),I(Ve.$$.fragment,e),ta=E(e),Xe=A(e,"P",{});var rs=g(Xe);mo=$(rs,"The model is identical to the one we have used over the last sections."),rs.forEach(a),aa=E(e),I(Le.$$.fragment,e),oa=E(e),I(Se.$$.fragment,e),sa=E(e),Qe=A(e,"P",{});var is=g(Qe);_o=$(is,`We print our metrics for the training and the valdiation dataset. You can
    see that the model starts to overfit relatively fast.`),is.forEach(a),la=E(e),I(Ie.$$.fragment,e),ra=E(e),I(Pe.$$.fragment,e),ia=E(e),Je=A(e,"P",{});var ns=g(Je);vo=$(ns,"To reinforce our results, we draw the progression for the two datasets."),ns.forEach(a),na=E(e),I(De.$$.fragment,e),fa=E(e),et=A(e,"P",{});var fs=g(et);$o=$(fs,"The divergence due to overfitting is obvious."),fs.forEach(a),ha=E(e),I(Ce.$$.fragment,e),ua=E(e),We=A(e,"IMG",{src:!0,alt:!0}),da=E(e),tt=A(e,"P",{});var hs=g(tt);wo=$(hs,`The metrics for the test dataset are relatively close to those based on the
    validation dataset.`),hs.forEach(a),pa=E(e),I(Fe.$$.fragment,e),ca=E(e),I(Ne.$$.fragment,e),ma=E(e),at=A(e,"DIV",{class:!0}),g(at).forEach(a),_a=E(e),ot=A(e,"H2",{});var us=g(ot);go=$(us,"K-Fold Cross-Validation"),us.forEach(a),va=E(e),de=A(e,"P",{});var Na=g(de);bo=$(Na,`In the approach above we divided the dataset into three distinct buckets and
    kept them constant during the whole training process, but ideally we would
    like to somehow use all available data in training and testing
    simultaneously. This is especially important if our dataset is relatively
    small. While we need to keep the test data separate, untouched by training,
    we can use the rest of the data simultaneously for training and validation
    by using `),I(pe.$$.fragment,Na),yo=$(Na,"."),Na.forEach(a),$a=E(e),st=A(e,"P",{});var ds=g(st);Eo=$(ds,`We divide the data (excluding the test set) into k equal sets, called folds.
    k is a hyperparameter, but usually we construct 5 or 10 folds. Each fold is
    basically a bucket of data that can be used either for trainig or
    validation. We use one of the folds for validation and the rest (k-1 folds)
    for training and we repeat the trainig process k times, switching the fold
    that is used for validation each time. After the k iterations we are left
    with k models and k measures of overfitting.`),ds.forEach(a),wa=E(e),I(Oe.$$.fragment,e),ga=E(e),lt=A(e,"P",{});var ps=g(lt);ko=$(ps,`K-Fold cross-validation provides a much more robust measure of performance.
    At the end of the trainig process we average over the results of the k-folds
    to get a more accurate estimate of how our model performs. Once we are
    satisfied with the choise of our hyperparameters, we could retrain the model
    on the full k folds.`),ps.forEach(a),ba=E(e),rt=A(e,"P",{});var cs=g(rt);To=$(cs,`There is obviously also a downside to using k models. Training a neural
    network just once requires a lot of computaional resources. By using k folds
    we will more or less increase the training time by a factor of k-1.`),cs.forEach(a),ya=E(e),J=A(e,"P",{});var ce=g(J);Ao=$(ce,`To implement k-fold cross-validation with PyTorch we will mostly reuse the
    code from above, but we still require a couple more components. PyTorch does
    not offer k-fold out of the box, but once again sklearn is the perfect
    companion for that purpose. Additionally we import the `),vt=A(ce,"CODE",{});var ms=g(vt);Vo=$(ms,"SubsetRandomSampler"),ms.forEach(a),Lo=$(ce,". A sampler can be used as an input into the "),$t=A(ce,"CODE",{});var _s=g($t);So=$(_s,"DataLoader"),_s.forEach(a),Io=$(ce,`
    object, in order to determine how the samples in a dataset are going to be
    drawn. A random subset sampler specifically allows us to determine a subset,
    like a fold and the data is going to be sampled in a random fashion. We
    could have used the `),wt=A(ce,"CODE",{});var vs=g(wt);Po=$(vs,"Subset"),vs.forEach(a),Do=$(ce,` object from above to accomplish the same,
    but we wanted to teach you different approaches.`),ce.forEach(a),Ea=E(e),I(Be.$$.fragment,e),ka=E(e),X=A(e,"P",{});var te=g(X);Co=$(te,"We use the "),gt=A(te,"CODE",{});var $s=g(gt);Wo=$($s,"seed"),$s.forEach(a),Fo=$(te," variable as input into "),bt=A(te,"CODE",{});var ws=g(bt);No=$(ws,"KFold"),ws.forEach(a),Oo=$(te,` and
    `),yt=A(te,"CODE",{});var gs=g(yt);Bo=$(gs,"torch.maual_seed"),gs.forEach(a),zo=$(te,`. A seed is variable that is used as input into
    the random number generator. The initial weights and biases of a neural
    network are generated randomly. By providing a seed into the function
    `),Et=A(te,"CODE",{});var bs=g(Et);Ho=$(bs,"torch.manual_seed()"),bs.forEach(a),Mo=$(te,` we make the parameters of the neural network
    identical for each of the folds and make our results reproduceble.`),te.forEach(a),Ta=E(e),I(ze.$$.fragment,e),Aa=E(e),I(He.$$.fragment,e),Va=E(e),it=A(e,"P",{});var ys=g(it);Ro=$(ys,`There is some variability in the folds, but that is not too bad for our
    first attempt.`),ys.forEach(a),La=E(e),nt=A(e,"DIV",{class:!0}),g(nt).forEach(a),this.h()},h(){u(n,"class","separator"),Ks(We.src,Zo=il)||u(We,"src",Zo),u(We,"alt","Signs of overfitting"),u(at,"class","separator"),u(nt,"class","separator")},m(e,r){i(e,t,r),p(t,l),i(e,s,r),i(e,n,r),i(e,d,r),i(e,h,r),p(h,f),i(e,m,r),i(e,k,r),p(k,c),i(e,V,r),i(e,L,r),p(L,Z),P(j,L,null),p(L,b),i(e,F,r),i(e,G,r),p(G,q),P(Y,G,null),p(G,U),i(e,Q,r),P(B,e,r),i(e,w,r),i(e,N,r),p(N,_),P(z,N,null),p(N,H),i(e,St,r),i(e,qe,r),p(qe,za),i(e,It,r),P(se,e,r),i(e,Pt,r),i(e,Ue,r),p(Ue,Ha),i(e,Dt,r),i(e,Ke,r),p(Ke,Ma),i(e,Ct,r),P(ve,e,r),i(e,Wt,r),i(e,Ze,r),p(Ze,Ra),i(e,Ft,r),i(e,le,r),p(le,ja),P(re,le,null),p(le,qa),i(e,Nt,r),P($e,e,r),i(e,Ot,r),i(e,xe,r),p(xe,Ua),i(e,Bt,r),P(we,e,r),i(e,zt,r),i(e,ie,r),p(ie,Ka),p(ie,ht),p(ht,Za),p(ie,xa),i(e,Ht,r),P(ge,e,r),i(e,Mt,r),i(e,ne,r),p(ne,Ga),p(ne,ut),p(ut,Ya),p(ne,Xa),i(e,Rt,r),P(be,e,r),i(e,jt,r),i(e,fe,r),p(fe,Qa),p(fe,dt),p(dt,Ja),p(fe,eo),i(e,qt,r),P(ye,e,r),i(e,Ut,r),i(e,Ge,r),p(Ge,to),i(e,Kt,r),P(Ee,e,r),i(e,Zt,r),i(e,Ye,r),p(Ye,ao),i(e,xt,r),P(ke,e,r),i(e,Gt,r),i(e,he,r),p(he,oo),p(he,pt),p(pt,so),p(he,lo),i(e,Yt,r),P(Te,e,r),i(e,Xt,r),i(e,ue,r),p(ue,ro),p(ue,ct),p(ct,io),p(ue,no),i(e,Qt,r),P(Ae,e,r),i(e,Jt,r),i(e,ee,r),p(ee,fo),p(ee,mt),p(mt,ho),p(ee,uo),p(ee,_t),p(_t,po),p(ee,co),i(e,ea,r),P(Ve,e,r),i(e,ta,r),i(e,Xe,r),p(Xe,mo),i(e,aa,r),P(Le,e,r),i(e,oa,r),P(Se,e,r),i(e,sa,r),i(e,Qe,r),p(Qe,_o),i(e,la,r),P(Ie,e,r),i(e,ra,r),P(Pe,e,r),i(e,ia,r),i(e,Je,r),p(Je,vo),i(e,na,r),P(De,e,r),i(e,fa,r),i(e,et,r),p(et,$o),i(e,ha,r),P(Ce,e,r),i(e,ua,r),i(e,We,r),i(e,da,r),i(e,tt,r),p(tt,wo),i(e,pa,r),P(Fe,e,r),i(e,ca,r),P(Ne,e,r),i(e,ma,r),i(e,at,r),i(e,_a,r),i(e,ot,r),p(ot,go),i(e,va,r),i(e,de,r),p(de,bo),P(pe,de,null),p(de,yo),i(e,$a,r),i(e,st,r),p(st,Eo),i(e,wa,r),P(Oe,e,r),i(e,ga,r),i(e,lt,r),p(lt,ko),i(e,ba,r),i(e,rt,r),p(rt,To),i(e,ya,r),i(e,J,r),p(J,Ao),p(J,vt),p(vt,Vo),p(J,Lo),p(J,$t),p($t,So),p(J,Io),p(J,wt),p(wt,Po),p(J,Do),i(e,Ea,r),P(Be,e,r),i(e,ka,r),i(e,X,r),p(X,Co),p(X,gt),p(gt,Wo),p(X,Fo),p(X,bt),p(bt,No),p(X,Oo),p(X,yt),p(yt,Bo),p(X,zo),p(X,Et),p(Et,Ho),p(X,Mo),i(e,Ta,r),P(ze,e,r),i(e,Aa,r),P(He,e,r),i(e,Va,r),i(e,it,r),p(it,Ro),i(e,La,r),i(e,nt,r),Sa=!0},p(e,r){const kt={};r&16777216&&(kt.$$scope={dirty:r,ctx:e}),j.$set(kt);const Tt={};r&16777216&&(Tt.$$scope={dirty:r,ctx:e}),Y.$set(Tt);const Me={};r&16777216&&(Me.$$scope={dirty:r,ctx:e}),B.$set(Me);const Re={};r&16777216&&(Re.$$scope={dirty:r,ctx:e}),z.$set(Re);const je={};r&16777216&&(je.$$scope={dirty:r,ctx:e}),se.$set(je);const At={};r&16777216&&(At.$$scope={dirty:r,ctx:e}),re.$set(At);const Vt={};r&16777216&&(Vt.$$scope={dirty:r,ctx:e}),pe.$set(Vt)},i(e){Sa||(D(j.$$.fragment,e),D(Y.$$.fragment,e),D(B.$$.fragment,e),D(z.$$.fragment,e),D(se.$$.fragment,e),D(ve.$$.fragment,e),D(re.$$.fragment,e),D($e.$$.fragment,e),D(we.$$.fragment,e),D(ge.$$.fragment,e),D(be.$$.fragment,e),D(ye.$$.fragment,e),D(Ee.$$.fragment,e),D(ke.$$.fragment,e),D(Te.$$.fragment,e),D(Ae.$$.fragment,e),D(Ve.$$.fragment,e),D(Le.$$.fragment,e),D(Se.$$.fragment,e),D(Ie.$$.fragment,e),D(Pe.$$.fragment,e),D(De.$$.fragment,e),D(Ce.$$.fragment,e),D(Fe.$$.fragment,e),D(Ne.$$.fragment,e),D(pe.$$.fragment,e),D(Oe.$$.fragment,e),D(Be.$$.fragment,e),D(ze.$$.fragment,e),D(He.$$.fragment,e),Sa=!0)},o(e){C(j.$$.fragment,e),C(Y.$$.fragment,e),C(B.$$.fragment,e),C(z.$$.fragment,e),C(se.$$.fragment,e),C(ve.$$.fragment,e),C(re.$$.fragment,e),C($e.$$.fragment,e),C(we.$$.fragment,e),C(ge.$$.fragment,e),C(be.$$.fragment,e),C(ye.$$.fragment,e),C(Ee.$$.fragment,e),C(ke.$$.fragment,e),C(Te.$$.fragment,e),C(Ae.$$.fragment,e),C(Ve.$$.fragment,e),C(Le.$$.fragment,e),C(Se.$$.fragment,e),C(Ie.$$.fragment,e),C(Pe.$$.fragment,e),C(De.$$.fragment,e),C(Ce.$$.fragment,e),C(Fe.$$.fragment,e),C(Ne.$$.fragment,e),C(pe.$$.fragment,e),C(Oe.$$.fragment,e),C(Be.$$.fragment,e),C(ze.$$.fragment,e),C(He.$$.fragment,e),Sa=!1},d(e){e&&a(t),e&&a(s),e&&a(n),e&&a(d),e&&a(h),e&&a(m),e&&a(k),e&&a(V),e&&a(L),W(j),e&&a(F),e&&a(G),W(Y),e&&a(Q),W(B,e),e&&a(w),e&&a(N),W(z),e&&a(St),e&&a(qe),e&&a(It),W(se,e),e&&a(Pt),e&&a(Ue),e&&a(Dt),e&&a(Ke),e&&a(Ct),W(ve,e),e&&a(Wt),e&&a(Ze),e&&a(Ft),e&&a(le),W(re),e&&a(Nt),W($e,e),e&&a(Ot),e&&a(xe),e&&a(Bt),W(we,e),e&&a(zt),e&&a(ie),e&&a(Ht),W(ge,e),e&&a(Mt),e&&a(ne),e&&a(Rt),W(be,e),e&&a(jt),e&&a(fe),e&&a(qt),W(ye,e),e&&a(Ut),e&&a(Ge),e&&a(Kt),W(Ee,e),e&&a(Zt),e&&a(Ye),e&&a(xt),W(ke,e),e&&a(Gt),e&&a(he),e&&a(Yt),W(Te,e),e&&a(Xt),e&&a(ue),e&&a(Qt),W(Ae,e),e&&a(Jt),e&&a(ee),e&&a(ea),W(Ve,e),e&&a(ta),e&&a(Xe),e&&a(aa),W(Le,e),e&&a(oa),W(Se,e),e&&a(sa),e&&a(Qe),e&&a(la),W(Ie,e),e&&a(ra),W(Pe,e),e&&a(ia),e&&a(Je),e&&a(na),W(De,e),e&&a(fa),e&&a(et),e&&a(ha),W(Ce,e),e&&a(ua),e&&a(We),e&&a(da),e&&a(tt),e&&a(pa),W(Fe,e),e&&a(ca),W(Ne,e),e&&a(ma),e&&a(at),e&&a(_a),e&&a(ot),e&&a(va),e&&a(de),W(pe),e&&a($a),e&&a(st),e&&a(wa),W(Oe,e),e&&a(ga),e&&a(lt),e&&a(ba),e&&a(rt),e&&a(ya),e&&a(J),e&&a(Ea),W(Be,e),e&&a(ka),e&&a(X),e&&a(Ta),W(ze,e),e&&a(Aa),W(He,e),e&&a(Va),e&&a(it),e&&a(La),e&&a(nt)}}}function _l(o){let t,l,s,n,d,h,f,m,k;return m=new Zs({props:{$$slots:{default:[ml]},$$scope:{ctx:o}}}),{c(){t=T("meta"),l=y(),s=T("h1"),n=v("Train, Test, Validate"),d=y(),h=T("div"),f=y(),S(m.$$.fragment),this.h()},l(c){const V=Us("svelte-3baip0",document.head);t=A(V,"META",{name:!0,content:!0}),V.forEach(a),l=E(c),s=A(c,"H1",{});var L=g(s);n=$(L,"Train, Test, Validate"),L.forEach(a),d=E(c),h=A(c,"DIV",{class:!0}),g(h).forEach(a),f=E(c),I(m.$$.fragment,c),this.h()},h(){document.title="Train, Test, Validate - World4AI",u(t,"name","description"),u(t,"content","In order to measure the level of overfitting we need to split the dataset into the trainig, the validation and the test sets. The training dataset is used for backpropagation and gradient descent, the validation dataset is used to measure the generalization of the model and the test dataset is the final measure of performance that can only be used once."),u(h,"class","separator")},m(c,V){p(document.head,t),i(c,l,V),i(c,s,V),p(s,n),i(c,d,V),i(c,h,V),i(c,f,V),P(m,c,V),k=!0},p(c,[V]){const L={};V&16777216&&(L.$$scope={dirty:V,ctx:c}),m.$set(L)},i(c){k||(D(m.$$.fragment,c),k=!0)},o(c){C(m.$$.fragment,c),k=!1},d(c){a(t),c&&a(l),c&&a(s),c&&a(d),c&&a(h),c&&a(f),W(m,c)}}}function vl(o){let t=[],l=[],s=1,n=1;for(let _=0;_<100;_++){let z=_,H=s;t.push({x:z,y:H}),H=n,l.push({x:z,y:H}),s*=.94,_<=30?n*=.95:_<=40?n*=.96:_<=50?n*=.97:_<=60?n*=.98:_<=65?n*=1:_<=70?n*=1.01:n*=1.03}return[t,l,`import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision import transforms as T
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt`,`train_validation_dataset = MNIST(root="../datasets/", train=True, download=True, transform=T.ToTensor())
test_dataset = MNIST(root="../datasets", train=False, download=False, transform=T.ToTensor())`,`stratify = train_validation_dataset.targets.numpy()
train_idxs, val_idxs = train_test_split(
                                range(len(train_validation_dataset)),
                                stratify=stratify,
                                test_size=0.1)`,`train_dataset = Subset(train_validation_dataset, train_idxs)
val_dataset = Subset(train_validation_dataset, val_idxs)`,`# parameters
DEVICE = ("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS=50
BATCH_SIZE=32
HIDDEN_SIZE_1 = 100
HIDDEN_SIZE_2 = 50
NUM_LABELS = 10
NUM_FEATURES = 28*28
ALPHA = 0.1`,`train_dataloader = DataLoader(dataset=train_dataset, 
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              drop_last=True,
                              num_workers=4)

val_dataloader = DataLoader(dataset=val_dataset, 
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              drop_last=False,
                              num_workers=4)

test_dataloader = DataLoader(dataset=test_dataset, 
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              drop_last=False,
                              num_workers=4)`,`def track_performance(dataloader, model, criterion):
    # switch to evaluation mode
    num_samples = 0
    num_correct = 0
    loss_sum = 0
    
    # no need to calculate gradients
    with torch.inference_mode():
        for batch_idx, (features, labels) in enumerate(dataloader):
            features = features.view(-1, NUM_FEATURES).to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(features)
            
            predictions = logits.max(dim=1)[1]
            num_correct += (predictions == labels).sum().item()
            
            loss = criterion(logits, labels)
            loss_sum += loss.cpu().item()
            num_samples += len(features)
    
    # we return the average loss and the accuracy
    return loss_sum/num_samples, num_correct/num_samples`,`def train_epoch(dataloader, model, criterion, optimizer):
    for batch_idx, (features, labels) in enumerate(train_dataloader):
        # move features and labels to GPU
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        # ------ FORWARD PASS --------
        output = model(features)

        # ------CALCULATE LOSS --------
        loss = criterion(output, labels)

        # ------BACKPROPAGATION --------
        loss.backward()

        # ------GRADIENT DESCENT --------
        optimizer.step()

        # ------CLEAR GRADIENTS --------
        optimizer.zero_grad()`,`def train(epochs, train_dataloader, val_dataloader, model, criterion, optimizer):
    # track progress over time
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        train_epoch(train_dataloader, model, criterion, optimizer)
        
        # ------TRACK LOSS and ACCURACY --------
        train_loss, train_acc = track_performance(train_dataloader, model, criterion)
        val_loss, val_acc = track_performance(val_dataloader, model, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch: {epoch+1}/{epochs}|' 
                  f'Train Loss: {train_loss:.4f} |' 
                  f'Val Loss: {val_loss:.4f} |' 
                  f'Train Acc: {train_acc:.4f} |' 
                  f'Val Acc: {val_acc:.4f}')
    return history`,`class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(NUM_FEATURES, HIDDEN_SIZE_1),
                nn.Sigmoid(),
                nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
                nn.Sigmoid(),
                nn.Linear(HIDDEN_SIZE_2, NUM_LABELS),
            )
    
    def forward(self, features):
        return self.layers(features)`,`model = Model().to(DEVICE)
criterion = nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.SGD(model.parameters(), lr=ALPHA)`,"history = train(NUM_EPOCHS, train_dataloader, val_dataloader, model, criterion, optimizer)",`Epoch: 1/50|Train Loss: 0.1983 |Val Loss: 0.2199 |Train Acc: 0.9451 |Val Acc: 0.9372
Epoch: 10/50|Train Loss: 0.0416 |Val Loss: 0.1211 |Train Acc: 0.9871 |Val Acc: 0.9677
Epoch: 20/50|Train Loss: 0.0162 |Val Loss: 0.1297 |Train Acc: 0.9951 |Val Acc: 0.9713
Epoch: 30/50|Train Loss: 0.0141 |Val Loss: 0.1429 |Train Acc: 0.9953 |Val Acc: 0.9727
Epoch: 40/50|Train Loss: 0.0005 |Val Loss: 0.1363 |Train Acc: 1.0000 |Val Acc: 0.9780
Epoch: 50/50|Train Loss: 0.0003 |Val Loss: 0.1428 |Train Acc: 1.0000 |Val Acc: 0.9787
`,`def plot_history(history):
    fig = plt.figure(figsize=(12, 5))

    fig.add_subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy")
    plt.legend()
    
    fig.add_subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Training Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('overfitting.png')
    plt.show()`,"plot_history(history)",`test_loss, test_acc = track_performance(test_dataloader, model, criterion)
print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')`,"Test Loss: 0.1349 | Test Acc: 0.9768",`from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler`,`epochs = 5
seed = 42
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
for i, (train_index, val_index) in enumerate(kf.split(train_validation_dataset)):
    
    train_subsetsampler = SubsetRandomSampler(train_index)
    val_subsetsampler = SubsetRandomSampler(val_index)
    
    train_dataloader = DataLoader(train_validation_dataset, 
                                  batch_size=BATCH_SIZE, 
                                  sampler=train_subsetsampler)
    val_dataloader = DataLoader(train_validation_dataset, 
                              batch_size=BATCH_SIZE, 
                              sampler=val_subsetsampler)
    torch.manual_seed(seed)
    model = Model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.SGD(model.parameters(), lr=ALPHA)
    
    print('-'*50)
    print(f'Fold {i+1}')
    for epoch in range(epochs):
        train_epoch(train_dataloader, model, criterion, optimizer)
        val_loss, val_acc = track_performance(val_dataloader, model, criterion)
        print(f'Epoch: {epoch+1}/{epochs} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')`,`--------------------------------------------------
Fold 1
Epoch: 1/5 | Val Loss: 0.2120 | Val Acc: 0.9386
Epoch: 2/5 | Val Loss: 0.1681 | Val Acc: 0.9497
Epoch: 3/5 | Val Loss: 0.1360 | Val Acc: 0.9594
Epoch: 4/5 | Val Loss: 0.1169 | Val Acc: 0.9673
Epoch: 5/5 | Val Loss: 0.1295 | Val Acc: 0.9630
--------------------------------------------------
Fold 2
Epoch: 1/5 | Val Loss: 0.1770 | Val Acc: 0.9465
Epoch: 2/5 | Val Loss: 0.1451 | Val Acc: 0.9577
Epoch: 3/5 | Val Loss: 0.1387 | Val Acc: 0.9578
Epoch: 4/5 | Val Loss: 0.1375 | Val Acc: 0.9598
Epoch: 5/5 | Val Loss: 0.1140 | Val Acc: 0.9676
--------------------------------------------------
Fold 3
Epoch: 1/5 | Val Loss: 0.2109 | Val Acc: 0.9397
Epoch: 2/5 | Val Loss: 0.2501 | Val Acc: 0.9206
Epoch: 3/5 | Val Loss: 0.1246 | Val Acc: 0.9616
Epoch: 4/5 | Val Loss: 0.1163 | Val Acc: 0.9650
Epoch: 5/5 | Val Loss: 0.1192 | Val Acc: 0.9650
--------------------------------------------------
Fold 4
Epoch: 1/5 | Val Loss: 0.2243 | Val Acc: 0.9316
Epoch: 2/5 | Val Loss: 0.1793 | Val Acc: 0.9463
Epoch: 3/5 | Val Loss: 0.1270 | Val Acc: 0.9630
Epoch: 4/5 | Val Loss: 0.1172 | Val Acc: 0.9660
Epoch: 5/5 | Val Loss: 0.1158 | Val Acc: 0.9666
--------------------------------------------------
Fold 5
Epoch: 1/5 | Val Loss: 0.1983 | Val Acc: 0.9386
Epoch: 2/5 | Val Loss: 0.1628 | Val Acc: 0.9517
Epoch: 3/5 | Val Loss: 0.1320 | Val Acc: 0.9601
Epoch: 4/5 | Val Loss: 0.1161 | Val Acc: 0.9654
Epoch: 5/5 | Val Loss: 0.1173 | Val Acc: 0.9666
`]}class Sl extends qo{constructor(t){super(),Uo(this,t,vl,_l,Ko,{})}}export{Sl as default};
