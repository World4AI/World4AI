import{S as In,i as An,s as zn,y as P,z as I,A,g as z,d as S,B as N,Q as Z,R as ee,m as k,h as s,n as g,b as f,P as G,e as j,a as v,c as y,V as ql,C as qe,N as w,q as c,r as p,u as ca,k as T,W as pa,l as E,L as ga}from"../chunks/index.4d92b023.js";import{C as gl}from"../chunks/Container.b0705c7b.js";import{H as ue}from"../chunks/Highlight.b7c1de53.js";import{N as da}from"../chunks/NeuralNetwork.9b1e2957.js";import{S as wl}from"../chunks/SvgContainer.f70b5745.js";import{L as Bt}from"../chunks/Latex.e0b308c0.js";import{A as Cl}from"../chunks/Alert.25a852b3.js";import{P as dl}from"../chunks/PythonCode.212ba7a6.js";import{B as ma}from"../chunks/ButtonContainer.e9aac418.js";import{S as wa}from"../chunks/StepButton.2fb0289b.js";import{t as Ll}from"../chunks/index.4de27e87.js";import{C as he}from"../chunks/Convolution.b73a3805.js";import{P as $a}from"../chunks/PlayButton.85103c5a.js";function Hl(r,n,a){const t=r.slice();return t[5]=n[a],t[7]=a,t}function Bl(r,n,a){const t=r.slice();return t[8]=n[a],t[10]=a,t}function Ml(r,n,a){const t=r.slice();return t[8]=n[a],t[12]=a,t}function Dl(r,n,a){const t=r.slice();return t[8]=n[a],t[14]=a,t}function Ol(r){let n,a,t;return{c(){n=Z("rect"),this.h()},l(i){n=ee(i,"rect",{x:!0,y:!0,width:!0,height:!0,fill:!0,stroke:!0}),k(n).forEach(s),this.h()},h(){g(n,"x",a=r[14]*(r[2]+r[3])),g(n,"y",t=r[12]*(r[2]+r[3])),g(n,"width",r[2]),g(n,"height",r[2]),g(n,"fill","white"),g(n,"stroke","black")},m(i,o){f(i,n,o)},p(i,o){o&12&&a!==(a=i[14]*(i[2]+i[3]))&&g(n,"x",a),o&12&&t!==(t=i[12]*(i[2]+i[3]))&&g(n,"y",t),o&4&&g(n,"width",i[2]),o&4&&g(n,"height",i[2])},d(i){i&&s(n)}}}function Ul(r){let n,a=Array(r[5].width),t=[];for(let i=0;i<a.length;i+=1)t[i]=Ol(Dl(r,a,i));return{c(){for(let i=0;i<t.length;i+=1)t[i].c();n=j()},l(i){for(let o=0;o<t.length;o+=1)t[o].l(i);n=j()},m(i,o){for(let l=0;l<t.length;l+=1)t[l]&&t[l].m(i,o);f(i,n,o)},p(i,o){if(o&13){a=Array(i[5].width);let l;for(l=0;l<a.length;l+=1){const u=Dl(i,a,l);t[l]?t[l].p(u,o):(t[l]=Ol(u),t[l].c(),t[l].m(n.parentNode,n))}for(;l<t.length;l+=1)t[l].d(1);t.length=a.length}},d(i){G(t,i),i&&s(n)}}}function Rl(r){let n,a=Array(r[5].height),t=[];for(let i=0;i<a.length;i+=1)t[i]=Ul(Ml(r,a,i));return{c(){n=Z("g");for(let i=0;i<t.length;i+=1)t[i].c();this.h()},l(i){n=ee(i,"g",{transform:!0});var o=k(n);for(let l=0;l<t.length;l+=1)t[l].l(o);o.forEach(s),this.h()},h(){g(n,"transform","translate("+r[10]*5+", "+r[10]*5+")")},m(i,o){f(i,n,o);for(let l=0;l<t.length;l+=1)t[l]&&t[l].m(n,null)},p(i,o){if(o&13){a=Array(i[5].height);let l;for(l=0;l<a.length;l+=1){const u=Ml(i,a,l);t[l]?t[l].p(u,o):(t[l]=Ul(u),t[l].c(),t[l].m(n,null))}for(;l<t.length;l+=1)t[l].d(1);t.length=a.length}},d(i){i&&s(n),G(t,i)}}}function Fl(r){let n,a,t=Array(r[5].channels),i=[];for(let o=0;o<t.length;o+=1)i[o]=Rl(Bl(r,t,o));return{c(){n=Z("g");for(let o=0;o<i.length;o+=1)i[o].c();this.h()},l(o){n=ee(o,"g",{transform:!0});var l=k(n);for(let u=0;u<i.length;u+=1)i[u].l(l);l.forEach(s),this.h()},h(){g(n,"transform",a="translate("+r[7]*r[4]+", 0)")},m(o,l){f(o,n,l);for(let u=0;u<i.length;u+=1)i[u]&&i[u].m(n,null)},p(o,l){if(l&13){t=Array(o[5].channels);let u;for(u=0;u<t.length;u+=1){const $=Bl(o,t,u);i[u]?i[u].p($,l):(i[u]=Rl($),i[u].c(),i[u].m(n,null))}for(;u<i.length;u+=1)i[u].d(1);i.length=t.length}l&16&&a!==(a="translate("+o[7]*o[4]+", 0)")&&g(n,"transform",a)},d(o){o&&s(n),G(i,o)}}}function _a(r){let n,a=r[0],t=[];for(let i=0;i<a.length;i+=1)t[i]=Fl(Hl(r,a,i));return{c(){n=Z("svg");for(let i=0;i<t.length;i+=1)t[i].c();this.h()},l(i){n=ee(i,"svg",{viewBox:!0});var o=k(n);for(let l=0;l<t.length;l+=1)t[l].l(o);o.forEach(s),this.h()},h(){g(n,"viewBox","0 0 "+ba+" "+ya)},m(i,o){f(i,n,o);for(let l=0;l<t.length;l+=1)t[l]&&t[l].m(n,null)},p(i,o){if(o&29){a=i[0];let l;for(l=0;l<a.length;l+=1){const u=Hl(i,a,l);t[l]?t[l].p(u,o):(t[l]=Fl(u),t[l].c(),t[l].m(n,null))}for(;l<t.length;l+=1)t[l].d(1);t.length=a.length}},d(i){i&&s(n),G(t,i)}}}function va(r){let n,a;return n=new wl({props:{maxWidth:r[1]+"px",$$slots:{default:[_a]},$$scope:{ctx:r}}}),{c(){P(n.$$.fragment)},l(t){I(n.$$.fragment,t)},m(t,i){A(n,t,i),a=!0},p(t,[i]){const o={};i&2&&(o.maxWidth=t[1]+"px"),i&32797&&(o.$$scope={dirty:i,ctx:t}),n.$set(o)},i(t){a||(z(n.$$.fragment,t),a=!0)},o(t){S(n.$$.fragment,t),a=!1},d(t){N(n,t)}}}let ya=400,ba=1600;function ka(r,n,a){let{layers:t=[{width:10,height:10,channels:3},{width:8,height:8,channels:8},{width:6,height:6,channels:16},{width:4,height:4,channels:32},{width:1,height:1,channels:64}]}=n,{maxWidth:i=1e3}=n,{blockSize:o=20}=n,{gap:l=5}=n,{layerDistance:u=300}=n;return r.$$set=$=>{"layers"in $&&a(0,t=$.layers),"maxWidth"in $&&a(1,i=$.maxWidth),"blockSize"in $&&a(2,o=$.blockSize),"gap"in $&&a(3,l=$.gap),"layerDistance"in $&&a(4,u=$.layerDistance)},[t,i,o,l,u]}class Ta extends In{constructor(n){super(),An(this,n,ka,va,zn,{layers:0,maxWidth:1,blockSize:2,gap:3,layerDistance:4})}}function Yl(r,n,a){const t=r.slice();return t[9]=n[a],t[11]=a,t}function jl(r,n,a){const t=r.slice();return t[12]=n[a],t[14]=a,t}function Ea(r){let n,a;return n=new wa({}),n.$on("click",r[7]),{c(){P(n.$$.fragment)},l(t){I(n.$$.fragment,t)},m(t,i){A(n,t,i),a=!0},p:qe,i(t){a||(z(n.$$.fragment,t),a=!0)},o(t){S(n.$$.fragment,t),a=!1},d(t){N(n,t)}}}function Vl(r){let n,a,t,i;return{c(){n=Z("rect"),this.h()},l(o){n=ee(o,"rect",{x:!0,y:!0,width:!0,height:!0,fill:!0,stroke:!0}),k(n).forEach(s),this.h()},h(){g(n,"x",a=r[14]*(r[1]+r[2])),g(n,"y",t=r[11]*(r[1]+r[2])),g(n,"width",r[1]),g(n,"height",r[1]),g(n,"fill",i=r[12]===1?"black":"white"),g(n,"stroke","black")},m(o,l){f(o,n,l)},p(o,l){l&6&&a!==(a=o[14]*(o[1]+o[2]))&&g(n,"x",a),l&6&&t!==(t=o[11]*(o[1]+o[2]))&&g(n,"y",t),l&2&&g(n,"width",o[1]),l&2&&g(n,"height",o[1]),l&1&&i!==(i=o[12]===1?"black":"white")&&g(n,"fill",i)},d(o){o&&s(n)}}}function Gl(r){let n,a,t=r[9],i=[];for(let o=0;o<t.length;o+=1)i[o]=Vl(jl(r,t,o));return{c(){n=Z("g");for(let o=0;o<i.length;o+=1)i[o].c();this.h()},l(o){n=ee(o,"g",{transform:!0});var l=k(n);for(let u=0;u<i.length;u+=1)i[u].l(l);l.forEach(s),this.h()},h(){g(n,"transform",a="translate("+r[3]*r[11]*r[9].length*(r[1]+r[2])+", "+-r[4]*r[11]*(r[1]+r[2])+")")},m(o,l){f(o,n,l);for(let u=0;u<i.length;u+=1)i[u]&&i[u].m(n,null)},p(o,l){if(l&7){t=o[9];let u;for(u=0;u<t.length;u+=1){const $=jl(o,t,u);i[u]?i[u].p($,l):(i[u]=Vl($),i[u].c(),i[u].m(n,null))}for(;u<i.length;u+=1)i[u].d(1);i.length=t.length}l&31&&a!==(a="translate("+o[3]*o[11]*o[9].length*(o[1]+o[2])+", "+-o[4]*o[11]*(o[1]+o[2])+")")&&g(n,"transform",a)},d(o){o&&s(n),G(i,o)}}}function Wa(r){let n,a=r[0],t=[];for(let i=0;i<a.length;i+=1)t[i]=Gl(Yl(r,a,i));return{c(){n=Z("svg");for(let i=0;i<t.length;i+=1)t[i].c();this.h()},l(i){n=ee(i,"svg",{viewBox:!0});var o=k(n);for(let l=0;l<t.length;l+=1)t[l].l(o);o.forEach(s),this.h()},h(){g(n,"viewBox","0 0 "+Ia+" "+Pa)},m(i,o){f(i,n,o);for(let l=0;l<t.length;l+=1)t[l]&&t[l].m(n,null)},p(i,o){if(o&31){a=i[0];let l;for(l=0;l<a.length;l+=1){const u=Yl(i,a,l);t[l]?t[l].p(u,o):(t[l]=Gl(u),t[l].c(),t[l].m(n,null))}for(;l<t.length;l+=1)t[l].d(1);t.length=a.length}},d(i){i&&s(n),G(t,i)}}}function xa(r){let n,a,t,i;return n=new ma({props:{$$slots:{default:[Ea]},$$scope:{ctx:r}}}),t=new wl({props:{maxWidth:"1000px",$$slots:{default:[Wa]},$$scope:{ctx:r}}}),{c(){P(n.$$.fragment),a=v(),P(t.$$.fragment)},l(o){I(n.$$.fragment,o),a=y(o),I(t.$$.fragment,o)},m(o,l){A(n,o,l),f(o,a,l),A(t,o,l),i=!0},p(o,[l]){const u={};l&32768&&(u.$$scope={dirty:l,ctx:o}),n.$set(u);const $={};l&32799&&($.$$scope={dirty:l,ctx:o}),t.$set($)},i(o){i||(z(n.$$.fragment,o),z(t.$$.fragment,o),i=!0)},o(o){S(n.$$.fragment,o),S(t.$$.fragment,o),i=!1},d(o){N(n,o),o&&s(a),N(t,o)}}}let Pa=200,Ia=1e3;function Aa(r,n,a){let t,i,{image:o=[[0,0,0,0,0],[0,1,1,1,0],[0,0,0,1,0],[0,1,1,1,0],[0,1,0,0,0],[0,1,1,1,0],[0,0,0,0,0]]}=n,{blockSize:l=23}=n,{gap:u=5}=n;const $=Ll(0,{duration:400});ql(r,$,W=>a(3,t=W));const L=Ll(0,{duration:400});ql(r,L,W=>a(4,i=W));let x=!1;async function q(){x?(await L.set(0),await $.set(0),x=!1):(await $.set(1),await L.set(1),x=!0)}return r.$$set=W=>{"image"in W&&a(0,o=W.image),"blockSize"in W&&a(1,l=W.blockSize),"gap"in W&&a(2,u=W.gap)},[o,l,u,t,i,$,L,q]}class Xl extends In{constructor(n){super(),An(this,n,Aa,xa,zn,{image:0,blockSize:1,gap:2})}}function Ql(r,n,a){const t=r.slice();return t[16]=n[a],t[18]=a,t}function Jl(r,n,a){const t=r.slice();return t[19]=n[a],t[21]=a,t}function Kl(r,n,a){const t=r.slice();return t[16]=n[a],t[18]=a,t}function Zl(r,n,a){const t=r.slice();return t[19]=n[a],t[24]=a,t}function ea(r,n,a){const t=r.slice();return t[25]=n[a],t[18]=a,t}function ta(r,n,a){const t=r.slice();return t[25]=n[a],t[21]=a,t}function na(r,n,a){const t=r.slice();return t[25]=n[a],t[18]=a,t}function la(r,n,a){const t=r.slice();return t[25]=n[a],t[21]=a,t}function za(r){let n,a;return n=new $a({props:{f:r[14],delta:500}}),{c(){P(n.$$.fragment)},l(t){I(n.$$.fragment,t)},m(t,i){A(n,t,i),a=!0},p:qe,i(t){a||(z(n.$$.fragment,t),a=!0)},o(t){S(n.$$.fragment,t),a=!1},d(t){N(n,t)}}}function aa(r){let n,a,t,i;return{c(){n=Z("rect"),this.h()},l(o){n=ee(o,"rect",{x:!0,y:!0,width:!0,height:!0,class:!0,stroke:!0}),k(n).forEach(s),this.h()},h(){g(n,"x",a=1+r[21]*(r[4]+r[5])),g(n,"y",t=1+r[18]*(r[4]+r[5])),g(n,"width",r[4]),g(n,"height",r[4]),g(n,"class",i=`stroke ${r[21]<r[0].column+oe&&r[18]<r[0].row+oe&&r[21]>=r[0].column&&r[18]>=r[0].row?"fill-lime-200":"fill-slate-400"}`),g(n,"stroke","black")},m(o,l){f(o,n,l)},p(o,l){l&48&&a!==(a=1+o[21]*(o[4]+o[5]))&&g(n,"x",a),l&48&&t!==(t=1+o[18]*(o[4]+o[5]))&&g(n,"y",t),l&16&&g(n,"width",o[4]),l&16&&g(n,"height",o[4]),l&1&&i!==(i=`stroke ${o[21]<o[0].column+oe&&o[18]<o[0].row+oe&&o[21]>=o[0].column&&o[18]>=o[0].row?"fill-lime-200":"fill-slate-400"}`)&&g(n,"class",i)},d(o){o&&s(n)}}}function ia(r){let n,a=Array(r[1]),t=[];for(let i=0;i<a.length;i+=1)t[i]=aa(la(r,a,i));return{c(){for(let i=0;i<t.length;i+=1)t[i].c();n=j()},l(i){for(let o=0;o<t.length;o+=1)t[o].l(i);n=j()},m(i,o){for(let l=0;l<t.length;l+=1)t[l]&&t[l].m(i,o);f(i,n,o)},p(i,o){if(o&51){a=Array(i[1]);let l;for(l=0;l<a.length;l+=1){const u=la(i,a,l);t[l]?t[l].p(u,o):(t[l]=aa(u),t[l].c(),t[l].m(n.parentNode,n))}for(;l<t.length;l+=1)t[l].d(1);t.length=a.length}},d(i){G(t,i),i&&s(n)}}}function oa(r){let n,a,t,i;return{c(){n=Z("rect"),this.h()},l(o){n=ee(o,"rect",{x:!0,y:!0,width:!0,height:!0,class:!0}),k(n).forEach(s),this.h()},h(){g(n,"x",a=r[1]*(r[4]+r[5])+bt+r[21]*(r[4]+r[5])),g(n,"y",t=r[11]+r[18]*(r[4]+r[5])),g(n,"width",r[4]),g(n,"height",r[4]),g(n,"class",i=`stroke-black ${r[0].row/r[8]==r[18]&&r[0].column/r[8]==r[21]?"fill-w4ai-red":"fill-w4ai-lightblue"}`)},m(o,l){f(o,n,l)},p(o,l){l&50&&a!==(a=o[1]*(o[4]+o[5])+bt+o[21]*(o[4]+o[5]))&&g(n,"x",a),l&48&&t!==(t=o[11]+o[18]*(o[4]+o[5]))&&g(n,"y",t),l&16&&g(n,"width",o[4]),l&16&&g(n,"height",o[4]),l&1&&i!==(i=`stroke-black ${o[0].row/o[8]==o[18]&&o[0].column/o[8]==o[21]?"fill-w4ai-red":"fill-w4ai-lightblue"}`)&&g(n,"class",i)},d(o){o&&s(n)}}}function ra(r){let n,a=Array(r[9]),t=[];for(let i=0;i<a.length;i+=1)t[i]=oa(ta(r,a,i));return{c(){for(let i=0;i<t.length;i+=1)t[i].c();n=j()},l(i){for(let o=0;o<t.length;o+=1)t[o].l(i);n=j()},m(i,o){for(let l=0;l<t.length;l+=1)t[l]&&t[l].m(i,o);f(i,n,o)},p(i,o){if(o&2355){a=Array(i[9]);let l;for(l=0;l<a.length;l+=1){const u=ta(i,a,l);t[l]?t[l].p(u,o):(t[l]=oa(u),t[l].c(),t[l].m(n.parentNode,n))}for(;l<t.length;l+=1)t[l].d(1);t.length=a.length}},d(i){G(t,i),i&&s(n)}}}function sa(r){let n,a=r[19]+"",t,i,o;return{c(){n=Z("text"),t=c(a),this.h()},l(l){n=ee(l,"text",{fill:!0,x:!0,y:!0,class:!0});var u=k(n);t=p(u,a),u.forEach(s),this.h()},h(){g(n,"fill","black"),g(n,"x",i=1+r[24]*(r[4]+r[5])+r[4]/2),g(n,"y",o=1+r[18]*(r[4]+r[5])+r[4]/2),g(n,"class","svelte-15l9vb6")},m(l,u){f(l,n,u),w(n,t)},p(l,u){u&64&&a!==(a=l[19]+"")&&ca(t,a),u&48&&i!==(i=1+l[24]*(l[4]+l[5])+l[4]/2)&&g(n,"x",i),u&48&&o!==(o=1+l[18]*(l[4]+l[5])+l[4]/2)&&g(n,"y",o)},d(l){l&&s(n)}}}function fa(r){let n,a=r[16],t=[];for(let i=0;i<a.length;i+=1)t[i]=sa(Zl(r,a,i));return{c(){for(let i=0;i<t.length;i+=1)t[i].c();n=j()},l(i){for(let o=0;o<t.length;o+=1)t[o].l(i);n=j()},m(i,o){for(let l=0;l<t.length;l+=1)t[l]&&t[l].m(i,o);f(i,n,o)},p(i,o){if(o&112){a=i[16];let l;for(l=0;l<a.length;l+=1){const u=Zl(i,a,l);t[l]?t[l].p(u,o):(t[l]=sa(u),t[l].c(),t[l].m(n.parentNode,n))}for(;l<t.length;l+=1)t[l].d(1);t.length=a.length}},d(i){G(t,i),i&&s(n)}}}function ha(r){let n,a=r[19]+"",t,i,o;return{c(){n=Z("text"),t=c(a),this.h()},l(l){n=ee(l,"text",{x:!0,y:!0,fill:!0,class:!0});var u=k(n);t=p(u,a),u.forEach(s),this.h()},h(){g(n,"x",i=r[1]*(r[4]+r[5])+bt+r[21]*(r[4]+r[5])+r[4]/2),g(n,"y",o=r[11]+r[18]*(r[4]+r[5])+r[4]/2),g(n,"fill","black"),g(n,"class","svelte-15l9vb6")},m(l,u){f(l,n,u),w(n,t)},p(l,u){u&50&&i!==(i=l[1]*(l[4]+l[5])+bt+l[21]*(l[4]+l[5])+l[4]/2)&&g(n,"x",i),u&48&&o!==(o=l[11]+l[18]*(l[4]+l[5])+l[4]/2)&&g(n,"y",o)},d(l){l&&s(n)}}}function ua(r){let n,a=r[16],t=[];for(let i=0;i<a.length;i+=1)t[i]=ha(Jl(r,a,i));return{c(){for(let i=0;i<t.length;i+=1)t[i].c();n=j()},l(i){for(let o=0;o<t.length;o+=1)t[o].l(i);n=j()},m(i,o){for(let l=0;l<t.length;l+=1)t[l]&&t[l].m(i,o);f(i,n,o)},p(i,o){if(o&2226){a=i[16];let l;for(l=0;l<a.length;l+=1){const u=Jl(i,a,l);t[l]?t[l].p(u,o):(t[l]=ha(u),t[l].c(),t[l].m(n.parentNode,n))}for(;l<t.length;l+=1)t[l].d(1);t.length=a.length}},d(i){G(t,i),i&&s(n)}}}function Sa(r){let n,a,t,i,o=Array(r[2]),l=[];for(let _=0;_<o.length;_+=1)l[_]=ia(na(r,o,_));let u=Array(r[10]),$=[];for(let _=0;_<u.length;_+=1)$[_]=ra(ea(r,u,_));let L=r[6],x=[];for(let _=0;_<L.length;_+=1)x[_]=fa(Kl(r,L,_));let q=r[7],W=[];for(let _=0;_<q.length;_+=1)W[_]=ua(Ql(r,q,_));return{c(){n=Z("svg");for(let _=0;_<l.length;_+=1)l[_].c();a=j();for(let _=0;_<$.length;_+=1)$[_].c();t=j();for(let _=0;_<x.length;_+=1)x[_].c();i=j();for(let _=0;_<W.length;_+=1)W[_].c();this.h()},l(_){n=ee(_,"svg",{viewBox:!0});var d=k(n);for(let m=0;m<l.length;m+=1)l[m].l(d);a=j();for(let m=0;m<$.length;m+=1)$[m].l(d);t=j();for(let m=0;m<x.length;m+=1)x[m].l(d);i=j();for(let m=0;m<W.length;m+=1)W[m].l(d);d.forEach(s),this.h()},h(){g(n,"viewBox","0 0 "+(r[13]+2)+" "+(r[12]+2))},m(_,d){f(_,n,d);for(let m=0;m<l.length;m+=1)l[m]&&l[m].m(n,null);w(n,a);for(let m=0;m<$.length;m+=1)$[m]&&$[m].m(n,null);w(n,t);for(let m=0;m<x.length;m+=1)x[m]&&x[m].m(n,null);w(n,i);for(let m=0;m<W.length;m+=1)W[m]&&W[m].m(n,null)},p(_,d){if(d&55){o=Array(_[2]);let m;for(m=0;m<o.length;m+=1){const H=na(_,o,m);l[m]?l[m].p(H,d):(l[m]=ia(H),l[m].c(),l[m].m(n,a))}for(;m<l.length;m+=1)l[m].d(1);l.length=o.length}if(d&2867){u=Array(_[10]);let m;for(m=0;m<u.length;m+=1){const H=ea(_,u,m);$[m]?$[m].p(H,d):($[m]=ra(H),$[m].c(),$[m].m(n,t))}for(;m<$.length;m+=1)$[m].d(1);$.length=u.length}if(d&112){L=_[6];let m;for(m=0;m<L.length;m+=1){const H=Kl(_,L,m);x[m]?x[m].p(H,d):(x[m]=fa(H),x[m].c(),x[m].m(n,i))}for(;m<x.length;m+=1)x[m].d(1);x.length=L.length}if(d&2226){q=_[7];let m;for(m=0;m<q.length;m+=1){const H=Ql(_,q,m);W[m]?W[m].p(H,d):(W[m]=ua(H),W[m].c(),W[m].m(n,null))}for(;m<W.length;m+=1)W[m].d(1);W.length=q.length}},d(_){_&&s(n),G(l,_),G($,_),G(x,_),G(W,_)}}}function Na(r){let n,a,t,i;return n=new ma({props:{$$slots:{default:[za]},$$scope:{ctx:r}}}),t=new wl({props:{maxWidth:r[3]+"px",$$slots:{default:[Sa]},$$scope:{ctx:r}}}),{c(){P(n.$$.fragment),a=v(),P(t.$$.fragment)},l(o){I(n.$$.fragment,o),a=y(o),I(t.$$.fragment,o)},m(o,l){A(n,o,l),f(o,a,l),A(t,o,l),i=!0},p(o,[l]){const u={};l&1073741824&&(u.$$scope={dirty:l,ctx:o}),n.$set(u);const $={};l&8&&($.maxWidth=o[3]+"px"),l&1073741943&&($.$$scope={dirty:l,ctx:o}),t.$set($)},i(o){i||(z(n.$$.fragment,o),z(t.$$.fragment,o),i=!0)},o(o){S(n.$$.fragment,o),S(t.$$.fragment,o),i=!1},d(o){N(n,o),o&&s(a),N(t,o)}}}let oe=2;const bt=50;function qa(r,n,a){let{imageWidth:t=4}=n,{imageHeight:i=4}=n,{maxWidth:o=500}=n,{blockSize:l=20}=n,{gap:u=5}=n,{imageNumbers:$=[]}=n,L=[],{filterLocation:x={row:0,column:0}}=n,q=oe,W=Math.floor((t-oe)/q+1),_=Math.floor((i-oe)/q+1),d=(i-_)/2*(u+l);const m=bt+W*l+(W-1)*u;let H=i*l+(i-1)*u,O=t*l+(t-1)*u+m+bt;function U(){let C=x.row,M=x.column;M+=q,M+oe>t&&(M=0,C+=q),C+oe>i&&(C=0),a(0,x={row:C,column:M})}return $.forEach((C,M)=>{if(M%q===0){let te=[];C.forEach((R,re)=>{if(re%q===0){let Q=0;for(let D=0;D<oe;D++)for(let V=0;V<oe;V++){let F=$[M+D][re+V];F>Q&&(Q=F)}te.push(Q)}}),L.push(te)}}),r.$$set=C=>{"imageWidth"in C&&a(1,t=C.imageWidth),"imageHeight"in C&&a(2,i=C.imageHeight),"maxWidth"in C&&a(3,o=C.maxWidth),"blockSize"in C&&a(4,l=C.blockSize),"gap"in C&&a(5,u=C.gap),"imageNumbers"in C&&a(6,$=C.imageNumbers),"filterLocation"in C&&a(0,x=C.filterLocation)},[x,t,i,o,l,u,$,L,q,W,_,d,H,O,U]}class Ca extends In{constructor(n){super(),An(this,n,qa,Na,zn,{imageWidth:1,imageHeight:2,maxWidth:3,blockSize:4,gap:5,imageNumbers:6,filterLocation:0})}}const La=""+new URL("../assets/alex-holzreiter--unsplash.d10fccce.webp",import.meta.url).href;function Ha(r){let n;return{c(){n=c(`"Why is a fully connected neural network not the ideal tool for computer
      vision tasks?"`)},l(a){n=p(a,`"Why is a fully connected neural network not the ideal tool for computer
      vision tasks?"`)},m(a,t){f(a,n,t)},d(a){a&&s(n)}}}function Ba(r){let n;return{c(){n=c("convolution neural network")},l(a){n=p(a,"convolution neural network")},m(a,t){f(a,n,t)},d(a){a&&s(n)}}}function Ma(r){let n;return{c(){n=c("Pixels in a small region of an image are highly correlated.")},l(a){n=p(a,"Pixels in a small region of an image are highly correlated.")},m(a,t){f(a,n,t)},d(a){a&&s(n)}}}function Da(r){let n;return{c(){n=c(`The receptive field of a neuron describes the area of an image that a
    neuron has access to.`)},l(a){n=p(a,`The receptive field of a neuron describes the area of an image that a
    neuron has access to.`)},m(a,t){f(a,n,t)},d(a){a&&s(n)}}}function Oa(r){let n;return{c(){n=c("sparse connectivity")},l(a){n=p(a,"sparse connectivity")},m(a,t){f(a,n,t)},d(a){a&&s(n)}}}function Ua(r){let n;return{c(){n=c("stride")},l(a){n=p(a,"stride")},m(a,t){f(a,n,t)},d(a){a&&s(n)}}}function Ra(r){let n=String.raw`
  \begin{bmatrix}
     x_{11} & x_{12} \\
     x_{21} & x_{22}
  \end{bmatrix}
  `+"",a;return{c(){a=c(n)},l(t){a=p(t,n)},m(t,i){f(t,a,i)},p:qe,d(t){t&&s(a)}}}function Fa(r){let n=String.raw`2\times2`+"",a;return{c(){a=c(n)},l(t){a=p(t,n)},m(t,i){f(t,a,i)},p:qe,d(t){t&&s(a)}}}function Ya(r){let n=String.raw`
  \begin{bmatrix}
     w_{11} & w_{12} \\
     w_{21} & w_{22}
  \end{bmatrix}
  `+"",a;return{c(){a=c(n)},l(t){a=p(t,n)},m(t,i){f(t,a,i)},p:qe,d(t){t&&s(a)}}}function ja(r){let n;return{c(){n=c("filter")},l(a){n=p(a,"filter")},m(a,t){f(a,n,t)},d(a){a&&s(n)}}}function Va(r){let n;return{c(){n=c("kernel")},l(a){n=p(a,"kernel")},m(a,t){f(a,n,t)},d(a){a&&s(n)}}}function Ga(r){let n=String.raw`
  \begin{aligned}
  z &= x_{11}w_{11} + x_{12}w_{12} + x_{21}w_{21} + x_{22}w_{22} + b \\
  a &= f(z)
  \end{aligned}
  `+"",a;return{c(){a=c(n)},l(t){a=p(t,n)},m(t,i){f(t,a,i)},p:qe,d(t){t&&s(a)}}}function Xa(r){let n;return{c(){n=c("weight sharing")},l(a){n=p(a,"weight sharing")},m(a,t){f(a,n,t)},d(a){a&&s(n)}}}function Qa(r){let n=String.raw`
  \begin{bmatrix}
     w_{1} & w_{2} \\
     w_{3} & w_{4}
  \end{bmatrix}
  `+"",a;return{c(){a=c(n)},l(t){a=p(t,n)},m(t,i){f(t,a,i)},p:qe,d(t){t&&s(a)}}}function Ja(r){let n;return{c(){n=c("translation invariant")},l(a){n=p(a,"translation invariant")},m(a,t){f(a,n,t)},d(a){a&&s(n)}}}function Ka(r){let n;return{c(){n=c("feature map")},l(a){n=p(a,"feature map")},m(a,t){f(a,n,t)},d(a){a&&s(n)}}}function Za(r){let n,a,t,i,o,l,u,$,L,x,q,W,_,d,m,H,O,U,C,M,te,R,re,Q,D,V,F,ne,me,X,ge,le,se,ce,b,B,Y,ae,Ce,pe,de,Le,J,He,$l,Sn,Ze,kt,Nn,Mt,et,qn,Dt,we,Ot,tt,Cn,Ut,Be,Rt,$e,Ln,_e,Hn,Ft,nt,Bn,Yt,Me,jt,lt,Mn,Vt,De,Gt,ve,Dn,ye,On,Xt,Oe,Qt,at,Un,Jt,Ue,Kt,it,Rn,Zt,Re,en,ie,Fn,be,Yn,ke,jn,Te,tn,fe,Vn,Ee,Gn,We,Xn,nn,ot,Qn,ln,xe,an,K,Jn,Pe,Kn,Ie,Zn,Ae,el,on,ze,tl,Se,nl,rn,rt,ll,sn,Fe,fn,st,al,hn,Ye,un,ft,il,mn,je,cn,ht,ol,pn,Ve,gn,ut,dn,mt,rl,wn,ct,sl,$n,pt,fl,_n,gt,hl,vn,dt,ul,yn,Ge,bn,wt,ml,kn,$t,Tn,_t,cl,En,vt,pl,Wn;return o=new ue({props:{$$slots:{default:[Ha]},$$scope:{ctx:r}}}),x=new Xl({props:{image:r[1]}}),m=new Xl({props:{image:r[2]}}),M=new da({props:{layers:r[0],height:250,verticalGap:5,rectSize:9}}),F=new ue({props:{$$slots:{default:[Ba]},$$scope:{ctx:r}}}),ae=new Cl({props:{type:"info",$$slots:{default:[Ma]},$$scope:{ctx:r}}}),we=new Cl({props:{type:"info",$$slots:{default:[Da]},$$scope:{ctx:r}}}),Be=new he({props:{imageWidth:6,imageHeight:6,maxWidth:200,kernel:2}}),_e=new ue({props:{$$slots:{default:[Oa]},$$scope:{ctx:r}}}),Me=new he({props:{imageWidth:6,imageHeight:6,maxWidth:500,kernel:2,showOutput:!0}}),De=new he({props:{imageWidth:6,imageHeight:6,maxWidth:500,kernel:3,stride:1,showOutput:!0}}),ye=new ue({props:{$$slots:{default:[Ua]},$$scope:{ctx:r}}}),Oe=new he({props:{imageWidth:6,imageHeight:6,maxWidth:500,kernel:2,stride:2,showOutput:!0}}),Ue=new he({props:{maxWidth:650,imageWidth:6,imageHeight:6,kernel:3,stride:1,padding:1,showOutput:!0}}),Re=new he({props:{imageWidth:6,imageHeight:6,maxWidth:200,kernel:2}}),be=new Bt({props:{$$slots:{default:[Ra]},$$scope:{ctx:r}}}),ke=new Bt({props:{$$slots:{default:[Fa]},$$scope:{ctx:r}}}),Te=new Bt({props:{$$slots:{default:[Ya]},$$scope:{ctx:r}}}),Ee=new ue({props:{$$slots:{default:[ja]},$$scope:{ctx:r}}}),We=new ue({props:{$$slots:{default:[Va]},$$scope:{ctx:r}}}),xe=new Bt({props:{$$slots:{default:[Ga]},$$scope:{ctx:r}}}),Pe=new ue({props:{$$slots:{default:[Xa]},$$scope:{ctx:r}}}),Ie=new Bt({props:{$$slots:{default:[Qa]},$$scope:{ctx:r}}}),Ae=new ue({props:{$$slots:{default:[Ja]},$$scope:{ctx:r}}}),Se=new ue({props:{$$slots:{default:[Ka]},$$scope:{ctx:r}}}),Fe=new he({props:{imageWidth:5,imageHeight:5,maxWidth:500,kernel:3,imageNumbers:[[0,1,0,0,1],[1,1,0,0,1],[0,-1,0,0,1],[0,0,-1,-1,1],[0,1,0,-1,-1]],kernelNumbers:[[-1,0,1],[-1,1,1],[0,0,1]],showOutput:!0,showNumbers:!0}}),Ye=new he({props:{imageWidth:5,imageHeight:5,maxWidth:500,kernel:3,imageNumbers:[[0,1,0,0,1],[1,1,0,0,1],[0,-1,0,0,1],[0,0,-1,-1,1],[0,1,0,-1,-1]],kernelNumbers:[[1,0,-1],[1,1,-1],[-1,0,1]],showOutput:!0,showNumbers:!0}}),je=new he({props:{imageWidth:5,imageHeight:5,maxWidth:500,kernel:3,numFilters:5,showOutput:!0}}),Ve=new he({props:{imageWidth:5,imageHeight:5,maxWidth:500,kernel:3,numChannels:3,showOutput:!0}}),Ge=new Ca({props:{imageNumbers:[[9,0,3,5],[2,1,7,2],[0,0,1,3],[1,2,6,0]]}}),{c(){n=T("p"),a=c("Let's start this section with the following question."),t=v(),i=T("p"),P(o.$$.fragment),l=v(),u=T("p"),$=c(`Let's assume for a second, that we look at an image of a digit. If you look
    at the digit directly, you will have no problem recognizing the number. But
    when you interract with the example and flatten the image (as we did with
    MNIST so far), the task gets a lot harder. Yet that is exactly the problem
    our fully connected neural network has to face.`),L=v(),P(x.$$.fragment),q=v(),W=T("p"),_=c(`The loss of all spatial information makes our model also quite sensitive to
    different types of transformations: like translation, rotation, scaling,
    color and lightning. The image below is shifted sligtly to the right and to
    the top. When you compare the two flattened images you will notices, that
    there is hardly any overlap in pixel values, even thought we are dealing
    with an almost identical image.`),d=v(),P(m.$$.fragment),H=v(),O=T("p"),U=c(`Even if we didn't lose any spatial information, the combination of a fully
    connected neural network and images is problematic. The neural network below
    processes a flattened greyscale image of size 4x4 pixels. You can hardly
    argue that that is an image at all, yet the 16 inputs and the ten neurons in
    the hidden layer require 160 weights and 10 biases and the output neuron
    requires 11 more parameters.`),C=v(),P(M.$$.fragment),te=v(),R=T("p"),re=c(`Real-life images are vastly larger than that and we require a neural network
    with hundreds or thousands of neurons and several hidden layers to solve
    even a simple task. Even for an image of size 100x100 and 100 neuron we are
    dealing with 1,000,000 weights. Training fully connected neural networks can
    become extremely inefficient.`),Q=v(),D=T("p"),V=c(`Given those problems, we need a new neural network architecture. An
    architecture that is able to deal with images and video without destroying
    spatial information and requires fewer learnable parameters at the same
    time. The neural network that would alleviate our problems is called a `),P(F.$$.fragment),ne=c(", often abbreviated as CNN or ConvNet."),me=v(),X=T("div"),ge=v(),le=T("h2"),se=c("Convolutional Layer"),ce=v(),b=T("p"),B=c(`Let's take it one step at a time and think about how we could design such a
    neural network. We start with a basic assumption.`),Y=v(),P(ae.$$.fragment),Ce=v(),pe=T("p"),de=c(`Look at the image below. If you look at any pixel of that image, then with a
    very high probability the connecting pixels that surround that location are
    going to be part of the same object and will exhibit similar color values.
    Pixels that are part of the sky are surrounded by other sky pixels and
    mountain pixels are surrounded by other mountain pixels.`),Le=v(),J=T("figure"),He=T("img"),Sn=v(),Ze=T("figcaption"),kt=T("em"),Nn=c("Source: Alex Holzreiter, Unsplash"),Mt=v(),et=T("p"),qn=c(`In order to somehow leverage the spatial correlation that is contained in a
    local patch of pixels we could construct a neural network that limits the
    receptive field of each neuron.`),Dt=v(),P(we.$$.fragment),Ot=v(),tt=T("p"),Cn=c(`In a convolutional layer a neuron gets assigned a small patch of the image.
    Below for example the first neuron in the first hidden layer would focus
    only on the top left corner of the input image.`),Ut=v(),P(Be.$$.fragment),Rt=v(),$e=T("p"),Ln=c(`In a fully connected neural network a neuron had to be connected to all
    input pixels (hence the name fully connected). If we limit the number of
    pixels to a local patch of 2x2, that reduces the number of weights for a
    single neuron from 28*28 (MNIST dimensions) to just four. This is called `),P(_e.$$.fragment),Hn=c("."),Ft=v(),nt=T("p"),Bn=c(`Each neuron is calculated using a different patch of pixels and you can
    imagine that those calculations are conducted by using a sliding window on
    the input image. The output neurons are placed in a way that keeps the
    spatial structure of the image. For example the neuron that has the upper
    left corner in its receptive field is located in the upper left corner of
    the hidden layer. The neuron that attends to the patch that is to the right
    of the upper left corner, is put to the right of the before mentioned
    neuron. When the receptive field moves a row below, the neurons that attend
    to that receptive field also move below. This results in a new two
    dimensional image. You can start the interactive example below and observe
    how the receptive field moves and how the neurons are placed in a 2D grid.
    Notice also that the output image shrinks. This is expected, because a 2x2
    patch is required to construct a single neuron.`),Yt=v(),P(Me.$$.fragment),jt=v(),lt=T("p"),Mn=c(`You have a lot of control over the behaviour of the receptive field. You can
    for example control the size of the receptive field. Above we used the
    window of size 2x2, but 3x3 is also a common size.`),Vt=v(),P(De.$$.fragment),Gt=v(),ve=T("p"),Dn=c("The "),P(ye.$$.fragment),On=c(` is also a hyperparameter you will be interested
    in. The stride controls the number of steps the receptive field is moved. Above
    the field was moved 1 step to right and 1 step below, which corresponds to a
    stride of 1. In the example below we use a stride of 2. A larger stride obviously
    makes the output image smaller.`),Xt=v(),P(Oe.$$.fragment),Qt=v(),at=T("p"),Un=c(`As you have probability noticed, the output image is always smaller than the
    input image. If you want to keep the dimensionality between the input and
    ouput images consistent, you can pad the input image. Basically that means
    that you add artificial pixels by surrounding the input image with zeros.`),Jt=v(),P(Ue.$$.fragment),Kt=v(),it=T("p"),Rn=c(`When it comes to the actual calculation of the neuron values, we are dealing
    with an almost identical procedure that we used in the previous chapters.
    Let's asume we want to calculate the activation value for the patch in the
    upper left corner.`),Zt=v(),P(Re.$$.fragment),en=v(),ie=T("p"),Fn=c(`The patch
    `),P(be.$$.fragment),Yn=c(`
    is `),P(ke.$$.fragment),jn=c(`, therefore we need exactly 4
    weights.
    `),P(Te.$$.fragment),tn=v(),fe=T("p"),Vn=c(`This collection of weights that is applied to a limited receptive field is
    called a `),P(Ee.$$.fragment),Gn=c(" or a "),P(We.$$.fragment),Xn=c("."),nn=v(),ot=T("p"),Qn=c(`Similar to a fully connected neural network we calcualate a weighted sum,
    add a bias and apply a non-linear activation function to get the value of a
    neuron in the next layer.`),ln=v(),P(xe.$$.fragment),an=v(),K=T("p"),Jn=c("What is unique about convolutional neural networks is the "),P(Pe.$$.fragment),Kn=c(` among all neurons. When we slide the window of the receptive field, we do not
    replace the weights and biases, but always keep the same identical filter
    `),P(Ie.$$.fragment),Zn=c(". Weight sharing allows a filter to be "),P(Ae.$$.fragment),el=c(`, which means that a filter learns to detect particular features (like
    edges) of an image independent of where those features are located.`),on=v(),ze=T("p"),tl=c("The image that is produced by a filter is called a "),P(Se.$$.fragment),nl=c(`. Essentially a convolutional operation uses a filter to map an input image
    to an output image that highlights the features that are encoded in the
    filter.`),rn=v(),rt=T("p"),ll=c(`In the example below the input image and the kernel have pixel values of -1,
    0 or 1. The convolution layer produces positive values when a sufficient
    amount of either positive or negative numbers overlap. In our case the
    filter and the image only sufficiently overlap on the right edge. Remember
    that we are most likely going to apply a ReLU non-linearity, which means
    that most of those numbers are going to be set to 0.`),sn=v(),P(Fe.$$.fragment),fn=v(),st=T("p"),al=c(`Different filters would generate different types of overlaps and thereby
    focus on different features of an image. Using the same image, but a
    different filter produces a feature map, that hightlights the upper edge.`),hn=v(),P(Ye.$$.fragment),un=v(),ft=T("p"),il=c(`Usually we want a convolutional layer to calculate several feature maps. For
    that purpose a convolution layer learns several filters, each with with
    different weights and bias. The result of a convolutional layer is therefore
    not a single 2d image, but a 3d cube.`),mn=v(),P(je.$$.fragment),cn=v(),ht=T("p"),ol=c(`Similarly we will not always deal with 1-channel greyscale images. Instead
    we will either deal with colored images or with three dimensional feature
    maps that come from a previous convolutional layer. When we are dealing with
    several channels as inputs, our filters gain a channel dimension as well.
    That means that each neuron attends to a 3 dimensional receptive field.
    Below for example the receptive field is 3x3x3 which in turn requires a
    filter with 27 weights.`),pn=v(),P(Ve.$$.fragment),gn=v(),ut=T("div"),dn=v(),mt=T("h2"),rl=c("Pooling Layer"),wn=v(),ct=T("p"),sl=c(`While a convolution layer is more efficient than a fully connected layer due
    to sparse connectivity and weight sharing, you can still get into trouble
    when you are dealing with images of high resolution. The requirements on
    your computational resources can grow out of proportion. The pooling layer
    is intended to alleviate the problem by downsampling the image. That means
    that we use a pooling layer to reduce the resolution of an image.`),$n=v(),pt=T("p"),fl=c(`The convolutional layer downsamples an image automatically. If you don't use
    padding when you apply the convolutional operation, your image is going to
    shrink, especially if you use a stride above 1. The pooling layer does that
    in a different manner, while requiring no additional weights at all. That
    makes the pooling operation extremely efficient.`),_n=v(),gt=T("p"),hl=c(`Similar to a convolutional layer, a pooling layer has a receptive field and
    a stride. Usually the size of the receptive field and the stride are
    identical. If the receptive field is 2x2 the stride is also 2x2. That means
    each output of the pooling layer attends to a unique patch of the input
    image and there is never an overlap.`),vn=v(),dt=T("p"),ul=c(`The pooling layer applies simple operations to the patch in order to
    downsample the image. The average pooling layer for example calculates the
    average of the receptive field. But the most common pooling layer is
    probably the so called max pooling. As the name suggest, the pooling
    operation only keeps the largest value of the receptive field. Below we
    provide an interactive example of max pooling in order to make the
    explanations more intuitive.`),yn=v(),P(Ge.$$.fragment),bn=v(),wt=T("p"),ml=c(`There is one downside to downsampling though. While you make your images
    more managable by reducing the resolution, you also lose some spatial
    information. The max pooling operation for example example only keeps one of
    the four values and it is impossible to determine at a later stage in which
    location the value was stored. Pooling is often used for image
    classification and works generally great, but if you can not afford to lose
    spatial information, you should avoid the layer.`),kn=v(),$t=T("div"),Tn=v(),_t=T("h2"),cl=c("Hierarchy of Features"),En=v(),vt=T("p"),pl=c(`A neural network architecture, that is based on convolutional layers often
    has a very familiar procedure. First we take an image with a low number of
    channels and apply a convolutional layer to it. That procedure results in a
    stack of feature maps, let's say 16. We can regard the number of produced
    feature maps as a channel dimension, so that now we are faced with an image
    of dimension (16, W, H). As we know how to apply a convolution layer to an
    image with many channels, we can stack several convolutional layers. The
    dimension of channels grows (usually as of power of 2: 16, 32, 64, 128 ...)
    as we move forward in the convolutional neural network, while the width and
    height dimensions shrink either naturally by avoiding padding or through
    pooling layers. Once the number of feature maps has grown sufficiently and
    the width and height of images has shrunk dramatically, we can flatten all
    the feature maps and use a fully connected neural network in a familar
    manner.`),this.h()},l(e){n=E(e,"P",{});var h=k(n);a=p(h,"Let's start this section with the following question."),h.forEach(s),t=y(e),i=E(e,"P",{});var Tt=k(i);I(o.$$.fragment,Tt),Tt.forEach(s),l=y(e),u=E(e,"P",{});var Et=k(u);$=p(Et,`Let's assume for a second, that we look at an image of a digit. If you look
    at the digit directly, you will have no problem recognizing the number. But
    when you interract with the example and flatten the image (as we did with
    MNIST so far), the task gets a lot harder. Yet that is exactly the problem
    our fully connected neural network has to face.`),Et.forEach(s),L=y(e),I(x.$$.fragment,e),q=y(e),W=E(e,"P",{});var Wt=k(W);_=p(Wt,`The loss of all spatial information makes our model also quite sensitive to
    different types of transformations: like translation, rotation, scaling,
    color and lightning. The image below is shifted sligtly to the right and to
    the top. When you compare the two flattened images you will notices, that
    there is hardly any overlap in pixel values, even thought we are dealing
    with an almost identical image.`),Wt.forEach(s),d=y(e),I(m.$$.fragment,e),H=y(e),O=E(e,"P",{});var xt=k(O);U=p(xt,`Even if we didn't lose any spatial information, the combination of a fully
    connected neural network and images is problematic. The neural network below
    processes a flattened greyscale image of size 4x4 pixels. You can hardly
    argue that that is an image at all, yet the 16 inputs and the ten neurons in
    the hidden layer require 160 weights and 10 biases and the output neuron
    requires 11 more parameters.`),xt.forEach(s),C=y(e),I(M.$$.fragment,e),te=y(e),R=E(e,"P",{});var Pt=k(R);re=p(Pt,`Real-life images are vastly larger than that and we require a neural network
    with hundreds or thousands of neurons and several hidden layers to solve
    even a simple task. Even for an image of size 100x100 and 100 neuron we are
    dealing with 1,000,000 weights. Training fully connected neural networks can
    become extremely inefficient.`),Pt.forEach(s),Q=y(e),D=E(e,"P",{});var Xe=k(D);V=p(Xe,`Given those problems, we need a new neural network architecture. An
    architecture that is able to deal with images and video without destroying
    spatial information and requires fewer learnable parameters at the same
    time. The neural network that would alleviate our problems is called a `),I(F.$$.fragment,Xe),ne=p(Xe,", often abbreviated as CNN or ConvNet."),Xe.forEach(s),me=y(e),X=E(e,"DIV",{class:!0}),k(X).forEach(s),ge=y(e),le=E(e,"H2",{});var It=k(le);se=p(It,"Convolutional Layer"),It.forEach(s),ce=y(e),b=E(e,"P",{});var At=k(b);B=p(At,`Let's take it one step at a time and think about how we could design such a
    neural network. We start with a basic assumption.`),At.forEach(s),Y=y(e),I(ae.$$.fragment,e),Ce=y(e),pe=E(e,"P",{});var zt=k(pe);de=p(zt,`Look at the image below. If you look at any pixel of that image, then with a
    very high probability the connecting pixels that surround that location are
    going to be part of the same object and will exhibit similar color values.
    Pixels that are part of the sky are surrounded by other sky pixels and
    mountain pixels are surrounded by other mountain pixels.`),zt.forEach(s),Le=y(e),J=E(e,"FIGURE",{class:!0});var Qe=k(J);He=E(Qe,"IMG",{src:!0,alt:!0,class:!0}),Sn=y(Qe),Ze=E(Qe,"FIGCAPTION",{class:!0});var St=k(Ze);kt=E(St,"EM",{});var Nt=k(kt);Nn=p(Nt,"Source: Alex Holzreiter, Unsplash"),Nt.forEach(s),St.forEach(s),Qe.forEach(s),Mt=y(e),et=E(e,"P",{});var qt=k(et);qn=p(qt,`In order to somehow leverage the spatial correlation that is contained in a
    local patch of pixels we could construct a neural network that limits the
    receptive field of each neuron.`),qt.forEach(s),Dt=y(e),I(we.$$.fragment,e),Ot=y(e),tt=E(e,"P",{});var Ct=k(tt);Cn=p(Ct,`In a convolutional layer a neuron gets assigned a small patch of the image.
    Below for example the first neuron in the first hidden layer would focus
    only on the top left corner of the input image.`),Ct.forEach(s),Ut=y(e),I(Be.$$.fragment,e),Rt=y(e),$e=E(e,"P",{});var Je=k($e);Ln=p(Je,`In a fully connected neural network a neuron had to be connected to all
    input pixels (hence the name fully connected). If we limit the number of
    pixels to a local patch of 2x2, that reduces the number of weights for a
    single neuron from 28*28 (MNIST dimensions) to just four. This is called `),I(_e.$$.fragment,Je),Hn=p(Je,"."),Je.forEach(s),Ft=y(e),nt=E(e,"P",{});var Lt=k(nt);Bn=p(Lt,`Each neuron is calculated using a different patch of pixels and you can
    imagine that those calculations are conducted by using a sliding window on
    the input image. The output neurons are placed in a way that keeps the
    spatial structure of the image. For example the neuron that has the upper
    left corner in its receptive field is located in the upper left corner of
    the hidden layer. The neuron that attends to the patch that is to the right
    of the upper left corner, is put to the right of the before mentioned
    neuron. When the receptive field moves a row below, the neurons that attend
    to that receptive field also move below. This results in a new two
    dimensional image. You can start the interactive example below and observe
    how the receptive field moves and how the neurons are placed in a 2D grid.
    Notice also that the output image shrinks. This is expected, because a 2x2
    patch is required to construct a single neuron.`),Lt.forEach(s),Yt=y(e),I(Me.$$.fragment,e),jt=y(e),lt=E(e,"P",{});var Ht=k(lt);Mn=p(Ht,`You have a lot of control over the behaviour of the receptive field. You can
    for example control the size of the receptive field. Above we used the
    window of size 2x2, but 3x3 is also a common size.`),Ht.forEach(s),Vt=y(e),I(De.$$.fragment,e),Gt=y(e),ve=E(e,"P",{});var xn=k(ve);Dn=p(xn,"The "),I(ye.$$.fragment,xn),On=p(xn,` is also a hyperparameter you will be interested
    in. The stride controls the number of steps the receptive field is moved. Above
    the field was moved 1 step to right and 1 step below, which corresponds to a
    stride of 1. In the example below we use a stride of 2. A larger stride obviously
    makes the output image smaller.`),xn.forEach(s),Xt=y(e),I(Oe.$$.fragment,e),Qt=y(e),at=E(e,"P",{});var _l=k(at);Un=p(_l,`As you have probability noticed, the output image is always smaller than the
    input image. If you want to keep the dimensionality between the input and
    ouput images consistent, you can pad the input image. Basically that means
    that you add artificial pixels by surrounding the input image with zeros.`),_l.forEach(s),Jt=y(e),I(Ue.$$.fragment,e),Kt=y(e),it=E(e,"P",{});var vl=k(it);Rn=p(vl,`When it comes to the actual calculation of the neuron values, we are dealing
    with an almost identical procedure that we used in the previous chapters.
    Let's asume we want to calculate the activation value for the patch in the
    upper left corner.`),vl.forEach(s),Zt=y(e),I(Re.$$.fragment,e),en=y(e),ie=E(e,"P",{});var Ke=k(ie);Fn=p(Ke,`The patch
    `),I(be.$$.fragment,Ke),Yn=p(Ke,`
    is `),I(ke.$$.fragment,Ke),jn=p(Ke,`, therefore we need exactly 4
    weights.
    `),I(Te.$$.fragment,Ke),Ke.forEach(s),tn=y(e),fe=E(e,"P",{});var yt=k(fe);Vn=p(yt,`This collection of weights that is applied to a limited receptive field is
    called a `),I(Ee.$$.fragment,yt),Gn=p(yt," or a "),I(We.$$.fragment,yt),Xn=p(yt,"."),yt.forEach(s),nn=y(e),ot=E(e,"P",{});var yl=k(ot);Qn=p(yl,`Similar to a fully connected neural network we calcualate a weighted sum,
    add a bias and apply a non-linear activation function to get the value of a
    neuron in the next layer.`),yl.forEach(s),ln=y(e),I(xe.$$.fragment,e),an=y(e),K=E(e,"P",{});var Ne=k(K);Jn=p(Ne,"What is unique about convolutional neural networks is the "),I(Pe.$$.fragment,Ne),Kn=p(Ne,` among all neurons. When we slide the window of the receptive field, we do not
    replace the weights and biases, but always keep the same identical filter
    `),I(Ie.$$.fragment,Ne),Zn=p(Ne,". Weight sharing allows a filter to be "),I(Ae.$$.fragment,Ne),el=p(Ne,`, which means that a filter learns to detect particular features (like
    edges) of an image independent of where those features are located.`),Ne.forEach(s),on=y(e),ze=E(e,"P",{});var Pn=k(ze);tl=p(Pn,"The image that is produced by a filter is called a "),I(Se.$$.fragment,Pn),nl=p(Pn,`. Essentially a convolutional operation uses a filter to map an input image
    to an output image that highlights the features that are encoded in the
    filter.`),Pn.forEach(s),rn=y(e),rt=E(e,"P",{});var bl=k(rt);ll=p(bl,`In the example below the input image and the kernel have pixel values of -1,
    0 or 1. The convolution layer produces positive values when a sufficient
    amount of either positive or negative numbers overlap. In our case the
    filter and the image only sufficiently overlap on the right edge. Remember
    that we are most likely going to apply a ReLU non-linearity, which means
    that most of those numbers are going to be set to 0.`),bl.forEach(s),sn=y(e),I(Fe.$$.fragment,e),fn=y(e),st=E(e,"P",{});var kl=k(st);al=p(kl,`Different filters would generate different types of overlaps and thereby
    focus on different features of an image. Using the same image, but a
    different filter produces a feature map, that hightlights the upper edge.`),kl.forEach(s),hn=y(e),I(Ye.$$.fragment,e),un=y(e),ft=E(e,"P",{});var Tl=k(ft);il=p(Tl,`Usually we want a convolutional layer to calculate several feature maps. For
    that purpose a convolution layer learns several filters, each with with
    different weights and bias. The result of a convolutional layer is therefore
    not a single 2d image, but a 3d cube.`),Tl.forEach(s),mn=y(e),I(je.$$.fragment,e),cn=y(e),ht=E(e,"P",{});var El=k(ht);ol=p(El,`Similarly we will not always deal with 1-channel greyscale images. Instead
    we will either deal with colored images or with three dimensional feature
    maps that come from a previous convolutional layer. When we are dealing with
    several channels as inputs, our filters gain a channel dimension as well.
    That means that each neuron attends to a 3 dimensional receptive field.
    Below for example the receptive field is 3x3x3 which in turn requires a
    filter with 27 weights.`),El.forEach(s),pn=y(e),I(Ve.$$.fragment,e),gn=y(e),ut=E(e,"DIV",{class:!0}),k(ut).forEach(s),dn=y(e),mt=E(e,"H2",{});var Wl=k(mt);rl=p(Wl,"Pooling Layer"),Wl.forEach(s),wn=y(e),ct=E(e,"P",{});var xl=k(ct);sl=p(xl,`While a convolution layer is more efficient than a fully connected layer due
    to sparse connectivity and weight sharing, you can still get into trouble
    when you are dealing with images of high resolution. The requirements on
    your computational resources can grow out of proportion. The pooling layer
    is intended to alleviate the problem by downsampling the image. That means
    that we use a pooling layer to reduce the resolution of an image.`),xl.forEach(s),$n=y(e),pt=E(e,"P",{});var Pl=k(pt);fl=p(Pl,`The convolutional layer downsamples an image automatically. If you don't use
    padding when you apply the convolutional operation, your image is going to
    shrink, especially if you use a stride above 1. The pooling layer does that
    in a different manner, while requiring no additional weights at all. That
    makes the pooling operation extremely efficient.`),Pl.forEach(s),_n=y(e),gt=E(e,"P",{});var Il=k(gt);hl=p(Il,`Similar to a convolutional layer, a pooling layer has a receptive field and
    a stride. Usually the size of the receptive field and the stride are
    identical. If the receptive field is 2x2 the stride is also 2x2. That means
    each output of the pooling layer attends to a unique patch of the input
    image and there is never an overlap.`),Il.forEach(s),vn=y(e),dt=E(e,"P",{});var Al=k(dt);ul=p(Al,`The pooling layer applies simple operations to the patch in order to
    downsample the image. The average pooling layer for example calculates the
    average of the receptive field. But the most common pooling layer is
    probably the so called max pooling. As the name suggest, the pooling
    operation only keeps the largest value of the receptive field. Below we
    provide an interactive example of max pooling in order to make the
    explanations more intuitive.`),Al.forEach(s),yn=y(e),I(Ge.$$.fragment,e),bn=y(e),wt=E(e,"P",{});var zl=k(wt);ml=p(zl,`There is one downside to downsampling though. While you make your images
    more managable by reducing the resolution, you also lose some spatial
    information. The max pooling operation for example example only keeps one of
    the four values and it is impossible to determine at a later stage in which
    location the value was stored. Pooling is often used for image
    classification and works generally great, but if you can not afford to lose
    spatial information, you should avoid the layer.`),zl.forEach(s),kn=y(e),$t=E(e,"DIV",{class:!0}),k($t).forEach(s),Tn=y(e),_t=E(e,"H2",{});var Sl=k(_t);cl=p(Sl,"Hierarchy of Features"),Sl.forEach(s),En=y(e),vt=E(e,"P",{});var Nl=k(vt);pl=p(Nl,`A neural network architecture, that is based on convolutional layers often
    has a very familiar procedure. First we take an image with a low number of
    channels and apply a convolutional layer to it. That procedure results in a
    stack of feature maps, let's say 16. We can regard the number of produced
    feature maps as a channel dimension, so that now we are faced with an image
    of dimension (16, W, H). As we know how to apply a convolution layer to an
    image with many channels, we can stack several convolutional layers. The
    dimension of channels grows (usually as of power of 2: 16, 32, 64, 128 ...)
    as we move forward in the convolutional neural network, while the width and
    height dimensions shrink either naturally by avoiding padding or through
    pooling layers. Once the number of feature maps has grown sufficiently and
    the width and height of images has shrunk dramatically, we can flatten all
    the feature maps and use a fully connected neural network in a familar
    manner.`),Nl.forEach(s),this.h()},h(){g(X,"class","separator"),ga(He.src,$l=La)||g(He,"src",$l),g(He,"alt","Sky, mountains and sea"),g(He,"class","max-w-lg rounded-xl"),g(Ze,"class","text-sm text-center"),g(J,"class","flex flex-col justify-center items-center"),g(ut,"class","separator"),g($t,"class","separator")},m(e,h){f(e,n,h),w(n,a),f(e,t,h),f(e,i,h),A(o,i,null),f(e,l,h),f(e,u,h),w(u,$),f(e,L,h),A(x,e,h),f(e,q,h),f(e,W,h),w(W,_),f(e,d,h),A(m,e,h),f(e,H,h),f(e,O,h),w(O,U),f(e,C,h),A(M,e,h),f(e,te,h),f(e,R,h),w(R,re),f(e,Q,h),f(e,D,h),w(D,V),A(F,D,null),w(D,ne),f(e,me,h),f(e,X,h),f(e,ge,h),f(e,le,h),w(le,se),f(e,ce,h),f(e,b,h),w(b,B),f(e,Y,h),A(ae,e,h),f(e,Ce,h),f(e,pe,h),w(pe,de),f(e,Le,h),f(e,J,h),w(J,He),w(J,Sn),w(J,Ze),w(Ze,kt),w(kt,Nn),f(e,Mt,h),f(e,et,h),w(et,qn),f(e,Dt,h),A(we,e,h),f(e,Ot,h),f(e,tt,h),w(tt,Cn),f(e,Ut,h),A(Be,e,h),f(e,Rt,h),f(e,$e,h),w($e,Ln),A(_e,$e,null),w($e,Hn),f(e,Ft,h),f(e,nt,h),w(nt,Bn),f(e,Yt,h),A(Me,e,h),f(e,jt,h),f(e,lt,h),w(lt,Mn),f(e,Vt,h),A(De,e,h),f(e,Gt,h),f(e,ve,h),w(ve,Dn),A(ye,ve,null),w(ve,On),f(e,Xt,h),A(Oe,e,h),f(e,Qt,h),f(e,at,h),w(at,Un),f(e,Jt,h),A(Ue,e,h),f(e,Kt,h),f(e,it,h),w(it,Rn),f(e,Zt,h),A(Re,e,h),f(e,en,h),f(e,ie,h),w(ie,Fn),A(be,ie,null),w(ie,Yn),A(ke,ie,null),w(ie,jn),A(Te,ie,null),f(e,tn,h),f(e,fe,h),w(fe,Vn),A(Ee,fe,null),w(fe,Gn),A(We,fe,null),w(fe,Xn),f(e,nn,h),f(e,ot,h),w(ot,Qn),f(e,ln,h),A(xe,e,h),f(e,an,h),f(e,K,h),w(K,Jn),A(Pe,K,null),w(K,Kn),A(Ie,K,null),w(K,Zn),A(Ae,K,null),w(K,el),f(e,on,h),f(e,ze,h),w(ze,tl),A(Se,ze,null),w(ze,nl),f(e,rn,h),f(e,rt,h),w(rt,ll),f(e,sn,h),A(Fe,e,h),f(e,fn,h),f(e,st,h),w(st,al),f(e,hn,h),A(Ye,e,h),f(e,un,h),f(e,ft,h),w(ft,il),f(e,mn,h),A(je,e,h),f(e,cn,h),f(e,ht,h),w(ht,ol),f(e,pn,h),A(Ve,e,h),f(e,gn,h),f(e,ut,h),f(e,dn,h),f(e,mt,h),w(mt,rl),f(e,wn,h),f(e,ct,h),w(ct,sl),f(e,$n,h),f(e,pt,h),w(pt,fl),f(e,_n,h),f(e,gt,h),w(gt,hl),f(e,vn,h),f(e,dt,h),w(dt,ul),f(e,yn,h),A(Ge,e,h),f(e,bn,h),f(e,wt,h),w(wt,ml),f(e,kn,h),f(e,$t,h),f(e,Tn,h),f(e,_t,h),w(_t,cl),f(e,En,h),f(e,vt,h),w(vt,pl),Wn=!0},p(e,h){const Tt={};h&8&&(Tt.$$scope={dirty:h,ctx:e}),o.$set(Tt);const Et={};h&1&&(Et.layers=e[0]),M.$set(Et);const Wt={};h&8&&(Wt.$$scope={dirty:h,ctx:e}),F.$set(Wt);const xt={};h&8&&(xt.$$scope={dirty:h,ctx:e}),ae.$set(xt);const Pt={};h&8&&(Pt.$$scope={dirty:h,ctx:e}),we.$set(Pt);const Xe={};h&8&&(Xe.$$scope={dirty:h,ctx:e}),_e.$set(Xe);const It={};h&8&&(It.$$scope={dirty:h,ctx:e}),ye.$set(It);const At={};h&8&&(At.$$scope={dirty:h,ctx:e}),be.$set(At);const zt={};h&8&&(zt.$$scope={dirty:h,ctx:e}),ke.$set(zt);const Qe={};h&8&&(Qe.$$scope={dirty:h,ctx:e}),Te.$set(Qe);const St={};h&8&&(St.$$scope={dirty:h,ctx:e}),Ee.$set(St);const Nt={};h&8&&(Nt.$$scope={dirty:h,ctx:e}),We.$set(Nt);const qt={};h&8&&(qt.$$scope={dirty:h,ctx:e}),xe.$set(qt);const Ct={};h&8&&(Ct.$$scope={dirty:h,ctx:e}),Pe.$set(Ct);const Je={};h&8&&(Je.$$scope={dirty:h,ctx:e}),Ie.$set(Je);const Lt={};h&8&&(Lt.$$scope={dirty:h,ctx:e}),Ae.$set(Lt);const Ht={};h&8&&(Ht.$$scope={dirty:h,ctx:e}),Se.$set(Ht)},i(e){Wn||(z(o.$$.fragment,e),z(x.$$.fragment,e),z(m.$$.fragment,e),z(M.$$.fragment,e),z(F.$$.fragment,e),z(ae.$$.fragment,e),z(we.$$.fragment,e),z(Be.$$.fragment,e),z(_e.$$.fragment,e),z(Me.$$.fragment,e),z(De.$$.fragment,e),z(ye.$$.fragment,e),z(Oe.$$.fragment,e),z(Ue.$$.fragment,e),z(Re.$$.fragment,e),z(be.$$.fragment,e),z(ke.$$.fragment,e),z(Te.$$.fragment,e),z(Ee.$$.fragment,e),z(We.$$.fragment,e),z(xe.$$.fragment,e),z(Pe.$$.fragment,e),z(Ie.$$.fragment,e),z(Ae.$$.fragment,e),z(Se.$$.fragment,e),z(Fe.$$.fragment,e),z(Ye.$$.fragment,e),z(je.$$.fragment,e),z(Ve.$$.fragment,e),z(Ge.$$.fragment,e),Wn=!0)},o(e){S(o.$$.fragment,e),S(x.$$.fragment,e),S(m.$$.fragment,e),S(M.$$.fragment,e),S(F.$$.fragment,e),S(ae.$$.fragment,e),S(we.$$.fragment,e),S(Be.$$.fragment,e),S(_e.$$.fragment,e),S(Me.$$.fragment,e),S(De.$$.fragment,e),S(ye.$$.fragment,e),S(Oe.$$.fragment,e),S(Ue.$$.fragment,e),S(Re.$$.fragment,e),S(be.$$.fragment,e),S(ke.$$.fragment,e),S(Te.$$.fragment,e),S(Ee.$$.fragment,e),S(We.$$.fragment,e),S(xe.$$.fragment,e),S(Pe.$$.fragment,e),S(Ie.$$.fragment,e),S(Ae.$$.fragment,e),S(Se.$$.fragment,e),S(Fe.$$.fragment,e),S(Ye.$$.fragment,e),S(je.$$.fragment,e),S(Ve.$$.fragment,e),S(Ge.$$.fragment,e),Wn=!1},d(e){e&&s(n),e&&s(t),e&&s(i),N(o),e&&s(l),e&&s(u),e&&s(L),N(x,e),e&&s(q),e&&s(W),e&&s(d),N(m,e),e&&s(H),e&&s(O),e&&s(C),N(M,e),e&&s(te),e&&s(R),e&&s(Q),e&&s(D),N(F),e&&s(me),e&&s(X),e&&s(ge),e&&s(le),e&&s(ce),e&&s(b),e&&s(Y),N(ae,e),e&&s(Ce),e&&s(pe),e&&s(Le),e&&s(J),e&&s(Mt),e&&s(et),e&&s(Dt),N(we,e),e&&s(Ot),e&&s(tt),e&&s(Ut),N(Be,e),e&&s(Rt),e&&s($e),N(_e),e&&s(Ft),e&&s(nt),e&&s(Yt),N(Me,e),e&&s(jt),e&&s(lt),e&&s(Vt),N(De,e),e&&s(Gt),e&&s(ve),N(ye),e&&s(Xt),N(Oe,e),e&&s(Qt),e&&s(at),e&&s(Jt),N(Ue,e),e&&s(Kt),e&&s(it),e&&s(Zt),N(Re,e),e&&s(en),e&&s(ie),N(be),N(ke),N(Te),e&&s(tn),e&&s(fe),N(Ee),N(We),e&&s(nn),e&&s(ot),e&&s(ln),N(xe,e),e&&s(an),e&&s(K),N(Pe),N(Ie),N(Ae),e&&s(on),e&&s(ze),N(Se),e&&s(rn),e&&s(rt),e&&s(sn),N(Fe,e),e&&s(fn),e&&s(st),e&&s(hn),N(Ye,e),e&&s(un),e&&s(ft),e&&s(mn),N(je,e),e&&s(cn),e&&s(ht),e&&s(pn),N(Ve,e),e&&s(gn),e&&s(ut),e&&s(dn),e&&s(mt),e&&s(wn),e&&s(ct),e&&s($n),e&&s(pt),e&&s(_n),e&&s(gt),e&&s(vn),e&&s(dt),e&&s(yn),N(Ge,e),e&&s(bn),e&&s(wt),e&&s(kn),e&&s($t),e&&s(Tn),e&&s(_t),e&&s(En),e&&s(vt)}}}function ei(r){let n,a;return n=new Ta({props:{maxWidth:"1400"}}),{c(){P(n.$$.fragment)},l(t){I(n.$$.fragment,t)},m(t,i){A(n,t,i),a=!0},p:qe,i(t){a||(z(n.$$.fragment,t),a=!0)},o(t){S(n.$$.fragment,t),a=!1},d(t){N(n,t)}}}function ti(r){let n;return{c(){n=c("feature extractor")},l(a){n=p(a,"feature extractor")},m(a,t){f(a,n,t)},d(a){a&&s(n)}}}function ni(r){let n,a,t,i,o,l,u,$,L,x,q,W,_,d,m,H,O,U,C,M,te,R,re,Q,D,V,F,ne,me,X,ge,le,se,ce;return t=new ue({props:{$$slots:{default:[ti]},$$scope:{ctx:r}}}),U=new dl({props:{code:`class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
                
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
                
    def forward(self, features):
        features = self.feature_extractor(features)
        logits = self.classifier(features)
        return logits`}}),V=new dl({props:{code:`X = torch.randn(32, 1, 28, 28)
model = Model()
with torch.inference_mode():
    print(model.feature_extractor(X).shape)`}}),ne=new dl({props:{code:"torch.Size([32, 64, 2, 2])",isOutput:!0}}),{c(){n=T("p"),a=c(`This stacking of convolutional neural networks and the growing number of
    feature maps is usually attributed the unbelievable success of ConvNets. In
    the first layer the receptive field is limited to a small area, therefore
    the network learns local features. As the number of layers grows, the
    subsequent layers start to learn features of features. Because of that,
    subsequent layers will attend to a larger area of the original image. If the
    first neuron in the first layer attends to four pixels in the upper left
    corner, the first neuron in the second layer will attend to features build
    on the 16 pixels of the original image (assuming a stride of 2). This
    hierarchical structure of feature detectors allows to find higher and higher
    level features, going for example from edges and colors to distinct shapes
    to actual objects. By the time we arrive at the last convolutional layer, we
    usually have more than 100 feature maps, each theoretically containing some
    higher level feature. Those features would be able to answer questions like:
    "Is there a nose?" or "Is there a tail?" or "Are there whiskers?". That is
    why the first part of a convolutional neural network is often called a `),P(t.$$.fragment),i=c(`. The last fully connected layers leverage those features to predict the
    class of an image.`),o=v(),l=T("p"),u=c(`Below we present a convolutional neural network implemented in PyTorch. The
    convolutional layer and the pooling layer are implemented in the `),$=T("code"),L=c("nn.Conv2d()"),x=c(`
    and `),q=T("code"),W=c("nn.MaxPool2d()"),_=c(` respectively. We separate the feature
    extractor and the classifier steps into individual
    `),d=T("code"),m=c("nn.Sequential()"),H=c(` modules, but theoretically you could structure the
    model any way you desire.`),O=v(),P(U.$$.fragment),C=v(),M=T("p"),te=c(`If you ask yourself where the number 256 in the first linear layer comes
    from, this is the number of neurons that remain after the last max pooling
    operation. There is an explicit formula to calculate the size of your
    feature maps and you can read about it in the PyTorch `),R=T("a"),re=c("documentation"),Q=c(`, but it is usually much more convenient to create a dummy input, pass it
    though your feature extractor and to deduce the number of features.`),D=v(),P(V.$$.fragment),F=v(),P(ne.$$.fragment),me=v(),X=T("p"),ge=c(`Above for example we assume that we are dealing with the MNIST dataset. Each
    image is of shape (1, 28, 28) and the batch size is 32. After the input is
    processed by the feature extractor, we end up with a dimension of (32, 64,
    2, 2), which means that we have a batch of 32 images consisting of 64
    channels, each of size 2x2. When we multiply 64x2x2 we end up with the
    number 256.`),le=v(),se=T("div"),this.h()},l(b){n=E(b,"P",{});var B=k(n);a=p(B,`This stacking of convolutional neural networks and the growing number of
    feature maps is usually attributed the unbelievable success of ConvNets. In
    the first layer the receptive field is limited to a small area, therefore
    the network learns local features. As the number of layers grows, the
    subsequent layers start to learn features of features. Because of that,
    subsequent layers will attend to a larger area of the original image. If the
    first neuron in the first layer attends to four pixels in the upper left
    corner, the first neuron in the second layer will attend to features build
    on the 16 pixels of the original image (assuming a stride of 2). This
    hierarchical structure of feature detectors allows to find higher and higher
    level features, going for example from edges and colors to distinct shapes
    to actual objects. By the time we arrive at the last convolutional layer, we
    usually have more than 100 feature maps, each theoretically containing some
    higher level feature. Those features would be able to answer questions like:
    "Is there a nose?" or "Is there a tail?" or "Are there whiskers?". That is
    why the first part of a convolutional neural network is often called a `),I(t.$$.fragment,B),i=p(B,`. The last fully connected layers leverage those features to predict the
    class of an image.`),B.forEach(s),o=y(b),l=E(b,"P",{});var Y=k(l);u=p(Y,`Below we present a convolutional neural network implemented in PyTorch. The
    convolutional layer and the pooling layer are implemented in the `),$=E(Y,"CODE",{});var ae=k($);L=p(ae,"nn.Conv2d()"),ae.forEach(s),x=p(Y,`
    and `),q=E(Y,"CODE",{});var Ce=k(q);W=p(Ce,"nn.MaxPool2d()"),Ce.forEach(s),_=p(Y,` respectively. We separate the feature
    extractor and the classifier steps into individual
    `),d=E(Y,"CODE",{});var pe=k(d);m=p(pe,"nn.Sequential()"),pe.forEach(s),H=p(Y,` modules, but theoretically you could structure the
    model any way you desire.`),Y.forEach(s),O=y(b),I(U.$$.fragment,b),C=y(b),M=E(b,"P",{});var de=k(M);te=p(de,`If you ask yourself where the number 256 in the first linear layer comes
    from, this is the number of neurons that remain after the last max pooling
    operation. There is an explicit formula to calculate the size of your
    feature maps and you can read about it in the PyTorch `),R=E(de,"A",{href:!0,target:!0,rel:!0});var Le=k(R);re=p(Le,"documentation"),Le.forEach(s),Q=p(de,`, but it is usually much more convenient to create a dummy input, pass it
    though your feature extractor and to deduce the number of features.`),de.forEach(s),D=y(b),I(V.$$.fragment,b),F=y(b),I(ne.$$.fragment,b),me=y(b),X=E(b,"P",{});var J=k(X);ge=p(J,`Above for example we assume that we are dealing with the MNIST dataset. Each
    image is of shape (1, 28, 28) and the batch size is 32. After the input is
    processed by the feature extractor, we end up with a dimension of (32, 64,
    2, 2), which means that we have a batch of 32 images consisting of 64
    channels, each of size 2x2. When we multiply 64x2x2 we end up with the
    number 256.`),J.forEach(s),le=y(b),se=E(b,"DIV",{class:!0}),k(se).forEach(s),this.h()},h(){g(R,"href","https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d"),g(R,"target","_blank"),g(R,"rel","noreferrer"),g(se,"class","separator")},m(b,B){f(b,n,B),w(n,a),A(t,n,null),w(n,i),f(b,o,B),f(b,l,B),w(l,u),w(l,$),w($,L),w(l,x),w(l,q),w(q,W),w(l,_),w(l,d),w(d,m),w(l,H),f(b,O,B),A(U,b,B),f(b,C,B),f(b,M,B),w(M,te),w(M,R),w(R,re),w(M,Q),f(b,D,B),A(V,b,B),f(b,F,B),A(ne,b,B),f(b,me,B),f(b,X,B),w(X,ge),f(b,le,B),f(b,se,B),ce=!0},p(b,B){const Y={};B&8&&(Y.$$scope={dirty:B,ctx:b}),t.$set(Y)},i(b){ce||(z(t.$$.fragment,b),z(U.$$.fragment,b),z(V.$$.fragment,b),z(ne.$$.fragment,b),ce=!0)},o(b){S(t.$$.fragment,b),S(U.$$.fragment,b),S(V.$$.fragment,b),S(ne.$$.fragment,b),ce=!1},d(b){b&&s(n),N(t),b&&s(o),b&&s(l),b&&s(O),N(U,b),b&&s(C),b&&s(M),b&&s(D),N(V,b),b&&s(F),N(ne,b),b&&s(me),b&&s(X),b&&s(le),b&&s(se)}}}function li(r){let n,a,t,i,o,l,u,$,L,x,q,W,_;return $=new gl({props:{$$slots:{default:[Za]},$$scope:{ctx:r}}}),x=new gl({props:{maxWidth:"1400px",$$slots:{default:[ei]},$$scope:{ctx:r}}}),W=new gl({props:{$$slots:{default:[ni]},$$scope:{ctx:r}}}),{c(){n=T("meta"),a=v(),t=T("h1"),i=c("Convolutional Neural Networks"),o=v(),l=T("div"),u=v(),P($.$$.fragment),L=v(),P(x.$$.fragment),q=v(),P(W.$$.fragment),this.h()},l(d){const m=pa("svelte-xwufif",document.head);n=E(m,"META",{name:!0,content:!0}),m.forEach(s),a=y(d),t=E(d,"H1",{});var H=k(t);i=p(H,"Convolutional Neural Networks"),H.forEach(s),o=y(d),l=E(d,"DIV",{class:!0}),k(l).forEach(s),u=y(d),I($.$$.fragment,d),L=y(d),I(x.$$.fragment,d),q=y(d),I(W.$$.fragment,d),this.h()},h(){document.title="Convolutional Neural Networks - World4AI",g(n,"name","description"),g(n,"content","A convolutional neural network is a more efficient neural network due to weight sharing and sparse connections. The network learns hierarchies of features by stacking more and more convolutional layers. That architecture allows the network to go from local to global features."),g(l,"class","separator")},m(d,m){w(document.head,n),f(d,a,m),f(d,t,m),w(t,i),f(d,o,m),f(d,l,m),f(d,u,m),A($,d,m),f(d,L,m),A(x,d,m),f(d,q,m),A(W,d,m),_=!0},p(d,[m]){const H={};m&9&&(H.$$scope={dirty:m,ctx:d}),$.$set(H);const O={};m&8&&(O.$$scope={dirty:m,ctx:d}),x.$set(O);const U={};m&8&&(U.$$scope={dirty:m,ctx:d}),W.$set(U)},i(d){_||(z($.$$.fragment,d),z(x.$$.fragment,d),z(W.$$.fragment,d),_=!0)},o(d){S($.$$.fragment,d),S(x.$$.fragment,d),S(W.$$.fragment,d),_=!1},d(d){s(n),d&&s(a),d&&s(t),d&&s(o),d&&s(l),d&&s(u),N($,d),d&&s(L),N(x,d),d&&s(q),N(W,d)}}}function ai(r,n,a){const t=[[0,0,0,0,0],[0,1,1,1,0],[0,0,0,1,0],[0,1,1,1,0],[0,1,0,0,0],[0,1,1,1,0],[0,0,0,0,0]],i=[[0,0,1,1,1],[0,0,0,0,1],[0,0,1,1,1],[0,0,1,0,0],[0,0,1,1,1],[0,0,0,0,0],[0,0,0,0,0]];let{layers:o=[{title:"Input",nodes:[{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"}]},{title:"Hidden Layer",nodes:[{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"},{value:"",class:"fill-white"}]},{title:"",nodes:[{value:"",class:"fill-white"}]}]}=n;return r.$$set=l=>{"layers"in l&&a(0,o=l.layers)},[o,t,i]}class $i extends In{constructor(n){super(),An(this,n,ai,li,zn,{layers:0})}}export{$i as default};
