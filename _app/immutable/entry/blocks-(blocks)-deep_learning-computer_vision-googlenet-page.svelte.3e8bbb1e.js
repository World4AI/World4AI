import{S as Ft,i as Yt,s as jt,k as P,a as y,q as I,y as $,W as Ht,l as S,h as l,c as k,m as N,r as z,z as d,n as re,N as T,b as p,A as _,g as u,d as h,B as g,Q as Mt,R as Rt,C as we,e as Le,v as kt,f as Et,P as Vt}from"../chunks/index.4d92b023.js";import{C as Ot}from"../chunks/Container.b0705c7b.js";import{F as Ut,I as bt}from"../chunks/InternalLink.7deb899c.js";import{C as Jt}from"../chunks/Convolution.b73a3805.js";import{S as Tt}from"../chunks/SvgContainer.f70b5745.js";import{H as Dt}from"../chunks/Highlight.b7c1de53.js";import{P as F}from"../chunks/PythonCode.212ba7a6.js";import{T as Qt,a as Zt,b as Kt,R as qt,H as Xt,D as en}from"../chunks/HeaderEntry.2b6e8f51.js";import{B as O}from"../chunks/Block.059eddcd.js";import{A as Y}from"../chunks/Arrow.ae91874c.js";function Nt(x,t,r){const a=x.slice();return a[4]=t[r],a}function Pt(x,t,r){const a=x.slice();return a[7]=t[r],a}function St(x,t,r){const a=x.slice();return a[10]=t[r],a}function tn(x){let t;return{c(){t=I("GoogLeNet")},l(r){t=z(r,"GoogLeNet")},m(r,a){p(r,t,a)},d(r){r&&l(t)}}}function nn(x){let t,r,a,n,i,o,c;return r=new O({props:{x:oe/2,y:Be-Z/2-W,width:yt,height:Z,text:"Conv2d"}}),a=new O({props:{x:oe/2,y:Be/2,width:yt,height:Z,text:"BatchNorm2d"}}),n=new O({props:{x:oe/2,y:Z/2+W,width:yt,height:Z,text:"ReLU"}}),i=new Y({props:{data:[{x:oe/2,y:Be-Z-W},{x:oe/2,y:Be/2+Z/2+3}],dashed:!0,moving:!0}}),o=new Y({props:{data:[{x:oe/2,y:Be/2-Z/2},{x:oe/2,y:Z+4}],dashed:!0,moving:!0}}),{c(){t=Mt("svg"),$(r.$$.fragment),$(a.$$.fragment),$(n.$$.fragment),$(i.$$.fragment),$(o.$$.fragment),this.h()},l(s){t=Rt(s,"svg",{viewBox:!0});var E=N(t);d(r.$$.fragment,E),d(a.$$.fragment,E),d(n.$$.fragment,E),d(i.$$.fragment,E),d(o.$$.fragment,E),E.forEach(l),this.h()},h(){re(t,"viewBox","0 0 "+oe+" "+Be)},m(s,E){p(s,t,E),_(r,t,null),_(a,t,null),_(n,t,null),_(i,t,null),_(o,t,null),c=!0},p:we,i(s){c||(u(r.$$.fragment,s),u(a.$$.fragment,s),u(n.$$.fragment,s),u(i.$$.fragment,s),u(o.$$.fragment,s),c=!0)},o(s){h(r.$$.fragment,s),h(a.$$.fragment,s),h(n.$$.fragment,s),h(i.$$.fragment,s),h(o.$$.fragment,s),c=!1},d(s){s&&l(t),g(r),g(a),g(n),g(i),g(o)}}}function an(x){let t,r,a,n,i,o,c,s,E,D,C,w,V,G,R,U,j,M,H,q,J,Q;return r=new O({props:{x:B/2,y:b/2+W,width:A,height:b,text:"Concatenation",fontSize:25}}),a=new O({props:{x:A/2+W,y:v-b-v/2,width:A,height:b,text:"1x1 Basic Block",fontSize:25}}),n=new O({props:{x:B/3*1,y:v-b-v/2,width:A,height:b,text:"3x3 Basic Block",fontSize:25}}),i=new O({props:{x:B/3*2,y:v-b-v/2,width:A,height:b,text:"5x5 Basic Block",fontSize:25}}),o=new O({props:{x:B/3*3-A/2-W,y:v-b-v/2,width:A,height:b,text:"1x1 Basic Block",fontSize:25}}),c=new O({props:{x:B/3*1,y:v-v/3,width:A,height:b,text:"1x1 Basic Block",fontSize:25}}),s=new O({props:{x:B/3*2,y:v-v/3,width:A,height:b,text:"1x1 Basic Block",fontSize:25}}),E=new O({props:{x:B/3*3-A/2-W,y:v-v/3,width:A,height:b,text:"3x3 MaxPool",fontSize:25}}),D=new O({props:{x:B/2,y:v-b/2-W,width:A,height:b,text:"Input",fontSize:25}}),C=new Y({props:{data:[{x:B/2,y:v-b},{x:B/3*1,y:v-v/3+b/2+10}],moving:!0,strokeWidth:3,dashed:!0,strokeDashArray:"14 4",speed:50}}),w=new Y({props:{data:[{x:B/2,y:v-b},{x:B/3*2,y:v-v/3+b/2+10}],moving:!0,strokeWidth:3,dashed:!0,strokeDashArray:"14 4",speed:50}}),V=new Y({props:{data:[{x:B/2+A/2,y:v-b/2},{x:B/3*3-A/2-W,y:v-b/2},{x:B/3*3-A/2-W,y:v-v/3+b/2+10}],moving:!0,strokeWidth:3,dashed:!0,strokeDashArray:"14 4",speed:50}}),G=new Y({props:{data:[{x:B/2-A/2,y:v-b/2},{x:0+A/2+W,y:v-b/2},{x:0+A/2+W,y:v-b/2-v/2+10}],moving:!0,strokeWidth:3,dashed:!0,strokeDashArray:"14 4",speed:50}}),R=new Y({props:{data:[{x:B/3*1,y:v-v/3-b/2},{x:B/3*1,y:v-b/2-v/2+10}],moving:!0,strokeWidth:3,dashed:!0,strokeDashArray:"14 4",speed:50}}),U=new Y({props:{data:[{x:B/3*2,y:v-v/3-b/2},{x:B/3*2,y:v-b/2-v/2+10}],moving:!0,strokeWidth:3,dashed:!0,strokeDashArray:"14 4",speed:50}}),j=new Y({props:{data:[{x:B/3*3-A/2-W,y:v-v/3-b/2},{x:B/3*3-A/2-W,y:v-b/2-v/2+10}],moving:!0,strokeWidth:3,dashed:!0,strokeDashArray:"14 4",speed:50}}),M=new Y({props:{data:[{x:0+A/2+W,y:v-b-b/2-v/2},{x:0+A/2+W,y:b/2},{x:B/2-A/2-10,y:b/2}],moving:!0,strokeWidth:3,dashed:!0,strokeDashArray:"14 4",speed:50}}),H=new Y({props:{data:[{x:B/3*1,y:v-b-b/2-v/2},{x:B/2-A/2,y:b/2+b/2+10}],moving:!0,strokeWidth:3,dashed:!0,strokeDashArray:"14 4",speed:50}}),q=new Y({props:{data:[{x:B/3*2,y:v-b-b/2-v/2},{x:B/2+A/2,y:b/2+b/2+10}],moving:!0,strokeWidth:3,dashed:!0,strokeDashArray:"14 4",speed:50}}),J=new Y({props:{data:[{x:B/3*3-A/2-W,y:v-b-b/2-v/2},{x:B/3*3-A/2-W,y:b/2},{x:B/2+A/2+10,y:b/2}],moving:!0,strokeWidth:3,dashed:!0,strokeDashArray:"14 4",speed:50}}),{c(){t=Mt("svg"),$(r.$$.fragment),$(a.$$.fragment),$(n.$$.fragment),$(i.$$.fragment),$(o.$$.fragment),$(c.$$.fragment),$(s.$$.fragment),$(E.$$.fragment),$(D.$$.fragment),$(C.$$.fragment),$(w.$$.fragment),$(V.$$.fragment),$(G.$$.fragment),$(R.$$.fragment),$(U.$$.fragment),$(j.$$.fragment),$(M.$$.fragment),$(H.$$.fragment),$(q.$$.fragment),$(J.$$.fragment),this.h()},l(m){t=Rt(m,"svg",{viewBox:!0});var L=N(t);d(r.$$.fragment,L),d(a.$$.fragment,L),d(n.$$.fragment,L),d(i.$$.fragment,L),d(o.$$.fragment,L),d(c.$$.fragment,L),d(s.$$.fragment,L),d(E.$$.fragment,L),d(D.$$.fragment,L),d(C.$$.fragment,L),d(w.$$.fragment,L),d(V.$$.fragment,L),d(G.$$.fragment,L),d(R.$$.fragment,L),d(U.$$.fragment,L),d(j.$$.fragment,L),d(M.$$.fragment,L),d(H.$$.fragment,L),d(q.$$.fragment,L),d(J.$$.fragment,L),L.forEach(l),this.h()},h(){re(t,"viewBox","0 0 "+B+" "+v)},m(m,L){p(m,t,L),_(r,t,null),_(a,t,null),_(n,t,null),_(i,t,null),_(o,t,null),_(c,t,null),_(s,t,null),_(E,t,null),_(D,t,null),_(C,t,null),_(w,t,null),_(V,t,null),_(G,t,null),_(R,t,null),_(U,t,null),_(j,t,null),_(M,t,null),_(H,t,null),_(q,t,null),_(J,t,null),Q=!0},p:we,i(m){Q||(u(r.$$.fragment,m),u(a.$$.fragment,m),u(n.$$.fragment,m),u(i.$$.fragment,m),u(o.$$.fragment,m),u(c.$$.fragment,m),u(s.$$.fragment,m),u(E.$$.fragment,m),u(D.$$.fragment,m),u(C.$$.fragment,m),u(w.$$.fragment,m),u(V.$$.fragment,m),u(G.$$.fragment,m),u(R.$$.fragment,m),u(U.$$.fragment,m),u(j.$$.fragment,m),u(M.$$.fragment,m),u(H.$$.fragment,m),u(q.$$.fragment,m),u(J.$$.fragment,m),Q=!0)},o(m){h(r.$$.fragment,m),h(a.$$.fragment,m),h(n.$$.fragment,m),h(i.$$.fragment,m),h(o.$$.fragment,m),h(c.$$.fragment,m),h(s.$$.fragment,m),h(E.$$.fragment,m),h(D.$$.fragment,m),h(C.$$.fragment,m),h(w.$$.fragment,m),h(V.$$.fragment,m),h(G.$$.fragment,m),h(R.$$.fragment,m),h(U.$$.fragment,m),h(j.$$.fragment,m),h(M.$$.fragment,m),h(H.$$.fragment,m),h(q.$$.fragment,m),h(J.$$.fragment,m),Q=!1},d(m){m&&l(t),g(r),g(a),g(n),g(i),g(o),g(c),g(s),g(E),g(D),g(C),g(w),g(V),g(G),g(R),g(U),g(j),g(M),g(H),g(q),g(J)}}}function on(x){let t=x[10]+"",r;return{c(){r=I(t)},l(a){r=z(a,t)},m(a,n){p(a,r,n)},p:we,d(a){a&&l(r)}}}function Gt(x){let t,r;return t=new Xt({props:{$$slots:{default:[on]},$$scope:{ctx:x}}}),{c(){$(t.$$.fragment)},l(a){d(t.$$.fragment,a)},m(a,n){_(t,a,n),r=!0},p(a,n){const i={};n&8192&&(i.$$scope={dirty:n,ctx:a}),t.$set(i)},i(a){r||(u(t.$$.fragment,a),r=!0)},o(a){h(t.$$.fragment,a),r=!1},d(a){g(t,a)}}}function rn(x){let t,r,a=x[0],n=[];for(let o=0;o<a.length;o+=1)n[o]=Gt(St(x,a,o));const i=o=>h(n[o],1,1,()=>{n[o]=null});return{c(){for(let o=0;o<n.length;o+=1)n[o].c();t=Le()},l(o){for(let c=0;c<n.length;c+=1)n[c].l(o);t=Le()},m(o,c){for(let s=0;s<n.length;s+=1)n[s]&&n[s].m(o,c);p(o,t,c),r=!0},p(o,c){if(c&1){a=o[0];let s;for(s=0;s<a.length;s+=1){const E=St(o,a,s);n[s]?(n[s].p(E,c),u(n[s],1)):(n[s]=Gt(E),n[s].c(),u(n[s],1),n[s].m(t.parentNode,t))}for(kt(),s=a.length;s<n.length;s+=1)i(s);Et()}},i(o){if(!r){for(let c=0;c<a.length;c+=1)u(n[c]);r=!0}},o(o){n=n.filter(Boolean);for(let c=0;c<n.length;c+=1)h(n[c]);r=!1},d(o){Vt(n,o),o&&l(t)}}}function sn(x){let t,r;return t=new qt({props:{$$slots:{default:[rn]},$$scope:{ctx:x}}}),{c(){$(t.$$.fragment)},l(a){d(t.$$.fragment,a)},m(a,n){_(t,a,n),r=!0},p(a,n){const i={};n&8192&&(i.$$scope={dirty:n,ctx:a}),t.$set(i)},i(a){r||(u(t.$$.fragment,a),r=!0)},o(a){h(t.$$.fragment,a),r=!1},d(a){g(t,a)}}}function ln(x){let t=x[7]+"",r;return{c(){r=I(t)},l(a){r=z(a,t)},m(a,n){p(a,r,n)},p:we,d(a){a&&l(r)}}}function cn(x){let t,r=x[7]+"",a;return{c(){t=P("span"),a=I(r),this.h()},l(n){t=S(n,"SPAN",{class:!0});var i=N(t);a=z(i,r),i.forEach(l),this.h()},h(){re(t,"class","inline-block bg-yellow-100 px-3 py-1 rounded-full")},m(n,i){p(n,t,i),T(t,a)},p:we,d(n){n&&l(t)}}}function fn(x){let t,r=x[7]+"",a;return{c(){t=P("span"),a=I(r),this.h()},l(n){t=S(n,"SPAN",{class:!0});var i=N(t);a=z(i,r),i.forEach(l),this.h()},h(){re(t,"class","inline-block bg-slate-200 px-3 py-1 rounded-full")},m(n,i){p(n,t,i),T(t,a)},p:we,d(n){n&&l(t)}}}function pn(x){let t,r=x[7]+"",a;return{c(){t=P("span"),a=I(r),this.h()},l(n){t=S(n,"SPAN",{class:!0});var i=N(t);a=z(i,r),i.forEach(l),this.h()},h(){re(t,"class","inline-block bg-red-100 px-3 py-1 rounded-full")},m(n,i){p(n,t,i),T(t,a)},p:we,d(n){n&&l(t)}}}function un(x){let t;function r(i,o){return i[7]==="Inception"?pn:i[7]==="Max Pooling"?fn:i[7]==="Basic Block"?cn:ln}let n=r(x)(x);return{c(){n.c(),t=Le()},l(i){n.l(i),t=Le()},m(i,o){n.m(i,o),p(i,t,o)},p(i,o){n.p(i,o)},d(i){n.d(i),i&&l(t)}}}function Wt(x){let t,r;return t=new en({props:{$$slots:{default:[un]},$$scope:{ctx:x}}}),{c(){$(t.$$.fragment)},l(a){d(t.$$.fragment,a)},m(a,n){_(t,a,n),r=!0},p(a,n){const i={};n&8192&&(i.$$scope={dirty:n,ctx:a}),t.$set(i)},i(a){r||(u(t.$$.fragment,a),r=!0)},o(a){h(t.$$.fragment,a),r=!1},d(a){g(t,a)}}}function hn(x){let t,r,a=x[4],n=[];for(let o=0;o<a.length;o+=1)n[o]=Wt(Pt(x,a,o));const i=o=>h(n[o],1,1,()=>{n[o]=null});return{c(){for(let o=0;o<n.length;o+=1)n[o].c();t=y()},l(o){for(let c=0;c<n.length;c+=1)n[c].l(o);t=k(o)},m(o,c){for(let s=0;s<n.length;s+=1)n[s]&&n[s].m(o,c);p(o,t,c),r=!0},p(o,c){if(c&2){a=o[4];let s;for(s=0;s<a.length;s+=1){const E=Pt(o,a,s);n[s]?(n[s].p(E,c),u(n[s],1)):(n[s]=Wt(E),n[s].c(),u(n[s],1),n[s].m(t.parentNode,t))}for(kt(),s=a.length;s<n.length;s+=1)i(s);Et()}},i(o){if(!r){for(let c=0;c<a.length;c+=1)u(n[c]);r=!0}},o(o){n=n.filter(Boolean);for(let c=0;c<n.length;c+=1)h(n[c]);r=!1},d(o){Vt(n,o),o&&l(t)}}}function Ct(x){let t,r;return t=new qt({props:{$$slots:{default:[hn]},$$scope:{ctx:x}}}),{c(){$(t.$$.fragment)},l(a){d(t.$$.fragment,a)},m(a,n){_(t,a,n),r=!0},p(a,n){const i={};n&8192&&(i.$$scope={dirty:n,ctx:a}),t.$set(i)},i(a){r||(u(t.$$.fragment,a),r=!0)},o(a){h(t.$$.fragment,a),r=!1},d(a){g(t,a)}}}function mn(x){let t,r,a=x[1],n=[];for(let o=0;o<a.length;o+=1)n[o]=Ct(Nt(x,a,o));const i=o=>h(n[o],1,1,()=>{n[o]=null});return{c(){for(let o=0;o<n.length;o+=1)n[o].c();t=Le()},l(o){for(let c=0;c<n.length;c+=1)n[c].l(o);t=Le()},m(o,c){for(let s=0;s<n.length;s+=1)n[s]&&n[s].m(o,c);p(o,t,c),r=!0},p(o,c){if(c&2){a=o[1];let s;for(s=0;s<a.length;s+=1){const E=Nt(o,a,s);n[s]?(n[s].p(E,c),u(n[s],1)):(n[s]=Ct(E),n[s].c(),u(n[s],1),n[s].m(t.parentNode,t))}for(kt(),s=a.length;s<n.length;s+=1)i(s);Et()}},i(o){if(!r){for(let c=0;c<a.length;c+=1)u(n[c]);r=!0}},o(o){n=n.filter(Boolean);for(let c=0;c<n.length;c+=1)h(n[c]);r=!1},d(o){Vt(n,o),o&&l(t)}}}function $n(x){let t,r,a,n;return t=new Zt({props:{$$slots:{default:[sn]},$$scope:{ctx:x}}}),a=new Kt({props:{$$slots:{default:[mn]},$$scope:{ctx:x}}}),{c(){$(t.$$.fragment),r=y(),$(a.$$.fragment)},l(i){d(t.$$.fragment,i),r=k(i),d(a.$$.fragment,i)},m(i,o){_(t,i,o),p(i,r,o),_(a,i,o),n=!0},p(i,o){const c={};o&8192&&(c.$$scope={dirty:o,ctx:i}),t.$set(c);const s={};o&8192&&(s.$$scope={dirty:o,ctx:i}),a.$set(s)},i(i){n||(u(t.$$.fragment,i),u(a.$$.fragment,i),n=!0)},o(i){h(t.$$.fragment,i),h(a.$$.fragment,i),n=!1},d(i){g(t,i),i&&l(r),g(a,i)}}}function dn(x){let t;return{c(){t=I("87%")},l(r){t=z(r,"87%")},m(r,a){p(r,t,a)},d(r){r&&l(t)}}}function _n(x){let t,r,a,n,i,o,c,s,E,D,C,w,V,G,R,U,j,M,H,q,J,Q,m,L,be,lt,Pe,ye,ct,Se,ke,ft,Ge,K,We,X,pt,se,ut,Ce,ee,ht,ie,mt,Me,le,Re,ce,qe,fe,Fe,pe,Ye,ue,je,Ee,$t,He,he,Oe,te,dt,Ae,_t,gt,Ue,me,Je,Ve,xt,Qe,$e,Ze,de,Ke,_e,Xe,ge,et,ne,vt,ae,wt,tt,xe,nt,ve,at;return a=new Dt({props:{$$slots:{default:[tn]},$$scope:{ctx:x}}}),n=new bt({props:{type:"reference",id:1}}),V=new Tt({props:{maxWidth:xn,$$slots:{default:[nn]},$$scope:{ctx:x}}}),M=new Tt({props:{maxWidth:vn,$$slots:{default:[an]},$$scope:{ctx:x}}}),m=new Jt({props:{imageWidth:6,imageHeight:6,kernel:1,showOutput:"true",numChannels:4,numFilters:1}}),K=new Qt({props:{$$slots:{default:[$n]},$$scope:{ctx:x}}}),se=new bt({props:{type:"reference",id:2}}),ie=new bt({props:{type:"reference",id:3}}),le=new F({props:{code:`import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets.cifar import CIFAR10
from torchvision import transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`}}),ce=new F({props:{code:`train_transform = T.Compose([T.Resize((50, 50)), 
                             T.ToTensor(),
                             T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])`}}),fe=new F({props:{code:"train_val_dataset = CIFAR10(root='../datasets', download=True, train=True, transform=train_transform)"}}),pe=new F({props:{code:`# split dataset into train and validate
indices = list(range(len(train_val_dataset)))
train_idxs, val_idxs = train_test_split(
    indices, test_size=0.1, stratify=train_val_dataset.targets
)

train_dataset = Subset(train_val_dataset, train_idxs)
val_dataset = Subset(train_val_dataset, val_idxs)`}}),ue=new F({props:{code:`batch_size=128
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
)`}}),he=new F({props:{code:`class BasicBlock(nn.Module):
    def __init__(self,
               in_channels,
               out_channels,
               **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      bias=False,
                      **kwargs),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU())
  
    def forward(self, x):
        return self.block(x)`}}),me=new F({props:{code:`class InceptionBlock(nn.Module):

    def __init__(self, 
                 in_channels, 
                 conv1x1_channels,
                 conv3x3_input_channels,
                 conv3x3_channels,
                 conv5x5_input_channels,
                 conv5x5_channels,
                 projection_channels):
        super().__init__()
        
        self.branch_1 = BasicBlock(in_channels=in_channels, 
                                  out_channels=conv1x1_channels,
                                  kernel_size=1)
        
        self.branch_2 = nn.Sequential(
            BasicBlock(in_channels=in_channels,
                        out_channels=conv3x3_input_channels,
                        kernel_size=1),
            BasicBlock(in_channels=conv3x3_input_channels,
                        out_channels=conv3x3_channels,
                        kernel_size=3,
                        padding=1))
        
        self.branch_3 = nn.Sequential(
            BasicBlock(in_channels=in_channels, 
                       out_channels=conv5x5_input_channels,
                       kernel_size=1),
            BasicBlock(in_channels=conv5x5_input_channels,
                       out_channels=conv5x5_channels,
                       kernel_size=5,
                       padding=2))
        
        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicBlock(in_channels, projection_channels, kernel_size=1),
        )

        
    def forward(self, x):
        return torch.cat([self.branch_1(x), 
                          self.branch_2(x), 
                          self.branch_3(x), 
                          self.branch_4(x)], dim=1)`}}),$e=new F({props:{code:`class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = BasicBlock(in_channels=3, 
                                out_channels=64, 
                                kernel_size=7, 
                                stride=2,
                                padding=3)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=3, 
                                       stride=2, 
                                       ceil_mode=True)

        self.conv_2 = BasicBlock(in_channels=64, 
                                out_channels=64,
                                kernel_size=1)

        self.conv_3 = BasicBlock(in_channels=64, 
                                out_channels=192, 
                                kernel_size=3, 
                                stride=1,
                                padding=1)
    
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception_3a = InceptionBlock(
            in_channels=192, 
            conv1x1_channels=64,
            conv3x3_input_channels=96,
            conv3x3_channels=128,
            conv5x5_input_channels=16,
            conv5x5_channels=32,
            projection_channels=32)

        self.inception_3b = InceptionBlock(
            in_channels=256, 
            conv1x1_channels=128,
            conv3x3_input_channels=128,
            conv3x3_channels=192,
            conv5x5_input_channels=32,
            conv5x5_channels=96,
            projection_channels=64)

        self.max_pool_3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.inception_4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.inception_5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, 10)


    def forward(self, x):
        x = self.conv_1(x)
        #x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.max_pool_2(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.max_pool_3(x)
        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        x = self.max_pool_4(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x`}}),de=new F({props:{code:`def track_performance(dataloader, model, criterion):
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
    return loss_sum / num_samples, num_correct / num_samples`}}),_e=new F({props:{code:`def train(
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
            scheduler.step(val_loss)`}}),ge=new F({props:{code:`model = Model()
optimizer = optim.Adam(params=model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=2, verbose=True
)
criterion = nn.CrossEntropyLoss(reduction="sum")`}}),ae=new Dt({props:{$$slots:{default:[dn]},$$scope:{ctx:x}}}),xe=new F({props:{code:`train(
    num_epochs=30,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
)`}}),ve=new F({props:{isOutput:!0,code:`Epoch:  1/30 | Epoch Duration: 21.662 sec | Val Loss: 1.02850 | Val Acc: 0.637 |
Epoch:  2/30 | Epoch Duration: 20.848 sec | Val Loss: 0.86810 | Val Acc: 0.713 |
Epoch:  3/30 | Epoch Duration: 20.950 sec | Val Loss: 0.75014 | Val Acc: 0.744 |
Epoch:  4/30 | Epoch Duration: 21.108 sec | Val Loss: 0.62963 | Val Acc: 0.785 |
Epoch:  5/30 | Epoch Duration: 21.120 sec | Val Loss: 0.62424 | Val Acc: 0.793 |
Epoch:  6/30 | Epoch Duration: 20.876 sec | Val Loss: 0.59486 | Val Acc: 0.814 |
Epoch:  7/30 | Epoch Duration: 20.860 sec | Val Loss: 0.59696 | Val Acc: 0.811 |
Epoch:  8/30 | Epoch Duration: 20.894 sec | Val Loss: 0.60809 | Val Acc: 0.818 |
Epoch:  9/30 | Epoch Duration: 21.068 sec | Val Loss: 0.87457 | Val Acc: 0.769 |
Epoch 00009: reducing learning rate of group 0 to 1.0000e-04.
Epoch: 10/30 | Epoch Duration: 20.961 sec | Val Loss: 0.45824 | Val Acc: 0.868 |
Epoch: 11/30 | Epoch Duration: 20.983 sec | Val Loss: 0.49140 | Val Acc: 0.867 |
Epoch: 12/30 | Epoch Duration: 21.150 sec | Val Loss: 0.51830 | Val Acc: 0.868 |
Epoch: 13/30 | Epoch Duration: 20.836 sec | Val Loss: 0.56201 | Val Acc: 0.869 |
Epoch 00013: reducing learning rate of group 0 to 1.0000e-05.
Epoch: 14/30 | Epoch Duration: 20.986 sec | Val Loss: 0.56474 | Val Acc: 0.868 |
Epoch: 15/30 | Epoch Duration: 20.946 sec | Val Loss: 0.55584 | Val Acc: 0.869 |
Epoch: 16/30 | Epoch Duration: 20.966 sec | Val Loss: 0.56621 | Val Acc: 0.870 |
Epoch 00016: reducing learning rate of group 0 to 1.0000e-06.
Epoch: 17/30 | Epoch Duration: 21.122 sec | Val Loss: 0.57022 | Val Acc: 0.868 |
Epoch: 18/30 | Epoch Duration: 20.916 sec | Val Loss: 0.57037 | Val Acc: 0.870 |
Epoch: 19/30 | Epoch Duration: 21.257 sec | Val Loss: 0.57713 | Val Acc: 0.867 |
Epoch 00019: reducing learning rate of group 0 to 1.0000e-07.
Epoch: 20/30 | Epoch Duration: 20.884 sec | Val Loss: 0.56507 | Val Acc: 0.869 |
Epoch: 21/30 | Epoch Duration: 21.388 sec | Val Loss: 0.56922 | Val Acc: 0.869 |
Epoch: 22/30 | Epoch Duration: 21.201 sec | Val Loss: 0.56943 | Val Acc: 0.869 |
Epoch 00022: reducing learning rate of group 0 to 1.0000e-08.
Epoch: 23/30 | Epoch Duration: 20.873 sec | Val Loss: 0.56877 | Val Acc: 0.869 |
Epoch: 24/30 | Epoch Duration: 21.181 sec | Val Loss: 0.56954 | Val Acc: 0.867 |
Epoch: 25/30 | Epoch Duration: 20.905 sec | Val Loss: 0.56815 | Val Acc: 0.871 |
Epoch: 26/30 | Epoch Duration: 20.958 sec | Val Loss: 0.56737 | Val Acc: 0.869 |
Epoch: 27/30 | Epoch Duration: 21.091 sec | Val Loss: 0.56780 | Val Acc: 0.868 |
Epoch: 28/30 | Epoch Duration: 21.017 sec | Val Loss: 0.56894 | Val Acc: 0.869 |
Epoch: 29/30 | Epoch Duration: 21.117 sec | Val Loss: 0.56901 | Val Acc: 0.871 |
Epoch: 30/30 | Epoch Duration: 20.952 sec | Val Loss: 0.56453 | Val Acc: 0.869 |`}}),{c(){t=P("p"),r=I("The "),$(a.$$.fragment),$(n.$$.fragment),i=I(` architecture was developed by researchers at Google, but the name is also
    a reference to the original LeNet-5 architecture, a sign of respect for Yann
    LeCun. GoogLeNet achieved a top-5 error rate of 6.67% (VGG achieved 7.32%) and
    won the 2014 ImageNet classification challenge.`),o=y(),c=P("p"),s=I(`The GoogLeNet network is a specific, 22 layer, realization of the so called
    Inception architecture. This architecture uses an Inception block, a
    multibranch block that applies convolutions of different filter sizes to the
    same input and concatenates the results in the final step. This architecture
    choice removes the need to search for the optimal patch size and allows the
    creation of much deeper neural networks, while being very efficient at the
    same time. In fact the GoogLeNet architecture uses 12x fewer parameters than
    AlexNet.`),E=y(),D=P("p"),C=I(`In the very first step we create a basic building block that is going to be
    utilized in each convolutional layer. The block constists of a convolutional
    layer with variable filter and feature map size. The convolution is followed
    by a batch norm layer and a ReLU activation function. In the original
    implementation batch normalization was not used, instead in order to deal
    with vanishing gradients, the authors implemented several losses along the
    path of the neural network. This approach is very uncommon and we are not
    going to implement these so called auxilary losses. Batch normalization is a
    much simpler and practical approach.`),w=y(),$(V.$$.fragment),G=y(),R=P("p"),U=I(`The Inception block takes an input from a previous layer and applies
    calculations in 4 different branches using the basic block from above. At
    the end the four branches are concatenated into a single output.`),j=y(),$(M.$$.fragment),H=y(),q=P("p"),J=I(`You will notice that aside from the expected 3x3 convolutions, 5x5
    convolutions and max pooling, there is a 1x1 convolution in each single
    branch. You might suspect that the 1x1 convolution operation produces an
    output, that is equal to the input. If you think that, then your intuition
    is wrong. Remember that the convolution operation is applied to all feature
    maps in the previous layer. While the width and the height after the 1x1
    convolution remain the same, the number of filters can be changed
    arbitrarily. Below for example we take 4 feature maps as input and return
    just one single feature map.`),Q=y(),$(m.$$.fragment),L=y(),be=P("p"),lt=I(`This operation allows us to reduce the number of feature maps in order to
    save computational power. This is especially relevant for the 3x3 and 5x5
    filters, as those require a lot of weights, when the number of filter grows.
    That means that in the inception block we reduce the number of filters,
    before we apply the 3x3 and 5x5 filters.`),Pe=y(),ye=P("p"),ct=I(`You should also bear in mind that in each branch the size of the feature
    maps have to match. If they wouldn't, you would not be able to concatenate
    the branches in the last step. The number of channels after the
    concatenation corresponds to the sum of the channels from the four branches.`),Se=y(),ke=P("p"),ft=I(`The overall GoogLeNet architecture combines many layers of Inception and
    Pooling blocks. You can get the exact parameters either by studying the
    original paper.`),Ge=y(),$(K.$$.fragment),We=y(),X=P("p"),pt=I(`Aside from the inception blocks, there are more details, that we have not
    seen so far. With AlexNet for example we used several fully connected layers
    in the classification block. We did that to slowly move from the flattened
    vector to the number of neurons that are used as input into the softmax
    layer. In the GoogLeNet architecture the last pooling layer removes the
    width and length and we use a single fully connected layer before the
    sigmoid/softmax layer. Such a procedure is quite common nowadays. Fully
    connected layers require many parameters and the approach above avoids
    unnecessary calculations (see Lin et al. `),$(se.$$.fragment),ut=I(" for more info)."),Ce=y(),ee=P("p"),ht=I(`Be aware that the architecture we have discussed above is often called
    InceptionV1. The field of deep learning improved very fast, which resulted
    in the improved InceptionV2 and InceptionV3`),$(ie.$$.fragment),mt=I(" architectues. Below we will implement the original GoogLeNet architecture."),Me=y(),$(le.$$.fragment),Re=y(),$(ce.$$.fragment),qe=y(),$(fe.$$.fragment),Fe=y(),$(pe.$$.fragment),Ye=y(),$(ue.$$.fragment),je=y(),Ee=P("p"),$t=I(`The basic block module is going to be used extensively for the inception
    module below.`),He=y(),$(he.$$.fragment),Oe=y(),te=P("p"),dt=I(`The four branches of the inception module run separately first, but are then
    concatenated on the channel dimension (`),Ae=P("code"),_t=I("dim=1"),gt=I(")."),Ue=y(),$(me.$$.fragment),Je=y(),Ve=P("p"),xt=I(`Finally we implement the GoogLeNet module. The parameters used below were
    taken from the original paper. In the forward pass we comment out one of the
    max pooling layers. We do that, because our images are significantly smaller
    than the ImageNet images and we would run into errors if we included the
    additional pooling layer.`),Qe=y(),$($e.$$.fragment),Ze=y(),$(de.$$.fragment),Ke=y(),$(_e.$$.fragment),Xe=y(),$(ge.$$.fragment),et=y(),ne=P("p"),vt=I("The GoogLeNet model produces an accurancy of close to "),$(ae.$$.fragment),wt=I(`. We do not beat our VGG implementation, but our runtime is significantly
    reduced.`),tt=y(),$(xe.$$.fragment),nt=y(),$(ve.$$.fragment)},l(e){t=S(e,"P",{});var f=N(t);r=z(f,"The "),d(a.$$.fragment,f),d(n.$$.fragment,f),i=z(f,` architecture was developed by researchers at Google, but the name is also
    a reference to the original LeNet-5 architecture, a sign of respect for Yann
    LeCun. GoogLeNet achieved a top-5 error rate of 6.67% (VGG achieved 7.32%) and
    won the 2014 ImageNet classification challenge.`),f.forEach(l),o=k(e),c=S(e,"P",{});var Ie=N(c);s=z(Ie,`The GoogLeNet network is a specific, 22 layer, realization of the so called
    Inception architecture. This architecture uses an Inception block, a
    multibranch block that applies convolutions of different filter sizes to the
    same input and concatenates the results in the final step. This architecture
    choice removes the need to search for the optimal patch size and allows the
    creation of much deeper neural networks, while being very efficient at the
    same time. In fact the GoogLeNet architecture uses 12x fewer parameters than
    AlexNet.`),Ie.forEach(l),E=k(e),D=S(e,"P",{});var ze=N(D);C=z(ze,`In the very first step we create a basic building block that is going to be
    utilized in each convolutional layer. The block constists of a convolutional
    layer with variable filter and feature map size. The convolution is followed
    by a batch norm layer and a ReLU activation function. In the original
    implementation batch normalization was not used, instead in order to deal
    with vanishing gradients, the authors implemented several losses along the
    path of the neural network. This approach is very uncommon and we are not
    going to implement these so called auxilary losses. Batch normalization is a
    much simpler and practical approach.`),ze.forEach(l),w=k(e),d(V.$$.fragment,e),G=k(e),R=S(e,"P",{});var Te=N(R);U=z(Te,`The Inception block takes an input from a previous layer and applies
    calculations in 4 different branches using the basic block from above. At
    the end the four branches are concatenated into a single output.`),Te.forEach(l),j=k(e),d(M.$$.fragment,e),H=k(e),q=S(e,"P",{});var De=N(q);J=z(De,`You will notice that aside from the expected 3x3 convolutions, 5x5
    convolutions and max pooling, there is a 1x1 convolution in each single
    branch. You might suspect that the 1x1 convolution operation produces an
    output, that is equal to the input. If you think that, then your intuition
    is wrong. Remember that the convolution operation is applied to all feature
    maps in the previous layer. While the width and the height after the 1x1
    convolution remain the same, the number of filters can be changed
    arbitrarily. Below for example we take 4 feature maps as input and return
    just one single feature map.`),De.forEach(l),Q=k(e),d(m.$$.fragment,e),L=k(e),be=S(e,"P",{});var Ne=N(be);lt=z(Ne,`This operation allows us to reduce the number of feature maps in order to
    save computational power. This is especially relevant for the 3x3 and 5x5
    filters, as those require a lot of weights, when the number of filter grows.
    That means that in the inception block we reduce the number of filters,
    before we apply the 3x3 and 5x5 filters.`),Ne.forEach(l),Pe=k(e),ye=S(e,"P",{});var Bt=N(ye);ct=z(Bt,`You should also bear in mind that in each branch the size of the feature
    maps have to match. If they wouldn't, you would not be able to concatenate
    the branches in the last step. The number of channels after the
    concatenation corresponds to the sum of the channels from the four branches.`),Bt.forEach(l),Se=k(e),ke=S(e,"P",{});var Lt=N(ke);ft=z(Lt,`The overall GoogLeNet architecture combines many layers of Inception and
    Pooling blocks. You can get the exact parameters either by studying the
    original paper.`),Lt.forEach(l),Ge=k(e),d(K.$$.fragment,e),We=k(e),X=S(e,"P",{});var ot=N(X);pt=z(ot,`Aside from the inception blocks, there are more details, that we have not
    seen so far. With AlexNet for example we used several fully connected layers
    in the classification block. We did that to slowly move from the flattened
    vector to the number of neurons that are used as input into the softmax
    layer. In the GoogLeNet architecture the last pooling layer removes the
    width and length and we use a single fully connected layer before the
    sigmoid/softmax layer. Such a procedure is quite common nowadays. Fully
    connected layers require many parameters and the approach above avoids
    unnecessary calculations (see Lin et al. `),d(se.$$.fragment,ot),ut=z(ot," for more info)."),ot.forEach(l),Ce=k(e),ee=S(e,"P",{});var rt=N(ee);ht=z(rt,`Be aware that the architecture we have discussed above is often called
    InceptionV1. The field of deep learning improved very fast, which resulted
    in the improved InceptionV2 and InceptionV3`),d(ie.$$.fragment,rt),mt=z(rt," architectues. Below we will implement the original GoogLeNet architecture."),rt.forEach(l),Me=k(e),d(le.$$.fragment,e),Re=k(e),d(ce.$$.fragment,e),qe=k(e),d(fe.$$.fragment,e),Fe=k(e),d(pe.$$.fragment,e),Ye=k(e),d(ue.$$.fragment,e),je=k(e),Ee=S(e,"P",{});var At=N(Ee);$t=z(At,`The basic block module is going to be used extensively for the inception
    module below.`),At.forEach(l),He=k(e),d(he.$$.fragment,e),Oe=k(e),te=S(e,"P",{});var st=N(te);dt=z(st,`The four branches of the inception module run separately first, but are then
    concatenated on the channel dimension (`),Ae=S(st,"CODE",{});var It=N(Ae);_t=z(It,"dim=1"),It.forEach(l),gt=z(st,")."),st.forEach(l),Ue=k(e),d(me.$$.fragment,e),Je=k(e),Ve=S(e,"P",{});var zt=N(Ve);xt=z(zt,`Finally we implement the GoogLeNet module. The parameters used below were
    taken from the original paper. In the forward pass we comment out one of the
    max pooling layers. We do that, because our images are significantly smaller
    than the ImageNet images and we would run into errors if we included the
    additional pooling layer.`),zt.forEach(l),Qe=k(e),d($e.$$.fragment,e),Ze=k(e),d(de.$$.fragment,e),Ke=k(e),d(_e.$$.fragment,e),Xe=k(e),d(ge.$$.fragment,e),et=k(e),ne=S(e,"P",{});var it=N(ne);vt=z(it,"The GoogLeNet model produces an accurancy of close to "),d(ae.$$.fragment,it),wt=z(it,`. We do not beat our VGG implementation, but our runtime is significantly
    reduced.`),it.forEach(l),tt=k(e),d(xe.$$.fragment,e),nt=k(e),d(ve.$$.fragment,e)},m(e,f){p(e,t,f),T(t,r),_(a,t,null),_(n,t,null),T(t,i),p(e,o,f),p(e,c,f),T(c,s),p(e,E,f),p(e,D,f),T(D,C),p(e,w,f),_(V,e,f),p(e,G,f),p(e,R,f),T(R,U),p(e,j,f),_(M,e,f),p(e,H,f),p(e,q,f),T(q,J),p(e,Q,f),_(m,e,f),p(e,L,f),p(e,be,f),T(be,lt),p(e,Pe,f),p(e,ye,f),T(ye,ct),p(e,Se,f),p(e,ke,f),T(ke,ft),p(e,Ge,f),_(K,e,f),p(e,We,f),p(e,X,f),T(X,pt),_(se,X,null),T(X,ut),p(e,Ce,f),p(e,ee,f),T(ee,ht),_(ie,ee,null),T(ee,mt),p(e,Me,f),_(le,e,f),p(e,Re,f),_(ce,e,f),p(e,qe,f),_(fe,e,f),p(e,Fe,f),_(pe,e,f),p(e,Ye,f),_(ue,e,f),p(e,je,f),p(e,Ee,f),T(Ee,$t),p(e,He,f),_(he,e,f),p(e,Oe,f),p(e,te,f),T(te,dt),T(te,Ae),T(Ae,_t),T(te,gt),p(e,Ue,f),_(me,e,f),p(e,Je,f),p(e,Ve,f),T(Ve,xt),p(e,Qe,f),_($e,e,f),p(e,Ze,f),_(de,e,f),p(e,Ke,f),_(_e,e,f),p(e,Xe,f),_(ge,e,f),p(e,et,f),p(e,ne,f),T(ne,vt),_(ae,ne,null),T(ne,wt),p(e,tt,f),_(xe,e,f),p(e,nt,f),_(ve,e,f),at=!0},p(e,f){const Ie={};f&8192&&(Ie.$$scope={dirty:f,ctx:e}),a.$set(Ie);const ze={};f&8192&&(ze.$$scope={dirty:f,ctx:e}),V.$set(ze);const Te={};f&8192&&(Te.$$scope={dirty:f,ctx:e}),M.$set(Te);const De={};f&8192&&(De.$$scope={dirty:f,ctx:e}),K.$set(De);const Ne={};f&8192&&(Ne.$$scope={dirty:f,ctx:e}),ae.$set(Ne)},i(e){at||(u(a.$$.fragment,e),u(n.$$.fragment,e),u(V.$$.fragment,e),u(M.$$.fragment,e),u(m.$$.fragment,e),u(K.$$.fragment,e),u(se.$$.fragment,e),u(ie.$$.fragment,e),u(le.$$.fragment,e),u(ce.$$.fragment,e),u(fe.$$.fragment,e),u(pe.$$.fragment,e),u(ue.$$.fragment,e),u(he.$$.fragment,e),u(me.$$.fragment,e),u($e.$$.fragment,e),u(de.$$.fragment,e),u(_e.$$.fragment,e),u(ge.$$.fragment,e),u(ae.$$.fragment,e),u(xe.$$.fragment,e),u(ve.$$.fragment,e),at=!0)},o(e){h(a.$$.fragment,e),h(n.$$.fragment,e),h(V.$$.fragment,e),h(M.$$.fragment,e),h(m.$$.fragment,e),h(K.$$.fragment,e),h(se.$$.fragment,e),h(ie.$$.fragment,e),h(le.$$.fragment,e),h(ce.$$.fragment,e),h(fe.$$.fragment,e),h(pe.$$.fragment,e),h(ue.$$.fragment,e),h(he.$$.fragment,e),h(me.$$.fragment,e),h($e.$$.fragment,e),h(de.$$.fragment,e),h(_e.$$.fragment,e),h(ge.$$.fragment,e),h(ae.$$.fragment,e),h(xe.$$.fragment,e),h(ve.$$.fragment,e),at=!1},d(e){e&&l(t),g(a),g(n),e&&l(o),e&&l(c),e&&l(E),e&&l(D),e&&l(w),g(V,e),e&&l(G),e&&l(R),e&&l(j),g(M,e),e&&l(H),e&&l(q),e&&l(Q),g(m,e),e&&l(L),e&&l(be),e&&l(Pe),e&&l(ye),e&&l(Se),e&&l(ke),e&&l(Ge),g(K,e),e&&l(We),e&&l(X),g(se),e&&l(Ce),e&&l(ee),g(ie),e&&l(Me),g(le,e),e&&l(Re),g(ce,e),e&&l(qe),g(fe,e),e&&l(Fe),g(pe,e),e&&l(Ye),g(ue,e),e&&l(je),e&&l(Ee),e&&l(He),g(he,e),e&&l(Oe),e&&l(te),e&&l(Ue),g(me,e),e&&l(Je),e&&l(Ve),e&&l(Qe),g($e,e),e&&l(Ze),g(de,e),e&&l(Ke),g(_e,e),e&&l(Xe),g(ge,e),e&&l(et),e&&l(ne),g(ae),e&&l(tt),g(xe,e),e&&l(nt),g(ve,e)}}}function gn(x){let t,r,a,n,i,o,c,s,E,D,C;return s=new Ot({props:{$$slots:{default:[_n]},$$scope:{ctx:x}}}),D=new Ut({props:{references:x[2]}}),{c(){t=P("meta"),r=y(),a=P("h1"),n=I("GoogLeNet"),i=y(),o=P("div"),c=y(),$(s.$$.fragment),E=y(),$(D.$$.fragment),this.h()},l(w){const V=Ht("svelte-1w8nka8",document.head);t=S(V,"META",{name:!0,content:!0}),V.forEach(l),r=k(w),a=S(w,"H1",{});var G=N(a);n=z(G,"GoogLeNet"),G.forEach(l),i=k(w),o=S(w,"DIV",{class:!0}),N(o).forEach(l),c=k(w),d(s.$$.fragment,w),E=k(w),d(D.$$.fragment,w),this.h()},h(){document.title="GoogLeNet - World4Ai",re(t,"name","description"),re(t,"content","The GoogLeNet architecture combines several layers of Inception modules to create a deep convolutional neural network. An inception module simultaneously calculates convolutions with different kernel sizes using the same input and the results are then concatenated."),re(o,"class","separator")},m(w,V){T(document.head,t),p(w,r,V),p(w,a,V),T(a,n),p(w,i,V),p(w,o,V),p(w,c,V),_(s,w,V),p(w,E,V),_(D,w,V),C=!0},p(w,[V]){const G={};V&8192&&(G.$$scope={dirty:V,ctx:w}),s.$set(G)},i(w){C||(u(s.$$.fragment,w),u(D.$$.fragment,w),C=!0)},o(w){h(s.$$.fragment,w),h(D.$$.fragment,w),C=!1},d(w){l(t),w&&l(r),w&&l(a),w&&l(i),w&&l(o),w&&l(c),g(s,w),w&&l(E),g(D,w)}}}let W=1,oe=90,Be=100,yt=80,Z=20,xn="200px",B=1e3,v=500,A=210,b=70,vn="800px";function wn(x){return[["Type","Input Size","Output Size"],[["Basic Block","224x224x3","112x112x64"],["Max Pooling","112x112x64","56x56x64"],["Basic Block","56x56x64","56x56x64"],["Basic Block","56x56x64","56x56x192"],["Max Pooling","56x56x192","28x28x192"],["Inception","28x28x192","28x28x256"],["Inception","28x28x256","28x28x480"],["Max Pooling","28x28x480","14x14x480"],["Inception","14x14x480","14x14x512"],["Inception","14x14x512","14x14x512"],["Inception","14x14x512","14x14x512"],["Inception","14x14x512","14x14x528"],["Inception","14x14x528","14x14x832"],["Max Pooling","14x14x832","7x7x832"],["Inception","7x7x832","7x7x832"],["Inception","7x7x832","7x7x1024"],["Avg. Pooling","7x7x1024","1x1x1024"],["Dropout","-","-"],["Fully Connected","1024","1000"],["Softmax","1000","1000"]],[{author:"Szegedy, Christian and Wei Liu and Yangqing Jia and Sermanet, Pierre and Reed, Scott and Anguelov, Dragomir and Erhan, Dumitru and Vanhoucke, Vincent and Rabinovich, Andrew",title:"Going deeper with convolutions",journal:"2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",year:"2015",pages:"1-9",volume:"",issue:""},{author:"Lin, M., Chen, Q., & Yan, S.",title:"Network in Network",journal:"",year:"2013",pages:"",volume:"",issue:""},{author:"Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, ZB",title:"Rethinking the Inception Architecture for Computer Vision",journal:"",year:"2016",pages:"",volume:"",issue:""}]]}class Tn extends Ft{constructor(t){super(),Yt(this,t,wn,gn,jt,{})}}export{Tn as default};
