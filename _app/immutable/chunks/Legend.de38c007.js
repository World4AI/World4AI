import{S as d,i as k,s as v,Q as z,a as w,q as L,R as C,m as g,h as u,c as q,r as E,n as s,b as h,N,u as Q,C as S,a8 as x}from"./index.4d92b023.js";function R(n){let t,f,l,_,r,o;return{c(){t=z("rect"),f=w(),l=z("text"),_=L(n[0]),this.h()},l(e){t=C(e,"rect",{x:!0,y:!0,width:!0,height:!0,fill:!0,stroke:!0}),g(t).forEach(u),f=q(e),l=C(e,"text",{fill:!0,x:!0,y:!0,"font-size":!0,class:!0});var i=g(l);_=E(i,n[0]),i.forEach(u),this.h()},h(){s(t,"x",n[8]),s(t,"y",n[9]),s(t,"width",n[4]),s(t,"height",n[4]),s(t,"fill",n[2]),s(t,"stroke","black"),s(l,"fill",n[1]),s(l,"x",r=n[4]*2+n[6](n[3].x)),s(l,"y",o=n[7](n[3].y)),s(l,"font-size",n[5]),s(l,"class","svelte-9icl3e")},m(e,i){h(e,t,i),h(e,f,i),h(e,l,i),N(l,_)},p(e,[i]){i&16&&s(t,"width",e[4]),i&16&&s(t,"height",e[4]),i&4&&s(t,"fill",e[2]),i&1&&Q(_,e[0]),i&2&&s(l,"fill",e[1]),i&24&&r!==(r=e[4]*2+e[6](e[3].x))&&s(l,"x",r),i&8&&o!==(o=e[7](e[3].y))&&s(l,"y",o),i&32&&s(l,"font-size",e[5])},i:S,o:S,d(e){e&&u(t),e&&u(f),e&&u(l)}}}function X(n,t,f){let{text:l=""}=t,{textColor:_="black"}=t,{legendColor:r="black"}=t,{coordinates:o={}}=t,{size:e=5}=t,{fontSize:i=12}=t;const c=x("xScale"),m=x("yScale");let y=c(o.x)-e/2,b=m(o.y)-e/2;return n.$$set=a=>{"text"in a&&f(0,l=a.text),"textColor"in a&&f(1,_=a.textColor),"legendColor"in a&&f(2,r=a.legendColor),"coordinates"in a&&f(3,o=a.coordinates),"size"in a&&f(4,e=a.size),"fontSize"in a&&f(5,i=a.fontSize)},[l,_,r,o,e,i,c,m,y,b]}class j extends d{constructor(t){super(),k(this,t,X,R,v,{text:0,textColor:1,legendColor:2,coordinates:3,size:4,fontSize:5})}}export{j as L};
