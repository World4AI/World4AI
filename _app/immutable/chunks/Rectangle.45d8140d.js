import{S as y,i as v,s as S,e as f,b as d,C as _,P as w,h as r,a8 as u,Q as z,R,m as b,n as h}from"./index.4d92b023.js";function g(i,t,s){const l=i.slice();return l[5]=t[s],l}function m(i){let t,s,l;return{c(){t=z("rect"),this.h()},l(e){t=R(e,"rect",{fill:!0,x:!0,y:!0,height:!0,width:!0,class:!0}),b(t).forEach(r),this.h()},h(){h(t,"fill",i[1]),h(t,"x",s=i[3](i[5].x)-i[2]/2),h(t,"y",l=i[4](i[5].y)-i[2]/2),h(t,"height",i[2]),h(t,"width",i[2]),h(t,"class","svelte-qywhw")},m(e,a){d(e,t,a)},p(e,a){a&2&&h(t,"fill",e[1]),a&5&&s!==(s=e[3](e[5].x)-e[2]/2)&&h(t,"x",s),a&5&&l!==(l=e[4](e[5].y)-e[2]/2)&&h(t,"y",l),a&4&&h(t,"height",e[2]),a&4&&h(t,"width",e[2])},d(e){e&&r(t)}}}function C(i){let t,s=i[0],l=[];for(let e=0;e<s.length;e+=1)l[e]=m(g(i,s,e));return{c(){for(let e=0;e<l.length;e+=1)l[e].c();t=f()},l(e){for(let a=0;a<l.length;a+=1)l[a].l(e);t=f()},m(e,a){for(let n=0;n<l.length;n+=1)l[n]&&l[n].m(e,a);d(e,t,a)},p(e,[a]){if(a&31){s=e[0];let n;for(n=0;n<s.length;n+=1){const c=g(e,s,n);l[n]?l[n].p(c,a):(l[n]=m(c),l[n].c(),l[n].m(t.parentNode,t))}for(;n<l.length;n+=1)l[n].d(1);l.length=s.length}},i:_,o:_,d(e){w(l,e),e&&r(t)}}}function k(i,t,s){let{data:l}=t,{color:e="var(--main-color-1)"}=t,{size:a=5}=t,n=u("xScale"),c=u("yScale");return i.$$set=o=>{"data"in o&&s(0,l=o.data),"color"in o&&s(1,e=o.color),"size"in o&&s(2,a=o.size)},[l,e,a,n,c]}class E extends y{constructor(t){super(),v(this,t,k,C,S,{data:0,color:1,size:2})}}export{E as R};
