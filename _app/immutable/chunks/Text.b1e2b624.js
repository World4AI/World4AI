import{S as m,i as y,s as S,Q as z,q as C,R as g,m as v,r as T,h as x,n as i,b,N as q,u as d,C as u,a8 as h}from"./index.4d92b023.js";function k(n){let e,a,f,_;return{c(){e=z("text"),a=C(n[0]),this.h()},l(t){e=g(t,"text",{"font-size":!0,fill:!0,x:!0,y:!0,class:!0});var l=v(e);a=T(l,n[0]),l.forEach(x),this.h()},h(){i(e,"font-size",n[4]),i(e,"fill",n[1]),i(e,"x",f=n[5](n[2])),i(e,"y",_=n[6](n[3])),i(e,"class","svelte-9icl3e")},m(t,l){b(t,e,l),q(e,a)},p(t,[l]){l&1&&d(a,t[0]),l&16&&i(e,"font-size",t[4]),l&2&&i(e,"fill",t[1]),l&4&&f!==(f=t[5](t[2]))&&i(e,"x",f),l&8&&_!==(_=t[6](t[3]))&&i(e,"y",_)},i:u,o:u,d(t){t&&x(e)}}}function E(n,e,a){let{text:f=""}=e,{textColor:_="black"}=e,{x:t}=e,{y:l}=e,{fontSize:o=12}=e;const r=h("xScale"),c=h("yScale");return n.$$set=s=>{"text"in s&&a(0,f=s.text),"textColor"in s&&a(1,_=s.textColor),"x"in s&&a(2,t=s.x),"y"in s&&a(3,l=s.y),"fontSize"in s&&a(4,o=s.fontSize)},[f,_,t,l,o,r,c]}class Q extends m{constructor(e){super(),y(this,e,E,k,S,{text:0,textColor:1,x:2,y:3,fontSize:4})}}export{Q as T};
