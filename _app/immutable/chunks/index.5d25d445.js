import{c as d,a as h}from"./index.7e899070.js";import{_ as l}from"./index.4d92b023.js";function x(e,{delay:a=0,duration:n=400,easing:t=d,x:f=0,y:s=0,opacity:o=0}={}){const c=getComputedStyle(e),r=+c.opacity,y=c.transform==="none"?"":c.transform,p=r*(1-o),[u,m]=l(f),[$,g]=l(s);return{delay:a,duration:n,easing:t,css:(i,_)=>`
			transform: ${y} translate(${(1-i)*u}${m}, ${(1-i)*$}${g});
			opacity: ${r-p*_}`}}function C(e,{delay:a=0,speed:n,duration:t,easing:f=h}={}){let s=e.getTotalLength();const o=getComputedStyle(e);return o.strokeLinecap!=="butt"&&(s+=parseInt(o.strokeWidth)),t===void 0?n===void 0?t=800:t=s/n:typeof t=="function"&&(t=t(s)),{delay:a,duration:t,easing:f,css:(c,r)=>`
			stroke-dasharray: ${s};
			stroke-dashoffset: ${r*s};
		`}}export{C as d,x as f};
