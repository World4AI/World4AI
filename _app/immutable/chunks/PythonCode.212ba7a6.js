import{S as pe,i as be,s as _e,D as F,k as fe,l as he,m as de,h as k,M as z,$ as Ie,b as $,N as $e,E as Xe,C as Ce,U as se,G as Ye,q as wt,r as mt,u as Ot,a0 as Je,e as ie,a1 as Ze,H as Nt,I as At,J as xt,K as kt,g as G,d as W,a2 as Tt,j as jt,y as Ee,z as ye,A as Re,F as vt,B as Me,a as Bt,W as Lt,c as It,n as Ct,v as Dt,f as Pt}from"./index.4d92b023.js";var Se={exports:{}};function we(e){return e instanceof Map?e.clear=e.delete=e.set=function(){throw new Error("map is read-only")}:e instanceof Set&&(e.add=e.clear=e.delete=function(){throw new Error("set is read-only")}),Object.freeze(e),Object.getOwnPropertyNames(e).forEach(function(t){var n=e[t];typeof n=="object"&&!Object.isFrozen(n)&&we(n)}),e}Se.exports=we;Se.exports.default=we;class De{constructor(t){t.data===void 0&&(t.data={}),this.data=t.data,this.isMatchIgnored=!1}ignoreMatch(){this.isMatchIgnored=!0}}function Ve(e){return e.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#x27;")}function D(e,...t){const n=Object.create(null);for(const s in e)n[s]=e[s];return t.forEach(function(s){for(const c in s)n[c]=s[c]}),n}const Ht="</span>",Pe=e=>!!e.scope||e.sublanguage&&e.language,Ut=(e,{prefix:t})=>{if(e.includes(".")){const n=e.split(".");return[`${t}${n.shift()}`,...n.map((s,c)=>`${s}${"_".repeat(c+1)}`)].join(" ")}return`${t}${e}`};class Kt{constructor(t,n){this.buffer="",this.classPrefix=n.classPrefix,t.walk(this)}addText(t){this.buffer+=Ve(t)}openNode(t){if(!Pe(t))return;let n="";t.sublanguage?n=`language-${t.language}`:n=Ut(t.scope,{prefix:this.classPrefix}),this.span(n)}closeNode(t){Pe(t)&&(this.buffer+=Ht)}value(){return this.buffer}span(t){this.buffer+=`<span class="${t}">`}}const He=(e={})=>{const t={children:[]};return Object.assign(t,e),t};class me{constructor(){this.rootNode=He(),this.stack=[this.rootNode]}get top(){return this.stack[this.stack.length-1]}get root(){return this.rootNode}add(t){this.top.children.push(t)}openNode(t){const n=He({scope:t});this.add(n),this.stack.push(n)}closeNode(){if(this.stack.length>1)return this.stack.pop()}closeAllNodes(){for(;this.closeNode(););}toJSON(){return JSON.stringify(this.rootNode,null,4)}walk(t){return this.constructor._walk(t,this.rootNode)}static _walk(t,n){return typeof n=="string"?t.addText(n):n.children&&(t.openNode(n),n.children.forEach(s=>this._walk(t,s)),t.closeNode(n)),t}static _collapse(t){typeof t!="string"&&t.children&&(t.children.every(n=>typeof n=="string")?t.children=[t.children.join("")]:t.children.forEach(n=>{me._collapse(n)}))}}class Ft extends me{constructor(t){super(),this.options=t}addKeyword(t,n){t!==""&&(this.openNode(n),this.addText(t),this.closeNode())}addText(t){t!==""&&this.add(t)}addSublanguage(t,n){const s=t.root;s.sublanguage=!0,s.language=n,this.add(s)}toHTML(){return new Kt(this,this.options).value()}finalize(){return!0}}function X(e){return e?typeof e=="string"?e:e.source:null}function qe(e){return U("(?=",e,")")}function Gt(e){return U("(?:",e,")*")}function Wt(e){return U("(?:",e,")?")}function U(...e){return e.map(n=>X(n)).join("")}function zt(e){const t=e[e.length-1];return typeof t=="object"&&t.constructor===Object?(e.splice(e.length-1,1),t):{}}function Oe(...e){return"("+(zt(e).capture?"":"?:")+e.map(s=>X(s)).join("|")+")"}function Qe(e){return new RegExp(e.toString()+"|").exec("").length-1}function $t(e,t){const n=e&&e.exec(t);return n&&n.index===0}const Xt=/\[(?:[^\\\]]|\\.)*\]|\(\??|\\([1-9][0-9]*)|\\./;function Ne(e,{joinWith:t}){let n=0;return e.map(s=>{n+=1;const c=n;let r=X(s),i="";for(;r.length>0;){const a=Xt.exec(r);if(!a){i+=r;break}i+=r.substring(0,a.index),r=r.substring(a.index+a[0].length),a[0][0]==="\\"&&a[1]?i+="\\"+String(Number(a[1])+c):(i+=a[0],a[0]==="("&&n++)}return i}).map(s=>`(${s})`).join(t)}const Yt=/\b\B/,et="[a-zA-Z]\\w*",Ae="[a-zA-Z_]\\w*",tt="\\b\\d+(\\.\\d+)?",nt="(-?)(\\b0[xX][a-fA-F0-9]+|(\\b\\d+(\\.\\d*)?|\\.\\d+)([eE][-+]?\\d+)?)",st="\\b(0b[01]+)",Jt="!|!=|!==|%|%=|&|&&|&=|\\*|\\*=|\\+|\\+=|,|-|-=|/=|/|:|;|<<|<<=|<=|<|===|==|=|>>>=|>>=|>=|>>>|>>|>|\\?|\\[|\\{|\\(|\\^|\\^=|\\||\\|=|\\|\\||~",Zt=(e={})=>{const t=/^#![ ]*\//;return e.binary&&(e.begin=U(t,/.*\b/,e.binary,/\b.*/)),D({scope:"meta",begin:t,end:/$/,relevance:0,"on:begin":(n,s)=>{n.index!==0&&s.ignoreMatch()}},e)},Y={begin:"\\\\[\\s\\S]",relevance:0},Vt={scope:"string",begin:"'",end:"'",illegal:"\\n",contains:[Y]},qt={scope:"string",begin:'"',end:'"',illegal:"\\n",contains:[Y]},Qt={begin:/\b(a|an|the|are|I'm|isn't|don't|doesn't|won't|but|just|should|pretty|simply|enough|gonna|going|wtf|so|such|will|you|your|they|like|more)\b/},ae=function(e,t,n={}){const s=D({scope:"comment",begin:e,end:t,contains:[]},n);s.contains.push({scope:"doctag",begin:"[ ]*(?=(TODO|FIXME|NOTE|BUG|OPTIMIZE|HACK|XXX):)",end:/(TODO|FIXME|NOTE|BUG|OPTIMIZE|HACK|XXX):/,excludeBegin:!0,relevance:0});const c=Oe("I","a","is","so","us","to","at","if","in","it","on",/[A-Za-z]+['](d|ve|re|ll|t|s|n)/,/[A-Za-z]+[-][a-z]+/,/[A-Za-z][a-z]{2,}/);return s.contains.push({begin:U(/[ ]+/,"(",c,/[.]?[:]?([.][ ]|[ ])/,"){3}")}),s},en=ae("//","$"),tn=ae("/\\*","\\*/"),nn=ae("#","$"),sn={scope:"number",begin:tt,relevance:0},rn={scope:"number",begin:nt,relevance:0},an={scope:"number",begin:st,relevance:0},ln={begin:/(?=\/[^/\n]*\/)/,contains:[{scope:"regexp",begin:/\//,end:/\/[gimuy]*/,illegal:/\n/,contains:[Y,{begin:/\[/,end:/\]/,relevance:0,contains:[Y]}]}]},on={scope:"title",begin:et,relevance:0},cn={scope:"title",begin:Ae,relevance:0},un={begin:"\\.\\s*"+Ae,relevance:0},gn=function(e){return Object.assign(e,{"on:begin":(t,n)=>{n.data._beginMatch=t[1]},"on:end":(t,n)=>{n.data._beginMatch!==t[1]&&n.ignoreMatch()}})};var ne=Object.freeze({__proto__:null,MATCH_NOTHING_RE:Yt,IDENT_RE:et,UNDERSCORE_IDENT_RE:Ae,NUMBER_RE:tt,C_NUMBER_RE:nt,BINARY_NUMBER_RE:st,RE_STARTERS_RE:Jt,SHEBANG:Zt,BACKSLASH_ESCAPE:Y,APOS_STRING_MODE:Vt,QUOTE_STRING_MODE:qt,PHRASAL_WORDS_MODE:Qt,COMMENT:ae,C_LINE_COMMENT_MODE:en,C_BLOCK_COMMENT_MODE:tn,HASH_COMMENT_MODE:nn,NUMBER_MODE:sn,C_NUMBER_MODE:rn,BINARY_NUMBER_MODE:an,REGEXP_MODE:ln,TITLE_MODE:on,UNDERSCORE_TITLE_MODE:cn,METHOD_GUARD:un,END_SAME_AS_BEGIN:gn});function fn(e,t){e.input[e.index-1]==="."&&t.ignoreMatch()}function hn(e,t){e.className!==void 0&&(e.scope=e.className,delete e.className)}function dn(e,t){t&&e.beginKeywords&&(e.begin="\\b("+e.beginKeywords.split(" ").join("|")+")(?!\\.)(?=\\b|\\s)",e.__beforeBegin=fn,e.keywords=e.keywords||e.beginKeywords,delete e.beginKeywords,e.relevance===void 0&&(e.relevance=0))}function pn(e,t){Array.isArray(e.illegal)&&(e.illegal=Oe(...e.illegal))}function bn(e,t){if(e.match){if(e.begin||e.end)throw new Error("begin & end are not supported with match");e.begin=e.match,delete e.match}}function _n(e,t){e.relevance===void 0&&(e.relevance=1)}const En=(e,t)=>{if(!e.beforeMatch)return;if(e.starts)throw new Error("beforeMatch cannot be used with starts");const n=Object.assign({},e);Object.keys(e).forEach(s=>{delete e[s]}),e.keywords=n.keywords,e.begin=U(n.beforeMatch,qe(n.begin)),e.starts={relevance:0,contains:[Object.assign(n,{endsParent:!0})]},e.relevance=0,delete n.beforeMatch},yn=["of","and","for","in","not","or","if","then","parent","list","value"],Rn="keyword";function it(e,t,n=Rn){const s=Object.create(null);return typeof e=="string"?c(n,e.split(" ")):Array.isArray(e)?c(n,e):Object.keys(e).forEach(function(r){Object.assign(s,it(e[r],t,r))}),s;function c(r,i){t&&(i=i.map(a=>a.toLowerCase())),i.forEach(function(a){const l=a.split("|");s[l[0]]=[r,Mn(l[0],l[1])]})}}function Mn(e,t){return t?Number(t):Sn(e)?0:1}function Sn(e){return yn.includes(e.toLowerCase())}const Ue={},H=e=>{console.error(e)},Ke=(e,...t)=>{console.log(`WARN: ${e}`,...t)},K=(e,t)=>{Ue[`${e}/${t}`]||(console.log(`Deprecated as of ${e}. ${t}`),Ue[`${e}/${t}`]=!0)},re=new Error;function rt(e,t,{key:n}){let s=0;const c=e[n],r={},i={};for(let a=1;a<=t.length;a++)i[a+s]=c[a],r[a+s]=!0,s+=Qe(t[a-1]);e[n]=i,e[n]._emit=r,e[n]._multi=!0}function wn(e){if(Array.isArray(e.begin)){if(e.skip||e.excludeBegin||e.returnBegin)throw H("skip, excludeBegin, returnBegin not compatible with beginScope: {}"),re;if(typeof e.beginScope!="object"||e.beginScope===null)throw H("beginScope must be object"),re;rt(e,e.begin,{key:"beginScope"}),e.begin=Ne(e.begin,{joinWith:""})}}function mn(e){if(Array.isArray(e.end)){if(e.skip||e.excludeEnd||e.returnEnd)throw H("skip, excludeEnd, returnEnd not compatible with endScope: {}"),re;if(typeof e.endScope!="object"||e.endScope===null)throw H("endScope must be object"),re;rt(e,e.end,{key:"endScope"}),e.end=Ne(e.end,{joinWith:""})}}function On(e){e.scope&&typeof e.scope=="object"&&e.scope!==null&&(e.beginScope=e.scope,delete e.scope)}function Nn(e){On(e),typeof e.beginScope=="string"&&(e.beginScope={_wrap:e.beginScope}),typeof e.endScope=="string"&&(e.endScope={_wrap:e.endScope}),wn(e),mn(e)}function An(e){function t(i,a){return new RegExp(X(i),"m"+(e.case_insensitive?"i":"")+(e.unicodeRegex?"u":"")+(a?"g":""))}class n{constructor(){this.matchIndexes={},this.regexes=[],this.matchAt=1,this.position=0}addRule(a,l){l.position=this.position++,this.matchIndexes[this.matchAt]=l,this.regexes.push([l,a]),this.matchAt+=Qe(a)+1}compile(){this.regexes.length===0&&(this.exec=()=>null);const a=this.regexes.map(l=>l[1]);this.matcherRe=t(Ne(a,{joinWith:"|"}),!0),this.lastIndex=0}exec(a){this.matcherRe.lastIndex=this.lastIndex;const l=this.matcherRe.exec(a);if(!l)return null;const g=l.findIndex((_,y)=>y>0&&_!==void 0),E=this.matchIndexes[g];return l.splice(0,g),Object.assign(l,E)}}class s{constructor(){this.rules=[],this.multiRegexes=[],this.count=0,this.lastIndex=0,this.regexIndex=0}getMatcher(a){if(this.multiRegexes[a])return this.multiRegexes[a];const l=new n;return this.rules.slice(a).forEach(([g,E])=>l.addRule(g,E)),l.compile(),this.multiRegexes[a]=l,l}resumingScanAtSamePosition(){return this.regexIndex!==0}considerAll(){this.regexIndex=0}addRule(a,l){this.rules.push([a,l]),l.type==="begin"&&this.count++}exec(a){const l=this.getMatcher(this.regexIndex);l.lastIndex=this.lastIndex;let g=l.exec(a);if(this.resumingScanAtSamePosition()&&!(g&&g.index===this.lastIndex)){const E=this.getMatcher(0);E.lastIndex=this.lastIndex+1,g=E.exec(a)}return g&&(this.regexIndex+=g.position+1,this.regexIndex===this.count&&this.considerAll()),g}}function c(i){const a=new s;return i.contains.forEach(l=>a.addRule(l.begin,{rule:l,type:"begin"})),i.terminatorEnd&&a.addRule(i.terminatorEnd,{type:"end"}),i.illegal&&a.addRule(i.illegal,{type:"illegal"}),a}function r(i,a){const l=i;if(i.isCompiled)return l;[hn,bn,Nn,En].forEach(E=>E(i,a)),e.compilerExtensions.forEach(E=>E(i,a)),i.__beforeBegin=null,[dn,pn,_n].forEach(E=>E(i,a)),i.isCompiled=!0;let g=null;return typeof i.keywords=="object"&&i.keywords.$pattern&&(i.keywords=Object.assign({},i.keywords),g=i.keywords.$pattern,delete i.keywords.$pattern),g=g||/\w+/,i.keywords&&(i.keywords=it(i.keywords,e.case_insensitive)),l.keywordPatternRe=t(g,!0),a&&(i.begin||(i.begin=/\B|\b/),l.beginRe=t(l.begin),!i.end&&!i.endsWithParent&&(i.end=/\B|\b/),i.end&&(l.endRe=t(l.end)),l.terminatorEnd=X(l.end)||"",i.endsWithParent&&a.terminatorEnd&&(l.terminatorEnd+=(i.end?"|":"")+a.terminatorEnd)),i.illegal&&(l.illegalRe=t(i.illegal)),i.contains||(i.contains=[]),i.contains=[].concat(...i.contains.map(function(E){return xn(E==="self"?i:E)})),i.contains.forEach(function(E){r(E,l)}),i.starts&&r(i.starts,a),l.matcher=c(l),l}if(e.compilerExtensions||(e.compilerExtensions=[]),e.contains&&e.contains.includes("self"))throw new Error("ERR: contains `self` is not supported at the top-level of a language.  See documentation.");return e.classNameAliases=D(e.classNameAliases||{}),r(e)}function at(e){return e?e.endsWithParent||at(e.starts):!1}function xn(e){return e.variants&&!e.cachedVariants&&(e.cachedVariants=e.variants.map(function(t){return D(e,{variants:null},t)})),e.cachedVariants?e.cachedVariants:at(e)?D(e,{starts:e.starts?D(e.starts):null}):Object.isFrozen(e)?D(e):e}var kn="11.7.0";class Tn extends Error{constructor(t,n){super(t),this.name="HTMLInjectionError",this.html=n}}const ge=Ve,Fe=D,Ge=Symbol("nomatch"),jn=7,vn=function(e){const t=Object.create(null),n=Object.create(null),s=[];let c=!0;const r="Could not find the language '{}', did you forget to load/include a language module?",i={disableAutodetect:!0,name:"Plain text",contains:[]};let a={ignoreUnescapedHTML:!1,throwUnescapedHTML:!1,noHighlightRe:/^(no-?highlight)$/i,languageDetectRe:/\blang(?:uage)?-([\w-]+)\b/i,classPrefix:"hljs-",cssSelector:"pre code",languages:null,__emitter:Ft};function l(o){return a.noHighlightRe.test(o)}function g(o){let h=o.className+" ";h+=o.parentNode?o.parentNode.className:"";const b=a.languageDetectRe.exec(h);if(b){const M=B(b[1]);return M||(Ke(r.replace("{}",b[1])),Ke("Falling back to no-highlight mode for this block.",o)),M?b[1]:"no-highlight"}return h.split(/\s+/).find(M=>l(M)||B(M))}function E(o,h,b){let M="",w="";typeof h=="object"?(M=o,b=h.ignoreIllegals,w=h.language):(K("10.7.0","highlight(lang, code, ...args) has been deprecated."),K("10.7.0",`Please use highlight(code, options) instead.
https://github.com/highlightjs/highlight.js/issues/2277`),w=o,M=h),b===void 0&&(b=!0);const x={code:M,language:w};q("before:highlight",x);const L=x.result?x.result:_(x.language,x.code,b);return L.code=x.code,q("after:highlight",L),L}function _(o,h,b,M){const w=Object.create(null);function x(u,f){return u.keywords[f]}function L(){if(!d.keywords){m.addText(S);return}let u=0;d.keywordPatternRe.lastIndex=0;let f=d.keywordPatternRe.exec(S),p="";for(;f;){p+=S.substring(u,f.index);const R=C.case_insensitive?f[0].toLowerCase():f[0],O=x(d,R);if(O){const[T,Mt]=O;if(m.addText(p),p="",w[R]=(w[R]||0)+1,w[R]<=jn&&(te+=Mt),T.startsWith("_"))p+=f[0];else{const St=C.classNameAliases[T]||T;m.addKeyword(f[0],St)}}else p+=f[0];u=d.keywordPatternRe.lastIndex,f=d.keywordPatternRe.exec(S)}p+=S.substring(u),m.addText(p)}function Q(){if(S==="")return;let u=null;if(typeof d.subLanguage=="string"){if(!t[d.subLanguage]){m.addText(S);return}u=_(d.subLanguage,S,!0,Le[d.subLanguage]),Le[d.subLanguage]=u._top}else u=N(S,d.subLanguage.length?d.subLanguage:null);d.relevance>0&&(te+=u.relevance),m.addSublanguage(u._emitter,u.language)}function A(){d.subLanguage!=null?Q():L(),S=""}function I(u,f){let p=1;const R=f.length-1;for(;p<=R;){if(!u._emit[p]){p++;continue}const O=C.classNameAliases[u[p]]||u[p],T=f[p];O?m.addKeyword(T,O):(S=T,L(),S=""),p++}}function je(u,f){return u.scope&&typeof u.scope=="string"&&m.openNode(C.classNameAliases[u.scope]||u.scope),u.beginScope&&(u.beginScope._wrap?(m.addKeyword(S,C.classNameAliases[u.beginScope._wrap]||u.beginScope._wrap),S=""):u.beginScope._multi&&(I(u.beginScope,f),S="")),d=Object.create(u,{parent:{value:d}}),d}function ve(u,f,p){let R=$t(u.endRe,p);if(R){if(u["on:end"]){const O=new De(u);u["on:end"](f,O),O.isMatchIgnored&&(R=!1)}if(R){for(;u.endsParent&&u.parent;)u=u.parent;return u}}if(u.endsWithParent)return ve(u.parent,f,p)}function bt(u){return d.matcher.regexIndex===0?(S+=u[0],1):(ue=!0,0)}function _t(u){const f=u[0],p=u.rule,R=new De(p),O=[p.__beforeBegin,p["on:begin"]];for(const T of O)if(T&&(T(u,R),R.isMatchIgnored))return bt(f);return p.skip?S+=f:(p.excludeBegin&&(S+=f),A(),!p.returnBegin&&!p.excludeBegin&&(S=f)),je(p,u),p.returnBegin?0:f.length}function Et(u){const f=u[0],p=h.substring(u.index),R=ve(d,u,p);if(!R)return Ge;const O=d;d.endScope&&d.endScope._wrap?(A(),m.addKeyword(f,d.endScope._wrap)):d.endScope&&d.endScope._multi?(A(),I(d.endScope,u)):O.skip?S+=f:(O.returnEnd||O.excludeEnd||(S+=f),A(),O.excludeEnd&&(S=f));do d.scope&&m.closeNode(),!d.skip&&!d.subLanguage&&(te+=d.relevance),d=d.parent;while(d!==R.parent);return R.starts&&je(R.starts,u),O.returnEnd?0:f.length}function yt(){const u=[];for(let f=d;f!==C;f=f.parent)f.scope&&u.unshift(f.scope);u.forEach(f=>m.openNode(f))}let ee={};function Be(u,f){const p=f&&f[0];if(S+=u,p==null)return A(),0;if(ee.type==="begin"&&f.type==="end"&&ee.index===f.index&&p===""){if(S+=h.slice(f.index,f.index+1),!c){const R=new Error(`0 width match regex (${o})`);throw R.languageName=o,R.badRule=ee.rule,R}return 1}if(ee=f,f.type==="begin")return _t(f);if(f.type==="illegal"&&!b){const R=new Error('Illegal lexeme "'+p+'" for mode "'+(d.scope||"<unnamed>")+'"');throw R.mode=d,R}else if(f.type==="end"){const R=Et(f);if(R!==Ge)return R}if(f.type==="illegal"&&p==="")return 1;if(ce>1e5&&ce>f.index*3)throw new Error("potential infinite loop, way more iterations than matches");return S+=p,p.length}const C=B(o);if(!C)throw H(r.replace("{}",o)),new Error('Unknown language: "'+o+'"');const Rt=An(C);let oe="",d=M||Rt;const Le={},m=new a.__emitter(a);yt();let S="",te=0,P=0,ce=0,ue=!1;try{for(d.matcher.considerAll();;){ce++,ue?ue=!1:d.matcher.considerAll(),d.matcher.lastIndex=P;const u=d.matcher.exec(h);if(!u)break;const f=h.substring(P,u.index),p=Be(f,u);P=u.index+p}return Be(h.substring(P)),m.closeAllNodes(),m.finalize(),oe=m.toHTML(),{language:o,value:oe,relevance:te,illegal:!1,_emitter:m,_top:d}}catch(u){if(u.message&&u.message.includes("Illegal"))return{language:o,value:ge(h),illegal:!0,relevance:0,_illegalBy:{message:u.message,index:P,context:h.slice(P-100,P+100),mode:u.mode,resultSoFar:oe},_emitter:m};if(c)return{language:o,value:ge(h),illegal:!1,relevance:0,errorRaised:u,_emitter:m,_top:d};throw u}}function y(o){const h={value:ge(o),illegal:!1,relevance:0,_top:i,_emitter:new a.__emitter(a)};return h._emitter.addText(o),h}function N(o,h){h=h||a.languages||Object.keys(t);const b=y(o),M=h.filter(B).filter(Te).map(A=>_(A,o,!1));M.unshift(b);const w=M.sort((A,I)=>{if(A.relevance!==I.relevance)return I.relevance-A.relevance;if(A.language&&I.language){if(B(A.language).supersetOf===I.language)return 1;if(B(I.language).supersetOf===A.language)return-1}return 0}),[x,L]=w,Q=x;return Q.secondBest=L,Q}function j(o,h,b){const M=h&&n[h]||b;o.classList.add("hljs"),o.classList.add(`language-${M}`)}function v(o){let h=null;const b=g(o);if(l(b))return;if(q("before:highlightElement",{el:o,language:b}),o.children.length>0&&(a.ignoreUnescapedHTML||(console.warn("One of your code blocks includes unescaped HTML. This is a potentially serious security risk."),console.warn("https://github.com/highlightjs/highlight.js/wiki/security"),console.warn("The element with unescaped HTML:"),console.warn(o)),a.throwUnescapedHTML))throw new Tn("One of your code blocks includes unescaped HTML.",o.innerHTML);h=o;const M=h.textContent,w=b?E(M,{language:b,ignoreIllegals:!0}):N(M);o.innerHTML=w.value,j(o,b,w.language),o.result={language:w.language,re:w.relevance,relevance:w.relevance},w.secondBest&&(o.secondBest={language:w.secondBest.language,relevance:w.secondBest.relevance}),q("after:highlightElement",{el:o,result:w,text:M})}function le(o){a=Fe(a,o)}const Z=()=>{V(),K("10.6.0","initHighlighting() deprecated.  Use highlightAll() now.")};function ot(){V(),K("10.6.0","initHighlightingOnLoad() deprecated.  Use highlightAll() now.")}let xe=!1;function V(){if(document.readyState==="loading"){xe=!0;return}document.querySelectorAll(a.cssSelector).forEach(v)}function ct(){xe&&V()}typeof window<"u"&&window.addEventListener&&window.addEventListener("DOMContentLoaded",ct,!1);function ut(o,h){let b=null;try{b=h(e)}catch(M){if(H("Language definition for '{}' could not be registered.".replace("{}",o)),c)H(M);else throw M;b=i}b.name||(b.name=o),t[o]=b,b.rawDefinition=h.bind(null,e),b.aliases&&ke(b.aliases,{languageName:o})}function gt(o){delete t[o];for(const h of Object.keys(n))n[h]===o&&delete n[h]}function ft(){return Object.keys(t)}function B(o){return o=(o||"").toLowerCase(),t[o]||t[n[o]]}function ke(o,{languageName:h}){typeof o=="string"&&(o=[o]),o.forEach(b=>{n[b.toLowerCase()]=h})}function Te(o){const h=B(o);return h&&!h.disableAutodetect}function ht(o){o["before:highlightBlock"]&&!o["before:highlightElement"]&&(o["before:highlightElement"]=h=>{o["before:highlightBlock"](Object.assign({block:h.el},h))}),o["after:highlightBlock"]&&!o["after:highlightElement"]&&(o["after:highlightElement"]=h=>{o["after:highlightBlock"](Object.assign({block:h.el},h))})}function dt(o){ht(o),s.push(o)}function q(o,h){const b=o;s.forEach(function(M){M[b]&&M[b](h)})}function pt(o){return K("10.7.0","highlightBlock will be removed entirely in v12.0"),K("10.7.0","Please use highlightElement now."),v(o)}Object.assign(e,{highlight:E,highlightAuto:N,highlightAll:V,highlightElement:v,highlightBlock:pt,configure:le,initHighlighting:Z,initHighlightingOnLoad:ot,registerLanguage:ut,unregisterLanguage:gt,listLanguages:ft,getLanguage:B,registerAliases:ke,autoDetection:Te,inherit:Fe,addPlugin:dt}),e.debugMode=function(){c=!1},e.safeMode=function(){c=!0},e.versionString=kn,e.regex={concat:U,lookahead:qe,either:Oe,optional:Wt,anyNumberOfTimes:Gt};for(const o in ne)typeof ne[o]=="object"&&Se.exports(ne[o]);return Object.assign(e,ne),e};var J=vn({}),Bn=J;J.HighlightJS=J;J.default=J;const We=Bn;function Ln(e){let t;return{c(){t=wt(e[2])},l(n){t=mt(n,e[2])},m(n,s){$(n,t,s)},p(n,s){s&4&&Ot(t,n[2])},d(n){n&&k(t)}}}function In(e){let t,n;return{c(){t=new Je(!1),n=ie(),this.h()},l(s){t=Ze(s,!1),n=ie(),this.h()},h(){t.a=n},m(s,c){t.m(e[1],s,c),$(s,n,c)},p(s,c){c&2&&t.p(s[1])},d(s){s&&k(n),s&&t.d()}}}function Cn(e){let t,n;function s(l,g){return l[1]?In:Ln}let c=s(e),r=c(e),i=[{"data-language":e[3]},e[4]],a={};for(let l=0;l<i.length;l+=1)a=F(a,i[l]);return{c(){t=fe("pre"),n=fe("code"),r.c(),this.h()},l(l){t=he(l,"PRE",{"data-language":!0});var g=de(t);n=he(g,"CODE",{});var E=de(n);r.l(E),E.forEach(k),g.forEach(k),this.h()},h(){z(n,"hljs",!0),Ie(t,a),z(t,"langtag",e[0]),z(t,"svelte-1h28s4b",!0)},m(l,g){$(l,t,g),$e(t,n),r.m(n,null)},p(l,[g]){c===(c=s(l))&&r?r.p(l,g):(r.d(1),r=c(l),r&&(r.c(),r.m(n,null))),Ie(t,a=Xe(i,[g&8&&{"data-language":l[3]},g&16&&l[4]])),z(t,"langtag",l[0]),z(t,"svelte-1h28s4b",!0)},i:Ce,o:Ce,d(l){l&&k(t),r.d()}}}function Dn(e,t,n){const s=["langtag","highlighted","code","languageName"];let c=se(t,s),{langtag:r=!1}=t,{highlighted:i}=t,{code:a}=t,{languageName:l="plaintext"}=t;return e.$$set=g=>{t=F(F({},t),Ye(g)),n(4,c=se(t,s)),"langtag"in g&&n(0,r=g.langtag),"highlighted"in g&&n(1,i=g.highlighted),"code"in g&&n(2,a=g.code),"languageName"in g&&n(3,l=g.languageName)},[r,i,a,l,c]}class Pn extends pe{constructor(t){super(),be(this,t,Dn,Cn,_e,{langtag:0,highlighted:1,code:2,languageName:3})}}const Hn=Pn,Un=e=>({highlighted:e&8}),ze=e=>({highlighted:e[3]});function Kn(e){let t,n;const s=[e[4],{languageName:e[0].name},{langtag:e[2]},{highlighted:e[3]},{code:e[1]}];let c={};for(let r=0;r<s.length;r+=1)c=F(c,s[r]);return t=new Hn({props:c}),{c(){Ee(t.$$.fragment)},l(r){ye(t.$$.fragment,r)},m(r,i){Re(t,r,i),n=!0},p(r,i){const a=i&31?Xe(s,[i&16&&vt(r[4]),i&1&&{languageName:r[0].name},i&4&&{langtag:r[2]},i&8&&{highlighted:r[3]},i&2&&{code:r[1]}]):{};t.$set(a)},i(r){n||(G(t.$$.fragment,r),n=!0)},o(r){W(t.$$.fragment,r),n=!1},d(r){Me(t,r)}}}function Fn(e){let t;const n=e[6].default,s=Nt(n,e,e[5],ze),c=s||Kn(e);return{c(){c&&c.c()},l(r){c&&c.l(r)},m(r,i){c&&c.m(r,i),t=!0},p(r,[i]){s?s.p&&(!t||i&40)&&At(s,n,r,r[5],t?kt(n,r[5],i,Un):xt(r[5]),ze):c&&c.p&&(!t||i&31)&&c.p(r,t?i:-1)},i(r){t||(G(c,r),t=!0)},o(r){W(c,r),t=!1},d(r){c&&c.d(r)}}}function Gn(e,t,n){const s=["language","code","langtag"];let c=se(t,s),{$$slots:r={},$$scope:i}=t,{language:a}=t,{code:l}=t,{langtag:g=!1}=t;const E=Tt();let _="";return jt(()=>{_&&E("highlight",{highlighted:_})}),e.$$set=y=>{t=F(F({},t),Ye(y)),n(4,c=se(t,s)),"language"in y&&n(0,a=y.language),"code"in y&&n(1,l=y.code),"langtag"in y&&n(2,g=y.langtag),"$$scope"in y&&n(5,i=y.$$scope)},e.$$.update=()=>{e.$$.dirty&3&&(We.registerLanguage(a.name,a.register),n(3,_=We.highlight(l,{language:a.name}).value))},[a,l,g,_,c,i,r]}class Wn extends pe{constructor(t){super(),be(this,t,Gn,Fn,_e,{language:0,code:1,langtag:2})}}const lt=Wn;function zn(e){const t=e.regex,n=/[\p{XID_Start}_]\p{XID_Continue}*/u,s=["and","as","assert","async","await","break","case","class","continue","def","del","elif","else","except","finally","for","from","global","if","import","in","is","lambda","match","nonlocal|10","not","or","pass","raise","return","try","while","with","yield"],a={$pattern:/[A-Za-z]\w+|__\w+__/,keyword:s,built_in:["__import__","abs","all","any","ascii","bin","bool","breakpoint","bytearray","bytes","callable","chr","classmethod","compile","complex","delattr","dict","dir","divmod","enumerate","eval","exec","filter","float","format","frozenset","getattr","globals","hasattr","hash","help","hex","id","input","int","isinstance","issubclass","iter","len","list","locals","map","max","memoryview","min","next","object","oct","open","ord","pow","print","property","range","repr","reversed","round","set","setattr","slice","sorted","staticmethod","str","sum","super","tuple","type","vars","zip"],literal:["__debug__","Ellipsis","False","None","NotImplemented","True"],type:["Any","Callable","Coroutine","Dict","List","Literal","Generic","Optional","Sequence","Set","Tuple","Type","Union"]},l={className:"meta",begin:/^(>>>|\.\.\.) /},g={className:"subst",begin:/\{/,end:/\}/,keywords:a,illegal:/#/},E={begin:/\{\{/,relevance:0},_={className:"string",contains:[e.BACKSLASH_ESCAPE],variants:[{begin:/([uU]|[bB]|[rR]|[bB][rR]|[rR][bB])?'''/,end:/'''/,contains:[e.BACKSLASH_ESCAPE,l],relevance:10},{begin:/([uU]|[bB]|[rR]|[bB][rR]|[rR][bB])?"""/,end:/"""/,contains:[e.BACKSLASH_ESCAPE,l],relevance:10},{begin:/([fF][rR]|[rR][fF]|[fF])'''/,end:/'''/,contains:[e.BACKSLASH_ESCAPE,l,E,g]},{begin:/([fF][rR]|[rR][fF]|[fF])"""/,end:/"""/,contains:[e.BACKSLASH_ESCAPE,l,E,g]},{begin:/([uU]|[rR])'/,end:/'/,relevance:10},{begin:/([uU]|[rR])"/,end:/"/,relevance:10},{begin:/([bB]|[bB][rR]|[rR][bB])'/,end:/'/},{begin:/([bB]|[bB][rR]|[rR][bB])"/,end:/"/},{begin:/([fF][rR]|[rR][fF]|[fF])'/,end:/'/,contains:[e.BACKSLASH_ESCAPE,E,g]},{begin:/([fF][rR]|[rR][fF]|[fF])"/,end:/"/,contains:[e.BACKSLASH_ESCAPE,E,g]},e.APOS_STRING_MODE,e.QUOTE_STRING_MODE]},y="[0-9](_?[0-9])*",N=`(\\b(${y}))?\\.(${y})|\\b(${y})\\.`,j=`\\b|${s.join("|")}`,v={className:"number",relevance:0,variants:[{begin:`(\\b(${y})|(${N}))[eE][+-]?(${y})[jJ]?(?=${j})`},{begin:`(${N})[jJ]?`},{begin:`\\b([1-9](_?[0-9])*|0+(_?0)*)[lLjJ]?(?=${j})`},{begin:`\\b0[bB](_?[01])+[lL]?(?=${j})`},{begin:`\\b0[oO](_?[0-7])+[lL]?(?=${j})`},{begin:`\\b0[xX](_?[0-9a-fA-F])+[lL]?(?=${j})`},{begin:`\\b(${y})[jJ](?=${j})`}]},le={className:"comment",begin:t.lookahead(/# type:/),end:/$/,keywords:a,contains:[{begin:/# type:/},{begin:/#/,end:/\b\B/,endsWithParent:!0}]},Z={className:"params",variants:[{className:"",begin:/\(\s*\)/,skip:!0},{begin:/\(/,end:/\)/,excludeBegin:!0,excludeEnd:!0,keywords:a,contains:["self",l,v,_,e.HASH_COMMENT_MODE]}]};return g.contains=[_,v,l],{name:"Python",aliases:["py","gyp","ipython"],unicodeRegex:!0,keywords:a,illegal:/(<\/|->|\?)|=>/,contains:[l,v,{begin:/\bself\b/},{beginKeywords:"if",relevance:0},_,le,e.HASH_COMMENT_MODE,{match:[/\bdef/,/\s+/,n],scope:{1:"keyword",3:"title.function"},contains:[Z]},{variants:[{match:[/\bclass/,/\s+/,n,/\s*/,/\(\s*/,n,/\s*\)/]},{match:[/\bclass/,/\s+/,n]}],scope:{1:"keyword",3:"title.class",6:"title.class.inherited"}},{className:"meta",begin:/^[\t ]*@/,end:/(?=#)|$/,contains:[v,Z,_]}]}}const $n={name:"python",register:zn},Xn=$n;function Yn(e){return{aliases:["pycon"],contains:[{className:"meta.prompt",starts:{end:/ |$/,starts:{end:"$",subLanguage:"python"}},variants:[{begin:/^>>>(?=[ ]|$)/},{begin:/^\.\.\.(?=[ ]|$)/}]}]}}const Jn={name:"python-repl",register:Yn},Zn=Jn,Vn=`<style>/*!
  Theme: Github
  Author: Defman21
  License: ~ MIT (or more permissive) [via base16-schemes-source]
  Maintainer: @highlightjs/core-team
  Version: 2021.09.0
*/pre code.hljs{display:block;overflow-x:auto;padding:1em}code.hljs{padding:3px 5px}.hljs{color:#333;background:#fff}.hljs ::selection,.hljs::selection{background-color:#c8c8fa;color:#333}.hljs-comment{color:#969896}.hljs-tag{color:#e8e8e8}.hljs-operator,.hljs-punctuation,.hljs-subst{color:#333}.hljs-operator{opacity:.7}.hljs-bullet,.hljs-deletion,.hljs-name,.hljs-selector-tag,.hljs-template-variable,.hljs-variable{color:#ed6a43}.hljs-attr,.hljs-link,.hljs-literal,.hljs-number,.hljs-symbol,.hljs-variable.constant_{color:#0086b3}.hljs-class .hljs-title,.hljs-title,.hljs-title.class_{color:#795da3}.hljs-strong{font-weight:700;color:#795da3}.hljs-addition,.hljs-built_in,.hljs-code,.hljs-doctag,.hljs-keyword.hljs-atrule,.hljs-quote,.hljs-regexp,.hljs-string,.hljs-title.class_.inherited__{color:#183691}.hljs-attribute,.hljs-function .hljs-title,.hljs-section,.hljs-title.function_,.ruby .hljs-property{color:#795da3}.diff .hljs-meta,.hljs-keyword,.hljs-template-tag,.hljs-type{color:#a71d5d}.hljs-emphasis{color:#a71d5d;font-style:italic}.hljs-meta,.hljs-meta .hljs-keyword,.hljs-meta .hljs-string{color:#333}.hljs-meta .hljs-keyword,.hljs-meta-keyword{font-weight:700}</style>`,qn=Vn;function Qn(e){let t,n;return t=new lt({props:{language:Zn,code:e[0]}}),{c(){Ee(t.$$.fragment)},l(s){ye(t.$$.fragment,s)},m(s,c){Re(t,s,c),n=!0},p(s,c){const r={};c&1&&(r.code=s[0]),t.$set(r)},i(s){n||(G(t.$$.fragment,s),n=!0)},o(s){W(t.$$.fragment,s),n=!1},d(s){Me(t,s)}}}function es(e){let t,n;return t=new lt({props:{language:Xn,code:e[0]}}),{c(){Ee(t.$$.fragment)},l(s){ye(t.$$.fragment,s)},m(s,c){Re(t,s,c),n=!0},p(s,c){const r={};c&1&&(r.code=s[0]),t.$set(r)},i(s){n||(G(t.$$.fragment,s),n=!0)},o(s){W(t.$$.fragment,s),n=!1},d(s){Me(t,s)}}}function ts(e){let t,n,s,c,r,i,a;const l=[es,Qn],g=[];function E(_,y){return _[1]?1:0}return r=E(e),i=g[r]=l[r](e),{c(){t=new Je(!1),n=ie(),s=Bt(),c=fe("div"),i.c(),this.h()},l(_){const y=Lt("svelte-32u38c",document.head);t=Ze(y,!1),n=ie(),y.forEach(k),s=It(_),c=he(_,"DIV",{class:!0});var N=de(c);i.l(N),N.forEach(k),this.h()},h(){t.a=n,Ct(c,"class","border my-2 text-sm")},m(_,y){t.m(qn,document.head),$e(document.head,n),$(_,s,y),$(_,c,y),g[r].m(c,null),a=!0},p(_,[y]){let N=r;r=E(_),r===N?g[r].p(_,y):(Dt(),W(g[N],1,1,()=>{g[N]=null}),Pt(),i=g[r],i?i.p(_,y):(i=g[r]=l[r](_),i.c()),G(i,1),i.m(c,null))},i(_){a||(G(i),a=!0)},o(_){W(i),a=!1},d(_){k(n),_&&t.d(),_&&k(s),_&&k(c),g[r].d()}}}function ns(e,t,n){let{code:s=""}=t,{isOutput:c=!1}=t;return e.$$set=r=>{"code"in r&&n(0,s=r.code),"isOutput"in r&&n(1,c=r.isOutput)},[s,c]}class is extends pe{constructor(t){super(),be(this,t,ns,ts,_e,{code:0,isOutput:1})}}export{is as P};
