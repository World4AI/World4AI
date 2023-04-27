import{S as za,i as Na,s as Wa,k as b,a as w,q as s,y as p,W as Ma,l as x,h as l,c as y,m as S,r,z as m,n as J,N as $,b as f,A as h,g as c,d as g,B as d,V as ma,C as B}from"../chunks/index.4d92b023.js";import{C as Ja}from"../chunks/Container.b0705c7b.js";import{L as v}from"../chunks/Latex.e0b308c0.js";import{G as Ra,g as ja,a as Oa}from"../chunks/maps.0f079072.js";import{w as Aa}from"../chunks/index.e1ba4c1e.js";class Ca{constructor(n,t,a,u,q){this.observationSpace=n,this.actionSpace=t,this.policy={},this.valueFunction={},this.model=a,this.theta=u,this.gamma=q,this.maxDelta=0,n.forEach((E,T)=>{this.policy[E.r]||(this.policy[E.r]={}),this.policy[E.r][E.c]=this.randomChoice(t)}),n.forEach((E,T)=>{this.valueFunction[E.r]||(this.valueFunction[E.r]={}),this.valueFunction[E.r][E.c]=0}),this.policyStore=Aa(this.policy),this.valueStore=Aa(this.valueFunction)}policyEvaluation(){do this.policyEvaluationStep();while(this.maxDelta>this.theta)}policyEvaluationStep(){let n=JSON.parse(JSON.stringify(this.valueFunction)),t=JSON.parse(JSON.stringify(this.valueFunction));this.maxDelta=0,this.observationSpace.forEach(a=>{let u=this.policy[a.r][a.c],q=0;this.model[a.r][a.c][u].forEach(T=>{let O=T.probability,D=T.reward,_=T.observation,k;T.done===!0?k=0:k=1,q+=O*(D+this.gamma*n[_.r][_.c]*k)}),t[a.r][a.c]=q;let E=Math.abs(q-n[a.r][a.c]);E>this.maxDelta&&(this.maxDelta=E)}),this.valueFunction=t,this.valueStore.set(this.valueFunction)}randomChoice(n){return n[Math.floor(n.length*Math.random())]}policyImprovement(){let n=JSON.parse(JSON.stringify(this.policy));return this.observationSpace.forEach(t=>{let a=-1e6,u=0;this.actionSpace.forEach(q=>{let E=0;this.model[t.r][t.c][q].forEach(T=>{let O=T.probability,D=T.reward,_=T.observation,k;T.done===!0?k=0:k=1,E+=O*(D+this.gamma*this.valueFunction[_.r][_.c]*k)}),E>a&&(a=E,u=q)}),n[t.r][t.c]=u}),n}policyIteration(){for(;;){this.policyEvaluation();let n=this.policyImprovement();if(JSON.stringify(n)==JSON.stringify(this.policy))break;this.policy=n,this.policyStore.set(this.policy)}}}function La(o){let n=String.raw`\pi`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function Ga(o){let n=String.raw`v_{\pi}(s)`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function Ha(o){let n=String.raw`\pi_*`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function Qa(o){let n=String.raw`v_*`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function Ya(o){let n=String.raw`v_{\pi}`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function Ka(o){let n=String.raw`\pi`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function Ua(o){let n=String.raw`
    \begin{aligned}
   v_{\pi}(s)  & = \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(s') \mid S_t=s] \\
& = \sum_a \pi(a \mid s)  R(a, s) + \gamma \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)v_{\pi}(s')
\end{aligned}
`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function Xa(o){let n;return{c(){n=s("r")},l(t){n=r(t,"r")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function Za(o){let n;return{c(){n=s("s'")},l(t){n=r(t,"s'")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function ei(o){let n;return{c(){n=s("s")},l(t){n=r(t,"s")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function ti(o){let n;return{c(){n=s("a")},l(t){n=r(t,"a")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function ni(o){let n;return{c(){n=s("p(s', r \\mid s, a)")},l(t){n=r(t,"p(s', r \\mid s, a)")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function ai(o){let n=String.raw`
    \begin{aligned}
    v_{\pi}(s) & = \mathbb{E}_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1}) \mid S_t = s] \\
    & = \sum_a \pi(a \mid s)  R(a, s) + \gamma \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)v_{\pi}(s') \\
    & = \sum_{a}\pi(a \mid s)\sum_{s', r}p(s', r \mid s, a)[r + \gamma v_{\pi}(s')]
    \end{aligned}
  `+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function ii(o){let n=String.raw`\underbrace{v_{\pi}(s)}_{\text{left side}} = \underbrace{\sum_{a}\pi(a \mid s)\sum_{s', r}p(s', r \mid s, a)[r + \gamma v_{\pi}(s')]}_{\text{right side}}`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function li(o){let n;return{c(){n=s("s")},l(t){n=r(t,"s")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function oi(o){let n=String.raw`s \in \mathcal{S}`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function si(o){let n=String.raw`V_{k+1}(s) = \sum_a \pi(a \mid s) \sum_{s', r}p(s', r \mid s, a)[r + \gamma V_{k}(s')]`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function ri(o){let n;return{c(){n=s("V(s)")},l(t){n=r(t,"V(s)")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function fi(o){let n;return{c(){n=s("v(s)")},l(t){n=r(t,"v(s)")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function $i(o){let n;return{c(){n=s("v(s)")},l(t){n=r(t,"v(s)")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function ui(o){let n;return{c(){n=s("\\pi")},l(t){n=r(t,"\\pi")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function pi(o){let n;return{c(){n=s("V(s)")},l(t){n=r(t,"V(s)")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function mi(o){let n;return{c(){n=s("k+1")},l(t){n=r(t,"k+1")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function hi(o){let n=String.raw`V(s)`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function ci(o){let n=String.raw`\mu(s)`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function gi(o){let n;return{c(){n=s("s")},l(t){n=r(t,"s")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function di(o){let n;return{c(){n=s("a")},l(t){n=r(t,"a")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function vi(o){let n=String.raw`a \neq \mu(s)`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function _i(o){let n=String.raw`\mu(s)`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function wi(o){let n;return{c(){n=s("T")},l(t){n=r(t,"T")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function yi(o){let n;return{c(){n=s("a")},l(t){n=r(t,"a")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function bi(o){let n;return{c(){n=s("s")},l(t){n=r(t,"s")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function xi(o){let n=String.raw`\mu(s)`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function Si(o){let n=String.raw`q_{\mu}(s, a) \doteq \mathbb{E}[R_{t+1} + \gamma v_{\mu}(S_{t+1}) \mid S_t = s, A_t = a]`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function Ei(o){let n=String.raw`V_{\mu}(s)`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function Ti(o){let n=String.raw`Q_{\mu}(s, a)`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function ki(o){let n;return{c(){n=s("a")},l(t){n=r(t,"a")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function qi(o){let n;return{c(){n=s("\\mu")},l(t){n=r(t,"\\mu")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function Pi(o){let n=String.raw`\mu`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function Ii(o){let n=String.raw`Q_{\mu}(s, a) > V_{\mu}(s)`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function Bi(o){let n;return{c(){n=s("a")},l(t){n=r(t,"a")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function Fi(o){let n;return{c(){n=s("s")},l(t){n=r(t,"s")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function Vi(o){let n;return{c(){n=s("a")},l(t){n=r(t,"a")},m(t,a){f(t,n,a)},d(t){t&&l(n)}}}function Di(o){let n=String.raw`\mu'`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function Oi(o){let n=String.raw`s \in \mathcal{S}`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function Ai(o){let n=String.raw`\mu'(s) = \arg\max_a Q_{\mu}(s, a)`+"",t;return{c(){t=s(n)},l(a){t=r(a,n)},m(a,u){f(a,t,u)},p:B,d(a){a&&l(t)}}}function zi(o){let n,t,a,u,q,E,T,O,D,_,k,W,Le,K,C,Et,R,on,U,sn,X,rn,Tt,Z,kt,N,fn,ee,$n,te,un,ne,pn,ae,mn,ie,hn,qt,le,Pt,et,cn,It,oe,Bt,se,gn,re,dn,Ft,fe,vn,$e,_n,Vt,tt,wn,Dt,ue,Ot,A,yn,pe,bn,me,xn,he,Sn,ce,En,ge,Tn,de,kn,At,L,qn,ut,Pn,In,zt,nt,Bn,Nt,at,Fn,Wt,G,ve,Vn,pt,Mt,it,Jt,lt,Dn,Rt,_e,On,we,An,jt,P,zn,ye,Nn,be,Wn,xe,Mn,Se,Jn,Ee,Rn,Te,jn,ke,Cn,qe,Ln,Pe,Gn,Ct,Ie,Lt,I,Hn,Be,Qn,Fe,Yn,Ve,Kn,De,Un,Oe,Xn,Ae,Zn,ze,ea,Ne,ta,We,na,Gt,j,aa,Me,ia,Je,la,Ht,Re,Qt,ot,oa,Yt,st,Kt,rt,sa,Ut,ft,ra,Xt,je,fa,mt,$a,ua,Zt,H,Ce,pa,ht,en,$t,tn;return a=new v({props:{$$slots:{default:[La]},$$scope:{ctx:o}}}),q=new v({props:{$$slots:{default:[Ga]},$$scope:{ctx:o}}}),T=new v({props:{$$slots:{default:[Ha]},$$scope:{ctx:o}}}),D=new v({props:{$$slots:{default:[Qa]},$$scope:{ctx:o}}}),U=new v({props:{$$slots:{default:[Ya]},$$scope:{ctx:o}}}),X=new v({props:{$$slots:{default:[Ka]},$$scope:{ctx:o}}}),Z=new v({props:{$$slots:{default:[Ua]},$$scope:{ctx:o}}}),ee=new v({props:{$$slots:{default:[Xa]},$$scope:{ctx:o}}}),te=new v({props:{$$slots:{default:[Za]},$$scope:{ctx:o}}}),ne=new v({props:{$$slots:{default:[ei]},$$scope:{ctx:o}}}),ae=new v({props:{$$slots:{default:[ti]},$$scope:{ctx:o}}}),ie=new v({props:{$$slots:{default:[ni]},$$scope:{ctx:o}}}),le=new v({props:{$$slots:{default:[ai]},$$scope:{ctx:o}}}),oe=new v({props:{$$slots:{default:[ii]},$$scope:{ctx:o}}}),re=new v({props:{$$slots:{default:[li]},$$scope:{ctx:o}}}),$e=new v({props:{$$slots:{default:[oi]},$$scope:{ctx:o}}}),ue=new v({props:{$$slots:{default:[si]},$$scope:{ctx:o}}}),pe=new v({props:{$$slots:{default:[ri]},$$scope:{ctx:o}}}),me=new v({props:{$$slots:{default:[fi]},$$scope:{ctx:o}}}),he=new v({props:{$$slots:{default:[$i]},$$scope:{ctx:o}}}),ce=new v({props:{$$slots:{default:[ui]},$$scope:{ctx:o}}}),ge=new v({props:{$$slots:{default:[pi]},$$scope:{ctx:o}}}),de=new v({props:{$$slots:{default:[mi]},$$scope:{ctx:o}}}),ve=new Oa({props:{cells:o[3],valueFunction:o[0]?null:o[2],policy:o[0]?o[1]:null}}),we=new v({props:{$$slots:{default:[hi]},$$scope:{ctx:o}}}),ye=new v({props:{$$slots:{default:[ci]},$$scope:{ctx:o}}}),be=new v({props:{$$slots:{default:[gi]},$$scope:{ctx:o}}}),xe=new v({props:{$$slots:{default:[di]},$$scope:{ctx:o}}}),Se=new v({props:{$$slots:{default:[vi]},$$scope:{ctx:o}}}),Ee=new v({props:{$$slots:{default:[_i]},$$scope:{ctx:o}}}),Te=new v({props:{$$slots:{default:[wi]},$$scope:{ctx:o}}}),ke=new v({props:{$$slots:{default:[yi]},$$scope:{ctx:o}}}),qe=new v({props:{$$slots:{default:[bi]},$$scope:{ctx:o}}}),Pe=new v({props:{$$slots:{default:[xi]},$$scope:{ctx:o}}}),Ie=new v({props:{$$slots:{default:[Si]},$$scope:{ctx:o}}}),Be=new v({props:{$$slots:{default:[Ei]},$$scope:{ctx:o}}}),Fe=new v({props:{$$slots:{default:[Ti]},$$scope:{ctx:o}}}),Ve=new v({props:{$$slots:{default:[ki]},$$scope:{ctx:o}}}),De=new v({props:{$$slots:{default:[qi]},$$scope:{ctx:o}}}),Oe=new v({props:{$$slots:{default:[Pi]},$$scope:{ctx:o}}}),Ae=new v({props:{$$slots:{default:[Ii]},$$scope:{ctx:o}}}),ze=new v({props:{$$slots:{default:[Bi]},$$scope:{ctx:o}}}),Ne=new v({props:{$$slots:{default:[Fi]},$$scope:{ctx:o}}}),We=new v({props:{$$slots:{default:[Vi]},$$scope:{ctx:o}}}),Me=new v({props:{$$slots:{default:[Di]},$$scope:{ctx:o}}}),Je=new v({props:{$$slots:{default:[Oi]},$$scope:{ctx:o}}}),Re=new v({props:{$$slots:{default:[Ai]},$$scope:{ctx:o}}}),Ce=new Oa({props:{cells:o[3],valueFunction:o[0]?null:o[2],policy:o[0]?o[1]:null}}),{c(){n=b("p"),t=s(`Dynamic programming algorithms that are designed to solve a Markov decision
    process are iterative algorithms, which consist of two basic steps: policy
    evaluation and policy improvement. The purpose of policy evaluation is to
    measure the performance of a given policy `),p(a.$$.fragment),u=s(`
    by estimating the corresponding value function `),p(q.$$.fragment),E=s(`
    . Policy improvement on the other hand generates a new policy, that is better
    (or at least not worse) than the previous policy. The output of policy evaluation
    is used as an input into policy improvement and vice versa. The iterative process
    of evaluation and improvement produces value and policy functions that converge
    towards the optimal policy function `),p(T.$$.fragment),O=s(` and optimal
    value function `),p(D.$$.fragment),_=s(` over time. The policy iteration
    algorithm that is covered in this section is one such iterative algorithm.`),k=w(),W=b("div"),Le=w(),K=b("h2"),C=s("Policy Evaluation"),Et=w(),R=b("p"),on=s("The goal of policy evaluation is to find the true value function "),p(U.$$.fragment),sn=s(`
    of the policy `),p(X.$$.fragment),rn=s("."),Tt=w(),p(Z.$$.fragment),kt=w(),N=b("p"),fn=s(`Often it is more convenient to use the joint probability of simultaneously
    getting the reward `),p(ee.$$.fragment),$n=s(" and the next state "),p(te.$$.fragment),un=s(` given
    current state `),p(ne.$$.fragment),pn=s(" and action "),p(ae.$$.fragment),mn=s(`. This joint
    probability function is depicted as `),p(ie.$$.fragment),hn=s(`. This
    notation is more compact and is likely to make the transition from theory to
    practice easier.`),qt=w(),p(le.$$.fragment),Pt=w(),et=b("p"),cn=s(`If we look closely at the Bellman equation, we can observe that the equation
    basically consists of two sides. The left side and the right side.`),It=w(),p(oe.$$.fragment),Bt=w(),se=b("p"),gn=s("The left side is the function that returns the value of a state "),p(re.$$.fragment),dn=s(`, it is the mapping from states to values. The left side is essentially the
    function we are trying to find. The right side is the definition of the
    value function, that is based on the expectation of the returns and is
    expressed using the Bellman equation.`),Ft=w(),fe=b("p"),vn=s(`When we initialize the policy evaluation algorithm, the first step is to
    generate a value function that is used as a benchmark that needs to be
    constantly improved in the interative process. The initial values are set
    either randomly or to zero. When we start to use the above equation we will
    not surprisingly discover that the random/zero value (the left side of the
    above equation) and the expected value of the reward plus value for the next
    state (the right side of the above equation) will diverge quite a lot. The
    goal of the policy evaluation algorithm is to make the left side of the
    equation and the right side of the equation to be exactly equal. That is
    done in an iterative process where at each step the difference between both
    sides is reduced. In practice we do not expect the difference between the
    two to go all the way down to zero. Instead we define a threshold value. For
    example a threshold value of 0.0001 indicates that we can interrupt the
    iterative process as soon as for all the states `),p($e.$$.fragment),_n=s(` the difference between the left and the right side of the equation is below
    the threshold value.`),Vt=w(),tt=b("p"),wn=s(`The policy estimation algorithm is relatively straightforward. All we need
    to do is to turn the definition of the of the Bellman equation into the
    update rule.`),Dt=w(),p(ue.$$.fragment),Ot=w(),A=b("p"),yn=s("Above we use "),p(pe.$$.fragment),bn=s(` instead of
    `),p(me.$$.fragment),xn=s(". This notational difference is to show that "),p(he.$$.fragment),Sn=s(" is the true value function of a policy "),p(ce.$$.fragment),En=s(", while "),p(ge.$$.fragment),Tn=s(" is it's estimate. At each iteration step "),p(de.$$.fragment),kn=s(` the left side of
    the equation (the esimate of the value function) is replaced by the right hand
    of the equation. At this point it should become apparent why the Bellman equation
    is useful. Only the reward from the next time step is required to improve the
    approximation, because all subsequent rewards are already condensed into the
    value function from the next time step. That allows the algorithm to use the
    model to look only one step into the future for the reward and use the approximated
    value function for the next time step. By repeating the update rule over and
    over again the rewards are getting embedded into the value function and the approximation
    gets better and better.`),At=w(),L=b("p"),qn=s(`The process of using past estimates to improve current estimates is called
    `),ut=b("strong"),Pn=s("bootstrapping"),In=s(`. Bootstrapping is used heavily through
    reinforcement learning and can generally be used without the full knowledge
    of the model of the environment.`),zt=w(),nt=b("p"),Bn=s(`Below you can find the Python implementation of the policy evaluation
    algorithm.`),Nt=w(),at=b("p"),Fn=s(`Once again below we deal with a simple grid world where the task is to
    arrive at the bottom left corner starting from the top left corner. The
    environment transitions with 50% probability into the desired direction
    (unless there is some barrier) and with 50% chance the environment takes a
    randomm action. The playground below allows you to calculate the value
    function for a randomly initialized deterministic policy, using the policy
    evaluation algorithm. You can switch between the display of the policy and
    the value function. Start by taking one step of the algorithm at the time
    and observe how the value function propagates and the difference between
    steps keeps decreasing. Finally you can run the full policy evaluation
    algorithm, where the iterative process keeps going until the difference
    betwenn the left and the right side of the Bellman equation is less than
    0.00001.`),Wt=w(),G=b("div"),p(ve.$$.fragment),Vn=w(),pt=b("div"),Mt=w(),it=b("div"),Jt=w(),lt=b("h2"),Dn=s("Policy Improvement"),Rt=w(),_e=b("p"),On=s(`The goal of policy improvement is to create a new and improved policy using
    the value function `),p(we.$$.fragment),An=s(` from the previous policy
    evaluation step.`),jt=w(),P=b("p"),zn=s("Let us assume for simplicity that the agent follows a deterministic policy "),p(ye.$$.fragment),Nn=s(", but in the current state "),p(be.$$.fragment),Wn=s(` the agent contemplates to pick the
    action `),p(xe.$$.fragment),Mn=s(` that contradicts the policy, therefore
    `),p(Se.$$.fragment),Jn=s(`. After that action the agent will
    stick to the old policy `),p(Ee.$$.fragment),Rn=s(` and follow it until
    the terminal state `),p(Te.$$.fragment),jn=s(`. We can measure the value of using the
    action `),p(ke.$$.fragment),Cn=s(" at state "),p(qe.$$.fragment),Ln=s(` and then following the policy
    `),p(Pe.$$.fragment),Gn=s(" using the action-value function."),Ct=w(),p(Ie.$$.fragment),Lt=w(),I=b("p"),Hn=s("What if the agent compares the estimates "),p(Be.$$.fragment),Qn=s(" and "),p(Fe.$$.fragment),Yn=s(`
    and determines that taking some action `),p(Ve.$$.fragment),Kn=s(` and then following
    `),p(De.$$.fragment),Un=s(" is of higher value than strictly following "),p(Oe.$$.fragment),Xn=s(`, showing that
    `),p(Ae.$$.fragment),Zn=s(`? Does that imply
    that the agent should change the policy and always take the action `),p(ze.$$.fragment),ea=s(`
    when facing the state `),p(Ne.$$.fragment),ta=s(`? Does the short term gain from the
    new action `),p(We.$$.fragment),na=s(` justifies changing the policy? It turns out that
    this is exactly the case.`),Gt=w(),j=b("p"),aa=s("In the policy improvement step the we create a new policy "),p(Me.$$.fragment),ia=s(" where the agent chooses the greedy action at each state "),p(Je.$$.fragment),la=s("."),Ht=w(),p(Re.$$.fragment),Qt=w(),ot=b("p"),oa=s("Below you can find a Python example of the policy improvement step."),Yt=w(),st=b("div"),Kt=w(),rt=b("h2"),sa=s("The Policy Iteration Algorithm"),Ut=w(),ft=b("p"),ra=s(`The idea of policy iteration is to alternate between policy evaluation and
    policy improvement until the optimal policy has been reached. Once the new
    policy and the old policy are exactly the same we have reached the optimal
    policy.`),Xt=w(),je=b("p"),fa=s(`Below is a playground from the same gridworld, that demonstrates the policy
    iteration algorithm. The algorithm finds the optimal policy and
    corresponding optimal value function, once you click on the `),mt=b("em"),$a=s('"policy iteration"'),ua=s(" button."),Zt=w(),H=b("div"),p(Ce.$$.fragment),pa=w(),ht=b("div"),en=w(),$t=b("div"),this.h()},l(e){n=x(e,"P",{});var i=S(n);t=r(i,`Dynamic programming algorithms that are designed to solve a Markov decision
    process are iterative algorithms, which consist of two basic steps: policy
    evaluation and policy improvement. The purpose of policy evaluation is to
    measure the performance of a given policy `),m(a.$$.fragment,i),u=r(i,`
    by estimating the corresponding value function `),m(q.$$.fragment,i),E=r(i,`
    . Policy improvement on the other hand generates a new policy, that is better
    (or at least not worse) than the previous policy. The output of policy evaluation
    is used as an input into policy improvement and vice versa. The iterative process
    of evaluation and improvement produces value and policy functions that converge
    towards the optimal policy function `),m(T.$$.fragment,i),O=r(i,` and optimal
    value function `),m(D.$$.fragment,i),_=r(i,` over time. The policy iteration
    algorithm that is covered in this section is one such iterative algorithm.`),i.forEach(l),k=y(e),W=x(e,"DIV",{class:!0}),S(W).forEach(l),Le=y(e),K=x(e,"H2",{});var ct=S(K);C=r(ct,"Policy Evaluation"),ct.forEach(l),Et=y(e),R=x(e,"P",{});var Q=S(R);on=r(Q,"The goal of policy evaluation is to find the true value function "),m(U.$$.fragment,Q),sn=r(Q,`
    of the policy `),m(X.$$.fragment,Q),rn=r(Q,"."),Q.forEach(l),Tt=y(e),m(Z.$$.fragment,e),kt=y(e),N=x(e,"P",{});var M=S(N);fn=r(M,`Often it is more convenient to use the joint probability of simultaneously
    getting the reward `),m(ee.$$.fragment,M),$n=r(M," and the next state "),m(te.$$.fragment,M),un=r(M,` given
    current state `),m(ne.$$.fragment,M),pn=r(M," and action "),m(ae.$$.fragment,M),mn=r(M,`. This joint
    probability function is depicted as `),m(ie.$$.fragment,M),hn=r(M,`. This
    notation is more compact and is likely to make the transition from theory to
    practice easier.`),M.forEach(l),qt=y(e),m(le.$$.fragment,e),Pt=y(e),et=x(e,"P",{});var gt=S(et);cn=r(gt,`If we look closely at the Bellman equation, we can observe that the equation
    basically consists of two sides. The left side and the right side.`),gt.forEach(l),It=y(e),m(oe.$$.fragment,e),Bt=y(e),se=x(e,"P",{});var Ge=S(se);gn=r(Ge,"The left side is the function that returns the value of a state "),m(re.$$.fragment,Ge),dn=r(Ge,`, it is the mapping from states to values. The left side is essentially the
    function we are trying to find. The right side is the definition of the
    value function, that is based on the expectation of the returns and is
    expressed using the Bellman equation.`),Ge.forEach(l),Ft=y(e),fe=x(e,"P",{});var He=S(fe);vn=r(He,`When we initialize the policy evaluation algorithm, the first step is to
    generate a value function that is used as a benchmark that needs to be
    constantly improved in the interative process. The initial values are set
    either randomly or to zero. When we start to use the above equation we will
    not surprisingly discover that the random/zero value (the left side of the
    above equation) and the expected value of the reward plus value for the next
    state (the right side of the above equation) will diverge quite a lot. The
    goal of the policy evaluation algorithm is to make the left side of the
    equation and the right side of the equation to be exactly equal. That is
    done in an iterative process where at each step the difference between both
    sides is reduced. In practice we do not expect the difference between the
    two to go all the way down to zero. Instead we define a threshold value. For
    example a threshold value of 0.0001 indicates that we can interrupt the
    iterative process as soon as for all the states `),m($e.$$.fragment,He),_n=r(He,` the difference between the left and the right side of the equation is below
    the threshold value.`),He.forEach(l),Vt=y(e),tt=x(e,"P",{});var dt=S(tt);wn=r(dt,`The policy estimation algorithm is relatively straightforward. All we need
    to do is to turn the definition of the of the Bellman equation into the
    update rule.`),dt.forEach(l),Dt=y(e),m(ue.$$.fragment,e),Ot=y(e),A=x(e,"P",{});var z=S(A);yn=r(z,"Above we use "),m(pe.$$.fragment,z),bn=r(z,` instead of
    `),m(me.$$.fragment,z),xn=r(z,". This notational difference is to show that "),m(he.$$.fragment,z),Sn=r(z," is the true value function of a policy "),m(ce.$$.fragment,z),En=r(z,", while "),m(ge.$$.fragment,z),Tn=r(z," is it's estimate. At each iteration step "),m(de.$$.fragment,z),kn=r(z,` the left side of
    the equation (the esimate of the value function) is replaced by the right hand
    of the equation. At this point it should become apparent why the Bellman equation
    is useful. Only the reward from the next time step is required to improve the
    approximation, because all subsequent rewards are already condensed into the
    value function from the next time step. That allows the algorithm to use the
    model to look only one step into the future for the reward and use the approximated
    value function for the next time step. By repeating the update rule over and
    over again the rewards are getting embedded into the value function and the approximation
    gets better and better.`),z.forEach(l),At=y(e),L=x(e,"P",{class:!0});var Qe=S(L);qn=r(Qe,`The process of using past estimates to improve current estimates is called
    `),ut=x(Qe,"STRONG",{});var vt=S(ut);Pn=r(vt,"bootstrapping"),vt.forEach(l),In=r(Qe,`. Bootstrapping is used heavily through
    reinforcement learning and can generally be used without the full knowledge
    of the model of the environment.`),Qe.forEach(l),zt=y(e),nt=x(e,"P",{});var _t=S(nt);Bn=r(_t,`Below you can find the Python implementation of the policy evaluation
    algorithm.`),_t.forEach(l),Nt=y(e),at=x(e,"P",{});var wt=S(at);Fn=r(wt,`Once again below we deal with a simple grid world where the task is to
    arrive at the bottom left corner starting from the top left corner. The
    environment transitions with 50% probability into the desired direction
    (unless there is some barrier) and with 50% chance the environment takes a
    randomm action. The playground below allows you to calculate the value
    function for a randomly initialized deterministic policy, using the policy
    evaluation algorithm. You can switch between the display of the policy and
    the value function. Start by taking one step of the algorithm at the time
    and observe how the value function propagates and the difference between
    steps keeps decreasing. Finally you can run the full policy evaluation
    algorithm, where the iterative process keeps going until the difference
    betwenn the left and the right side of the Bellman equation is less than
    0.00001.`),wt.forEach(l),Wt=y(e),G=x(e,"DIV",{class:!0});var Ye=S(G);m(ve.$$.fragment,Ye),Vn=y(Ye),pt=x(Ye,"DIV",{class:!0});var nn=S(pt);nn.forEach(l),Ye.forEach(l),Mt=y(e),it=x(e,"DIV",{class:!0}),S(it).forEach(l),Jt=y(e),lt=x(e,"H2",{});var yt=S(lt);Dn=r(yt,"Policy Improvement"),yt.forEach(l),Rt=y(e),_e=x(e,"P",{});var Ke=S(_e);On=r(Ke,`The goal of policy improvement is to create a new and improved policy using
    the value function `),m(we.$$.fragment,Ke),An=r(Ke,` from the previous policy
    evaluation step.`),Ke.forEach(l),jt=y(e),P=x(e,"P",{});var F=S(P);zn=r(F,"Let us assume for simplicity that the agent follows a deterministic policy "),m(ye.$$.fragment,F),Nn=r(F,", but in the current state "),m(be.$$.fragment,F),Wn=r(F,` the agent contemplates to pick the
    action `),m(xe.$$.fragment,F),Mn=r(F,` that contradicts the policy, therefore
    `),m(Se.$$.fragment,F),Jn=r(F,`. After that action the agent will
    stick to the old policy `),m(Ee.$$.fragment,F),Rn=r(F,` and follow it until
    the terminal state `),m(Te.$$.fragment,F),jn=r(F,`. We can measure the value of using the
    action `),m(ke.$$.fragment,F),Cn=r(F," at state "),m(qe.$$.fragment,F),Ln=r(F,` and then following the policy
    `),m(Pe.$$.fragment,F),Gn=r(F," using the action-value function."),F.forEach(l),Ct=y(e),m(Ie.$$.fragment,e),Lt=y(e),I=x(e,"P",{});var V=S(I);Hn=r(V,"What if the agent compares the estimates "),m(Be.$$.fragment,V),Qn=r(V," and "),m(Fe.$$.fragment,V),Yn=r(V,`
    and determines that taking some action `),m(Ve.$$.fragment,V),Kn=r(V,` and then following
    `),m(De.$$.fragment,V),Un=r(V," is of higher value than strictly following "),m(Oe.$$.fragment,V),Xn=r(V,`, showing that
    `),m(Ae.$$.fragment,V),Zn=r(V,`? Does that imply
    that the agent should change the policy and always take the action `),m(ze.$$.fragment,V),ea=r(V,`
    when facing the state `),m(Ne.$$.fragment,V),ta=r(V,`? Does the short term gain from the
    new action `),m(We.$$.fragment,V),na=r(V,` justifies changing the policy? It turns out that
    this is exactly the case.`),V.forEach(l),Gt=y(e),j=x(e,"P",{});var Y=S(j);aa=r(Y,"In the policy improvement step the we create a new policy "),m(Me.$$.fragment,Y),ia=r(Y," where the agent chooses the greedy action at each state "),m(Je.$$.fragment,Y),la=r(Y,"."),Y.forEach(l),Ht=y(e),m(Re.$$.fragment,e),Qt=y(e),ot=x(e,"P",{});var bt=S(ot);oa=r(bt,"Below you can find a Python example of the policy improvement step."),bt.forEach(l),Yt=y(e),st=x(e,"DIV",{class:!0}),S(st).forEach(l),Kt=y(e),rt=x(e,"H2",{});var xt=S(rt);sa=r(xt,"The Policy Iteration Algorithm"),xt.forEach(l),Ut=y(e),ft=x(e,"P",{});var St=S(ft);ra=r(St,`The idea of policy iteration is to alternate between policy evaluation and
    policy improvement until the optimal policy has been reached. Once the new
    policy and the old policy are exactly the same we have reached the optimal
    policy.`),St.forEach(l),Xt=y(e),je=x(e,"P",{});var Ue=S(je);fa=r(Ue,`Below is a playground from the same gridworld, that demonstrates the policy
    iteration algorithm. The algorithm finds the optimal policy and
    corresponding optimal value function, once you click on the `),mt=x(Ue,"EM",{});var Xe=S(mt);$a=r(Xe,'"policy iteration"'),Xe.forEach(l),ua=r(Ue," button."),Ue.forEach(l),Zt=y(e),H=x(e,"DIV",{class:!0});var Ze=S(H);m(Ce.$$.fragment,Ze),pa=y(Ze),ht=x(Ze,"DIV",{class:!0});var an=S(ht);an.forEach(l),Ze.forEach(l),en=y(e),$t=x(e,"DIV",{class:!0}),S($t).forEach(l),this.h()},h(){J(W,"class","separator"),J(L,"class","info"),J(pt,"class","flex-vertical"),J(G,"class","flex-space"),J(it,"class","separator"),J(st,"class","separator"),J(ht,"class","flex-vertical"),J(H,"class","flex-space"),J($t,"class","separator")},m(e,i){f(e,n,i),$(n,t),h(a,n,null),$(n,u),h(q,n,null),$(n,E),h(T,n,null),$(n,O),h(D,n,null),$(n,_),f(e,k,i),f(e,W,i),f(e,Le,i),f(e,K,i),$(K,C),f(e,Et,i),f(e,R,i),$(R,on),h(U,R,null),$(R,sn),h(X,R,null),$(R,rn),f(e,Tt,i),h(Z,e,i),f(e,kt,i),f(e,N,i),$(N,fn),h(ee,N,null),$(N,$n),h(te,N,null),$(N,un),h(ne,N,null),$(N,pn),h(ae,N,null),$(N,mn),h(ie,N,null),$(N,hn),f(e,qt,i),h(le,e,i),f(e,Pt,i),f(e,et,i),$(et,cn),f(e,It,i),h(oe,e,i),f(e,Bt,i),f(e,se,i),$(se,gn),h(re,se,null),$(se,dn),f(e,Ft,i),f(e,fe,i),$(fe,vn),h($e,fe,null),$(fe,_n),f(e,Vt,i),f(e,tt,i),$(tt,wn),f(e,Dt,i),h(ue,e,i),f(e,Ot,i),f(e,A,i),$(A,yn),h(pe,A,null),$(A,bn),h(me,A,null),$(A,xn),h(he,A,null),$(A,Sn),h(ce,A,null),$(A,En),h(ge,A,null),$(A,Tn),h(de,A,null),$(A,kn),f(e,At,i),f(e,L,i),$(L,qn),$(L,ut),$(ut,Pn),$(L,In),f(e,zt,i),f(e,nt,i),$(nt,Bn),f(e,Nt,i),f(e,at,i),$(at,Fn),f(e,Wt,i),f(e,G,i),h(ve,G,null),$(G,Vn),$(G,pt),f(e,Mt,i),f(e,it,i),f(e,Jt,i),f(e,lt,i),$(lt,Dn),f(e,Rt,i),f(e,_e,i),$(_e,On),h(we,_e,null),$(_e,An),f(e,jt,i),f(e,P,i),$(P,zn),h(ye,P,null),$(P,Nn),h(be,P,null),$(P,Wn),h(xe,P,null),$(P,Mn),h(Se,P,null),$(P,Jn),h(Ee,P,null),$(P,Rn),h(Te,P,null),$(P,jn),h(ke,P,null),$(P,Cn),h(qe,P,null),$(P,Ln),h(Pe,P,null),$(P,Gn),f(e,Ct,i),h(Ie,e,i),f(e,Lt,i),f(e,I,i),$(I,Hn),h(Be,I,null),$(I,Qn),h(Fe,I,null),$(I,Yn),h(Ve,I,null),$(I,Kn),h(De,I,null),$(I,Un),h(Oe,I,null),$(I,Xn),h(Ae,I,null),$(I,Zn),h(ze,I,null),$(I,ea),h(Ne,I,null),$(I,ta),h(We,I,null),$(I,na),f(e,Gt,i),f(e,j,i),$(j,aa),h(Me,j,null),$(j,ia),h(Je,j,null),$(j,la),f(e,Ht,i),h(Re,e,i),f(e,Qt,i),f(e,ot,i),$(ot,oa),f(e,Yt,i),f(e,st,i),f(e,Kt,i),f(e,rt,i),$(rt,sa),f(e,Ut,i),f(e,ft,i),$(ft,ra),f(e,Xt,i),f(e,je,i),$(je,fa),$(je,mt),$(mt,$a),$(je,ua),f(e,Zt,i),f(e,H,i),h(Ce,H,null),$(H,pa),$(H,ht),f(e,en,i),f(e,$t,i),tn=!0},p(e,i){const ct={};i&65536&&(ct.$$scope={dirty:i,ctx:e}),a.$set(ct);const Q={};i&65536&&(Q.$$scope={dirty:i,ctx:e}),q.$set(Q);const M={};i&65536&&(M.$$scope={dirty:i,ctx:e}),T.$set(M);const gt={};i&65536&&(gt.$$scope={dirty:i,ctx:e}),D.$set(gt);const Ge={};i&65536&&(Ge.$$scope={dirty:i,ctx:e}),U.$set(Ge);const He={};i&65536&&(He.$$scope={dirty:i,ctx:e}),X.$set(He);const dt={};i&65536&&(dt.$$scope={dirty:i,ctx:e}),Z.$set(dt);const z={};i&65536&&(z.$$scope={dirty:i,ctx:e}),ee.$set(z);const Qe={};i&65536&&(Qe.$$scope={dirty:i,ctx:e}),te.$set(Qe);const vt={};i&65536&&(vt.$$scope={dirty:i,ctx:e}),ne.$set(vt);const _t={};i&65536&&(_t.$$scope={dirty:i,ctx:e}),ae.$set(_t);const wt={};i&65536&&(wt.$$scope={dirty:i,ctx:e}),ie.$set(wt);const Ye={};i&65536&&(Ye.$$scope={dirty:i,ctx:e}),le.$set(Ye);const nn={};i&65536&&(nn.$$scope={dirty:i,ctx:e}),oe.$set(nn);const yt={};i&65536&&(yt.$$scope={dirty:i,ctx:e}),re.$set(yt);const Ke={};i&65536&&(Ke.$$scope={dirty:i,ctx:e}),$e.$set(Ke);const F={};i&65536&&(F.$$scope={dirty:i,ctx:e}),ue.$set(F);const V={};i&65536&&(V.$$scope={dirty:i,ctx:e}),pe.$set(V);const Y={};i&65536&&(Y.$$scope={dirty:i,ctx:e}),me.$set(Y);const bt={};i&65536&&(bt.$$scope={dirty:i,ctx:e}),he.$set(bt);const xt={};i&65536&&(xt.$$scope={dirty:i,ctx:e}),ce.$set(xt);const St={};i&65536&&(St.$$scope={dirty:i,ctx:e}),ge.$set(St);const Ue={};i&65536&&(Ue.$$scope={dirty:i,ctx:e}),de.$set(Ue);const Xe={};i&8&&(Xe.cells=e[3]),i&5&&(Xe.valueFunction=e[0]?null:e[2]),i&3&&(Xe.policy=e[0]?e[1]:null),ve.$set(Xe);const Ze={};i&65536&&(Ze.$$scope={dirty:i,ctx:e}),we.$set(Ze);const an={};i&65536&&(an.$$scope={dirty:i,ctx:e}),ye.$set(an);const ha={};i&65536&&(ha.$$scope={dirty:i,ctx:e}),be.$set(ha);const ca={};i&65536&&(ca.$$scope={dirty:i,ctx:e}),xe.$set(ca);const ga={};i&65536&&(ga.$$scope={dirty:i,ctx:e}),Se.$set(ga);const da={};i&65536&&(da.$$scope={dirty:i,ctx:e}),Ee.$set(da);const va={};i&65536&&(va.$$scope={dirty:i,ctx:e}),Te.$set(va);const _a={};i&65536&&(_a.$$scope={dirty:i,ctx:e}),ke.$set(_a);const wa={};i&65536&&(wa.$$scope={dirty:i,ctx:e}),qe.$set(wa);const ya={};i&65536&&(ya.$$scope={dirty:i,ctx:e}),Pe.$set(ya);const ba={};i&65536&&(ba.$$scope={dirty:i,ctx:e}),Ie.$set(ba);const xa={};i&65536&&(xa.$$scope={dirty:i,ctx:e}),Be.$set(xa);const Sa={};i&65536&&(Sa.$$scope={dirty:i,ctx:e}),Fe.$set(Sa);const Ea={};i&65536&&(Ea.$$scope={dirty:i,ctx:e}),Ve.$set(Ea);const Ta={};i&65536&&(Ta.$$scope={dirty:i,ctx:e}),De.$set(Ta);const ka={};i&65536&&(ka.$$scope={dirty:i,ctx:e}),Oe.$set(ka);const qa={};i&65536&&(qa.$$scope={dirty:i,ctx:e}),Ae.$set(qa);const Pa={};i&65536&&(Pa.$$scope={dirty:i,ctx:e}),ze.$set(Pa);const Ia={};i&65536&&(Ia.$$scope={dirty:i,ctx:e}),Ne.$set(Ia);const Ba={};i&65536&&(Ba.$$scope={dirty:i,ctx:e}),We.$set(Ba);const Fa={};i&65536&&(Fa.$$scope={dirty:i,ctx:e}),Me.$set(Fa);const Va={};i&65536&&(Va.$$scope={dirty:i,ctx:e}),Je.$set(Va);const Da={};i&65536&&(Da.$$scope={dirty:i,ctx:e}),Re.$set(Da);const ln={};i&8&&(ln.cells=e[3]),i&5&&(ln.valueFunction=e[0]?null:e[2]),i&3&&(ln.policy=e[0]?e[1]:null),Ce.$set(ln)},i(e){tn||(c(a.$$.fragment,e),c(q.$$.fragment,e),c(T.$$.fragment,e),c(D.$$.fragment,e),c(U.$$.fragment,e),c(X.$$.fragment,e),c(Z.$$.fragment,e),c(ee.$$.fragment,e),c(te.$$.fragment,e),c(ne.$$.fragment,e),c(ae.$$.fragment,e),c(ie.$$.fragment,e),c(le.$$.fragment,e),c(oe.$$.fragment,e),c(re.$$.fragment,e),c($e.$$.fragment,e),c(ue.$$.fragment,e),c(pe.$$.fragment,e),c(me.$$.fragment,e),c(he.$$.fragment,e),c(ce.$$.fragment,e),c(ge.$$.fragment,e),c(de.$$.fragment,e),c(ve.$$.fragment,e),c(we.$$.fragment,e),c(ye.$$.fragment,e),c(be.$$.fragment,e),c(xe.$$.fragment,e),c(Se.$$.fragment,e),c(Ee.$$.fragment,e),c(Te.$$.fragment,e),c(ke.$$.fragment,e),c(qe.$$.fragment,e),c(Pe.$$.fragment,e),c(Ie.$$.fragment,e),c(Be.$$.fragment,e),c(Fe.$$.fragment,e),c(Ve.$$.fragment,e),c(De.$$.fragment,e),c(Oe.$$.fragment,e),c(Ae.$$.fragment,e),c(ze.$$.fragment,e),c(Ne.$$.fragment,e),c(We.$$.fragment,e),c(Me.$$.fragment,e),c(Je.$$.fragment,e),c(Re.$$.fragment,e),c(Ce.$$.fragment,e),tn=!0)},o(e){g(a.$$.fragment,e),g(q.$$.fragment,e),g(T.$$.fragment,e),g(D.$$.fragment,e),g(U.$$.fragment,e),g(X.$$.fragment,e),g(Z.$$.fragment,e),g(ee.$$.fragment,e),g(te.$$.fragment,e),g(ne.$$.fragment,e),g(ae.$$.fragment,e),g(ie.$$.fragment,e),g(le.$$.fragment,e),g(oe.$$.fragment,e),g(re.$$.fragment,e),g($e.$$.fragment,e),g(ue.$$.fragment,e),g(pe.$$.fragment,e),g(me.$$.fragment,e),g(he.$$.fragment,e),g(ce.$$.fragment,e),g(ge.$$.fragment,e),g(de.$$.fragment,e),g(ve.$$.fragment,e),g(we.$$.fragment,e),g(ye.$$.fragment,e),g(be.$$.fragment,e),g(xe.$$.fragment,e),g(Se.$$.fragment,e),g(Ee.$$.fragment,e),g(Te.$$.fragment,e),g(ke.$$.fragment,e),g(qe.$$.fragment,e),g(Pe.$$.fragment,e),g(Ie.$$.fragment,e),g(Be.$$.fragment,e),g(Fe.$$.fragment,e),g(Ve.$$.fragment,e),g(De.$$.fragment,e),g(Oe.$$.fragment,e),g(Ae.$$.fragment,e),g(ze.$$.fragment,e),g(Ne.$$.fragment,e),g(We.$$.fragment,e),g(Me.$$.fragment,e),g(Je.$$.fragment,e),g(Re.$$.fragment,e),g(Ce.$$.fragment,e),tn=!1},d(e){e&&l(n),d(a),d(q),d(T),d(D),e&&l(k),e&&l(W),e&&l(Le),e&&l(K),e&&l(Et),e&&l(R),d(U),d(X),e&&l(Tt),d(Z,e),e&&l(kt),e&&l(N),d(ee),d(te),d(ne),d(ae),d(ie),e&&l(qt),d(le,e),e&&l(Pt),e&&l(et),e&&l(It),d(oe,e),e&&l(Bt),e&&l(se),d(re),e&&l(Ft),e&&l(fe),d($e),e&&l(Vt),e&&l(tt),e&&l(Dt),d(ue,e),e&&l(Ot),e&&l(A),d(pe),d(me),d(he),d(ce),d(ge),d(de),e&&l(At),e&&l(L),e&&l(zt),e&&l(nt),e&&l(Nt),e&&l(at),e&&l(Wt),e&&l(G),d(ve),e&&l(Mt),e&&l(it),e&&l(Jt),e&&l(lt),e&&l(Rt),e&&l(_e),d(we),e&&l(jt),e&&l(P),d(ye),d(be),d(xe),d(Se),d(Ee),d(Te),d(ke),d(qe),d(Pe),e&&l(Ct),d(Ie,e),e&&l(Lt),e&&l(I),d(Be),d(Fe),d(Ve),d(De),d(Oe),d(Ae),d(ze),d(Ne),d(We),e&&l(Gt),e&&l(j),d(Me),d(Je),e&&l(Ht),d(Re,e),e&&l(Qt),e&&l(ot),e&&l(Yt),e&&l(st),e&&l(Kt),e&&l(rt),e&&l(Ut),e&&l(ft),e&&l(Xt),e&&l(je),e&&l(Zt),e&&l(H),d(Ce),e&&l(en),e&&l($t)}}}function Ni(o){let n,t,a,u,q,E,T,O,D;return O=new Ja({props:{$$slots:{default:[zi]},$$scope:{ctx:o}}}),{c(){n=b("meta"),t=w(),a=b("h1"),u=s("Policy Iteration"),q=w(),E=b("div"),T=w(),p(O.$$.fragment),this.h()},l(_){const k=Ma("svelte-1rhre4",document.head);n=x(k,"META",{name:!0,content:!0}),k.forEach(l),t=y(_),a=x(_,"H1",{});var W=S(a);u=r(W,"Policy Iteration"),W.forEach(l),q=y(_),E=x(_,"DIV",{class:!0}),S(E).forEach(l),T=y(_),m(O.$$.fragment,_),this.h()},h(){document.title="World4AI | Reinforcement Learning | Policy Iteration Algorithm",J(n,"name","description"),J(n,"content","Policy iteration is an iterative (dynamic programming) algorithm. The algorithm alternates between policy evaluation and policy improvement to arrive at the optimal policy and value functions"),J(E,"class","separator")},m(_,k){$(document.head,n),f(_,t,k),f(_,a,k),$(a,u),f(_,q,k),f(_,E,k),f(_,T,k),h(O,_,k),D=!0},p(_,[k]){const W={};k&65551&&(W.$$scope={dirty:k,ctx:_}),O.$set(W)},i(_){D||(c(O.$$.fragment,_),D=!0)},o(_){g(O.$$.fragment,_),D=!1},d(_){l(n),_&&l(t),_&&l(a),_&&l(q),_&&l(E),_&&l(T),d(O,_)}}}function Wi(o,n,t){let a,u,q,E,T,O,D=new Ra(ja,!0),_=new Ca(D.observationSpace,D.actionSpace,D.getModel(),1e-5,.99);const k=D.cellsStore;ma(o,k,C=>t(9,O=C));const W=_.valueStore;ma(o,W,C=>t(8,T=C));const Le=_.policyStore;ma(o,Le,C=>t(7,E=C));let K=!0;return o.$$.update=()=>{o.$$.dirty&512&&t(3,a=O),o.$$.dirty&256&&t(2,u=T),o.$$.dirty&128&&t(1,q=E)},[K,q,u,a,k,W,Le,E,T,O]}class Li extends za{constructor(n){super(),Na(this,n,Wi,Ni,Wa,{})}}export{Li as default};
