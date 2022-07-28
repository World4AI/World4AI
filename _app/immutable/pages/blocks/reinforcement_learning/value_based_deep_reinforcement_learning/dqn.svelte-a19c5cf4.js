import{S as da,i as va,s as ga,L as Qt,a as m,l as g,w as T,M as zt,n as d,h as t,c,m as b,x as E,p as v,b as o,G as p,y as D,f as k,t as Q,B as z,Q as ft,a0 as ba,a1 as _a,E as P,r as l,T as ka,u as f}from"../../../../chunks/index-caa95cd4.js";import{C as xa}from"../../../../chunks/Container-5c6b7f6d.js";import{N as pa}from"../../../../chunks/NeuralNetworkOld-c5b87430.js";import{L as q}from"../../../../chunks/Latex-bf74aeea.js";import{C as Ta}from"../../../../chunks/CartPole-8bd7166b.js";import{M as Ea}from"../../../../chunks/MemoryBuffer-d03c34ea.js";import{B as Da}from"../../../../chunks/Button-13b097c1.js";import{d as ya}from"../../../../chunks/index-13c6122f.js";import{t as ut}from"../../../../chunks/index-658140a5.js";import"../../../../chunks/Table-a51fb5ea.js";import"../../../../chunks/index-313c094f.js";import"../../../../chunks/index-1ac4b6a2.js";function ma(s){let i,n,a,u;return{c(){i=Qt("line"),this.h()},l(w){i=zt(w,"line",{x1:!0,y1:!0,x2:!0,y2:!0,stroke:!0}),d(i).forEach(t),this.h()},h(){v(i,"x1",s[10]),v(i,"y1",s[9]),v(i,"x2",n=s[10]+s[2]),v(i,"y2",a=s[9]+s[3]),v(i,"stroke","var(--text-color)")},m(w,_){o(w,i,_)},p(w,_){_&1024&&v(i,"x1",w[10]),_&512&&v(i,"y1",w[9]),_&1028&&n!==(n=w[10]+w[2])&&v(i,"x2",n),_&520&&a!==(a=w[9]+w[3])&&v(i,"y2",a)},i(w){u||ba(()=>{u=_a(i,ya,{duration:1e3}),u.start()})},o:P,d(w){w&&t(i)}}}function ca(s){let i,n,a,u;return{c(){i=Qt("line"),this.h()},l(w){i=zt(w,"line",{x1:!0,y1:!0,x2:!0,y2:!0,stroke:!0}),d(i).forEach(t),this.h()},h(){v(i,"x1",s[8]),v(i,"y1",s[7]),v(i,"x2",n=s[8]+s[5]),v(i,"y2",a=s[7]+s[6]),v(i,"stroke","var(--text-color)")},m(w,_){o(w,i,_)},p(w,_){_&256&&v(i,"x1",w[8]),_&128&&v(i,"y1",w[7]),_&288&&n!==(n=w[8]+w[5])&&v(i,"x2",n),_&192&&a!==(a=w[7]+w[6])&&v(i,"y2",a)},i(w){u||ba(()=>{u=_a(i,ya,{duration:1e3}),u.start()})},o:P,d(w){w&&t(i)}}}function Qa(s){let i,n,a,u,w,_,I,A,x=s[1]&&ma(s),h=s[4]&&ca(s);return I=new Da({props:{disabled:s[0],value:"Step"}}),I.$on("click",s[19]),{c(){i=Qt("svg"),x&&x.c(),n=Qt("circle"),h&&h.c(),a=Qt("circle"),w=m(),_=g("div"),T(I.$$.fragment),this.h()},l($){i=zt($,"svg",{viewBox:!0});var y=d(i);x&&x.l(y),n=zt(y,"circle",{fill:!0,stroke:!0,cx:!0,cy:!0,r:!0}),d(n).forEach(t),h&&h.l(y),a=zt(y,"circle",{fill:!0,stroke:!0,cx:!0,cy:!0,r:!0}),d(a).forEach(t),y.forEach(t),w=c($),_=b($,"DIV",{class:!0});var M=d(_);E(I.$$.fragment,M),M.forEach(t),this.h()},h(){v(n,"fill","var(--main-color-1)"),v(n,"stroke","black"),v(n,"cx",s[10]),v(n,"cy",s[9]),v(n,"r","10"),v(a,"fill","var(--main-color-2)"),v(a,"stroke","black"),v(a,"cx",s[8]),v(a,"cy",s[7]),v(a,"r","10"),v(i,"viewBox",u=s[12]+" "+s[11]+" "+At+" "+It),v(_,"class","flex-center")},m($,y){o($,i,y),x&&x.m(i,null),p(i,n),h&&h.m(i,null),p(i,a),o($,w,y),o($,_,y),D(I,_,null),A=!0},p($,[y]){$[1]?x?(x.p($,y),y&2&&k(x,1)):(x=ma($),x.c(),k(x,1),x.m(i,n)):x&&(x.d(1),x=null),(!A||y&1024)&&v(n,"cx",$[10]),(!A||y&512)&&v(n,"cy",$[9]),$[4]?h?(h.p($,y),y&16&&k(h,1)):(h=ca($),h.c(),k(h,1),h.m(i,a)):h&&(h.d(1),h=null),(!A||y&256)&&v(a,"cx",$[8]),(!A||y&128)&&v(a,"cy",$[7]),(!A||y&6144&&u!==(u=$[12]+" "+$[11]+" "+At+" "+It))&&v(i,"viewBox",u);const M={};y&1&&(M.disabled=$[0]),I.$set(M)},i($){A||(k(x),k(h),k(I.$$.fragment,$),A=!0)},o($){Q(I.$$.fragment,$),A=!1},d($){$&&t(i),x&&x.d(),h&&h.d(),$&&t(w),$&&t(_),z(I)}}}let za=3,At=500,It=200,$a=100;function Aa(s,i,n){let a,u,w,_,I,A,x,h,$,y,M,Z,{frozen:N=!1}=i,be=0,W=!1,_e=ut(At/2,{duration:1400,delay:1e3});ft(s,_e,S=>n(26,Z=S));let C=ut(It/2,{duration:1400,delay:1e3});ft(s,C,S=>n(25,M=S));let ye=ut(At/2-20,{duration:1400,delay:1e3});ft(s,ye,S=>n(24,y=S));let H=ut(It/2-40,{duration:1400,delay:1e3});ft(s,H,S=>n(23,$=S));let ke=!1,O=0,xe=0,R=-100,V;N?V=.5:V=.9;let ee=!1,F=0,Te=0;const U=ut(0,{duration:500});ft(s,U,S=>n(22,h=S));const Ee=ut(0,{duration:500});ft(s,Ee,S=>n(21,x=S));function Ne(){be+=1,n(0,W=!0),n(5,F=(w-a)*V),n(6,Te=(_-u)*V),n(4,ee=!0),_e.set(a+F),C.set(u+Te),setTimeout(function(){n(4,ee=!1)},1e3),(!N||be>za)&&(be=0,n(2,O=Math.random()*($a-R)+R),n(3,xe=Math.random()*($a-R)+R),n(1,ke=!0),ye.set(w+O),H.set(_+xe),setTimeout(function(){n(1,ke=!1)},1e3)),setTimeout(function(){U.set(a-At/2),Ee.set(u-It/2),n(0,W=!1)},3500)}return s.$$set=S=>{"frozen"in S&&n(20,N=S.frozen)},s.$$.update=()=>{s.$$.dirty&67108864&&n(8,a=Z),s.$$.dirty&33554432&&n(7,u=M),s.$$.dirty&16777216&&n(10,w=y),s.$$.dirty&8388608&&n(9,_=$),s.$$.dirty&4194304&&n(12,I=h),s.$$.dirty&2097152&&n(11,A=x)},[W,ke,O,xe,ee,F,Te,u,a,_,w,A,I,_e,C,ye,H,U,Ee,Ne,N,x,h,$,y,M,Z]}class wa extends da{constructor(i){super(),va(this,i,Aa,Qa,ga,{frozen:20})}}function Ia(s){let i=String.raw`\hat{Q}(s, a, \mathbf{w})`+"",n;return{c(){n=l(i)},l(a){n=f(a,i)},m(a,u){o(a,n,u)},p:P,d(a){a&&t(n)}}}function Sa(s){let i;return{c(){i=l("s")},l(n){i=f(n,"s")},m(n,a){o(n,i,a)},d(n){n&&t(i)}}}function qa(s){let i;return{c(){i=l("a")},l(n){i=f(n,"a")},m(n,a){o(n,i,a)},d(n){n&&t(i)}}}function Pa(s){let i=String.raw`\mathbf{w}`+"",n;return{c(){n=l(i)},l(a){n=f(a,i)},m(a,u){o(a,n,u)},p:P,d(a){a&&t(n)}}}function Na(s){let i;return{c(){i=l("s")},l(n){i=f(n,"s")},m(n,a){o(n,i,a)},d(n){n&&t(i)}}}function Ma(s){let i=String.raw`
\mathbf{x} \doteq 
\begin{bmatrix}
  x_1 \\
  x_2 \\ 
  x_3 \\
  x_4 
\end{bmatrix}
  `+"",n;return{c(){n=l(i)},l(a){n=f(a,i)},m(a,u){o(a,n,u)},p:P,d(a){a&&t(n)}}}function Ba(s){let i=String.raw`\mathcal{A}`+"",n;return{c(){n=l(i)},l(a){n=f(a,i)},m(a,u){o(a,n,u)},p:P,d(a){a&&t(n)}}}function Wa(s){let i=String.raw`\hat{Q}(s, a, \mathbf{w})`+"",n;return{c(){n=l(i)},l(a){n=f(a,i)},m(a,u){o(a,n,u)},p:P,d(a){a&&t(n)}}}function Ca(s){let i=String.raw`\mathbf{x}`+"",n;return{c(){n=l(i)},l(a){n=f(a,i)},m(a,u){o(a,n,u)},p:P,d(a){a&&t(n)}}}function Ha(s){let i=String.raw`\arg\max_a \hat{Q}(s, a, \mathbf{w})`+"",n;return{c(){n=l(i)},l(a){n=f(a,i)},m(a,u){o(a,n,u)},p:P,d(a){a&&t(n)}}}function Oa(s){let i=String.raw`
  w_{t+1} \doteq w_t - \frac{1}{2}\alpha\nabla[r + \gamma \max_{a'} \hat{Q}(s', a', \mathbf{w}_t) - \hat{Q}(s, a, \mathbf{w}_t)]^2
  `+"",n;return{c(){n=l(i)},l(a){n=f(a,i)},m(a,u){o(a,n,u)},p:P,d(a){a&&t(n)}}}function Va(s){let i=String.raw`e_t = (s_t, a_t, r_t, s_{t+1}, d_t)`+"",n;return{c(){n=l(i)},l(a){n=f(a,i)},m(a,u){o(a,n,u)},p:P,d(a){a&&t(n)}}}function Fa(s){let i=String.raw`D_t = \{e_1, ... , e_t\}`+"",n;return{c(){n=l(i)},l(a){n=f(a,i)},m(a,u){o(a,n,u)},p:P,d(a){a&&t(n)}}}function Ua(s){let i=String.raw`MSE \doteq \mathbb{E}_{(s, a, r, s', d) \sim U(D)}[(r + \gamma \max_{a'} Q(s', a', \mathbf{w}) - Q(s, a, \mathbf{w}))^2]`+"",n;return{c(){n=l(i)},l(a){n=f(a,i)},m(a,u){o(a,n,u)},p:P,d(a){a&&t(n)}}}function Xa(s){let i=String.raw`Q(s, a, \mathbf{w}`+"",n,a;return{c(){n=l(i),a=l(")")},l(u){n=f(u,i),a=f(u,")")},m(u,w){o(u,n,w),o(u,a,w)},p:P,d(u){u&&t(n),u&&t(a)}}}function Ya(s){let i=String.raw`r + \gamma \max_{a'} Q(s', a', \mathbf{w})`+"",n;return{c(){n=l(i)},l(a){n=f(a,i)},m(a,u){o(a,n,u)},p:P,d(a){a&&t(n)}}}function ja(s){let i=String.raw`w^-`+"",n;return{c(){n=l(i)},l(a){n=f(a,i)},m(a,u){o(a,n,u)},p:P,d(a){a&&t(n)}}}function La(s){let i=String.raw`Q(s, a, \mathbf{w})`+"",n;return{c(){n=l(i)},l(a){n=f(a,i)},m(a,u){o(a,n,u)},p:P,d(a){a&&t(n)}}}function Ga(s){let i=String.raw`Q(s', a', \mathbf{w}^-)`+"",n;return{c(){n=l(i)},l(a){n=f(a,i)},m(a,u){o(a,n,u)},p:P,d(a){a&&t(n)}}}function Ra(s){let i=String.raw`(s, a, r, s', t) \sim U(D)`+"",n;return{c(){n=l(i)},l(a){n=f(a,i)},m(a,u){o(a,n,u)},p:P,d(a){a&&t(n)}}}function Ja(s){let i;return{c(){i=l("w^-")},l(n){i=f(n,"w^-")},m(n,a){o(n,i,a)},d(n){n&&t(i)}}}function Ka(s){let i=String.raw`MSE \doteq \mathbb{E}_{(s, a, r, s', t) \sim U(D)}[(r + \gamma \max_{a'} Q(s', a', \mathbf{w}^-) - Q(s, a, \mathbf{w}))^2]`+"",n;return{c(){n=l(i)},l(a){n=f(a,i)},m(a,u){o(a,n,u)},p:P,d(a){a&&t(n)}}}function Za(s){let i,n,a,u,w,_,I,A,x,h,$,y,M,Z,N,be,W,_e,C,ye,H,ke,O,xe,R,V,ee,F,Te,U,Ee,Ne,S,St,te,_n,ne,yn,qt,Me,kn,Pt,De,Nt,ae,xn,re,Tn,Mt,Be,En,Bt,Qe,Wt,Y,Dn,ie,Qn,oe,zn,Ct,ht,Ht,We,Ot,Ce,An,Vt,He,In,Ft,se,Ut,Oe,Sn,Xt,Ve,Yt,Fe,qn,jt,Ue,Pn,Lt,Xe,Nn,Gt,j,Mn,le,Bn,fe,Wn,Rt,Ye,Cn,Jt,je,Hn,Kt,ue,Zt,Le,On,en,ze,tn,Ge,Vn,nn,Re,an,Je,Fn,rn,L,Un,he,Xn,pe,Yn,on,Ke,jn,sn,Ae,ln,X,Ln,me,Gn,ce,Rn,$e,Jn,fn,Ze,Kn,un,Ie,hn,et,pn,tt,Zn,mn,G,ea,we,ta,de,na,cn,ve,$n,nt,wn,at,aa,dn,rt,ra,vn,it,ia,gn,ot,bn;return W=new q({props:{$$slots:{default:[Ia]},$$scope:{ctx:s}}}),C=new q({props:{$$slots:{default:[Sa]},$$scope:{ctx:s}}}),H=new q({props:{$$slots:{default:[qa]},$$scope:{ctx:s}}}),O=new q({props:{$$slots:{default:[Pa]},$$scope:{ctx:s}}}),V=new Ta({}),U=new q({props:{$$slots:{default:[Na]},$$scope:{ctx:s}}}),S=new q({props:{$$slots:{default:[Ma]},$$scope:{ctx:s}}}),ne=new q({props:{$$slots:{default:[Ba]},$$scope:{ctx:s}}}),De=new pa({props:{config:s[0]}}),re=new q({props:{$$slots:{default:[Wa]},$$scope:{ctx:s}}}),Qe=new pa({props:{config:s[1]}}),ie=new q({props:{$$slots:{default:[Ca]},$$scope:{ctx:s}}}),oe=new q({props:{$$slots:{default:[Ha]},$$scope:{ctx:s}}}),se=new q({props:{$$slots:{default:[Oa]},$$scope:{ctx:s}}}),le=new q({props:{$$slots:{default:[Va]},$$scope:{ctx:s}}}),fe=new q({props:{$$slots:{default:[Fa]},$$scope:{ctx:s}}}),ue=new q({props:{$$slots:{default:[Ua]},$$scope:{ctx:s}}}),ze=new Ea({}),he=new q({props:{$$slots:{default:[Xa]},$$scope:{ctx:s}}}),pe=new q({props:{$$slots:{default:[Ya]},$$scope:{ctx:s}}}),Ae=new wa({}),me=new q({props:{$$slots:{default:[ja]},$$scope:{ctx:s}}}),ce=new q({props:{$$slots:{default:[La]},$$scope:{ctx:s}}}),$e=new q({props:{$$slots:{default:[Ga]},$$scope:{ctx:s}}}),Ie=new wa({props:{frozen:!0}}),we=new q({props:{$$slots:{default:[Ra]},$$scope:{ctx:s}}}),de=new q({props:{$$slots:{default:[Ja]},$$scope:{ctx:s}}}),ve=new q({props:{$$slots:{default:[Ka]},$$scope:{ctx:s}}}),{c(){i=g("p"),n=l(`The paper on the deep Q-network (often abbreviated as DQN) by DeepMind is
    regarded as one of the most seminal papers on modern reinforcement learning.
    The research that came from DeepMind showed how a combination of Q-learning
    with deep neural networks can be applied to Atari games. For many Atari
    games the DQN Agent even outperformed professional human players. The
    results were inspiring and groundbreaking in many respects, but the most
    important contribution of the paper was probably the rejuvenation of a field
    that seemed to be forgotten by the public. DQN spurred a research streak
    that continues up to this day.`),a=m(),u=g("p"),w=l(`Many of the solutions by the DQN seem to show creativity and even if you are
    not a reinforcement learning enthusiast, you will most likely find the
    playthroughs of Atari games by the DQN agent to be almost magical.`),_=m(),I=g("p"),A=l(`In this chapter we are going to explore the components that made the deep
    Q-network successful. We will look at how Atari games can be solved, but
    before that we are going to explore solutions to simpler OpenAI gym
    environments, because those can be solved quicker, especially if you do not
    possess a modern Nvidia graphics card.`),x=m(),h=g("div"),$=m(),y=g("h2"),M=l("Architecture"),Z=m(),N=g("p"),be=l(`In value based deep reinforcement learning we are utilizing a neural network
    to represent the approximate action value function `),T(W.$$.fragment),_e=l(", where "),T(C.$$.fragment),ye=l(" is the state of the environment, "),T(H.$$.fragment),ke=l(` is
    the action and `),T(O.$$.fragment),xe=l(` is the weight vector of
    the neural network.`),R=m(),T(V.$$.fragment),ee=m(),F=g("p"),Te=l(`If we take the cart pole environment as an example of a simple OpenAI gym
    environment, we will face a state `),T(U.$$.fragment),Ee=l(` represented by a 4 dimensional
    vector.`),Ne=m(),T(S.$$.fragment),St=m(),te=g("p"),_n=l("There are two possible actions in the action space "),T(ne.$$.fragment),yn=l(`. The agent can choose the action to move left (action 0) or to move right
    (action 1).`),qt=m(),Me=g("p"),kn=l(`Alltogether there are 5 inputs into the neural network, the four variables
    representing the state and one variable representing the value of the chosen
    action (0 or 1). The input is processed in a series of hidden layers by
    applying linear transformations and non-linear activation functions. The
    output layer is a single neuron that represents the action value of the
    state action combination.`),Pt=m(),T(De.$$.fragment),Nt=m(),ae=g("p"),xn=l(`The architecture above is theoretically sound, but has a significant
    practical drawback. In order to improve the policy the agent needs to act
    greedy with respect to the action value function`),T(re.$$.fragment),Tn=l(`. The agent needs to take the action with the hightest action value and
    that implies that we need to calculate the action values for all available
    actions in order to determine which action has the highest value. Each
    calculation would require the full pass through the neural network which
    gets computationally costly with the increased complexity of the network and
    the number of possible actions.`),Mt=m(),Be=g("p"),En=l(`The below architecture on the other hand is the one that is similar in
    spirit to the solution that DeepMind ended up using for DQN.`),Bt=m(),T(Qe.$$.fragment),Wt=m(),Y=g("p"),Dn=l("The input of the neural network consists only of the state representation"),T(ie.$$.fragment),Qn=l(`. The output layer has as many neurons as there are available actions in
    the environment. In the cart pole environment for example only two actions
    are available, therefore the neural network has two output neurons. With the
    above architecture only one pass through the network is required and
    choosing the action with the hightest value corresponds to taking the
    `),T(oe.$$.fragment),zn=l(" operation."),Ct=m(),ht=g("p"),Ht=m(),We=g("div"),Ot=m(),Ce=g("h2"),An=l("Divergence"),Vt=m(),He=g("p"),In=l(`In tabular Q-learning each experience is thrown away as soon as it has been
    used for training. The tabular Q-function contains all the required
    information to arrive at the optimal value function. If we apply the same
    logic to approximative value functions, that would imply that we would take
    a single gradient descent step after taking a single action and throw that
    experience away.`),Ft=m(),T(se.$$.fragment),Ut=m(),Oe=g("p"),Sn=l(`This naive implementation of deep Q-learning will most likely diverge. We
    might observe improvements until a breaking point, when the neural network
    will start failing catastrophically, never to recover again. In some cases
    we would be able to achieve descent results in simple environments like the
    cart pole environment, but achieving good results for Atari games is almost
    impossible with the naive above implementation.`),Xt=m(),Ve=g("div"),Yt=m(),Fe=g("h2"),qn=l("Experience Replay"),jt=m(),Ue=g("p"),Pn=l(`One of the reasons for the divergence of value based deep reinforcement
    learning algorithms is the high correlation of sequential observations. When
    we use gradient descent at each timestep we need to assume, that the current
    observation of the environment is very close to the previous observation. To
    convince yourself you can look at the example of the cart pole above. The
    cart position does not jump from 0 to 1, but moves slowly in incremental
    steps. From supervised learning we know, that the gradient descent algorithm
    assumes the data to be i.i.d. (independently and identically distributed),
    which is a hard limitation given the sequential nature of reinforcement
    learning. Thus sequential observations contribute to the destabilization of
    the learning process.`),Lt=m(),Xe=g("p"),Nn=l(`Additionally we should also remember that in supervised learning, we rarely
    use stochastic gradient descent, instead we use batch gradient descent. This
    has the advantage of reducing the variance of a single observation and of
    making the calculations more efficient through parallel processing on the
    GPU.`),Gt=m(),j=g("p"),Mn=l(`The experience replay technique uses a data structure called memory buffer
    to alleviate the above problem. Each experience tuple `),T(le.$$.fragment),Bn=l(` is stored in the memory buffer: a data structure with limited capacity. At
    each time step the agent faces a certain observation, uses epsilon-greedy action
    selection and collects the corresponding reward and next observation. The whole
    tuple is pushed into the memory buffer `),T(fe.$$.fragment),Wn=l(`. At full capacity the memory buffer removes the oldest tuple to make room
    for the new experience.`),Rt=m(),Ye=g("p"),Cn=l(`Before the agent starts optimizing the value function, there is usually a so
    called warm-up phase. In that phase the agent collects random experience, in
    order to fill the memory buffer with a minimum amount of tuples.`),Jt=m(),je=g("p"),Hn=l(`The agent learns only from the collected experiences and never online. At
    each time step the agent gets a randomized batch from the memory buffer and
    uses the whole batch to apply batch gradient descent. Using experience
    replay the mean squared error can be defined as follows.`),Kt=m(),T(ue.$$.fragment),Zt=m(),Le=g("p"),On=l(`We provide the interactive example of the experience replay below. We
    suggest you play with it in order to get a better intuitive understanding.
    The batch size corresponds to 3, the warmup phase is 6 and the maximum
    length of the buffer is 15. When you take a step a new experience is
    collected and the batch is chosen randomly from the available experiece
    tuples in the memory buffer. The oldest tuple is thrown away once the
    maximum size is reached.`),en=m(),T(ze.$$.fragment),tn=m(),Ge=g("p"),Vn=l(`The maximum length of the buffer and the batch size depend on the task the
    agent needs to solve. In the original implementation by DeepMind the memory
    size corresponded to 1,000,000 and batch size to 32. Depending on your
    hardware you might need to reduce the memory size. Especially for simpler
    tasks a memory size between 10,000 and 100,000 is usually sufficient.`),nn=m(),Re=g("div"),an=m(),Je=g("h2"),Fn=l("Frozen Target Network"),rn=m(),L=g("p"),Un=l(`The second problem that the agent faces is the correlation between the
    action values `),T(he.$$.fragment),Xn=l(` and the target
    values
    `),T(pe.$$.fragment),Yn=l(`,
    because the same action-value function is used for the target value and the
    current action value.`),on=m(),Ke=g("p"),jn=l(`Below we see an interactive game to explain the problem in a more intuitive
    manner. The blue circle represents our estimate we are trying to improve.
    The red circle is our target. When you take a step our estimate moves into
    the direction of the target. This is the whole point of gradient descent,
    you want to minimized the distance between the estimte and the target.
    Because the estimate and the bootstrapped value share the same weights of
    the neural network the weights of the target change as well, which forces
    the target to move. A great analogy that is often used in the literature is
    a dog chasing it's own tail. Catching up seems imporssible, which can have a
    destabilizing effect on the neural network.`),sn=m(),T(Ae.$$.fragment),ln=m(),X=g("p"),Ln=l(`In order to mitigate the destabilization the researchers at DeepMind used a
    neural network with frozen weights `),T(me.$$.fragment),Gn=l(` for the calculation
    of the target value. This weights are held constant for a period of time and
    only periodically are the weights copied from the action value function `),T(ce.$$.fragment),Rn=l(" to the neural network used for the calculation of the target "),T($e.$$.fragment),Jn=l(`. In the original paper the update frequency is 10,000 steps for Atari
    games, but for simpler environments the frequency is usually much shorter.`),fn=m(),Ze=g("p"),Kn=l(`Below if a similar interactive example using a frozen target network. The
    estimate moves into the direction of the target for 3 steps. In those three
    steps the weights of the target network are held constant. After the three
    steps the weights are copied from the blue estimate into the red target.`),un=m(),T(Ie.$$.fragment),hn=m(),et=g("div"),pn=m(),tt=g("h2"),Zn=l("Mean Squared Error"),mn=m(),G=g("p"),ea=l(`The final mean squared error calculation of the DQN is defined below. The
    optimization is done by drawing a batch of experiences from the memory
    buffer `),T(we.$$.fragment),ta=l(` using the uniform
    distribution. The estimated action value function and the action value function
    used for bootstrapping use the same architecture, but the bootstrapped calculation
    utilizes frozen weights `),T(de.$$.fragment),na=l("."),cn=m(),T(ve.$$.fragment),$n=m(),nt=g("div"),wn=m(),at=g("h2"),aa=l("Atari"),dn=m(),rt=g("h3"),ra=l("Architecture"),vn=m(),it=g("h3"),ia=l("Preprocessing"),gn=m(),ot=g("div"),this.h()},l(e){i=b(e,"P",{});var r=d(i);n=f(r,`The paper on the deep Q-network (often abbreviated as DQN) by DeepMind is
    regarded as one of the most seminal papers on modern reinforcement learning.
    The research that came from DeepMind showed how a combination of Q-learning
    with deep neural networks can be applied to Atari games. For many Atari
    games the DQN Agent even outperformed professional human players. The
    results were inspiring and groundbreaking in many respects, but the most
    important contribution of the paper was probably the rejuvenation of a field
    that seemed to be forgotten by the public. DQN spurred a research streak
    that continues up to this day.`),r.forEach(t),a=c(e),u=b(e,"P",{});var pt=d(u);w=f(pt,`Many of the solutions by the DQN seem to show creativity and even if you are
    not a reinforcement learning enthusiast, you will most likely find the
    playthroughs of Atari games by the DQN agent to be almost magical.`),pt.forEach(t),_=c(e),I=b(e,"P",{});var mt=d(I);A=f(mt,`In this chapter we are going to explore the components that made the deep
    Q-network successful. We will look at how Atari games can be solved, but
    before that we are going to explore solutions to simpler OpenAI gym
    environments, because those can be solved quicker, especially if you do not
    possess a modern Nvidia graphics card.`),mt.forEach(t),x=c(e),h=b(e,"DIV",{class:!0}),d(h).forEach(t),$=c(e),y=b(e,"H2",{});var ct=d(y);M=f(ct,"Architecture"),ct.forEach(t),Z=c(e),N=b(e,"P",{});var B=d(N);be=f(B,`In value based deep reinforcement learning we are utilizing a neural network
    to represent the approximate action value function `),E(W.$$.fragment,B),_e=f(B,", where "),E(C.$$.fragment,B),ye=f(B," is the state of the environment, "),E(H.$$.fragment,B),ke=f(B,` is
    the action and `),E(O.$$.fragment,B),xe=f(B,` is the weight vector of
    the neural network.`),B.forEach(t),R=c(e),E(V.$$.fragment,e),ee=c(e),F=b(e,"P",{});var Se=d(F);Te=f(Se,`If we take the cart pole environment as an example of a simple OpenAI gym
    environment, we will face a state `),E(U.$$.fragment,Se),Ee=f(Se,` represented by a 4 dimensional
    vector.`),Se.forEach(t),Ne=c(e),E(S.$$.fragment,e),St=c(e),te=b(e,"P",{});var qe=d(te);_n=f(qe,"There are two possible actions in the action space "),E(ne.$$.fragment,qe),yn=f(qe,`. The agent can choose the action to move left (action 0) or to move right
    (action 1).`),qe.forEach(t),qt=c(e),Me=b(e,"P",{});var $t=d(Me);kn=f($t,`Alltogether there are 5 inputs into the neural network, the four variables
    representing the state and one variable representing the value of the chosen
    action (0 or 1). The input is processed in a series of hidden layers by
    applying linear transformations and non-linear activation functions. The
    output layer is a single neuron that represents the action value of the
    state action combination.`),$t.forEach(t),Pt=c(e),E(De.$$.fragment,e),Nt=c(e),ae=b(e,"P",{});var Pe=d(ae);xn=f(Pe,`The architecture above is theoretically sound, but has a significant
    practical drawback. In order to improve the policy the agent needs to act
    greedy with respect to the action value function`),E(re.$$.fragment,Pe),Tn=f(Pe,`. The agent needs to take the action with the hightest action value and
    that implies that we need to calculate the action values for all available
    actions in order to determine which action has the highest value. Each
    calculation would require the full pass through the neural network which
    gets computationally costly with the increased complexity of the network and
    the number of possible actions.`),Pe.forEach(t),Mt=c(e),Be=b(e,"P",{});var wt=d(Be);En=f(wt,`The below architecture on the other hand is the one that is similar in
    spirit to the solution that DeepMind ended up using for DQN.`),wt.forEach(t),Bt=c(e),E(Qe.$$.fragment,e),Wt=c(e),Y=b(e,"P",{});var J=d(Y);Dn=f(J,"The input of the neural network consists only of the state representation"),E(ie.$$.fragment,J),Qn=f(J,`. The output layer has as many neurons as there are available actions in
    the environment. In the cart pole environment for example only two actions
    are available, therefore the neural network has two output neurons. With the
    above architecture only one pass through the network is required and
    choosing the action with the hightest value corresponds to taking the
    `),E(oe.$$.fragment,J),zn=f(J," operation."),J.forEach(t),Ct=c(e),ht=b(e,"P",{}),d(ht).forEach(t),Ht=c(e),We=b(e,"DIV",{class:!0}),d(We).forEach(t),Ot=c(e),Ce=b(e,"H2",{});var dt=d(Ce);An=f(dt,"Divergence"),dt.forEach(t),Vt=c(e),He=b(e,"P",{});var vt=d(He);In=f(vt,`In tabular Q-learning each experience is thrown away as soon as it has been
    used for training. The tabular Q-function contains all the required
    information to arrive at the optimal value function. If we apply the same
    logic to approximative value functions, that would imply that we would take
    a single gradient descent step after taking a single action and throw that
    experience away.`),vt.forEach(t),Ft=c(e),E(se.$$.fragment,e),Ut=c(e),Oe=b(e,"P",{});var gt=d(Oe);Sn=f(gt,`This naive implementation of deep Q-learning will most likely diverge. We
    might observe improvements until a breaking point, when the neural network
    will start failing catastrophically, never to recover again. In some cases
    we would be able to achieve descent results in simple environments like the
    cart pole environment, but achieving good results for Atari games is almost
    impossible with the naive above implementation.`),gt.forEach(t),Xt=c(e),Ve=b(e,"DIV",{class:!0}),d(Ve).forEach(t),Yt=c(e),Fe=b(e,"H2",{});var bt=d(Fe);qn=f(bt,"Experience Replay"),bt.forEach(t),jt=c(e),Ue=b(e,"P",{});var _t=d(Ue);Pn=f(_t,`One of the reasons for the divergence of value based deep reinforcement
    learning algorithms is the high correlation of sequential observations. When
    we use gradient descent at each timestep we need to assume, that the current
    observation of the environment is very close to the previous observation. To
    convince yourself you can look at the example of the cart pole above. The
    cart position does not jump from 0 to 1, but moves slowly in incremental
    steps. From supervised learning we know, that the gradient descent algorithm
    assumes the data to be i.i.d. (independently and identically distributed),
    which is a hard limitation given the sequential nature of reinforcement
    learning. Thus sequential observations contribute to the destabilization of
    the learning process.`),_t.forEach(t),Lt=c(e),Xe=b(e,"P",{});var yt=d(Xe);Nn=f(yt,`Additionally we should also remember that in supervised learning, we rarely
    use stochastic gradient descent, instead we use batch gradient descent. This
    has the advantage of reducing the variance of a single observation and of
    making the calculations more efficient through parallel processing on the
    GPU.`),yt.forEach(t),Gt=c(e),j=b(e,"P",{});var K=d(j);Mn=f(K,`The experience replay technique uses a data structure called memory buffer
    to alleviate the above problem. Each experience tuple `),E(le.$$.fragment,K),Bn=f(K,` is stored in the memory buffer: a data structure with limited capacity. At
    each time step the agent faces a certain observation, uses epsilon-greedy action
    selection and collects the corresponding reward and next observation. The whole
    tuple is pushed into the memory buffer `),E(fe.$$.fragment,K),Wn=f(K,`. At full capacity the memory buffer removes the oldest tuple to make room
    for the new experience.`),K.forEach(t),Rt=c(e),Ye=b(e,"P",{});var kt=d(Ye);Cn=f(kt,`Before the agent starts optimizing the value function, there is usually a so
    called warm-up phase. In that phase the agent collects random experience, in
    order to fill the memory buffer with a minimum amount of tuples.`),kt.forEach(t),Jt=c(e),je=b(e,"P",{});var xt=d(je);Hn=f(xt,`The agent learns only from the collected experiences and never online. At
    each time step the agent gets a randomized batch from the memory buffer and
    uses the whole batch to apply batch gradient descent. Using experience
    replay the mean squared error can be defined as follows.`),xt.forEach(t),Kt=c(e),E(ue.$$.fragment,e),Zt=c(e),Le=b(e,"P",{});var Tt=d(Le);On=f(Tt,`We provide the interactive example of the experience replay below. We
    suggest you play with it in order to get a better intuitive understanding.
    The batch size corresponds to 3, the warmup phase is 6 and the maximum
    length of the buffer is 15. When you take a step a new experience is
    collected and the batch is chosen randomly from the available experiece
    tuples in the memory buffer. The oldest tuple is thrown away once the
    maximum size is reached.`),Tt.forEach(t),en=c(e),E(ze.$$.fragment,e),tn=c(e),Ge=b(e,"P",{});var Et=d(Ge);Vn=f(Et,`The maximum length of the buffer and the batch size depend on the task the
    agent needs to solve. In the original implementation by DeepMind the memory
    size corresponded to 1,000,000 and batch size to 32. Depending on your
    hardware you might need to reduce the memory size. Especially for simpler
    tasks a memory size between 10,000 and 100,000 is usually sufficient.`),Et.forEach(t),nn=c(e),Re=b(e,"DIV",{class:!0}),d(Re).forEach(t),an=c(e),Je=b(e,"H2",{});var Dt=d(Je);Fn=f(Dt,"Frozen Target Network"),Dt.forEach(t),rn=c(e),L=b(e,"P",{});var st=d(L);Un=f(st,`The second problem that the agent faces is the correlation between the
    action values `),E(he.$$.fragment,st),Xn=f(st,` and the target
    values
    `),E(pe.$$.fragment,st),Yn=f(st,`,
    because the same action-value function is used for the target value and the
    current action value.`),st.forEach(t),on=c(e),Ke=b(e,"P",{});var oa=d(Ke);jn=f(oa,`Below we see an interactive game to explain the problem in a more intuitive
    manner. The blue circle represents our estimate we are trying to improve.
    The red circle is our target. When you take a step our estimate moves into
    the direction of the target. This is the whole point of gradient descent,
    you want to minimized the distance between the estimte and the target.
    Because the estimate and the bootstrapped value share the same weights of
    the neural network the weights of the target change as well, which forces
    the target to move. A great analogy that is often used in the literature is
    a dog chasing it's own tail. Catching up seems imporssible, which can have a
    destabilizing effect on the neural network.`),oa.forEach(t),sn=c(e),E(Ae.$$.fragment,e),ln=c(e),X=b(e,"P",{});var ge=d(X);Ln=f(ge,`In order to mitigate the destabilization the researchers at DeepMind used a
    neural network with frozen weights `),E(me.$$.fragment,ge),Gn=f(ge,` for the calculation
    of the target value. This weights are held constant for a period of time and
    only periodically are the weights copied from the action value function `),E(ce.$$.fragment,ge),Rn=f(ge," to the neural network used for the calculation of the target "),E($e.$$.fragment,ge),Jn=f(ge,`. In the original paper the update frequency is 10,000 steps for Atari
    games, but for simpler environments the frequency is usually much shorter.`),ge.forEach(t),fn=c(e),Ze=b(e,"P",{});var sa=d(Ze);Kn=f(sa,`Below if a similar interactive example using a frozen target network. The
    estimate moves into the direction of the target for 3 steps. In those three
    steps the weights of the target network are held constant. After the three
    steps the weights are copied from the blue estimate into the red target.`),sa.forEach(t),un=c(e),E(Ie.$$.fragment,e),hn=c(e),et=b(e,"DIV",{class:!0}),d(et).forEach(t),pn=c(e),tt=b(e,"H2",{});var la=d(tt);Zn=f(la,"Mean Squared Error"),la.forEach(t),mn=c(e),G=b(e,"P",{});var lt=d(G);ea=f(lt,`The final mean squared error calculation of the DQN is defined below. The
    optimization is done by drawing a batch of experiences from the memory
    buffer `),E(we.$$.fragment,lt),ta=f(lt,` using the uniform
    distribution. The estimated action value function and the action value function
    used for bootstrapping use the same architecture, but the bootstrapped calculation
    utilizes frozen weights `),E(de.$$.fragment,lt),na=f(lt,"."),lt.forEach(t),cn=c(e),E(ve.$$.fragment,e),$n=c(e),nt=b(e,"DIV",{class:!0}),d(nt).forEach(t),wn=c(e),at=b(e,"H2",{});var fa=d(at);aa=f(fa,"Atari"),fa.forEach(t),dn=c(e),rt=b(e,"H3",{});var ua=d(rt);ra=f(ua,"Architecture"),ua.forEach(t),vn=c(e),it=b(e,"H3",{});var ha=d(it);ia=f(ha,"Preprocessing"),ha.forEach(t),gn=c(e),ot=b(e,"DIV",{class:!0}),d(ot).forEach(t),this.h()},h(){v(h,"class","separator"),v(We,"class","separator"),v(Ve,"class","separator"),v(Re,"class","separator"),v(et,"class","separator"),v(nt,"class","separator"),v(ot,"class","separator")},m(e,r){o(e,i,r),p(i,n),o(e,a,r),o(e,u,r),p(u,w),o(e,_,r),o(e,I,r),p(I,A),o(e,x,r),o(e,h,r),o(e,$,r),o(e,y,r),p(y,M),o(e,Z,r),o(e,N,r),p(N,be),D(W,N,null),p(N,_e),D(C,N,null),p(N,ye),D(H,N,null),p(N,ke),D(O,N,null),p(N,xe),o(e,R,r),D(V,e,r),o(e,ee,r),o(e,F,r),p(F,Te),D(U,F,null),p(F,Ee),o(e,Ne,r),D(S,e,r),o(e,St,r),o(e,te,r),p(te,_n),D(ne,te,null),p(te,yn),o(e,qt,r),o(e,Me,r),p(Me,kn),o(e,Pt,r),D(De,e,r),o(e,Nt,r),o(e,ae,r),p(ae,xn),D(re,ae,null),p(ae,Tn),o(e,Mt,r),o(e,Be,r),p(Be,En),o(e,Bt,r),D(Qe,e,r),o(e,Wt,r),o(e,Y,r),p(Y,Dn),D(ie,Y,null),p(Y,Qn),D(oe,Y,null),p(Y,zn),o(e,Ct,r),o(e,ht,r),o(e,Ht,r),o(e,We,r),o(e,Ot,r),o(e,Ce,r),p(Ce,An),o(e,Vt,r),o(e,He,r),p(He,In),o(e,Ft,r),D(se,e,r),o(e,Ut,r),o(e,Oe,r),p(Oe,Sn),o(e,Xt,r),o(e,Ve,r),o(e,Yt,r),o(e,Fe,r),p(Fe,qn),o(e,jt,r),o(e,Ue,r),p(Ue,Pn),o(e,Lt,r),o(e,Xe,r),p(Xe,Nn),o(e,Gt,r),o(e,j,r),p(j,Mn),D(le,j,null),p(j,Bn),D(fe,j,null),p(j,Wn),o(e,Rt,r),o(e,Ye,r),p(Ye,Cn),o(e,Jt,r),o(e,je,r),p(je,Hn),o(e,Kt,r),D(ue,e,r),o(e,Zt,r),o(e,Le,r),p(Le,On),o(e,en,r),D(ze,e,r),o(e,tn,r),o(e,Ge,r),p(Ge,Vn),o(e,nn,r),o(e,Re,r),o(e,an,r),o(e,Je,r),p(Je,Fn),o(e,rn,r),o(e,L,r),p(L,Un),D(he,L,null),p(L,Xn),D(pe,L,null),p(L,Yn),o(e,on,r),o(e,Ke,r),p(Ke,jn),o(e,sn,r),D(Ae,e,r),o(e,ln,r),o(e,X,r),p(X,Ln),D(me,X,null),p(X,Gn),D(ce,X,null),p(X,Rn),D($e,X,null),p(X,Jn),o(e,fn,r),o(e,Ze,r),p(Ze,Kn),o(e,un,r),D(Ie,e,r),o(e,hn,r),o(e,et,r),o(e,pn,r),o(e,tt,r),p(tt,Zn),o(e,mn,r),o(e,G,r),p(G,ea),D(we,G,null),p(G,ta),D(de,G,null),p(G,na),o(e,cn,r),D(ve,e,r),o(e,$n,r),o(e,nt,r),o(e,wn,r),o(e,at,r),p(at,aa),o(e,dn,r),o(e,rt,r),p(rt,ra),o(e,vn,r),o(e,it,r),p(it,ia),o(e,gn,r),o(e,ot,r),bn=!0},p(e,r){const pt={};r&4&&(pt.$$scope={dirty:r,ctx:e}),W.$set(pt);const mt={};r&4&&(mt.$$scope={dirty:r,ctx:e}),C.$set(mt);const ct={};r&4&&(ct.$$scope={dirty:r,ctx:e}),H.$set(ct);const B={};r&4&&(B.$$scope={dirty:r,ctx:e}),O.$set(B);const Se={};r&4&&(Se.$$scope={dirty:r,ctx:e}),U.$set(Se);const qe={};r&4&&(qe.$$scope={dirty:r,ctx:e}),S.$set(qe);const $t={};r&4&&($t.$$scope={dirty:r,ctx:e}),ne.$set($t);const Pe={};r&4&&(Pe.$$scope={dirty:r,ctx:e}),re.$set(Pe);const wt={};r&4&&(wt.$$scope={dirty:r,ctx:e}),ie.$set(wt);const J={};r&4&&(J.$$scope={dirty:r,ctx:e}),oe.$set(J);const dt={};r&4&&(dt.$$scope={dirty:r,ctx:e}),se.$set(dt);const vt={};r&4&&(vt.$$scope={dirty:r,ctx:e}),le.$set(vt);const gt={};r&4&&(gt.$$scope={dirty:r,ctx:e}),fe.$set(gt);const bt={};r&4&&(bt.$$scope={dirty:r,ctx:e}),ue.$set(bt);const _t={};r&4&&(_t.$$scope={dirty:r,ctx:e}),he.$set(_t);const yt={};r&4&&(yt.$$scope={dirty:r,ctx:e}),pe.$set(yt);const K={};r&4&&(K.$$scope={dirty:r,ctx:e}),me.$set(K);const kt={};r&4&&(kt.$$scope={dirty:r,ctx:e}),ce.$set(kt);const xt={};r&4&&(xt.$$scope={dirty:r,ctx:e}),$e.$set(xt);const Tt={};r&4&&(Tt.$$scope={dirty:r,ctx:e}),we.$set(Tt);const Et={};r&4&&(Et.$$scope={dirty:r,ctx:e}),de.$set(Et);const Dt={};r&4&&(Dt.$$scope={dirty:r,ctx:e}),ve.$set(Dt)},i(e){bn||(k(W.$$.fragment,e),k(C.$$.fragment,e),k(H.$$.fragment,e),k(O.$$.fragment,e),k(V.$$.fragment,e),k(U.$$.fragment,e),k(S.$$.fragment,e),k(ne.$$.fragment,e),k(De.$$.fragment,e),k(re.$$.fragment,e),k(Qe.$$.fragment,e),k(ie.$$.fragment,e),k(oe.$$.fragment,e),k(se.$$.fragment,e),k(le.$$.fragment,e),k(fe.$$.fragment,e),k(ue.$$.fragment,e),k(ze.$$.fragment,e),k(he.$$.fragment,e),k(pe.$$.fragment,e),k(Ae.$$.fragment,e),k(me.$$.fragment,e),k(ce.$$.fragment,e),k($e.$$.fragment,e),k(Ie.$$.fragment,e),k(we.$$.fragment,e),k(de.$$.fragment,e),k(ve.$$.fragment,e),bn=!0)},o(e){Q(W.$$.fragment,e),Q(C.$$.fragment,e),Q(H.$$.fragment,e),Q(O.$$.fragment,e),Q(V.$$.fragment,e),Q(U.$$.fragment,e),Q(S.$$.fragment,e),Q(ne.$$.fragment,e),Q(De.$$.fragment,e),Q(re.$$.fragment,e),Q(Qe.$$.fragment,e),Q(ie.$$.fragment,e),Q(oe.$$.fragment,e),Q(se.$$.fragment,e),Q(le.$$.fragment,e),Q(fe.$$.fragment,e),Q(ue.$$.fragment,e),Q(ze.$$.fragment,e),Q(he.$$.fragment,e),Q(pe.$$.fragment,e),Q(Ae.$$.fragment,e),Q(me.$$.fragment,e),Q(ce.$$.fragment,e),Q($e.$$.fragment,e),Q(Ie.$$.fragment,e),Q(we.$$.fragment,e),Q(de.$$.fragment,e),Q(ve.$$.fragment,e),bn=!1},d(e){e&&t(i),e&&t(a),e&&t(u),e&&t(_),e&&t(I),e&&t(x),e&&t(h),e&&t($),e&&t(y),e&&t(Z),e&&t(N),z(W),z(C),z(H),z(O),e&&t(R),z(V,e),e&&t(ee),e&&t(F),z(U),e&&t(Ne),z(S,e),e&&t(St),e&&t(te),z(ne),e&&t(qt),e&&t(Me),e&&t(Pt),z(De,e),e&&t(Nt),e&&t(ae),z(re),e&&t(Mt),e&&t(Be),e&&t(Bt),z(Qe,e),e&&t(Wt),e&&t(Y),z(ie),z(oe),e&&t(Ct),e&&t(ht),e&&t(Ht),e&&t(We),e&&t(Ot),e&&t(Ce),e&&t(Vt),e&&t(He),e&&t(Ft),z(se,e),e&&t(Ut),e&&t(Oe),e&&t(Xt),e&&t(Ve),e&&t(Yt),e&&t(Fe),e&&t(jt),e&&t(Ue),e&&t(Lt),e&&t(Xe),e&&t(Gt),e&&t(j),z(le),z(fe),e&&t(Rt),e&&t(Ye),e&&t(Jt),e&&t(je),e&&t(Kt),z(ue,e),e&&t(Zt),e&&t(Le),e&&t(en),z(ze,e),e&&t(tn),e&&t(Ge),e&&t(nn),e&&t(Re),e&&t(an),e&&t(Je),e&&t(rn),e&&t(L),z(he),z(pe),e&&t(on),e&&t(Ke),e&&t(sn),z(Ae,e),e&&t(ln),e&&t(X),z(me),z(ce),z($e),e&&t(fn),e&&t(Ze),e&&t(un),z(Ie,e),e&&t(hn),e&&t(et),e&&t(pn),e&&t(tt),e&&t(mn),e&&t(G),z(we),z(de),e&&t(cn),z(ve,e),e&&t($n),e&&t(nt),e&&t(wn),e&&t(at),e&&t(dn),e&&t(rt),e&&t(vn),e&&t(it),e&&t(gn),e&&t(ot)}}}function er(s){let i,n,a,u,w,_,I,A,x;return A=new xa({props:{$$slots:{default:[Za]},$$scope:{ctx:s}}}),{c(){i=g("meta"),n=m(),a=g("h1"),u=l("Deep Q-Network (DQN)"),w=m(),_=g("div"),I=m(),T(A.$$.fragment),this.h()},l(h){const $=ka('[data-svelte="svelte-1n7tt4a"]',document.head);i=b($,"META",{name:!0,content:!0}),$.forEach(t),n=c(h),a=b(h,"H1",{});var y=d(a);u=f(y,"Deep Q-Network (DQN)"),y.forEach(t),w=c(h),_=b(h,"DIV",{class:!0}),d(_).forEach(t),I=c(h),E(A.$$.fragment,h),this.h()},h(){document.title="World4AI | Reinforcement Learning | DQN",v(i,"name","description"),v(i,"content","The deep Q-network (DQN) is considered to be one of the seminal works in deep reinforcement learning. The agent was able to achieve human level control in many Atari games and even outperformed professional players."),v(_,"class","separator")},m(h,$){p(document.head,i),o(h,n,$),o(h,a,$),p(a,u),o(h,w,$),o(h,_,$),o(h,I,$),D(A,h,$),x=!0},p(h,[$]){const y={};$&4&&(y.$$scope={dirty:$,ctx:h}),A.$set(y)},i(h){x||(k(A.$$.fragment,h),x=!0)},o(h){Q(A.$$.fragment,h),x=!1},d(h){t(i),h&&t(n),h&&t(a),h&&t(w),h&&t(_),h&&t(I),z(A,h)}}}function tr(s){return[{parameters:{0:{layer:0,type:"input",count:5,annotation:"Input"},1:{layer:1,type:"fc",count:7,input:[0]},2:{layer:2,type:"fc",count:5,input:[1]},3:{layer:3,input:[2],type:"fc",count:1,annotation:"Q(s, a)"}}},{parameters:{0:{layer:0,type:"input",count:4,annotation:"Input"},1:{layer:1,type:"fc",count:7,input:[0]},2:{layer:2,type:"fc",count:5,input:[1]},3:{layer:3,input:[2],type:"fc",count:2,annotation:"Q(s, a)"}}}]}class cr extends da{constructor(i){super(),va(this,i,tr,er,ga,{})}}export{cr as default};
