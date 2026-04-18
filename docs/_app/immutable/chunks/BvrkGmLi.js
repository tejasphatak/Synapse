import"./CWj6FrbW.js";import{p as U,o as V,h as W,i as X,a as f,b as Y,k as Z,l as p,b2 as $,t as _,r as ee,f as M}from"./Coqcv2G2.js";import{i as te}from"./CZeb1kGS.js";import{e as ae,i as ne}from"./X-6ks5ij.js";import{b as C,s as oe}from"./deyIpViy.js";import{p as t}from"./Bbm3UalE.js";const ve=i=>{typeof document>"u"||document.documentElement.style.setProperty("--app-text-scale",`${i}`)};var ie=M('<div class="confetti svelte-rtt661"></div>'),re=M("<div></div>");function ye(i,e){U(e,!0);const b=t(e,"size",3,10),u=t(e,"x",19,()=>[-.5,.5]),m=t(e,"y",19,()=>[.25,1]),r=t(e,"duration",3,2e3),l=t(e,"infinite",3,!1),s=t(e,"delay",19,()=>[0,50]),v=t(e,"colorRange",19,()=>[0,360]),d=t(e,"colorArray",19,()=>[]),z=t(e,"amount",3,50),c=t(e,"iterationCount",3,1),R=t(e,"fallDistance",3,"100px"),k=t(e,"rounded",3,!1),S=t(e,"cone",3,!1),w=t(e,"noGravity",3,!1),A=t(e,"xSpread",3,.15),D=t(e,"destroyOnComplete",3,!0),F=t(e,"disableForReducedMotion",3,!1);let y=$(!1);V(()=>{!D()||l()||typeof c()=="string"||setTimeout(()=>Z(y,!0),(r()+s()[1])*c())});function a(o,n){return Math.random()*(n-o)+o}function G(){return d().length?d()[Math.round(Math.random()*(d().length-1))]:`hsl(${Math.round(a(v()[0],v()[1]))}, 75%, 50%)`}var g=W(),O=X(g);{var T=o=>{var n=re();let h;ae(n,21,()=>({length:z()}),ne,(B,le)=>{var x=ie();_((E,P,j,q,H,I,J,K,L,N,Q)=>C(x,`
        --color: ${E??""};
        --skew: ${P??""}deg,${j??""}deg;
        --rotation-xyz: ${q??""}, ${H??""}, ${I??""};
        --rotation-deg: ${J??""}deg;
        --translate-y-multiplier: ${K??""};
        --translate-x-multiplier: ${L??""};
        --scale: ${N??""};
        --transition-delay: ${Q??""}ms;
        --transition-duration: ${l()?`calc(${r()}ms * var(--scale))`:`${r()}ms`};`),[G,()=>a(-45,45),()=>a(-45,45),()=>a(-10,10),()=>a(-10,10),()=>a(-10,10),()=>a(0,360),()=>a(m()[0],m()[1]),()=>a(u()[0],u()[1]),()=>.1*a(2,10),()=>a(s()[0],s()[1])]),f(B,x)}),ee(n),_(()=>{h=oe(n,1,"confetti-holder svelte-rtt661",null,h,{rounded:k(),cone:S(),"no-gravity":w(),"reduced-motion":F()}),C(n,`
    --fall-distance: ${R()??""};
    --size: ${b()??""}px;
    --x-spread: ${1-A()};
    --transition-iteration-count: ${(l()?"infinite":c())??""};`)}),f(o,n)};te(O,o=>{p(y)||o(T)})}f(i,g),Y()}export{ye as C,ve as s};
//# sourceMappingURL=BvrkGmLi.js.map
