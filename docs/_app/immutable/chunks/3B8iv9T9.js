import"./CWj6FrbW.js";import"./69_IOA4Y.js";import{p as Ve,g as ze,w as le,x as P,y as He,i as Oe,c as a,r,q as de,a as m,l,n as d,t as h,z as o,k as g,d as f,b as Ue,e as We,s as je,m as M,B as U,u as b,f as _}from"./Coqcv2G2.js";import{i as Le}from"./CZeb1kGS.js";import{r as W,a as j}from"./deyIpViy.js";import{b as L}from"./5Vjw72sW.js";import{b as ue}from"./a0CMuves.js";import{p as Ge}from"./Bfc47y5P.js";import{i as Je}from"./CKxXGAmO.js";import{p}from"./Bbm3UalE.js";import{g as Ke}from"./B7vJtv02.js";import{n as fe}from"./DrRO4pwJ.js";import{C as Qe}from"./C0ZmZmYF.js";import{C as Re}from"./BDvDVhcE.js";import{B as Xe}from"./B7FNS67a.js";import{T as q}from"./C8eAJkD3.js";import{C as Ye}from"./a667zsp6.js";var Ze=_('<button class="w-full text-left text-sm py-1.5 px-1 rounded-lg dark:text-gray-300 dark:hover:text-white hover:bg-black/5 dark:hover:bg-gray-850" type="button"><!></button>'),et=_('<input class="w-full text-2xl font-medium bg-transparent outline-hidden font-primary" type="text" required=""/>'),tt=_('<div class="text-sm text-gray-500 shrink-0"> </div>'),at=_('<input class="w-full text-sm disabled:text-gray-500 bg-transparent outline-hidden" type="text" required=""/>'),rt=_('<input class="w-full text-sm bg-transparent outline-hidden" type="text" required=""/>'),it=_('<div class="text-sm text-gray-500"><div class=" bg-yellow-500/20 text-yellow-700 dark:text-yellow-200 rounded-lg px-4 py-3"><div> </div> <ul class=" mt-1 list-disc pl-4 text-xs"><li> </li> <li> </li></ul></div> <div class="my-3"> </div></div>'),st=_('<div class=" flex flex-col justify-between w-full overflow-y-auto h-full"><div class="mx-auto w-full md:px-0 h-full"><form class=" flex flex-col max-h-[100dvh] h-full"><div class="flex flex-col flex-1 overflow-auto h-0 rounded-lg"><div class="w-full mb-2 flex flex-col gap-0.5"><div class="flex w-full items-center"><div class=" shrink-0 mr-2"><!></div> <div class="flex-1"><!></div> <div><!></div></div> <div class=" flex gap-2 px-1 items-center"><!> <!></div></div> <div class="mb-2 flex-1 overflow-auto h-0 rounded-lg"><!></div> <div class="pb-3 flex justify-between"><div class="flex-1 pr-3"><div class="text-xs text-gray-500 line-clamp-2"><span class=" font-semibold dark:text-gray-200"> </span> <br/>— <span class=" font-medium dark:text-gray-400"> </span></div></div> <button class="px-3.5 py-1.5 text-sm font-medium bg-black hover:bg-gray-900 text-white dark:bg-white dark:text-black dark:hover:bg-gray-100 transition rounded-full" type="submit"> </button></div></div></form></div></div> <!>',1);function kt(ce,c){Ve(c,!1);const e=()=>We(pe,"$i18n",ve),[ve,me]=je(),pe=ze("i18n");let w=M(null),A=M(!1),_e=p(c,"onSave",8,()=>{}),x=p(c,"edit",8,!1),G=p(c,"clone",8,!1),k=p(c,"id",12,""),y=p(c,"name",12,""),$=p(c,"meta",28,()=>({description:""})),v=p(c,"content",12,""),I=M("");const he=()=>{g(I,v())};let F=M(),ge=`"""
title: Example Filter
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.1
"""

from pydantic import BaseModel, Field
from typing import Optional


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        max_turns: int = Field(
            default=8, description="Maximum allowable conversation turns for a user."
        )
        pass

    class UserValves(BaseModel):
        max_turns: int = Field(
            default=4, description="Maximum allowable conversation turns for a user."
        )
        pass

    def __init__(self):
        # Indicates custom file handling logic. This flag helps disengage default routines in favor of custom
        # implementations, informing the WebUI to defer file-related operations to designated methods within this class.
        # Alternatively, you can remove the files directly from the body in from the inlet hook
        # self.file_handler = True

        # Initialize 'valves' with specific configurations. Using 'Valves' instance helps encapsulate settings,
        # which ensures settings are managed cohesively and not confused with operational flags like 'file_handler'.
        self.valves = self.Valves()
        pass

    def inlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        # Modify the request body or validate it before processing by the chat completion API.
        # This function is the pre-processor for the API where various checks on the input can be performed.
        # It can also modify the request before sending it to the API.
        print(f"inlet:{__name__}")
        print(f"inlet:body:{body}")
        print(f"inlet:user:{__user__}")

        if __user__.get("role", "admin") in ["user", "admin"]:
            messages = body.get("messages", [])

            max_turns = min(__user__["valves"].max_turns, self.valves.max_turns)
            if len(messages) > max_turns:
                raise Exception(
                    f"Conversation turn limit exceeded. Max turns: {max_turns}"
                )

        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        # Modify or analyze the response body after processing by the API.
        # This function is the post-processor for the API, which can be used to modify the response
        # or perform additional checks and analytics.
        print(f"outlet:{__name__}")
        print(f"outlet:body:{body}")
        print(f"outlet:user:{__user__}")

        return body
`;const xe=async()=>{_e()({id:k(),name:y(),meta:$(),content:v()})},J=async()=>{if(l(F)){v(l(I)),await U();const t=await l(F).formatPythonCodeHandler();await U(),v(l(I)),await U(),t||console.warn("Code formatting failed or was skipped, saving unformatted code"),xe()}};le(()=>P(v()),()=>{v()&&he()}),le(()=>(P(y()),P(x()),P(G()),fe),()=>{y()&&!x()&&!G()&&k(fe(y()))}),He(),Je();var K=st(),B=Oe(K),Q=a(B),C=a(Q),R=a(C),T=a(R),E=a(T),D=a(E),ye=a(D);{let t=b(()=>(e(),o(()=>e().t("Back"))));q(ye,{get content(){return l(t)},children:(n,u)=>{var i=Ze(),s=a(i);Ye(s,{strokeWidth:"2.5"}),r(i),de("click",i,()=>{Ke("/admin/functions")}),m(n,i)},$$slots:{default:!0}})}r(D);var N=d(D,2),be=a(N);{let t=b(()=>(e(),o(()=>e().t("e.g. My Filter"))));q(be,{get content(){return l(t)},placement:"top-start",children:(n,u)=>{var i=et();W(i),h(s=>j(i,"placeholder",s),[()=>(e(),o(()=>e().t("Function Name")))]),L(i,y),m(n,i)},$$slots:{default:!0}})}r(N);var X=d(N,2),we=a(X);{let t=b(()=>(e(),o(()=>e().t("Function"))));Xe(we,{type:"muted",get content(){return l(t)}})}r(X),r(E);var Y=d(E,2),Z=a(Y);{var ke=t=>{var n=tt(),u=a(n,!0);r(n),h(()=>f(u,k())),m(t,n)},$e=t=>{{let n=b(()=>(e(),o(()=>e().t("e.g. my_filter"))));q(t,{className:"w-full",get content(){return l(n)},placement:"top-start",children:(u,i)=>{var s=at();W(s),h(H=>{j(s,"placeholder",H),s.disabled=x()},[()=>(e(),o(()=>e().t("Function ID")))]),L(s,k),m(u,s)},$$slots:{default:!0}})}};Le(Z,t=>{x()?t(ke):t($e,-1)})}var Ie=d(Z,2);{let t=b(()=>(e(),o(()=>e().t("e.g. A filter to remove profanity from text"))));q(Ie,{className:"w-full self-center items-center flex",get content(){return l(t)},placement:"top-start",children:(n,u)=>{var i=rt();W(i),h(s=>j(i,"placeholder",s),[()=>(e(),o(()=>e().t("Function Description")))]),L(i,()=>$().description,s=>$($().description=s,!0)),m(n,i)},$$slots:{default:!0}})}r(Y),r(T);var S=d(T,2),Fe=a(S);ue(Qe(Fe,{get value(){return v()},lang:"python",boilerplate:ge,onChange:t=>{g(I,t)},onSave:async()=>{l(w)&&l(w).requestSubmit()},$$legacy:!0}),t=>g(F,t),()=>l(F)),r(S);var ee=d(S,2),V=a(ee),te=a(V),z=a(te),Ce=a(z,!0);r(z);var ae=d(z),re=d(ae,3),Pe=a(re,!0);r(re),r(te),r(V);var ie=d(V,2),Me=a(ie,!0);r(ie),r(ee),r(R),r(C),ue(C,t=>g(w,t),()=>l(w)),r(Q),r(B);var qe=d(B,2);Re(qe,{get show(){return l(A)},set show(t){g(A,t)},$$events:{confirm:()=>{J()}},children:(t,n)=>{var u=it(),i=a(u),s=a(i),H=a(s,!0);r(s);var se=d(s,2),O=a(se),Ae=a(O,!0);r(O);var oe=d(O,2),Be=a(oe,!0);r(oe),r(se),r(i);var ne=d(i,2),Te=a(ne,!0);r(ne),r(u),h((Ee,De,Ne,Se)=>{f(H,Ee),f(Ae,De),f(Be,Ne),f(Te,Se)},[()=>(e(),o(()=>e().t("Please carefully review the following warnings:"))),()=>(e(),o(()=>e().t("Functions allow arbitrary code execution."))),()=>(e(),o(()=>e().t("Do not install functions from sources you do not fully trust."))),()=>(e(),o(()=>e().t("I acknowledge that I have read and I understand the implications of my action. I am aware of the risks associated with executing arbitrary code and I have verified the trustworthiness of the source.")))]),m(t,u)},$$slots:{default:!0},$$legacy:!0}),h((t,n,u,i)=>{f(Ce,t),f(ae,` ${n??""} `),f(Pe,u),f(Me,i)},[()=>(e(),o(()=>e().t("Warning:"))),()=>(e(),o(()=>e().t("Functions allow arbitrary code execution."))),()=>(e(),o(()=>e().t("don't install random functions from sources you don't trust."))),()=>(e(),o(()=>e().t("Save")))]),de("submit",C,Ge(()=>{x()?J():g(A,!0)})),m(ce,K),Ue(),me()}export{kt as F};
//# sourceMappingURL=3B8iv9T9.js.map
