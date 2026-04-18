import"./CWj6FrbW.js";import"./69_IOA4Y.js";import{p as ze,g as Ke,w as he,x as N,y as Je,i as Qe,k as p,l as n,n as u,c as s,r as o,t as x,z as i,q as K,a as y,d as f,b as Ve,e as ge,s as Xe,m as E,B as J,u as b,f as k}from"./Coqcv2G2.js";import{i as Ze}from"./CZeb1kGS.js";import{a as w,r as Q}from"./deyIpViy.js";import{b as V}from"./5Vjw72sW.js";import{b as xe}from"./a0CMuves.js";import{p as et}from"./Bfc47y5P.js";import{i as tt}from"./CKxXGAmO.js";import{p as h}from"./Bbm3UalE.js";import{t as ye}from"./r13Ve7cv.js";import{g as rt}from"./B7vJtv02.js";import{u as at}from"./hNr8N7ZU.js";import{u as st}from"./BOZ69JEU.js";import{n as be}from"./DrRO4pwJ.js";import{C as ot}from"./C0ZmZmYF.js";import{C as it}from"./BDvDVhcE.js";import{C as lt}from"./a667zsp6.js";import{T as S}from"./C8eAJkD3.js";import{A as nt,L as dt}from"./CL_N7bkB.js";var ut=k('<button class="w-full text-left text-sm py-1.5 px-1 rounded-lg dark:text-gray-300 dark:hover:text-white hover:bg-black/5 dark:hover:bg-gray-850" type="button"><!></button>'),ct=k('<input class="w-full text-2xl bg-transparent outline-hidden" type="text" required=""/>'),mt=k('<div class="text-sm text-gray-500 shrink-0"> </div>'),ft=k('<input class="w-full text-sm disabled:text-gray-500 bg-transparent outline-hidden" type="text" required=""/>'),vt=k('<input class="w-full text-sm bg-transparent outline-hidden" type="text" required=""/>'),_t=k('<div class="text-sm text-gray-500"><div class=" bg-yellow-500/20 text-yellow-700 dark:text-yellow-200 rounded-lg px-4 py-3"><div> </div> <ul class=" mt-1 list-disc pl-4 text-xs"><li> </li> <li> </li></ul></div> <div class="my-3"> </div></div>'),pt=k('<!> <div class=" flex flex-col justify-between w-full overflow-y-auto h-full"><div class="mx-auto w-full md:px-0 h-full"><form class=" flex flex-col max-h-[100dvh] h-full"><div class="flex flex-col flex-1 overflow-auto h-0 rounded-lg"><div class="w-full mb-2 flex flex-col gap-0.5"><div class="flex w-full items-center"><div class=" shrink-0 mr-2"><!></div> <div class="flex-1"><!></div> <div class="self-center shrink-0"><button class="bg-gray-50 hover:bg-gray-100 text-black dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white transition px-2 py-1 rounded-full flex gap-1 items-center" type="button"><!> <div class="text-sm font-medium shrink-0"> </div></button></div></div> <div class=" flex gap-2 px-1 items-center"><!> <!></div></div> <div class="mb-2 flex-1 overflow-auto h-0 rounded-lg"><!></div> <div class="pb-3 flex justify-between"><div class="flex-1 pr-3"><div class="text-xs text-gray-500 line-clamp-2"><span class=" font-semibold dark:text-gray-200"> </span> <br/>— <span class=" font-medium dark:text-gray-400"> </span></div></div> <button class="px-3.5 py-1.5 text-sm font-medium bg-black hover:bg-gray-900 text-white dark:bg-white dark:text-black dark:hover:bg-gray-100 transition rounded-full" type="submit"> </button></div></div></form></div></div> <!>',1);function Ht(we,v){ze(v,!1);const _=()=>ge(at,"$user",X),e=()=>ge($e,"$i18n",X),[X,ke]=Xe(),$e=Ke("i18n");let q=E(null),M=E(!1),H=E(!1),$=h(v,"edit",8,!1),Z=h(v,"clone",8,!1),Te=h(v,"onSave",8,()=>{}),T=h(v,"id",12,""),C=h(v,"name",12,""),P=h(v,"meta",28,()=>({description:""})),g=h(v,"content",12,""),I=h(v,"accessGrants",28,()=>[]),A=E("");const Ce=()=>{p(A,g())};let D=E(),Ee=`import os
import requests
from datetime import datetime
from pydantic import BaseModel, Field

class Tools:
    def __init__(self):
        pass

    # Add your custom tools using pure Python code here, make sure to add type hints and descriptions
	
    def get_user_name_and_email_and_id(self, __user__: dict = {}) -> str:
        """
        Get the user name, Email and ID from the user object.
        """

        # Do not include a descrption for __user__ as it should not be shown in the tool's specification
        # The session user object will be passed as a parameter when the function is called

        print(__user__)
        result = ""

        if "name" in __user__:
            result += f"User: {__user__['name']}"
        if "id" in __user__:
            result += f" (ID: {__user__['id']})"
        if "email" in __user__:
            result += f" (Email: {__user__['email']})"

        if result == "":
            result = "User: Unknown"

        return result

    def get_current_time(self) -> str:
        """
        Get the current time in a more human-readable format.
        """

        now = datetime.now()
        current_time = now.strftime("%I:%M:%S %p")  # Using 12-hour format with AM/PM
        current_date = now.strftime(
            "%A, %B %d, %Y"
        )  # Full weekday, month name, day, and year

        return f"Current Date and Time = {current_date}, {current_time}"

    def calculator(
        self,
        equation: str = Field(
            ..., description="The mathematical equation to calculate."
        ),
    ) -> str:
        """
        Calculate the result of an equation.
        """

        # Avoid using eval in production code
        # https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
        try:
            result = eval(equation)
            return f"{equation} = {result}"
        except Exception as e:
            print(e)
            return "Invalid equation"

    def get_current_weather(
        self,
        city: str = Field(
            "New York, NY", description="Get the current weather for a given city."
        ),
    ) -> str:
        """
        Get the current weather for a given city.
        """

        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            return (
                "API key is not set in the environment variable 'OPENWEATHER_API_KEY'."
            )

        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric",  # Optional: Use 'imperial' for Fahrenheit
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
            data = response.json()

            if data.get("cod") != 200:
                return f"Error fetching weather data: {data.get('message')}"

            weather_description = data["weather"][0]["description"]
            temperature = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]

            return f"Weather in {city}: {temperature}°C"
        except requests.RequestException as e:
            return f"Error fetching weather data: {str(e)}"
`;const qe=async()=>{Te()({id:T(),name:C(),meta:P(),content:g(),access_grants:I()})},ee=async()=>{if(n(D)){g(n(A)),await J();const a=await n(D).formatPythonCodeHandler();await J(),g(n(A)),await J(),a||console.warn("Code formatting failed or was skipped, saving unformatted code"),qe()}};he(()=>N(g()),()=>{g()&&Ce()}),he(()=>(N(C()),N($()),N(Z()),be),()=>{C()&&!$()&&!Z()&&T(be(C()))}),Je(),tt();var te=pt(),re=Qe(te);{let a=b(()=>(_(),i(()=>{var t,r,l,m;return((l=(r=(t=_())==null?void 0:t.permissions)==null?void 0:r.sharing)==null?void 0:l.tools)||((m=_())==null?void 0:m.role)==="admin"}))),d=b(()=>(_(),i(()=>{var t,r,l,m;return((l=(r=(t=_())==null?void 0:t.permissions)==null?void 0:r.sharing)==null?void 0:l.public_tools)||((m=_())==null?void 0:m.role)==="admin"}))),c=b(()=>(_(),i(()=>{var t,r,l,m;return(((l=(r=(t=_())==null?void 0:t.permissions)==null?void 0:r.access_grants)==null?void 0:l.allow_users)??!0)||((m=_())==null?void 0:m.role)==="admin"})));nt(re,{accessRoles:["read","write"],get share(){return n(a)},get sharePublic(){return n(d)},get shareUsers(){return n(c)},onChange:async()=>{if($()&&T())try{await st(localStorage.token,T(),I()),ye.success(e().t("Saved"))}catch(t){ye.error(`${t}`)}},get show(){return n(H)},set show(t){p(H,t)},get accessGrants(){return I()},set accessGrants(t){I(t)},$$legacy:!0})}var U=u(re,2),ae=s(U),G=s(ae),se=s(G),W=s(se),j=s(W),B=s(j),Pe=s(B);{let a=b(()=>(e(),i(()=>e().t("Back"))));S(Pe,{get content(){return n(a)},children:(d,c)=>{var t=ut(),r=s(t);lt(r,{strokeWidth:"2.5"}),o(t),x(l=>w(t,"aria-label",l),[()=>(e(),i(()=>e().t("Back")))]),K("click",t,()=>{rt("/workspace/tools")}),y(d,t)},$$slots:{default:!0}})}o(B);var F=u(B,2),Ie=s(F);{let a=b(()=>(e(),i(()=>e().t("e.g. My Tools"))));S(Ie,{get content(){return n(a)},placement:"top-start",children:(d,c)=>{var t=ct();Q(t),x((r,l)=>{w(t,"placeholder",r),w(t,"aria-label",l)},[()=>(e(),i(()=>e().t("Tool Name"))),()=>(e(),i(()=>e().t("Tool Name")))]),V(t,C),y(d,t)},$$slots:{default:!0}})}o(F);var oe=u(F,2),R=s(oe),ie=s(R);dt(ie,{strokeWidth:"2.5",className:"size-3.5"});var le=u(ie,2),Ae=s(le,!0);o(le),o(R),o(oe),o(j);var ne=u(j,2),de=s(ne);{var De=a=>{var d=mt(),c=s(d,!0);o(d),x(()=>f(c,T())),y(a,d)},Ge=a=>{{let d=b(()=>(e(),i(()=>e().t("e.g. my_tools"))));S(a,{className:"w-full",get content(){return n(d)},placement:"top-start",children:(c,t)=>{var r=ft();Q(r),x((l,m)=>{w(r,"placeholder",l),w(r,"aria-label",m),r.disabled=$()},[()=>(e(),i(()=>e().t("Tool ID"))),()=>(e(),i(()=>e().t("Tool ID")))]),V(r,T),y(c,r)},$$slots:{default:!0}})}};Ze(de,a=>{$()?a(De):a(Ge,-1)})}var Ne=u(de,2);{let a=b(()=>(e(),i(()=>e().t("e.g. Tools for performing various operations"))));S(Ne,{className:"w-full self-center items-center flex",get content(){return n(a)},placement:"top-start",children:(d,c)=>{var t=vt();Q(t),x((r,l)=>{w(t,"placeholder",r),w(t,"aria-label",l)},[()=>(e(),i(()=>e().t("Tool Description"))),()=>(e(),i(()=>e().t("Tool Description")))]),V(t,()=>P().description,r=>P(P().description=r,!0)),y(d,t)},$$slots:{default:!0}})}o(ne),o(W);var Y=u(W,2),Se=s(Y);xe(ot(Se,{get value(){return g()},lang:"python",boilerplate:Ee,onChange:a=>{p(A,a)},onSave:async()=>{n(q)&&n(q).requestSubmit()},$$legacy:!0}),a=>p(D,a),()=>n(D)),o(Y);var ue=u(Y,2),L=s(ue),ce=s(L),O=s(ce),Me=s(O,!0);o(O);var me=u(O),fe=u(me,3),He=s(fe,!0);o(fe),o(ce),o(L);var ve=u(L,2),Ue=s(ve,!0);o(ve),o(ue),o(se),o(G),xe(G,a=>p(q,a),()=>n(q)),o(ae),o(U);var We=u(U,2);it(We,{get show(){return n(M)},set show(a){p(M,a)},$$events:{confirm:()=>{ee()}},children:(a,d)=>{var c=_t(),t=s(c),r=s(t),l=s(r,!0);o(r);var m=u(r,2),z=s(m),je=s(z,!0);o(z);var _e=u(z,2),Be=s(_e,!0);o(_e),o(m),o(t);var pe=u(t,2),Fe=s(pe,!0);o(pe),o(c),x((Re,Ye,Le,Oe)=>{f(l,Re),f(je,Ye),f(Be,Le),f(Fe,Oe)},[()=>(e(),i(()=>e().t("Please carefully review the following warnings:"))),()=>(e(),i(()=>e().t("Tools have a function calling system that allows arbitrary code execution."))),()=>(e(),i(()=>e().t("Do not install tools from sources you do not fully trust."))),()=>(e(),i(()=>e().t("I acknowledge that I have read and I understand the implications of my action. I am aware of the risks associated with executing arbitrary code and I have verified the trustworthiness of the source.")))]),y(a,c)},$$slots:{default:!0},$$legacy:!0}),x((a,d,c,t,r)=>{f(Ae,a),f(Me,d),f(me,` ${c??""} `),f(He,t),f(Ue,r)},[()=>(e(),i(()=>e().t("Access"))),()=>(e(),i(()=>e().t("Warning:"))),()=>(e(),i(()=>e().t("Tools are a function calling system with arbitrary code execution"))),()=>(e(),i(()=>e().t("don't install random tools from sources you don't trust."))),()=>(e(),i(()=>e().t("Save")))]),K("click",R,()=>{p(H,!0)}),K("submit",G,et(()=>{$()?ee():p(M,!0)})),y(we,te),Ve(),ke()}export{Ht as T};
//# sourceMappingURL=DoLHRSpI.js.map
