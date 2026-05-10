---
prev-chapter: "Responsible HCLLMs"
prev-url: "https://rlhfbook.com/"
page-title: HCLLMs and the Future of Work
search-title: "Chapter 7:  HCLLMs and the Future of Work"
next-chapter: "Conclusion"
next-url: "https://rlhfbook.com/"
---
# Case Study: HCLLMs and the Future of Work


[]{#sec:applications label="sec:applications"}

LLMs are not only research technologies that a select group of people
examine, use, and probe. They have become a part of the lives of
everyday users, supporting tasks ranging from assisting with coding to
giving relationship
advice [@tamkin2024clio; @chatterji2025people; @yang2024swe; @jimenezswe].
For example, as of September 2025, OpenAI reported that their flagship
chatbot, ChatGPT, saw over 700 million weekly active
users [@chatterji2025people]. So, we conclude by discussing how our
considerations around the multiple facets of HCLLMs apply in real-world
applications. For example, are HCLLMs practically feasible? How do
HCLLMs affect individuals and what macro effects do HCLLMs have on
society?

We focus this chapter on one particular application area: the future of
work. Of course, this is not the only area where LLMs are used; these
models have a wide scope of applications including healthcare,
education, political science, and so
on [@thirunavukarasu2023large; @adiguzel2023revolutionizing; @ornstein2023train].
Nonetheless, we focus on HCLLMs' impact on labor and the future of work
given public interest, as evidenced by the many headlines and
speculation, as well as the individual and societal level impacts that
models will have in this area [@shao2025future; @acemoglu2026building].

The adoption of LLMs in the labor market has led to significant shifts
in human productivity, in-demand skills, and even possible macroeconomic
changes [@chen2025code; @eloundou2024gpts]. We will now show how model
developers can incorporate the human-centered principles from previous
sections to define, develop, and deploy HCLLMs within this evolving
ecosystem. To define HCLLMs here, we will first discuss *who* the
stakeholders are and how to account for these differing parties in
§[1.1](#subsub:stakeholders_future_of_work){reference-type="ref"
reference="subsub:stakeholders_future_of_work"}. Then, we will cover
HCLLM development, how we ought to be training and evaluating models for
future of work purposes in
§[1.2](#subsub:evaluating_future_of_work){reference-type="ref"
reference="subsub:evaluating_future_of_work"}. Finally, in
§[1.3](#subsub:case_deploy){reference-type="ref"
reference="subsub:case_deploy"}, we conclude with considerations for
responsibly deploying HCLLMs, such as the potential for widening
inequalities or overreliance. The road map is visualized in
Figure [1](#fig:future_of_work){reference-type="ref"
reference="fig:future_of_work"}.

![We present a case study on HCLLMs and the future of work, covering the
three key areas of defining, developing, and deploying HCLLMs. We start
by identifying relevant stakeholders
([1.1](#subsub:stakeholders_future_of_work){reference-type="ref"
reference="subsub:stakeholders_future_of_work"}), then move to examining
the model capabilities needed to better suit LLMs for workplace settings
([1.2](#subsub:evaluating_future_of_work){reference-type="ref"
reference="subsub:evaluating_future_of_work"}), and conclude by
discussing key societal considerations
([1.3](#subsub:case_deploy){reference-type="ref"
reference="subsub:case_deploy"}).](../assets/CaseStudy.png){#fig:future_of_work
width="\\linewidth"}

## Defining the Stakeholders {#subsub:stakeholders_future_of_work}

To understand how we can design LLMs for the future of work in a
human-centered fashion, we must start by understanding the *who*. Who is
being impacted by HCLLMs in the workforce? Is this the same group of
people that these models are being designed for? How might different
groups of stakeholders --- both direct and indirect --- be impacted
differently? There are many potential stakeholders, including
**workers** who directly interact with LLMs; **employers** who may be in
charge of procuring the technology for their organization;
**shareholders** who are interested in productivity or financial gains;
and **customers** who may see the final artifact or output created from
workers using LLMs.

A natural group to start with are the **workers** who interface and
utilize these technologies as part of their work. Here, a recurring
theme is a fundamental mismatch between what people actually want from
LLMs and how those technologies are currently being designed. In part,
this misalignment can be attributed to organizational constraints, which
are more present in the work setting compared to personal use. For
instance, @nanda2025state found that employees frequently resort to
using personal accounts to access LLMs (e.g., ChatGPT, Claude) as they
found enterprise deployments fail to meet their needs or feel too
restrictive. Yet, this workaround behavior is symptomatic of a deeper
issue. The way in which LLMs are currently being used in the workplace
is misaligned with worker priorities. For example, @shao2025future
conducted a survey with 1,500 U.S. workers to understand what tasks they
want AI agents to be used for. Critically, they found that the tasks
workers *want* to use agents for differ substantially from how tools are
currently deployed and where industry funding is going. In tandem with
the human-centered reasons for centering workers' perspectives, this
finding suggests that current AI development trajectories risk
prioritizing displacement over augmentation. Doing so risks repeating
dangerous historical patterns in which automation technologies lead to
displacement without commensurate productivity gains or wage
increases [@acemoglu2021tasks]. To close this gap, we must ensure that
the needs and desires of workers are incorporated into the design of
these systems from the outset.

Taking into account the perspectives of **employers** or
**shareholders**, the question becomes whether introducing LLMs improves
the productivity and quality of work produced. What complicates this
question is the "jagged" nature by which LLMs are
useful [@dell2023navigating]. For some tasks, LLMs are particularly
performant and can automate existing process; some roles will see more
of a synmbiotic relationship where there is augmentation rather than
replacement; and in others, LLMs are in fact not capable at performing
requisite tasks at all [@mazeika2025remote]. There is a parallel
consideration around literacy --- whether workers are well-equipped to
use the technology. Harkening back to the "gulf of envisioning"
discussed in Chapter [\[sec:hci\]](#sec:hci){reference-type="ref"
reference="sec:hci"}, it is possible that LLMs may actually be useful
for workers, but workers may not know how to best specify their intent
or be unaware that such capabilities exist. This places a responsibility
on employers, organizations, and policymakers to invest in AI literacy
programs that equip workers with the conceptual and practical knowledge
needed to participate in an AI-augmented workplace, rather than simply
assuming adoption will follow deployment [@ma2025not; @nanda2025state].

Finally, another stakeholder we will highlight are **customers** that
serve as another end-user in this setting. For instance, as LLM-based
systems increasingly replace human touchpoints in domains such as
customer service or healthcare, the impact on customers warrants equal
consideration. The evidence here is mixed: while interacting with LLM
systems offers speed and accuracy compared to human support, these
interactions can also raise frustrations around the lack of empathy or
potential for miscommunication [@huang2024can; @li2025artificial].
Furthermore, recent survey data shows that customers in the U.S. still
prefer talking with a human representative over an LLM
system [@customer2026]. This concern is heightened in contexts that may
be more high-stakes or emotionally-laden. Nonetheless, we also want to
highlight that treating this question of AI usages as a binary choice
between human-only and AI-only presents an inchoate view of the issue as
there is meaningful middle ground that leverages the strengths of both.
For instance, existing work has explored how we can use LLMs to upskill
professionals to offer better human support, preserving this human
"touch" in interaction while enhancing the quality of accessibility.

## Developing HCLLMs for the Future of Work {#subsub:evaluating_future_of_work}

Next, we discuss how we can apply a human-centered lens to training and
evaluating LLMs to enable a future of work that jointly benefits the
different stakeholders we have discussed. To start, much of the current
discourse around LLM systems emphasizes autonomous execution, or systems
that complete complex, multi-step tasks with minimal human
involvement [@shen2025completion]. Yet, focusing primarily on this form
of autonomous interaction, overlooks the potential gains for systems in
which humans and LLMs collaborate. For example, researchers such as
@wang2026position have raised exactly this concern in the context of AI
coding agents --- a domain where models are quite performant and has
seen significant adoption especially amongst professionals in this
field [@appel2025anthropic]. They argue that the field has moved too
quickly toward full automation and that human involvement deserves to be
treated as a first-class design consideration rather than a transitional
inconvenience. Taking this critique seriously requires asking what it
would actually take, technically, to build human-centered LLMs for the
workforce.

#### Designing Models as Collaborators.

One prerequisite is that models need to know *how* to collaborate.
However, collaboration is not simply a matter of adding a confirmation
step before a model takes action. It requires a more nuanced
understanding of when to act autonomously, when to defer, and how to
communicate uncertainty or request input in ways that feel natural
rather than disruptive. As @shen2025completion show, building more
capable models does not necessarily equate to better collaboration, thus
necessitating dedicated efforts. In fact, current training paradigms,
such as preference tuning over single conversational turns, can even
hinder collaboration abilities [@wu2025collabllm]. Furthermore,
successful collaboration does not mean indiscriminately putting a human
in the loop. While ostensibly beneficial, this interaction paradigm can
lead to unnecessary verification and inefficiencies that ultimately slow
down the collaboration process. The best form of collaboration will
necessarily vary, depending on the task at hand as well as the skillset
of the human and LLM involved. For example, sometimes the human may need
to audit the LLM's work; other times the right form of collaboration may
need to be a back-and-forth conversation; and in others end-to-end
execution by the model may suffice.

To start, we must return to the core question: what are the ingredients
for a successful collaboration? In human organizations, a common
collaboration paradigm is *delegation*. At the moment, agents are
capable of decomposing complex tasks into more manageable units of
action. However, the next step here is being able to assign these
actions. Thus, we need to know what tasks to hand-off to these models
and which may be better suited for humans to complete or even to
delegate across multiple different models [@wang2025ai]. This delegation
capability must furthermore be adaptive across users, tasks, context,
and model
capabilities [@tomavsev2026intelligent; @guggenberger2023task; @fugener2022cognitive].
Beyond being able to delegate, there are many other proprties that help
make a good collaborator which are lacking in existing models, such as
proactivity [@lu2024proactive], transparency [@liao2023ai], and social
norm
adherence [@shaikh2025gum; @ziems2023normbank; @forbes2020social]--- all
presenting fruitful areas for future inquiry.

#### Improving our Understanding of Workers.

A second technical requirement is a richer understanding of the users
these systems are meant to serve. One push in this direction is
developing better user simulators which can capture the behavioral
patterns and habits of
individuals [@lei2026humanllm; @paglieri2026persona]. As our ability to
simulate users improves, a question we ought to return to is *which*
users are being simulated. At the moment, many efforts are concentrated
on generic user behavior or focused more so on knowledge workers (e.g.,
developers) [@naous2025flipping; @wang2026position]. This provides a
deep but narrow viewpoint of the workforce. The population of workers
whose jobs will intersect with LLMs is far broader, spanning domains
like healthcare, retail, education, logistics, and skilled
trades [@handa2025economic; @tomlinson2025working]. Developing better
user simulators for these contexts requires going beyond theoretical
modeling to gain a deeper understanding of how people are actually using
these systems in the real world. This is a genuine challenge, because
the most valuable behavioral data tends to be held by the companies
deploying models and is rarely made available to the broader research
community. Finding ways to bridge this gap --- whether through
privacy-preserving data sharing, partnerships, or the careful design of
in-the-wild studies --- is a prerequisite for building systems that
serve a broader swath of workers.

#### Evaluating for Ecologically Valid Tasks.

Finally, across chapters we have emphasized the importance of data in
the development and progress towards human-centered LLMs. To see
progress in this area, we must also have benchmarks that actually
reflect the complexity of real work. Many of the standard benchmarks
used to evaluate LLM performance consist of synthetic or highly
constrained tasks that bear little resemblance to the messy,
context-dependent, and often ambiguous nature of professional work. More
general-purpose benchmarks, such as GDPval [@patwardhan2025gdpval],
represent a meaningful step toward measuring model performance on more
ecologically valid tasks, and domain-specific benchmarks provides an
even more precise look as to how these models perform in settings such
as finance [@fan2025ai],
law [@guha2024legalbench; @li2025legalagentbench], and
medicine [@arora2025healthbench]. In tandem, we must also consider what
these benchmarks are failing to capture but that we may also want to
evaluate. For instance, how can we better account for how workers are
actually interacting with the systems or how the performance holds up in
the messy, open-ended conditions of real work?

## Responsibly Deploying HCLLMs in the Workforce {#subsub:case_deploy}

Perhaps one of the top-of-mind questions when it comes to the future of
work are the long-term societal impacts that these technologies will
have. How will employment be affected by the increasing popularity of
LLMs? What skills will be in-demand and which will be less important?
Deploying HCLLMs responsibly means taking into account these long-term
externalities now, and mitigating against potential harms before they
occur.

#### The Paradox of Productivity and Work Intensification.

A common perception is that integrating LLMs into workflows will
inherently reduce the volume of human effort. Initial stand-alone
studies looking at whether using LLMs improves how quickly workers
complete tasks show positive evidence: people complete tasks more
quickly with AI
assistance [@cui2025effects; @peng2023impact; @shen2026ai; @karny2024learning].
However, when it comes to the workforce, people are not completing
singular tasks in a vacuum. They must negotiate tasks with many demands,
coordinate with coworkers, navigate company politics, and so on. Thus,
when we look at the impact on productivity in a more holistic setting,
the impact of LLMs is less clear. Counter to the earlier studies, recent
work has suggested that rather than freeing up workers, LLMs may
actually amplify the intensity of
labor [@hbr_ai_work_intensification; @harvey2025don]. For example, a
recent study conducted by the Harvard Business Review observed 200
employees at a U.S.-based technology company, finding that LLM usage
promotes the phenomenon of "work slippage" in which AI leads to task
expansion and increased cognitive load rather than a net reduction in
hours [@hbr_ai_work_intensification]. This phenomenon is not solely
confined to the technology industry where LLM usage is more
prevalent [@harvey2025don; @johnson2025ai; @acemoglu2020wrong]. For
example, despite the purported benefits that LLMs have for educators,
using models often lead to more teacher time devoted towards validating
generated outputs beyond current time spent [@harvey2025don]. Thus,
instead of saving time, these tools may in fact create new tasks for
workers or potentially introduce friction into existing tasks.

To address these risks, we provide two approaches. First, we argue that
the way "productivity" is measured in the workplace in light of
advancements with LLMs must be updated. Traditionally, productivity has
centered on output per unit time, capturing how efficient people
work [@mckinsey2025]. However, now that LLMs can quickly produce
outputs, efficiency-based metrics may be defunct. Instead, echoing
practices in existing work [@shao2024collaborative; @wang2025ai], we
call for broader evaluations that consider not only whether tasks are
completed but also the *quality* of these outcomes. More broadly,
updating productivity metrics to match the current state of the world
provides a more precise picture of how LLMs are impacting work. Our
second approach is to find methods that center workers' voices. Survey
efforts such as @shao2025future provide insight into what workers want.
Complementing this approach, we argue for additional studies that can
capture the "thick" descriptions of what users are experiencing, such as
through interviews or ethnographic
approaches [@geertz2008thick; @anugraha2026sparkme]. Such methods can
shed light on what is not captured through numbers alone, such as hidden
forms of labor, additional cognitive load, or coordination costs.
Together, these directions point toward a more holistic framework for
evaluating AI in the workplace that accounts for both outcomes and
worker experience.

#### The Risk of Cognitive Deskilling.

A second concern is the potential for deskilling, where reliance on LLMs
erodes the foundational expertise of human workers. When models handle
the "first draft" (or sometimes the end-to-end execution) of complex
tasks, workers may lose the opportunity to engage in learning processes
necessary to develop deep domain mastery. Perhaps in an isolated setting
workers may be completing the task faster, but they are also less likely
to learn transferable skills or knowledge that can help them in future
tasks [@shen2026ai]. It is important to note that these effects are
heterogeneous across workers. How someone engages with these
technologies matters: automating tasks wholesale is likely to accelerate
deskilling, whereas more collaborative, iterative interactions where the
worker critically evaluates, corrects, and builds on model outputs may
partially preserve or even scaffold skill development [@shen2026ai].
Another moderator is expertise level. For instance, novices may rely on
these tools more heavily and more uncritically than experts, which is
especially concerning given that AI assistance can impart an illusion of
comprehension without providing sufficient expertise in an
area [@messeri2024artificial; @shen2026ai; @macnamara2024does].

These concerns highlight the importance of carefully designing how
humans collaborate with LLMs, as discussed in
§[1.2](#subsub:evaluating_future_of_work){reference-type="ref"
reference="subsub:evaluating_future_of_work"}. For example, the risk of
deskilling can be included as a factor when considering how to delegate
tasks. In this vein, rather than framing LLMs as tools that automate
tasks end-to-end, systems should be designed to empower users,
supporting human judgment, reflection, and skill development throughout
the workflow. To achieve this goal, this requires intentional design
interventions that encourage workers to remain actively engaged with the
task, such as mechanisms that promote critical evaluation of model
outputs or structures that preserve opportunities for learning and
expertise development [@ma2025towards; @reicherts2025ai].

#### Macroeconomic Divides and Digital Accessibility.

Finally, the integration of LLMs also threatens to exacerbate existing
economic disparities. Traditional automation has historically widened
the wage gap between high-skilled and low-skilled workers
[@acemoglu2021tasks], but LLMs introduce a specific "productivity
divide" rooted in accessibility. Research into usage patterns suggests
that ChatGPT adoption is positively correlated with higher education,
high socioeconomic status, and residency in urbanized zip codes
[@daepp2024emerging]. If access to state-of-the-art models remains
concentrated within privileged demographics, the resulting disparity in
AI-augmented productivity could further exacerbate systemic inequality.

Addressing this divide requires interventions at multiple levels. For
one, as open-source models grow more capable, this can help reduce
barriers to entry and broaden who is able to benefit from AI assistance.
However, access alone is insufficient. As discussed in
§[\[subsec:bias_eval\]](#subsec:bias_eval){reference-type="ref"
reference="subsec:bias_eval"}, model performance and usability can vary
across social groups, meaning that some populations may benefit less
from interacting with LLMs even when they are
available [@hofmann2024dialectprejudicepredictsai; @hassan2025dialectic; @durmus2023towards].
Continued efforts to identify and mitigate such disparities in model
behavior remain critical. Finally, improving AI literacy will be
essential as a new generation of workers grows up with these
technologies at their fingertips. Educational initiatives that teach
users how to critically evaluate, effectively collaborate with, and
responsibly leverage AI systems can help ensure that the productivity
gains from LLMs are distributed more
equitably [@solyst2025investigating; @morales2024youth; @morales2025learning; @okolo2024beyond; @cardon2023challenges].
