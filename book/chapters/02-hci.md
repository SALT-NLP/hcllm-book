<!--
  Copyright (c) 2025-2026 Nathan Lambert.
  Licensed under CC BY-NC-SA 4.0:
  https://creativecommons.org/licenses/by-nc-sa/4.0/
  Full license: https://github.com/natolambert/rlhf-book/blob/main/LICENSE-CHAPTERS
-->
---
prev-chapter: "Home"
prev-url: "https://rlhfbook.com/"
page-title: HCI for HCLLMs
search-title: "Chapter 2: HCI for HCLLMs"
next-chapter: "Chapter 3: Data"
next-url: "https://rlhfbook.com/"
---
::: {#hci}
# HCI for HCLLMs
:::

[]{#sec:hci label="sec:hci"} How do we center humans in the design of
LLMs? To start, we can turn to the field of human-computer interaction
(HCI), which offers the foundational principles for realizing the vision
of HCLLMs. In particular, HCI provides the established theories,
methods, and frameworks for understanding and designing the critical
interface between the human user and the complex system (see
Figure [1](#fig:hci){reference-type="ref" reference="fig:hci"}). This
field has long grappled with how to make technology not just functional,
but also usable, understandable, and aligned with human values and
needs.

In the first subsection of this chapter, we start by tracing how
principles of ***human-centered design*** apply to HCLLMs and
understanding *who* are the stakeholders for designing HCLLMs
(§[1.1](#subsec:hci_who){reference-type="ref"
reference="subsec:hci_who"}). We then discuss ***design principles and
challenges*** for creating HCLLMs in
§[1.2](#subsec:hci_challenges){reference-type="ref"
reference="subsec:hci_challenges"}. These challenges range in scope from
the individual level (i.e., how we can improve end-user interactions
with these models) to the societal level (i.e., how to account for the
diverse cultures and contexts in which these models will be deployed).
We then discuss how HCI ***methodologies*** can be used in conjunction
with techniques commonly used in NLP to build and evaluate HCLLMs. We
conclude in §[1.3](#subsec:hci_methods){reference-type="ref"
reference="subsec:hci_methods"} with an overview of three methodological
orientations --- experimental methods, participatory approaches, and
qualitative inquiry --- from HCI and discuss how they can be adopted for
HCLLMs.

![We can draw on the field of human-computer interaction (HCI) to help
inform human-centered LLM design. The first in this process is
understanding *who* the relevant stakeholders are --- both direct and
indirect --- in both the development and deployment of
HCLLMs [1.1](#subsec:hci_who){reference-type="ref"
reference="subsec:hci_who"}. Second, we identify a set of unique
interaction challenges when it comes to designing
HCLLMs [1.2](#subsec:hci_challenges){reference-type="ref"
reference="subsec:hci_challenges"}. Finally, we discuss how HCI methods
can be used for providing new design
perspectives [1.3](#subsec:hci_methods){reference-type="ref"
reference="subsec:hci_methods"}. We synthesize these points in a case
study on designing LLMs for motivating physical activity in
Sec. [1.4.1](#subsec:hci_case){reference-type="ref"
reference="subsec:hci_case"}.](../assets/02_HCI.png){#fig:hci
width="\\linewidth"}

## Understanding the *Who*: the Humans in HCLLMs {#subsec:hci_who}

Human-centered design (HCD) is an approach that centers the needs and
experiences of users in all steps of the design process. There are four
guiding principles within HCD [@hcd2025]:

1.  Centering real people, their needs, and experiences in the design
    process

2.  Solving the core problems

3.  Thinking of everything as existing in an interconnected system

4.  Engaging in iterative prototyping to find solutions

When applying these principles of HCD to LLMs, it reframes the
development away from solely optimizing technical performance toward
understanding how these systems impact and fit into people's real lives.
Rather than treating metrics such as accuracy or perplexity as the
ultimate goal, HCD foregrounds the needs, values, and contexts of the
humans who develop, use, and are affected by these models. Thus, to
start, we must first understand *who* the real people affected by LLMs
are and *what* are the problems they face. In our consideration, we
account for people involved across the entire lifecycle of LLM
development, starting from the data used to train these models or the
objectives and principles that model developers prioritize to the way
that end users interact with models. In this subsection, we provide an
overview of important stakeholder groups and their corresponding roles
in the development of HCLLMs: data workers, model developers, end users,
and indirect stakeholders.

#### Humans as Data Workers.

**Data workers** are group of stakeholders that are essential to the
HCLLM ecosystem but also often overlooked. Data workers form the often
invisible human labor that undergirds the datasets and methods that make
LLM development possible. Within this category of data workers, we also
have classes of individuals. For example, there are those who directly
label datasets or rank model outputs as **data annotators**. Their
judgments instantiate the "human preferences" that guide alignment,
embedding positionality into what the model learns to
value [@ouyang2022training; @kirk2024the]. While annotators are the ones
directly labeling the data, selecting which values or ideologies are
prioritized often comes from other parties who hold power or authority
over data workers (e.g., project manager, client requesting annotated
data) [@miceli2020between; @wang2022whose]. In addition to data
annotators, there are also **safety and moderation workers**, who engage
in various methods to expose and rectify potential harms in the model
before deployment. Similar to content moderation in other domains, such
as social media, this work often entails sustained exposure to toxic or
disturbing material, raising serious questions about the ethics and
mental health implications of this invisible labor [@pendse2025testing].
Finally, there are **data subjects**, or the individuals whose internet
data forms the bulk of the web corpora used to pre-train LLMs.
Importantly, many of those who are data subjects are doing so
unwittingly as their data is included without consent or even
awareness [@paullada2021data; @birhane2021large]. Together, these data
workers form the backbone of HCLLM development, yet their contributions
are systematically undervalued and underprotected.

#### Humans as Model Developers.

Next, **model developers** of HCLLMs are the ones most immediately
tasked with shaping the model architecture, training pipeline, and
deliverable behaviors of HCLLMs. They make crucial decisions about what
data to collect, how to preprocess it, and which training objectives and
evaluation protocols to employ. Prior works have shown how the values of
model developers can be implicitly imbued in resulting technical
artifacts, such as in how quality filters are defined for
data [@gururangan-etal-2022-whose] and what research problems are
prioritized [@birhane2022values]. In designing HCLLMs, model developers
must harmonize the trade-offs between performance, fairness,
inclusivity, and safety. For example, they must ensure the fairness,
privacy, and safety of the training data of LLMs
(§[\[sec:data\]](#sec:data){reference-type="ref" reference="sec:data"}),
design comprehensive evaluation frameworks to capture real-world usage
scenarios (§[\[sec:evaluation\]](#sec:evaluation){reference-type="ref"
reference="sec:evaluation"}), and devise governance mechanisms and
transparency guidelines that help ensure responsible deployment
(§[\[sec:responsible\]](#sec:responsible){reference-type="ref"
reference="sec:responsible"}).

#### Humans as End Users.

**End users** --- the individuals and groups who directly interact with
LLM systems --- represent one of the largest and most diverse
stakeholder groups. The general-purpose nature of LLMs means that end
users span a vast range: students seeking homework help; professionals
drafting reports; people with mental health concerns looking for support
and companionship; creative writers brainstorming ideas; non-native
speakers translating text; and countless
others [@handa2025economic; @chatterji2025people; @handa2025education].
Each user brings their own distinct goals, expertise levels, cultural
backgrounds, and needs when interfacing with models. Thus, as LLMs shift
from nascent technologies into being increasingly ingrained in users'
lives, understanding their real-world impact becomes critical. Despite
the outsized impact that LLMs may have on their lives, most end users
have relatively little control or influence in the design and creation
of these technologies. There are methods that allow end-users to exert
some degree of agency over models, such as through fine-tuning, which
offers a more heavyweight
intervention [@tan2024democratizing; @wang2025end]. Research prototypes
and product features also offer ways to tailor user interactions with
LLMs through creating personalized memory stores or drawing on community
knowledge
banks [@zhao2025knoll; @memories; @ryan2025synthesizeme; @zhong2024memorybank].
While these personalization mechanisms offer some user agency, they also
raise fundamental questions about the broader relationship between end
users and LLMs.

Key questions related to end users include: How do these systems affect
productivity, creativity, learning, and decision-making skills? What are
the risks of over-reliance, deskilling, or perpetuating existing biases?
And how can we design HCLLMs that empower users rather than constrain or
harm them? Answering these questions requires moving beyond system
capabilities --- bridging existing methods in NLP with those from HCI
--- to examine how LLMs may reshape human capacities, agency, and
well-being in practice.

#### Humans as Indirect Stakeholders

Finally, even people who are not direct end users of LLMs, or **indirect
stakeholders** can still be meaningfully impacted by
them [@friedman1996value]. If we consider an example of an LLM used to
assist physicians with taking clinical notes, patients are also affected
by this system even if they are not directly interfacing with
it [@haberle2024impact; @korom2025ai]. Beyond patients, we can also
think of many other potential indirect stakeholders in this example,
such as families or caregivers, insurance providers, hospital
administrators, and so on. Enumerating all indirect stakeholders can
seem like an intractable problem; in part, this underscores that
human-centered LLMs are not just technical artifacts; they are
sociotechnical systems with many externalities to consider. There is no
prescriptive formula for deciding which indirect stakeholders to
prioritize; however, thinking about factors, such as who may not be
well-represented in making design decisions, who is most likely to be
harmed by such systems (both immediately and over longer time horizons),
or conversely who may stand to benefit in ways that are not explicitly
intended, can inform designers. More broadly, rather than treating
indirect stakeholders as an overwhelming checklist, HCLLM designers
should use this complexity as motivation to identify the most
consequential stakeholders early and involve them throughout the design
process, rather than only including them as an afterthought
(Sec. [1.3.2](#subsub:participatory){reference-type="ref"
reference="subsub:participatory"}).

## Defining the *What*: Principles and Challenges for Designing HCLLMs {#subsec:hci_challenges}

While involving humans throughout the LLM lifecycle helps us understand
who is affected by these systems, it also raises fundamental questions
about *how* to design for their needs. Designing human-centered LLMs
requires more than simply optimizing model capabilities. It demands
careful attention to how users interact with, make sense of, and form
relationships with these systems across diverse contexts.

Designing human-AI interaction is not a new challenge. Pre-dating LLMs,
prior work @yang2020humanai has delineated the unique challenges of
human-AI interaction, highlighting the (1) uncertainties about model
capabilities and (2) the complexity of AI outputs --- both of which are
problems that remain even as model capabilities have improved. In
tandem, many others have enumerated many best practices for designing
human-AI
interaction [@amershi2019guidelines; @yildirim2023investigating]. These
principles span the lifecycle of model development, including the
initial conceptualization (e.g., what values are imbued in the system,
how is privacy and fairness handled), the model development process
(e.g., what data is used, how is the model trained), the model
deployment (e.g., how are errors handled, how are users' preferences
accounted for), and the interface layer where humans interact with the
system (e.g., how are users' expectations calibrated, how transparent
are the model's outputs) [@wright2020comparative]. While these
principles provide a strong foundation, human-centered LLMs require
adapting and extending them to address the unique characteristics of
large language models, accounting for their open-ended generation
capabilities, evolving social and relational roles, and wide-scale
deployment.

In this section, we explore specific challenges that arise when
designing HCLLMs. While of course there are challenges beyond those
covered in this subsection, our goal is to illustrate the differing
levels of consideration we must attend to. These range from
considerations at the unit of the individual, such as scaffolding users'
interactions with models and model outputs, to societal level questions
when it comes to deploying models across diverse cultures and contexts.

### Bridging the Gulf of Envisioning.

Within HCI, foundational frameworks for design are @norman1988design's
gulfs of execution and evaluation. The gulf of execution pertains to
gaps related to users figuring out how to do certain actions that they
want, whereas the gulf of evaluation arises when the user is not able to
interpret the system's output. These gulfs are persistent across the
design of many technologies, ranging from everyday, physical objects
like doorhandles to cutting edge technologies. However, LLMs pose a new
challenge for users: the "**gulf of
envisioning**" [@subramonyam2024bridging]. This gulf refers to the
distance that emerges between what users may intend to do with an LLM
and the prompts that are ultimately produced.

Why does this gulf of envisioning occur? Although LLMs have proven
capable across a wide-range of tasks, they still require users to guide
how they are being used. The de facto form of user interaction with LLMs
is prompting, or providing textual instructions delineating the desired
interactions. While prompting seems an ostensibly simple task, users
consistently struggle with writing prompts, underspecifying the
instructions and foregoing the appropriate level of detail, ultimately
yielding unsatisfactory model outputs [@zamfirescu2023johnny].
Furthermore, users must contend with the indeterminacy of model outputs
as well as the "black-box" nature of how their inputs are transformed
into
outputs [@subramonyam2024bridging; @yang2020humanai; @agrawala2023unpredictable].

Thus, one core challenge for designing human-centered LLMs is overcoming
this gulf of envisioning. One approach to do so has been focused on
designing interfaces that can better scaffold users' interactions with
models. For example, recent works have introduced direct manipulation
interfaces as an alternative to prompt
writing [@masson2024directgpt; @wu2022ai; @arawjo2024chainforge], such
as providing a visual programming environment that helps users create
more complex prompt chains [@arawjo2024chainforge]. Others have
operationalized prompting pipelines as modular code components to
provide a more systematic process for optimizing and refining
inputs [@khattab2024dspy]. There are also approaches beyond intervening
at the prompt level that are intended to improve users' interactions
with LLMs. One line of work has explored how to improve the model's
understanding of the user, such as through building better user models
or improving context that LLMs have about the
user [@shaikh2025gum; @naous2025flipping; @lei2026humanllm]. These
approaches aim to reduce this gap between users' intentions and what
they specify when interacting with an LLM. The improving capabilities of
models are only beneficial to users in so far as they can harness them.
Efforts coming from both directions --- making it easier for users to
guide model outputs with less effort and improving models' understanding
of users --- are required to reduce this gulf of envisioning.

### Interpreting LLM Outputs.

So far, our discussion of human-LLM interaction has mainly focused on
how users communicate intent to models. We now turn our attention to how
users evaluate and make sense of model outputs. As-is, model outputs can
be verbose and unstructured, making it difficult for users to understand
and leaving them feeling overwhelmed [@jiang2023graphologue]. To address
this challenge, prior work in human-computer interaction has proposed
new interfaces to support user *sensemaking* --- or providing external
representations to encode data for task-specific purposes. For example,
works such as Sensecape [@suh2023sensecape] and
Graphologue [@jiang2023graphologue] provide interactive visualizations
that allow users to visually explore LLMs' outputs in a structured
format rather than having to parse large amounts of text.

Beyond helping users understand the content that LLMs output, we also
must interrogate how users subsequently interpret and act upon these
outputs. Questions surrounding user trust and reliance on AI systems are
long-standing issues that predate the rise of
LLMs [@papenmeier2022s; @vasconcelos2023explanations]. Nonetheless, LLMs
introduce new challenges to these established problems. The complex
nature of these models means that understanding why and how models
produced the outputs is inscrutable to experts and end users
alike [@ameisen2025circuit]. However, user trust and reliance are not
solely model-specific issues; the design of LLM applications also
introduces new challenges. For example, @swoopes2025impact demonstrate
how the chat-based design of most user-facing LLMs can hide the inherent
stochasticity of the models, making it difficult for users to calibrate
trust. Furthermore, anthropomorphic features of models can further
foster user trust, even when it may be unwarranted [@cohn2024believing].
To combat these issues, researchers have proposed design features that
can foster appropriate reliance, such as generating explanations,
expressing uncertainty, and adding sources to
claims [@zhou2023navigating; @kim2025fostering]. These solutions are not
perfect; for instance, significant technical challenges remain in
ensuring that cited sources are correct and relevant. The hypothesized
benefit of features like sourcing is their potential to engage users in
slower, more careful thinking, but how to design LLMs that effectively
empower users' critical thinking over their outputs remains a key open
question.

### Navigating Human-LLM Relationships.

Finally, we consider the evolving role of LLMs relative to users.
Traditionally, AI systems have functioned as assistants: tools that can
augment human capabilities in restricted ways and that require users to
delegate tasks. However, as model capabilities approach or exceed human
performance in certain domains, visions of models as equal collaborators
are becoming increasingly plausible. For instance,
@shao2024collaborative articulates a framework in which humans engage in
bidirectional collaboration with LLM-based agents across a diverse set
of tasks. Shifting roles from assistant to collaborator carries
significant implications for the design of LLMs. For example, while
assistants wait for explicit instructions to execute tasks, a model that
serves as a collaborator might proactively suggest alternative
approaches, challenge assumptions, or redirect problem-solving
strategies, demanding fundamentally different interface affordances.
These questions echo and build upon classic debates in HCI between
direct manipulation interfaces, which afford users high degrees of
control, and interface agents, which can act autonomously on users'
behalf [@shneiderman1997direct]. Determining how much control is
required is not prescriptive but rather will be modulated by contextual
factors such as the type of task or user expertise.

Furthermore, as LLMs adopt more expansive roles beyond serving as
functional tools in our lives, we must also consider the *affective*
dimension in human-LLM interaction. Models are increasingly
anthropomorphized, meaning that they are perceived as having human-like
characteristics --- a fact that is exacerbated by the linguistic
expressions in generated
outputs [@cheng_anthroscore_2024; @cheng-etal-2025-dehumanizing].
Already, users are interacting with these models not only as coworkers
or collaborators but also as friends, companions, and romantic
partners [@zhang2025rise; @pataranutaporn2025my]. LLMs are also
increasingly used in sensitive domains, such as providing emotional
support or being used for therapy purposes [@zaosanders2025how]. These
affective interactions are also not always intentional. For example,
@zhang2025rise observed that companionship-oriented interactions emerge
even when users are not primarily interacting with LLMs for emotionally
laden tasks. While human-LLM relationships can have positive outcomes
for users, including combating loneliness and reducing
distress [@de2025ai], there are also many documented adverse impacts,
such as fostering emotional
dependence [@laestadius2024too; @pentina2023exploring], encouraging
harmful behavior [@zhang2025dark; @dupre2024aichatbots], or, in the
extreme, triggering cases of intense mental health
crises [@morrin2025delusions]. Balancing the trade-offs between these
benefits that human-LLM relationships can have with these very real
harms is an open challenge that necessitates interdisciplinary
interventions from technologists, social scientists, ethicists, and
policymakers. As an example of a work in this area, @kirk2025human
called for the community to prioritize the socio-affective alignment of
LLMs, accounting for how models fit into and actively shape individual
users' social and psychological ecosystems. Nonetheless, how exactly we
design human-LLM relationships that promote user well-being in the
long-term or other pro-social outcomes remains an open area for
exploration.

### Designing for Diverse Cultures and Contexts.

Finally, we turn our attention to the critical challenge of designing
LLMs that are sensitive and adaptive to diverse cultural contexts. LLMs
are not culturally neutral artifacts. As discussed in more detail in
§[\[sec:data\]](#sec:data){reference-type="ref" reference="sec:data"},
they are trained predominantly on data from Western, Educated,
Industrialized, Rich, and Democratic (WEIRD)
societies [@mihalcea2025ai]. As such, models inherently encode and
propagate specific cultural values, communication styles, and social
norms [@naous2024havingbeerprayermeasuring; @durmus2023towards; @ryan2024unintendedimpactsllmalignment].
These misalignments can render models unhelpful, at best, and culturally
insensitive or actively harmful to users.

HCI offers critical theoretical lenses to help us not only question the
assumptions underlying LLM design but also offer more generative
opportunities for design. For example, @bardzell2010feminist's
foundational work on Feminist HCI argues that we ought to be attending
to marginalized user groups when designing rather than focusing only on
a presumed "default". In this vein, other critical theories on
postcolonial computing and literature on decolonial practices have urged
designers to decenter dominant Western perspectives and account for the
plurality of worldviews and epistemologies that
exist [@irani2010postcolonial; @alvarado2021decolonial]. These works
push designers to move beyond simply "de-biasing" models and instead
question the fundamental assumptions embedded within them: whose
knowledge is centered, whose values are prioritized, and whose ways of
being are marginalized? For example, users from different cultural
backgrounds may have different expectations from what they wanted out of
an ideal AI system and use cases, requiring that we are able to localize
models to these needs rather than assuming a single
default [@ge2024culture; @qadri2025case; @phutane2025disability].
Systematically prioritizing knowledge from certain groups or cultures
over others is not simply a design challenge for better human-LLM
interaction but can present quality-of-service differences with material
impacts on users [@wilson2025no; @dev2022measures]. How to design,
build, and evaluate LLMs that are genuinely context-aware and culturally
adaptive remains a significant and vital open question for the field.

## Expanding the *How*: Methods from HCI for HCLLMs {#subsec:hci_methods}

Finally, in addition to providing new perspectives on how we ought to
design HCLLMs, HCI also introduces a set of methods that researchers and
practitioners can employ in pursuit of these goals. Given HCI's
interdisciplinary roots, there are many "ways of knowing" or
methodological orientations that researchers employ within the field. In
this section, we will discuss the applicability of three classes of
research methods --- experimental methods, participatory research, and
grounded theory --- that are particularly applicable to HCLLMs and are
less utilized within NLP. For a comprehensive overview of other
methodological practices in HCI, we refer readers to @olson2014ways.

### Experimental Research

An important step in designing and evaluating HCLLMs requires measuring
their impact on human outcomes, rather than only characterizing
behavior. While traditional methods in NLP, such as benchmarking,
provide a fast and scalable way to evaluate models, these numbers are
often divorced from the context that LLMs are applied and do not capture
their impact in practice [@raji2021ai; @mcintosh2024inadequacies].
Unlike observational studies, which are useful in informing us whether
variables are related, experimental research helps reveal many types of
relationships (e.g., association, causality) between variables of
interest [@gergle2014experimental]. Experiments can vary along the
spectrum of research control, ranging from laboratory experiments to
field experiments that are conducted in real-world settings but often
have many factors that researchers cannot control
for [@oulasvirta2008field].

For HCLLMs, experimental methods enable researchers to move beyond
measuring what models can do to understanding what they *actually* do
and *why*. First, experimental methods also allow researchers to better
understand mechanisms that explain observed patterns of behavior. For
example, to better understand why users become dependent or overreliant
on LLMs, recent work [@cheng2025sycophantic] focused on the construct of
*social sycophancy*, finding through a series of laboratory experiments,
that users are more likely to trust and also use sycophantic models.
Isolating specific mechanisms can then inform design interventions that
are first validated intrinsically (i.e., benchmarks) and then
extrinsically with additional experiments. Second, experimental methods
can quantify the utility of LLMs when deployed in practice, allowing
researchers to evaluate outcomes rather than capability. A number of
field experiments have started, looking at LLMs' impacts across domains
including
education [@cuna2025hidden; @wang2025tutorcopilothumanaiapproach],
software engineering [@becker2025measuring], healthcare [@korom2025ai],
and so on. For example, @becker2025measuring's work provides evidence
that challenges the assumption that LLMs increase developer
productivity, demonstrating through a field experiment that using these
tools increases the time it takes for developers to complete their
tasks. These findings complicate our understanding of models and point
to the exogenous factors (e.g., institutional endorsement, workflow
integration, user onboarding) that affect outcomes. Overall, moving from
what a model can perform within a clearly scoped benchmark setting to
how it impacts users or society writ large --- and why --- is a critical
contribution from HCI's methodological toolkit that can complement
traditional NLP evaluations.

### Participatory Approaches {#subsub:participatory}

Building HCLLMs requires us to engage with the communities that are
impacted by and using these technologies. A useful approach for
understanding how we can work toward this engagement is to draw on
participatory research methodologies. More than being a prescriptive
methodology to follow, "participatory research" delineates an
orientation towards involving relevant stakeholders in the knowledge
creation process [@bergold2012participatory]. Examples of methods that
fall under this broader umbrella of "participatory research" include
participatory action research [@hayes2011relationship] and
community-based participatory
research [@unertl2016integrating; @wallerstein2017theoretical]. For a
more extensive list of participatory research frameworks, we refer the
reader to @vaughn2020participatory. Despite the many variants that fall
under participatory research, a unifying emphasis is on fostering a
democratic and inclusive process that treats stakeholders or community
members as equal research collaborators rather than
subjects [@vaughn2020participatory; @bergold2012participatory].

Participatory approaches have been adopted in creating machine learning
solutions, such as developing context-specific models for detecting
feminicide [@suresh2022towards] or providing frameworks for communities
to articulate algorithmic policies [@lee2019webuildai].
@tseng2025ownership provides an example of how participatory research
approaches can be applied to designing HCLLMs. Through interviews with
different stakeholders in the journalism ecosystem (e.g., reporters,
editors, executives), @tseng2025ownership articulate how LLMs must be
designed to address journalists' needs, such as using an open-sourced
model that can be fine-tuned for their tasks rather than relying on
commercial offerings.

While we have listed exemplars of work that have adopted participatory
approaches, it is important to state that participation is neither a
panacea to the many of the challenges with creating HCLLM nor is it easy
to institute these methods in practice. These methods require building
trust with communities, which can be a long-term process requiring
significant time investment [@le2015strangers]. To ensure that
participatory approaches are equitable in practice, it further requires
centering communities, especially those who do not hold positions of
power, and ensuring that research tangibly benefits these individuals
rather than acting as an extractive
force [@harrington2019deconstructing].

### Qualitative Inquiry

Qualitative methods encompass a broad range of approaches, including
ethnographic studies, interviews, participatory design workshops,
thematic analysis, and other interpretive
techniques [@blandford2016qualitative]. For HCLLMs, existing work has
also sought to extract insights by analyzing users' chat histories with
models, such as common interaction patterns or high-level values
reflected in model responses [@tamkin2024clio; @huang2025values]. While
quantitative measures reveal how well models perform or what users do,
qualitative approaches provide a deeper understanding of why and how
individuals think about, interact with, and make sense of these systems.
As @geertz2008thick illustrates through the notion of "thick
description," qualitative methods enable researchers to capture the
meanings, contexts, and social dynamics that underlie behavior, rather
than merely documenting surface-level patterns.

Qualitative methods offer several particularly valuable contributions to
HCLLMs. First, they can uncover needs and harms in underrepresented
populations, revealing how specific groups engage with LLMs and the
nuanced benefits and harms they experience---insights that aggregate
metrics often obscure (e.g., @ma2024evaluating's interviews with LGBTQ+
individuals exploring LLMs in mental health support; @qadri2025case's
workshops with South Asian participants examining cultural
misrepresentations in AI). Qualitative methods also help generate new
design insights and hypotheses by surfacing unexpected use patterns,
workarounds, and unmet needs that can inform future system design.
Through providing this rich description about how users interact with
these models, qualitative work enables contextual understanding of how
LLM interactions are embedded in broader social and work practices,
revealing dependencies and consequences that otherwise may be missed.
For example, @tamkin2024clio's analysis of user interactions with Claude
revealed areas of unsafe behavior that their current guardrails were not
designed to catch, allowing them to refine their system design through
empirical insights. Finally, qualitative approaches complement and
enrich quantitative findings by helping triangulate results and
providing interpretive depth that explains why certain patterns
emerge [@zhao2025sphere].

## From Human-Centered Design Challenges to Technical Artifacts {#subsec:technical_artifacts}

Addressing these design challenges for HCLLMs requires coordinated
interventions and across the entire LLM development pipeline --- from
the data curation and model training processes to the user interface
design. Already when introducing these challenges, we have discussed
interface-level interventions. For example, at the model level,
improving model capabilities via post-training, such as instruction
tuning and preference learning (as discussed in
§[\[sec:nlp\]](#sec:nlp){reference-type="ref" reference="sec:nlp"}), can
help models better interpret underspecified prompts, helping bridge the
gulf of envisioning. Decisions that developers make around what sources
to include in the training data will affect models' capabilities across
different cultures and context
(§[\[sec:data\]](#sec:data){reference-type="ref" reference="sec:data"}).
Beyond the model training pipeline, choices around the technical design
of the system, such as how to handle source attribution or what safety
guardrails are put in place
(§[\[sec:responsible\]](#sec:responsible){reference-type="ref"
reference="sec:responsible"}), will mold users' interactions, impacting
relationships that may form. Nonetheless, the critical point of this
chapter is that these technical interventions are most effective when
informed by and evaluated against the human-centered principles outlined
above. The challenges we have charted are fundamentally *sociotechnical*
problems; they cannot be solved by better models alone, nor by better
interfaces alone, but through the careful co-design of both.

### Case Study: Motivating Physical Activity with HCLLMs {#subsec:hci_case}

To make the human-centered LLM design process concrete, we present a
case study based on two related systems for LLM physical activity
coaching: (1) GPTCoach [@joerke2025gptcoach], an LLM-based coach that
implements motivational interviewing, and (2) Bloom [@joerke2026bloom],
a mobile application that integrates GPTCoach with established, UI-based
behavior change interactions. Physical inactivity is a major public
health concern, with large portions of the population falling short of
recommended guidelines for physical activity. LLMs present a promising
opportunity to combine the scalability of existing mobile health
interventions with the personalization of human coaching. Through this
case study of designing an LLM health coach, we illustrate how a
*human-centered* process can help realize this opportunity.

Let us start by considering the status quo approach that treats training
a good LLM health coach as an instruction-following problem. The
approach is to collect user data (e.g., common barriers to activity,
wearable data), feed it to a model, and have the model generate
personalized nudges, exercise plans, or advice. This framing is an
intuitive starting point, and it maps cleanly onto standard LLM training
pipelines. However, it also implicitly encodes a set of assumptions:
that users always want or need recommendations and advice, that more
information yields better outcomes, and that the primary bottleneck is
the model's ability to produce accurate health advice.

GPTCoach and Bloom instead took a human-centered approach, exemplifying
the following three concepts discussed in the chapter:

-   **Working with stakeholders to shape the system.** Rather than
    starting from the status quo, @joerke2025gptcoach conducted
    formative interviews with health experts and prospective end-users.
    These interviews revealed that health experts emphasized the
    importance of facilitative, non-prescriptive support that refrains
    from giving unsolicited advice---a mode of engagement that runs
    counter to how a standard LLM chatbot operates. Experts described
    their role as staying "in the passenger seat" and helping clients
    surface their own goals and barriers, rather than telling them what
    to do. This learning fundamentally reframed the problem to be
    solved, the system that was designed to address it, and the
    evaluation criteria. Moreover, engagement with experts did not end
    after the formative study. After the lab study, the authors hired
    trained experts to code all transcripts to measure adherence to
    motivational interviewing. Expert coding indicated that GPTCoach
    used conversational strategies that were consistent with
    motivational interviewing or neutral over 93% of the time, but
    qualitative feedback revealed important gaps compared to skilled
    human practitioners, highlighting specific areas for improvement
    that would not have surfaced without expert involvement.

-   **Focusing on the interaction.** A second theme from this case study
    is that model capability, while important, is not sufficient in and
    of itself for a successful human-centered system. While the authors
    could have devoted significant efforts to training, prompt chaining
    proved sufficient to enable LLM-based motivational interviewing.
    This created space for the authors to focus on how different aspects
    of the interaction design could shape users' experiences in
    substantive ways. For example, GPTCoach's non-prescriptive and
    non-judgmental communication style had a greater impact on
    participants' overall experience than its analysis of their health
    data. In Bloom, the authors represented the coaching agent as a
    *bee* avatar named Beebo. Beebo's capabilities are the same as those
    of a generic chatbot, but it's representation substantially changed
    the nature of the interaction. Many participants resonated with the
    avatar and described Beebo in relational terms, leading to increased
    engagement and adherence. Beebo's clear role as a "coach," not a
    general purpose assistant, helped set expectations when Beebo
    redirected conversations back towards physical activity, or when
    guardrails triggered refusals for medical advice. This dynamic
    points to the design challenge of navigating human-LLM relationships
    from Sec. [1.2](#subsec:hci_challenges){reference-type="ref"
    reference="subsec:hci_challenges"}. Taken together, these design
    choices reflect a move beyond thinking narrowly about model
    performance toward a more holistic understanding of how users will
    interact with these systems.

-   **Evaluating with users.** Finally, this case study showcases how
    the different methods from HCI offer new ways of knowing for
    evaluating HCLLMs. In a four-week randomized field study (N=54)
    comparing Bloom to a no-LLM control, the authors used a mixed
    methods evaluation, synthesizing insights across qualitative coding
    of participant interviews, survey data, app usage logs, and wearable
    data. The quantitative data revealed a 5x increase in overall app
    usage time in the LLM condition, while mean physical activity levels
    stayed comparable in both conditions. Meanwhile, survey measures
    revealed substantial shifts in physical activity mindsets and
    satisfaction. Qualitative coding added rich nuance to these
    findings, with participants in the LLM condition reporting stronger
    beliefs that activity was beneficial to their health, greater
    enjoyment of exercise, an expanded appreciation of "what counts" as
    activity, and increased self-compassion when goals were missed. Most
    importantly, participants attributed these mindset shifts to
    interactions with Beebo that kept them in control of their own
    behavior change, such as finding flexible alternatives when plans
    fell through. More broadly, this illustrates how qualitative "thick"
    understanding can surface the *why* behind user experiences, which
    are insights that a purely quantitative evaluation might miss.

Overall, this case study demonstrates how critically engaging with
humans *early* in the design process can reframe the problem being
solved and the evaluation target, leading to qualitatively different
solutions that better serve human needs. Notably, the most consequential
outcomes in both studies---positive changes in participants' beliefs
about physical activity and their own abilities---emerged from the
design process rather than from improvements in model capability.
