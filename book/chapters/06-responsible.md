---
prev-chapter: "Evaluations for HCLLMs"
prev-url: "https://rlhfbook.com/"
page-title: Evaluation
search-title: "Chapter 6: Responsible Human-Centered LLMs"
next-chapter: "Case Study"
next-url: "https://rlhfbook.com/"
---

# Responsible Human-Centered LLMs


[]{#sec:responsible label="sec:responsible"} In this chapter, we
highlight three broad properties that underpin a responsible deployment
of HCLLMs, and we explore the tensions and relationships between these
ideals (Figure [1](#fig:responsible){reference-type="ref"
reference="fig:responsible"}). The first property is ***interpretability
(§[\[subsec:interpretability\]](#subsec:interpretability){reference-type="ref"
reference="subsec:interpretability"})***: an HCLLM's input-output
transformations should be understood. This property is first, since it
complements our next two properties, ***steerability
(§[\[subsec:steerability\]](#subsec:steerability){reference-type="ref"
reference="subsec:steerability"})*** and ***safety
(§[1.3](#subsec:safety){reference-type="ref"
reference="subsec:safety"})***. Steerable models can be aligned along a
pre-selected dimension, and safe models do not produce undesirable
outputs. If we have an interpretable model, we may obtain a more
steerable model through feature-level control, and we may obtain a safer
model by isolating and removing harmful representations.

What makes responsible HCLLMs challenging is that these properties are
not only complementary, but also in tension. If we make a model more
steerable to individual user preferences, this may undermine safety
constraints, while overly rigid safety guardrails can limit a model's
ability to adapt to legitimate, diverse human needs. And although
interpretability supports steerability and safety in many ways, certain
alignment and steering methods can make models less interpretable. For
example, reward functions used in alignment introduce an additional
layer of complexity, and these functions are non-identifiable
[@joselowitz2025insights]. Multiple distinct reward functions can yield
similar policy behavior. As we discuss each of these three properties,
steerability, safety, and interpretability, we will cover current
approaches in the literature and then provide directions for future
research.

![We enumerate three properties for the responsible HCLLM deployment:
***interpretability
(§[\[subsec:interpretability\]](#subsec:interpretability){reference-type="ref"
reference="subsec:interpretability"})***, ***steerability
(§[\[subsec:steerability\]](#subsec:steerability){reference-type="ref"
reference="subsec:steerability"})***, and ***safety
(§[1.3](#subsec:safety){reference-type="ref"
reference="subsec:safety"})***. These properties are generally
complementary, but tensions between them can make deployment
difficult.](../assets/sec6.png){#fig:responsible width="\\linewidth"}

## Interpretable and Explainable HCLLMs

[]{#subsec:interpretability label="subsec:interpretability"}

The first dimension we emphasize is **interpretability and
explainability**. Neural networks, as the fundamental building blocks of
LLMs, remain largely opaque; the complex interactions between weights
and activations make both training dynamics and inference behavior
difficult to understand [@8631448]. Yet understanding these systems is
critical for ensuring the alignment of LLMs with human values and
objectives. We distinguish between two complementary goals:
interpretability, which focuses on understanding *how* LLMs operate in
general settings; and explainability, which seeks causal explanations
for *why* LLMs produce specific behaviors, decisions, or outcomes. Both
are essential for human-centered applications, but serve different
purposes. Interpretability provides a better understanding of LLM
internals, which can help address undesired behaviors such as
hallucinations, vulnerabilities to adversarial attacks, and encoded
biases. Explainability, by contrast, provides users with comprehensible
justification for individual outputs, informing appropriate trust and
enabling contestability.

### Current Approaches to Interpretability

#### Three interconnected areas of modern interpretability research.

First, work on understanding internal mechanisms has revealed that
transformer components can function as interpretable key-value memories
[@geva-etal-2021-transformer] and has begun to uncover how LLMs
represent multilingual knowledge
[@tang-etal-2024-language; @zhang2024differentstructuralsimilaritiesdifferences].
Second, these mechanistic insights have enabled practical interventions
on model behavior, such as inference-time steering
methods[@li2023inferencetime; @DBLP:journals/corr/abs-2308-10248; @zou2023representationengineeringtopdownapproach; @wu2024reft],
while model editing and machine unlearning techniques allow for targeted
removal of undesirable traits
[@meng2023massediting; @ilharco2023editing; @liu2024rethinkingmachineunlearninglarge].
Third, interpretability serves as a diagnostic tool for safety, helping
researchers understand jailbreaking vulnerabilities
[@arditi2024refusal; @kirch2024featurespromptsjailbreakllms] and
identify adversarial attack vectors
[@lucki2024adversarialperspectivemachineunlearning; @yu2024robustllmsafeguardingrefusal; @jain2024what].

#### Interpretability methods for human-centered purposes.

It is important to understand why certain model behaviors arise, such as
sycophancy or deception; however, this cannot be done simply by
examining model outputs in a post-hoc
fashion [@sharma2024towards; @hubinger2024sleeperagentstrainingdeceptive].
This shortcoming motivates the need to apply interpretability methods
for human-centered purposes. Building on theories such as the linear
representation hypothesis [@park2023the], the platonic representation
hypothesis [@huh2024position], and universal feature representations
across all LLMs [@lan2024sparse], interpretability has been used as a
tool to understand different model biases, including social
biases [@liu2024devil], cultural biases [@yu2025entangled], and cultural
knowledge [@veselovsky2025localized]. Additionally, recent work has
sought to identify models' internal representations of important model
behaviors, finding that models encode harmfulness and refusal separately
[@zhao2025llms] and that three dimensions of sycophancy --- sycophantic
agreement, sycophantic praise, and genuine praise --- are all encoded
along different linear directions in latent space and can be amplified
and suppressed without affecting the other [@vennemeyer2025sycophancy].

Another recent application of interpretability on HCLLM can help better
model human-AI interactions. In order for LLMs to act as helpful
assistants for users, they need to not only understand the user query
but develop an understanding of a user's latent traits and needs. A
misalignment between a model's representation of the user and a user's
true needs can lead to various harmful outcomes, ranging from
conversational grounding failures [@shaikh2023grounding] to sycophancy
and deception. For example, to make a model's user representation more
transparent, @chen2024designing designs a system to extract data related
to a user's demographic features and a dashboard that displays this
representation. @choi2025scalably similarly extracts latent
representations of users in LLMs, and these methods have also been
applied to predict the behaviors of personalized LLMs
[@karny2025neural].

### Current Approaches to Explainability

#### Modern explainability research for LLMs pursues several complementary goals.

Feature attribution methods identify which inputs most influence
outputs, natural language rationales provide human-readable
justifications, and counterfactual explanations show how minimal input
changes would alter predictions [@10.1145/3639372]. Unlike
interpretability, which seeks general understanding of internal
mechanisms, explainability focuses on justifying individual predictions
in terms that users and stakeholders can act upon. This goal has proven
challenging, as traditional explainable AI (XAI) techniques such as LIME
and SHAP [@ribeiro2016should] become computationally impractical at the
scale of billions of parameters, while LLM-specific approaches such as
chain-of-thought reasoning and post-hoc citation generation often
prioritize plausibility over faithfulness
[@lanham2023measuringfaithfulnesschainofthoughtreasoning; @turpin2023languagemodelsdontsay].
For a comprehensive taxonomy of explainability techniques for LLMs, we
refer readers to @10.1145/3639372.

#### How explainability methods can be used for human-centered purposes.

Explainability serves as a foundational element for building user trust
and enabling accountability in LLM systems. The ability to assign
responsibility for model decisions is essential not only for developing
transparent systems but also for supporting downstream regulatory
efforts -- for instance, AI in hiring systems, compensation for content
creators, and copyright law [@guha2024alignment]. These concerns have
motivated legislative action: for instance, the EU AI Act, which became
enforceable in 2024, establishes explainability as a legal requirement
in critical domains [@euaiact2024].

For end users, trust fundamentally depends on calibration (i.e., whether
models can reliably express what they know and don't know). Models often
struggle to convey uncertainty, both through log-probabilities and
linguistic hedging [@zhou2023navigating], although recent work has made
progress on both fronts
[@tian-etal-2023-just; @li2024fewshotrecalibrationlanguagemodels].
Closely related is the problem of citation and attribution. Effective
attribution can provide causal explanations for LLM behavior, but
current approaches have significant limitations. While RAG systems
supply LLMs with relevant context, there is no guarantee that models
actually use that context to generate responses
[@du-etal-2024-context; @li-etal-2023-large]. Post-hoc citation
generation similarly suffers from severe faithfulness issues
@liu2023evaluating, motivating work on parametric attribution
[@khalifa2024sourceawaretrainingenablesknowledge] and measuring training
data influence more broadly
[@pmlr-v202-park23c; @grosse2023studyinglargelanguagemodel; @guu2023simfluencemodelinginfluenceindividual].

Chain-of-thought (CoT) reasoning represents a particularly contested
approach to explainability. On one hand, CoT outputs provide an
accessible window into model reasoning that users can inspect without
technical expertise. On the other hand, research has shown that these
explanations can systematically misrepresent the true reasons for a
model's predictions
[@lanham2023measuringfaithfulnesschainofthoughtreasoning; @turpin2023languagemodelsdontsay].
This creates a paradox for human-centered design: CoT explanations may
increase user trust precisely because they appear plausible, even if
they fail to faithfully reflect a model's decision process.

Finally, LLMs have shown potential for advancing explainability in other
scientific domains. For instance, LLM-powered simulations have enabled
HCI designers to explore counterfactual scenarios and reason about
design decisions [@park2022social] while LLM-inspired approaches can
extract interpretable biological features from protein language models
[@simon2024interplm].

### Looking Forward

#### Providing understanding for model developers.

The black-box nature of LLMs, particularly the closed-source ones, makes
it difficult to predict and control how models behave. For example, when
models provide unsolicited affirmation to the user, it is unclear what
*causes* the model to provide that affirmation. As the range of
questions and interactions becomes more and more complex and open-ended,
interpretability becomes a key tool to answer questions like: how can we
determine if a model is personalized? Does the model truly understand a
user's intent? Developing an understanding of a model's representation
of the user is especially important as people use LLMs for personal
questions and even as AI companions. Without an understanding of models'
behaviors, model builders risk harming users' well-being
[@cheng2025sycophantic].

#### Uncovering unintended effects of post-training.

Another key application of interpretability for HCLLM is a better
understanding of post-training procedures like preference alignment
[@ferrao2025anatomy; @movva2025s]. Without more interpretable
approaches, post-training can lead to various unintended effects (e.g.
sycophancy) that can be difficult to monitor or mitigate post-hoc.
Building on existing approaches that refine our understanding of what
preference alignment actually optimizes for, model providers can better
control and steer behaviors towards desirable directions. For example,
representation finetuning and steering
[@rimsky2024steering; @wu2025improved; @wu2025axbench; @wu2024reft] have
been shown to be a promising way to control an LLM's behaviors, and
these methods can be applied to elicit behaviors that are aligned with
users' long-term development. The key first step towards making models
safer and more steerable towards long-term beneficial objectives would
be to understand how they work.

## Steerable HCLLMs

[]{#subsec:steerability label="subsec:steerability"}

### Current Approaches to Steerability

**Steerability** is the second dimension we highlight for HCLLM
deployment. Steerability is the degree to which a model can be aligned
along a particular dimension
[@miehling2024evaluatingpromptsteerabilitylarge], such as preferences,
norms, or user-constraints [@chen2025steer]. In contrast to static
notions of alignment that aim to produce a single globally acceptable
behavior, steerability emphasizes conditional control, or the ability to
modulate LLM outputs in accord with its particular users. This property
is especially important for HCLLMs, which are designed to interact with
diverse users embedded in heterogeneous social, cultural, and
institutional contexts.

Steerability spans multiple dimensions. First, **personalization** is
the goal in which an LLM's outputs adaptively reflect the preferences of
individual users or groups of users, along with their prior knowledge,
goals, and needs [@tseng2024two]. Personalized models should be able to
provide more relevant recommendations [@hu2024enhancing], and they
should be calibrated to the user's preferred writing styles
[@zhang2024personalization], learning styles [@park2024empowering], and
norms around privacy [@shao2024privacylens; @asthana2024know] and social
behavior [@li2024agentalignmentevolvingsocial]. A user's preferences can
be derived from explicit feedback, as in pairwise preference datasets,
or from implicit feedback like historical interaction data. For more
in-depth discussion of personalization methods, see
§[\[subsec:personalization\]](#subsec:personalization){reference-type="ref"
reference="subsec:personalization"}.

Related to personalization is **persona alignment** or role play. Here,
the goal is that HCLLMs will adopt consistent identities or roles, like
software developers, expert tutors, empathetic counselors, or skeptical
reviewers, which remain stable across interactions
[@li2024steerability; @samuel2024personagym; @shanahan2023role]. Persona
alignment is especially important in multi-agent settings where multiple
LLM personas interact and collaborate
[@park2023generative; @guo2024large].

In addition to better personalization and role playing, steerable HCLLMs
should be able to adaptively understand low-resource languages,
dialects, or sociolinguistic varieties [@ziems2023multi]. We refer to
this target as **linguistic alignment**. This form of steerability is
critical for equitable access, as models trained predominantly on
high-resource, standardized corpora often underperform for marginalized
linguistic communities (see
§[\[subsec:data_representation\]](#subsec:data_representation){reference-type="ref"
reference="subsec:data_representation"}). Finally, **cultural
alignment** means that models can be steered to reflect the norms,
values, and narratives of particular communities or demographic groups
[@santurkar2023opinionslanguagemodelsreflect]. Unlike personalization,
which targets individuals, cultural alignment operates at the level of
shared practices and collective meaning-making.

A range of technical mechanisms support steerability. At inference time,
models may be steered through in-context learning, prompt engineering,
or output filtering [@wies2023learnability]. Post-hoc control methods
can condition generation on specific attributes or enforce constraints
via decoding strategies. More structurally, personalized reward models
and fine-tuning procedures can encode user- or group-specific objectives
into model parameters [@chen2024pal].

Steerability is still fundamentally constrained by the representational
biases embedded in pre-training and post-training data
[@mihalcea2025ai]. If certain identities, linguistic forms, or cultural
narratives are underrepresented or stereotyped in the training corpus,
then prompt-based steering may have limited expressive range. In this
sense, steerability is not just a matter of control at inference time,
but is rooted much earlier in data provenance
(§[\[subsec:data_provenance\]](#subsec:data_provenance){reference-type="ref"
reference="subsec:data_provenance"}) and evaluation
(§[\[sec:evaluation\]](#sec:evaluation){reference-type="ref"
reference="sec:evaluation"}). Steerability starts with measuring where
biases arise, localizing their sources in the data pipeline, and
redesigning collection and annotation practices accordingly.

### Looking Forward

Looking forward, we envision several research directions that extend
current notions of steerability. First, we consider continual learning
for preference, culture, and pluralistic alignment. Continual preference
learning would allow models to adapt dynamically to evolving user needs
by tracking implicit cues in interaction patterns or contextual data
[@shaikh2025gum]. Such systems often rely on persistent memory stores
like chat histories or retrieval-augmented generation. A central
challenge is achieving this adaptivity while preserving user autonomy
and privacy [@zhang2025autonomy]. In cultural and pluralistic alignment,
rather than optimizing toward a static representation of *culture* or
*user values*, HCLLMs should accommodate evolving norms and intra-group
disagreement, as supported by continual learning. One way to elicit
these evolving norms is to facilitate community-centered discussion and
debate, following methods like STELA [@bergman2024stela]. Rather than
treating communities as homogeneous preference aggregates, pluralistic
alignment frameworks should model disagreement as a first-class signal.

Pluralistic alignment comes with an array of technical and political
challenges that will need to be addressed. On the technical side,
post-training can induce mode-collapse, in which heterogeneous group
preferences and opinions are compressed in a lossy manner, suppressing
minority viewpoints and preserving only the majority preferences for a
given group
[@bisbee2024synthetic; @durmus2024towards; @rottger2024political]. To
elicit diverse generations from LLMs, some inference time methods use
iterative prompting [@hayati-etal-2024-far; @feng-etal-2024-modular],
but these methods are not calibrated to real-world distributions. Other
methods involve modifying the prompt with explicit identity terms for
diverse user groups [@Giulianelli2023WhatCN], but identity coded names
induce models to draw on distributions of stereotypical representations
or out-group perceptions rather than in-group perspectives
[@wang2025large]. To address mode-collapse in a manner that preserves
in-group perspectives, it may be necessary to update LLM priors
implicitly through distributionally-aligned in-context examples,
following methods like spectrum tuning [@sorensen2025spectrum].

It is not only a methodological challenge, but also a political
challenge to achieve localized, domain-specific alignment processes that
enable communities and stakeholders to meaningfully shape model behavior
[@delgado2023participatory]. Methodologically, we have the participatory
HCI approaches covered in
§[\[subsub:participatory\]](#subsub:participatory){reference-type="ref"
reference="subsub:participatory"}. However, politically, most model
developers lack incentives to share control with communities
[@gabriel2020artificial]. Current alignment pipelines are centralized by
the small set of companies with resources to develop LLMs. Similarly,
academic institutions and governments can serve to centralize
decision-making across the LLM development pipeline
[@suresh2024participation]. There are a number of less centralized
alternatives. For example, Masakhane [@orife2020masakhane] is a network
of NLP researchers working on NLP for local African languages, with
community involvement at every stage, from dataset creation and
annotation to model training. EleutherAI is a distributed,
volunteer-driven research collective that has created open pre-training
corpora [@gao2020pile800gbdatasetdiverse], models [@black2022gpt], and
scaling checkpoints [@biderman2023pythia]. BigScience was a research
effort in which over 1,000 researchers from academia and industry
coordinated through HuggingFace and built BLOOM [@workshop2022bloom], a
176B-parameter multilingual autoregressive language model, with
decision-making steps clearly and publicly documented.

These initiatives illustrate that steerability does not need to be
confined to top-down fine-tuning interfaces or proprietary alignment
pipelines. Distributed model development allows communities to steer
models from the ground up, establishing linguistic resources, value
functions, and development practices that ultimately shape downstream
behavior. At the same time, decentralization introduces its own
tensions, including coordination costs, uneven resource distribution,
and challenges of accountability. Open and community-led efforts may
broaden participation, but they must still grapple with questions of
safety, quality control, and transparent maintenance of HCLLMs. In the
following sections, we will discuss safety
(§[1.3](#subsec:safety){reference-type="ref" reference="subsec:safety"})
and interpretability
(§[\[subsec:interpretability\]](#subsec:interpretability){reference-type="ref"
reference="subsec:interpretability"}), as well as the tensions between
these objectives.

## Safe HCLLMs {#subsec:safety}

The third dimension we emphasize when building responsible HCLLMs is
**safety**. As defined in
§[\[subsec:safety_eval\]](#subsec:safety_eval){reference-type="ref"
reference="subsec:safety_eval"}, safety is conceptualized as preventing
LLMs from producing undesirable outputs (i.e., those that may be toxic,
harmful, discriminatory, or dangerous), even when prompted to do so. For
example, the widespread use of LLMs raises critical concerns about
ethical and social risks related to their outputs, including
discrimination, hate speech, exclusion, misinformation harms, malicious
uses, and so on [@Zhang2022OPTOP; @bender2021dangers]. At the same time,
there are concerns that LLMs can be used as agents of harm, such as
using AI-generated propaganda for misinformation
purposes [@goldstein2024persuasive] or spreading information that can
facilitate harmful actions like the manufacturing of
weapons [@shaikh-etal-2023-second].

We begin by discussing the existing methods that are employed for
addressing safety concerns both at the model training and interaction
layers. Looking forward, we advocate for expanding beyond this current
definition of safety, which focuses on preventing harms, to encompass
how we can build HCLLMs that also maximize user benefits.

### Current Approaches to Safety

First, we will discuss current methods for measuring and mitigating
safety concerns. *Red-teaming* is a common practice for identifying
harmful behavior through adversarial testing prior to deployment.
Red-teaming approaches differ across model providers, and details are
often not publicly disclosed as these practices are conducted in
industry settings [@feffer2024red]. As @feffer2024red survey,
red-teamers typically come from three pools: subject-matter
experts [@ahmad2025openai], crowdworkers [@ganguli2022red], or automated
methods, such as the language models
themselves [@ganguli2022red; @perez2022red]. The objectives for
red-teaming can range from broad mandates to identify any harmful
behavior to targeted assessments of specific risks, such as those
related to national security. After deployment, model providers may also
run bug bounty programs that offer incentives for discovering safety or
security
vulnerabilities [@openai2025_gpt5_bio_bug_bounty; @anthropic2025_model_safety_bug_bounty].
In addition to red-teaming efforts, model developers make use of
benchmarks and other safety evaluations, which we discuss in detail in
§[\[subsec:safety_eval\]](#subsec:safety_eval){reference-type="ref"
reference="subsec:safety_eval"}.

Mitigations can appear at various stages of the model development
pipeline. At the pre-training phase, there is interest in filtering
datasets to remove toxic content as a preventative
safeguard [@obrien2025deepignorance; @mendu2025towards; @stranisci2025they].
At the same time, other work has argued that filtering toxic data during
pre-training can have detrimental downstream effects, and that including
such data at pre-training can actually make these behaviors easier to
remove through
fine-tuning [@li2025bad; @maini2025safety; @longpre-etal-2024-pretrainers].
Many efforts also address safety concerns during post-training.
Foundational techniques for modern LLMs, such as RLHF, are useful not
only for increasing model helpfulness but also for aligning models to be
more harmless [@ouyang2022training]. Building on these principles,
additional post-training methods can help automate parts of this
process. For instance, Constitutional AI allows researchers to
pre-determine a set of ethical principles for the model to adhere
to [@bai2022constitutional]. Instruction-tuning methods can also reduce
toxicity (see
§[\[subsubsec:succeses_instruction_tuning\]](#subsubsec:succeses_instruction_tuning){reference-type="ref"
reference="subsubsec:succeses_instruction_tuning"}). Once deployed,
guardrail models are used to help moderate both user inputs and
generated outputs [@inan2023llama; @dong2024building; @rebedea2023nemo].

### Looking Forward

Like other human-centered objectives, *safety* can be an underspecified,
ambiguous, or contested target. As we anticipate the development of more
human-centered LLMs, we should be asking whose definitions of safety are
prioritized, and how we can design models that not only prevent harm but
can actively promote user flourishing.

#### Paying heed to long-term harms.

As surveyed above, existing AI safety research tends to focus on
immediate harms that users face when interacting with models, such as
exposure to toxic speech or the production of misinformation. Of course,
these harms carry long-term societal consequences. However, an
underexplored class of safety problems involves behavior that appears
innocuous in the short term but can compound over repeated usage to
become problematic. The recent work in this vein has identified specific
model properties, such as sycophancy, which can affect users'
psychological states and behaviors [@cheng2025sycophantic]. An open
challenge lies in measuring these long-term interaction harms, as they
are difficult to capture with standard evaluation practices like
benchmarking. One alternative approach, as discussed in
§[\[sec:hci\]](#sec:hci){reference-type="ref" reference="sec:hci"}, can
be to run controlled experiments to understand the effect of model
properties on users [@cheng2025sycophantic; @kirk2025neural] or to
conduct qualitative inquiry [@mathur2025sometimes]. However, this
process can be time-intensive and furthermore potentially exposes
participants to the very harms being studied. This concern raises the
question of what alternative valid methods exist, such as those that can
simulate these harms in silica. Beyond measurement, mitigations also
remain largely underexplored --- both in terms of interventions at the
model design and user interaction paradigms. Researchers have identified
properties that can exacerbate harms (e.g., steering models towards
being relationship-seeking in model design [@kirk2025neural], sending
emotionally laden messages as users try to exit a
platform [@de2025emotional]), but translating these insights into
preventative measures remains an important and open area for
exploration.

#### Expanding the definition of safety.

A second area of exploration involves expanding *whose* definition of
safety is prioritized. Definitions of safety vary considerably across
demographic groups, along factors such as ethnicity, age, and
gender [@rastogi2025whose; @ali2025operationalizing; @movva2024annotation; @gabriel2020artificial; @aroyo2023dices].
These variances are then encoded into models through alignment
processes, significantly changing model
behavior [@ali2025operationalizing]. Thus, instead of assuming a generic
definition of safety, there is interest in better capturing and modeling
the diversity of conceptualizations that exist. This direction presents
a natural continuation of the existing focus on pluralistic alignment
within human-centered LLM research. How do we capture these differing
definitions of safety? Some work has tackled the issue through
recruiting diverse sets of annotators to rate safety
perceptions [@rastogi2025whose; @aroyo2023dices]. Others advocate for
engaging with communities in a more participatory fashion to elicit
safety goals, which offer a richer understanding of how different
communities understand the potential harms of these
technologies [@qadri2025case; @bergman2024stela].

Despite the benefits of moving away from this "view from nowhere"
conception of safety, it is important to remember that communities
themselves are not monolithic. Disagreements inevitably arise about what
constitutes safe or harmful content or what model behaviors are deemed
desired or unacceptable [@gordon2022jury; @gordon2021disagreement].
There is a legitimate concern that implementing democratic methods could
inadvertently drown out the voices of minority groups. Yet this is not
grounds to dismiss democratic or participatory methods for AI safety as
a lost cause. As @zimmermanndon outline in their work, there are viable
paths forward for reconciling these tensions by drawing on practices
from political philosophy. As we move towards a more pluralistic
definition of safety, this requires thinking normatively about the
contexts in which we jointly maximize or balance considerations across
groups of people, recognizing that collective decisions may at times
conflict with individual desires or goals.

#### Considering not only harms but also benefits.

Finally, much of the discussion so far has focused on mitigating harmful
behaviors. However, we emphasize that avoiding harm is not the same as
maximizing user benefits. In pursuit of human-centered LLMs, we must
also prioritize building models that bring positive change for users.
This raises important questions about what beneficial model behavior
looks like, and whether our current conceptions of safety align with
what is truly beneficial. First, we must challenge the assumption that
"safe" models are necessarily the best for promoting benefits. As
@cai2024antagonistic challenge, perhaps there is a need to design
*antagonistic AI* systems that are "actively dismissive, disagreeable,
closed-off, critical, flippant, difficult, interrupting." Much like, for
instance, a student may be challenged by their teacher in the learning
process, when we design for benefits rather than merely minimizing
harms, the desired model behavior changes.

A second provocation concerns the scope of benefit: rather than thinking
only about the one-to-one benefit of a model on an individual user, what
about one-to-many benefits? We can envision designing models for
collective or group-level good --- for example, models deployed to
promote democratic health by finding common ground through
deliberation [@tessler2024ai], or models designed to benefit teams by
serving as collaborators within group settings. Just as there are
differing definitions of what safety means, there are similarly diverse
conceptions of benefit, raising parallel questions about whose
definition should be prioritized and how we reconcile conflicting views
of what constitutes a beneficial outcome.
