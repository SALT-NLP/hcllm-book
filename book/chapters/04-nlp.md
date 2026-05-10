---
prev-chapter: "Data for HCLLMs"
prev-url: "03-data.html"
page-title: NLP for HCLLMs
search-title: "Chapter 4: NLP for HCLLMs"
next-chapter: "Evaluations"
next-url: "05-eval.html"
---

::: {#nlp}
# NLP for HCLLMs
:::

[]{#sec:nlp label="sec:nlp"}

Human-centered LLMs are products of the multifaceted technical processes
used to create them. NLP techniques determine not only what models can
do but also the boundaries of what they *cannot*. These limitations can
have particular consequences as users across diverse linguistic and
cultural contexts interact with LLMs.

Prior survey papers cover the technical practicalities and details of
NLP methods for LLMs [@minaee2024large; @zhao2023survey]. In this
chapter, we instead focus on the human-centered considerations across
the language model training pipeline. We have already discussed
pre-training practices in
§[\[sec:data\]](#sec:data){reference-type="ref" reference="sec:data"}
and will focus on ***post-training techniques*** in this chapter.
Although post-training recipes differ across models, two core components
include a supervised fine-tuning (SFT) stage
([4.1](#subsec:instruction_tuning){reference-type="ref"
reference="subsec:instruction_tuning"}) and a reinforcement learning
stage that incorporates human preferences
([4.2](#subsec:preference_tuning){reference-type="ref"
reference="subsec:preference_tuning"}). We next discuss how the
predominant paradigm of scaling applies to human-centered objectives
([4.3](#subsec:scaling){reference-type="ref"
reference="subsec:scaling"}). Finally, we conclude by discussing three
currently open challenges and future research directions for HCLLMs,
covering ***personalization
([4.4](#subsec:personalization){reference-type="ref"
reference="subsec:personalization"}), pluralistic alignment
([4.5](#subsec:pluralism){reference-type="ref"
reference="subsec:pluralism"}), and multilinguality
([4.6](#subsec:multilinguality){reference-type="ref"
reference="subsec:multilinguality"})***. For a roadmap, see
Figure [1](#fig:nlp){reference-type="ref" reference="fig:nlp"}.

![This chapter applies human-centered considerations to ***existing
post-training techniques*** like SFT and RLHF
([4.1](#subsec:instruction_tuning){reference-type="ref"
reference="subsec:instruction_tuning"}-[4.2](#subsec:preference_tuning){reference-type="ref"
reference="subsec:preference_tuning"}), and explores the limitations of
***scaling for human-centered outcomes***
([4.3](#subsec:scaling){reference-type="ref"
reference="subsec:scaling"}). Finally, we cover open challenges in
***personalization ([4.4](#subsec:personalization){reference-type="ref"
reference="subsec:personalization"}), pluralistic alignment
([4.5](#subsec:pluralism){reference-type="ref"
reference="subsec:pluralism"}), and multilinguality
([4.6](#subsec:multilinguality){reference-type="ref"
reference="subsec:multilinguality"})***.](../assets/04_NLP.png){#fig:nlp
width="\\linewidth"}

## Supervised Fine-tuning for HCLLMs
Following pre-training, the subsequent stage in the pipeline involves
some form of supervised fine-tuning (SFT), where models are trained on
curated datasets to align their outputs with specific objectives or use
cases. The goals of fine-tuning vary depending on the desired
capabilities and target applications. For instance, existing models have
employed SFT on step-by-step rationales and chain-of-thought reasoning
examples to enhance their problem-solving and reasoning
capabilities [@muennighoff2025s1; @olmo3_arxiv2025]. In other cases, SFT
can also be used to adapt models for more bespoke, domain-specific
applications [@cheng2025finemedlm; @yue2023disc]. Our discussion here
focuses specifically on **instruction tuning** --- the supervised
process of training LLMs on instruction-response pairs --- where models
learn to follow diverse user instructions and generate appropriate
responses. Instruction-tuning has become central to creating usable,
general-purpose conversational AI systems. We briefly survey current
practices in the literature before examining key tensions and exploring
emerging frontiers for instruction tuning HCLLMs.

### Current Practices in Instruction Tuning


Instruction tuning is critical in achieving the instruction-adherance
and generalized problem-solving capabilities that have helped
popularized LLMs. By guiding the models to follow explicit instructions
and domain-specific prompts, Instruction tuning improves LLMs'
capabilities to communicate in a human-centered and user-friendly
manner. While massive pre-training on self-supervised tasks improves the
model's understanding of language conventions and semantics, an LLM with
an assistant-like ability to respond conversationally and complete tasks
makes LLMs far more useful as human tools.

Already, we have seen the many successes of instruction tuning. It has
improved performance across diverse language models, from zero-shot
reasoning to domain-specific tasks. Beyond general alignment with human
preferences, studies have explored ethical and domain-specific
challenges, highlighting the versatility of instruction tuning. For
instance, @prabhumoye2023adding demonstrated how simply attaching
toxicity metadata as part of the instruction template significantly
reduces toxicity present in model outputs. This approach suggests that
explicitly encoding desirable or undesirable text qualities in
instructions might enable the LLM to learn to promote or withhold
similar texts with more ease. This demonstrates how instructions can
guide models to better align with social norms and ethical
considerations.

Moreover, domain-specific instruction tuning for coding
[@muennighoff2024octopackinstructiontuningcode], dialogue systems
[@ouyang2022training], financial analysis
[@xie2023pixiulargelanguagemodel], and multilingual translation
[@zhu2024finetuning] shows that modest sets of targeted instructions can
unlock robust capabilities without sacrificing general knowledge. By
specifying instructions to meet the requirements of each domain, LLMs
can provide more reliable and appropriate outputs. These successes
underscore instruction fine-tuning's vital role in shaping LLMs into
reliable, human-oriented assistants.

### Human-Centered Challenges with Instruction Tuning {#subsubsec:challenges_instruction_tuning}

While instruction tuning can generally improve the capabilities and
usability of LLMs, there are still tensions that emerge in
human-centered contexts. First, as mentioned before, instruction tuning
helps shape models to respond more conversationally, making them more
usable for users. However, instruction tuning can lead to *superficial*
improvements, rather than improving the reasoning capabilities of
models. For example, prior work found that models can merely learn to
mimic the structure of the input data without heed to factual
correctness or reliable reasoning and can fail to generalize to tasks
outside of the training
dataset [@gudibande2023false; @kung2023instruction]. When users interact
with models, these patterns become especially concerning, as users may
be more inclined to trust outputs as the instruction-following style
appears more polished, confident, and authoritative, creating a veneer
of competence that masks underlying reasoning
failures [@park2025critical; @rathi2025humans]. This misalignment
between surface-level fluency and actual reliability can lead users to
over-rely on model outputs in high-stakes contexts where factual
accuracy is critical.

A second tension with instruction tuning relates to *safety* concerns,
which we discuss in more detail in
[5.2.3](05-eval.html#subsec:safety_eval){reference-type="ref"
reference="subsec:safety_eval"}. Since instruction-tuned models are
trained to comply with the provided prompt, instruction tuning can make
models more susceptible to backdoor or poisoning attacks that embed
malicious behaviors in
datasets [@prabhumoye2023adding; @wan2023poisoninglanguagemodelsinstruction; @shu2023exploitability].
As a result, models are more likely to produce unsafe responses, such as
offensive or disallowed content. Furthermore, other work has
demonstrated that instruction tuning can make models more susceptible to
jailbreaking attacks, as they have been trained to follow human
requests [@zeng2024johnny]. Ostensibly, model designers can add more
safety data to the training dataset or align models to avoid these
harmful instructions. However, this practice can lead to overly cautious
models that refuse to answer even benign queries [@bianchi2024safety].
This resulting behavior is similarly unhelpful for users. Thus, drawing
this line between what instructions are permissible to follow in service
of helpfulness versus what instructions may lead to unsafe behavior
remains a core design tension.

### Future of Instruction Tuning for HCLLMs

What are the next frontiers for instruction tuning? A recurring theme
when discussing tensions in instruction tuning is the role that data
plays. As mentioned in
[3.1](03-data.html#subsec:data_provenance){reference-type="ref"
reference="subsec:data_provenance"}, ensuring dataset diversity may be
crucial for avoiding biases and enhancing generalization across varied
tasks. To tackle this question, we can think about diversifying
modalities of instruction tuning data and how data is sourced. In
addition, there are new frontiers for evaluating how instruction tuned
models along dimensions.

#### Multimodal data for instruction-tuning.

While our focus has centered on textual corpora for instruction-tuning,
human-LLM interaction is not confined to text alone. For hands-free
assistance, interacting via speech in addition to text is easier to
manage for users [@udandarao2025data]. For visual editing (e.g., figure
design, poster-making), models ought to operate in both visual and
textual modalities [@pang2025paper2poster; @si2025design2code]. Even in
early demonstrations of intelligent assistants, our interactions are
envisioned to seamlessly span multiple modalities. including speech,
gesture, and images [@bolt1980put]. Advancements in multimodal
instruction tuning and dataset creation [@li-etal-2024-multi] are thus
critical for developing HCLLMs. Moving beyond text introduces
modality-specific factors that are essential for human-centered
applications. For instance, prosody in speech can help disambiguate user
intent, and deictic gestures can provide spatial
grounding [@sasu2025enhancing; @brooks2006working]. Furthermore, these
multimodal capabilities require new approaches to both pretraining and
fine-tuning on interleaved multimodal data, enabling models to process
the rich integrated signals that characterize human
communication [@udandarao2025data; @cui2025emu3].

#### Synthetic data for instruction-tuning.

How we obtain the requisite data for instruction-tuning remains an
important open question. Synthetic data generation is an important
research area that presents potential challenges for HCLLMs. From a
technical perspective, synthetic data generation offers benefits for
scaling data collection by reducing reliance on human labor while
maintaining data diversity. There are some arguable human-centered
benefits: synthetic data can democratize model development by making
fine-tuning more accessible to researchers and practitioners without
large annotation budgets. Empirical work has demonstrated that synthetic
instruction data can be particularly valuable in low-resource settings,
enabling more data-efficient fine-tuning [@pengpun2024seed]. However,
synthetic data generation also poses new challenges, exacerbating
tensions discussed in
[4.1.2](#subsubsec:challenges_instruction_tuning){reference-type="ref"
reference="subsubsec:challenges_instruction_tuning"}. Models trained on
synthetic instructions may overfit to specific patterns present in the
generated data, and despite claims of increased diversity, synthetic
datasets can paradoxically reduce the authentic variation found in
human-generated instructions [@chen2024unveiling]. Figuring out how to
address these flaws in synthetic data is important for leveraging this
suite of methods to ensure models can handle the full range of
real-world user needs and interaction styles.

#### Human-centered evaluation frameworks.

Emerging research shows that instruction tuning aligns LLMs with human
brain activity, particularly in larger models with extensive world
knowledge [@aw2024instructiontuningalignsllmshuman]. These findings open
the door to more human-centered evaluation frameworks, where models are
examined by how closely they mirror human-like reasoning, empathy, and
context awareness. This direction has implications for designing systems
that better respect ethical norms, cultural sensitivities, and user
well-being.

## Learning from Human Preferences 
In recent years, the idea of fine-tuning LLMs on human preferences has
seen remarkable success in improving their behavior. Previously, it was
believed that training on more samples and increasing model size was
sufficient to increase performance. However, researchers found that
these scaling rules ignored *alignment*, defined by
@askell2021generallanguageassistantlaboratory as helpfulness,
harmlessness, and honesty in model responses. To this end, it was
discovered that incorporating human feedback directly into the training
process achieved massive gains in human preference alignment
[@askell2021generallanguageassistantlaboratory; @leike2018scalableagentalignmentreward; @bai2022traininghelpfulharmlessassistant].
As such, in this section we discuss these developments chronologically,
starting with reinforcement learning from human feedback (RLHF), its
spinoffs, including direct preference optimization (DPO), and recent
frameworks like Constitutional AI which aim for a future of fully
self-supervised AI alignment.

### RL-Based Methods

Much of RLHF is built upon landmark research by @christiano2017deep,
@stiennon2022learningsummarizehumanfeedback, and @ouyang2022training.
Together, these works demonstrated the feasibility of learning a reward
function from human preferences and optimizing that function, first in
the domain of simple robotics and Atari video games
[@christiano2017deep], then in the ability to improve LLM performance on
summarization tasks [@stiennon2022learningsummarizehumanfeedback], and
finally in improving LLM behavior on a wide breadth of tasks, including
open generation, chatting, and question and answering
[@ouyang2022training].

Today, the canonical algorithm used to perform RLHF is proximal policy
optimization (PPO) [@schulman2017proximalpolicyoptimizationalgorithms],
originally introduced as a simpler and more general improvement upon
older RL methods like trust region policy optimization (TRPO)
[@schulman2017trustregionpolicyoptimization]. However, there now also
exists considerable research into exploring alternatives to PPO. To
address PPO's high computational cost and sensitivity to hyperparamater
tuning, @ahmadian2024basicsrevisitingreinforcestyle explore breaking PPO
into its component pieces, and show that revisiting the formulation of
human preferences in RL, discarding aspects that are unnecessarily
complex for fine-tuning pre-trained LLMs, and returning to the most
basic policy gradient algorithm, has yielded notable performance and
efficiency gains. Other works propose alternatives to PPO entirely, such
as bringing the process online for online iterative RLHF
[@dong2024rlhfworkflowrewardmodeling] or scoring sampled responses from
different sources and aligning these with human preferences (RRHF)
[@yuan2023rrhfrankresponsesalign].

Other works on extending RLHF focus specifically on the data that goes
into aligning LLMs, whether that be improving accessibility by filling
gaps in existing datasets or addressing issues of scale. For example,
Okapi [@lai2023okapiinstructiontunedlargelanguage] is introduced as the
first system and dataset to focus on RLHF for multiple languages,
covering 26 different languages. For issues of scale, researchers from
Google's DeepMind propose reinforced self-training (ReST)
[@gulcehre2023reinforcedselftrainingrestlanguage], which takes
inspiration from growing batch RL to produce a dataset consisting of
samples generated from the policy, which can then be used for offline
training.

Beyond PPO and its proposed alternatives, there also exists considerable
discussion on other portions of the RLHF pipeline, targeting better
alignment through task formulation and dataset augmentation. Safe RLHF,
proposed by @dai2023saferlhfsafereinforcement, explicitly decouples
human preferences around helpfulness and harmlessness into two separate
optimization objectives, and uses the Lagrangian method to balance
trade-offs between the two.

### Non-RL Methods

DPO has gained significant attention as an RLHF alternative because it
enables preference tuning without an explicit reward model. It does so
by directly including the probability ratio between preferred and
dispreferred responses in its loss function
[@rafailov2024directpreferenceoptimizationlanguage]. The original
authors show that DPO-trained models generate responses that are
preferred more frequently than those trained by PPO, and that DPO
converges faster.

Another sample-efficient alternative, which not only avoids RL but also
requires fewer than 10 samples, is Demonstration Iterated Task
Optimization (DITTO) by [@shaikh2024show]. This method uses online
imitation learning to create pairwise comparisons, treating user
demonstrations as the gold standard and the model's own outputs as
dispreferred. DITTO's improvement in model alignment was shown through
an average improvement of 19% in win rates, compared to few-shot
prompting and supervised fine-tuning on various human-centric tasks like
news writing, emails, and blog posts.

### Beyond Human Feedback

Despite the success of methods like RLHF and DPO, recent research has
sought to address the potential drawbacks of only using human-sourced
feedback for LLM alignment. One drawback is the incompleteness of human
feedback, which may only represent a partial view of collective human
values [@kirk2023pastpresentbetterfuture]. Furthermore, alignment is
difficult to specify with explicit objectives
[@tamkin2021understandingcapabilitieslimitationssocietal; @bommasani2022opportunitiesrisksfoundationmodels].Additionally,
scaling quality and representative human feedback will become
increasingly difficult with larger and more powerful LLMs
[@casper2023openproblemsfundamentallimitations; @santurkar2023opinionslanguagemodelsreflect].

#### Constitutional AI.

To resolve these intricate issues, recent work has shifted from using
RLHF to enlisting AI help along with human collaboration to supervise
other AIs to train helpful and harmless AI systems
[@bowman2022measuringprogressscalableoversight; @bai2022constitutional; @saunders2022selfcritiquingmodelsassistinghuman].
As previously mentioned, one issue with alignment is that it is not
clearly defined.

Anthropic's work on Constitutional AI establishes a framework designed
to answer this exact question [@Bai2022ConstitutionalAH]. Rather than
have humans provide explicit feedback, which may be inherently biased or
incomplete, Constitutional AI only includes human feedback through the
creation of a set of alignment principles (i.e. a \"constitution\").
Then, a model undergoes a training process similar to RLHF, but where
the rewards are given by an LLM fine-tuned according to the values in
the constitution
[@ouyang2022training; @bai2022traininghelpfulharmlessassistant]. This
approach also uses chain of thought reasoning to maximize LLM
self-reasoning capabilities throughout the entire process
[@nye2021workscratchpadsintermediatecomputation; @wei2023chainofthoughtpromptingelicitsreasoning].

The end goal of Constitutional AI is not to get rid of human involvement
or supervision entirely, but rather to have humans involved in only the
most necessary aspects to move towards a self-supervised approach to
alignment. Although Constitutional AI helped resolve many lingering
issues with RLHF, this approach also brings up new questions in the
ongoing research of alignment. First, how does the global AI research
community come up with a widely accepted constitution that incorporates
the pluralistic values of human beings
[@hendrycks2023aligningaisharedhuman]? Second, how do we ensure a
universal understanding and interpretation of the presumed constitution?
How do we make sure there is a robust system for editing and improving
the principles and rules as the society evolves? And when constitutional
guidelines fail in ambiguous situations, how do we ensure that the
models with minimal human supervision can still behave in a safe and
useful way?

## Scaling Human Centered LLMs
In NLP, "scaling" refers to the relationship between a model's
performance and factors such as the number of parameters $n$, dataset
size $d$, and computational resources $c$ [@kaplan2020scaling].
Understanding these scaling laws is important for developing HCLLMs that
balance efficiency and performance with accessibility and fairness.

### Scaling Laws in LLMs

@kaplan2020scaling conducted foundational research on empirical scaling
laws for language model performance, particularly focusing on
cross-entropy loss. Their work established that model performance
improves predictably with increases in $n$, $d$, and $c$, following a
power-law relationship. Importantly, they discovered that returns
diminish when either $n$ or $d$ is held constant, underscoring the need
for a strategic, balanced approach to scaling in NLP.

@tay2022scaling expand on @kaplan2020scaling's work to investigate the
effect of scaling properties of different inductive biases and model
architectures. They find via extensive experiments that (1) architecture
is an important consideration, (2) the best performing model can
fluctuate at different scales, and (3) the choice of whether to scale
depth (number of layers) or width (more neurons per layer) is important,
especially in resource-constrained environments. Models often excel at
pretraining but underperform on downstream tasks, underscoring the need
to evaluate models based on human utility rather than just raw
performance metrics. @hoffmann2022training introduced \"Chinchilla
scaling,\" which showed that smaller models trained on larger datasets
achieve better performance per compute budget. This finding is
applicable in resource-constrained human-centered applications where
compute and data availability may be limited due to ethical or
logistical constraints.

@ivgi2022scaling investigate the applicability of scaling laws for
different NLP tasks and find that benefits vary. Tasks aligned with
pretraining objectives, such as question answering, show clearer scaling
behavior compared to specialized tasks like sentiment analysis. Thus,
for human-centered applications, practitioners should assess whether
scaling laws are applicable to their specific use case or if full-scale
testing is necessary. While scaling can improve performance for some
tasks, it may not always be the most efficient path - a point further
developed by @liang2022holistic, who show that similar or greater
improvements to model accuracy can be achieved through more efficient
human-centric means, such as training with human feedback.

### Scaling in Human-Centered Domains

Although scaling improves the overall performance of LLMs, it does not
proportionally improve performance at the same rate for all
subpopulations and human-centered knowledge domains. Representational
biases in training data
([3.1](03-data.html#subsec:data_provenance){reference-type="ref"
reference="subsec:data_provenance"} and
[3.2](03-data.html#subsec:data_representation){reference-type="ref"
reference="subsec:data_representation"}) can lead to disparities in
scaling [@rolf2021representation]. However, data is not the only cause
of relative disparities in scaling, and may not even be the principal
cause. @held2025relative found that, when holding the data scale
constant, model-size scaling is responsible for *widening* the
performance gap between LLM performance on certain varieties of English
relative to other varieties. In addition to dialect, the authors
investigated AI risk behaviors [@perez2023discovering] and found that
scaling mitigates some risks more than others.

Scaling model size alone cannot address human-domain-specific challenges
such as cultural biases, as
@bommasani2022opportunitiesrisksfoundationmodels highlight. The
effectiveness of scaling laws varies across different domains, with some
human-centered areas experiencing diminishing returns when either the
number of parameters ($n$) or dataset size ($d$) is limited. For
instance, @brownLanguageModelsAre2020 observed that large models like
GPT-3 show reduced performance gains on human-centered tasks unless they
are fine-tuned with context-specific data, emphasizing the need for
tailored approaches in these applications.

@gururangan2020don conducted a study to investigate whether it is still
helpful to tailor a pretrained model to the domain of a target task in a
world where large-scale models, which form the foundation of today's NLP
landscape, have found much success in a broad-coverage approach trained
on a wide variety of sources. Overall they found that multi-phase
adaptive pretraining offers large gains in task performance, implying
that the quality and treatment of the data $d$ is more important than
the quantity, which is relevant when dealing with sensitive or
specialized human domains.

@bender2021dangers contribute by arguing that increasing $d$ without
considering diverse and ethical data sources can lead to biased or
non-representative outcomes, negatively impacting human-centered goals
such as equitable access and cultural inclusivity. Furthering the
discussion on fine-tuning for specific human-centric domains,
@zhang2024scaling examine how different scaling factors influence the
fine-tuning performance of LLMs. The study aligns with
@kaplan2020scaling and @hoffmann2022training, exhibiting a
multiplicative joint scaling law that links fine-tuning performance to
model size, fine-tuning data size, and other scaling factors. It
indicates that fine-tuning performance improves predictably when scaling
both dataset size $d$ and model size $n$ and finds that LLM fine-tuning
benefits more from LLM model scaling than pretraining data scaling. The
work also highlights that the effectiveness of fine-tuning varies
significantly based on the downstream task and the size and quality of
the available fine-tuning data, underscoring the need for task-specific
approaches in human-centered applications.

### Scaling in Human-Centered Goals

Looking at specific human-centered evaluations, such as bias and
fairness, there are potentially unexpected results when models are
scaled. @ethayarajh2020utility show that leaderboard-driven scaling can
create misaligned incentives in model development. By analyzing examples
like the SNLI leaderboard, they demonstrate that focusing solely on
state-of-the-art performance discourages practical models. For instance,
with SNLI baselines at 78% (n-gram) and 81% (LSTM) versus a 92%
BERT-based SOTA, there is little motivation to develop lightweight
models with  85% accuracy, which could balance performance and
computational efficiency and be more practically useful.

@ganguli2022red explored model safety through red teaming, where testers
try to provoke harmful outputs. Testing models from 2.7B to 52B
parameters, they found that RLHF-trained models became harder to 'break'
as they scaled, reducing the success of harmful attacks (the
harmlessness score increased from approximately -0.5 with 2.7B
parameters to 0.5 with 52B parameters). In contrast, other models showed
no improvement in resisting such outputs with increased size. This study
stresses the importance of incorporating human feedback during training
to develop safer AI systems. Larger models exhibit heightened privacy
risks. @hernandez2022scaling demonstrate that an 800M parameter model
could be degraded to that of a 400M model by repeating just 0.1% of the
training data 100 times, suggesting that larger models aren't
automatically more robust to certain types of data-based attacks (see
[3.3](03-data.html#subsec:data_privacy){reference-type="ref"
reference="subsec:data_privacy"}).

In regards to emulating human values, @biedma2024beyond showed that as
language models get larger, they show an increased preference for the
task-oriented values like accuracy and factual consistency at the slight
expense of social intelligence or moral fiber and adherence to ethical
norms. While larger models may become more capable, their value systems
have the potential to become increasingly misaligned with human values.

Transfer learning is often a prerequisite for the application of LLMs in
human-centered tasks. Scaling laws suggest that a model's ability to
transfer knowledge improves as its performance increases. This
relationship generally holds in human-centered evaluations, with studies
showing that transfer learning benefits from scaling in broad tasks
[@hernandez2021scaling; @raffel2020exploring]. However,
@hernandez2021scaling and @raffel2020exploring highlight the importance
of domain-specific fine-tuning and alignment techniques to achieve
human-centered objectives in specialized or sensitive areas such as
legal and medical contexts.

### Inference time scaling

Recent advances in inference-time scaling offer pathways to improve
HCLLMs without retraining. Now more targeted approaches to
inference-time adaptation are emerging that specifically address
human-centered concerns.
[@zhang2024controllablesafetyalignmentinferencetime] introduce
Controllable Safety Alignment (CoSA), a framework that enables
inference-time adaptation to diverse safety requirements without model
retraining. Rather than following a one-size-fits-all approach to safety
alignment where models refuse any potentially unsafe content, CoSA
allows authorized users to modify safety configurations at inference
time through natural language descriptions of desired safety behaviors.

Despite these promising directions, there remain open questions about
whether increased inference compute might undermine certain
human-centered objectives. OpenAI's emphasis on chain-of-thought (CoT)
in their o-series of models, for example, underscores the prevailing
focus on inference-time reasoning strategies
[@learningtoreasonwithllms]. However, @shaikh-etal-2023-second find that
explicitly prompting models to "think step by step" can inadvertently
increase harmful biases and toxic outputs, observing an 8.8% rise in
biased responses and a 19.4% increase in toxicity across relevant
benchmarks. Their study suggests that while inference-time reasoning
holds promise for improved performance or controllable safety alignment,
it may also expose underlying biases.

## Personalization {#subsec:personalization}

We established that the goal of the post-training stages is to align
LLMs to human preferences. Nonetheless, the term "human preferences" is
blanket statement that obscures the diverse and potentially conflicting
desires that different users have. Rather than trying to create LLMs
that can satisfy everyone, the goal of personalization is to align a
model's outputs with the preferences of a single individual, both in
terms of content and style [@zhang2024personalization; @tseng2024two].

### Current Approaches

We cover three families of techniques for personalizing LLMs:
prompting-based approaches, retrieval-based approaches, and personalized
alignment (e.g., learning from human preferences). There is no "best"
technique for personalization and instead depends on factors, such as
what data is available, compute efficiency, degree of personalization
required, and so on.

#### Prompting-based Approaches.

Given the capabilities of LLM, prompt-based approaches provide an
efficient and popular approach to personalizing model outputs. One
prompt-based technique is to provide a user persona directly in the text
instructions provided to the model. Here, the critical research decision
is figuring out what content to provide for personalization. For
example, prior work has explored providing demographic information
(e.g., race, gender) to the model for personalization, although these
personas run the risk of stereotypes or
caricaturization [@cheng-etal-2023-compost; @cheng-etal-2023-marked; @gupta2023calm; @huang2023humanity].
In addition, this approach has been adopted to imbue character traits
upon the model, such as prompting the responses to reflect certain
personality qualities (e.g., extroversion, warmth) or different
tones `\cite{}`{=latex}. Since these methods can suffer from
inefficiency and information loss [@Liu2023LostIT], other works have
also explored embedding user information into tokenized prompt
embeddings [@li2024personalizedlanguagemodelingpersonalized; @hebert2024persomapersonalizedsoftprompt; @huang2024selectivepromptingtuningpersonalized].

#### Retrieval-based Approaches.

What information about the user is relevant depends on the context.
Retrieval augmented personalization methods retrieve information from an
external knowledge base that is then incorporated at inference-time to
personalize the model's output. As an initial foray into this area,
@lamp benchmark different retrievers, including BM25 and a dense
retrieval model, on a series of personalization tasks. Other work has
continued to refine retrieval methods, exploring how to improve
capabilities while reducing the amount of retrieved data via
summarization [@rag_summary] or adopting techniques from other areas
such as collaborative filtering [@shi2025retrieval]. Although
retrieval-based methods allow for on-the-fly personalization of models
at inference-time, the performance is constrained by retriever quality.
For example, lexical retrievers, such as BM25, may match on shallow
keyword similarity rather than deep semantic understanding of user
needs. Furthermore, this approach may be more constrained in cold-start
situations or domains with sparse user data, where the knowledge base
itself may be insufficient to support meaningful personalization.

#### Personalized Alignment.

Finally, rather than adapting model behavior at inference time,
training-based approaches can embed individual preferences into
alignment objectives. For example, several works present methods for
creating personalized reward models, such as approaches that train
multiple reward models that are later merged [@jang2023personalized].
Critically, these approaches require that the dimensions for
personalization are defined a priori, limiting the scope to which they
can be applied. Alternative approaches have sought to loosen these
constraints, proposing methods that learn preferences from historical
user interactions, which can then be used to train personalized reward
models or directly align
LLMs [@ryan2025synthesizeme; @poddar2024personalizing; @balepur-etal-2025-whose].
However, training-based personalization faces practical limitations:
these methods require substantial amounts of user data in domains where
data is often scarce, risk overfitting to individual user patterns, and
incur significantly higher computational costs compared to
retrieval-based or prompting
approaches [@ryan2025synthesizeme; @shaikh2024show; @zhang2024personalization].
These tradeoffs make training-based personalization most suitable for
scenarios where there is sufficient data present and customization
justifies the additional resource investment.

### Future of Personalization for HCLLMs

A central challenge in personalization is deciding what the model should
adapt to. For example, economists will distinguish between stated
preferences --- what people say they want --- and revealed
preferences---inferred from behavior [@samuelson1948consumption]. In
language model personalization, this tension also manifests. Behavioral
signals such as query reformulations, response ratings, or conversation
length may reflect immediate satisfaction but diverge from users' stated
goals or long-term values. Consider a user learning a new subject who
frequently requests direct answers. Based on inferred signals, the model
might be personalized to comply whereas a model targeting learning
outcomes might instead offer scaffolded hints. This raises fundamental
questions about system objectives and user agency: which preferences
should dominate when conflict arises, and how should systems handle
patterns users exhibit but might not endorse upon reflection?

Temporal dynamics present an additional challenge. User preferences and
needs evolve over timescales, ranging from within-session learning to
long-term skill development and shifting life contexts. Yet, current
approaches treat personalization as a static task. While recent work has
extended the length of conversation in personalization datasets, we lack
real-world benchmarks that capture these longer-term dynamics or
benchmarks for evaluating personalization over
time [@kirk2024the; @zhang2024personalization; @zhao2025llms]. There is
some early work that explores updating user profiles over time, but
critical questions remain open [@wang2024lifelong]. For instance, how
can systems distinguish transient preferences (a user exploring a new
hobby) from enduring ones (a domain expert's consistent working style)?
What mechanisms allow for efficient model updates under these
circumstances? These temporal considerations compound the preference
alignment challenges; even if we can perfectly identify and incorporate
a user's current preferences into LLM outputs, those preferences
themselves may be moving targets.

## Pluralism {#subsec:pluralism}

While we have discussed methods for aligning models to human preferences
and values, the important, unaddressed question is *whose* values. Not
all humans share the same set of values [@durmus2024towards], and what
is permissible to some may be irrelevant or harmful to others. Rather
than assuming values are monolithic and aligning models to a "Silicon
Valley default" set of preferences, *pluralistic alignment* is an
approach of aligning language models to simultaneously serve diverse
preferences. As @DBLP:conf/icml/SorensenMFGMRYJ24 define, pluralistic
alignment is the process of creating language models capable of
representing a diverse set of human values and perspectives.

### Current Approaches

#### Defining Pluralism

Prior work has proposed three definitions as to how pluralism can be
embedded into models [@sorensen2024roadmap]:

1.  **Overton Pluralism**: An Overton pluralistc model generates all
    *reasonable* perspectives for a given subject. The model draws on
    the concept of the "Overton window," which encompasses the range of
    ideas and perspectives that are considered acceptable within the
    mainstream. Scholars such as @lake2024from have posited that Overton
    pluralism is compatible with the status quo alignment process, as
    longer conversational outputs typically share a few varied
    perspectives.

2.  **Steerable Pluralism**: Steerable pluralism advocates for using
    situational context to steer the model towards a particular
    perspective. One popular use case of steerable pluralism is
    improving the cultural alignment of language models to particular
    groups and cultures
    [@masoud2024culturalalignmentlargelanguage; @tao2024cultural].
    Personalization also falls under steerable pluralism as the model is
    steered to the values of a particular user.

3.  **Distributional Pluralism**: Finally, in distributional pluralism,
    developers steer models to produce responses that roughly correspond
    with a population distribution. That is, if 70% of the population
    holds a particular opinion, the model will generate that opinion 70%
    of the time. As @lake2024from note, base models are somewhat
    distributionally aligned to the opinions present in pre-training
    data.

#### Methods for Pluralistic Alignment.

Several works have proposed different methods for alignment, depending
on what type of pluralism they seek to achieve. For Overton pluralism,
researchers have explored prompting
techniques [@meincke2024promptingdiverseideasincreasing] and approaches
that generate multiple perspectives before synthesizing
them [@feng-etal-2024-modular; @hayati-etal-2024-far; @li2024largelanguagemodelsecretly].
For steerable pluralism, personalized reward
models [@jang2023personalized; @poddar2024personalizing; @chen2024pal]
and optimization techniques (e.g., Group Preference
Optimization [@zhao2023group]) enable alignment to specific user or
group preferences with minimal context. Other common implementation
methods include targeted prompting [@alkhamissi-etal-2024-investigating]
and generate-then-filter approaches [@feng-etal-2024-modular]. Finally,
for distributional pluralism, researchers generate diverse perspectives
and filter based on population distributions [@feng-etal-2024-modular],
with surveys serving as key measurement tools since they enable matching
LLM probability distributions to actual population responses.
OpinionsQA [@santurkar2023opinionslanguagemodelsreflect] and
GlobalOpinionsQA [@durmus2024towards] are two widely-used survey
benchmarks for evaluating distributional alignment.

In tandem, a data-centric approach to pluralistic alignment has focused
on collecting datasets that represent a diversity of values and
perspectives. Chatbot Arena [@DBLP:conf/icml/ChiangZ0ALLZ0JG24] and
PRISM [@kirk2024the] are two such collections of preference data that
retain user labels. The PERSONA dataset attempts to simulate 1,500 users
with synthetic personas for the purpose of studying pluralistic
alignment, providing synthetic prompts and feedback pairs
[@castricato2024personareproducibletestbedpluralistic]. The ValuePrism
dataset [@sorensen2024value] is a collection of values, rights, and
duties applied to various scenarios. They also release the KALEIDO model
for measuring how certain statements agree or disagree with particular
values. ValueConsistency [@moore-etal-2024-large] is a dataset of 300
controversial topics in four languages with corresponding controversial
questions. Moral Stories [@emelin-etal-2021-moral] is a dataset of
narratives with moral dilemmas with multiple endings. The stories were
written from annotators of various demographic backgrounds and express
different social and moral norms.

### Future of Pluralistic Alignment for HCLLMs

Achieving effective pluralistic alignment faces several challenges.
First, figuring out how to appropriately and effectively model diverse
perspectives is a key precursor to effective pluralistic alignment.
Existing literature has shown that models tend to emphasize
stereotypical representations when asked to simulate perspectives from
different demographic
groups [@cheng-etal-2023-marked; @cheng-etal-2023-compost; @deshpande2023toxicity],
meaning that approaches that generate and synthesize varied perspectives
require careful design to avoid misrepresenting communities. While
prompting LLMs with demographic features can improve alignment with
group opinions [@alkhamissi-etal-2024-investigating], the choice of base
model sometimes has a more significant effect than the sociodemographic
features themselves [@beck-etal-2024-sensitivity]. Effective perspective
modeling remains an open problem.

A second challenge involves understanding and balancing the societal
impacts of pluralistic models on public opinion. It is well-established
that aligning LLMs to human preferences can exacerbate model sycophancy
[@sharma2024towards], where models agree with users regardless of
validity. Personalized models developed through steerable pluralism risk
creating personalized echo chambers that reinforce rather than challenge
user beliefs. Already there are concerns about "echo chambers" in media
consumption, which may only be further augmented with
LLMs [@cinelli2021echo; @barbera2020social]. Even though pluralistic
alignment aims to broaden the viewpoints that models may espouse, these
methods still define what perspectives are considered "in bounds" of
acceptability or follow ostensible public opinion distributions. In
doing so, these methods may inadvertently silence marginalized
viewpoints that fall outside mainstream acceptability. Future work
should empirically investigate how different pluralistic alignment
strategies affect opinion formation and marginalization in practice,
moving beyond theoretical concerns to measurable impacts on diverse user
populations.

Finally, pluralistic alignment raises questions of data sovereignty and
consent. Many communities, particularly indigenous groups, do not want
their data and opinions collected for training AI systems
[@rainie2019indigenous]. Even when motivated by the goal of broad
representation, developers must carefully consider whether all
communities want their perspectives embedded in AI systems, respecting
the right of groups to opt out of technological representation entirely.

## Multilinguality {#subsec:multilinguality}

Beyond being able to capture the needs, values, etc. of users, users
ought to also be able to interact with models in their preferred
language. Although there are over 7,000 languages spoken worldwide, most
of LLM development focuses on English [@held2023material]. Expanding the
multilingual capabilities plays a critical role in helping democratize
access to language models, particularly bridging the gap between
high-resource and low-resource language technologies.

### Current Approaches

Multilingual large language models (MLLMs) are systems capable of both
understanding and generating text in multiple languages. While most LLMs
perform best in English, there is a concerted effort to improve their
multilingual capabilities. From a data-centric perspective, these
efforts focus on curating diverse linguistic resources, including
pre-training corpora and post-training datasets for supervised
fine-tuning (SFT) and preference learning. Many pre-training corpora,
such as RedPajama [@weber2024redpajama] and CC-100 [@conneau_etal_2020],
are collected via web scraping. Furthermore, many existing pre-training
corpora that are primarily in English already include a small percentage
of non-English data; incorporating even this small amount of data during
pre-training can improve cross-lingual
capabilities [@blevins2022language]. Nonetheless, a core challenge is
that the Internet remains heavily English-centric, leading to
performance gaps for less-resourced languages that appear infrequently
online or, in some cases, lack standardized written forms. For an
in-depth survey on multilingual LLMs, please refer to @qin2025survey.

Rather than relying solely on naturally occurring multilingual text,
researchers have turned to translating high-resource English text into
other languages. For example, @wang2025multilingual translate the
pre-training corpus FineWeb into multiple languages for pre-training.
Similar approaches exist for creating post-training datasets, such as
the Aya [@ustun2024aya] and CrossAlpaca [@ranaldi2024empowering]
corpora, which rely on either machine translation or existing MLLMs to
assist with
translation [@lai2023okapiinstructiontunedlargelanguage; @yue2024pangea].
However, translated datasets can introduce artifacts (referred to as
*translationese*) that shift linguistic patterns and degrade data
quality, particularly for downstream reasoning tasks and open-ended
generation [@dang2024rlhf; @vanmassenhove2021machine]. While translation
is an efficient way to generate large amounts of multilingual data,
another question we must ask is what gets lost in translation? For
example, when translating from text in English, the resulting output
might lack important cultural context and nuance that would be found in
text originating in places that speak the selected
language [@qin2025survey].

Finally, complementary to training interventions are non-training
approaches enabled through prompting. A wide range of prompting
strategies for multilingual use has been proposed, including those
surveyed in recent work by @vatsal2025multilingual. One common strategy
is "translate-test", or to translate user input into English before
performing the task, leveraging the stronger English proficiency of most
current
LLMs [@liu2025translation; @artetxe2023revisiting; @huang2022zero; @etxaniz2024multilingual; @huang2023not].
While this approach often boosts performance, it can fall short for
tasks requiring cultural nuance, idiomatic understanding, or
language-specific world knowledge, where remaining in the original
language is critical for faithful interpretation and
generation [@liu2025translation].

### Future of Multilinguality for HCLLMs

Looking forward, efforts toward *human*-centered multilingual
capabilities must consider the following areas. First, there is a
growing interest in multilingual safety. As @yong2025state identify,
multilingual safety remains massively underrepresented as a research
domain, resulting in safety standards built for English that do not
translate effectively to other linguistic contexts. Treating English as
the universal reference point obscures sociolinguistic variation and
produces a gap between how models behave and how safety norms should
operate for real users across languages. Understanding and addressing
multilingual safety thus remains an open frontier.

A second challenge involves determining what it means to represent
language in ways that do not exploit the communities that speak it. The
desire for multilingual NLP is not new. Projects such as Meta's No
Language Left Behind demonstrate longstanding investment in broad
language coverage. However, as @bird2024must argues, these efforts often
treat language as a detached artifact rather than something rooted in
communities, cultures, and social practices. When language is treated as
a pure optimization target, scraped data, or a resource to be
"unlocked," the resulting systems risk extraction without contributing
tangible value to the communities whose linguistic labor enables them.

Beyond technical considerations, achieving authentic multilinguality in
human-centered systems demands rethinking how data is gathered, whose
language practices are modeled, and for what ends. Simply scaling web
pre-training and technical fixes may improve multilingual benchmarks,
but it does not confront the deeper question of whether these systems
advance the needs, agency, and self-determination of speakers. Efforts
to "democratize" access to LLMs echo earlier narratives that cast
computing as a universal solution. As the Information and Communication
Technology for Development (ICT4D) literature reminds
us [@toyama2015geek; @harris2016ict4d], such narratives risk reproducing
existing inequities when social, cultural, and political contexts are
ignored.
