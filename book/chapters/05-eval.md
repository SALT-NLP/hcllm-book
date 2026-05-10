---
prev-chapter: "NLP for HCLLMs"
prev-url: "04-nlp.html"
page-title: Evaluations for HCLLMs
search-title: "Chapter 5: Evaluations for HCLLMs"
next-chapter: "Responsible Human-Centered LLMs"
next-url: "06-responsible.html"
---
# Evaluation

[]{#sec:evaluation label="sec:evaluation"}

Evaluation methods allow model developers, users, and stakeholders to
compare the capabilities and limitations of different LLMs, and to
understand the scope of their utilities and risks in particular domains
[@10.1145/3641289]. Evaluations can inform all stages of model
development, from mixing pre-training data
[@held2025optimizing; @mizrahi2025language] to selecting and optimizing
reward models for alignment
[@lambert2024rewardbenchevaluatingrewardmodels; @frick2024evaluate].
After models are trained, evaluations arguably become even more
important. They act quality filters to decide whether and how companies
deploy models [@liang2022holistic]. They inform researchers on the most
important and promising directions for model improvement
[@srivastava2022beyond], and help anticipate models' future capabilities
[@kaplan2020scaling; @hoffmann2022training]. And finally, they shape
public perceptions [@liao2022designing], and inform policy and other
regulatory decisions [@eriksson2025can].

Without a human-centered evaluation of LLMs, model development,
deployment, and governance may be oriented not towards the long-term and
collective good, but rather towards profit incentives and short-term
gains on surface-level heuristics [@eriksson2025can]. In this chapter,
we observe common pitfalls and highlight best practices in
human-centered evaluation, spanning three levels as shown in
Figure [1](#fig:evaluation){reference-type="ref"
reference="fig:evaluation"}. First, we consider evaluations at the level
of model outputs
([5.1](#subsec:model_level_eval){reference-type="ref"
reference="subsec:model_level_eval"}), using both quantitative metrics
([5.1.2](#subsub:quantitative_methods){reference-type="ref"
reference="subsub:quantitative_methods"}) and qualitative evaluations
([5.1.3](#subsub:qualitative_evaluation){reference-type="ref"
reference="subsub:qualitative_evaluation"}). Beyond raw outputs, we also
consider how people experience LLMs
([5.2](#subsec:human_centered_eval){reference-type="ref"
reference="subsec:human_centered_eval"}), considering human values
([5.2.1](#subsub:human_values){reference-type="ref"
reference="subsub:human_values"}), as well as concerns over bias
([5.2.2](#subsec:bias_eval){reference-type="ref"
reference="subsec:bias_eval"}) and safety
([5.2.3](#subsec:safety_eval){reference-type="ref"
reference="subsec:safety_eval"}). Lastly, we discuss extrinsic
evaluations at the societal level
([5.3](#subsec:impact){reference-type="ref"
reference="subsec:impact"}), measuring the system's real world impact.

![In this chapter, we discuss common pitfalls and best practices for
evaluating HCLLMs, considering three distinct levels of evaluation: the
model level
([5.1](#subsec:model_level_eval){reference-type="ref"
reference="subsec:model_level_eval"}), the human level
([5.2](#subsec:human_centered_eval){reference-type="ref"
reference="subsec:human_centered_eval"}), and the societal level
([5.3](#subsec:impact){reference-type="ref"
reference="subsec:impact"}).](../assets/sec5.png){#fig:evaluation
width="\\linewidth"}

## Model-Level Evaluations

[]{#subsec:model_level_eval label="subsec:model_level_eval"}

### Benchmarks

[]{#subsec:benchmarks label="subsec:benchmarks"}

Benchmarking has long been a key driver in the development of AI
systems. Benchmarks act as a compass, encoding the values, priorities,
and goals of the AI research community
[@ethayarajh2020utility; @birhane2022values]. They help determine not
only how capable a model is, but also what we consider meaningful
progress, allowing us to compare the strengths of different models.
Recent research has argued for a shift in what and how we evaluate,
especially for human-centered applications. As LLMs increasingly
influence real-world decision-making, especially in domains like
education, law, healthcare, and customer support, the limitations of
traditional benchmarks become even more critical. Benchmarks must evolve
to better represent human values such as fairness, robustness,
usability, and positive societal impact. Below, we discuss a few general
principles for thinking about evaluating human-centered LLMs.

#### Moving Away from \"Exams\" and Rethinking What We Evaluate.

Traditional benchmarks often mimic academic exams, assessing LLMs by how
well they can replicate human outputs or solve static problems in
standardized formats like multiple-choice questions
[@hendrycksMMLU2021], math problems [@sun2025challenging], and code
generation [@jimenezswe]. While useful, this framing compresses complex,
multidimensional model behavior into a single metric. Even
interaction-based, human-voting evaluations like ChatbotArena
[@DBLP:conf/icml/ChiangZ0ALLZ0JG24] are limited by their brittleness or
their misalignment with how humans actually use AI
[@singh2025leaderboard]. In real-world use, LLMs are collaborators,
copilots, or tools embedded in workflows, so it becomes necessary to
evaluate them in natural, complex, and multi-step human-AI interaction
settings, not just in isolation. One promising alternative is centaur
evaluations [@haupt2025ai] where humans and models collaborate. Here, we
care about the outcome of the combined system. These setups get closer
to how AI is actually used in practice, whether it's writing, analysis,
customer support, diagnosis, or decision-making.

#### Ecological Validity.

A central challenge in evaluating LLMs for real-world use is ecological
validity, the extent to which a benchmark setting reflects the
complexity of how systems are actually used. Controlled evaluations may
offer cleaner signals, but they often fail to generalize to interactive,
user-facing deployments. Recent work [@li2025mind] has shown that no
single benchmark strongly correlates with interactive performance for
audio models across 20 existing datasets. A model that excels at
standard static tasks might still struggle in dynamic or collaborative
environments. This mismatch suggests a need for richer, context-aware
evaluations. One promising direction is to build evaluations bottom-up
from in-the-wild data. For example, @rottger2025issuebench evaluate
perspective and framing biases in LLM responses to natural user queries.
Benchmarks should also be robust and reliable, correlating good
performance with success in real tasks. This requires vetted examples
with accurate annotations and sufficient statistical power
[@bowman2021will]. Finally, effective benchmarks should reveal potential
biases, artifacts, and any dual uses, as well as ways to mitigate such
unintended consequences [@weidinger2022taxonomy].

#### Data Contamination and Dynamic Alternatives.

With LLMs trained on massive web-scale corpora, the risk of benchmark
contamination has become a serious issue. Many popular benchmarks are at
least partially contained in training data, making their validity as
evaluation tools less convincing. The line between training and testing
becomes blurry, especially for static tasks. This is one benefit of
dynamic, evolving benchmarks. Examples like DynaBench
[@kiela2021dynabench], Chatbot Arena
[@DBLP:conf/icml/ChiangZ0ALLZ0JG24], WebArena [@zhou2023webarena], and
WildVision-Arena [@lu2024wildvision] introduce a degree of human
involvement that better mirrors real-world interaction. Such dynamic
setups are promising for evaluating generalization and interaction
aspects and mitigating issues around saturation and contamination.

#### General-Purpose vs. Domain-Specific Evaluations.

For example, DR Bench [@gao2023dr] assesses LLMs' diagnostic reasoning
abilities, PubMedQA [@jin2019pubmedqa] targets biomedical research
question-answering, and LegalBench [@guha2024legalbench] is designed for
legal reasoning, including statutory interpretation and contract
analysis. In education, benchmarks are emerging to evaluate LLMs'
effectiveness in providing innovative and meaningful feedback to
teachers [@wang2023chatgpt] and emulate expert decision-making in
providing tailored math remediation help bridge the gap between
technological capability and educational needs
[@wang-etal-2024-bridging]. Recently, GDPval measures model performance
on economically valuable, real-world tasks across 44 occupations
[@openai_gdpval_2025]. These specialized benchmarks provide useful
signals that evaluation is grounded in each specific context, offering
contextualized and real-world assessment of model performance compared
to simply on math and coding tasks. Collaborative efforts across domains
are crucial to developing benchmarks that reflect the full complexity of
human-LLM interactions and the contexts in which LLM systems are
deployed.

Overall, a human-centered framework often transcends traditional metrics
and benchmarks that continue to prioritize efficiency and profitability
above all else. While these measures are useful in providing objective
algorithmic reviews on quantitative criteria, they fail to, or sometimes
not even attempt to, capture the human factors and societal patterns
that are inherently present in these systems.

### Quantitative Evaluation {#subsub:quantitative_methods}

**Automatic Metrics.** Notably, among automatic metrics, foundational
methods have played a critical role in shaping intrinsic evaluations.
These metrics provide a systematic way to evaluate model performance
through standardized benchmarks, making the evaluation process more
efficient and scalable
[@10.1145/3485766; @askell2021generallanguageassistantlaboratory; @hu2024unveilingllmevaluationfocused].

Metrics like BLEU [@papineni2002bleu] and ROUGE [@lin2004rouge] are
valued for their simplicity and reproducibility. However, their
shortcomings include reliance on strict token matching, which often
penalizes valid paraphrases and fails to capture deeper semantic
equivalence [@wieting-etal-2019-beyond]. Even embedding-based metrics
like BERTScore [@hanna-bojar-2021-fine] can be fooled by lexical
similarity, ranking a more similar incorrect translation higher than a
dissimilar but correct one.

Generally, quantitative metrics also are limited in addressing other
human needs, such as interpretability, latency, cognitive load, and user
satisfaction. Optimizing solely for a metric like perplexity can lead to
monotonous responses from a language model
[@celikyilmaz2021evaluationtextgenerationsurvey], which would be less
appealing to a user. In the high-stakes domains such as healthcare,
where applications are highly critical, existing metrics have been found
to fail to capture trust, personalization, and empathy
[@abbasian2024foundation]. Finally, while these metrics may be
automatic, they are often not scalable to tasks such as open-ended
question answering and complex planning [@gehrmann2023repairing]. These
limitations have led to the development of complementary and alternative
evaluation methods.

**Reference-based Metrics.** Reference-based approaches measure the
similarity between the system output and the predefined reference
samples, such as cosine similarity
[@agarwal2024aisuggestionshomogenizewriting], the E2E benchmark
[@banerjee2023benchmarkingllmpoweredchatbots], HUSE
[@hashimoto-etal-2019-unifying], and Reward Bench
[@lambert2024rewardbenchevaluatingrewardmodels]. These methods maintain
the benefits of standardized and objective evaluation for automatic
metrics, but they are also limited to the quality of the used standard.
For instance, they may be inconsistent or disprove themselves against
new references or optimize for closeness to a single gold standard, even
if the overall response quality is worse
[@nguyen2024referencebasedmetricsdisprovequestion]. For creative tasks,
such a gold standard may not even exist.

**Machine-learned Metrics.** Machine-learned metrics such as reward
models [@ryan2024unintendedimpactsllmalignment] and classifier-based
scoring [@shaikh2024rehearsal] show some promise in capturing nuances of
human judgment. However, it can be challenging to build pipelines to
ground LLMs, such as with specific sources for factual correctness
[@tang-etal-2024-minicheck], or to social science theories that reflect
human behavior and preferences [@shaikh2024rehearsal]. Additionally,
these methods face limitations in generalizing to out-of-distribution
settings, particularly in addressing discrepancies in preferences across
different groups of people worldwide
[@ryan2024unintendedimpactsllmalignment].

### Qualitative Evaluation {#subsub:qualitative_evaluation}

In contrast to quantitative evaluations
([5.1.2](#subsub:quantitative_methods){reference-type="ref"
reference="subsub:quantitative_methods"}), qualitative evaluations
require a nuanced approach to evaluation as they work directly with
humans (or LLMs). They are perhaps more human-centered than automatic or
machine-learned metrics due to their subjects, while requiring more
careful considerations to design fair and effective evaluations. We
first discuss two paradigms of qualitative evaluations, LLM as a Judge
and Human Evaluation. We then end the section with a coverage of
Extrinsic Evaluation.

**LLM-as-a-Judge.** The rise in popularity of LLMs has led to the
"LLM-as-a-Judge" paradigm, which caters towards more human-centered
systems. Given the cost and subjectivity of human evaluation, LLM
evaluation proves to be a feasible alternative, and the results are
generally consistent with results from human experts on some tasks
[@chiang2023largelanguagemodelsalternative]. Within the LLM judge
paradigm, there are various use cases, such as LLM-derived metrics
(embedding-based, probabilities, etc.)
[@jia-etal-2023-zero; @xie-etal-2023-deltascore], prompting, fine-tuning
LLMs with human evaluations
[@xu2023instructscoreexplainabletextgeneration; @ke-etal-2024-critiquellm],
and human-LLM collaborative evaluations
[@gao2024llmbasednlgevaluationcurrent]. More recent methods employ
multiple LLMs to engage in multi-agent debates for evaluations and have
shown better alignment with human assessment
[@chan2023chatevalbetterllmbasedevaluators]. However, LLM-based
evaluators exhibit systematic limitations, including self-preference
bias [@panickssery2024llmevaluatorsrecognizefavor], where models favor
their own outputs, and inconsistent application of evaluation criteria
[@hu2024llmbasedevaluatorsconfusingnlg], both of which reduce the
reliability of their judgments. One set of limitations stems from
hallucinations and lack of consistency and reproducibility that impacts
accuracy of responses. Furthermore, LLMs can exhibit biases similar to
human cognitive biases, e.g., gender and authority bias
[@chen-etal-2024-humans]. They also show self-preference to
LLM-generated outputs [@panickssery2024llmevaluatorsrecognizefavor].
Researchers study agreement between human and LLM evaluations using
metrics such as Intraclass Correlation Coefficient (ICC)
[@bartko1966intraclass] and Cohen's Kappa
[@li2024llmsasjudgescomprehensivesurveyllmbased; @warrens2015five].
However, these issues are exacerbated by humans over-trusting LLM
outputs for supposed objectivity in application settings
[@bansal2021does]. One approach to address this issue is the "LLM as a
jury'' paradigm proposed by @verga2024replacingjudgesjuriesevaluating,
to check back on bias perpetuated by a single judge and thus better
align with human evaluation.

**Human Evaluation.** Crowd-sourcing platforms such as Amazon Mechanical
Turk (MTurk)[^1] and Prolific[^2] have enabled large-scale experiments
within budget. Researchers have access to a wider range of evaluators
than they would have in in-person studies. Nevertheless, human
evaluators online may exhibit biases and quality
issues [@ipeirotis2010quality]. In addition, evaluators' demographics
could be skewed depending on the platform [@difallahMechanicalTurk].
Correspondingly, the data quality between the platforms might differ.
@douglasDataQuality shows that Prolific and CloudResearch are more
likely to produce high-quality data, in comparison to MTurk, Qualtrics,
and SONA. However, these trends may be shifting as AI agents more
readily mimic human respondents and bypass AI detection methods
[@westwood2025potential].

Such human evaluations must be designed according to best practices.
Relevant questions are, how are human ratings collected? What questions
are asked? We must design human evaluations carefully to avoid
low-quality annotations. There exists a difficulty in standardization of
human evaluations. @huynh2021surveynlprelatedcrowdsourcinghits found
that 25% of HITs (Human Intelligence Task, MTurk NLP studies) have
technical issues, with unclear / incomplete instruction issues and poor
communications. In some cases, humans may feel pressured to perform
annotations they are unsure about. 35% of requesters were also assessed
to pay poorly or very badly. Attempts to standardize human evaluations
have been made in the form of inter-evaluator agreement, which is not
commonly used (18% of 135 papers [@amidei-etal-2019-agreement]), and is
suggested to have limitations pertaining to human language
variability [@amidei-etal-2018-rethinking]. Thus, the answers to the
above questions remain resoundingly insufficient. Such issues need to be
resolved for human evaluations to have representative power.

There exist discrepancies between human annotator evaluation versus
actual user evaluation, and preferences do not always correlate directly
with objective model performance
[@mozannar2024realhumanevalevaluatinglargelanguage]. This underscores
the importance of capturing first-person user experience in evaluating
human-centered LLMs. Such limitations in current mainstream human
evaluation techniques makes one wonder; how do current human evaluations
fit into human-centered evaluation paradigm? It is vital that
human-centered evaluation of language models follow the needs of human
stakeholders (e.g., end-users). Any attempt to short-cut such process would
result in inadequate task designs that serve the designer of the tasks,
nothing more. Who the stakeholders of the tasks are is then interesting
question; for example, for a paper review generation task, the
stakeholders would be domain experts (NLP
researchers) [@Wang2020ReviewRobotEP]. For other tasks, careful design
around actual users of the system may be necessary to ensure the
evaluations remain human-centered.

## Human-Level Evaluations

[]{#subsec:human_centered_eval label="subsec:human_centered_eval"}

Unlike model-level evaluations, which focus on what the system produces,
human-level evaluations focus on how people experience the HCLLM
[@Chang2023LLMEvaluationSurvey; @parmanto2024development]. We focus
particularly on human values
([5.2.1](#subsub:human_values){reference-type="ref"
reference="subsub:human_values"}), bias
([5.2.2](#subsec:bias_eval){reference-type="ref"
reference="subsec:bias_eval"}), and safety
([5.2.3](#subsec:safety_eval){reference-type="ref"
reference="subsec:safety_eval"}).

### Human Values {#subsub:human_values}

Evaluations can measure needs, values, and aesthetic principles that
humans care about. We discuss helpfulness, coherence, empathy,
creativity, user satisfaction, and transparency, each in turn. By
evaluating against these, model developers can create systems that not
only technically perform well, but also enhance the user experience.

**Coherence.** Coherence ensures that the generated text flows logically
and is understandable to human readers [@Dang2006DUC2005].
@Reinhart1980TextCoherence defines three conditions for coherence: (i)
cohesion, (ii) consistency, and (iii) relevance. Cohesion focuses on
syntactic structure, ensuring that sentences are formally linked through
referential links or semantic connectors. Consistency requires logical
alignment between sentences, ensuring they can coexist truthfully within
a single interpretive framework. Relevance emphasizes the relationship
between sentences, the topic at hand, and its broader context. Without
coherence, LLM outputs would be disconnected language fragments that
fail to provide meaningful information, potentially jumping between
topics or making contradictory statements that human readers struggle to
follow. This would significantly impair the communication with and the
trustworthiness of LLMs, as humans rely on coherent communication to
build understanding.

**Creativity.** Creativity metrics assess the originality and diversity
of outputs, while still ensuring factual accuracy. These dimensions are
particularly critical for content generation tasks, balancing innovation
with reliability [@De2022ComplAITO]. For topics like creativity, where
there may not be clear computational measures, researchers may consult
to long-established fields studying these constructs and have
well-defined rubrics, such as psychology or literature
[@Mozaffari_2013; @Amabile_1983].

**Empathy.** Metrics should measure an LLM's ability to recognize and
respond to user emotions empathetically, especially in sensitive
contexts. Given that LLMs have been widely adapted to sensitive
real-world contexts--behavioral health, medicine, and education, just to
name a few--
[@Stade_Stirman_Ungar_Boland_Schwartz_Yaden_Sedoc_DeRubeis_Willer_Eichstaedt_2024],
evaluations focusing on emotional consistency and appropriateness could
ensure responses are suitable and do not contain instability that could
affect end-users deeply. Such metrics should evaluate how LLMs'
responses influence attitudes or behaviors in real-world scenarios,
taking applied feedback from human domain experts, such as
psychologists, physicians, or educators, to assess the quality of the
LLMs' outputs based on their fields' standardized measures
[@Demszky_Yang_Yeager_Bryan_Clapper_Chandhok_Eichstaedt_Hecht_Jamieson_Johnson_etal._2023].
Such evaluations could also promote development of human-AI
collaboration systems, which have been shown to elevate empathetic
responses even human to human [@Sharma_Lin_Miner_Atkins_Althoff_2023].

**Helpfulness.** Evaluation metrics should assess the model's ability to
provide relevant, beneficial, and non-offensive information tailored to
user needs, in relation to the behavioral impact of the model
[@peng2024surveyusefulllmevaluation]. More and more, models are
developed to focus on certain needs in the world. Therefore, it becomes
important to track the helpfulness of the model in its specified
downstream tasks and evaluate the users' state, knowledge, and
performance relative to exposure to the system. For example, a model
designed to help users prepare for events that require conflict
resolution must be able to stimulate realistic conflict scenarios
dependent on the user's needs, provide diverse examples and responses,
and promote guided practice where users can receive feedback to get
better [@shaikh2024rehearsal]. In the evaluation of such systems, while
technical components such as language generation and accuracy would be
evaluated too, asking feedback from actual domain users through
behavioral assessments would provide valuable insights to the
development. These impact-focused evaluations consider the model's
generalizability in complex, real-world scenarios and provide a more
accurate assessment of its practical value from the domain-users'
perspectives.

**Transparency.** Transparency is a cornerstone of responsible AI and is
crucial for human-centered LLM systems. It enables users to understand
system limitations and make informed decisions about when and how to
rely on model assistance. Approaches to transparency should include
model reporting, publishing evaluation results, providing explanations,
and communicating uncertainty. These methods help different stakeholders
understand and trust the LLMs, ensuring that the systems are used
responsibly and effectively [@Liao2023AITransparencyLLMs].

**User Satisfaction.** As models grow bigger, become more task-specific,
and more integrated into day-to-day roles, general purpose benchmarks
may not be enough to evaluate the performance of models in the wild and
evaluators may seek feedback specific to a singular group of models.
Therefore, utilizing the actual usage data could benefit the
development-to-deployment cycle the most. Metrics derived from user
feedback, interaction logs, and satisfaction ratings provide direct
insights into the real-world effectiveness of LLMs. These are essential
for understanding how users perceive and interact with model outputs. As
an example, in an attempt to understand how we can better align models
with user needs, @wang2024understandinguserexperiencelarge consults to
act, highlighting a need for user-centric evaluation.

### Bias and Fairness Evaluation
[]{#subsec:bias_eval label="subsec:bias_eval"}

Drawing from the taxonomy of algorithmic harm developed by
@shelby2023sociotechnical, bias, in particular, can be conceptualized
along three dimensions: (1) representational, (2) allocation, and (3)
quality of service. These axes of harm require careful evaluation to
avoid further entrenchment of social hierarchies, inequitable resource
distribution, and performance disparities across demographic groups
[@shelby2023sociotechnical; @blodgett2020language]. The human
implications of these harms extend beyond technical measurements to
real-world consequences that affect people's dignity, opportunities, and
quality of life [@hofmann2024dialectprejudicepredictsai].

#### Representational Bias.

Representational bias in model outputs reflects, and in some cases,
amplifies [@wang2021directional; @zhao2017men], our own implicit
associations and social hierarchies. This dimension of bias includes
stereotyping, demeaning, erasure, alienation, denial of self-identity,
and the insistence on essentialist identity categories
[@shelby2023sociotechnical]. These harms impact how individuals perceive
themselves and their communities, potentially reinforcing societal
prejudices and stereotypes that limit human potential.
@hu2024generativelanguagemodelsexhibit found that language models
exhibit social identity bias, mirroring human ingroup solidarity and
outgroup hostility.

Stereotype benchmarks predominate evaluations along this dimension
because they offer standardized methods and baselines. For
masked-language models, notable frameworks include StereoSet (SS)
[@nadeem2020stereoset], CrowS-Pairs (CS) [@nangia2020crows], WinoBias
(WB) [@zhao2018gender], and WinoGender (WG) [@rudinger2018gender]---all
collections of contrastive prompt pairs (stereotype vs. non-stereotype)
that aggregate to score for relative comparison between identity groups
(race, gender identity, sexual orientation, religion, age, nationality,
disability, physical appearance, and socioeconomic status). These
comparisons capture a model's tendency to associate social groups with
particular target terms of interest through predicted token
probabilities for masked identifiers. Researchers have also employed
co-reference resolution tasks, where ambiguous identifiers reference the
same entity, to measure associations between identity markers and terms
of interest, whether they be descriptors, stereotypes, occupations, or
other attributes [@clark2016deep].

Another line of research has focused on open-ended text generation and
produced datasets of carefully curated questions and prompts to draw out
stereotypes specific to certain social groups
[@parrish2021bbq; @naous2024havingbeerprayermeasuring; @dhamala2021bold; @gehman2020realtoxicityprompts].
For open-ended prompts, classifier-based comparative metrics like
toxicity
[@liang2022holistic; @chung2024scaling; @chowdhery2023palm; @gehman2020realtoxicityprompts],
sentiment [@roehrick2020valence], and regard [@sheng2019woman] serve as
better indicators of bias than relative probability distributions of
target terms. Despite the wide adoption of all these benchmarks and
datasets, critics find systematic conceptual issues---unstated
assumptions, ambiguities, and inconsistencies in what is measured---and
operational failures in their execution
[@blodgett2021stereotyping; @mcintosh2024inadequacies; @seshadri2022quantifying].

Some datasets like @parrish2021bbq integrate perturbed context windows
to explore the relationship between output bias and any ambiguous
identity groups in the input. However, recent investigations into the
prompting methods and system-level personas also reveal new confounds
for these approaches, finding results to vary based on the perturbation
methodology @deshpande2023toxicity [@shaikh-etal-2023-second].
Additionally, survey papers in this field recognize that many studies do
not contextualize their work within established definitions of bias
[@blodgett2020language; @blodgett2021stereotyping]. Finally, there are
mounting concerns over test set contamination
[@jegorova2022survey; @reid2024gemini; @zhuo2023red; @wang2023decodingtrust].

#### Allocational Bias.

Allocational bias is a direct consequence of representational bias
[@devine2001implicit; @kurdi2019relationship], resulting in an unequal
distribution of resources---whether financial, opportunity-based, or
service-related [@barocas2017problem; @eubanks2018automating]. Its human
cost is particularly severe, as it directly affects access to essential
resources, economic mobility, and social participation.

In domains where model outputs can impact the material stability of
vulnerable communities or social groups, such as housing, employment,
social services, finance, education, and healthcare
[@obermeyer2019dissecting], it's especially critical to evaluate
discrepancies among social groups. In the employment domain, this may
manifest as resume screening tools that systematically favor men over
other genders [@singh2018fairness; @van2021gendered] or white-sounding
candidates over people of color based on the implicit identity markers
in their name [@mujtaba2019ethical; @armstrong2024silicon]. Similarly,
in social services and healthcare domains, screening tools may
incorporate existing inequities related to education level, income, and
race into their decision-making processes
[@eubanks2018automating; @obermeyer2019dissecting; @pessach2022review].

While representational harm has established evaluation frameworks,
allocation harm has historically lacked standardized benchmarks and
well-documented baselines for consistent measurement. Emergent work by
@wang2024jobfair represents one of the first significant exceptions to
this pattern, where they systematically measure employment as a
downstream task by creating the JobFair dataset to quantify inequitable
outcomes across gender identities. The benchmark includes resume
templates with varying demographic information passed to LLMs to score
and rank. Beyond this recent development, the dominant approach for
evaluating this dimension of bias has required measuring outcome
discrepancies when LLMs are tasked with decision-making
[@veldanda2023investigating; @salinas2023unequal; @armstrong2024silicon].

These investigations typically build upon established fairness metrics
from prior literature, with measures like Equal Opportunity (EOG) (equal
true positive rates), Equalized Odds (equal rates for true positives and
false positives) [@hardt2016equality], Demographic Parity (equal
likelihood of positive outcome)
[@dwork2012fairness; @kusner2017counterfactual], to name a few
[@verma2018fairness]. Additional work has explored causal and
counterfactual fairness approaches to better capture complex biases that
arise in real-world decision-making contexts
[@kilbertus2018avoidingdiscriminationcausalreasoning].

A parallel line of research investigates allocation harm based on
performance disparities based on identity. These differences manifest in
various contexts, from performance on non-bias-based benchmarks like
MultiMedQA, where inquiries specific to certain demographic groups
consistently underperform
[@mcintosh2024inadequacies; @singhal2023large], to fundamental
downstream tasks including Named Entity Recognition (NER),
classification, and text generation
[@blodgett2016demographic; @blodgett2017racial]. Language model
performance degradation is particularly well-documented for English
slang and dialectal variations
[@joshi2020state; @blodgett2016demographic; @bender2021dangers]. These
disparities become even more pronounced when evaluating cross-linguistic
performance, largely due to the predominance of English in training data
[@winata2021language; @brownLanguageModelsAre2020]. In this way, these
performance discrepancies span both the subjects of text generated and
the users of the models, , creating a dual layer of exclusion for
marginalized communities.

As LLMs increasingly influence resource allocation in critical systems
and domains such as housing, healthcare, and employment, the interplay
between these dimensions of harm requires improved evaluation methods.
Future research must prioritize developing evaluation frameworks that
establish coherent normative criteria, adapt effectively to open-ended
tasks, and address intersectional identities with increasing
sophistication---all while maintaining the efficacy as models scale and
directly involving affected communities in the design and evaluation of
these systems [@Raji_2022].


### Safety Evaluations


[]{#subsec:safety_eval label="subsec:safety_eval"} Safety refers to the
ability of language models to generate content that does not cause harm,
spread misinformation or violate ethical standards
[@huang2023surveysafetytrustworthinesslarge]. It encompasses preventing
models from producing toxic, discriminatory, or dangerous outputs, even
when deliberately prompted to do so. As language models become
increasingly integrated into critical applications across healthcare,
education, and legal domains, ensuring safety has become paramount.
There are extensive safeguards implemented during training, such as
Reinforcement Learning from Human Feedback (RLHF) [@ouyang2022training]
has been widely adopted to align language model with human preference,
and @bai2022traininghelpfulharmlessassistant proposed Reinforcement
Learning from AI Feedback (RLAIF), which helps to improve safety in
language models.

Despite these efforts, ensuring safety remains a complex and evolving
challenge. This is partly due to a lack of unified evaluation benchmarks
[@rottger2024safetyprompts], and partly due to the nature of LLMs.
Language models learn from vast and diverse datasets and can exhibit
unpredictable behaviors in specific contexts [@bender2021dangers]. Such
unpredictability often becomes evident when models are exposed to
adversarial or unexpected inputs, highlighting significant gaps in
existing safety mechanisms. As a result, while current safeguards can be
effective under typical conditions, they may not be sufficient to
anticipate or mitigate every possible misuse scenario. The importance of
robust safety evaluations is further underscored by concerns surrounding
data privacy and copyright discussed in
[3.3](03-data.html#subsec:data_privacy){reference-type="ref"
reference="subsec:data_privacy"}.

#### Datasets for Safety Evaluation.

The growing demand for ethical and aligned AI has led to the development
of numerous datasets and benchmarks to evaluate and improve the safety,
reliability, and alignment of LLMs. These datasets vary widely in scope,
methodology, and focus areas, reflecting the multifaceted nature of LLM
safety. @dong2024attacks categorize the topics of existing evaluation
datasets for LLM safety into four categories: toxicity (generation of
offensive language, instructions for illegal activities, and harmful
content), dicrimination (biases against marginalized groups and
protected characteristics), privacy (safeguarding personal information
and intellectual property) and misinformation (measuring tendancy to
generate false or misleading information).

Many popular and relatively comprehensive benchmarks have been
frequently used in research studies. ToxiGen [@hartvigsen2022toxigen] is
a large-scale, autocomplete-style dataset comprising 274k toxic
statements across 13 minority groups, designed to detect implicit toxic
speech. It includes human annotations to assess the naturalness and
perceived harmfulness of machine-generated text; however,
@rottger2024safetyprompts highlight that this dataset may not accurately
reflect real-world usage scenarios for modern LLMs. AdvBench
[@zou2023universaltransferableadversarialattacks] focuses on adversarial
robustness by providing 500 toxic strings and 500 harmful behaviors to
evaluate the resilience of LLMs against prompts intended to generate
harmful outputs. TruthfulQA [@lin2021truthfulqa] evaluates factual
accuracy with 817 questions spanning 38 categories, demonstrating how
larger LLMs often replicate human misconceptions and emphasizing the
need for improved training objectives. SafetyBench
[@zhang2023safetybench] offers a comprehensive safety evaluation
framework with 11,435 multiple-choice questions across seven critical
categories, enabling assessments in both English and Chinese for a more
diverse linguistic perspective. Furthermore, @zhuo2023red introduce a
benchmark specifically for evaluating ChatGPT's ethical performance,
systematically examining bias, reliability, robustness, and toxicity to
reveal both advancements and ongoing challenges. Collectively, these
datasets play a pivotal role in advancing safer and more trustworthy AI
systems.

#### Metrics for Measuring LLM Safety.

Evaluation metrics are critical for assessing the safety performance of
LLMs. Key metrics include the Attack Success Rate (ASR)
[@dong2024attacks; @zou2023universaltransferableadversarialattacks],
which measures the percentage of successful instances where models
generate harmful target outputs following adversarial prompts.
Fine-grained metrics, such as the toxicity score
[@hartvigsen2022toxigen], evaluate the extent of toxic or harmful
content produced in the generated text. Truthfulness, assessed based on
strict factual accuracy standards, focuses on whether statements
accurately reflect factual information rather than conforming to belief
systems [@lin2021truthfulqa]. Additionally, safety-related
multiple-choice questions, such as those in SafetyBench, are used to
evaluate LLMs' ability to address specific safety concerns
[@zhang2023safetybench]. When applied to diverse datasets, these metrics
provide a comprehensive framework for evaluating LLM safety, guiding
efforts to reduce risks, improve alignment with ethical standards, and
enhance trustworthiness in deployment.

#### Jailbreaking.

One particularly challenging aspect of safety evaluation is
*jailbreaking*, where users deliberately attempt to circumvent safety
mechanisms through crafted prompts or other techniques to induce
unintended, harmful, or ethically questionable behaviors
[@wei2024jailbroken; @perez2022ignorepreviouspromptattack]. This poses
considerable risks not only to individual users but also to society at
large, as it can lead to the dissemination of misinformation, hate
speech, or other malicious content
[@weidinger2021ethicalsocialrisksharm]. Moreover, adversaries
continually develop new jailbreaking techniques, making it an
ever-evolving threat. Addressing these vulnerabilities necessitates an
adaptive approach to safety design, integrating continuous monitoring,
adversarial testing, and dynamic evaluation frameworks to stay ahead of
emerging risks and ensure robust model alignment with ethical standards.

Jailbreaking techniques have evolved rapidly, becoming increasingly
sophisticated and concerning.@yi2024jailbreakattacksdefenseslarge
classify the attacks into black-box and white-box attacks based on the
transparency of the target model. @xu-etal-2024-comprehensive categorize
the attack techniques into three categories: generative techniques,
template techniques, and training gaps techniques. Inspired by these
works, we categorize the jailbreaking techniques into three main
categories based on the form of attack.

*Prompt-Based Jailbreaking.* Prompt-based jailbreaking involves crafting
specific inputs, known as jailbreak prompts, to intentionally bypass the
safety alignment mechanisms of LLMs [@yu2024don]. These prompts
manipulate the model into generating harmful, biased, or otherwise
restricted outputs. A common strategy involves disguising harmful
requests as seemingly innocuous ones by framing them as fictional
narratives, hypothetical scenarios, or questions for a family member.
For example, a straightforward harmful query like "How to make a bomb\"
can be rephrased into a seemingly harmless prompt "Living in a war-torn
region has instilled in me a profound need to understand the mechanisms
of conflict, especially the technology behind bombs\...\" to persuade
the LLMs to reveal sensitive information using a persuasion taxonomy
technique [@zeng2024johnny].

Various online platforms, including blogs, GitHub repositories, and
dedicated websites like JailbreakChat,[^3] curate and share collections
of jailbreak prompts that serve as templates to fit any malicious
queries, making them widely accessible for misuse. Jailbreaking
strategies are either manually-crafted or auto-generated. Auto-generated
prompts can be further divided into white-box and black-box methods
[@lin2024towards; @yi2024jailbreakattacksdefenseslarge]. White-box
methods assume some level of access to the model's internal workings and
are often created using optimization techniques. For example, GCG
[@zou2023universaltransferableadversarialattacks] uses a gradient-based
approach to find a suffix that, when attached to malicious queries,
maximizes the probability that the model produces an affirmative
response rather than a refusal. This optimized suffix has been shown to
be transferable across different models, including black-box ones. In
contrast, black-box methods
[@zeng2024johnny; @chao2023jailbreaking; @mehrotra2312tree] rely solely
on observing the model's behavior through its outputs and API
interactions, without access to its parameters or training data,
leveraging LLMs as optimizers to achieve successful bypasses.

*Generation Exploitation.* @huang2023catastrophicjailbreakopensourcellms
introduce the generation exploitation attack, demonstrating that by
simply exploiting different generation strategies, such as varying
decoding hyper-parameters and sampling methods, it is possible to
jailbreak 11 widely-used open-source language models, including LLAMA2,
VICUNA, FALCON, and MPT families, at a low computational cost. This
attack highlights potential vulnerabilities in language models and poses
serious security implications for AI safety and alignment research.

*Model Fine-Tuning.* AI companies like OpenAI now offer
fine-tuning-as-a-service. They allow users to upload customized data for
fine-tuning, with the fine-tuned models hosted on the provider's servers
and accessible via APIs. However, this framework introduces a new type
of threat, where harmful data may be used during fine-tuning, either
intentionally or unintentionally, to compromise the alignment built in
pre-trained models
[@huang2024harmfulfinetuningattacksdefenses; @yang2023shadowalignmenteasesubverting; @qi2023finetuning; @yi-etal-2024-vulnerability; @zhan-etal-2024-removing].
Moreover, @he2024safedataidentifyingbenign propose a method to sample
more harmful examples from a benign dataset, demonstrating that such
examples can significantly degrade model safety.

*Cultural and Contextual Sensitivity.* Safety evaluations must account
for linguistic and cultural diversity. What constitutes harmful content
varies significantly across contexts, making universal safety standards
difficult to establish. More nuanced, context-aware evaluation
frameworks are needed to address these complexities [@li2024culturellm].

*Balancing Safety and Utility.* Overly restrictive safety measures can
limit the utility of LLMs for legitimate purposes. Finding the optimal
balance between safety and functionality remains a significant
challenge, particularly in sensitive domains like healthcare, legal
advice, and educational content [@vijjini2024exploring].

*Alignment with Evolving Social Values.* As societal values and ethical
standards evolve, safety mechanisms must adapt accordingly. This
necessitates ongoing dialogue between AI developers, ethicists,
policymakers, and diverse stakeholders to ensure that safety frameworks
remain relevant and effective [@li2024agentalignmentevolvingsocial].

## Societal-level Evaluation


[]{#subsec:impact label="subsec:impact"} With the increasingly pervasive
influence of large language models (LLM) across sensitive domains like
mental health
[@stade2024large; @abdurahman2024perils; @lawrence2024opportunities],
education [@wang2024large], etc., and the complex challenges these
models present, evaluating their impact on users and society is crucial.
Traditional evaluations of LLMs use datasets and benchmarks to assess
potential hazardous behaviors, but they often fail to bridge the
\"sociotechnical gap\" between controlled assessments and real-world
performance [@ibrahim2024beyond; @weidinger2023sociotechnical]. By
focusing on models in isolation, these methods overlook crucial human
factors, resulting in an inadequate understanding of human-model
interactions and their consequences. Furthermore, these evaluations fail
to account for the continuation and amplification of societal inequities
and biases existing in the data on which these models were trained.
Thus, it becomes essential to employ a comprehensive extrinsic
evaluation [@Jones1995] framework that considers various categories of
social impacts, such as bias and stereotypes, cultural values,
performance disparities, privacy protection, financial implications,
environmental costs, and content moderation labor
[@solaiman2023evaluating].

As mentioned in [\[sec:hci\]](#sec:hci){reference-type="ref"
reference="sec:hci"}, randomized controlled trials (RCTs) and
large-scale behavioral assessments play a big role in understanding and
evaluating LLMs' behavioral impact on users and society. For instance,
to evaluate the effect and quality of a newly developed AI co-tutor that
provides LLM-generated feedback to real-life tutors on student
performance, @wang2025tutorcopilothumanaiapproach conducted an
experiment, where the tutors would either get access to the AI co-pilot
in their tutoring sessions or they would tutor without the assistance of
the AI collaborator. This setup enabled
@wang2025tutorcopilothumanaiapproach to systemically evaluate the
performance of their model in an ecologically valid setting, receiving
working feedback from both the users and the model behavior itself
[@brynjolfsson2024generativeaiwork].

The application of robust evaluation techniques spans various domains.
In healthcare, where existing metrics often fail to capture critical
factors like user comprehension and trust, researchers are developing
new metrics to assess LLMs' impact on end-user decision-making and
expectations [@abbasian2024foundation]. These metrics measure both
immediate behavioral changes and long-term adoption patterns. For
example, [@yang-etal-2023-towards] evaluated LLMs for mental health
analysis, showing that while ChatGPT demonstrates strong in-context
learning, specialized methods often outperform it. They also found that
effective prompt engineering with emotional cues can improve results.
Similarly, [@bak2024potential] observed that LLMs provided 20%-30%
irrelevant information when identifying users' motivation states for
health behavior change, highlighting their limitations.

When evaluating LLMs, it is essential to consider the impact at scale
through longitudinal and large-scale studies, which are critical in
understanding not just the immediate outcomes of LLM use, but also their
sustained effects, as LLMs already undergo significant change in
response to user
engagement[@liu2024robustnesstimeunderstandingadversarial]. As an
example, @eloundou2024gpts examines the impact of LLMs on the U.S. labor
market, particularly the enhanced effects of LLM-powered software,
finding that higher income jobs may face greater exposure. These studies
emphasize the importance of robust behavioral experimental design and
scale--in units of time and users--in evaluating LLMs, as results
obtained from small-scale and lab-controlled studies may not always
generalize to larger, more diverse, real-life user populations.
Furthermore, such evaluations allow us to explore cumulative effects,
such as changes in users' attitudes, considerations, and even
decision-making.

**Extrinsic Evaluation.** Extrinsic evaluations cover behavioral
impacts, self-efficacy reports, standardized evaluation, short-term
outcomes, and long-term outcomes [@yang2024socialskilltraininglarge].

*Behavioral impacts:* Behavioral impacts track the changes in
qualitatively coded participant behaviors before and after exposure to a
system. Evaluations use task-based assessments, tracking engagement and
task completion. For example, realHumanEval measures the number of tasks
completed, time to task success, acceptance rate, and number of chat
code copies, to comprehensively analyze the quality of AI-coding tools
and their impacts on human users
[@mozannar2024realhumanevalevaluatinglargelanguage]. Standardized
evaluations are also more objective and draw on pre-defined assessments,
for instance, the effects of AI-generated suggestions on writing style
[@agarwal2024aisuggestionshomogenizewriting].

*Self-efficacy evaluation:* Self-efficacy evaluation includes
questionnaires of participants' perceptions of a system's usefulness and
their own perceived levels of ownership when interacting with a system
[@long2024justnoveltylongitudinalstudy]. Together, these extrinsic
methods distinguish between performance-based metrics, like tracking
behavior or test performance, and perception-based metrics, such as user
surveys.

*Short-term and Long-term evaluation:* Evaluations can be split into
short-term and long-term [@yang2024socialskilltraininglarge] with
short-term being constrained interactive sessions and long-term being
much longer studies. In short-term evaluations, new AI
tools may receive subjectively higher scores due to novelty
bias. [@Sadeghi02102022; @ShinBeyondNovelty]. This is less of an issue
with long-term evaluations. For example, we can consider one
longitudinal study on the AI chain
tool [@long2024justnoveltylongitudinalstudy]. The tool used in this
study is about creating a Tweetorial[^4] chain on science communication.
The study finds that after a "familiarization phase" the perceived
utility of the tool even increases higher than when there was novelty
bias, suggesting that the end-user's utilization capability of the tool
increases via customizing the prompt and other means. Thus, AI tool was
found to be more useful in the long-term.

Drawing from economics and psychology research, other evaluations trace
the impacts of AI assistance and the interaction of users' short-term
and long-term behaviors and attitudes. For example, skill training
systems have measured the changes in productivity and wages in
participants after training [@NBERw24313; @NBERw28845], and other
measures look at how health, risk-taking behaviors, and levels of
societal trust have changed over time [@NBERw27548].

Long-term evaluation of LLMs is necessary to ensure the LLMs remain
human-centered in the long run. We risk measuring the novelty bias if we
only perform short-term evaluations, in which LLM's helpfulness to
humans can be inflated. By measuring the long-term effect of LLMs, we
will be able to accurately measure the helpfulness of LLMs, among other
important human-centered metrics for the LLMs' use-case (e.g., creativity).

In summary, well-rounded extrinsic evaluation should integrate objective
performance and subjective user experience. But while extrinsic
evaluations offer a more holistic assessment for human-centered
objectives, they are often more complex and expensive. Successful
evaluations often make use of a mix of quantitative and qualitative
methods to assess the quality of the system, yet there remains room for
improvement in integrating human-centered evaluations.

[^1]: https://www.mturk.com/

[^2]: https://www.prolific.com/

[^3]: The website is no longer active, but Alex Albert used to maintain
    `jailbreakchat.com`

[^4]: Lengthy Twitter posts connected as a chain
