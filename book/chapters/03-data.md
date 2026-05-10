---
prev-chapter: "Home"
prev-url: "https://rlhfbook.com/"
page-title: Data for HCLLMs
search-title: "Chapter 3: Data for HCLLMs"
next-chapter: "NLP for HCLLMs"
next-url: "https://rlhfbook.com/"
---
# Data for HCLLMs

[]{#sec:data label="sec:data"}

Data is central to the language modeling paradigm, just as it has been
throughout the history of machine learning [@halevy2009unreasonable].
Every stage of language model development, from pretraining to
evaluation and deployment, depends on the availability of massive text
corpora [@sun2017revisiting]. Critically, the scale, diversity, and
quality of this linguistic data can determine a model's downstream
utility for users [@kaplan2020scaling; @liu2024makes; @zhou2023lima].
Data quality and quantity can quickly become bottlenecks
[@10.5555/3692070.3694094], limiting progress in AI
[@longpre2024consent]. Thus one of the most urgent challenges in AI is
in identifying diverse and representative sources of data.

From the human perspective, data is more than the fuel behind AI
progress. Data is a dynamic reflection of lived human experience. It
reflects the people, institutions, cultures, histories, and social
contexts that produce it. In this sense, data is never neutral. Rather,
data encodes viewpoints and values [@dotan2020value], assumptions and
biases [@paullada2021data], and even political and social structures
[@scheuerman2021datasets; @miceli2022studying; @capel2023human]. The
origins of such data may be the subject of legal claims and privacy
concerns, and its content may be highly personal or sensitive
[@bender2018data]. To understand the human impact of LLMs, it becomes
necessary to consider the human origins of data that shapes our models,
particularly in pre-training, instruction tuning, and alignment
(Figure [1](#fig:data){reference-type="ref" reference="fig:data"}).

In the first subsection of this chapter, we examine the
**(§[\[subsec:data_provenance\]](#subsec:data_provenance){reference-type="ref"
reference="subsec:data_provenance"}) *provenance*** of data used to
develop LLMs. We ask where this data comes from, who produced it, under
what conditions it was produced, and how it was transformed throughout
this process. In this way, we recognize how data encodes implicit
values, perspectives, and cultures that shape LLM behavior. From here,
we are positioned to understand human-centered concerns around
**(§[\[subsec:data_representation\]](#subsec:data_representation){reference-type="ref"
reference="subsec:data_representation"}) *representation and bias***,
how the data's origins systematically skews, misrepresents, and erases
the perspectives of underrepresented groups, leading to representational
and allocational harms. While rich community and personal data may be
used to mitigate some of these harms, we consider issues around
**(§[\[subsec:data_privacy\]](#subsec:data_privacy){reference-type="ref"
reference="subsec:data_privacy"}) *consent and ownership***. Finally, we
consider some of the biggest data challenges facing LLM developers
today, and how proposed solutions like
**(§[1.4](#subsec:synthetic_data){reference-type="ref"
reference="subsec:synthetic_data"}) *synthetic data*** account or fail
to account for the human-centered objectives we have outlined.

## Data Provenance


[]{#subsec:data_provenance label="subsec:data_provenance"}

![This chapter focuses on the human origins of data
(§[\[subsec:data_provenance\]](#subsec:data_provenance){reference-type="ref"
reference="subsec:data_provenance"}), and how data encodes perspectives
and values that impact HCLLM outcomes, from representation and bias
(§[\[subsec:data_representation\]](#subsec:data_representation){reference-type="ref"
reference="subsec:data_representation"}) to consent and ownership
(§[\[subsec:data_privacy\]](#subsec:data_privacy){reference-type="ref"
reference="subsec:data_privacy"}). In particular, we consider
***pre-training data***, ***instruction-tuning data***, and ***alignment
data***.](../assets/03_Data.png){#fig:data width="\\linewidth"}

**Data provenance** [@longpredata], also called **dataset genealogy**
[@denton2021genealogy], is the record of a dataset's origins, history,
and transformations throughout its lifecycle. An understanding of data
provenance is critical for achieving transparency in LLM development
[@bommasani2023transparency]. Without transparent data practices, it
becomes difficult to predict and understand why LLMs leak private
information
[@19_kandpal2022deduplicating; @bubeck2023sparksartificialgeneralintelligence],
violate copyrights [@carlini2021extracting], or perpetuate social biases
[@denton2021genealogy]. But with data provenance, stakeholders become
more equipped to audit models [@mokander2024auditing] and tackle these
human-centered concerns. In this section, we will investigate the
provenance of data used for pre-training, instruction-tuning, and
aligning LLMs. In particular, we ask where data comes from, who produced
it, and under what conditions it was produced.

### Pretraining Data

[]{#subsec:data_provenance_pretraining
label="subsec:data_provenance_pretraining"} **Data Sources.** The
provenance of LLM pretraining data is often complex, layered, and
opaque. Unlike the small, curated datasets of traditional machine
learning, LLM pretraining corpora tend to be huge, multi-trillion token
aggregations across heterogeneous and potentially noisy sources. These
sources traditionally include web text, digitized books, and open-source
code repositories
[@wenzek2019ccnetextractinghighquality; @raffel2020exploring; @soldaini2024dolmaopencorpustrillion; @devlin-etal-2019-bert].
Although different model developers use different data mixtures, most
incorporate an open web crawl that at least partially intersects the
[*Common Crawl*](https://commoncrawl.org/). The Common Crawl contains
monthly snapshots of "open web" --- partial samples of machine-crawlable
sites reached from seed URLs that were initially crowdsourced in 2008
[@baack2024critical]. This is not a random sample of the internet. Large
web crawls like this favor wikis, news sites, blogs, and other
user-generated content platforms, which are generally multilingual, but
heavily skew towards English [@baack2024critical]. Much of this data has
been found to be socially undesirable, with a high prevalence of hate
speech and sexually explicit content
[@luccioni2021whatsboxpreliminaryanalysis]. The data can also reify
social, cultural, and political biases
[@naous2024havingbeerprayermeasuring; @feng2023pretrainingdatalanguagemodels; @navigli2023biases].
Finally, this web-scale data inextricably encodes the structural biases
of the web itself, where the majority of content is produced by an
active minority of users, and these users overly represent Western,
Educated, Industrialized, Rich and Democratic (WEIRD) populations
[@baeza2018bias].

**Quality Filtering.** Because open web data is noisy, redundant,
low-quality, and often socially undesirable, model developers use
classifiers or heuristics
[@chen2021evaluatinglargelanguagemodels; @penedo2023refinedweb; @rae2021scaling; @soldaini2024dolmaopencorpustrillion]
to filter their pre-training corpora for unique and high-quality
documents that are information dense and free from toxicity or
personally identifiable information [@longpre2024responsible]. By
filtering pre-training data in this way, researchers can train safer
models with better performance at lower computational costs
[@du2022glamefficientscalinglanguage; @rae2021scaling; @albalak2024dataselection].
The distributions of these filtered corpora are shaped by sampling
decisions, including the filters used to determine document quality.
These quality filters often systematically exclude both communities and
discursive topics [@lucy-etal-2024-aboutme]. For instance, toxicity
classifiers often exhibit racial and linguistic bias
[@sap2019risk; @dodge-etal-2021-documenting]. Quality filters trained on
Wikipedia and OpenWebText tend to favor text from wealthy, educated,
urban areas [@gururangan-etal-2022-whose].

After language identification
[@conneau2019cross; @laurenccon2022bigscience] and deduplication
[@lee2022deduplicatingtrainingdatamakes], model-based filtering is one
popular quality filtering approach. For example, perplexity-based
methods filter noisy documents that appear as highly surprising to a
much smaller language model. CCNet
[@wenzek2019ccnetextractinghighquality] was constructed as a subset of
the Common Crawl, filtered with 5-gram language models that the authors
trained on Wikipedia data for each target language. They use a fastText
classifier [@joulin2017bag] for language identification and run each
deduplicated document through the appropriate 5-gram model to compute
perplexity, filtering based on a heuristic and language-specific
perplexity threshold. Disconcertingly, this pipeline effectively filters
out minority dialects and low-resource languages
[@albalak2024dataselection] for which language ID is unreliable
[@caswell-etal-2020-language; @kudugunta2023madlad], or the perplexity
model is overfit to only a small corpus
[@feng2023pretrainingdatalanguagemodels; @lucy-etal-2024-aboutme].
Perplexity-based filtering will retain text that matches the language
distribution the filtering model was fit to. When the standard is
Wikipedia, filtering will primarily preserve Standard American English
in the third person, written in a neutral, semi-formal and broadly
readable register, with clear declarative sentences. Another idea is to
prompt existing LLMs to estimate the quality of pre-training data
zero-shot using some manually-written definition of high quality data
[@sachdeva2024train; @wettig2024qurating; @penedo2024fineweb]. This was
the approach used for Llama-3 [@llama3modelcard]. However,
manually-written definitions are brittle and may not encompass
task-specific or user-specific notions of LLM utility
[@held2025optimizing]. A third idea is to fine-tune a small model like
fastText [@joulin2017bag] as a binary quality classifier. The binary
classifier was the approach used by DataComp for Language Models [DCLM;
@li2024datacomp], as this resulted in models with higher scores on
general benchmarks like MMLU [@hendrycksMMLU2021]. However, methods like
this are prone to overfitting on the training set, the construction of
which itself reflects and reifies the values of model developers.

Another popular filtering mechanism is to use content heuristics like
domain name blacklists, and toxic keyword dictionaries, which were used
to construct the Colossal Clean Crawled Corpus (C4)
[@xue2021mt5; @raffel2020exploring]. The English C4 was found to skew
heavily towards Wikipedia articles, patents, and United States news
articles, such as the New York Times
[@dodge-etal-2021-documenting; @elazar2024s]. Most documents in the
corpus had been published after the year 2011. The multilingual variant,
mC4 [@xue2021mt5], represents 101 identified languages, but many of
these languages are under-represented [@snaebjarnarson2022warm]. For
example, compared to 2.7T tokens of English, mC4 contains only 600,000
tokens of Javanese [@aji2022one]. Data for these lower-resource
languages is also much noisier than that of the English subset
[@kreutzer2022quality; @van2024language].

Many pretraining corpora aggregate documents from a variety of sources.
Popular examples include the Pile [@gao2020pile800gbdatasetdiverse],
RedPajama [@weber2024redpajama], and Nemotron-CC [@su2025nemotron],
which contain 300B, 1T, and 7T tokens respectively, sampled from the
Common Crawl, as well as academic texts, books, coding, medical, and
legal documents [@biderman2022datasheet; @weber2024redpajama]. Over half
of the documents in these data mixes are duplicated at least once, and
some of these contain personally identifiable information like email and
IP addresses, as well as toxic language [@elazar2024s].

### Instruction-tuning Data

[]{#subsec:data_provenance_instruct
label="subsec:data_provenance_instruct"}

Compared to the origin story of pre-training data, the provenance of
instruction-tuning datasets is relatively well-known. Data is typically
aggregated by a single organization for the purpose of fine-tuning a
model to follow instructions. Therefore the provenance of instruction
dataset development resembles the distribution of model developers, over
half of whom originate in the US or China [@held2023material]. Notable
examples include Google's FLAN [@chung2024scaling], AI2's Natural
Instructions [@mishra2021crossnaturallanguage], Stanford's Alpaca
[@alpaca], and Cohere's Aya corpus [@singh2024aya].

The instruction aggregation process often involves selecting a diverse
range of tasks over which are constructed well-formatted prompt-output
pairs. Some organizations may opt to annotate these pairs entirely from
scratch, like in the @databricks2023dolly15k Dolly-15k. There are
ethical and scientific benefits in such cases where data is sourced with
explicit consent, attribution, and compensation. However, this is not
the norm, especially since doing so demands significant human labor.

In many cases, instructions are sourced automatically from evaluation
benchmarks via templates [@chung2024scaling; @longpre2023flan], which
may be further translated [@muennighoff2022crosslingualBLOOMZ] or
restructured using tertiary models. The templates themselves typically
have a human origin. For example, the Natural Instructions dataset
[@mishra2021crossnaturallanguage] was sourced from annotation guidelines
that the benchmark developers constructed to onboard crowdworkers.
Sometimes, humans also write templates from scratch, especially in the
early days of UnifiedQA [@khashabi2020unifiedqa] and FLAN
[@wei2021finetuned], and in low-resource settings like the multi-lingual
Aya corpus [@singh2024aya]. However, much of the data construction
pipeline is automated. This trend is growing as instruction-tuning
datasets are generated synthetically. For example, the instructions used
to fine-tune Stanford's Alpaca model [@alpaca] were distilled from
GPT-3.5, a larger model which was itself instruction-fine-tuned. This
approach, called self-instruction tuning
[@wang2023selfinstructaligninglanguagemodels], has been adopted in a
range of more recent work
[@peng2023instructiontuninggpt4; @li2023ottermultimodalmodelincontext].
As we will discuss in
§[1.4](#subsec:synthetic_data){reference-type="ref"
reference="subsec:synthetic_data"}, the use of synthetic data for
self-instruction tuning complicates data provenance, and may exacerbate
the human-centered concerns raised in this chapter.

### Alignment Data

[]{#subsec:data_provenance_posttraining
label="subsec:data_provenance_posttraining"} Additional datasets are
used for *model alignment*, or the process of training more helpful and
less harmful models via supervised fine-tuning, preference tuning, and
reinforcement learning from human feedback (RLHF)
[@askell2021generallanguageassistantlaboratory]. Since alignment data is
what produces models that are useful to humans, it constitutes a major
force behind the sudden proliferation in the number of LLM users
worldwide.

Since the notion of helpfulness or harmfulness is ambiguous and varies
with different cultures and contexts, one might expect a commensurate
heterogeneity in both the source and format of alignment data
[@ethayarajh2024behavior]. This is generally not the case. With respect
to the format, many alignment datasets assume a Bradley--Terry model of
pairwise human preferences. Datasets like Anthropic's HH-RLHF
[@bai2022traininghelpfulharmlessassistant], OpenAI's InstructGPT
[@ouyang2022training], and Peking University's PKU-SafeRLHF
[@NEURIPS2023_4dbb61cb] couple a user prompt with a pair of model
responses: one preferred and one dispreferred. With respect to data
sources, many preference judgments come from a very small pool of
annotators, sometimes within the organization itself. For example,
Peking University hired 28 internal annotators to construct
PKU-SafeRLHF, and Anthropic's internal research team similarly hired and
trained a small group of contractors to construct HH-RLHF.

Crowdsourcing and citizen science can serve to democratize the process
of collecting alignment data. One drawback of these approaches is
sampling bias, which may favor researchers, AI enthusiasts, and
individuals from industrialized nations. Chatbot Arena
[@DBLP:conf/icml/ChiangZ0ALLZ0JG24], also known as LMArena, is one
example of a public web platform with open-user participation in which
volunteers engage with pairs of anonymous models and provide preference
feedback in the standard binary format. The project was initiated at the
University of California Berkeley in 2023, and covers 96 languages,
although the vast majority are in English. OpenAssistant Conversations
[@kopf2023openassistant] is a similar crowdsourcing effort, initiated by
the German non-profit LAION in 2022. Over 13k volunteers contributed
alignment data in 35 different languages, particularly in English (50%),
German (20%), and Spanish (10%). Of these annotators, 89.1% identified
as male, with a median age of 26. These clear demographic biases above
will skew the values, perspectives, and interests represented by this
data.

To address issues of demographic bias, some dataset developers
intentionally target underrepresented demographics in their recruitment
efforts. For example, the PRISM Alignment Dataset [@kirk2024the] is an
academic project initiated at the University of Oxford, where the
developers recruited Prolific workers from 33 underrepresented
countries. The Meta Community Alignment Dataset [@zhang2025cultivating]
is a similarly-motivated multilingual preference dataset in which its
15k participants were recruited from five countries on YouGov. Still,
there remain limitations in recruiting diverse populations from
crowdwork platforms, which have limited global coverage
[@rinderknecht2025daily; @douglasDataQuality; @palan2018prolific].

Synthetic data is an emerging trend among subsections in this chapter,
and it is largely motivated by the need to scale AI beyond what human
annotation labor can support
[@casper2023openproblemsfundamentallimitations; @santurkar2023opinionslanguagemodelsreflect].
Some LLM developers have considered synthetic data in the alignment step
as well. Variants of this approach include Constitutional AI
[@Bai2022ConstitutionalAH] and Reinforcement Learning from AI Feedback
[RLAIF; @lee2024rlaif]. Both approaches shift critical alignment
decisions from data contributors to more centralized authorities:
namely, the LLM-as-a-Judge, and those who prompt it. For example, in
Constitutional AI, models judge their own output against the standards
of a human-written constitution, and then re-write a better,
constitution-aligned response. Anthropic's original 2022 constitution
was sourced from Western liberal-democratic sources like the United
Nations Declaration of Human Rights, the OECD, and Google's AI
Principles. These frameworks employ individualist, rights-based moral
reasoning [@haidt2012righteous], which may not represent other global
ethical traditions, or incorporate the voices of pluralistic user bases
[@sorensen2024roadmap].

## Data Representation, Bias and Ethics

[]{#subsec:data_representation label="subsec:data_representation"}

The story of LLM training data is a story about whose voices become
computationally legible and whose are overwritten or erased. In
§[\[subsec:data_provenance\]](#subsec:data_provenance){reference-type="ref"
reference="subsec:data_provenance"}, we considered how the story of data
is shaped by its sources, filtering decisions, annotation pipelines, and
synthetic generation practices. Imbalances or biases in the provenance
of LLM pre- and post-training data can exacerbate representational,
allocational, and quality-of-service harms for those who use these
models. Representational harms include stereotyping, denigration, and
misrecognition, when LLMs perpetuate and amplify distorted and harmful
portrayals of personal identities and social groups
[@Blodgett2021SociolinguisticNLP]. Allocational harms arise when LLMs
reinforce or amplify inequality in the distribution of opportunities and
resources [@barocas2017problem; @eubanks2018automating].
Quality-of-service harms involve performance disparities across
different user groups, which may cascade into both representational and
allocational harms. For further discussion on how to define and measure
these harms, see
§[\[subsec:bias_eval\]](#subsec:bias_eval){reference-type="ref"
reference="subsec:bias_eval"}.

Sociotechnical harms become harder to diagnose when data provenance is
incomplete. Without visibility into the linguistic, cultural, and
geographic origins of the data, as well as the filtering and curation
pipelines, researchers cannot identify: (1) why the model stereotypes
certain voices, (2) why specific groups are absent from generated
outputs, or (3) how certain narrative tropes became dominant. We will
briefly discuss the relationship between data provenance and each of
these harm outcome categories, as well as data-based mitigation
strategies.

### Quality-of-Service Harms

Quality-of-service harms are disparities in model utility for users from
different sociodemographic groups [@shelby2023sociotechnical]. These
disparities are often rooted in the composition and curation of data as
discussed in
§[\[subsec:data_provenance\]](#subsec:data_provenance){reference-type="ref"
reference="subsec:data_provenance"}. Pre-training data scraping, quality
filtering, instruction-tuning templates, and alignment data collection
tend to over-represent native English speakers from wealthy, Western
nations and under-represent the language and perspectives of
marginalized communities. As a result, LLMs introduce quality-of-service
harms for individuals from these communities [@shah2020predictive].

LLM performance degrades for speakers of non-standard language varieties
or dialects on a wide range of tasks
[@kantharuban2023quantifying; @joshi2025natural], from text
classification [@lwowski2021risk] and machine translation [@ahia2023all]
to question answering [@ziems2023multi; @fleisig2024linguistic] and
conversational AI [@artemova2024exploring]. This inequitable
distribution of utility to LLM users can result in allocational harms
like inequitable wages and quality of life, especially as LLMs are
integrated into the workplace [@shao2025future] and become *general
purpose technologies* [@eloundou2024gpts]. Quality-of-service bias also
contributes to the representational harm of erasure, and may derive in
part from representational biases.

### Representational Harms

LLMs demonstrate representational harms when they propagate negative or
skewed representations of social groups, including cultural
misrepresentation, stereotypes, essentialist language, and erasure
[@chien2024beyond]. Representational harms can derive from pre-training
data [@10.1145/3682112.3682117], not only from its explicitly harmful,
stereotypical, and toxic language
[@luccioni2021whatsboxpreliminaryanalysis], but also from implicitly
biased language [@Caliskan2016SemanticsDA; @navigli2023biases], framing
effects [@feng2023pretrainingdatalanguagemodels], and the sparsity of
socioculturally representative data
[@naous2024havingbeerprayermeasuring]. Data quality filters exacerbate
racial and linguistic biases that skew pre-training data away from
in-group perspectives in favor of unrepresentative and misinformed
out-group perspectives [@wang2025large]. Post-training data can further
induce mode-collapse, effectively flattening their representational
distributions to portray groups one-dimensionally
[@bisbee2024synthetic; @durmus2024towards; @rottger2024political]. This
kind of distributional flattening is a form of *essentializing* that is
particularly harmful for groups historically portrayed as
one-dimensional [@wang2025large].

Unsurprisingly, LLMs are known to generate harmful stereotypes in
question-answering [@naous2024havingbeerprayermeasuring], machine
translation [@ghosh2023chatgpt], and open-ended generation
[@dhamala2021bold]. These issues are only exacerbated when the prompts
are written in non-standard dialects, which may trigger demeaning or
condescending responses from models [@fleisig2024linguistic].
LLM-simulated personas also collapse into stereotypical caricatures
[@cheng-etal-2023-marked; @cheng-etal-2023-compost; @gupta2023calm].
These simulations systematically misrepresent, flatten, and essentialize
the perspectives of underrepresented groups based on protected
characteristics like age, gender, and disability [@wang2025large].

### Allocational harms

Allocational harms are disparities in individuals' access to material
resources like jobs, housing, credit, healthcare, childcare, education,
and transportation [@cyberey2025prevalent]. When LLMs are embedded in
decision-making systems, they can introduce, amplify, or otherwise
reinforce allocational disparities, in part as a result of
representational biases in the training data
[@sen2025missing; @chien2024beyond]. In pre-training, skewed
representations can lead models to encode assumptions about who is
qualified, creditworthy, employable, or deserving of services
[@mehrabi2021survey]. During post-training, alignment data privileges
the annotators' norms of professionalism, risk, and appropriate behavior
[@conitzerposition2024], which appear in downstream allocational biases
as follows.

LLMs used in hiring decisions can be more likely to recommend
less-prestigious jobs to speakers of marginalized dialects
[@hofmann2024ai]. In content moderation, LLMs are prejudiced against
speakers of African American English [@sap2019risk]. In automated exam
scoring, LLMs show disparate performance for students with backgrounds
not represented in training data [@schaller2024fairness]. And more
broadly, LLM decisions are biased against underrepresented groups across
domains such as business (i.e., funding a startup), finance (i.e.,
approving a credit card), relationships (i.e., resolving conflicts), law
(i.e., issuing a passport), science (i.e., approving a research study),
and the arts (i.e., awarding a filmmaking prize)
[@tamkin2023evaluating; @levy2024gender].

### Mitigating Harms

Mitigating sociotechnical harms requires interventions across the data
pipeline. The first step is to establish transparent data provenance
through documentation practices like *Datasheets for Datasets* and *Data
Statements* [@gebru2021datasheets; @bender2018data]. By explicitly
recording the linguistic, demographic, and geographic composition of
datasets, as well as filtering and annotation decisions, making it
easier to diagnose representational gaps and biases. Model cards and
system cards further extend this transparency to downstream users by
documenting intended use, performance disparities, and known limitations
[@model_cards]. Recent work argues that provenance-aware documentation
should include not only source descriptions but also transformation
histories, like filtering, deduplication, and synthetic augmentation
[@scheuerman2021datasets].

With transparent data provenance, a second mitigation step is to involve
stakeholders in the process of data creation and diversification, using
participatory methods [@vaughn2020participatory], following
§[\[subsub:participatory\]](#subsub:participatory){reference-type="ref"
reference="subsub:participatory"}. With community-level organization, it
is possible to develop rich data resources for low-resource languages
and underrepresented communities
[@orife2020masakhane; @heidt2025walking]. However, diversification alone
may prove insufficient without governance structures that prevent
extractive data practices and ensure ongoing community oversight
[@benjamin2023race].

A third mitigation approach is to collect learnable data from user
interactions with LLMs at the individual level. Personalized alignment
methods may be considered in which individual preference data is
collected from user interactions and used to shape subsequent model
behavior through prompt-based
[@hebert2024persomapersonalizedsoftprompt], retrieval-based [@lamp], or
alignment-based methods [@ryan2025synthesizeme]. For more discussion on
this direction, see
§[\[subsec:personalization\]](#subsec:personalization){reference-type="ref"
reference="subsec:personalization"}.


## Consent and Ownership


[]{#subsec:data_privacy label="subsec:data_privacy"}

The data used to pre- and post-train LLMs may include sensitive personal
information
(§[\[subsec:data_provenance\]](#subsec:data_provenance){reference-type="ref"
reference="subsec:data_provenance"}). Such personal data may actually
help LLM systems become more capable, useful, and proactive. For
example, LLMs can infer from users' hidden behaviors, personal habits,
and broader computer use patterns, what the user might need before they
even make a request [@shaikh2025gum]. However, the use of private or
personal data comes with an array of legal and ethical challenges
[@1_privacy_yan2024protecting; @2_subramani2023detecting]. Sensitive,
private, or copyrighted information can be inadvertently leaked or
reproduced without authorization, further complicating compliance with
laws and regulations
[@6_zhang2024privacy; @31_khan2022subjects; @32_wachter2019data]. Here,
we will consider these concerns regarding the ownership of data.

### Data Privacy Considerations

Data privacy is broadly defined as the ability of individuals to control
their personal information. Privacy leaks can occur in data, either when
sensitive personal information is explicitly encoded, or when this
information can be inferred
[@1_privacy_yan2024protecting; @kshetri2023cybercrime; @staab2024memorizationviolatingprivacyinference].
In the former setting, LLMs can memorize personally identifiable
information from training data and expose these details at inference
time
[@1_privacy_yan2024protecting; @3_memory_carlini2019secret; @staab2024memorizationviolatingprivacyinference].
Attackers can exploit vulnerabilities in LLMs through methods such as
backdoor attacks, membership inference attacks, and model inversion
attacks, which can extract sensitive information embedded in the model
during pre-training or fine-tuning
[@1_privacy_yan2024protecting; @3_memory_carlini2019secret]. For
instance, @carlini2021extracting demonstrated that it is possible to
recover individual training examples, including names and phone numbers,
by attacking the language model. Similarly, @4_zhao2024measuring
revealed that LLMs could generate infringing content when prompted with
partial information from copyrighted materials. These risks are more
severe in larger models with more parameters and when longer contexts
are used in prompting, making it increasingly challenging to address
these vulnerabilities effectively for current LLMs
[@karamolegkou2023copyright; @carlini2021extracting; @11_carlini2022quantifying].

Additionally, legal frameworks play a critical role in governing the use
of personal and copyrighted data. Since 2018, the General Data
Protection Regulation (GDPR) in the European Union has mandated data
minimization, consent requirement, and the "Right to Erasure.\" This can
be re-interpreted to apply to AI systems, though with limitations after
the data collection process [@neel2023privacy; @32_wachter2019data].
More regulations and protocols are needed to comply with ethical
obligations. Copyright law introduces another layer of complexity.
Copyright law grants creators exclusive rights to use and distribute
their work, with specific exceptions. Under §107 of the United States
Copyright Law, the fair use doctrine permits limited usage of
copyrighted materials without permission, typically for purposes such as
commentary, research, or information extraction, but not for verbatim
reproduction [@karamolegkou2023copyright]. With the increasing influence
of LLMs, the use of online data has come under heightened scrutiny;
justifications under principles like \"Legitimate Interests\" for
personal data and \"Fair Use\" for copyrighted content are being
questioned more rigorously [@33_franceschelli2022copyright]. Notably,
companies such as OpenAI, Stability AI, and Microsoft have faced various
legal challenges, including consumer privacy lawsuits and copyright
infringement claims, underscoring the growing contention surrounding
privacy and copyright issues in AI development
[@news_1; @news_2; @news_3; @news_4].

### Proactive vs. Reactive Privacy Strategies

Adopting a proactive approach to privacy is essential. Rather than
deferring mitigation until after model training, privacy considerations
should inform every stage of data collection and curation. This includes
implementing privacy-preserving data collection protocols, robust
anonymization techniques, and consent-based frameworks from the outset.
For instance, it is critical to obtain consent and minimize sensitive
information collection, employ more tools to detect and remove
personally identifiable information, and use more sophisticated data
anonymization techniques to better protect aganst privacy leakage
[@1_privacy_yan2024protecting; @2_subramani2023detecting]. Consent-Based
Data Collection should be adopted in scenarios like web scraping to
respect individual's rights [@2_subramani2023detecting]. Web
architectures like SOLID [@sambra2016solid] and Consent Tagging
[@zhang2023tag] aim to streamline consent acquisition
[@6_zhang2024privacy].

For more reactive privacy strategies after the data collection stage,
various techniques have been proposed to mitigate these issues. Data
cleaning methods aim to remove or generalize sensitive information from
datasets before training
[@brownLanguageModelsAre2020; @ouyang2022training; @bai2022traininghelpfulharmlessassistant; @19_kandpal2022deduplicating].
Federated Learning approaches decentralize the training process to
enhance privacy by keeping data local and aggregating updates instead of
sharing raw data
[@20_chen2023federated; @21_yu2023federated; @22_xu2024fwdllm; @24_hoory2021learning].
Differential Privacy methods extract useful statistical information from
datasets without revealing individual data by introducing controlled
random noise or applying aggregation techniques
[@24_hoory2021learning; @25_du2021dp; @26_li2021large; @27_shi2022just; @28_wu2022adaptive].
Additionally, Knowledge Unlearning techniques selectively forget or
remove sensitive information from models to mitigate privacy risks
[@5_seyitouglu2024extracting; @29_chen2023unlearn; @30_eldan2023s].

### Open Challenges in Data Privacy

Currently, privacy risks persist across the entire LLM lifecycle,
encompassing not only model-centric issues but also human-centered
factors. From the data side, stronger anonymization techniques and tools
capable of identifying memorized personal information must keep up with
LLMs' evolving capabilities
[@staab2024memorizationviolatingprivacyinference; @2_subramani2023detecting].
It should also be cautioned when scaling HCLLMs, as discussed in
§[\[subsec:scaling\]](#subsec:scaling){reference-type="ref"
reference="subsec:scaling"}, that risks from memorization also increase
with scale if repeated data are in the training stage
[@hernandez2022scaling]. In addition, the complexities of obtaining
consent, especially in scenarios involving third-party or inaccessible
data sources, underscore the need for more robust frameworks to ensure
transparent data sourcing and meaningful user control
[@6_zhang2024privacy]. HCI researchers now also advocate for improved
LLM interaction paradigms, a deeper understanding of user mental models,
and systems that enable end-users to reclaim ownership over their
personal data [@li2024human]. Despite significant progress in addressing
data privacy concerns, much of the research focuses on well-known LLMs
with relatively small scales. In contrast, recently released models with
larger parameter sizes have received less attention due to the
challenges posed by their scale, data transparency issues, and the
lagging development of privacy-preserving technologies
[@1_privacy_yan2024protecting]. Overall, greater efforts are needed to
enhance legal frameworks, strengthen regulatory oversight, and advance
research and technology to better safeguard privacy and copyright in the
era of LLMs that developers, users, and policymakers can jointly share.

## Expanding Data Sources: Synthetic and Non-Traditional Data {#subsec:synthetic_data}

### Synthetic Data

We often lack high-quality, diverse, and privacy-compliant data
[@almeida_2024_sdg_part1]. Filtering methods
(§[\[subsec:data_provenance\]](#subsec:data_provenance){reference-type="ref"
reference="subsec:data_provenance"}) can filter out as much as 90% of
raw web text data from the Common Crawl. To replace this data, synthetic
generation is one solution employed in Nemotron-CC [@su2025nemotron] and
other popular pre-training corpora. Synthetic data generation preserves
individuals' confidentiality, replicating only the statistical
properties of real datasets without retaining any personally
identifiable information. LLM-generated synthetic text can also serve as
fine-tuning and evaluation data [@vongthongsri2025synthetic], where it
is invaluable for addressing class imbalances [@moon2024synaug],
especially in domains like healthcare [@guo2024generative] where data is
sensitive, and mathematical reasoning where gold examples are costly to
produce [@chan2024balancing].

#### Methods Used to Generate Synthetic Data.

Even medium-size language models can effectively expand pre-training
corpora by paraphrasing existing data [@maini2024rephrasing]. Moreover,
LLMs can effectively generate entirely new content from scratch,
including textbooks for pre-training [@gunasekar2023textbooks] and
instruction-tuning data for post-training
[@wang2023selfinstructaligninglanguagemodels]. Procuring high-quality
synthetic data with LLMs typically involves three stages: generation,
curation and evaluation [@long2024llms]. Generation often involves
prompt engineering to elicit LLM responses in the required format. This
involves using strategies such as task definition, conditional
prompting, in-context learning, and multi-step generation, which address
context limitations and degradation over reasoning steps
[@long2024llms; @wang2024qstar].

The generated data often contains noise or corrupted samples due to
hallucination, and is generally curated using sample filtering and label
enhancement techniques. Sample filtering could involve simple
heuristic-based strategies or leverage the advanced
language-understanding capabilities of LLMs to generate confidence
scores for data points based on quality and reject samples with low
scores [@chung-etal-2023-increasing]. Label enhancement strategies could
include human inspection and annotation of low-confidence samples. These
techniques are described in
[\[subsec:data_provenance_pretraining\]](#subsec:data_provenance_pretraining){reference-type="ref"
reference="subsec:data_provenance_pretraining"}.

After curation, the generated data must be evaluated for several
components, including the statistical similarity between synthetic and
real data, impact on model performance, and ensuring that synthetic data
preserves essential patterns and relationships- @xia2024advancing
capture these requirements in their proposed fidelity, utility, and
privacy framework.

#### Making Synthetic Data More Human-Centric.

A human-centered approach to synthetic data creation should explicitly
incorporate human values, perspectives, and audits at all stages of
development, from generation to curation and evaluation. First,
generation should serve to reflect authentic human interactions and
preferences when real data collection proves slow or costly
[@synthetic_hci]. Rather than simply increasing dataset sizes, synthetic
data should contain realistic social interactions between individuals
with diverse personalities and backgrounds. This requires persona
alignment
(§[\[subsec:steerability\]](#subsec:steerability){reference-type="ref"
reference="subsec:steerability"}) or role-play in which the LLM portrays
a consistent identity [@tseng2024two], possibly simulating a person from
a particular sociodemographic background [@lutz2025prompt], or an agent
with a role, like a tutor or counselor
[@li2024steerability; @samuel2024personagym; @shanahan2023role]. Persona
alignment has been used to generate synthetic dialogues
[@prodigy; @stargate] and preference data
[@castricato2024personareproducibletestbedpluralistic].

At the curation stage, stratified sampling should reflect real-world
distributions along known axes of variation, such as opinions and
preferences [@sorensen2025spectrum]. Finally, robust human-in-the-loop
validation and auditing is essential. Human annotators and experts can
review synthetic outputs, flag problematic patterns, and iteratively
refine generation procedures. One major concern is that LLMs may
reproduce biases and harms present in their training data, leaking
private information or reinforcing existing social inequalities.
@compare compare LLM-generated datasets with human-annotated benchmarks
and highlight ethical concerns related to disparities in task
performance and representational coverage. @chen2024unveiling identify
several failure modes in LLM-generated query--answer pairs, including
instruction-following errors. To mitigate these risks,
@chen2024unveiling propose unlearning techniques to improve the
reliability of synthetic queries. To preserve privacy, @privacy_perserve
propose decentralized frameworks designed to reduce the likelihood of
sensitive information exposure during data synthesis. These steps will
ultimately enhance the quality, fairness, and usability of synthetic
data to align with ethical standards and user expectations.

### Non-traditional Data {#subsec:non_traditional}

Recent progress in LLM research has shown the value of using
non-traditional data to make models more human-centered. This discussion
focuses on three primary areas. The first is multimodal data, which
allows LLMs to work with inputs like speech, images, and touch. The
second is human-AI interaction data, such as user feedback, edits, and
eye-tracking, which helps improve how well LLMs understand and respond
to user needs. Lastly, human-human interaction data uses examples of
real human interactions to teach LLMs how people communicate, enabling
models to better handle context, complex emotions, and relationships.

#### Multimodal Data.

Recent research has sought to expand Large Language Models to enable
multimodality [@yin2024survey; @li2024multimodal], significantly
enhancing human-LLM interaction by allowing systems to process and
respond to a diverse range of input formats beyond text, such as speech
[@rubenstein2023audiopalmlargelanguagemodel; @huang2024audiogpt], sound
[@zhang-etal-2023-video; @huang2024audiogpt], vision
[@achiam2023gpt; @zhang-etal-2023-video; @li2023ottermultimodalmodelincontext; @pmlr-v202-li23q; @fu2024a],
and tactile data [@fu2024a; @yu2024octopi], creating richer and
increasingly human-like communication channels. By integrating multiple
sensory modalities, AI can better mirror human communication, which
could further improve Human-AI interaction. For instance, a multimodal
AI assistant could analyze a user's tone of voice, facial expressions,
and spoken words to assess emotional states
[@zhang2024llms; @cheng2024emotion], tailoring its responses
accordingly. Recent works also explore integrating human physiological
data (e.g. EEG, BVP) with LLMs to enhance empathic human-AI interaction
[@dongre2024integratingphysiologicaldatalarge]. In applications such as
education, healthcare, and accessibility, multimodality fosters
inclusivity by accommodating users with diverse needs
[@yildirim2024multimodal; @belyaeva2023multimodalllmshealthgrounded; @chang2024worldscribe].
Ultimately, multimodal AI systems bridge the gap between machine
efficiency and human communication, making interactions more seamless,
adaptive, and human-centered.

#### Human-AI Interaction Data.

Expanding the scope of human-AI interaction data has opened new pathways
for enhancing Large Language Models through both supervised fine-tuning
and reinforcement learning with human feedback (RLHF). For example,
Vicuna is trained with massive user-shared conversations with GPT to
achieve high quality outputs [@vicuna2023]. Another valuable type of
human-AI interaction data is human edits, where users adjust the outputs
of LLMs to better match their desired results. This data can be
leveraged to fine-tune LLMs for improved preference alignment
[@shaikh2024show] or to extract user preferences more effectively
[@gao2024aligningllmagentslearning]. Beyond text-based interaction data,
untraditional modalities such as eye-gaze signals offer additional
interaction. Eye-gaze data, in particular, provides a real-time,
implicit feedback mechanism that enhances context awareness and
alignment with user intent
[@konrad2024gazegpt; @prokofieva2019eye; @engel2023project; @lopez2024seeing].
These gaze-based interactions have been shown to improve multi-modal
conversational understanding and can be leveraged in RLHF workflows to
refine LLM outputs dynamically [@lopez2024seeing]. The integration of
gaze data into multi-modal frameworks would help create richer,
contextually adaptive systems, fostering more intuitive, personalized,
and effective interactions across diverse applications.

#### Human-Human Interaction Data.

Real world human-human interaction data captures the nuances of human
communication, including implicit cues, turn-taking dynamics, and
diverse conversational contexts. Such data has the potential to improve
LLMs by fostering deeper understanding of relational and situational
context, thereby enabling models to generate responses that feel more
natural, empathetic, and contextually appropriate. Recent advancements
demonstrate how mining teacher-student interaction data, such as
dialogue transcripts and collaborative problem-solving sessions, can
align LLM outputs with human cognitive and emotional patterns, which
allow LLMs to address complex, interdisciplinary challenges in fields
such as education, psychology, and social science by emulating and
learning from authentic human interaction styles
[@wang-etal-2024-bridging; @XU2024100325; @wang2023sight; @wang2024educonvokit].
By leveraging human-human interaction as an informative data source, we
can expand the capacity of LLMs to foster meaningful, human-centered
interactions in diverse real-world applications.

The integration of multimodal data, human-AI interaction data and
human-human interaction data can all help LLMs more closely approximate
the complexity of human communication, in turn making models more usable
and reliable across high-impact domains like healthcare, education, and
social services. As we exhaust traditional text data sources, recent
efforts, such as MINT-1T, a multimodal text and image interleaved
open-source dataset generated by [@awadalla2024mint1t] will be
fundamental to advance the performance of frontier models.
