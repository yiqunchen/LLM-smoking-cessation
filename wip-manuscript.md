Can LLMs Read Between the Cravings? Lessons from Predicting the Effectiveness of Smoking Cessation Messages



Some key messages and findings:
Zero-shot LLMs promise to transform how we do behavioural research – we are not quite there yet. 
Supporting evidence: across models we tried and even some prompt engineering: we get ~35% average accuracy across the 3 fields; this is much lower than running a regression model on some training data (high 40%)
It does provide some prior information
Structured prompt engineering tailored to persona/behavior science (“digital twin”) is still critical to get good performance:
Evidence 1: The digital twin prompting approach gives us ~50% across the three text based fields 
Evidence 2: Using the CBT/ACT strategy seems to help for some messages and participant splits
There is still a large variation in terms of model performance for pure LLM based; we should always have some pilot validation to check performance before going all in LLM directly
Evidence 1: all the good results from a simple regression
Evidence 2: whether you use grok4/gpt-4o vs 4omini vs deepseek actually still makes a difference (not every LLM is built the same way for your application)
Think about a list of best practices of using LLM for rating prediction problems in smoking behavior research:
One is to think about baseline model you have (regression model? Random guess?)
Consider the level of noise (signal to noise ratio) in your outcome: 80% is reproducible in one study - we are plateuing
Metrics: 5 class accuracy; or 3 accuracy  (does it really matter if i couldnt distinguish very poor vs poor or very good versus good)



Set up for Jasmin’s experiments 
10 messages each (splitting 7 for training, 3 for testing); used all the characteristics (existing one)
Categorizing into CBT and ACT (coping and quitting selection; design & content); Jasmin has uploaded the message categorization here. Have to specify coping and quitting for it to help consistently 




2. Methods
2.1 Study design and population
We conducted a secondary analysis of data from an online panel study in which young adult smokers evaluated smoking cessation messages. The parent study was designed to assess the effectiveness of evidence-based digital health interventions grounded in Cognitive Behavioral Therapy (CBT) and Acceptance and Commitment Therapy (ACT) frameworks for promoting smoking cessation among diverse young adults. CBT-based messages emphasize strategies for managing cravings by redirecting attention through specific tasks or behaviors that serve as distractions. In contrast, ACT-based messages encourage individuals to acknowledge and accept cravings without judgment while maintaining focus on the present moment. Both therapeutic approaches have been extensively investigated in prior behavioral intervention research and provide empirically supported foundations for digital cessation programs.
The study population consisted of 301 young adults aged 18 to 30 years, recruited through an online Qualtrics research panel. Eligible participants had smoked at least 100 cigarettes in their lifetime, reported smoking every day or on some days, and were either currently attempting to quit or intending to quit within the next month. During the trial, participants evaluated 124 smoking cessation messages that were developed by the research team based on evidence from prior studies  and paired with image content obtained from free stock photo websites (Pexels and Unsplash). Each participant rated 10 randomly selected messages, including five based on CBT strategies and five based on ACT strategies, resulting in a total of 3,010 message ratings.
Participants completed an online survey in which they rated each message across four dimensions: perceived quality of content (“How would you rate the content (that is, the words and meaning) of this message?”); (2) perceived quality of design (“How would you rate the design (that is, how the message looks) of this message?”); (3) perceived message support for coping with smoking urges (“How helpful would this message be to support you in coping with a smoking urge or craving?”); (4) perceived message support for quitting smoking (“How helpful would this message be to support you in quitting or reducing smoking?”). Responses for content and design were rated on a five-point Likert scale ranging from Very poor to Very good, while responses for coping and quitting support were rated on a five-point Likert scale ranging from Not at all helpful to Extremely helpful. In addition to message ratings, participants completed questions assessing their sociodemographic characteristics (age, sex, sexual or gender minority status, race/ethnicity, and education level), current smoking behaviors (daily smoking status, time to first cigarette, and readiness to quit), and psychological flexibility, which was measured using the Acceptance and Action Questionnaire–II (AAQ- II).
2.2 Models
To explore how LLMs can support the design of more effective smoking cessation interventions, we implemented a series of models to predict different dimensions of smoking cessation message ratings.
2.2.1 Benchmark models
Because human behavior data are often noisy, we used two benchmark models to contextualize prediction accuracy: a random-guess model and a regression model. The random-guess model represents the lower bound of performance by assigning each prediction randomly from a uniform distribution across all possible smoking cessation message ratings. The regression model represents a traditional statistical baseline, providing a structured comparison that captures linear relationships between predictors and the outcome. Specifically, we used a xxx model, with xxx as predictors and xxx as the outcome variable.
2.2.2 Generic LLM models
We evaluated a set of generic LLM configurations that varied by example inclusion, feature inclusion, and response format. Specifically, we tested:
 (1) a zero-shot model incorporating all individual characteristics (sociodemographic, smoking behaviors, and psychological flexibility);
 (2) a zero-shot model using a subset of selected individual characteristics (selection based on xxx);
 (3) a few-shot model with all individual characteristics;
 (4) a few-shot model with selected individual characteristics;
 (5) a model producing continuous numerical ratings (1–5) for calibration using all individual characteristics; and
 (6) a continuous-rating model using only selected individual characteristics.
2.2.3 Hybrid machine learning–LLM models
In the hybrid machine learning– LLM models, we aimed to combine the strengths of machine learning and LLMs. Machine learning models were used exclusively with individual characteristics to capture structured relationships between person-specific features and message evaluations. In contrast, LLMs were applied solely to message text and associated ratings, leveraging their capacity to model complex semantic patterns and linguistic nuances that are difficult to represent in traditional variables. This separation enables an integrated framework that reflects both structured and unstructured sources of information. Specifically, we tested a combination of different machine learning models and LLM-derived feature representations, including:
(1) xxx ML model & LLM PCA
(2) …
2.2.4 Digital twin
Unlike the previous models which focused on population-level or feature-based prediction, the digital twin approach generated a personalized computational representation of each participant by integrating their unique characteristics, behavioral patterns, and message evaluations into a personalized profile. This framework allows the model to approximate how a specific individual might respond to a given message, thereby capturing heterogeneity in intervention responses and supporting more adaptive, person-centered intervention design. Specifically, we first built an individual profile that integrated each participant’s characteristics (including sociodemographic, smoking behaviors, and psychological flexibility), along with their prior message exposures and corresponding responses. The resulting profile served as a personalized context for predicting ratings of new messages for that individual. To examine the effects of different profile configurations, we evaluated a set of digital twin model variants:
(1) full-feature digital twin (50/50 split): included all individual characteristics and used a 50%/50% split of messages between the individual profile and the testing set;
(2) selected-feature digital twin (50/50 split): included only selected individual characteristics (selected based on xxx) with the same 50%/50% message split;
(3) full-feature + feedback digital twin (50/50 split): incorporated all individual characteristics and participant feedback (i.e., additional information on person-specific cognitive appraisals) on messages within the profile, using a 50%/50% message split for profile versus testing;
(4) domain-informed digital twin (variable split): included all individual characteristics and messages categorized as CBT or ACT based, using varying proportions of messages in the profile versus testing set (50/50, 70/30, and 90/10 splits) and different LLM models (GPT, Grok, DeepSeek).
It is important to note that our analyses focused on three of the four rating dimensions: content, coping, and quitting. The design dimension, which primarily related to the visual composition of messages’ pairing images, was excluded from the main analyses because incorporating image embeddings into the prompts did not improve prediction accuracy and substantially increased computational complexity. Nevertheless, the vision-enabled prompt, along with other prompt implementations, are available in our GitHub repository (link), and additional results are provided in the Supplementary.
Results

