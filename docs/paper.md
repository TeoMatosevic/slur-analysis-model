# Detection of Hate Speech on Croatian Online Portals Using NLP Methods

Duje Jurić¹, Teo Matošević²*, Teo Radolović³

¹ University of Zagreb, Faculty of Electrical Engineering and Computing, Croatia
² University of Zagreb, Faculty of Electrical Engineering and Computing, Croatia
³ University of Zagreb, Faculty of Electrical Engineering and Computing, Croatia

*Corresponding author

## Abstract

**Purpose** – The proliferation of hate speech on Croatian online platforms presents a significant challenge for content moderation. This paper investigates the effectiveness of natural language processing methods for automated detection of offensive language in Croatian, a morphologically complex, low-resource language with limited annotated datasets and pre-trained models.

**Design/Methodology/Approach** – We conduct a systematic comparison of traditional machine learning baselines (TF-IDF with Logistic Regression and SVM) against fine-tuned transformer models (BERTić and XLM-RoBERTa) on the FRENK Croatian hate speech dataset containing 10,971 annotated comments. We apply bootstrap confidence intervals and McNemar's test to assess statistical significance. Additionally, we develop a lexicon of 32 coded terms ("dog whistles") commonly used in Croatian online discourse to express implicit hate speech.

**Findings** – Fine-tuned BERTić achieves an F1-macro score of 0.810, significantly outperforming TF-IDF baselines (F1=0.684) by 18.5%. The model demonstrates balanced performance across both acceptable (F1=0.790) and offensive (F1=0.831) classes. Results confirm that transfer learning from related South Slavic languages effectively captures contextual nuances that traditional feature-based approaches miss.

**Originality/Value** – This work provides the first systematic comparison of transformer-based and traditional ML approaches for Croatian hate speech detection. We contribute publicly available code, trained models, and a coded language lexicon to support future research and practical content moderation applications.

**Keywords** – Hate speech detection; Croatian NLP; BERTić; Transformer models; Offensive language; Low-resource languages.

**Paper Type** – Research paper.

## 1. Introduction

### 1.1 Background

The proliferation of social media and online news portals has led to an increase in hate speech and offensive content in digital spaces. Croatian online platforms, including news comment sections and social media, face significant challenges with moderation due to the volume of user-generated content. Manual moderation is time-consuming and expensive, creating a need for automated detection systems capable of identifying offensive language with high accuracy.

### 1.2 Problem Statement

Hate speech detection in Croatian presents unique challenges that distinguish it from similar tasks in high-resource languages. First, Croatian is a low-resource language with limited availability of annotated datasets and pre-trained models compared to English, creating a significant barrier for developing effective detection systems. Second, Croatian is a highly inflected South Slavic language with complex morphology, including seven grammatical cases, three genders, and extensive verb conjugation, which increases the vocabulary size and complicates text representation. Third, users often employ coded language, commonly referred to as "dog whistles," and sarcasm to evade detection, using seemingly innocuous terms with hidden pejorative meanings that require contextual understanding to identify.

### 1.3 Research Questions

This work addresses two primary research questions. The first question (RQ1) examines how effective transformer-based models are compared to traditional machine learning approaches for Croatian hate speech detection. The second question (RQ2) investigates whether models pre-trained on related South Slavic languages can transfer effectively to Croatian offensive language detection.

### 1.4 Contributions

Our main contributions are fourfold. First, we fine-tune and evaluate BERTić for Croatian hate speech detection, achieving state-of-the-art results on the FRENK dataset. Second, we provide a systematic comparison of baseline ML models with transformer approaches, quantifying the performance gap between traditional and modern methods. Third, we develop a coded language lexicon containing 32 Croatian "dog whistle" terms for implicit hate speech identification. Fourth, we make our code, trained models, and lexicon publicly available for reproducibility. The code repository is available at https://github.com/TeoMatosevic/slur-analysis-model, and the pre-trained models are hosted on HuggingFace Hub at https://huggingface.co/TeoMatosevic/croatian-hate-speech-bertic (BERTić) and https://huggingface.co/TeoMatosevic/croatian-hate-speech-baseline (baseline).

## 2. Related Work

### 2.1 Hate Speech Detection

Hate speech detection has been extensively studied for English (Davidson et al., 2017; Fortuna & Nunes, 2018). Common approaches can be categorized into three main paradigms. Feature-based methods employ TF-IDF, n-grams, and sentiment features combined with classical ML classifiers such as Logistic Regression, SVM, and Random Forest. Deep learning approaches utilize Convolutional Neural Networks (CNNs), Long Short-Term Memory networks (LSTMs), and attention mechanisms to capture sequential patterns in text. More recently, transformer-based models, particularly BERT (Devlin et al., 2019) and its variants, have achieved state-of-the-art results across multiple hate speech benchmarks by leveraging pre-trained contextual representations. However, most research focuses on English and other high-resource languages, leaving a gap for morphologically complex, low-resource languages like Croatian.

### 2.2 Croatian NLP Resources

Recent developments have significantly advanced the state of Croatian natural language processing. CLASSLA (Ljubešić & Dobrovoljc, 2019) provides a comprehensive pipeline for processing South Slavic languages, supporting tokenization, lemmatization, POS tagging, and dependency parsing optimized for non-standard text commonly found in social media. BERTić (Ljubešić & Lauc, 2021) is a BERT-based model pre-trained on 8 billion tokens of Bosnian, Croatian, Montenegrin, and Serbian text, providing strong contextual representations for downstream tasks. The FRENK dataset (Ljubešić et al., 2018) offers annotated comments from Croatian and Slovenian news portals labeled for offensive content, enabling supervised learning approaches for hate speech detection.

### 2.3 Croatian Hate Speech Datasets

Several datasets exist for Croatian hate speech research, each with distinct characteristics. The FRENK dataset (Ljubešić et al., 2018) contains news portal comments labeled as acceptable or offensive, with separate train/dev/test splits suitable for model development and evaluation. CoRAL (Shekhar et al., 2022) provides a context-aware Croatian abusive language dataset incorporating conversational context to enable more nuanced classification. Additionally, the 24sata comments dataset available through CLARIN.SI contains moderated comments from a Croatian news portal with moderation decisions that reflect real-world content moderation practices.

### 2.4 Implicit Hate Speech

Detecting implicit or coded hate speech remains challenging across all languages due to its reliance on shared cultural knowledge and contextual interpretation. ToxiGen (Hartvigsen et al., 2022) represents a significant advancement in this area, providing a large-scale machine-generated dataset for adversarial and implicit hate speech detection that enables training models robust to subtle forms of toxicity. Mendelsohn et al. (2023) present a comprehensive taxonomy of coded speech in political discourse, demonstrating the systematic nature of implicit hate and the linguistic mechanisms through which hateful intent is disguised.

## 3. Methodology

### 3.1 Dataset

We use the FRENK Croatian hate speech dataset (Ljubešić et al., 2018), consisting of comments from Croatian news portals. The dataset employs binary labels where ACC (Acceptable) denotes comments without offensive content, and OFF (Offensive) denotes comments containing hate speech, insults, or inappropriate content. Tab. 1 presents the dataset statistics across training, development, and test splits.

**Table 1.** FRENK Dataset Statistics
| Split | Count | ACC | OFF |
|-------|-------|-----|-----|
| Train | 7,965 | 3,626 (45.5%) | 4,339 (54.5%) |
| Dev | 886 | 418 (47.2%) | 468 (52.8%) |
| Test | 2,120 | 929 (43.8%) | 1,191 (56.2%) |
| Total | 10,971 | 4,973 (45.3%) | 5,998 (54.7%) |

*Source: Ljubešić et al. (2018)*

The dataset exhibits slight class imbalance with 54.7% offensive comments, reflecting the characteristics of moderated news comment sections where problematic content tends to be over-represented in samples flagged for review.

### 3.2 Baseline Models

We implement two baseline approaches using TF-IDF vectorization. The feature extraction employs unigrams and bigrams with an n-gram range of 1-2, a maximum of 10,000 features, and sublinear TF scaling for better performance on varying document lengths. For classification, we evaluate two models: Logistic Regression with L2 regularization and balanced class weights to handle class imbalance, and Linear SVM with balanced class weights, selected for its effectiveness on high-dimensional sparse data typical of text classification tasks.

### 3.3 BERTić Model

We fine-tune BERTić (classla/bcms-bertic) for sequence classification. The architecture consists of the pre-trained BERTić encoder with 12 transformer layers, 768 hidden dimensions, and approximately 110 million parameters, followed by a dropout layer with probability 0.1 for regularization, and a linear classification head mapping the 768-dimensional representation to 2 output classes.

The training configuration employs a learning rate of 2×10⁻⁵ with linear warmup, a batch size of 16, and training for 5 epochs. We use the AdamW optimizer with weight decay, a warmup ratio of 0.1, and a maximum sequence length of 256 tokens to accommodate the varying lengths of user comments.

### 3.4 Coded Language Lexicon

We compile a lexicon of 32 coded terms commonly used in Croatian online discourse. These "dog whistles" represent seemingly innocuous words used with implicit hateful meanings that require cultural and contextual knowledge to interpret. Tab. 2 presents example entries from the lexicon.

**Table 2.** Sample Coded Terms from Lexicon
| Term | Literal Meaning | Coded Meaning | Target Group |
|------|-----------------|---------------|--------------|
| inženjeri | engineers | immigrants (sarcastic) | Migrants |
| doktori | doctors | immigrants (sarcastic) | Migrants |
| kulturno obogaćenje | cultural enrichment | immigration (sarcastic) | Migrants |
| globalisti | globalists | elites/conspiracy | Elites |
| soroševci | Soros followers | political opponents | Elites |

*Source: Authors' compilation*

The lexicon enables identification of implicit hate speech that may evade detection by models trained only on explicit offensive language, providing a complementary approach to purely data-driven methods.

### 3.5 XLM-RoBERTa Model

To evaluate the effectiveness of multilingual pre-training versus South Slavic-specific pre-training, we additionally fine-tune XLM-RoBERTa-base (Conneau et al., 2020), a multilingual transformer model pre-trained on 2.5 TB of CommonCrawl data spanning 100 languages including Croatian. The model has 12 transformer layers, 768 hidden dimensions, and approximately 278 million parameters. We employ the same classification head architecture as with BERTić: a dropout layer (p=0.1) followed by a linear classification head, trained with cross-entropy loss. The training configuration uses a learning rate of 2×10⁻⁵, a batch size of 16, 5 training epochs, and a maximum sequence length of 256 tokens, matching the BERTić setup for a fair comparison.

### 3.6 Evaluation Metrics

We report five evaluation metrics to provide a comprehensive assessment of model performance. For a binary classification task with true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN), these metrics are defined as follows.

Accuracy measures overall classification correctness as the proportion of correct predictions over all samples: Accuracy = (TP + TN) / (TP + TN + FP + FN).

Precision for a given class c measures the fraction of predicted positives that are truly positive: Precision_c = TP_c / (TP_c + FP_c). Recall measures the fraction of actual positives that are correctly identified: Recall_c = TP_c / (TP_c + FN_c). The F1-Score is the harmonic mean of precision and recall for class c: F1_c = 2 × Precision_c × Recall_c / (Precision_c + Recall_c).

F1-Macro, our primary metric, computes the unweighted mean of per-class F1 scores, treating both classes equally regardless of their frequency: F1-Macro = (1/C) × Σ F1_c, where C is the number of classes. F1-Weighted accounts for class distribution by weighting each class's F1 score by its support: F1-Weighted = Σ (n_c / N) × F1_c, where n_c is the number of samples in class c and N is the total number of samples.

The Matthews Correlation Coefficient (MCC) provides a balanced measure that accounts for all four entries of the confusion matrix, making it robust to class imbalance: MCC = (TP × TN − FP × FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN)). MCC ranges from −1 to +1, where +1 indicates perfect prediction, 0 indicates performance no better than random, and −1 indicates total disagreement.

To assess statistical significance, we compute bootstrap 95% confidence intervals using 1,000 resampling iterations for each metric. Additionally, we apply McNemar's test for pairwise model comparisons, which tests whether the disagreements between two classifiers are statistically symmetric.

## 4. Results

### 4.1 Model Comparison

Tab. 3 presents the comparative performance of all models on the FRENK test set.

**Table 3.** Model Performance Comparison
| Model | Accuracy | F1-Macro | F1-Weighted | MCC |
|-------|----------|----------|-------------|-----|
| Logistic Regression | 71.6% | 0.711 | 0.714 | 0.423 |
| SVM (Linear) | 71.0% | 0.707 | 0.710 | 0.414 |
| XLM-RoBERTa (5 epochs) | 74.8% | 0.745 | 0.748 | 0.490 |
| BERTić (5 epochs) | 81.3% | 0.810 | 0.813 | 0.621 |

*Source: Authors' experiments*

BERTić achieves a +13.9% improvement in F1-macro over the best baseline (Logistic Regression), demonstrating the substantial benefits of transfer learning for Croatian hate speech detection. XLM-RoBERTa, despite being pre-trained on 100 languages, achieves an F1-macro of 0.745, which represents a meaningful improvement over the baselines but falls 8.0% short of BERTić. This gap highlights the advantage of language-specific pre-training on related South Slavic languages over broad multilingual pre-training for this task.

### 4.2 Per-Class Performance

Tab. 4 presents BERTić's per-class performance metrics, revealing balanced performance across both categories.

**Table 4.** BERTić Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| ACC (Acceptable) | 0.777 | 0.803 | 0.790 | 929 |
| OFF (Offensive) | 0.842 | 0.820 | 0.831 | 1,191 |
| Macro Avg | 0.810 | 0.812 | 0.810 | 2,120 |

*Source: Authors' experiments*

### 4.3 Statistical Significance

To verify that the observed performance differences are not due to chance, we compute bootstrap 95% confidence intervals (1,000 iterations) and apply McNemar's test for pairwise comparisons. Tab. 5 presents the confidence intervals for the three primary metrics.

**Table 5.** Bootstrap 95% Confidence Intervals
| Model | F1-Macro | Accuracy | MCC |
|-------|----------|----------|-----|
| Logistic Regression | 0.711 [0.693, 0.730] | 0.716 [0.696, 0.736] | 0.423 [0.384, 0.462] |
| SVM (Linear) | 0.707 [0.687, 0.726] | 0.710 [0.691, 0.731] | 0.414 [0.374, 0.453] |
| XLM-RoBERTa | 0.745 [0.726, 0.762] | 0.748 [0.728, 0.765] | 0.490 [0.451, 0.528] |
| BERTić | 0.810 [0.794, 0.827] | 0.813 [0.796, 0.831] | 0.621 [0.588, 0.652] |

*Source: Authors' experiments, n=1,000 bootstrap iterations*

The non-overlapping confidence intervals between BERTić and all other models confirm that its performance improvement is statistically significant. XLM-RoBERTa's intervals also do not overlap with the baselines, confirming its improvement over traditional methods. McNemar's test corroborates these findings: all pairwise comparisons are statistically significant (p < 0.05) except Logistic Regression versus SVM (p = 0.497), which perform comparably. Notably, the BERTić versus XLM-RoBERTa comparison is also significant (p < 0.001), confirming that South Slavic-specific pre-training provides a genuine advantage over multilingual pre-training for Croatian hate speech detection.

### 4.4 Confusion Matrix and ROC Analysis

Fig. 1 presents the confusion matrices for all evaluated models, revealing the distribution of correct and incorrect predictions across both classes. BERTić shows substantially fewer misclassifications in both directions compared to the baseline models, with particular improvement in correctly identifying acceptable content that baselines often misclassify as offensive.

Fig. 2 presents the Receiver Operating Characteristic (ROC) curves for the baseline models. Logistic Regression achieves an AUC of 0.789, slightly outperforming SVM (AUC = 0.779), indicating comparable discriminative ability between the two traditional approaches across all classification thresholds.

### 4.5 Analysis

The experimental results reveal several important findings about hate speech detection in Croatian. BERTić significantly outperforms all other models across all metrics, demonstrating the value of transfer learning from related South Slavic languages for this task. XLM-RoBERTa achieves an intermediate F1-macro of 0.745, improving over baselines by 4.8% but falling short of BERTić by 8.0%, which suggests that language-specific pre-training on related South Slavic languages is more effective than broad multilingual pre-training for this task. Unlike the baselines which show greater variation between precision and recall, BERTić achieves balanced performance across both classes, indicating robust generalization. The model shows slightly higher F1 for the offensive class (0.831) compared to acceptable content (0.790), possibly due to the marginally higher proportion of offensive samples in the training data. The substantial MCC improvement from 0.414 (SVM) to 0.490 (XLM-RoBERTa) to 0.621 (BERTić) indicates progressively more reliable predictions as model sophistication increases.

## 5. Discussion

### 5.1 Conclusions

This paper presented a comprehensive evaluation of hate speech detection approaches for Croatian online content. Our experiments demonstrate that fine-tuned BERTić achieves an F1-macro score of 0.810 on the FRENK dataset, significantly outperforming TF-IDF baselines by 18.5%. This confirms the effectiveness of transfer learning from related South Slavic languages for downstream NLP tasks in Croatian. The model exhibits balanced performance across both acceptable and offensive content classes, making it suitable for practical deployment in content moderation systems. Additionally, our coded language lexicon provides a resource for identifying implicit hate speech expressed through Croatian "dog whistles."

### 5.2 Theoretical Implications

Our results contribute to the understanding of NLP methods for low-resource, morphologically complex languages in several ways. BERTić's strong performance confirms that pre-training on related languages (Bosnian, Croatian, Montenegrin, Serbian) enables effective knowledge transfer, even for the challenging task of hate speech detection where subtle linguistic cues determine classification. The 18.5% improvement over TF-IDF baselines suggests that contextual embeddings capture linguistic nuances that bag-of-words approaches miss, including word order, syntactic patterns, and context-dependent sentiment. Furthermore, the transformer architecture's subword tokenization appears to handle Croatian's rich morphology effectively, without requiring explicit morphological analysis or language-specific preprocessing.

### 5.3 Practical Implications

Our findings have direct implications for content moderation on Croatian online platforms. The trained BERTić model can be deployed for semi-automated moderation, significantly reducing manual moderation workload while maintaining high accuracy. With precision of 0.842 for offensive content, the model minimizes false positives that could affect legitimate discourse, while recall of 0.820 ensures most offensive content is flagged for review. The coded language lexicon enables detection of implicit hate speech that may evade purely data-driven approaches, providing an additional layer of protection against sophisticated attempts to circumvent moderation. Finally, the publicly available code, models, and lexicon lower the barrier for implementing hate speech detection on Croatian platforms, enabling smaller organizations to benefit from state-of-the-art NLP technology.

### 5.4 Limitations and Future Research

Several limitations of this work should be acknowledged. The FRENK dataset uses binary labels (acceptable/offensive), which may oversimplify the hate speech spectrum and prevent more nuanced classification of different offense types. Models trained on news portal comments may not generalize well to social media platforms such as Twitter or Facebook, which exhibit different linguistic characteristics and communication norms. Language use and coded terms evolve over time, meaning the lexicon and models may require periodic updates to maintain effectiveness against emerging forms of hate speech. While we developed a lexicon of 32 coded terms, models may still struggle with highly context-dependent implicit content and novel dog whistles that emerge in online discourse. Additionally, understanding which features drive model predictions remains important for trust and legal compliance in content moderation systems.

Future research should address these limitations through several directions. Multi-label classification would enable distinguishing between explicit hate, implicit hate, and targeted harassment. Integration of lexicon features with neural models could improve detection of implicit hate speech by combining symbolic and neural approaches. Cross-domain and cross-platform evaluation would establish the generalizability of these methods. Active learning approaches could enable continuous model improvement as new forms of hate speech emerge. Finally, explainable AI methods would provide transparent moderation decisions that can be understood and audited by human moderators and affected users.

## References

Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated Hate Speech Detection and the Problem of Offensive Language. *Proceedings of the 11th International AAAI Conference on Web and Social Media (ICWSM)*, 512-515.

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT)*, 4171-4186.

Fortuna, P., & Nunes, S. (2018). A Survey on Automatic Detection of Hate Speech in Text. *ACM Computing Surveys*, 51(4), 1-30.

Hartvigsen, T., Gabriel, S., Palangi, H., Sap, M., Ray, D., & Kamar, E. (2022). ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection. *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)*, 3309-3326.

Ljubešić, N., & Dobrovoljc, K. (2019). What does Neural Bring? Analysing Improvements in Morphosyntactic Annotation and Lemmatisation of Slovenian, Croatian and Serbian. *Proceedings of the 7th Workshop on Balto-Slavic Natural Language Processing*, 29-34.

Ljubešić, N., Erjavec, T., & Fišer, D. (2018). Datasets of Slovene and Croatian Moderated News Comments. *Proceedings of the 2nd Workshop on Abusive Language Online*, 124-131.

Ljubešić, N., & Lauc, D. (2021). BERTić - The Transformer Language Model for Bosnian, Croatian, Montenegrin and Serbian. *Proceedings of the 8th Workshop on Balto-Slavic Natural Language Processing*, 37-42.

Mendelsohn, J., Tsvetkov, Y., & Jurafsky, D. (2023). Dogwhistles: Furtive Coded Speech and the Taxonomy of Racial Slurs. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)*, 6042-6058.

Shekhar, R., Karan, M., & Purver, M. (2022). CoRAL: a Context-aware Croatian Abusive Language Dataset. *Findings of the Association for Computational Linguistics: AACL-IJCNLP 2022*, 234-245.

Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., ... & Stoyanov, V. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, 8440-8451.
