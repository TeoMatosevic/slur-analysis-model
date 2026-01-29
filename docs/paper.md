# Detection of Hate Speech on Croatian Online Portals Using NLP Methods

Duje Jurić¹, Teo Matošević²*, Teo Radolović³

¹ University of Zagreb, Faculty of Electrical Engineering and Computing, Croatia
² University of Zagreb, Faculty of Electrical Engineering and Computing, Croatia
³ University of Zagreb, Faculty of Electrical Engineering and Computing, Croatia

*Corresponding author

## Abstract

**Purpose** – The proliferation of hate speech on Croatian online platforms presents a significant challenge for content moderation. This paper investigates the effectiveness of natural language processing methods for automated detection of offensive language in Croatian, a morphologically complex, low-resource language with limited annotated datasets and pre-trained models.

**Design/Methodology/Approach** – We conduct a systematic comparison of traditional machine learning baselines (TF-IDF with Logistic Regression and SVM) against a fine-tuned transformer model (BERTić) on the FRENK Croatian hate speech dataset containing 10,971 annotated comments. Additionally, we develop a lexicon of 32 coded terms ("dog whistles") commonly used in Croatian online discourse to express implicit hate speech.

**Findings** – Fine-tuned BERTić achieves an F1-macro score of 0.810, significantly outperforming TF-IDF baselines (F1=0.684) by 18.5%. The model demonstrates balanced performance across both acceptable (F1=0.790) and offensive (F1=0.831) classes. Results confirm that transfer learning from related South Slavic languages effectively captures contextual nuances that traditional feature-based approaches miss.

**Originality/Value** – This work provides the first systematic comparison of transformer-based and traditional ML approaches for Croatian hate speech detection. We contribute publicly available code, trained models, and a coded language lexicon to support future research and practical content moderation applications.

**Keywords** – Hate speech detection; Croatian NLP; BERTić; Transformer models; Offensive language; Low-resource languages.

**Paper Type** – Research paper.

## 1. Introduction

### 1.1 Background

The proliferation of social media and online news portals has led to an increase in hate speech and offensive content in digital spaces. Croatian online platforms, including news comment sections and social media, face significant challenges with moderation due to the volume of user-generated content. Manual moderation is time-consuming and expensive, creating a need for automated detection systems capable of identifying offensive language with high accuracy.

### 1.2 Problem Statement

Hate speech detection in Croatian presents unique challenges:

1. **Low-resource language**: Limited availability of annotated datasets and pre-trained models compared to English creates a significant barrier for developing effective detection systems.
2. **Morphological complexity**: Croatian is a highly inflected South Slavic language with complex morphology, including seven grammatical cases, three genders, and extensive verb conjugation.
3. **Implicit hate speech**: Users often employ coded language ("dog whistles") and sarcasm to evade detection, using seemingly innocuous terms with hidden pejorative meanings.

### 1.3 Research Questions

This work addresses the following research questions:

- **RQ1**: How effective are transformer-based models compared to traditional ML approaches for Croatian hate speech detection?
- **RQ2**: Can models pre-trained on related South Slavic languages transfer effectively to Croatian offensive language detection?

### 1.4 Contributions

Our main contributions are:

1. Fine-tuning and evaluation of BERTić for Croatian hate speech detection, achieving state-of-the-art results on the FRENK dataset
2. Systematic comparison of baseline ML models with transformer approaches, quantifying the performance gap
3. Development of a coded language lexicon containing 32 Croatian "dog whistle" terms for implicit hate speech
4. Publicly available code, models, and lexicon for reproducibility at: https://github.com/TeoMatosevic/slur-analysis-model

## 2. Related Work

### 2.1 Hate Speech Detection

Hate speech detection has been extensively studied for English (Davidson et al., 2017; Fortuna & Nunes, 2018). Common approaches include:

- **Feature-based methods**: TF-IDF, n-grams, sentiment features combined with classical ML classifiers such as Logistic Regression, SVM, and Random Forest
- **Deep learning**: Convolutional Neural Networks (CNNs), Long Short-Term Memory networks (LSTMs), and attention mechanisms
- **Transformers**: BERT (Devlin et al., 2019) and its variants achieving state-of-the-art results across multiple hate speech benchmarks

However, most research focuses on English and other high-resource languages, leaving a gap for morphologically complex, low-resource languages like Croatian.

### 2.2 Croatian NLP Resources

Recent developments in Croatian NLP include:

- **CLASSLA** (Ljubešić & Dobrovoljc, 2019): A pipeline for processing South Slavic languages, supporting tokenization, lemmatization, POS tagging, and dependency parsing optimized for non-standard text commonly found in social media
- **BERTić** (Ljubešić & Lauc, 2021): A BERT-based model pre-trained on 8 billion tokens of Bosnian, Croatian, Montenegrin, and Serbian text, providing strong contextual representations for downstream tasks
- **FRENK dataset** (Ljubešić et al., 2018): Annotated comments from Croatian and Slovenian news portals labeled for offensive content

### 2.3 Croatian Hate Speech Datasets

Several datasets exist for Croatian hate speech research:

- **FRENK** (Ljubešić et al., 2018): News portal comments labeled as acceptable or offensive, with separate train/dev/test splits
- **CoRAL** (Shekhar et al., 2022): Context-aware Croatian abusive language dataset incorporating conversational context
- **24sata comments** (CLARIN.SI): Moderated comments from Croatian news portal with moderation decisions

### 2.4 Implicit Hate Speech

Detecting implicit or coded hate speech remains challenging across all languages. Related work includes:

- **ToxiGen** (Hartvigsen et al., 2022): Large-scale machine-generated dataset for adversarial and implicit hate speech detection
- **Dogwhistles** (Mendelsohn et al., 2023): Taxonomy of coded speech in political discourse, demonstrating the systematic nature of implicit hate

## 3. Methodology

### 3.1 Dataset

We use the FRENK Croatian hate speech dataset (Ljubešić et al., 2018), consisting of comments from Croatian news portals. The dataset contains binary labels:

- **ACC** (Acceptable): Comments without offensive content
- **OFF** (Offensive): Comments containing hate speech, insults, or inappropriate content

Tab. 1 presents the dataset statistics across training, development, and test splits.

*Table 1: FRENK Dataset Statistics*
| Split | Count | ACC | OFF |
|-------|-------|-----|-----|
| Train | 7,965 | 3,626 (45.5%) | 4,339 (54.5%) |
| Dev | 886 | 418 (47.2%) | 468 (52.8%) |
| Test | 2,120 | 929 (43.8%) | 1,191 (56.2%) |
| **Total** | **10,971** | **4,973 (45.3%)** | **5,998 (54.7%)** |

*Source: Ljubešić et al. (2018)*

The dataset exhibits slight class imbalance with 54.7% offensive comments, reflecting the characteristics of moderated news comment sections.

### 3.2 Baseline Models

We implement two baseline approaches using TF-IDF vectorization:

**TF-IDF Features:**
- Unigrams and bigrams (n-gram range: 1-2)
- Maximum 10,000 features
- Sublinear TF scaling for better performance on varying document lengths

**Classifiers:**
1. **Logistic Regression**: L2 regularization with balanced class weights to handle class imbalance
2. **Linear SVM**: Linear kernel with balanced class weights, selected for its effectiveness on high-dimensional sparse data

### 3.3 BERTić Model

We fine-tune BERTić (`classla/bcms-bertic`) for sequence classification:

**Architecture:**
- Pre-trained BERTić encoder (12 transformer layers, 768 hidden dimensions, ~110M parameters)
- Dropout layer (p=0.1) for regularization
- Linear classification head (768 → 2)

**Training Configuration:**
- Learning rate: 2×10⁻⁵ with linear warmup
- Batch size: 16
- Epochs: 5
- Optimizer: AdamW with weight decay
- Warmup ratio: 0.1
- Maximum sequence length: 256 tokens

### 3.4 Coded Language Lexicon

We compile a lexicon of 32 coded terms commonly used in Croatian online discourse. These "dog whistles" represent seemingly innocuous words used with implicit hateful meanings. Tab. 2 presents example entries.

*Table 2: Sample Coded Terms from Lexicon*
| Term | Literal Meaning | Coded Meaning | Target Group |
|------|-----------------|---------------|--------------|
| inženjeri | engineers | immigrants (sarcastic) | Migrants |
| doktori | doctors | immigrants (sarcastic) | Migrants |
| kulturno obogaćenje | cultural enrichment | immigration (sarcastic) | Migrants |
| globalisti | globalists | elites/conspiracy | Elites |
| soroševci | Soros followers | political opponents | Elites |

*Source: Authors' compilation*

The lexicon enables identification of implicit hate speech that may evade detection by models trained only on explicit offensive language.

### 3.5 Evaluation Metrics

We report the following metrics:

- **Accuracy**: Overall classification accuracy
- **F1-Macro**: Macro-averaged F1 score (primary metric, treating both classes equally)
- **F1-Weighted**: Weighted F1 score accounting for class distribution
- **Precision/Recall**: Per-class precision and recall
- **MCC**: Matthews Correlation Coefficient, robust to class imbalance

## 4. Results

### 4.1 Model Comparison

Tab. 3 presents the comparative performance of all models on the FRENK test set.

*Table 3: Model Performance Comparison*
| Model | Accuracy | F1-Macro | F1-Weighted | MCC |
|-------|----------|----------|-------------|-----|
| Logistic Regression | 69.0% | 0.684 | 0.689 | 0.371 |
| SVM (Linear) | 68.5% | 0.680 | 0.684 | 0.361 |
| **BERTić (5 epochs)** | **81.3%** | **0.810** | **0.813** | **0.621** |

*Source: Authors' experiments*

BERTić achieves a **+18.5% improvement** in F1-macro over the best baseline (Logistic Regression), demonstrating the substantial benefits of transfer learning for Croatian hate speech detection.

### 4.2 Per-Class Performance

Tab. 4 presents BERTić's per-class performance metrics.

*Table 4: BERTić Per-Class Performance*
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| ACC (Acceptable) | 0.777 | 0.803 | 0.790 | 929 |
| OFF (Offensive) | 0.842 | 0.820 | 0.831 | 1,191 |
| **Macro Avg** | **0.810** | **0.812** | **0.810** | 2,120 |

*Source: Authors' experiments*

### 4.3 Analysis

Key findings from our experiments:

1. **Transformer superiority**: BERTić significantly outperforms traditional ML baselines across all metrics, demonstrating the value of transfer learning from related South Slavic languages
2. **Balanced performance**: BERTić achieves balanced precision and recall across both classes, unlike baselines which show greater variation
3. **Offensive detection advantage**: Higher F1 for offensive class (0.831) compared to acceptable (0.790), possibly due to the slightly higher proportion of offensive samples in training data
4. **MCC improvement**: The substantial MCC improvement (0.621 vs 0.371) indicates BERTić produces more reliable predictions accounting for class distribution

## 5. Discussion

### 5.1 Conclusions

This paper presented a comprehensive evaluation of hate speech detection approaches for Croatian online content. Our experiments demonstrate that fine-tuned BERTić achieves an F1-macro score of 0.810 on the FRENK dataset, significantly outperforming TF-IDF baselines by 18.5%. This confirms the effectiveness of transfer learning from related South Slavic languages for downstream NLP tasks in Croatian.

The model exhibits balanced performance across both acceptable and offensive content classes, making it suitable for practical deployment in content moderation systems. Additionally, our coded language lexicon provides a resource for identifying implicit hate speech expressed through Croatian "dog whistles."

### 5.2 Theoretical Implications

Our results contribute to the understanding of NLP methods for low-resource, morphologically complex languages:

1. **Transfer learning effectiveness**: BERTić's strong performance confirms that pre-training on related languages (Bosnian, Croatian, Montenegrin, Serbian) enables effective knowledge transfer, even for the challenging task of hate speech detection
2. **Contextual understanding**: The 18.5% improvement over TF-IDF baselines suggests that contextual embeddings capture linguistic nuances that bag-of-words approaches miss, including word order, syntactic patterns, and context-dependent sentiment
3. **Morphological handling**: The transformer architecture's subword tokenization appears to handle Croatian's rich morphology effectively, without requiring explicit morphological analysis

### 5.3 Practical Implications

Our findings have direct implications for content moderation:

1. **Deployment readiness**: The trained BERTić model can be deployed for semi-automated moderation on Croatian online platforms, significantly reducing manual moderation workload
2. **Balanced performance**: With precision of 0.842 for offensive content, the model minimizes false positives that could affect legitimate discourse, while recall of 0.820 ensures most offensive content is flagged
3. **Coded language awareness**: The lexicon enables detection of implicit hate speech that may evade purely data-driven approaches, providing an additional layer of protection
4. **Open resources**: The publicly available code, models, and lexicon lower the barrier for implementing hate speech detection on Croatian platforms

### 5.4 Limitations and Future Research

Several limitations should be noted:

1. **Binary classification**: The FRENK dataset uses binary labels (acceptable/offensive), which may oversimplify the hate speech spectrum. Future work could explore multi-label classification distinguishing explicit hate, implicit hate, and targeted harassment.

2. **Domain specificity**: Models trained on news portal comments may not generalize well to social media platforms (Twitter, Facebook) with different linguistic characteristics. Cross-domain evaluation is needed.

3. **Temporal drift**: Language use and coded terms evolve over time. The lexicon and models may require periodic updates to maintain effectiveness.

4. **Implicit hate speech coverage**: While we developed a lexicon of 32 coded terms, models may still struggle with highly context-dependent implicit content and emerging dog whistles.

5. **Explainability**: Understanding which features drive model predictions remains important for trust and legal compliance. Future work should explore attention visualization and other interpretability methods.

Future research directions include:
- Multi-label classification distinguishing hate speech types
- Integration of lexicon features with neural models for improved implicit hate detection
- Cross-domain and cross-platform evaluation
- Active learning approaches for continuous model improvement
- Explainable AI methods for transparent moderation decisions

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
