# Detection of Hate Speech on Croatian Online Portals using NLP Methods

**Authors:** Duje Jurić, Teo Matošević, Teo Radolović

**Institution:** University of Zagreb, Faculty of Electrical Engineering and Computing

**Course:** Natural Language Processing (Obrada prirodnog jezika)

---

## Abstract

Hate speech detection in low-resource languages remains a significant challenge for natural language processing. This paper presents a comprehensive approach to detecting offensive language in Croatian online comments using both traditional machine learning baselines and transformer-based models. We fine-tune BERTić, a BERT-based model pre-trained on 8 billion tokens of Bosnian, Croatian, Montenegrin, and Serbian text, for binary classification of acceptable versus offensive content. Our experiments on the FRENK dataset (10,971 comments) demonstrate that fine-tuned BERTić achieves an F1-macro score of 0.810, significantly outperforming TF-IDF baselines (F1=0.684) by 18.5%. Additionally, we develop a lexicon of 32 coded terms ("dog whistles") commonly used in Croatian online discourse to express implicit hate speech. Our results highlight the effectiveness of transfer learning for hate speech detection in morphologically rich, low-resource languages.

**Keywords:** hate speech detection, Croatian NLP, BERTić, transformer models, offensive language

---

## 1. Introduction

### 1.1 Background

The proliferation of social media and online news portals has led to an increase in hate speech and offensive content in digital spaces. Croatian online platforms, including news comment sections and social media, face significant challenges with moderation due to the volume of user-generated content. Manual moderation is time-consuming and expensive, creating a need for automated detection systems.

### 1.2 Problem Statement

Hate speech detection in Croatian presents unique challenges:

1. **Low-resource language**: Limited availability of annotated datasets and pre-trained models compared to English
2. **Morphological complexity**: Croatian is a highly inflected language with complex morphology
3. **Implicit hate speech**: Users often employ coded language ("dog whistles") and sarcasm to evade detection

### 1.3 Research Questions

This work addresses the following research questions:

- **RQ1**: How effective are transformer-based models compared to traditional ML approaches for Croatian hate speech detection?
- **RQ2**: Can models pre-trained on related South Slavic languages transfer effectively to Croatian offensive language detection?

### 1.4 Contributions

Our main contributions are:

1. Fine-tuning and evaluation of BERTić for Croatian hate speech detection, achieving state-of-the-art results
2. Systematic comparison of baseline ML models with transformer approaches
3. Development of a coded language lexicon for Croatian implicit hate speech
4. Publicly available code and models for reproducibility

---

## 2. Related Work

### 2.1 Hate Speech Detection

Hate speech detection has been extensively studied for English (Davidson et al., 2017; Fortuna & Nunes, 2018). Common approaches include:

- **Feature-based methods**: TF-IDF, n-grams, sentiment features with classical ML classifiers
- **Deep learning**: CNNs, LSTMs, and attention mechanisms
- **Transformers**: BERT and variants achieving state-of-the-art results

### 2.2 Croatian NLP Resources

Recent developments in Croatian NLP include:

- **CLASSLA** (Ljubešić & Dobrovoljc, 2019): A pipeline for processing South Slavic languages, supporting tokenization, lemmatization, POS tagging, and dependency parsing for non-standard text
- **BERTić** (Ljubešić & Lauc, 2021): A BERT-based model pre-trained on 8 billion tokens of Bosnian, Croatian, Montenegrin, and Serbian text
- **FRENK dataset** (Ljubešić et al., 2019): Annotated comments from Croatian and Slovenian news portals

### 2.3 Croatian Hate Speech Datasets

Several datasets exist for Croatian hate speech research:

- **FRENK** (Ljubešić et al., 2019): News portal comments labeled as acceptable or offensive
- **CoRAL** (Shekhar et al., 2022): Context-aware Croatian abusive language dataset
- **24sata comments** (CLARIN.SI): Moderated comments from Croatian news portal

### 2.4 Implicit Hate Speech

Detecting implicit or coded hate speech remains challenging. Related work includes:

- **ToxiGen** (Hartvigsen et al., 2022): Large-scale machine-generated dataset for implicit hate
- **Silent Signals** (Mendelsohn et al., 2023): Detection of dog whistles in political discourse

---

## 3. Methodology

### 3.1 Dataset

We use the FRENK Croatian hate speech dataset, consisting of comments from Croatian news portals. The dataset contains binary labels:

- **ACC** (Acceptable): Comments without offensive content
- **OFF** (Offensive): Comments containing hate speech, insults, or inappropriate content

**Dataset Statistics:**

| Split | Count | ACC | OFF |
|-------|-------|-----|-----|
| Train | 7,965 | 3,626 (45.5%) | 4,339 (54.5%) |
| Dev | 886 | 418 (47.2%) | 468 (52.8%) |
| Test | 2,120 | 929 (43.8%) | 1,191 (56.2%) |
| **Total** | **10,971** | **4,973 (45.3%)** | **5,998 (54.7%)** |

### 3.2 Baseline Models

We implement two baseline approaches using TF-IDF vectorization:

**TF-IDF Features:**
- Unigrams and bigrams (n-gram range: 1-2)
- Maximum 10,000 features
- Sublinear TF scaling

**Classifiers:**
1. **Logistic Regression**: L2 regularization, balanced class weights
2. **Linear SVM**: Linear kernel, balanced class weights

### 3.3 BERTić Model

We fine-tune BERTić (`classla/bcms-bertic`) for sequence classification:

**Architecture:**
- Pre-trained BERTić encoder (12 layers, 768 hidden dimensions)
- Dropout layer (p=0.1)
- Linear classification head (768 → 2)

**Training Configuration:**
- Learning rate: 2×10⁻⁵
- Batch size: 16
- Epochs: 5
- Optimizer: AdamW
- Warmup ratio: 0.1
- Maximum sequence length: 256 tokens

### 3.4 Coded Language Lexicon

We compile a lexicon of 32 coded terms commonly used in Croatian online discourse. Examples include:

| Term | Literal Meaning | Coded Meaning | Target Group |
|------|-----------------|---------------|--------------|
| inženjeri | engineers | immigrants (sarcastic) | Migrants |
| doktori | doctors | immigrants (sarcastic) | Migrants |
| kulturno obogaćenje | cultural enrichment | immigration (sarcastic) | Migrants |
| globalisti | globalists | elites/conspiracy | Elites |
| soroševci | Soros followers | political opponents | Elites |

### 3.5 Evaluation Metrics

We report the following metrics:

- **Accuracy**: Overall classification accuracy
- **F1-Macro**: Macro-averaged F1 score (primary metric)
- **F1-Weighted**: Weighted F1 score
- **Precision/Recall**: Per-class precision and recall
- **MCC**: Matthews Correlation Coefficient

---

## 4. Results

### 4.1 Model Comparison

| Model | Accuracy | F1-Macro | F1-Weighted | MCC |
|-------|----------|----------|-------------|-----|
| Logistic Regression | 69.0% | 0.684 | 0.689 | 0.371 |
| SVM (Linear) | 68.5% | 0.680 | 0.684 | 0.361 |
| **BERTić (5 epochs)** | **81.3%** | **0.810** | **0.813** | **0.621** |

BERTić achieves a **+18.5% improvement** in F1-macro over the best baseline.

### 4.2 Per-Class Performance (BERTić)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| ACC (Acceptable) | 0.777 | 0.803 | 0.790 | 929 |
| OFF (Offensive) | 0.842 | 0.820 | 0.831 | 1,191 |
| **Macro Avg** | **0.810** | **0.812** | **0.810** | 2,120 |

### 4.3 Analysis

**Key Findings:**

1. **Transformer superiority**: BERTić significantly outperforms traditional ML baselines, demonstrating the value of transfer learning for Croatian NLP
2. **Balanced performance**: BERTić achieves balanced precision and recall across both classes
3. **Offensive detection**: Higher F1 for offensive class (0.831) compared to acceptable (0.790), possibly due to class distribution

---

## 5. Discussion

### 5.1 Effectiveness of Transfer Learning

Our results demonstrate that BERTić, pre-trained on South Slavic languages, transfers effectively to Croatian hate speech detection. The 18.5% improvement over baselines suggests that contextual embeddings capture nuances that TF-IDF features miss, including:

- Word order and syntax
- Context-dependent meanings
- Implicit negative sentiment

### 5.2 Limitations

Several limitations should be noted:

1. **Binary classification**: The FRENK dataset uses binary labels, which may oversimplify the hate speech spectrum
2. **Domain specificity**: Models trained on news comments may not generalize to social media
3. **Temporal drift**: Language use and coded terms evolve over time
4. **Implicit hate speech**: While we developed a lexicon, models may still struggle with highly implicit content

### 5.3 Future Work

Future directions include:

1. **Multi-label classification**: Distinguishing between hate speech types (explicit, implicit, targeted)
2. **Lexicon integration**: Combining lexicon features with neural models
3. **Cross-domain evaluation**: Testing on social media data
4. **Explainability**: Understanding which features drive model predictions

---

## 6. Conclusion

This paper presented a comprehensive approach to Croatian hate speech detection using both traditional ML and transformer-based models. Fine-tuned BERTić achieves an F1-macro score of 0.810, significantly outperforming TF-IDF baselines. Our results demonstrate the effectiveness of transfer learning for low-resource languages and provide a foundation for automated content moderation in Croatian online spaces.

The code, models, and lexicon are publicly available at: https://github.com/TeoMatosevic/slur-analysis-model

---

## References

Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated Hate Speech Detection and the Problem of Offensive Language. *Proceedings of ICWSM*.

Fortuna, P., & Nunes, S. (2018). A Survey on Automatic Detection of Hate Speech in Text. *ACM Computing Surveys*, 51(4).

Hartvigsen, T., Gabriel, S., Palangi, H., Sap, M., Ray, D., & Kamar, E. (2022). ToxiGen: A Large-Scale Machine-Generated Dataset for Adversarial and Implicit Hate Speech Detection. *Proceedings of ACL*.

Ljubešić, N., & Dobrovoljc, K. (2019). What does Neural Bring? Analysing Improvements in Morphosyntactic Annotation and Lemmatisation of Slovenian, Croatian and Serbian. *Proceedings of the 7th Workshop on Balto-Slavic Natural Language Processing*.

Ljubešić, N., Erjavec, T., & Fišer, D. (2018). Datasets of Slovene and Croatian Moderated News Comments. *Proceedings of the 2nd Workshop on Abusive Language Online*.

Ljubešić, N., & Lauc, D. (2021). BERTić - The Transformer Language Model for Bosnian, Croatian, Montenegrin and Serbian. *Proceedings of the 8th Workshop on Balto-Slavic Natural Language Processing*.

Mendelsohn, J., Tsvetkov, Y., & Jurafsky, D. (2023). Dogwhistles: Furtive Coded Speech and the Taxonomy of Racial Slurs. *Proceedings of ACL*.

Shekhar, R., Karan, M., & Purver, M. (2022). CoRAL: a Context-aware Croatian Abusive Language Dataset. *Findings of ACL*.

---

## Appendix A: Coded Terms Lexicon

The complete lexicon of 32 coded terms is available in `data/lexicon/coded_terms.json`.

**Sample entries:**

```json
{
  "term": "inženjeri",
  "literal": "engineers",
  "coded_meaning": "immigrants/refugees (sarcastic reference to claimed professional skills)",
  "target_group": "TGT_MIG",
  "category": "IHS"
}
```

## Appendix B: Reproducibility

**Training BERTić:**
```bash
python src/training/train.py \
    --data data/processed/frenk_train.jsonl \
    --model bertic \
    --output checkpoints/bertic
```

**Evaluation:**
```bash
python src/training/evaluate.py \
    --data data/processed/frenk_test.jsonl \
    --model bertic \
    --model-path checkpoints/bertic/best_model \
    --output evaluation_results
```

**Demo:**
```bash
python src/demo.py --text "Your text here"
```
