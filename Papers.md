# TANDA (WikiQA winner)
TANDA: Transfer and Adapt Pre-Trained Transformer Models
for Answer Sentence Selection

- Fine tuning technique for pretrained transformers
- Outperforms by a large margin on WikiQA dataset 
- Reduces effort required for selecting hparams
- IDEA: Answer sentence selection as a pretraining step ??
- IDEA: Use Natural Question dataset as pre-pre training step just cause
- IDEA: 
    different loss weights for different situdations
    -- predicted_no_answer_when_answer_present 
    -- predicted_different_answer 
- 2e-5 for pretraining, 1e-6 for finetuning 
restart_tpu(), delete_model()

# Splinter (for MRQA)
- Different pretraining objective than MLM
- Used in few shot question answering 
- Add question [QUESTION] token at the end of the question. Use this question's representation to get the answer
- QASS: Question Aware Span Selection layer, which uses question token's representation to select the answer span
- pretraining MLM objective only needs local context, while qa selection requires more global context
- IDEA: Replace question mark with question token and use it's representation in the end 

# Synthetic Data Augmentation for Zero-Shot Cross-Lingual Question Answering
- generate synthetic questions
- sota in mlqa, xquad
- On English
corpora, generating synthetic questions has been
shown to significantly improve the performance of
QA models 
## WikiScrap
Select the wikipedia articles for generation

# Synthetic QA Corpora Generation with Roundtrip Consistency
- pretraining on synthetic data
- add back translation for gold passages

# SpanBERT 
- pretraining method to better predict spans