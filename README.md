#   Discontinuous Named Entity Recognition by Fusing Ensemble Learning and Retrieval-Augmented Large Language Models

##  1. Datasets
*   [CADEC](https://data.csiro.au/collection/csiro:10948)
*   [ShARe13](https://physionet.org/content/shareclefehealth2013/1.0/)
*   [ShARe14](https://physionet.org/content/shareclefehealth2014task2/1.0/)

##  2. Data Preprocessing
### Different preprocessing for different datasets:  
*   CADEC: Preprocessing for CADEC following [Dai et al](https://github.com/dainlp/acl2020-transition-discontinuous-ner).
*   ShARe13_fixed: Preprocessing that merges sentences containing cross-sentence entities and recalculates entity index positions for ShARe13.
*   ShARe14_fixed: Preprocessing that merges sentences containing cross-sentence entities and recalculates entity index positions for ShARe14.
-   [stanford-corenlp-full-2018-10-05](https://stanfordnlp.github.io/CoreNLP/history.html)
-   [jdk-21_windows-x64_bin.exe](https://www.oracle.com/tw/java/technologies/downloads/)

### Preprocessing
*   CADEC  
    1.  In M1, put "original" and "text" folder from CADEC into "\data\Corpora\CADEC".
    2.  Run "extract_annotations.py" → "tokenization.py" → "convert_ann_using_token_idx.py" → "convert_text_inline.py" → "split_train_test.py". Please refer to "圖片說明.pptx".
    3.  In M2, put M1 output into "Input" folder and run "preprocess_2.py". Please refer to "圖片說明.pptx".
    4.  In M3, put M1 output into "Input" folder and run "preprocess_3.py".
    5.  In M4-5, put M3 output into "Input" folder and run "preprocess_4-5.py".

*   ShARe13_fixed
    1.  In M1, unzip "Task1TrainSetCorpus199.zip" from ShARe13 and put all .txt files into "\train\text".
    2.  Unzip "Task1TrainSetGOLD199knowtatorehost.zip" from ShARe13 and put all .xml files into "\train\ann".
    3.  Unzip "Task1TestSetCorpus100.zip" and put all .txt files into "\test\text".
    4.  Unzip "Task1Gold_SN2012.tar.bz2" and put all .txt files into "\test\ann".
    5.  Run "extract_annotations.py" → "tokenization.py" → "data_fixed.py" → "convert_ann_using_token_idx.py" → "convert_text_inline.py".
    6.  Step 3 ~ 5 in CADEC.

*   ShARe14_fixed
    1.  In M1, unzip "2014ShAReCLEFeHealthTasks2_training_10Jan2014.zip" from ShARe14.
    2.  Put all .txt files in "2014ShAReCLEFeHealthTask2_training_corpus" into "\train\text".
    3.  Put all .txt files in "2014ShAReCLEFeHealthTask2_training_pipedelimited.zip" into "\train\ann".
    4.  Unzip "ShAReCLEFeHealth2014Task2_test_default_values.zip" and put all .txt files in "ShAReCLEFeHealth2104Task2_test_data_corpus" into "\test\text".
    5.  Unzip "ShAReCLEFeHealth2014_test_data_gold.zip" from ShARe14 and put all .txt files into "\test\ann".
    6.  Run "extract_annotations.py" → "tokenization.py" → "data_fixed.py" → "convert_ann_using_token_idx.py" → "convert_text_inline.py".
    7.  Step 3 ~ 5 in CADEC.

##  3. DNER Models
*   M1: [Transition-based model](https://github.com/dainlp/acl2020-transition-discontinuous-ner).
*   M2: [Span-based model](https://github.com/foxlf823/sodner).
*   M3: [Mac model](https://github.com/131250208/infextraction).
*   M4: [W2NER model](https://github.com/ljynlp/W2NER).
*   M5: [TOE model](https://github.com/solkx/TOE).
-   P.S. Each model's folder contains a .yaml file for environment settings.
-   [elmo_2x4096_512_2048cnn_2xhighway_5.5B\weights.hdf5](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5)
-   [glove.6B.100d.txt](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt)
-   [scibert_scivocab_cased.tar](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_cased.tar)
-   [biobert-base-cased-v1.2](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2/tree/main)
-   [YelpBERT](https://www.aclweb.org/anthology/2020.findings-emnlp.151/)

##  4. Format Alignment
Put five model outputs into "input" folder and run "format_alignment.py".

##  5. Meta-Learner
Put format alignment output into "input" folder and run "meta_learner.py".

##  6. RAG
1.  Download "meddra.tsv" and "meddra_all_se.tsv" from [SIDER](https://sideeffects.embl.de/download/) and put them into "\data\meddra_sider".
2.  Put format alignment output into "\data\format_alignment".
3.  Run "sider_preprocessing.py" → "rag_preprocessing.py" → "embedding.py" → "rag.py".

##  7. GPT
1.  Put RAG result into "\data\input".
2.  Write system prompt in "\data\prompt_system.txt", user prompt in "\data\prompt_user.txt", and GPT's API key in "\data\api_key".
3.  Run "response.py".

##  8. Post-Processing
1.  Put format alignment result into "\input\format alignment", meta-learner result into "\input\meta_learner", and GPT output into "\input\gpt".
2.  Run "post_processing.py".

##  9. Evaluation
Every evaluator is in this folder. "2. Data Preprocessing\\{dataset}\M3\Output\test_data.json" is used as golden text entities and "2. Data Preprocessing\\{dataset}\M4-5\Output\test_data.json" is used as golden index entities.
*   m1_evaluator: M1 Micro-F1.
*   m2_evaluator: M2 Micro-F1.
*   m3_evaluator: M3 Micro-F1.
*   m4_evaluator: M4 Micro-F1.
*   m5_evaluator: M5 Micro-F1.
*   based_model_evaluator: Weighted-F1, Macro-F1, and Instance-F1 for M1 ~ M5.
*   ensemble_voting_evaluator: Meta-learner (five model outputs hard voting) output. (Chen and Lin)
*   ensemble_gpt_evaluator: Meta-learner (GPT as an adjudicator to determine the optimal base model output) output. (Chen and Lin)
*   GPT-5-mini_evaluator: GPT-5-mini extracts entities by itself.
*   rag_evaluator: RAG-GPT + hard voting.
*   union_evaluator: RAG-GPT + hard voting output ∪ hard voting output.
*   qbc_evaluator: FLARE-DNER.
*   reevaluator: Evaluator for re-evaluating model performance after data quality evaluation.

##  10. Ablation Experiments
*   DNE_filtering.py: RAG-GPT + hard voting + DNE filtering strategy.
*   QBC.py: RAG-GPT + hard voting + QBC-based mechanism.
*   input  
    *   m3: "2. Data Preprocessing\\{dataset}\M3\Output\test_data.json" is used as golden text entities.
    *   m5: "2. Data Preprocessing\\{dataset}\M4-5\Output\test_data.json" is used as golden index entities.
    *   format alignment: format alignment result.
    *   meta_learner: meta-learner result.
    *   gpt: GPT output.

##  11. Data Quality Evaluation
### Preprocessing
1.  Put "2. Data Preprocessing\\{dataset}\M4-5\Output\test_data.json" into "\input\m5" and RAG result into "\input\rag".
2.  Run "preprocessing.py".

### Gemini
1.  Put preprocessing result into "\Gemini\input".
2.  Write prompt in "\Gemini\prompt\\{dataset}_prompt.txt" and Gemini's API key in "\Gemini\\.env".
3.  Run "\Gemini\gemini.py".
-   P.S. The .yaml file contains the environment setting.

### Llama
1.  Put preprocessing result into "\Llama\data\input".
2.  Write system prompt in "\Llama\data\prompt\\{dataset}_prompt_system.txt" and user prompt in "\Llama\data\prompt\\{dataset}_prompt_user.txt".
3.  Run "\Llama\response.py".
-   P.S. The .yaml file contains the environment setting.
-   [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

### Post-processing
1.  Put Gemini output into "\input\gemini" and Llama output into "\input\llama".
2.  Run "post_processing.py".

### Re-evaluation
1.  Put file in "\output\post" into "9. Evaluation\llm_data_evaluation".
2.  Run "9. Evaluation\reevaluator.py".
