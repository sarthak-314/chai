



# Post Processing 
- Progressive Pruning: remove definatlely not answer after prediction with each model
- GoldS Pseduo labelling 
-  In case of multiple possible answers, incentevise the model to select the first sentence. Sentences should be in the same order as they appear, 
- Two Step: Answer Sentence selection -> answer text selection. 


# Other
- LB Probing for Wikipedia Indexes (TFIDF)

# Data Expansion
- Use different question generation strategies
- If question generation succeeds, try context expansion
- Generating qa from hi-ta wikipedia with MT5-XXL might be super useful. 

# Pre-Pre Training 
- Use wiki ner embeddings
Finetuning

# Training
- If you can cut the time to load data and the time for first epoch, you can run many many many experiments blazingly fast

# Finetuning
- Finetuning with lower context length, higher doc stride

Post Processing
Other 



- Team up with 4 people early and tell them to run the long time TPU code. This way you get 150 legit + 60 non legit hours / week
- Maybe local works slow because TPU reads from GCS bucket better
- cloud-tpu-tools
- Release TPU memory with del model, gc.collect() or tf.tpu.experimental.initialize_tpu_system(tpu) (reinitialize the tpu)
- Make the dataset -> cache the dataset -> try every possible model config on that dataset??
- Remove mixed precision when finetuning
- TODO: Move github repo to deep-learner-314/tea
- Buy g-stars after pushing finally
- Postprocessing: So what we did is to map the start and end token predictions of each window back to the original answer and create a answer length x answer length heatmap.

- Clip negative answer weights
- Hidden Layer
- add_dropout(), ...
- pass word count length and context len to bert somehow
- I am very burnt out. Once I get to the top, I'll take a one week sabbatical and cleansing

FUTURE 
- Write a long post, by gstars
- Win this competition in the first place by a landslide and use it to get a kickass dl job. In the end, a great Kaggle project and some actual experimece and money. Plus I can be a grandmaster just by targeting NLP competitions, 
