- If you can cut the time to load data and the time for first epoch, you can run many many many experiments blazingly fast
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
 