# Decoupling ✖
__Decoupling Hyperparameters__: You can keep the core logic and hyperparameters seprate by using a hierarical config framework like Hydra. 
__Decoupling Environment__: The notebook should be able to run without any change in code whether it's running on Kaggle, Colab, Google Cloud or a teammate's kaggle account. This helps me in parellization of experiments
__Decoupling Hardware__: This is easily done on Pytorch Lightning
__Decoupling Frameword__: This is the hardest and the most 'extreme' of the following. If you can decopuling this


# Working at Scale 📦
## Scaling up on Hardware Dimension
went 5x on TPU hardware by teaming up with non TPU users. Told them to run my code. This way I can go 5-10x on hardware by e̶x̶p̶l̶o̶i̶t̶i̶n̶g̶  using google's resources effectively. Our team made sure that we used up all the hardware quota on Kaggle every week. I wonder how life is for those who run 64 GPUs and 256 core TPUs

## Scaling up on External Data
I used huggingface's datasets to load huggingface QA datasets. This made loading external data very easy

## Scaling up on Competiton Data
The model should be able to fit the competition data. 

## Scaling on Time Dimension (Use with Caution)
If you have magically had more time, you can try more experiments, code up more ideas and read more papers. Unfortunately the number of hours in a day are fixed and you can't do anything about it. But I have found that I can still go 2-4x on time by skipping college classes, remaining unemployed & single and having no social life (not . recommended). There are things that I deeply care about and things I don't care about. College is one of the things I don't care about. 


## Scaling up on People Dimension
A team of 5 >> team of 1. Teaming up with top players clears out the gold medal area. Teaming up means more pooled hardware. There are very few reasons to not team up. Still I competeted solo because I wanted a solo 1st place and also keep the 2 grand to myself. I plan on reinvesting all my winnings back into kaggle and 2 grand will buy me a long lasting colab pro+ subscription and a high end laptop. Hopefully this will lead me to win some more,  which I'll use to buy gpus, which will lead me to win even more, which I'll use to buy more GPUs and TPU runtime and so on..  It's like a seed inverstment on my Kaggle startup. I was sure of winning, so an extra 1.6 grand is worth more than tpu hours. 
I got access to the Kaggle accounts from my college roommates. I know this falls a bit on the gray area for multiple accounts. 

## Future
I think hardware is still my biggest bottleneck. I will use the money I won from this competition to by a Colab Pro + subscriptionb. In future I hope to make enough money to be able to buy a few good GPUs. For now my best hope to scale up on hardware is to join a team where no one uses a TPU. My current main focus is to become a Kaggle competitions grandmaster, but I think I might take a part time deep learning job to gain some exposure and make a GPU-worthy extra income (referrels welcome!), The github link is the 'lite' version of my original codebase. I'm still keeping my code private so that I can use it in the futuer. 

## Easy Performance Hacks

### TPU related boosts
- mixed precision: availible in both GPUs and TPUs
- steps_per_execution: 32
- jit compilation
- tf.data hacks: cache, prefetch, autotune with map, using tfrecords, ignore order
- usually the first epoch takes the longest time
- Optimizer Hacks: Using SWA, using lookahead, weight decay, using cyclic scheduler

## My Mistakes
- I am a perfectinist my nature. I focused too much on low level that I was not able to make a real submission. I have wasted too much time focusing on the wrong stuff. 
- Burnout: Burnout is real and it kills creativity. I have found if I take some time off to relax and catch up on sleep, I come up with much better ideas. 

## Optimizing for Very Quick Experimentation
- I can test 50 combinations in hyperparameter space. This is partially fuelled by using TPU and partially by other 
- This was very important as the bottleneck was not being able to test out my ideas. 
Smart use of availible TPUs and focus on experimentation gave me a upper hand. 
You cannot compete with rich kagglers and nvidia backed researchers on raw compute power. Smart use of availible reources was my key strenghts. That is my best fighting chance, - that is until I become a rich kaggler or a nvidia backed researcher. I wonder how life is for someone with 64 gpus and 1024 core tpu pod. I don't want lack of hardware to be a bottleneck. 
I am not open sourcing the full repo because I don't want to lose my competive advantage in the future NLP competitions. 
I am a perfectionist and a hyper competitive person by nature. I either give my 100% or 0% to anything. I put so much thought and effort into this competition because I HAD to win this competition. The idea of even coming in 2nd was not enough. 

Made it up by working at way below minimum wage

Scaling out the software design: Ask abhisheak to make a video on it after winning? 
- Thanks for your kind comments. Would you consider making a video on it? 
Team up with Abhisheak and ask him to make a video on it later. Use it to jump start your YouTube career in Kaggle. 
IDEA: YouTube Channel