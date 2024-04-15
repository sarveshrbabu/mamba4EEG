# mamba4EEG
Training Mamba on EEG and clinical notes/labelling data. 

I think the initial goal is to first be able to run a Mamba model correctly. 

Some experiments I have gotten training right now are: 
- Finetuning a MAMBA model using LORA (lora_finetune.py)
- With HuggingFace and without HuggingFace (with huggingFace is lora_finetune.py without HuggingFace is lora_pytorch_tune.py)
- With my huggingFace edits and without my huggingFace edits (right now the hugging face changes are in a branch called transformers changes)
- Some weird stuff is happening with distributed training where its not training properly - where it either uses 1 GPU more than all the others or just fails all together. Debugging this is becoming a little difficult and I don't know if my changes to the hugging face transformers library that allow for finetuning are correct. 

Things I'm working on: 

Task A: 
- replicating the Mamba code in a much more readable way so I truly understand the architecture myself. Pytorch + CUDA
- replicating S4 in code as well myself pytorch + CUDA 
- This is also sort of a prerequisite to being a Trainable "model". I can't run a training run on something I didn't write. 
- Also would help me fix bugs a lot faster I think - especially also because debugging is kinda hard because I don't fully understand what the code is doing (my edit to the transformers library that let me finetune a couple of weeks ago might just be wrong). 

Task B: 
- Training a MAMBA, S4 model on joint text-audio pairs, doing speech 2 text.
- Then seeing how it generalizes to EEG -> text, with EEG-text audio pairs. 
- Compare it to the parakeet architecture:
- Try replicating parakeet.  
https://nvidia.github.io/NeMo/blogs/2024/2024-01-parakeet/

Task C: 
- Finetuning a MAMBA model using QLORA (Can I quantize it?) (Mamba model LORA quantized is being a bit finnicky)
- Building out a finetuning harness for RLHF for MAMBA (this would be interesting) 
https://arxiv.org/pdf/2402.14740.pdf

Task D: 
diffusion model for EEG, where you can use energy based models (kind of like the audio generators that do diffusion -> 30/40khz) 

S4 EEG code is available in the lab (EEG/S4 stuff). 

We have a fairly standard EEG dataset (H5?) Get results on smaller numbers. 
VAE for EEG (bland EEG, where it could generate brain data). 
Speed the process for a VAE. 1 second of EEG would be at 200HZ, you can downsample it to how every channels you need. 

Task E: 
Simran Arora @ Christopher Re's: 
- How do I want to do this with EEG? Any thoughts on the EEG (long memory thingies), permanent memory of things that can be relevant. How it could sort of work on interacting with KV-caching?
- The thing that Mamba sucks at is specifcally recalling something from context (KV-caching) 
- 10 seconds long/3 minutes long. Seizure always starts with 10/20 second pattern. 1/10th 
- Lisa's Yamada research: Compression for this specific modality 
- Information content during a seizure (you can zip it and it'll be found). The pattern becomes more predictable. Optimizations doing compression really well.
- This network where you have token coming in -> compress larger segments and forms representations for that
- Network that does the encoding and compression and is being trained to do that. A member of its dictionary VQ system, where you have a set of centers of vectors and you can represent the signal with a lot of compression with this set of vectors. You could use this in a very useful way - to come up with candidates that might be seizures (and be able to remember specific activities in the input token space).
- Attention mechanism could be a recall mechanism (kinda like a dynamic RAG system/ classical information retreival). Network learns which things to store. It always has those in context. 

Normally full context is a 1k steps. Look at the most recent 100. 

intuition: 
A certain amount of blurring is initially okay. naively tried S4 for EEG seizure detection. There's a certain level of detail that needs to be maintained that can just be dissappearing in the past. You need to maintain medium resolution of the features to make a good comparison. 

Hippo initialization intuition: 
By representing the inputs in terms of these linearly invariant polynomial systems (matrices lmao), you can kind of nicely approximate these functions well. They represent the space over a long period of time. 

