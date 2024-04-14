# mamba4EEG
Training Mamba on EEG and clinical notes/labelling data. 

I think the initial goal is to first be able to run a Mamba model correctly. 

Some experiments I have gotten training right now are: 
- Finetuning a MAMBA model using LORA (lora_finetune.py)
- With HuggingFace and without HuggingFace (with huggingFace is lora_finetune.py without HuggingFace is lora_pytorch_tune.py)
- With my huggingFace edits and without my huggingFace edits (right now the hugging face changes are in a branch called transformers changes)

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

