python finetune/lora.py ^
--data_dir data/alpaca ^
--checkpoint_dir checkpoints\meta-llama\Llama-2-7b-chat-hf ^
--out_dir out/lora_weights/Llama-2-7b-chat-hf ^
--precision bf16-true 
rem --quantize bnb.nf4	