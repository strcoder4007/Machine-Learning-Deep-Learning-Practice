from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers import pipeline
import torch

model_id = "MaziyarPanahi/Phi-3-mini-4k-instruct-v0.3"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    # attn_implementation="flash_attention_2"
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

streamer = TextStreamer(tokenizer)

messages = [
    {"role": "system", "content": "You are a business chatbot who always responds in business lingo!"},
    {"role": "user", "content": "Who are you?"},
]

# this should work perfectly for the model to stop generating
terminators = [
    tokenizer.eos_token_id, # this should be <|im_end|>
    tokenizer.convert_tokens_to_ids("<|assistant|>"), # sometimes model stops generating at <|assistant|>
    tokenizer.convert_tokens_to_ids("<|end|>") # sometimes model stops generating at <|end|>
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
    "streamer": streamer,
    "eos_token_id": terminators,
}

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])