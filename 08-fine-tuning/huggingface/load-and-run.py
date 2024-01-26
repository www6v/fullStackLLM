from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

# model_name_or_path="baichuan-inc/baichuan-7B"
model_name_or_path = "distilgpt2"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, trust_remote_code=True
).to(device)

inputs = tokenizer('I am', return_tensors='pt').to(device)

print("====inputs====")

print(inputs)

pred = model.generate(**inputs, max_new_tokens=10)

output = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)

print("====decoded====")

print(output)