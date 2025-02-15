from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load DeepSeek Coder model
model_name = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

def detect_bug(code_snippet):
    prompt = f"Analyze the following code and tell if it contains a bug. Answer 'Yes' or 'No'.\n\n{code_snippet}\n\nBug?:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return "Yes" in response  # If "Yes" is in the response, it's buggy

def suggest_fix(code_snippet):
    prompt = f"Fix the bug in the following code and return the corrected version:\n\n{code_snippet}\n\nFixed Code:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=200)
    fixed_code = tokenizer.decode(output[0], skip_special_tokens=True)
    return fixed_code

# Example test
if __name__ == "__main__":
    buggy_code = "def add_numbers(a, b): return a - b"
    print("Bug Detected:", detect_bug(buggy_code))
    print("Suggested Fix:", suggest_fix(buggy_code))