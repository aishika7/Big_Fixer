# git clone https://github.com/jkoppel/QuixBugs

# pip install datasets
from datasets import load_dataset

# Load the code_x_glue_cc_clone_detection_big_clone_bench dataset
dataset = load_dataset("code_x_glue_cc_clone_detection_big_clone_bench")

# Print the dataset information
print(dataset)

# pip install datasets
from datasets import load_dataset
from transformers import AutoTokenizer

# Load the code_x_glue_cc_clone_detection_big_clone_bench dataset
dataset = load_dataset("code_x_glue_cc_clone_detection_big_clone_bench")

# Print the dataset information (optional)
print(dataset)

# Use CodeT5-small tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")

# Function to tokenize code
def preprocess_data(example):
    # Assuming the dataset has 'func1' and 'func2' for code and 'id1','id2', and 'label' columns.
    # Modify these column names if your dataset has different names.
    code1 = example["func1"]
    code2 = example["func2"]
    inputs = tokenizer(code1, code2, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    # Concatenate code1 and code2 if necessary for your model
    # Adjust how 'labels' are assigned based on your dataset's structure
    inputs["labels"] = example["label"] # Removed extra brackets to assign the list of labels directly.
    return inputs

# Apply preprocessing
dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import os # Import os module

# et the WANDB_API_KEY environment variable:
os.environ["WANDB_API_KEY"] = "f719d44b4ec067ea3c144a8eb1180f8bed102f3e" # Replaced with actual API key

# Load CodeT5 model for classification
model = AutoModelForSequenceClassification.from_pretrained("Salesforce/codet5-small", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    push_to_hub=False,
    report_to="wandb"  # Enable wandb logging explicitly
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# Train the model
trainer.train()

def predict_bug(code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Buggy" if prediction == 1 else "Correct"

# Example buggy Java code
test_code = """
public class Example {
    public static void main(String[] args) {
        int a = 10;
        int b = 0;
        System.out.println(a / b); // Division by zero error
    }
}
"""

print("Prediction:", predict_bug(test_code))

from transformers import AutoModelForSeq2SeqLM

# Load CodeT5 model for code generation
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")

def fix_code(buggy_code):
    inputs = tokenizer("Fix: " + buggy_code, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    output = model.generate(**inputs, max_length=512)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test the model with a buggy snippet
print("Fixed Code:\n", fix_code(test_code))