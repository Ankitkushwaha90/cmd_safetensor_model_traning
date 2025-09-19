from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the fine-tuned model
model_dir = "./command_results/fine_tuned_model"
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Initialize the agent
agent = CommandAgent(model, tokenizer)

# Generate and execute a command
prompt = "List all files in the current directory"
command = agent.generate_command(prompt)
print(f"Prompt: {prompt}")
print(f"Generated Command: {command}")
print(f"Execution Result: {agent.execute_command(command)}")
