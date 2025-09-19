import random
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import subprocess
import platform
import csv
import os
from pathlib import Path

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def generate_and_save_custom_csv(original_data, target_size=2000, file_path='custom_command_dataset.csv'):
    print("\n📥 Generating and saving custom CSV dataset...")
    
    is_windows = platform.system() == "Windows"
    
    names = ["Ankit Kushwaha", "User", "Admin"]
    actions = ["list", "show", "create", "delete", "copy", "move", "search", "count", "check", "display"]
    objects = ["files", "directories", "processes", "disk usage", "lines", "logs", "users", "services"]
    modifiers = ["hidden", "all", "first 10", "last 20", "human-readable", "sorted", "recursive"]
    locations = [
        "C:\\Users\\Ankit Kushwaha" if is_windows else "/home/user/Documents",
        "C:\\ProgramData\\Logs" if is_windows else "/var/log",
        "C:\\Temp" if is_windows else "/tmp",
        ".\\backup" if is_windows else "./backup",
    ]

    templates = [
        "{action} {object} in the current directory",
        "{action} {modifier} {object}",
        "{action} a new {object} named {name}",
        "{action} {object} containing {name}",
        "{action} the {object} to {location}",
        "{action} {object} for {name}",
        "{action} {modifier} {object} in {location}",
    ]

    prompts = []
    commands = []
    original_prompts = [item["prompt"] for item in original_data]
    original_commands = [item["command"] for item in original_data]

    prompts.extend(original_prompts)
    commands.extend(original_commands)

    while len(prompts) < target_size:
        template = random.choice(templates)
        prompt = template.format(
            action=random.choice(actions),
            object=random.choice(objects),
            modifier=random.choice(modifiers),
            name=random.choice(names),
            location=random.choice(locations),
        )
        if "list" in prompt or "show" in prompt or "display" in prompt:
            if "files" in prompt or "directories" in prompt:
                command = "dir" if is_windows else "ls"
            elif "processes" in prompt:
                command = "tasklist" if is_windows else "ps aux"
            elif "disk usage" in prompt:
                command = "dir" if is_windows else "df -h"
            elif "lines" in prompt:
                command = "find /c /v \"\"" if is_windows else "wc -l"
            elif "logs" in prompt:
                command = "type" if is_windows else "cat"
            elif "users" in prompt:
                command = "net user" if is_windows else "who"
            elif "services" in prompt:
                command = "net start" if is_windows else "systemctl list-units"
            else:
                command = "dir" if is_windows else "ls"
            if "hidden" in prompt:
                command += " /a:h" if is_windows else " -a"
            if "sorted" in prompt:
                command += " /on" if is_windows else " -l | sort"
            if "recursive" in prompt:
                command += " /s" if is_windows else " -R"
            if "location" in prompt and "current directory" not in prompt:
                command += f" \"{prompt[prompt.find('in ') + 3:]}\"" if is_windows else f" {prompt[prompt.find('in ') + 3:]}"
        elif "create" in prompt:
            command = f"mkdir {random.choice(names)}" if "directories" in prompt else f"echo. > {random.choice(names)}.txt" if is_windows else f"touch {random.choice(names)}.txt"
        elif "delete" in prompt or "remove" in prompt:
            command = (
                f"del {random.choice(names)}.txt" if is_windows else f"rm {random.choice(names)}.txt"
                if "files" in prompt
                else f"rmdir {random.choice(names)}"
            )
        elif "copy" in prompt:
            command = (
                f"copy {random.choice(names)}.txt \"{random.choice(locations)}\""
                if is_windows
                else f"cp {random.choice(names)}.txt {random.choice(locations)}"
            )
        elif "move" in prompt:
            command = (
                f"move {random.choice(names)}.txt \"{random.choice(locations)}\""
                if is_windows
                else f"mv {random.choice(names)}.txt {random.choice(locations)}"
            )
        elif "search" in prompt:
            command = (
                f"findstr \"{random.choice(names)}\" \"{random.choice(locations)}\\*.txt"
                if is_windows
                else f"grep '{random.choice(names)}' {random.choice(locations)}/*"
                if "files" in prompt or "logs" in prompt
                else f"tasklist | findstr {random.choice(names)}" if is_windows else f"ps aux | grep {random.choice(names)}"
            )
        elif "count" in prompt:
            command = (
                f"find /c /v \"\" {random.choice(names)}.txt"
                if is_windows
                else f"wc -l {random.choice(names)}.txt"
                if "lines" in prompt
                else "dir | find /c \"dir\"" if is_windows else "ls | wc -l"
            )
        elif "check" in prompt:
            command = (
                "dir" if is_windows else "df -h"
                if "disk usage" in prompt
                else "sc query" if is_windows else "systemctl status"
                if "services" in prompt
                else "echo %CD%" if is_windows else "pwd"
            )
        else:
            command = "echo Unknown command"
        if prompt not in prompts:
            prompts.append(prompt)
            commands.append(command)

    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['prompt', 'command'])
        for p, c in zip(prompts[:target_size], commands[:target_size]):
            writer.writerow([p, c])
    
    print(f"✅ Custom CSV saved to {file_path} with {target_size} examples.")

original_data = [
    {"prompt": "List all files in the current directory", "command": "dir"},
    {"prompt": "Show hidden files too", "command": "dir /a:h"},
    {"prompt": "Print the current working directory", "command": "echo %CD%"},
    {"prompt": "Change directory to Documents", "command": "cd Documents"},
    {"prompt": "Create a new folder named test", "command": "mkdir test"},
    {"prompt": "Remove a file named old.txt", "command": "del old.txt"},
    {"prompt": "Copy file data.txt to backup.txt", "command": "copy data.txt backup.txt"},
    {"prompt": "Move file report.pdf to Documents", "command": "move report.pdf C:\\Users\\Ankit Kushwaha\\Documents"},
    {"prompt": "Show first 10 lines of log.txt", "command": "type log.txt | more +10"},
    {"prompt": "Show last 20 lines of log.txt", "command": "type log.txt | more -20"},
    {"prompt": "Search for the word error in log.txt", "command": "findstr \"error\" log.txt"},
    {"prompt": "Count number of lines in file.txt", "command": "find /c /v \"\" file.txt"},
    {"prompt": "Check disk usage in human readable form", "command": "dir"},
    {"prompt": "Display running processes", "command": "tasklist"},
    {"prompt": "Find python processes", "command": "tasklist | findstr python"},
    {"prompt": "Show hidden files in Documents", "command": "dir /a:h C:\\Users\\Ankit Kushwaha\\Documents"},
    {"prompt": "Create a new file named test.txt", "command": "echo. > test.txt"},
]

def fine_tune_model():
    try:
        # Generate dataset
        file_path = 'custom_command_dataset.csv'
        print("Generating new custom_command_dataset.csv...")
        generate_and_save_custom_csv(original_data, target_size=2000, file_path=file_path)
        dataset = load_dataset('csv', data_files=file_path, quoting=csv.QUOTE_MINIMAL)
        command_dataset = dataset['train']

        # Initialize model and tokenizer
        model_name = "t5-small"  # Use a known Seq2Seq model
        model_dir = "./command_results/fine_tuned_model"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Tokenize dataset
        def tokenize_command(examples):
            inputs = tokenizer(
                examples["prompt"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            targets = tokenizer(
                examples["command"],
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            inputs["labels"] = targets["input_ids"]
            return inputs

        command_dataset = command_dataset.map(tokenize_command, batched=True, num_proc=4)

        # Print sample of dataset
        print("Sample of command dataset:")
        sample_data = command_dataset[:5]
        for prompt, command in zip(sample_data["prompt"], sample_data["command"]):
            print(f"- Prompt: {prompt}")
            print(f"  Command: {command}")

        # Set format for training
        command_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./command_results/fine_tuned_v3",
            evaluation_strategy="no",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-5,
            warmup_steps=100,
            logging_dir="./logs",
            logging_steps=50,
            save_strategy="epoch",
            save_total_limit=1,
            report_to="none",
            fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU is available
        )

        # Use default data collator for Seq2Seq
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=command_dataset,
            data_collator=data_collator,
        )

        # Train the model
        print("🚀 Starting fine-tuning...")
        trainer.train()

        # Save the model and tokenizer
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"✅ Model and tokenizer saved to {model_dir}")

    except Exception as e:
        print(f"❌ Error during fine-tuning: {str(e)}")
        return None, None

    return model, tokenizer

class CommandAgent:
    def __init__(self, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def generate_command(self, prompt, max_length=64):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=5,
            early_stopping=True,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def execute_command(self, command):
        print(f"\n🔧 Executing command: {command}")
        try:
            if any(x in command for x in ["log.txt", ".txt"]):
                file_name = command.split()[-1].strip('"')
                if not os.path.exists(file_name):
                    return f"Execution Error: File {file_name} does not exist"
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
            return result.stdout or result.stderr
        except Exception as e:
            return f"Execution Error: {str(e)}"

def main():
    # Fine-tune the model
    model, tokenizer = fine_tune_model()
    if model is None or tokenizer is None:
        print("❌ Fine-tuning failed. Exiting...")
        return

    # Initialize the agent
    agent = CommandAgent(model, tokenizer)

    # Test the agent
    print("\n🧪 Testing the command agent...")
    test_prompts = [
        "List all files in the current directory",
        "Show hidden files too",
        "Print the current working directory",
        "Create a new folder named test",
        "Search for the word error in log.txt",
    ]

    for prompt in test_prompts:
        generated_command = agent.generate_command(prompt)
        print(f"Prompt: {prompt}")
        print(f"Generated Command: {generated_command}")
        execution_result = agent.execute_command(generated_command)
        print(f"Execution Result: {execution_result}")
        print("-" * 50)

if __name__ == "__main__":
    main()
