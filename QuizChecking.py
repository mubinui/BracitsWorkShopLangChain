import pandas as pd

# Load the XLSX file
input_file = "quiz_responses.xlsx"  # Change this to your actual input file
output_file = "graded_quiz.xlsx"  # Output file

df = pd.read_excel(input_file)

# Define the correct answers (normalize spaces for consistency)
correct_answers = {
    "What is the primary function of a Large Language Model (LLM)?": "Generating and understanding human-like text",
    "Which architecture do most LLMs use?": "Transformer Architecture",
    "What is the key mechanism in the Transformer model?": "Self-Attention",
    "What is tokenization in LLMs?": "Splitting text into smaller units like words or sub-words",
    "Which of the following is a prompt engineering technique?": "Chain of Thought (COT)",
    "What is embedding in LLMs?": "A process to convert text into numerical representations",
    "What is the purpose of fine-tuning an LLM?": "To optimize the model for specific tasks or domains",
    "What is Quantization in LLMs?": "Reducing model size and computational requirements",
    "What is Ollama used for?": "Running local LLMs",
    "What does Reinforcement Learning from Human Feedback (RLHF) help with?": "Improving the quality of model-generated responses"
}

# Normalize column names by stripping spaces and newlines
df.columns = df.columns.str.strip()

# Function to calculate scores with case-insensitive and space-normalized comparison
def calculate_score(row):
    score = 1
    for question, correct_answer in correct_answers.items():
        if question in row:
            user_answer = str(row[question]).strip().lower() if pd.notna(row[question]) else ""
            correct_answer = correct_answer.strip().lower()
            if user_answer == correct_answer:
                score += 1
    return score

# Apply scoring to each row
df["Score"] = df.apply(calculate_score, axis=1)

# Save the results as an Excel file (.xlsx)
df.to_excel(output_file, index=False)

print(f"Grading complete! Results saved to {output_file}")