import pandas as pd
import numpy as np

def analyze_scores(csv_path='question_generation_scores_llama_rag.csv'):
    # Read the CSV file
    print("Reading results from CSV...")
    df = pd.read_csv(csv_path)
    
    # Calculate mean scores
    means = {
        'BLEU': df['bleu_score'].mean(),
        'ROUGE-1': df['rouge1_f'].mean(),
        'ROUGE-2': df['rouge2_f'].mean(),
        'ROUGE-L': df['rougeL_f'].mean()
    }
    
    # Calculate standard deviations
    stds = {
        'BLEU': df['bleu_score'].std(),
        'ROUGE-1': df['rouge1_f'].std(),
        'ROUGE-2': df['rouge2_f'].std(),
        'ROUGE-L': df['rougeL_f'].std()
    }
    
    # Print results in a formatted way
    print("\nScore Analysis:")
    print("-" * 50)
    print(f"Total samples analyzed: {len(df)}")
    print("-" * 50)
    for metric in means.keys():
        print(f"{metric:8} = {means[metric]:.4f} ± {stds[metric]:.4f}")
    
    # Print some example predictions
    print("\nExample Predictions:")
    print("-" * 50)
    for i in range(min(3, len(df))):
        print(f"Example {i+1}:")
        print(f"Original:  {df['original_question'].iloc[i]}")
        print(f"Generated: {df['generated_question'].iloc[i]}")
        print(f"BLEU: {df['bleu_score'].iloc[i]:.4f}")
        print(f"ROUGE-1: {df['rouge1_f'].iloc[i]:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    analyze_scores()