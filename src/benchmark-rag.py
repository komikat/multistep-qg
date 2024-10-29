import ijson
import torch
from transformers import pipeline, AutoTokenizer
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import nltk
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt', quiet=True)

# Initialize the tokenizer with correct padding configuration
model_id = "meta-llama/Llama-3.2-1B-Instruct"
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("Model loaded successfully!")

def create_batches(lst, batch_size):
    """Create batches from a list"""
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

def calculate_scores(generated_question, original_question):
    """Calculate BLEU and ROUGE scores between generated and original questions"""
    try:
        # Calculate BLEU score
        reference = [tokenize(original_question)]
        candidate = tokenize(generated_question)
        bleu_score = sentence_bleu(reference, candidate)
        
        # Calculate ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(original_question, generated_question)
        
        return {
            'bleu_score': bleu_score,
            'rouge1_f': rouge_scores['rouge1'].fmeasure,
            'rouge2_f': rouge_scores['rouge2'].fmeasure,
            'rougeL_f': rouge_scores['rougeL'].fmeasure
        }
    except Exception as e:
        print(f"Error in calculate_scores: {str(e)}")
        print(f"Generated question: {generated_question}")
        print(f"Original question: {original_question}")
        return None

def tokenize(text):
    """Tokenize the text for BLEU score calculation"""
    return nltk.word_tokenize(text.lower())

def process_dataset(json_file, sample_size=0.05, batch_size=8):
    """Process the dataset and calculate scores using proper batch processing"""
    # First pass to count total items
    print("Counting total items in dataset...")
    total_items = sum(1 for _ in ijson.items(open(json_file, 'r'), 'item'))
    
    # Calculate number of items for 5% sample
    num_samples = int(total_items * sample_size)
    sample_interval = int(1/sample_size)
    
    print(f"Total items: {total_items}")
    print(f"Taking {num_samples} items ({sample_size*100}% sample)")
    
    # Read and prepare data
    print("Loading sampled data from JSON file...")
    data = []
    with open(json_file, 'r') as f:
        parser = ijson.items(f, 'item')
        for i, item in enumerate(tqdm(parser, desc="Loading items")):
            if i % sample_interval == 0:
                context = " ".join([" ".join(cont[1]) for cont in item["context"]])
                data.append({
                    'context': context,
                    'answer': item['answer'],
                    'original_question': item['question']
                })
    
    print(f"\nLoaded {len(data)} items. Processing in batches...")
    
    # Prepare prompts
    prompts = []
    for item in data:
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful question generator. Given some context and the correct answer, you need to generate what could have been the original question."
            },
            {
                "role": "user",
                "content": f"Context: {item['context']}\nAnswer: {item['answer']}\nGenerate a question that would lead to this answer:"
            }
        ]
        prompts.append(prompt)
    
    # Process in batches
    print(f"\nGenerating questions (batch size: {batch_size})...")
    all_outputs = []
    batches = create_batches(prompts, batch_size)
    
    # Process each batch
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        try:
            with torch.no_grad():
                outputs = pipe(
                    batch,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    batch_size=len(batch),
                    pad_token_id=tokenizer.pad_token_id,
                )
                all_outputs.extend(outputs)
                print(f"\nProcessed batch {batch_idx + 1}/{len(batches)}")
                print(f"Batch size: {len(batch)}")
                print(f"Outputs in this batch: {len(outputs)}")
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            print(f"Batch content: {batch}")
            continue
    
    print(f"\nTotal outputs collected: {len(all_outputs)}")
    print("\nCalculating scores...")
    
    results = []
    for i, (output, item) in enumerate(tqdm(zip(all_outputs, data), total=len(data), desc="Calculating scores")):
        try:
            generated_text = output[0]['generated_text'][-1]['content']
            generated_question = generated_text.split("Generate a question that would lead to this answer:")[-1].strip()
            
            # Print first few examples to verify output format
            if i < 3:
                print(f"\nExample {i + 1}:")
                print(f"Generated: {generated_question}")
                print(f"Original: {item['original_question']}")
            
            scores = calculate_scores(generated_question, item['original_question'])
            if scores is not None:
                result_dict = {
                    'original_question': item['original_question'],
                    'generated_question': generated_question,
                    'answer': item['answer'],
                    **scores
                }
                results.append(result_dict)
                
                # Print first few results to verify structure
                if i < 3:
                    print(f"Scores: {scores}")
        except Exception as e:
            print(f"Error processing item {i}: {str(e)}")
            continue
    
    print(f"\nTotal results collected: {len(results)}")
    
    if not results:
        print("Warning: No results were collected!")
        return pd.DataFrame()
    
    print("Creating final DataFrame...")
    df = pd.DataFrame(results)
    print(f"DataFrame shape: {df.shape}")
    print("DataFrame columns:", df.columns.tolist())
    print("\nFirst few rows of scores:")
    print(df[['bleu_score', 'rouge1_f', 'rouge2_f', 'rougeL_f']].head())
    
    return df

def main():
    json_file = "../hotpot_train_v1.1.json"
    print("\n=== Starting Question Generation Pipeline ===\n")
    
    df = process_dataset(
        json_file,
        sample_size=0.1,
        batch_size=50
    )
    
    if df.empty:
        print("Error: DataFrame is empty!")
        return
    
    print("\nSaving results to CSV...")
    df.to_csv('question_generation_scores.csv', index=False)
    
    print("\nScore Summary Statistics:")
    print(df[['bleu_score', 'rouge1_f', 'rouge2_f', 'rougeL_f']].describe())
    
    print("\n=== Pipeline Complete ===")
    return df

if __name__ == "__main__":
    main()