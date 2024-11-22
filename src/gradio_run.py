import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from safetensors.torch import load_file
from collections import OrderedDict

checkpoint_path = "/scratch/arjun.dosajh/llama_finetune/run_random_hops/checkpoint-1400"
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

state_dict = load_file(f"{checkpoint_path}/model.safetensors")

# Remove the 'module.' prefix from state dict keys
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace('module.', '') if k.startswith('module.') else k
    new_state_dict[name] = v

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-3.2-1B',  # Your base model
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map="auto"
)

# Load the cleaned state dict
missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
print(f"\nMissing keys: {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def generate_response(prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt, "").strip()

def create_decompose_prompt(question, context):
    return f"""Break down the following complex question into simple, single-hop questions that can be answered independently.

Complex Question: {question}
Context: {context}

Simple questions:"""

def create_answer_prompt(questions, context):
    return f"""Answer the following simple questions using the provided context.

Questions: {questions}
Context: {context}

Answers:"""

def process_question(question, context):
    context = context or "No context provided"
    
    # Get simple questions
    decompose_prompt = create_decompose_prompt(question, context)
    simple_questions = generate_response(decompose_prompt)
    
    # Get final answer
    answer_prompt = create_answer_prompt(simple_questions, context)
    final_answer = generate_response(answer_prompt)
    
    return simple_questions, final_answer

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Question Decomposition and Answering System")
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(label="Complex Question", lines=3)
            context_input = gr.Textbox(label="Context (Optional)", lines=5)
            submit_btn = gr.Button("Process")
            
        with gr.Column():
            simple_questions_output = gr.Textbox(label="Simple Questions", lines=5)
            final_answer_output = gr.Textbox(label="Final Answer", lines=5)
    
    submit_btn.click(
        fn=process_question,
        inputs=[question_input, context_input],
        outputs=[simple_questions_output, final_answer_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)