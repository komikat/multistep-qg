import ijson
from llama_cpp import Llama

f = open("../hotpot_train_v1.1.json")
json = f.read()
a = ijson.items(json, "item")

llm = Llama(
      model_path="../../../llama-3.2-1b-instruct-q6_k.gguf",
      n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      n_ctx=4096, # Uncomment to increase the context window
)

def c():
    s = next(a)
    q = s["question"]
    cont = " ".join([" ".join(cont[1]) for cont in s["context"]])
    print(q)
    answer = s["answer"]
    print(answer)
    return generate(cont, answer)

def generate(context, answer):
    return llm(
      f"""You are a helpful question generator. Given some context and the correct answer, you need to generate what could have been the original question.
      ***context: {context}***
      ***answer: {answer}***
      ***question:""",
      max_tokens=100,
      stop=["question:", "\n"], # Stop generating just before the model would generate a new question
      echo=False
    )["choices"][0]["text"]

### call llama for the next json item this way:
if __name__ == "__main__":
    print(c())