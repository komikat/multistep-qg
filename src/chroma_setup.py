__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm.autonotebook import tqdm
import ijson

client = chromadb.PersistentClient(path="./../chromadb")
ef = SentenceTransformerEmbeddingFunction(model_name="thenlper/gte-small", device="cuda")
db = client.create_collection(name="src", embedding_function=ef)

with open("./../../data/hotpot/hotpot_train_v1.1.json") as f:
    json = f.read()
collection = ijson.items(json, "item")

for ds_id, docs in tqdm(enumerate(collection)):
    for d_id, doc in enumerate(docs["context"]):
        for c_id, chunk in enumerate(doc[1]):
            f_id = "-".join([str(s) for s in [ds_id, d_id, c_id]])
            db.add(documents=[chunk], ids=[f_id])

sample = next(collection)
sample["question"]
sample["answer"]