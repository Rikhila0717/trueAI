from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import nltk
import json
import matplotlib.pyplot as plt

nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import wordnet

#load RAGTruth dataset from huggingface (public, no need of API key)
ds = load_dataset("wandb/RAGTruth-processed")
print(ds)


#info about the number of testing and training samples, columns
for split,d in ds.items():
    print(f"{split:10s} | rows: {d.num_rows:6d} | columns: {d.column_names}")

#inspect the context column

print(json.dumps(ds["train"][0], indent=2, ensure_ascii=False))
uniq_train = ds["train"].unique("context")
num_uniq_train = len(uniq_train)
all_train_contexts = len(ds["train"]["context"])
uniq_test = ds["test"].unique("context")
num_uniq_test = len(uniq_test)
all_test_contexts = len(ds["test"]["context"])


print("unique contexts in train: ", num_uniq_train)
print("total contexts in train: ",all_train_contexts)
print("unique contexts in test: ", num_uniq_test)
print("total contexts in test: ",all_test_contexts)

# this part of the code has duplicates in the document as well
# create the corpus(documents)
# docs_ds = Dataset.from_dict({
#     "doc_id": list(range(ds["train"].num_rows)),
#     "text": ds["train"]["context"]
# })

docs_ds = Dataset.from_dict({
    "doc_id": list(range(len(ds["train"].unique("context")))),
    "text": ds["train"].unique("context"),
})

#verify doc data
print(type(docs_ds))
print(docs_ds)


# clean doc corpus - remove whitespaces
def clean_ds(ex):
    text = ex["text"].strip().replace("\n", " ")
    # remove a trailing 'output:' or 'Output:' if present
    text = re.sub(r"\s*[Oo]utput:\s*$", "", text)
    ex["text"] = text
    return ex

docs_ds = docs_ds.map(clean_ds)
print("Document cleaning successful")


#initialize sentence-transformer for embeddings
embedder = SentenceTransformer('all-mpnet-base-v2')
print("initialized embedder")

#cosine similarity in
"""
#generate embeddings and indices for the docs
#to use cosine similarity in FAISS, we can simply normalize the data and compute the inner product- cosine equivalent
#https://medium.com/@devbytes/similarity-search-with-faiss-a-practical-guide-to-efficient-indexing-and-retrieval-e99dd0e55e8c
"""

def embed_batch(batch):
    embs = embedder.encode(
        batch["text"],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return {"embeddings": embs.tolist() }

docs_ds = docs_ds.map(
    embed_batch,
    batched=True,
    batch_size=128,
)

print("Created embeddings")

docs_ds.add_faiss_index(
    column="embeddings",
    metric_type=faiss.METRIC_INNER_PRODUCT
)

print("FAISS indexing successful")

#retrieval
def retrieve(query, k=5):
    q_emb= embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)
    index = docs_ds.get_index("embeddings")
    distances,indices = index.search(q_emb,k)
    flat_idxs   = indices.flatten()
    flat_scores = distances.flatten()
    results = []
    for i, idx in enumerate(flat_idxs):
        results.append({
            "doc_id": int(idx),
            "text":   docs_ds[int(idx)]["text"],
            "score":  float(flat_scores[i])
        })
    return results

#validation - sanity checks on retrieval
"""
1. Manual Check:
Hand-pick random queries and eyeball the top-3 retrieved passages, any irrelevant hits can easily be observed
2. Random, Out-of-Domain Query Test:
For any garbage queries, the score should be low.
3. Synonyms Test
Replace terms with synonyms and check the similarity scores. They should be similar to each other

4. Recall@K and MRR
Recall@K - measures proportion of relevant passages in top-K retrieved passages out of all relevant passages
(number of relevant items/total number of relevant items)
MRR - Mean Reciprocal Rank - 1/Ri - Ri - the position of the relevant item
if MRR is greater, then the answer was in the top few results.
********* HOWEVER, THESE METRICS MAKE MORE SENSE WHEN WE ARE CREATING OUR OWN RETRIEVER WHICH IS CURRENTLY BEYOND THE
SCOPE OF OUR PROJECT. WE ARE CURRENTLY TRYING TO INTRODUCE FACT-CHECKING MECHANISMS WITHIN THE PIPELINE, SO WE ARE JUST
PERFORMING A SANITY-CHECK ON HOW WELL OUR RETRIEVAL FUNCTION IS WORKING ***********

5. Score distribution plot
Understand how the top-1 similarity cosine scores are distributed, and identify a potential threshold. Any cosine
similarity above 0.5 can mean an acceptable response.

"""

print("Starting Manual Check")
#1. Manual Check
sample_queries = [
    "What caused Anne Frank’s death?",
    "When was Bergen-Belsen liberated?",
    "How does typhus spread?",
    "Symptoms of epidemic typhus",
    "What is the role of lice in typhus transmission?"
]

for q in sample_queries:
    print("\n=== QUERY:", q)
    for rank, hit in enumerate(retrieve(q, k=3),start=1):
        print(f"{rank}. (cos={hit['score']:.3f}) {hit['text'][:200]}…")
print("manual check complete")

print("Starting random, gibberish query test")
#Random, Out-of-Domain Query Test
junk_queries = ["asdfghjkl", "What is the capital of Mars?", "qwertyuiop"]
for q in junk_queries:
    hits = retrieve(q, k=3)
    print(f"\nQ: {q}\nScores:", [round(h["score"],3) for h in hits])
print("completed random query test")


#Synonyms Test
def synonym_variants(word):
    syns = wordnet.synsets(word)
    words = {lemma.name().replace("_"," ") for s in syns for lemma in s.lemmas()}
    return list(words)[:5]

print("Starting synonym variants test")
base_query = "How does typhus spread?"
variants = ["typhus", "fever"]  # you can pick your own terms
for word in variants:
    for syn in synonym_variants(word):
        q = base_query.replace(word, syn)
        top_score = retrieve(q,1)[0]["score"]
        print(f"Variant: '{syn}' → top‐1 cos = {top_score:.3f}")
print("Synonym variants test complete")

#Score Distribution and Threshold on the test data
val_split = "test"
val_qs  = ds[val_split]["query"][:200]
top1 = np.array([retrieve(q,1)[0]["score"] for q in val_qs ])

#text summary
print("Top-1 cosine score  statistics:")
print(f"  min  = {top1.min():.3f}")
print(f"  25%  = {np.percentile(top1,25):.3f}")
print(f"  median = {np.median(top1):.3f}")
print(f"  mean   = {top1.mean():.3f}")
print(f"  75%  = {np.percentile(top1,75):.3f}")
print(f"  max  = {top1.max():.3f}")

#plot generation
plt.hist(top1, bins=20)
plt.title("Distribution of Top‐1 Cosine Scores")
plt.xlabel("Cosine Score")
plt.ylabel("Count")
plt.show()

"""
#Recall@K and MRR

print("Starting Recall@k and MRR validation")
#take a sample of 200 queries and their answers


#recall@K
def recall_at_k(k):
    hits=0 #the number of queries where we successfully find a relevant passage in the top K
    for q,gold in zip(val_qs, val_gold):
        #for each query and its corresponding answer
        retrieved = [h["text"] for h in retrieve(q,k)]
        #check if any of the k passages contain the first 50 characters of the gold answer(Expected)
        #we are not exactly strict-matching the characters - just enough to capture the reference
        if any (gold[:50] in txt for txt in retrieved):
            hits+=1
    #return the recall@k ratio
    return hits/len(val_qs)

#MRR
def mean_reciprocal_rank(k):
    rr_total = 0
    for q,gold in zip(val_qs, val_gold):
        retrieved = [h["text"] for h in retrieve(q,k)]
        rr=0
        #if no relevant passage is found for the query, then the reciprocal rank will remain 0
        #assign rank 1 to first, rank 2 to the second and rank k to the kth retrieved passage
        for rank,txt in enumerate(retrieved,start=1):
            if gold[:50] in txt:
                rr=1/rank
                break #as soon as the first relevant passage is found, break
        rr_total += rr
    #compute the average reciprocal ranks over all queries
    return rr_total/len(val_qs)

#Higher MRR - means the relevant context showed up earlier - how soon are we hitting the correct piece of evidence
for k in [1,3,5]:
    print(f"Recall@{k} = {recall_at_k(k):.3f}, MRR@{k} = {mean_reciprocal_rank(k): .3f}")

print("Completed Recall@k and MRR validation")
"""


# #chromadb code
# """
# #set up chroma client
# chroma_client = PersistentClient(path="chroma_db")
# print("initialized chroma client")
#
# #create a vectordb collection
# collection = chroma_client.get_or_create_collection(
#     name="ragtruth"
# )
# print("created chroma collection")
#
# #upsert the docs in a single batch
# ids = [str(d["doc_id"]) for d in docs_ds]
# documents = [d["text"] for d in docs_ds]
# metadatas = [{"doc_id":d["doc_id"]} for d in docs_ds]
#
# print("Starting upsert")
#
# collection.upsert(
#     ids=ids,
#     documents=documents,
#     metadatas=metadatas
# )
#
# print("Completed upsert")
# """




