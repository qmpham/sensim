import faiss
import numpy as np
import argparse
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--vector_database", required=True , help="configuration file")
parser.add_argument("--text_database", required=True , help="configuration file")
parser.add_argument("--vector_query")
parser.add_argument("--text_query")
args = parser.parse_args()

database = np.load(args.vector_database)
query = np.load(args.vector_query)

with open(args.text_query,"r") as f:
    query_lines = [l.strip() for l in f.readlines()]

with open(args.text_database,"r") as f:
    database_lines = [l.strip() for l in f.readlines()]

v_db = database["sentence_embeddings"]
v_query = query["sentence_embeddings"]

v_db = normalize(v_db)
v_query = normalize(v_query)
cosine_similarity_ = cosine_similarity(v_query,v_db)
d = v_db.shape[-1]
assert v_query.shape[-1] == d
assert len(query_lines)==v_query.shape[0]
assert len(database_lines)==v_db.shape[0]
index = faiss.IndexFlatIP(d)   # build the index
print("faiss state: ", index.is_trained)
index.add(v_db)       # add vectors to the index
print("number of sentences in database: %d"%index.ntotal)
print("number of sentences in query: %d"%v_query.shape[0])
k = 5
D, I = index.search(v_query, k)     # tgt -> src search  
 
for i in range(len(I)):
    print("sentence: %s"%query_lines[i])
    print("%d neighbors: "%k)
    for j in range(k):
        print("\t - ",database_lines[I[i][j]])
        print("\t - at distance %f"%D[i][j])
    print("\t + real match: %s"%database_lines[i])
    print("\t at distance", cosine_similarity_[i,i])
print(accuracy_score(np.arange(v_query.shape[0]), I[:,0]))