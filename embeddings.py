from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-l6-v2"
)

CATEGORY_EXAMPLES = {
    "finance":[
        "invoice attached",
        "payment pending",
        "bill generated"
    ],

    "recruitment":[
        "interview scheduled",
        "candidate selected",
        "resume recieved"
    ],
    "support": [
        "unable to login",
        "application crashed",
        "account issue"
    ]
}

category_vectors = {}

for category,examples in CATEGORY_EXAMPLES.items():
    category_vectors[category] = model.encode(examples)

def embedding_classifier(email_text):
    email_vector = model.encode(email_text)
    best_category = None
    best_score = -1

    for category, vectors in category_vectors.items():
        scores = cosine_similarity(
            [email_vector],
            vectors
        )[0]
        score = np.max(scores)

        if score > best_score:
            best_score = score
            best_category = category
    
    return best_category, best_score