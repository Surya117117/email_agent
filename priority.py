from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

IMPORTANT_SENDERS = {
    "owner@company.com",
    "hr@company.com"
}

KEYWORDS = {
    "urgent": 20,
    "deadline": 20,
    "interview": 25,
    "offer letter": 30,
    "meeting": 15,
}

IMPORTANT_EMAILS = [
    "Your interview has been scheduled",
    "payment deadline is tomorrow",
    "Meeting with management team",
    "Offer letter attached",
]

IMPORTANT_VECTORS = model.encode(IMPORTANT_EMAILS)

def calculate_rule_score(email: dict) -> int:
    score = 0

    if email["sender"] in IMPORTANT_SENDERS:
        score += 30

    text = (
        email["subject"] + " " + email["body"]
    ).lower()

    for keyword, weight in KEYWORDS.items():
        if keyword in text:
            score += weight
    return score

def calculate_embedding_score(email:dict) -> float:
    email_text = (
        email["subject"] + " " + email["body"]
    )

    email_vector = model.encode(email_text)

    similarities = cosine_similarity(
        [email_vector],
        IMPORTANT_VECTORS
    )[0]

    return max(similarities)

def get_priority(email: dict) -> str:
    rule_score = calculate_rule_score(email)

    embedding_score = calculate_embedding_score(email)

    final_score = rule_score + (embedding_score * 30)

    if final_score >= 60:
        priority = "HIGH"

    elif final_score >=30:
        priority = "MEDIUM"
    
    else:
        priority = "LOW"
    
    return {
        "priority": priority,
        "rule_score": rule_score,
        "embedding_score": round(embedding_score, 3),
        "final_score":round(final_score, 2)
    }
