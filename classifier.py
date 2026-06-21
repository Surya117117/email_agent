from keywords import keyword_classifier
from embeddings import embedding_classifier

def classify_email(email_text):
    category = keyword_classifier(email_text)
    if category:
        return category
    
    category, score = embedding_classifier(
        email_text
    )
    if score > 0.75:
        return category
    
    return "unable to classify"
