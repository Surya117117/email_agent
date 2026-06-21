KEYWORDS = {
    "finance": [
        "invoice",
        "payment",
        "bill",
        "transaction"
    ],
    "recruitment":[
        "interview",
        "resume",
        "candidate",
        "hiring"
    ],
    "support": [
        "issue",
        "bug",
        "error",
        "help"
    ]
}

def keyword_classifier(email_text):
    email_text = email_text.lower()

    for category, words in KEYWORDS.items():
        for word in words:
            if word in email_text:
                return category
    return None
