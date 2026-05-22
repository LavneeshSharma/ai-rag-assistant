def classify_query(query):

    broad_keywords = [
        "list",
        "all",
        "summarize",
        "overview",
        "projects",
        "topics"
    ]

    query_lower = query.lower()

    for keyword in broad_keywords:
        if keyword in query_lower:
            return "broad"

    return "specific"