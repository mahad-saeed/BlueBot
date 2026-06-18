from src.retriever import retrieve

result = retrieve("What is the baggage allowance for Value fare?", debug=True)
print(f"is_relevant: {result.is_relevant}")
