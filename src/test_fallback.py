from pipeline import _is_fallback_answer

test_answer = "I don't have that information."
result = _is_fallback_answer(test_answer)
print(f"Result: {result}")

# Check for smart/curly apostrophe vs straight apostrophe
for c in test_answer:
    if not c.isalnum() and c not in (" ", "."):
        print(f"char: {repr(c)} hex: {hex(ord(c))}")