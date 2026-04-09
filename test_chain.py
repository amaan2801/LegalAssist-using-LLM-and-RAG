from src.chain.qa_chain import build_qa_chain, ask

print("Building chain...")
chain = build_qa_chain()

print("Asking question...")
result = ask(chain, "What is this document about?")

print("\nANSWER:")
print(result.answer)

print("\nSOURCES:")
for src in result.unique_sources:
    print("-", src)