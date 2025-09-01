from vectorstore.faiss_index import FAISSSearch

def main():
    search = FAISSSearch()
    print(f"Currently indexed documents: {len(search.metadata)}")

    query = input("Enter search query (or type 'clear' to reset index): ").strip()
    if query.lower() == "clear":
        confirm = input("Are you sure you want to clear the index? (y/n): ").strip()
        if confirm.lower() == "y":
            search.clear()
        return

    bias_filter = input("Filter by bias (Neutral/Left/Right/Mixed/Undetermined or leave blank): ").strip()
    results = search.search(query, top_k=5)

    if not results:
        print("No results found.")
        return

    for meta, distance in results:
        if bias_filter and bias_filter.lower() != meta["bias"].lower():
            continue
        print("\n=== Result ===")
        print(f"Headline: {meta['headline']}")
        print(f"URL: {meta['url']}")
        print(f"Bias: {meta['bias']}")
        print(f"Distance: {distance:.4f}")
        print(f"Report excerpt: {meta['report'][:200]}...")

if __name__ == "__main__":
    main()
