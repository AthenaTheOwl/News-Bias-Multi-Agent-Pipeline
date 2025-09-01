# main.py
from langgraph.news_graph import run_news_graph

if __name__ == "__main__":
    q = input("Enter query (e.g., 'Singapore today' or 'Africa last year'): ").strip()
    if not q:
        q = "Singapore today"
    report = run_news_graph(q)
    print("\n=== FINAL REPORT ===\n")
    print(report)
