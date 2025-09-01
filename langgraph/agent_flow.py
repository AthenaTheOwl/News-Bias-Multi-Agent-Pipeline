# LangGraph agent flow planning
from langgraph.graph import StateGraph
from tools import summarizer_chain, critic_chain, writer_chain

graph = StateGraph()
graph.add_node("summarizer", summarizer_chain)
graph.add_node("critic", critic_chain)
graph.add_node("writer", writer_chain)

graph.set_entry_point("summarizer")
graph.add_edge("summarizer", "critic")
graph.add_edge("critic", "writer")

graph.set_finish_point("writer")
compiled_graph = graph.compile()

if __name__ == "__main__":
    result = compiled_graph.invoke({
        "article_text": "Put your raw article text here",
        "bias_score": 4,
        "flags": ["loaded language"]
    })
    print(result)
