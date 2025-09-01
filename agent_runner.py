from langchain.agents import initialize_agent, AgentType, Tool
from models import llama3
from tools import summarizer_tool, critic_tool, writer_tool, search_tool, preprocessor_tool
from agents.preprocessor import run_preprocessor_agent  # ✅ Import direct preprocessor

tools = [
    Tool.from_function(preprocessor_tool, name="Preprocessor", description="Preprocess vague user queries into structured search requests"),
    Tool.from_function(summarizer_tool, name="SummarizeArticle", description="Summarize an article in 5 sentences or fewer"),
    Tool.from_function(critic_tool, name="BiasCritic", description="Critique article bias based on bias score and summary"),
    Tool.from_function(writer_tool, name="ArticleWriter", description="Write a professional article from summary and critique"),
    Tool.from_function(search_tool, name="SemanticSearch", description="Search recent stored summaries with FAISS/SQLite hybrid"),
]

agent_executor = initialize_agent(
    tools=tools,
    llm=llama3,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

def run_agent(prompt: str):
    print(f"\n🧠 Original prompt: {prompt}")
    
    # ✅ Step 1: Always preprocess first
    structured_prompt = run_preprocessor_agent(prompt)
    print(f"🔧 Preprocessed prompt: {structured_prompt}")
    
    # ✅ Step 2: Feed into main agent
    result = agent_executor.run(structured_prompt)
    
    print("✅ Agent finished.")
    return result
