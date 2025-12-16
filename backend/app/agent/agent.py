from langchain_community.graphs import Neo4jGraph
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.tools.tools_ranking import make_rank_tool
from app.tools.tools_matching import make_simple_match_tool
from app.tools.tools_graph import make_graph_qa_tool

def setup_agent(model, graph: Neo4jGraph, qa_chain):
    graph_qa_tool = make_graph_qa_tool(qa_chain)
    rank_tool = make_rank_tool(graph)
    match_tool = make_simple_match_tool(graph)

    tools = [graph_qa_tool, rank_tool, match_tool]

    prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant.\n"
     "MANDATORY TOOL POLICY:\n"
     "0) FIRST decide whether the user is asking to match/assign people to an RFP.\n"
     "   Matching intent keywords include: match, assign, staffed, staffing, allocate, who to assign, who should be assigned,\n"
     "   and also Polish equivalents: dopasuj, przypisz, przydziel.\n"
     "1) If the user is asking to match/assign people to an RFP:\n"
     "   - DO NOT call graph_qa.\n"
     "   - You MUST call match_devs_to_rfp_simple(rfpTitle=<RFP title extracted from the user input>) as the ONLY tool call.\n"
     "2) Otherwise (not matching intent): you MUST call graph_qa(question) first.\n"
     "3) You MUST call rank_best_devs_university ONLY if the user's input is EXACTLY one of:\n"
     "   - 'List developers with their project counts and university rankings'\n"
     "   - 'Give me best developers based on project counts and university rankings'\n"
     "   (case-insensitive, optional trailing punctuation)\n"
     "4) Never call rank_best_devs_university for any other question.\n"
     "5) Respond with a human-friendly final answer.\n"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm=model, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
