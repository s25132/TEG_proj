from langchain_community.graphs import Neo4jGraph
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from app.tools.tools_ranking import make_rank_tool
from app.tools.tools_matching import make_simple_match_tool
from app.tools.tools_graph import make_graph_qa_tool
from app.tools.tools_whatif import make_whatif_match_tool, make_compare_whatif_tool


# Prosty “store” historii w pamięci procesu (RAM)
_SESSION_STORE: dict[str, ChatMessageHistory] = {}

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = ChatMessageHistory()
    return _SESSION_STORE[session_id]

def setup_agent(model, graph: Neo4jGraph, qa_chain):
    graph_qa_tool = make_graph_qa_tool(qa_chain)
    rank_tool = make_rank_tool(graph)
    match_tool = make_simple_match_tool(graph)
    whatif_match_tool = make_whatif_match_tool(graph)
    compare_whatif_tool = make_compare_whatif_tool(whatif_match_tool)

    tools = [graph_qa_tool, rank_tool, match_tool, whatif_match_tool, compare_whatif_tool]

    prompt = ChatPromptTemplate.from_messages([
("system",
 "You are a helpful assistant.\n"
 "\n"
 "MANDATORY TOOL POLICY:\n"
 "\n"
 "0) FIRST classify the user intent into ONE of these categories:\n"
 "   A) WHAT-IF / SCENARIO intent (keywords: what-if, scenario, compare, diff, baseline, vs,\n"
 "      porównaj, scenariusz, co jeśli)\n"
 "   B) MATCHING intent (keywords: match, assign, staffed, staffing, allocate, who to assign,\n"
 "      who should be assigned, Polish: dopasuj, przypisz, przydziel)\n"
 "   C) RANKING intent (EXACT phrases only, see rule 5)\n"
 "   D) GENERAL GRAPH QA (everything else)\n"
 "\n"
 "FOR WHAT-IF SCENARIOS:\n"
 "- If the user mentions adding hypothetical or additional developers,\n"
 "  you MUST extract them into a JSON array called `extra_devs`.\n"
 "\n"
 "- Each extra developer MUST have this schema:\n"
 "  {{\n"
 "    \"personId\": string,\n"
 "    \"name\": string,\n"
 "    \"skills\": [string],\n"
 "    \"yearsExperience\": number,\n"
 "    \"projectCount\": number,\n"
 "    \"universityRanking\": number\n"
 "  }}\n"
 "\n"
 "- If the user provides incomplete data, you MUST infer reasonable defaults.\n"
 "- You MUST pass `extra_devs` to the tool call.\n"
 "- When calling compare_baseline_vs_whatif_for_rfp or match_devs_to_rfp_scored_whatif,\n"
 " ALWAYS include `extra_devs` explicitly (use [] if none).\n"
 "- For intent A: BEFORE calling the tool, you MUST extract rfpTitle and extra_devs from the user input,\n"
 " then call the tool with those exact values.\n"
 "\n"
 "1) If intent is A (WHAT-IF / SCENARIO):\n"
 "   - DO NOT call graph_qa.\n"
 "   - If the user asks to compare baseline vs what-if or asks for a diff,\n"
 "     you MUST call compare_baseline_vs_whatif_for_rfp.\n"
 "   - Otherwise, you MUST call match_devs_to_rfp_scored_whatif.\n"
 "\n"
 "2) If intent is B (MATCHING):\n"
 "   - DO NOT call graph_qa.\n"
 "   - Prefer scored matching: call match_devs_to_rfp_scored\n"
 "     (rfpTitle=<RFP title extracted from user input>).\n"
 "   - If the user explicitly asks for the simplest match, call match_devs_to_rfp_simple.\n"
 "\n"
 "3) If intent is D (GENERAL GRAPH QA):\n"
 "   - You MUST call graph_qa(question) first.\n"
 "\n"
 "4) You MUST call rank_best_devs_university ONLY if the user's input is EXACTLY one of:\n"
 "   - 'List developers with their project counts and university rankings'\n"
 "   - 'Give me best developers based on project counts and university rankings'\n"
 "   (case-insensitive, optional trailing punctuation)\n"
 "\n"
 "5) Never call rank_best_devs_university for any other question.\n"
 "\n"
 "6) Respond with a human-friendly final answer.\n"
),
MessagesPlaceholder(variable_name="chat_history"),
("human", "{input}"),
MessagesPlaceholder(variable_name="agent_scratchpad"),
])


    agent = create_openai_tools_agent(llm=model, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True
    )

    executor_with_history = RunnableWithMessageHistory(
        executor,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return executor_with_history
