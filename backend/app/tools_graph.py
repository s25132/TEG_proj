from typing import Dict, Any

from langchain_core.tools import tool


def make_graph_qa_tool(qa_chain):
    @tool
    def graph_qa(question: str) -> Dict[str, Any]:
        """
        Answer a question using GraphCypherQAChain over Neo4j.
        Returns answer + cypher_query + retrieved_contexts.
        """
        res = qa_chain.invoke({"query": question})

        out = {"answer": "", "cypher_query": "", "retrieved_contexts": []}

        if isinstance(res, dict):
            out["answer"] = res.get("result", "")

            steps = res.get("intermediate_steps", [])
            if len(steps) > 0 and isinstance(steps[0], dict):
                out["cypher_query"] = steps[0].get("query", "") or ""

            if len(steps) > 1 and isinstance(steps[1], dict):
                ctx = steps[1].get("context", [])
                if isinstance(ctx, list):
                    out["retrieved_contexts"] = [
                        ", ".join(f"{k}={v}" for k, v in rec.items()) if isinstance(rec, dict) else str(rec)
                        for rec in ctx
                    ]
        else:
            out["answer"] = str(res)

        return out

    return graph_qa