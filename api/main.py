"""
main.py — interactive CLI for testing the Recipe RAG agent.
"""

import asyncio
from langchain_core.messages import HumanMessage
from graph import get_app


async def run():
    print("🍳 Recipe RAG Agent (type 'quit' to exit)\n")
    app = await get_app()

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        initial_state = {
            "user_question":    query,
            "messages":         [HumanMessage(content=query)],
            "streaming":        False,
            "candidate_recipes": [],
            "final_recipes":    [],
        }

        print("\n⏳ Processing...\n")
        result = await app.ainvoke(initial_state)

        intent    = result.get("user_intent", {})
        candidates = result.get("candidate_recipes", [])
        final     = result.get("final_recipes", [])
        answer    = result.get("final_answer", "No answer.")
        error     = result.get("error")

        print("─── DEBUG ──────────────────────────────────")
        print(f"Intent:      {intent}")
        print(f"Candidates:  {len(candidates)} recipes from Recipe Agent")
        print(f"Final:       {len(final)} recipes from Nutrition Agent")
        if error:
            print(f"Error:       {error}")
        print("─── ANSWER ─────────────────────────────────")
        print(answer)
        print("────────────────────────────────────────────\n")


if __name__ == "__main__":
    asyncio.run(run())
