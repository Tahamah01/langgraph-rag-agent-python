from langchain_core.messages import HumanMessage
from agent import build_agent


session_id = "user-session_001"


def chat_loop():
    print("=" * 50)
    print("  AI Agent with Memory + RAG")
    print("=" * 50)
    print("Tip: run 'python ingest.py ./your_docs/' first")
    print("     to populate the knowledge base.\n")
    print("Type 'quit' to exit.\n")

    first_input = input('You: ')
    if not first_input or first_input.lower() == 'quit':
        return
    
    print('Agent Loading...')
    agent = build_agent(first_input)
    config = {'configurable': {'thread_id': session_id}}

    response = agent.invoke(
        {'messages': [HumanMessage(content=first_input)]},
        config=config
    )

    print(f"\nAgent: {response['messages'][-1].content}\n")

    while True:
        user_input = input('You: ').strip()
        if not user_input:
            continue
        if user_input.lower == 'quit':
            print('GoodBye...')
            break
        response = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        print(f"\nAgent: {response['messages'][-1].content}\n")



if __name__ == '__main__':
    chat_loop()
