import uuid
from langchain_core.messages import HumanMessage, AIMessageChunk
from agent import build_agent

QUIT_WORDS = {"quit", "exit"}


def ask(agent, text: str, config: dict) -> None:
    print('\nAgent: ', end="", flush=True)
    for chunk, _metadata in agent.stream(
        {"messages" : [HumanMessage(content=text)]},
        config=config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessageChunk) and isinstance(chunk.content, str) and chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")


def chat_loop() -> None:
    print("=" * 50)
    print(" AI Agent with Memory + RAG")
    print("=" * 50)
    print("Tip: run 'python rag.py ./your_docs/' first to populate the knowledge base.")
    print("Type 'quit' to exit.\n")

    first_input = input("You: ").strip()
    if not first_input or first_input.lower() in QUIT_WORDS:
        return

    print("Agent loading...")
    agent = build_agent(first_input)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    ask(agent, first_input, config)

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in QUIT_WORDS:
            print("Goodbye...")
            break
        ask(agent, user_input, config)


if __name__ == "__main__":
    try:
        chat_loop()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye...")
