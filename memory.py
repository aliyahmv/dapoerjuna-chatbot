from langchain_community.memory import ConversationBufferMemory

# Default memory (k = 8 message window)
memory = ConversationBufferMemory(
    return_messages=True
)

def remember(input_text: str, output_text: str):
    memory.save_context(
        {"input": input_text},
        {"output": output_text}
    )
