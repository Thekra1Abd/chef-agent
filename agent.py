import sys
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

def run_chef_agent_cli():
    # 1. Initialize the local Ollama model (Make sure 'ollama run llama3' is executed in your terminal)
    try:
        model = Ollama(model="llama3")
    except Exception as e:
        print(f"Failed to initialize Ollama: {e}")
        sys.exit(1)

    # 2. Define the Agent's persona and logic constraints via System Prompt
    system_instruction = """
    You are a professional, creative Personal Chef AI Agent. 
    Your mission is to help users innovate meals based on the ingredients they have in their fridge.
    
    Rules:
    1. Analyze the ingredients provided by the user.
    2. Suggest 2 realistic meals that can be cooked using these ingredients.
    3. For each meal, briefly list the extra basic pantry items needed (like salt, oil, or water) and short preparation steps.
    4. Maintain the conversation context using the chat history. If the user asks follow-up questions like 'How do I cook the first one?', refer back to your previous response.
    5. give the steps with the needed time required for each step.
    """

    # 3. Setup Chat Prompt Template with Memory Layer Injection
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    # 4. Bind components using LangChain Expression Language (LCEL)
    chain = prompt_template | model

    # 5. Initialize the state holder for short-term memory (Chat History Log)
    chat_history = []

    print("\n==================================================")
    print("      WELCOME TO THE INTERACTIVE CHEF AGENT       ")
    print("==================================================")
    print("The Chef is ready! Type your ingredients to start.")
    print("Type 'exit' or 'quit' to end the session.\n")

    # 6. Execution Loop for the Multi-turn CLI Application
    while True:
        try:
            user_input = input("\nYou: ")
            
            # Check for termination keywords
            if user_input.strip().lower() in ['exit', 'quit']:
                print("\nChef: Bye!")
                break
                
            if not user_input.strip():
                continue

            print("\nChef is thinking...")

            # Run the generation chain with historical context tracking
            response = chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })

            print("\n--- Chef Suggestion ---")
            print(response)
            print("-----------------------")

            # Update history cache with human query and AI response turns
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response))

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred during invocation: {e}")

if __name__ == "__main__":
    run_chef_agent_cli()