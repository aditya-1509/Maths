import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Math & Search Assistant",
    page_icon="🧮",
    layout="centered"
)
st.title("🧮 Math Problem Solver & Search Assistant")

os.environ['API_KEY']=os.getenv("API_KEY")

with st.sidebar:
    st.header("Configuration")
    st.markdown("The App is configured with Groq API key and is ready to start.")
    st.markdown("##Groq API KEY -")
    st.markdown('xxxxxxxxxxxxxxxxxxxxx')
    groq_api_key = os.environ['API_KEY']
    st.markdown("---")
    st.markdown(
        "This app uses the Llama 3.1 model via Groq to solve math word problems and search for information using Wikipedia."
    )

if not groq_api_key:
    st.info("Internal Server error! Reload the Page.")
    st.stop()


llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key, temperature=0)



wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A useful tool for searching the internet to find information on various topics."
)


math_chain = LLMMathChain.from_llm(llm=llm)
calculator_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A useful tool for answering math-related questions. This tool can handle complex mathematical expressions. Use this for any math calculations."
)


FORMAT_INSTRUCTIONS = """To use a tool, you MUST use the following format:

```
Thought: Do I need to use a tool? Yes. I should use the [tool name] tool to answer this.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have the final answer to the user's question, or if you do not need to use a tool,
you MUST use the format:

```
Thought: Do I need to use a tool? No. I have the final answer.
Final Answer: [your final, detailed, and well-explained answer here]
```

**CRITICAL INSTRUCTION**: When using the 'Calculator' tool, the 'Action Input' MUST be a pure mathematical expression and nothing else.
For example: `Action Input: 2 * (25 + 5)`
**DO NOT** write explanations in the Action Input, for example: `Action Input: Calculate 2 times 25`.
"""


tools = [wikipedia_tool, calculator_tool]


agent_kwargs = {
    "format_instructions": FORMAT_INSTRUCTIONS,
}

assistant_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True, 
    handle_parsing_errors=True, 
    agent_kwargs=agent_kwargs,
    max_iterations=5, 
    early_stopping_method="generate" 
)


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm an assistant who can solve math problems and search for information. How can I help you today?"}
    ]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])


question = st.chat_input("Enter your question here...", key="chat_input")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    with st.chat_message("assistant"):
        response_container = st.container()
        
        with st.spinner("Thinking..."):
         
            st_cb = StreamlitCallbackHandler(response_container, expand_new_thoughts=False)
            
          
            response_data = assistant_agent.invoke(
                {"input": question},
                {"callbacks": [st_cb]}
            )
            
            # The final answer is in the 'output' key of the response dictionary
            response = response_data['output']
            
            # Display the final answer and append to session state
            st.session_state.messages.append({'role': 'assistant', "content": response})
            response_container.markdown(response)

