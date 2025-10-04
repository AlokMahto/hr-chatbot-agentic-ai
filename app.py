import uuid
import streamlit as st
from main import conversation_runnable_with_history, llm

st.set_page_config(
    page_title="HR Chatbot Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 HR Chatbot with Agentic AI")
st.markdown("Ask me about HR policies, holidays, leave rules, and more!")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This HR Chatbot uses **Agentic AI** to:
    
    🔍 **Tools Available:**
    - Search HR policy knowledge base
    - Get current date
    - Check holidays
    - Check if today is a holiday
    - Get upcoming holidays
    
    💡 **Try asking:**
    - "What is the leave policy?"
    - "What's today's date?"
    - "Is today a holiday?"
    - "Show me upcoming holidays"
    - "What are the public holidays?"
    """)
    
    st.divider()
    
    st.markdown(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    st.divider()
    
    # System status
    st.subheader("System Status")
    if llm:
        st.success("✅ LLM Connected")
    else:
        st.error("❌ LLM Not Available")
    
    if conversation_runnable_with_history:
        st.success("✅ Agent Initialized")
    else:
        st.error("❌ Agent Not Initialized")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about HR policies or holidays..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🤔"):
            try:
                if not conversation_runnable_with_history:
                    st.error("❌ Chatbot is not initialized. Please check your configuration.")
                else:
                    # Call the agent
                    result = conversation_runnable_with_history.invoke(
                        input={"input": prompt},
                        config={"configurable": {"session_id": st.session_state.session_id}}
                    )
                    
                    # Extract response
                    if isinstance(result, dict) and 'output' in result:
                        response = result['output']
                    elif isinstance(result, str):
                        response = result
                    else:
                        response = str(result)
                    
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

                
# --- Run Instructions ---
# Run with: streamlit run app.py