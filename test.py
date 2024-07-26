import os
import json
import datetime
import asyncio
import streamlit as st
from groq import Groq
from langchain.memory import ConversationBufferMemory

try:
    with open('character-details.json', 'r', encoding='utf-8') as file:
        character_info = json.load(file)
except json.JSONDecodeError as e:
    st.error(f"Error loading JSON file: {e}")
    st.stop()

# Extract character names for dropdown
character_names = list(character_info.keys())

class ChatMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role, "content": self.content}

class ChatPipeline:
    def __init__(self, api_key, model_name='llama3-70b-8192'):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.memory = ConversationBufferMemory()

    async def preprocess(self, prompt, character_prompt, conversation_history):
        # Combine character prompt with user input and conversation history
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
        return f"As {character_prompt}, continue the following conversation:\n{history}\nUser: {prompt}"

    async def _forward(self, prompt):
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
        )
        return response

    async def postprocess(self, response):
        ai_response = response.choices[0].message.content
        return ai_response

    async def __call__(self, prompt, character_prompt, conversation_history):
        processed_prompt = await self.preprocess(prompt, character_prompt, conversation_history)
        response = await self._forward(processed_prompt)
        return await self.postprocess(response)

async def get_ai_response(user_input, character_prompt):
    if "my name is" in user_input.lower():
        name = user_input.split("my name is")[-1].strip()
        st.session_state.user_name = name

    ai_response = await st.session_state.chat_pipeline(user_input, character_prompt, st.session_state.chat_history)
    st.session_state.chat_history.append(ChatMessage("User", user_input).to_dict())
    st.session_state.chat_history.append(ChatMessage("Assistant", ai_response).to_dict())
    st.session_state.user_input = ""  # Clear input field

def handle_enter_key_press():
    user_input = st.session_state.user_input
    if user_input:
        character_prompt = character_info[st.session_state.selected_character]
        asyncio.run(get_ai_response(user_input, character_prompt))

# Streamlit app
def main():
    st.title("Chat with AI")

    if 'chat_pipeline' not in st.session_state:
        api_key = "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"  # Your Groq API key
        model_name = 'llama3-70b-8192'
        st.session_state.chat_pipeline = ChatPipeline(api_key=api_key, model_name=model_name)
        st.session_state.chat_history = []

    selected_character = st.selectbox("Select your preferred fictional character", character_names)

    # Refresh chat history if a different character is selected
    if "selected_character" not in st.session_state or st.session_state["selected_character"] != selected_character:
        st.session_state["chat_history"] = []
        st.session_state["selected_character"] = selected_character

    character_prompt = character_info[selected_character]
    st.subheader("Chat History:")
    for message in st.session_state.chat_history:
        if message["role"] == 'User':
            st.markdown(
                f"""
                <div style='display: flex; align-items: center; justify-content: flex-end; margin-bottom: 10px;'>
                    <div style='max-width: 70%; padding: 10px; background-color: black; border-radius: 15px;'>
                        <p style='margin: 0;'>{message["content"]}</p>
                    </div>
                    <img src='https://via.placeholder.com/30' alt='user' style='margin-left: 10px; border-radius: 50%;'>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                    <img src='https://via.placeholder.com/30' alt='assistant' style='margin-right: 10px; border-radius: 50%;'>
                    <div style='max-width: 70%; padding: 10px; background-color: black; border-radius: 15px;'>
                        <p style='margin: 0;'>{message["content"]}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.text_input("You:", key="user_input", on_change=handle_enter_key_press)

    if st.session_state.user_input.lower() == "stop":
        st.write("Chat ended. Thank you!")
        st.stop()

    st.text("Press Enter to send the message.")

if __name__ == "__main__":
    main()
