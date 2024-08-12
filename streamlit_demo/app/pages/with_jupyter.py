from langchain_community.chat_models import GigaChat
from langchain.agents import AgentExecutor
import streamlit as st

from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)

from code_agent import create_code_chat_agent
from code_agent.tools import tools

from utils import st_sidebar_render, render_attachments, AVATARS, MessageCallbackHandler

st_sidebar_render()

msgs = StreamlitChatMessageHistory()

llm = GigaChat(
    profanity_check=False,
    verify_ssl_certs=False,
    timeout=6000,
    model="GigaChat-Pro",
    top_p=0.7,
    repetition_penalty=1.1,
)


if len(msgs.messages) == 0:
    msgs.add_ai_message("Давай избавим этот мир... от нерешенных задач!")

for msg in msgs.messages:
    if msg.additional_kwargs.get("is_tool"):
        with st.chat_message("assistant", avatar=AVATARS.get("code")).status(
            "Выполняем код", state="complete"
        ):
            st.write(msg.content)
        render_attachments(msg.additional_kwargs.get("attach", []))
    else:
        st.chat_message(msg.type, avatar=AVATARS.get(msg.type)).write(msg.content)

agent = create_code_chat_agent(llm)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)

if prompt := st.chat_input("Ваш вопрос"):
    st.chat_message("human", avatar=AVATARS.get("user")).write(prompt)
    msgs.add_user_message(prompt)
    agent_messages = msgs.messages
    agent_executor.invoke(
        {"input": prompt, "history": agent_messages},
        {
            "callbacks": [
                MessageCallbackHandler(
                    lambda: st.chat_message("assistant", avatar=AVATARS.get("ai")),
                    history=msgs,
                )
            ]
        },
    )
