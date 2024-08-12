import time
import uuid

import streamlit as st
from langchain_community.chat_models import GigaChat
from langchain.agents import AgentExecutor

from streamlit.components.v1 import html

from code_agent import create_code_chat_agent, tools
from generate_tasks import task_prompts
from utils import st_sidebar_render, MessageCallbackHandler, AVATARS


def scroll():
    html(
        f"""
    <script>(() =>{{
    parent.document.querySelector('div[data-testid="stAppViewBlockContainer"]').parentNode.scrollTo({{top: parent.document.querySelector('div[data-testid="stAppViewBlockContainer"]').clientHeight - 300, behavior: 'smooth'}})
    }})("{str(uuid.uuid4())}")</script>""",  # noqa
        height=0,
        width=0,
    )


def scroll_with_delay():
    time.sleep(1)
    scroll()


def reload():
    html(
        f"""
    <script>((dummy) =>{{
    parent.location.reload()
    }})("{str(uuid.uuid4())}")</script>
        """
    )


st_sidebar_render()

st.write(
    """<style>iframe{height:0px} div[data-testid="stAppViewBlockContainer"]>div>div>div>.element-container{height:0px} div[data-testid="stAppViewBlockContainer"]>div>div>div {gap: 0}</style>""",  # noqa
    unsafe_allow_html=True,
)

llm = GigaChat(
    profanity_check=False,
    verify_ssl_certs=False,
    streaming=True,
    timeout=6000,
    model="GigaChat-Pro",
    top_p=0.7,
    repetition_penalty=1.1,
)


st.chat_message("assistant", avatar=AVATARS.get("ai")).write(
    "Давай избавим этот мир... от нерешенных задач!"
)
try:
    agent = create_code_chat_agent(llm)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        return_intermediate_steps=True,
        verbose=True,
        max_iterations=5,
    )

    try:
        st.current_iteration += 1
    except AttributeError:
        st.current_iteration = 0
    task_prompt = task_prompts[st.current_iteration % len(task_prompts)]
    task_chain = task_prompt | llm.bind(top_p=0.9)

    prompt = task_chain.invoke(
        {},
        {
            "callbacks": [
                MessageCallbackHandler(
                    lambda: st.chat_message("human", avatar=AVATARS.get("task_gen")),
                    scroll,
                ),
            ]
        },
    ).content

    agent_executor.invoke(
        {"input": prompt},
        {
            "callbacks": [
                MessageCallbackHandler(
                    lambda: st.chat_message("assistant", avatar=AVATARS.get("ai")),
                    scroll,
                ),
            ]
        },
    )
    time.sleep(10)
finally:
    reload()
