import base64
import json
import time
from typing import Any, Dict, List, Callable, Optional
from uuid import UUID

import streamlit as st
import plotly.io
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from streamlit.delta_generator import DeltaGenerator


AVATARS = {
    "ai": "media/avatars/solver.jpg",
    "human": "media/avatars/creator.jpg",
    "user": "media/avatars/creator.jpg",
    "task_gen": "media/avatars/task_generator_morpheus.jpg",
    "code": "media/avatars/compiler.jpg",
}


def st_sidebar_render():
    """Рендерим дефолтный sidebar"""
    st.markdown(
        """
    <style>
      div[data-testid="stSidebarHeader"] > img, div[data-testid="collapsedControl"] > img {
          height: 3rem;
          width: auto;
      }

      div[data-testid="stSidebarHeader"], div[data-testid="stSidebarHeader"] > *,
      div[data-testid="collapsedControl"], div[data-testid="collapsedControl"] > * {
          display: flex;
          align-items: center;
      }
    </style>""",  # noqa
        unsafe_allow_html=True,
    )
    st.sidebar.page_link("pages/with_jupyter.py", label="👨‍💻 Ввести задачу самому")
    st.sidebar.page_link("pages/with_jupyter_infinite.py", label="🤖 Авто-задачи")
    st.logo("media/logo.png")

    qr64 = image_to_base64("media/qr.png")
    st.sidebar.markdown(
        f"<div style='margin-top: 10px'>"
        f"<img style='max-width: 225px;margin-bottom:20px;' "
        f"src='data:image/png;base64, {qr64}'/>"
        f"</div>",
        unsafe_allow_html=True,
    )


def image_to_base64(image_path):
    with open(image_path, "rb") as file:
        logo = file.read()
    base64_image = base64.b64encode(logo).decode("utf-8")
    return base64_image


def render_attachments(attachments, callback_on_attachment=None):
    for attachment in attachments:
        if "application/vnd.plotly.v1+json" in attachment:
            data = json.dumps(attachment["application/vnd.plotly.v1+json"])
            plot = plotly.io.from_json(data)
            with st.chat_message("assistant", avatar=AVATARS.get("code")):
                st.plotly_chart(plot, key="iris")
        if "image/png" in attachment:
            data = attachment["image/png"]
            with st.chat_message("assistant", avatar=AVATARS.get("code")):
                st.markdown(
                    f"""<img src="data:png;base64,{data}" >""",
                    True,
                )
        if callable(callback_on_attachment):
            time.sleep(1)
            callback_on_attachment()


class MessageCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        parent_container: Callable[[], DeltaGenerator],
        visual_callback: Optional[Callable] = None,
        history: Optional[StreamlitChatMessageHistory] = None,
    ):
        self.parent_container = parent_container
        if callable(visual_callback):
            self.visual_callback = visual_callback
        else:
            self.visual_callback = lambda: None
        self.history = history
        self.text = ""
        self.token_count = 0

    def on_tool_start(self, *args, **kwargs: Any) -> Any:
        self.status = self.parent_container().status("Выполняем код", expanded=True)

    def on_chat_model_start(
        self,
        *args,
        **kwargs: Any,
    ) -> Any:
        with self.parent_container():
            self.message = st.empty()
            self.text = ""
            self.token_count = 0

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        self.text += token
        self.token_count += 1
        self.message.markdown(self.text)
        if self.token_count % 10 == 0:
            self.visual_callback()

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        self.visual_callback()
        if self.history:
            self.history.add_ai_message(action.log)

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        self.visual_callback()
        if self.history:
            self.history.add_ai_message(finish.log)

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        """Run when tool ends running."""
        attachments = output["attachments"]
        with self.status:
            st.text(output["message"])
            if output["is_exception"]:
                self.status.update(state="error")
            else:
                self.status.update(state="complete")
        render_attachments(attachments, self.visual_callback)
        self.visual_callback()
        if self.history:
            self.history.add_message(
                HumanMessage(
                    content=output["message"],
                    additional_kwargs={"attach": attachments, "is_tool": True},
                )
            )
