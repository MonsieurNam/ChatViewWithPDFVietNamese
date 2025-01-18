import streamlit as st
from langchain.llms.base import LLM
from pydantic import BaseModel, Field
from typing import Optional, List, Mapping, Any

from groq import Groq  # Make sure you installed groq: pip install groq

import os

class GroqWrapper(LLM, BaseModel):
    """
    A custom LLM wrapper that uses the Groq API for chat completion.
    """
    # We assume GROQ_API_TOKEN is set in your .env file or environment.
    api_key: str = os.getenv("GROQ_API_TOKEN")
    model_name: str = Field(default="llama-3.1-8b-instant")
    system_prompt: str = Field(
        default=(
            "Bạn là trợ lý AI chuyên về lịch sử Việt Nam và luôn luôn trả lời bằng tiếng Việt. "
            "Bạn cung cấp câu trả lời chính xác, chi tiết dựa trên nội dung tài liệu được cung cấp. "
            "Nếu không biết, bạn sẽ trả lời 'Tôi không biết.' hoặc "
            "'Trong tài liệu không có thông tin về câu trả lời cho câu hỏi của bạn.' "
            "Bạn không bao giờ trả lời bằng tiếng Anh hoặc bất kỳ ngôn ngữ nào khác ngoài tiếng Việt. "
            "Hãy chỉ sử dụng tiếng Việt trong tất cả các phản hồi của bạn."
        )
    )
    client: Groq = Field(default_factory=lambda: Groq(api_key=os.getenv("GROQ_API_TOKEN")))

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Send a chat completion request to the Groq API.
        """
        try:
            # Build messages list with system prompt
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]

            # Add conversation history from session_state (if any)
            if 'messages' in st.session_state and st.session_state.messages:
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        messages.append({"role": "user", "content": msg["content"]})
                    else:
                        messages.append({"role": "assistant", "content": msg["content"]})

            # Add user's new message
            messages.append({"role": "user", "content": prompt})

            # Create the chat completion
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=stop,
            )
            return completion.choices[0].message.content

        except Exception as e:
            st.error(f"Lỗi khi tạo phản hồi: {e}")
            return "Đã xảy ra lỗi khi tạo phản hồi."

    @property
    def _llm_type(self) -> str:
        return "groq"

    def get_num_tokens(self, text: str) -> int:
        # Just a basic tokenizer approach
        return len(text.split())

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Return identifying params for this LLM.
        """
        return {"model_name": self.model_name, "system_prompt": self.system_prompt}
