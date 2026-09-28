from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

fake_llm = FakeListChatModel(responses=["这是模拟回答。"])
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手。"),
    ("human", "{user_input}"),
])
chain = prompt | fake_llm | StrOutputParser()

result = chain.invoke({"user_input": "测试问题"})
assert result == "这是模拟回答。"
print("✅ 测试通过")