import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import csv_loader , DirectoryLoader
import openai
from langsmith import traceable
from langsmith.wrappers import wrap_openai

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
grok_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment or .env file.")
os.environ["OPENAI_API_KEY"] = openai_api_key

path = os.path.abspath("myntra_products_catalog.csv")



# loader = csv_loader.CSVLoader(path)
# docs = loader.load()


# text_splitter = RecursiveCharacterTextSplitter(chunk_size=4500, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)

# # Embed and store in Chroma
# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings() , persist_directory="./chroma")

# # Index
# retriever = vectorstore.as_retriever()

vectorstore = Chroma(persist_directory="./temp", embedding_function = OpenAIEmbeddings() )
retriever = vectorstore.as_retriever()
### RAG bot



class RagBot:

    def __init__(self, retriever, model: str = "gpt-4o-mini"):
        self._retriever = retriever
        # Wrapping the client instruments the LLM
        self._client = wrap_openai(openai.Client())
        self._model = model

    @traceable()
    def retrieve_docs(self, question):
        return self._retriever.invoke(question)

    @traceable()
    def invoke_llm(self, question, docs):
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": """

                            - أنت مساعد افتراضي متخصص في مساعدة العملاء في متجر الملابس الجاهزة. يمكنك تقديم المساعدة في:

                            - تقديم نصائح لاختيار المقاسات المناسبة.
                             - اقتراح تنسيقات ملابس تناسب المناسبة أو الأسلوب الشخصي.
                            - الإجابة على أسئلة العملاء المتعلقة بالمنتجات.

                                """
                    "  استخدم المستندات التالية لتقديم اجابة لسؤال المستخدم باللغة العامية المصرية.\n\n"
                    f"## Docs\n\n{docs}",
                },
                {"role": "user", "content": question},
            ],
        )

        # Evaluators will expect "answer" and "contexts"
        return {
            "answer": response.choices[0].message.content,
            "contexts": [str(doc) for doc in docs],
        }

    @traceable()
    def get_answer(self, question: str):
        docs = self.retrieve_docs(question)
        return self.invoke_llm(question, docs)

rag_bot = RagBot(retriever)



st.set_page_config(page_icon= "🛍️" , layout="wide")

st.header("🛍️ مرحبا بكم في محلات الملابس")
st.subheader("انا المساعد الشخصي ")

question = st.text_area("بما يمكنني ان اساعدك" , height=250)
btn = st.button("SUBMET")
if question:
    if btn:
        with st.spinner('Wait ...'):
   

            response = rag_bot.get_answer(question)
            st.write(response["answer"])



