from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

import replicate

from fastapi import FastAPI, Query
from pydantic import BaseModel

app = FastAPI()

# Custom prompt template
def get_prompt_template(context, question):
    custom_prompt_template = f"""[INST] <<SYS>>
    You are a trained bot to guide people about Indian Law. You will answer user's query with your knowledge and the context provided. 
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    Do not say thank you and tell you are an AI Assistant and be open about everything.
    <</SYS>>
    Use the following pieces of context to answer the users question.
    Context : {context}
    Question : {question}
    Answer : [/INST]
    """
    return custom_prompt_template

# Return the chain
def qa_pipeline(prompt):
    """
    Create the QA pipeline
    """
    # Load the HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings()

    # Load the index
    db = FAISS.load_local("vectorstore/", embeddings)

    # Get the prompt from the user
    # prompt = input("Enter the prompt\n")

    # Use RAG to get the needed context
    similar_chunks =  db.similarity_search(prompt)

    context1 = similar_chunks[0].page_content
    context2 = similar_chunks[1].page_content

    context = context1 + context2

    # Get custom prompt template
    custom_prompt_template = get_prompt_template(context, prompt)

    # Prompt the mistral-7b-instruct LLM
    mistral_response = replicate.run(
        "a16z-infra/mistral-7b-instruct-v0.1:83b6a56e7c828e667f21fd596c338fd4f0039b46bcfa18d973e8e70e455fda70",
        input={
            "prompt": custom_prompt_template,
            "temperature": 0.75,
            "max_new_tokens": 2048,
        },
    )

    # Concatenate the response into a single string.
    suggestions = "".join([str(s) for s in mistral_response])
   
    return suggestions

class TextPromptRequest(BaseModel):
    prompt: str

class GeneratedTextResponse(BaseModel):
    generated_text: str

@app.post("/generate_text/")
def generate_text(text_prompt: TextPromptRequest):
    
    # Get the outputs from the LLM
    llm_output = qa_pipeline(text_prompt.prompt)
    
    return GeneratedTextResponse(generated_text=llm_output)

@app.get("/")
def default():
    return "Welcome to LawChat"

if __name__ == '__main__':
    print(qa_pipeline())