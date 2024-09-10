
import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY_KONPROZ")
import sys
from langchain_openai import ChatOpenAI
import openai
from langchain_openai import OpenAIEmbeddings

sys.path.append(os.getcwd())

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the model and tokenizer from Hugging Face
# model_name = "vicuna-7b-v1.5"  # Replace with any other model you prefer like Falcon-7B
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Create a pipeline for text generation using the loaded model and tokenizer
hf_pipeline = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    max_length=512,
    temperature=0  # Equivalent to the temperature you used in ChatOpenAI
)

# Integrate with Langchain
llm = HuggingFacePipeline(pipeline=hf_pipeline)



# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0
# )

embeddings=OpenAIEmbeddings
embeddings=embeddings(model="text-embedding-ada-002")

if __name__=="__main__":
    from langchain.schema import HumanMessage
    message = HumanMessage(
        content="who is building you?"
    )
    # llm=llm
    # print(llm.invoke([message]))
    response = llm("What is the capital of France?")
    print(response)
    
    # embd=embeddings
    # emb=embd.embed_query("Who is the CEO of the company?")
    # print(len(emb))
    # print(emb[:5])