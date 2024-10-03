

# Agentic-RAG-using-Ollama-GoogleSearchAPI
This repository demonstrates a Retrieval-Augmented Generation (RAG) agent using LLaMA3 for local question answering with fallback and self-correction mechanisms. The project integrates several RAG strategies based on key papers, and runs a local LLaMA3 model to retrieve documents and generate responses.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Environment Setup](#environment-setup)
4. [Model Loading](#model-loading)
5. [Document Loading and Indexing](#document-loading-and-indexing)
6. [Retrieval Grading](#retrieval-grading)
7. [Response Generation](#response-generation)
8. [Hallucination Grading](#hallucination-grading)
9. [Answer Grading](#answer-grading)
10. [Routing Mechanism](#routing-mechanism)
11. [Web Search Fallback](#web-search-fallback)
12. [Graph Workflow](#graph-workflow)

---

## 1. Project Overview

This project implements a Retrieval-Augmented Generation (RAG) agent using LLaMA3 as the local language model. The system is capable of routing questions to either local document retrieval or fallback to web search, performing hallucination checks, and grading the relevance of responses. It is structured based on papers such as Adaptive RAG, Corrective RAG, and Self-RAG.

**Key Features:**
- Adaptive question routing.
- Document retrieval from vector stores.
- Hallucination detection and correction.
- Grading of response relevance.

---

## 2. Installation

First, install the required packages:
```bash
pip install langchain-nomic langchain_community tiktoken langchainhub
```

For embedding and language model:
```bash
pip install ollama
ollama pull llama3
```

---

## 3. Environment Setup



---

### **Prerequisites**

Before setting up the Retrieval-Augmented Generation (RAG) agent using LLaMA3, ensure that the following prerequisites are met:

#### a. **Secrets.json File Configuration**

Create a `secrets.json` file in the root directory of your project. It should look something like this:

```json
{
  "LANGCHAIN_API_KEY": "your_langchain_api_key",
  "GOOGLE_API_KEY": "your_google_api_key",
  "GOOGLE_CSE_ID": "your_google_cse_id"
}
```

This file stores the necessary API keys that the project requires for functioning, such as Langchain API, Google Custom Search API, and Custom Search Engine (CSE) ID.

#### b. **Search Engine ID (CSE ID)**

- **Search Engine Configuration**:  
  Before using the Google Custom Search JSON API, you need to create and configure a **Programmable Search Engine (CSE)**. You can start this process by visiting the [Programmable Search Engine control panel](https://programmablesearchengine.google.com/).

- **Configuration Options**:  
  Follow the [tutorial](https://developers.google.com/custom-search/docs/tutorial/introduction) to learn more about different configuration options, such as including specific sites or the entire web for your search engine.

- **Locating Your Search Engine ID**:  
  After creating the search engine, visit the [help center](https://support.google.com/programmable-search/answer/2649143?hl=en) to locate your **Search engine ID (CSE ID)**, which you'll add to the `secrets.json` file under the key `"GOOGLE_CSE_ID"`.

#### c. **API Key for Custom Search JSON API**

- **Obtaining an API Key**:  
  The Custom Search JSON API requires an API key for authentication. You can get your API key from the Google Developers page [here](https://developers.google.com/custom-search/v1/overview#search_engine_id).

After obtaining the API key, add it to the `secrets.json` file under the key `"GOOGLE_API_KEY"`.

Next, sign Up on Langchain Hub:

1. Go to the [Langchain Hub](https://smith.langchain.com/) and sign up for an account.
If you already have an account, simply log in.
2. Access API Settings:

* Once logged in, navigate to your profile or account settings.
* Look for the API Key section, where you'll find an option to generate an API key.
3. Generate Your API Key:

* Click on the "Generate API Key" button.

#### A new API key will be generated. Make sure to copy and save this key, as you will need it to authenticate your requests when using Langchain services.
Add the API Key to secrets.json:
---

Set up your environment by storing necessary API keys in a `secrets.json` file, and load them into the environment.



```python
import os
import json

def get_secrets():
    with open('secrets.json') as secrets_file:
        return json.load(secrets_file)

secrets = get_secrets()
os.environ["LANGCHAIN_API_KEY"]  = secrets.get("LANGCHAIN_API_KEY")
os.environ["GOOGLE_API_KEY"] = secrets.get("GOOGLE_API_KEY")
os.environ["GOOGLE_CSE_ID"] = secrets.get("GOOGLE_CSE_ID")
```

Enable tracing for debugging:
```bash
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
```

---

## 4. Model Loading

Specify and load the LLaMA3 model locally:
```python
local_llm = "llama3"
```

---

## 5. Document Loading and Indexing

Retrieve and load documents from URLs and split them into chunks using `RecursiveCharacterTextSplitter`. 

```python
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

urls = [
    "https://example-url-1.com",
    "https://example-url-2.com"
]

docs = [WebBaseLoader(url).load() for url in urls]
doc_splits = RecursiveCharacterTextSplitter(chunk_size=250).split_documents(docs)
```

Embed the documents and add them to a **Chroma Vector Store**:
```python
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed")
vectorstore = Chroma.from_documents(doc_splits, embedding=embedding_model)
retriever = vectorstore.as_retriever()
```

---

## 6. Retrieval Grading

Create a grading mechanism to check if retrieved documents are relevant to the query:
```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt = PromptTemplate(
    template="Is the document relevant to the query? Give a 'yes' or 'no' response.",
    input_variables=["document", "question"]
)

retrieval_grader = prompt | llm | JsonOutputParser()
```

---

## 7. Response Generation

Generate responses by combining retrieved context with the model:

```python
prompt = PromptTemplate(
    template="Use the following context to answer the query. Max 3 sentences.",
    input_variables=["question", "context"]
)

rag_chain = prompt | llm
docs = retriever.invoke("agent memory")
generation = rag_chain.invoke({"context": docs, "question": "agent memory"})
```

---

## 8. Hallucination Grading

Evaluate whether the generated answer is grounded in the provided facts:
```python
hallucination_prompt = PromptTemplate(
    template="Is the answer grounded in the facts provided? Give a 'yes' or 'no' response.",
    input_variables=["generation", "documents"]
)

hallucination_grader = hallucination_prompt | llm | JsonOutputParser()
result = hallucination_grader.invoke({"documents": docs, "generation": generation})
```

---

## 9. Answer Grading

Assess if the answer sufficiently resolves the user’s question:
```python
answer_prompt = PromptTemplate(
    template="Does the answer resolve the question? Give a 'yes' or 'no' response.",
    input_variables=["generation", "question"]
)

answer_grader = answer_prompt | llm | JsonOutputParser()
result = answer_grader.invoke({"generation": generation, "question": "agent memory"})
```

---

## 10. Routing Mechanism

Use a routing mechanism to decide if a query should use local documents or web search:
```python
routing_prompt = PromptTemplate(
    template="Should the query be routed to a vectorstore or web search?",
    input_variables=["question"]
)

question_router = routing_prompt | llm | JsonOutputParser()
result = question_router.invoke({"question": "agent memory"})
```

---

## 11. Web Search Fallback

Fallback to web search when documents are not relevant:
```python
from langchain.utilities import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper(k=3)
web_results = search.results("agent memory")
```

---

## 12. Graph Workflow

Define a workflow for state management and control flow:
```python
from langgraph.graph import StateGraph

workflow = StateGraph(GraphState)

# Define nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Define edges and conditional logic
workflow.add_edge("retrieve", "generate")
```

Compile and execute the workflow:
```python
app = workflow.compile()

inputs = {"question": "What are the types of agent memory?"}
for output in app.stream(inputs):
    print(output)
```

---

## Conclusion

This project showcases the integration of a local RAG agent using LLaMA3, capable of retrieving, generating, and grading responses while leveraging web search as a fallback mechanism.

--- 
