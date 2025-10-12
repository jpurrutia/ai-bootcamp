from openai import OpenAI
import requests
import json
from minsearch import Index

openai_client = OpenAI()

def search(query):
    boost = {'question': 3.0, 'section': 0.5}

    results = index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        boost_dict=boost,
        num_results=5
    )

    return results

def build_prompt(question, search_results):
    search_json = json.dumps(search_results)
    return prompt_template.format(
        question=question,
        context=search_json
    )

def llm(user_prompt, instructions=None, model="gpt-4o-mini"):
    messages = []

    if instructions:
        messages.append({
            "role": "system",
            "content": instructions
        })

    messages.append({
        "role": "user",
        "content": user_prompt
    })

    response = openai_client.responses.create(
        model=model,
        input=messages
    )

    return response.output_text

def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt, instructions=instructions)
    return answer


if __name__ == "__main__":

    instructions = """
        You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
        Use only the facts from the CONTEXT when answering the QUESTION.
        """.strip()

    prompt_template = """
        <QUESTION>
        {question}
        </QUESTION>

        <CONTEXT>
        {context}
        </CONTEXT>
        """.strip()



    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    documents = []

    for course in documents_raw:
        course_name = course['course']

        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)

    
    index = Index(
        text_fields=["question", "text", "section"],
        keyword_fields=["course"]
    )
    index.fit(documents)


    answer = rag('how do I install Kafka in Python?')

    print(answer)