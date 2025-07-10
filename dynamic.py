import os
import nest_asyncio
nest_asyncio.apply()  

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.readers.github import GitHubIssuesClient, GitHubRepositoryIssuesReader

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding



# --- Config ---

DB_PATH = "./chroma_db"
COLLECTION_NAME = "github_issues"
GITHUB_OWNER = "Openprise"  
GITHUB_REPO = "op-incidents"   

# --- Initialize Chroma and vector store ---

index_exists = os.path.exists(os.path.join(DB_PATH, "docstore.json"))

db = chromadb.PersistentClient(path=DB_PATH)
chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

if not index_exists:
    print("🔧 Creating new index from GitHub issues...")

    class IssueState:
        OPEN = type('EnumValue', (), {'value': 'open'})()
        CLOSED = type('EnumValue', (), {'value': 'closed'})()

    github_client = GitHubIssuesClient(github_token=os.getenv("GITHUB_TOKEN"))
    loader = GitHubRepositoryIssuesReader(
        github_client,
        owner=GITHUB_OWNER,
        repo=GITHUB_REPO,
        verbose=True,
    )
    
    docs_open = loader.load_data(state=IssueState.OPEN)
    docs_closed = loader.load_data(state=IssueState.CLOSED)
    docs = docs_open + docs_closed

    #print("Sample doc object:\n", docs[0])
    #print("Sample metadata:\n", docs[0].metadata)

    for doc in docs:
        doc.metadata["number"] = doc.doc_id  # e.g., '3664'

        lines = doc.text.strip().splitlines()
        doc.metadata["title"] = lines[0][:120] if lines else "No Title"

        doc.metadata["url"] = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/issues/{doc.metadata['number']}"


        for k, v in doc.metadata.items():
            if isinstance(v, (list, dict)):
                doc.metadata[k] = str(v)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        docs, 
        storage_context=storage_context 
    )
    index.storage_context.persist(persist_dir=DB_PATH)

else:
    print("📦 Loading existing index from disk...")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=DB_PATH,
    )
    index = load_index_from_storage(storage_context=storage_context)

# --- Adaptive Similarity Cutoff Query ---

def adaptive_query(prompt, index, top_k=9, initial_cutoff=0.9, min_cutoff=0.6, step=0.1):
    current_cutoff = initial_cutoff

    while current_cutoff >= min_cutoff:
        llm = OpenAI(model="o4-mini", temperature=0)
        print(f"Using LLM Model: {llm.model}, Temperature: {llm.temperature}")

        retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=get_response_synthesizer(llm=llm),
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=current_cutoff)],
        )


        response = query_engine.query(prompt)

        if response.source_nodes:
            print(f"✅ Used similarity_cutoff={current_cutoff:.2f}")
            return response, current_cutoff
        
        print(f"⚠️ No results at cutoff={current_cutoff:.2f}, trying lower...")
        current_cutoff -= step

    print("❌ No relevant results found.")
    return None, None

# --- Streamlit App ---

def main():
    st.title('RAG Application')

    if st.sidebar.button("New Conversation"):
        st.session_state.messages = []

    support_context = ("Create an answer that would assistant the support team and try to keep it understandable for support members rather than an engineering team. Create another answer that can go in depth for an engineering team. Seperate your answer as sections for support assistance and another section for the engineering team that can be more technical.\n")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a Question"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        chat_history = ""
        for msg in st.session_state.messages[-6:]:  # adjust window size as needed
            role = msg["role"]
            content = msg["content"]
            chat_history += f"{role.capitalize()}: {content}\n"
        full_prompt = support_context + f"{chat_history}User: {prompt}\nAssistant:"

        
        with st.spinner('Thinking...'):
            response, used_cutoff = adaptive_query(full_prompt, index)

        if response:
            with st.chat_message("assistant"):
                st.markdown(f"**Used similarity cutoff: `{used_cutoff:.2f}`**")
                st.write(response.response)

                st.subheader('Retrieved context')
                #for i, node in enumerate(response.source_nodes):
                    #score = node.score if hasattr(node, "score") else "N/A"
                    #st.markdown(f"**#{i+1} — Score: {score:.4f}**")
                    #st.markdown(node.node.get_content())
                    #st.markdown("---")
                for i, node in enumerate(response.source_nodes):
                    metadata = node.node.metadata

                    st.markdown(f"**#{i+1} — Retrieved GitHub Issue**")

                    if 'title' in metadata:
                        st.markdown(f"**Title:** {metadata['title']}")
                    if 'number' in metadata:
                        st.markdown(f"**Issue #:** {metadata['number']}")
                    if 'state' in metadata:
                        st.markdown(f"**Status:** {metadata['state']}")
                    if 'url' in metadata:
                        st.markdown(f"[View on GitHub]({metadata['url']})")

                st.markdown("---")

            st.session_state.messages.append({"role": "assistant", "content": response.response})

        else:
            with st.chat_message("assistant"):
                st.warning("No relevant results found at any similarity threshold.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Sorry, I couldn't find anything relevant."
            })


if __name__ == '__main__':
    main()
