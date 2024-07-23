import gradio as gr
from huggingface_hub import InferenceClient
from typing import List, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Placeholder for the app's state
class MyApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_pdf("Motivation.pdf")
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        """Extracts text from a PDF file and stores it in the app's documents."""
        doc = fitz.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        """Builds a vector database using the content of the PDF."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = model.encode([doc["content"] for doc in self.documents])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant documents found."]

app = MyApp()

def respond(
    message: str,
    history: List[Tuple[str, str]],
    system_message: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    system_message = "Welcome to our motivational chatbot! I’m here to inspire and support you on your journey to discovering your purpose and achieving your goals. Feel free to ask me questions about finding your passion, overcoming challenges, and staying motivated. Whether you're looking for practical advice, inspiring stories, or thought-provoking questions to reflect on, I’m here to help. Let’s work together to unlock your potential and make meaningful progress towards your dreams. How can I assist you today?"
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    # RAG - Retrieve relevant documents
    retrieved_docs = app.search_documents(message)
    context = "\n".join(retrieved_docs)
    messages.append({"role": "system", "content": "Relevant documents: " + context})

    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=100,
        stream=True,
        temperature=0.98,
        top_p=0.7,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

demo = gr.Blocks()

with demo:
    gr.Markdown(
        
    )
    
    chatbot = gr.ChatInterface(
        respond,
        examples=[
            ["How can someone balance their passion with their professional responsibilities to create a fulfilling career?"],
            ["What daily practices or habits do you follow to maintain your motivation and drive?"],
            ["What significant obstacles have you overcome on your path to achieving your dreams, and how did you stay motivated?"],
            ["How do you use your platform to inspire others to find and pursue their passions?"],
            ["What strategies do you use to set and achieve your long-term goals?"],
            ["How do you stay resilient and motivated when faced with setbacks or failures?"],
            ["What legacy do you want to leave behind, and how does it motivate your current actions?"],
            ["How do you find joy and fulfillment in the journey towards achieving your dreams?"]
        ],
        title='Motivational Speaker'
    )

if __name__ == "__main__":
    demo.launch(),
     