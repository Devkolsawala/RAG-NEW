"""
Local Document Summarizer + Q&A System
Uses Ollama (phi3), ChromaDB, and sentence-transformers for local RAG
Python 3.11+ compatible
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import ollama

# Document parsing
try:
    import PyPDF2
except ImportError:
    print("Warning: PyPDF2 not installed. PDF support disabled.")
    PyPDF2 = None


class DocumentQASystem:
    def __init__(self, persist_dir: str = "./chroma_db", model_name: str = "mistral"):
        """
        Initialize the Document Q&A system
        
        Args:
            persist_dir: Directory to persist ChromaDB data
            model_name: Ollama model name (default: mistral)
        """
        self.model_name = model_name
        print(f"Loading embedding model (all-mpnet-base-v2)...")
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')  # Better accuracy
        
        # Initialize ChromaDB with persistence
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        ))
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"✓ Initialized with {self.collection.count()} existing document chunks")
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text
            chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks in characters
        """
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_len:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.5:  # Only break if we're past halfway
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return [c for c in chunks if len(c) > 20]  # Filter very short chunks
    
    def read_document(self, file_path: str) -> str:
        """Read document content based on file type"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Handle PDFs
        if path.suffix.lower() == '.pdf':
            if PyPDF2 is None:
                raise ImportError("PyPDF2 required for PDF support. Install: pip install PyPDF2")
            
            text = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text())
            return '\n\n'.join(text)
        
        # Handle text files
        elif path.suffix.lower() in ['.txt', '.md', '.rst']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    
    def add_document(self, file_path: str, metadata: Dict = None) -> int:
        """
        Add a document to the vector store
        
        Returns:
            Number of chunks added
        """
        print(f"\n📄 Processing: {file_path}")
        
        # Read document
        text = self.read_document(file_path)
        print(f"   Extracted {len(text)} characters")
        
        # Chunk document
        chunks = self.chunk_text(text)
        print(f"   Created {len(chunks)} chunks")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False).tolist()
        
        # Prepare metadata
        doc_name = Path(file_path).name
        base_metadata = metadata or {}
        base_metadata['source'] = doc_name
        
        # Add to ChromaDB
        ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{**base_metadata, 'chunk_id': i} for i in range(len(chunks))]
        
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"   ✓ Added {len(chunks)} chunks to vector store")
        return len(chunks)
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search for relevant document chunks
        
        Returns:
            List of relevant chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # Format results
        chunks = []
        for i in range(len(results['documents'][0])):
            chunks.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return chunks
    
    def answer_question(self, question: str, n_context: int = 3) -> Dict:
        """
        Answer a question using document context
        
        Returns:
            Dict with answer, sources, and context
        """
        print(f"\n❓ Question: {question}")
        
        # Retrieve relevant chunks
        relevant_chunks = self.search(question, n_results=n_context)
        
        if not relevant_chunks:
            return {
                'answer': "I couldn't find any relevant information in the documents.",
                'sources': [],
                'context_used': []
            }
        
        # Build context
        context = "\n\n".join([
            f"[Document: {chunk['metadata']['source']}]\n{chunk['text']}"
            for chunk in relevant_chunks
        ])
        
        # Create prompt
        prompt = f"""Based on the following document excerpts, answer the question. Only use information from the provided context. If the answer is not in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        # Get answer from Ollama
        print(f"   Querying {self.model_name}...")
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt
            )
            answer = response['response'].strip()
        except Exception as e:
            return {
                'answer': f"Error querying Ollama: {str(e)}",
                'sources': [],
                'context_used': []
            }
        
        # Extract unique sources
        sources = list(set([chunk['metadata']['source'] for chunk in relevant_chunks]))
        
        return {
            'answer': answer,
            'sources': sources,
            'context_used': [chunk['text'][:200] + "..." for chunk in relevant_chunks]
        }
    
    def summarize_document(self, file_path: str) -> str:
        """Generate a summary of a document"""
        print(f"\n📋 Summarizing: {file_path}")
        
        text = self.read_document(file_path)
        
        # For long documents, chunk and summarize iteratively
        if len(text) > 3000:
            chunks = self.chunk_text(text, chunk_size=2000, overlap=100)
            summaries = []
            
            for i, chunk in enumerate(chunks[:5]):  # Limit to first 5 chunks
                prompt = f"Summarize the following text concisely:\n\n{chunk}\n\nSummary:"
                response = ollama.generate(model=self.model_name, prompt=prompt)
                summaries.append(response['response'].strip())
            
            # Combine summaries
            combined = "\n\n".join(summaries)
            final_prompt = f"Combine these summaries into one coherent summary:\n\n{combined}\n\nFinal Summary:"
            response = ollama.generate(model=self.model_name, prompt=final_prompt)
            return response['response'].strip()
        else:
            prompt = f"Summarize the following document:\n\n{text}\n\nSummary:"
            response = ollama.generate(model=self.model_name, prompt=prompt)
            return response['response'].strip()
    
    def clear_database(self):
        """Clear all documents from the database"""
        self.chroma_client.delete_collection("documents")
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        print("✓ Database cleared")


def main():
    """Interactive demo"""
    print("=" * 60)
    print("Local Document Q&A System (Ollama + Phi3)")
    print("=" * 60)
    
    # Initialize system
    qa_system = DocumentQASystem(model_name="phi3")
    
    # Demo mode
    print("\n📚 Demo Mode - Add documents and ask questions")
    print("\nCommands:")
    print("  add <filepath>     - Add a document")
    print("  ask <question>     - Ask a question")
    print("  summarize <filepath> - Summarize a document")
    print("  clear              - Clear database")
    print("  quit               - Exit")
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            
            if command == "quit":
                break
            
            elif command == "add" and len(parts) > 1:
                file_path = parts[1].strip('"\'')
                qa_system.add_document(file_path)
            
            elif command == "ask" and len(parts) > 1:
                question = parts[1]
                result = qa_system.answer_question(question)
                print(f"\n💡 Answer: {result['answer']}")
                print(f"\n📚 Sources: {', '.join(result['sources'])}")
            
            elif command == "summarize" and len(parts) > 1:
                file_path = parts[1].strip('"\'')
                summary = qa_system.summarize_document(file_path)
                print(f"\n📋 Summary:\n{summary}")
            
            elif command == "clear":
                qa_system.clear_database()
            
            else:
                print("Invalid command. Use: add, ask, summarize, clear, or quit")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()