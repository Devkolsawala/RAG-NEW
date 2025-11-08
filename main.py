"""
Enhanced Flask Web Application for Document Q&A System
Features: BGE embeddings, Multiple LLM support, Streaming responses, General Chat Mode
Run with: python main.py
Access at: http://localhost:5000
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import ollama
import PyPDF2
from typing import List, Dict, Generator
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'txt', 'md', 'docx'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

qa_system = None


class EnhancedDocumentQASystem:
    def __init__(self, persist_dir: str = "./chroma_db", model_name: str = "mistral"):
        self.model_name = model_name
        print(f"🚀 Loading BGE embeddings (best quality)...")
        
        # Use BGE - State of the art embeddings
        self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        
        # Add instruction for better retrieval
        self.query_instruction = "Represent this sentence for searching relevant passages: "
        
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        ))
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"✓ System ready with {self.collection.count()} chunks")
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Enhanced chunking with better boundary detection"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            
            if end < text_len:
                # Try multiple separators
                separators = ['\n\n', '\n', '. ', '! ', '? ', '; ', ', ']
                best_break = -1
                
                for sep in separators:
                    pos = chunk.rfind(sep)
                    if pos > chunk_size * 0.4:  # At least 40% through
                        best_break = pos + len(sep)
                        break
                
                if best_break > 0:
                    chunk = chunk[:best_break]
                    end = start + best_break
            
            chunk = chunk.strip()
            if len(chunk) > 50:  # Only add substantial chunks
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def read_document(self, file_path: str) -> str:
        """Enhanced document reading with better error handling"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.suffix.lower() == '.pdf':
            text = []
            try:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for i, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        if page_text.strip():
                            text.append(f"[Page {i+1}]\n{page_text}")
                return '\n\n'.join(text)
            except Exception as e:
                raise ValueError(f"Failed to read PDF: {str(e)}")
        
        elif path.suffix.lower() in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    
    def add_document(self, file_path: str) -> Dict:
        """Add document with enhanced metadata"""
        try:
            doc_name = Path(file_path).name
            text = self.read_document(file_path)
            
            # Enhanced chunking
            chunks = self.chunk_text(text, chunk_size=800, overlap=100)
            
            # Generate embeddings with batch processing
            print(f"   Generating embeddings for {len(chunks)} chunks...")
            embeddings = self.embedding_model.encode(
                chunks, 
                show_progress_bar=False,
                batch_size=32
            ).tolist()
            
            # Rich metadata
            ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [{
                'source': doc_name,
                'chunk_id': i,
                'chunk_size': len(chunks[i]),
                'total_chunks': len(chunks)
            } for i in range(len(chunks))]
            
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            
            return {
                'success': True,
                'filename': doc_name,
                'chunks': len(chunks),
                'characters': len(text)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Enhanced search with query instruction"""
        # Add instruction prefix for BGE
        instructed_query = self.query_instruction + query
        
        query_embedding = self.embedding_model.encode([instructed_query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, self.collection.count())
        )
        
        chunks = []
        if results['documents'][0]:
            for i in range(len(results['documents'][0])):
                # Calculate relevance score (1 - distance)
                relevance = 1 - results['distances'][0][i]
                chunks.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'relevance': relevance
                })
        
        return chunks
    
    def general_chat_stream(self, question: str) -> Generator:
        """Stream response for general chat (no document context)"""
        try:
            stream = ollama.generate(
                model=self.model_name,
                prompt=question,
                stream=True
            )
            
            for chunk in stream:
                if 'response' in chunk:
                    yield json.dumps({
                        'type': 'token',
                        'content': chunk['response']
                    }) + '\n'
            
            yield json.dumps({'type': 'done'}) + '\n'
            
        except Exception as e:
            yield json.dumps({
                'type': 'error',
                'content': f"Error: {str(e)}"
            }) + '\n'
    
    def answer_question_stream(self, question: str, n_context: int = 4) -> Generator:
        """Stream answer chunks for real-time display (with document context)"""
        relevant_chunks = self.search(question, n_results=n_context)
        
        if not relevant_chunks:
            yield json.dumps({
                'type': 'error',
                'content': "I couldn't find any relevant information in the uploaded documents."
            }) + '\n'
            return
        
        # Send sources first
        sources = list(set([chunk['metadata']['source'] for chunk in relevant_chunks]))
        yield json.dumps({
            'type': 'sources',
            'content': sources
        }) + '\n'
        
        # Build enhanced context with relevance scores
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            context_parts.append(
                f"[Source: {chunk['metadata']['source']}, Relevance: {chunk['relevance']:.2%}]\n"
                f"{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)
        
        # Enhanced prompt with better instructions
        prompt = f"""You are a helpful AI assistant answering questions based on provided documents. Follow these rules:

1. Answer ONLY using information from the context below
2. If the answer isn't in the context, clearly state that
3. Be specific and cite which document sections support your answer
4. Use a natural, conversational tone
5. If multiple documents have relevant info, synthesize them coherently

Context from documents:
{context}

Question: {question}

Provide a clear, accurate answer:"""
        
        # Stream response from Ollama
        try:
            stream = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=True
            )
            
            for chunk in stream:
                if 'response' in chunk:
                    yield json.dumps({
                        'type': 'token',
                        'content': chunk['response']
                    }) + '\n'
            
            yield json.dumps({'type': 'done'}) + '\n'
            
        except Exception as e:
            yield json.dumps({
                'type': 'error',
                'content': f"Error: {str(e)}"
            }) + '\n'
    
    def answer_question(self, question: str, n_context: int = 4) -> Dict:
        """Non-streaming version for compatibility"""
        relevant_chunks = self.search(question, n_results=n_context)
        
        if not relevant_chunks:
            return {
                'answer': "I couldn't find any relevant information in the uploaded documents.",
                'sources': [],
                'context_used': []
            }
        
        context = "\n\n".join([
            f"[Document: {chunk['metadata']['source']}]\n{chunk['text']}"
            for chunk in relevant_chunks
        ])
        
        prompt = f"""Based on the following document excerpts, answer the question. Only use information from the provided context.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = ollama.generate(model=self.model_name, prompt=prompt)
            answer = response['response'].strip()
        except Exception as e:
            return {
                'answer': f"Error: {str(e)}",
                'sources': [],
                'context_used': []
            }
        
        sources = list(set([chunk['metadata']['source'] for chunk in relevant_chunks]))
        
        return {
            'answer': answer,
            'sources': sources,
            'relevance_scores': [chunk['relevance'] for chunk in relevant_chunks]
        }
    
    def get_all_documents(self) -> List[Dict]:
        """Get detailed document list with statistics"""
        try:
            results = self.collection.get()
            if not results or not results['metadatas']:
                return []
            
            doc_stats = {}
            for meta in results['metadatas']:
                source = meta['source']
                if source not in doc_stats:
                    doc_stats[source] = {
                        'name': source,
                        'chunks': 0,
                        'total_size': 0
                    }
                doc_stats[source]['chunks'] += 1
                doc_stats[source]['total_size'] += meta.get('chunk_size', 0)
            
            return sorted(doc_stats.values(), key=lambda x: x['name'])
        except:
            return []
    
    def delete_document(self, filename: str) -> bool:
        try:
            results = self.collection.get()
            ids_to_delete = [
                results['ids'][i] 
                for i, meta in enumerate(results['metadatas']) 
                if meta['source'] == filename
            ]
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
            return True
        except:
            return False
    
    def clear_all_documents(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Get all IDs first
            results = self.collection.get()
            if results and results['ids']:
                # Delete by IDs instead of using where={}
                self.collection.delete(ids=results['ids'])
            return True
        except Exception as e:
            print(f"Error clearing documents: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        try:
            count = self.collection.count()
            docs = self.get_all_documents()
            return {
                'total_chunks': count,
                'total_documents': len(docs),
                'documents': docs
            }
        except:
            return {'total_chunks': 0, 'total_documents': 0, 'documents': []}
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            models = ollama.list()
            return [model['name'] for model in models['models']]
        except:
            return [self.model_name]
    
    def switch_model(self, model_name: str):
        """Switch to a different Ollama model"""
        self.model_name = model_name


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    result = qa_system.add_document(filepath)
    
    return jsonify(result)


@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    result = qa_system.answer_question(question)
    return jsonify(result)


@app.route('/api/ask/stream', methods=['POST'])
def ask_question_stream():
    """Streaming endpoint for document-based responses"""
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    def generate():
        for chunk in qa_system.answer_question_stream(question):
            yield chunk
    
    return Response(
        stream_with_context(generate()),
        mimetype='application/x-ndjson'
    )


@app.route('/api/chat/stream', methods=['POST'])
def general_chat_stream():
    """Streaming endpoint for general chat (no documents)"""
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    def generate():
        for chunk in qa_system.general_chat_stream(question):
            yield chunk
    
    return Response(
        stream_with_context(generate()),
        mimetype='application/x-ndjson'
    )


@app.route('/api/documents', methods=['GET'])
def get_documents():
    stats = qa_system.get_stats()
    return jsonify(stats)


@app.route('/api/documents/<filename>', methods=['DELETE'])
def delete_document(filename):
    success = qa_system.delete_document(filename)
    if success:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Failed to delete'}), 500


@app.route('/api/clear', methods=['POST'])
def clear_database():
    """Clear all documents"""
    try:
        success = qa_system.clear_all_documents()
        
        if success:
            # Clear uploaded files
            for file in os.listdir(app.config['UPLOAD_FOLDER']):
                try:
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
                except Exception as e:
                    print(f"Failed to delete file {file}: {str(e)}")
            
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Failed to clear database'}), 500
            
    except Exception as e:
        print(f"Clear database error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available Ollama models"""
    models = qa_system.get_available_models()
    return jsonify({
        'models': models,
        'current': qa_system.model_name
    })


@app.route('/api/models', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    data = request.get_json()
    model_name = data.get('model', '').strip()
    
    if not model_name:
        return jsonify({'error': 'No model specified'}), 400
    
    qa_system.switch_model(model_name)
    return jsonify({'success': True, 'model': model_name})


if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Enhanced Document Q&A System with General Chat")
    print("=" * 70)
    print("\n⚡ Initializing with BGE embeddings and advanced features...")
    
    qa_system = EnhancedDocumentQASystem(model_name="mistral")
    
    print("\n✅ System ready!")
    print("\n" + "=" * 70)
    print("🌐 Open your browser:")
    print("   http://localhost:5000")
    print("=" * 70)
    print("\n💡 Features:")
    print("   • BGE embeddings (best quality)")
    print("   • Streaming responses (real-time)")
    print("   • Multiple model support")
    print("   • General chat mode (no documents needed)")
    print("   • Document Q&A mode (RAG-powered)")
    print("   • Enhanced UI with dark mode")
    print("\nPress Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)