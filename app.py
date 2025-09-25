import streamlit as st
import re
import PyPDF2
import os
import glob
import requests
import json
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- CONFIG ---
DOCUMENTS_PATH = "documents"
LLAMA_SERVER_URL = "http://127.0.0.1:8000/completion"

# --- DETECTAR ENTORNO ---
def is_streamlit_cloud():
    """Detecta si está corriendo en Streamlit Cloud"""
    return (
        os.getenv("STREAMLIT_SHARING_MODE") is not None or
        os.getenv("STREAMLIT_SERVER_PORT") is not None or
        "streamlit" in os.getenv("HOME", "").lower() or
        not os.path.exists("/usr/local/bin")  # Indicador típico de entorno local
    )

# --- FUNCIÓN IA ADAPTATIVA ---
def generate_ai_response(query, relevant_chunks):
    if not relevant_chunks:
        return "No se encontró información relevante en los documentos disponibles."
    
    # Verificar si estamos en Streamlit Cloud
    if is_streamlit_cloud():
        return generate_cloud_response(query, relevant_chunks)
    else:
        return generate_local_response(query, relevant_chunks)

def generate_cloud_response(query, relevant_chunks):
    """Respuesta para Streamlit Cloud (sin llama.cpp)"""
    
    # Respuesta inteligente basada en búsqueda de patrones
    context = "\n\n".join([
        f"**{chunk['source']}:**\n{chunk['content'][:400]}" 
        for chunk in relevant_chunks[:2]
    ])
    
    # Análisis básico de palabras clave
    query_lower = query.lower()
    
    # Patrones comunes en hojas de seguridad
    safety_patterns = {
        "contacto": ["ojos", "piel", "contact", "skin", "eyes"],
        "inhalacion": ["inhalar", "respirar", "inhalation", "breathing"],
        "ingestion": ["ingerir", "tragar", "ingestion", "swallow"],
        "primeros auxilios": ["first aid", "emergency", "emergencia"],
        "almacenamiento": ["storage", "store", "almacenar"],
        "manipulacion": ["handling", "manipular", "use"],
        "peligros": ["hazard", "danger", "peligro", "risk"],
        "equipo": ["ppe", "equipment", "protection", "protección"]
    }
    
    # Encontrar el patrón más relevante
    relevant_pattern = None
    for pattern, keywords in safety_patterns.items():
        if any(keyword in query_lower for keyword in keywords):
            relevant_pattern = pattern
            break
    
    # Buscar información específica en el contexto
    response_parts = []
    
    if relevant_pattern:
        # Buscar oraciones relevantes
        sentences = context.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in safety_patterns[relevant_pattern]):
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            response_parts.extend(relevant_sentences[:3])
    
    # Si no encontró patrones específicos, usar las oraciones más relevantes
    if not response_parts:
        query_words = set(query_lower.split())
        sentences = context.split('.')
        scored_sentences = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 20:
                sentence_words = set(sentence.lower().split())
                score = len(query_words.intersection(sentence_words))
                if score > 0:
                    scored_sentences.append((score, sentence.strip()))
        
        # Ordenar por relevancia
        scored_sentences.sort(reverse=True)
        response_parts = [sent[1] for sent in scored_sentences[:3]]
    
    if response_parts:
        sources = list(set([chunk['source'] for chunk in relevant_chunks]))
        response = "**Información encontrada:**\n\n" + "\n\n".join(response_parts)
        response += f"\n\n*📚 Fuentes: {', '.join(sources)}*"
        return response
    else:
        return "La información específica no se encuentra claramente detallada en los documentos disponibles."

def generate_local_response(query, relevant_chunks):
    """Respuesta para entorno local (con llama.cpp)"""
    context = "\n\n".join([
        f"DOC: {chunk['source'][:20]}\n{chunk['content'][:400]}" 
        for chunk in relevant_chunks[:2]
    ])
    
    prompt = f"""Responde basándote SOLO en estos documentos:

{context}

Pregunta: {query}
Respuesta:"""

    payload = {
        "prompt": prompt,
        "n_predict": 100,
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 20,
        "repeat_penalty": 1.05,
        "stop": ["Pregunta:", "\n\n", "DOC:"],
        "stream": False
    }
    
    timeouts = [30, 60, 90]
    
    for attempt, timeout in enumerate(timeouts, 1):
        try:
            st.info(f"🤖 Procesando con IA local (intento {attempt}/{len(timeouts)})")
            
            response = requests.post(
                LLAMA_SERVER_URL,
                json=payload,
                timeout=timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                text = data.get("content", "").strip()
                
                if "Respuesta:" in text:
                    text = text.split("Respuesta:")[-1].strip()
                
                if text and len(text) > 15:
                    return text
                elif attempt < len(timeouts):
                    payload["n_predict"] = min(payload["n_predict"] + 50, 200)
                    time.sleep(1)
                    continue
                else:
                    return "Esta información no se encuentra claramente especificada en los documentos disponibles."
            
            else:
                if attempt < len(timeouts):
                    time.sleep(2)
                    continue
                else:
                    return f"Error del servidor: {response.status_code}"
                    
        except requests.exceptions.Timeout:
            if attempt < len(timeouts):
                st.warning(f"⏱️ Timeout de {timeout}s, reintentando...")
                time.sleep(1)
                continue
            else:
                return "⏱️ El servidor local está tardando demasiado en responder."
        
        except requests.exceptions.ConnectionError:
            return "❌ No se puede conectar al servidor llama.cpp local. Verifica que esté corriendo."
        
        except Exception as e:
            if attempt < len(timeouts):
                time.sleep(2)
                continue
            else:
                return f"❌ Error: {str(e)}"
    
    return "❌ No se pudo obtener respuesta después de múltiples intentos."

# --- RESTO DE FUNCIONES (iguales) ---
@st.cache_data
def load_documents():
    all_chunks = []
    
    if not os.path.exists(DOCUMENTS_PATH):
        os.makedirs(DOCUMENTS_PATH)
        return all_chunks
    
    pdf_files = glob.glob(os.path.join(DOCUMENTS_PATH, "*.pdf"))
    
    for pdf_file in pdf_files:
        try:
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                
                for page in pdf_reader.pages:
                    full_text += page.extract_text() + "\n"
                
                full_text = re.sub(r'\s+', ' ', full_text).strip()
                
                if len(full_text) > 50:
                    chunks = create_chunks(full_text, chunk_size=400, overlap=60)
                    
                    for chunk in chunks:
                        all_chunks.append({
                            'content': chunk,
                            'source': os.path.basename(pdf_file)
                        })
        except Exception as e:
            st.error(f"Error procesando {pdf_file}: {e}")
    
    return all_chunks

def create_chunks(text, chunk_size=400, overlap=60):
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start = end - overlap
        
        if end >= len(text):
            break
    
    return chunks

def hybrid_search(query, chunks, top_k=3, min_score=0.15):
    if not chunks:
        return []
    
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    contents = [chunk['content'] for chunk in chunks]
    
    try:
        tfidf_matrix = vectorizer.fit_transform(contents)
        query_vector = vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    except:
        tfidf_scores = np.zeros(len(chunks))
    
    query_words = set(query.lower().split())
    keyword_scores = []
    
    for chunk in chunks:
        content_words = set(chunk['content'].lower().split())
        overlap = len(query_words.intersection(content_words))
        score = overlap / len(query_words) if query_words else 0
        keyword_scores.append(score)
    
    keyword_scores = np.array(keyword_scores)
    combined_scores = 0.7 * tfidf_scores + 0.3 * keyword_scores
    
    top_indices = combined_scores.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        if combined_scores[idx] >= min_score:
            chunk_copy = chunks[idx].copy()
            chunk_copy['score'] = combined_scores[idx]
            results.append(chunk_copy)
    
    return results

# --- INTERFAZ ---
st.set_page_config(page_title="Chatbot - Hojas de Seguridad", layout="wide", page_icon="🧑‍🔬")

# CSS
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .response-box {
        background: #e3f9ed;
        border-radius: 12px;
        padding: 1.2em;
        margin-bottom: 1em;
        border-left: 4px solid #009639;
    }
    .cloud-badge {
        background: #007bff;
        color: white;
        padding: 0.3em 0.8em;
        border-radius: 15px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .local-badge {
        background: #28a745;
        color: white;
        padding: 0.3em 0.8em;
        border-radius: 15px;
        font-size: 0.8em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Detectar entorno
is_cloud = is_streamlit_cloud()
env_badge = "☁️ CLOUD" if is_cloud else "🏠 LOCAL"
badge_class = "cloud-badge" if is_cloud else "local-badge"

st.markdown(
    f"""
    <div style='text-align:center;'>
        <h1 style='color:#009639; font-size: 2.8rem; margin-bottom:0.2em;'>🧑‍🔬 Chatbot - Hojas de Seguridad</h1>
        <span class='{badge_class}'>{env_badge}</span>
        <p style='color:#666; font-size:1.1rem; margin-top:1em;'>
            {'Búsqueda inteligente sin IA externa' if is_cloud else 'IA local con llama.cpp'}
        </p>
    </div>
    <hr style='border:1px solid #009639; margin-top:1em; margin-bottom:1.5em;'/>
    """,
    unsafe_allow_html=True
)

# Cargar documentos
all_chunks = load_documents()

if not all_chunks:
    st.warning("📁 No se encontraron documentos PDF en la carpeta 'documents'")
    if is_cloud:
        st.info("💡 **Para Streamlit Cloud:** Sube los PDFs a la carpeta 'documents' en tu repositorio de GitHub")
else:
    st.success(f"✅ {len(all_chunks)} fragmentos cargados")

# Chat
if all_chunks:
    query = st.text_input(
        "💬 Escribe tu pregunta:",
        placeholder="Ejemplo: ¿Qué hacer en caso de contacto con los ojos?",
        key="query_input"
    )
    
    if query:
        with st.spinner("🔍 Analizando documentos..."):
            results = hybrid_search(query, all_chunks, top_k=3, min_score=0.15)
        
        if results:
            ai_response = generate_ai_response(query, results)
            
            st.markdown(
                f'<div class="response-box">'
                f'<h4 style="color:#009639; margin-top:0;">{"🔍 Análisis Inteligente" if is_cloud else "🤖 Respuesta IA"}:</h4>'
                f'<div>{ai_response}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.warning("❌ No encontré información relevante para tu pregunta.")

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Estado del Sistema")
    st.write(f"🌍 **Entorno:** {'Streamlit Cloud' if is_cloud else 'Local'}")
    st.write(f"🤖 **IA:** {'Búsqueda inteligente' if is_cloud else 'llama.cpp'}")
    st.write(f"📄 **Documentos:** {len(all_chunks)} fragmentos")
    
    if is_cloud:
        st.markdown("### ☁️ Modo Cloud")
        st.info("""
        **Funcionalidades:**
        ✅ Búsqueda avanzada por patrones
        ✅ Análisis de palabras clave
        ✅ Extracción de información relevante
        ✅ Sin dependencias externas
        """)
    else:
        st.markdown("### 🏠 Modo Local")
        st.info("""
        **Funcionalidades:**
        ✅ IA generativa con llama.cpp
        ✅ Respuestas contextuales
        ✅ Procesamiento avanzado
        """)
    