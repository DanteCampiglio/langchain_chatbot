import streamlit as st
import re
import PyPDF2
import os
import glob
import requests

# --- CONFIG VISUAL ---
st.set_page_config(page_title="Chatbot - Hojas de Seguridad", layout="wide", page_icon="🧑‍🔬")

# Fondo personalizado (puedes cambiar el color o usar una imagen si quieres)
st.markdown("""
    <style>
        body {
            background-color: #f3f6fa;
        }
        .main {
            background-color: #f9fafb;
        }
        .block-container {
            padding-top: 2rem;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
            border: 1px solid #009639;
        }
        .stButton>button {
            border-radius: 8px;
            background-color: #009639;
            color: white;
        }
        .stAlert {
            border-radius: 10px;
        }
        .st-bb {
            background: #e3f9ed;
            border-radius: 8px;
        }
        hr {
            border:1px solid #009639 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG PATHS ---
DOCUMENTS_PATH = r"C:\Users\s1246678\OneDrive - Syngenta\Documents\Syngenta\IA\Data Science\RAG Pipeline\langchain_chatbot\documents"
LLAMA_SERVER_URL = "http://127.0.0.1:8000/completion"

# --- FUNCIONES LLM ---
def generate_ai_response(query, relevant_chunks):
    if not relevant_chunks:
        return "No se encontró información relevante en los documentos disponibles."
    
    context = "\n".join([chunk['content'][:400] for chunk in relevant_chunks[:2]])
    
    # Prompt mejorado con restricciones claras
    prompt = f"""Eres un asistente especializado en hojas de seguridad. SOLO puedes responder basándote en la información proporcionada a continuación.

REGLAS IMPORTANTES:
- Si la información no está en el contexto, responde: "Esta información no se encuentra en los documentos disponibles."
- NO inventes ni agregues información que no esté explícitamente en el contexto
- Cita específicamente de qué documento proviene la información cuando sea posible

CONTEXTO DE LOS DOCUMENTOS:
{context}

PREGUNTA: {query}

RESPUESTA (solo basada en el contexto anterior):"""

    try:
        response = requests.post(
            LLAMA_SERVER_URL,
            json={
                "prompt": prompt,
                "n_predict": 150,  # Aumenté un poco para respuestas más completas
                "temperature": 0.3,  # Reduje para respuestas más conservadoras
                "stop": ["</s>", "PREGUNTA:", "CONTEXTO:"]
            },
            timeout=60
        )
        
        if response.ok:
            data = response.json()
            text = data.get("content", "")
            
            # Limpiar la respuesta
            if "RESPUESTA:" in text:
                text = text.split("RESPUESTA:")[-1].strip()
            
            # Validar que la respuesta no esté vacía
            if not text.strip():
                return "No pude generar una respuesta clara basada en los documentos disponibles."
                
            return text.strip()
        else:
            return f"Error del servidor: {response.status_code}"
            
    except Exception as e:
        return f"Error de conexión: {e}"

def validate_response_relevance(response, query_keywords, context_keywords):
    """Valida si la respuesta está relacionada con el contexto"""
    response_words = extract_keywords(response.lower())
    
    # Verificar que la respuesta contenga palabras del contexto
    context_overlap = len(set(response_words) & set(context_keywords))
    
    # Si no hay overlap significativo, es probable que sea una respuesta inventada
    if context_overlap < 2 and len(response_words) > 5:
        return False
    return True

# --- FUNCIONES PDF y búsqueda ---
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception:
        return ""

def simple_chunking(text, source, chunk_size=400):
    chunks = []
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) < 50:
        return chunks
    overlap = 50
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            space_pos = text.rfind(' ', end - 50, end)
            if space_pos > start:
                end = space_pos
        chunk_text = text[start:end].strip()
        if len(chunk_text) > 50:
            chunks.append({
                'content': chunk_text,
                'source': source
            })
        start = end - overlap
        if start >= len(text) - 50:
            break
    return chunks

def load_documents():
    all_chunks = []
    if not os.path.exists(DOCUMENTS_PATH):
        return all_chunks
    pdf_files = glob.glob(os.path.join(DOCUMENTS_PATH, "*.pdf"))
    for file_path in pdf_files:
        filename = os.path.basename(file_path)
        text = extract_text_from_pdf(file_path)
        if text.strip():
            chunks = simple_chunking(text, filename)
            all_chunks.extend(chunks)
    return all_chunks

def extract_keywords(text):
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()
    words = [w for w in words if len(w) > 2]
    stop_words = {
        'que', 'para', 'con', 'por', 'una', 'del', 'las', 'los', 'este', 'esta',
        'como', 'ser', 'son', 'está', 'están', 'fue', 'sido', 'tiene', 'tienen',
        'puede', 'pueden', 'debe', 'deben', 'hacer', 'hace', 'muy', 'más', 'menos',
        'todo', 'todos', 'toda', 'todas', 'cada', 'algunos', 'algunas', 'otro', 'otra'
    }
    words = [w for w in words if w not in stop_words]
    return words

def hybrid_search(query, all_chunks, top_k=3, min_score=0.15):
    if not all_chunks:
        return []
    
    query_keywords = extract_keywords(query)
    results = []
    
    for chunk in all_chunks:
        chunk_text = chunk['content'].lower()
        chunk_keywords = extract_keywords(chunk_text)
        
        exact_matches = len(set(query_keywords) & set(chunk_keywords))
        exact_score = exact_matches / max(len(query_keywords), 1)
        
        partial_score = 0
        for qword in query_keywords:
            for cword in chunk_keywords:
                if qword in cword or cword in qword:
                    partial_score += 0.5
        partial_score = partial_score / max(len(query_keywords), 1)
        
        phrase_score = 0
        query_lower = query.lower()
        if len(query_lower) > 10:
            query_words = query_lower.split()
            for i in range(len(query_words) - 2):
                phrase = ' '.join(query_words[i:i+3])
                if phrase in chunk_text:
                    phrase_score += 1
        
        final_score = (
            exact_score * 0.5 +
            partial_score * 0.2 + 
            phrase_score * 0.3
        )
        
        # Solo incluir chunks con score mínimo
        if final_score >= min_score:
            results.append({
                'content': chunk['content'],
                'source': chunk['source'],
                'score': final_score,
                'exact_matches': exact_matches,
                'keywords': chunk_keywords,  # Agregar para validación
                'debug': f"Exact:{exact_score:.2f} Partial:{partial_score:.2f} Phrase:{phrase_score:.2f}"
            })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

# --- INTERFAZ ---
st.markdown(
    """
    <div style='text-align:center;'>
        <h1 style='color:#009639; font-size: 2.8rem; margin-bottom:0.2em;'>🧑‍🔬 Chatbot - Hojas de Seguridad</h1>
    </div>
    <hr style='border:1px solid #009639; margin-top:1em; margin-bottom:1.5em;'/>
    """,
    unsafe_allow_html=True
)

all_chunks = load_documents()

if all_chunks:
    query = st.text_input(
        "💬 Escribe tu pregunta sobre seguridad, aplicación, dosis, etc.",
        placeholder="Ejemplo: ¿Qué hacer en caso de contacto con los ojos?",
        key="input_pregunta"
    )
    
    if query:
        results = hybrid_search(query, all_chunks, top_k=3, min_score=0.15)
        
        if results:
            with st.spinner("🤖 Analizando documentos..."):
                ai_response = generate_ai_response(query, results)
                
                # Validación adicional
                query_keywords = extract_keywords(query)
                context_keywords = []
                for result in results:
                    context_keywords.extend(result.get('keywords', []))
                
                # Verificar si la respuesta parece inventada
                if "no se encuentra" not in ai_response.lower() and not validate_response_relevance(ai_response, query_keywords, context_keywords):
                    ai_response = "Esta información no se encuentra claramente especificada en los documentos disponibles."
            
            if ai_response:
                # Mostrar fuentes
                sources = list(set([r['source'] for r in results]))
                
                st.markdown(
                    "<div style='background:#e3f9ed; border-radius:12px; padding:1.2em 1em; margin-bottom:1em; border: 1px solid #6fcf97;'><b>🤖 Respuesta:</b><br>"
                    + ai_response + 
                    f"<br><br><small><b>Fuentes:</b> {', '.join(sources)}</small></div>",
                    unsafe_allow_html=True
                )
                
                # Mostrar score de confianza
                max_score = max([r['score'] for r in results])
                confidence = "Alta" if max_score > 0.4 else "Media" if max_score > 0.2 else "Baja"
                
                st.markdown(
                    f"<div style='text-align:center; color:#666; font-size:0.9em;'>Confianza: {confidence} | Basado en documentos oficiales</div>",
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    "<div style='text-align:center; color:#aaa; font-size:0.95em;'>---<br>Recuerda siempre consultar la hoja oficial ante dudas graves.<br>---</div>",
                    unsafe_allow_html=True
                )
        else:
            st.warning("❌ No encontré información relevante para tu consulta en los documentos disponibles.")
else:
    st.error("❌ No se pudieron cargar documentos.")

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://www.syngenta.com/sites/g/files/zhg576/f/styles/large/public/media/2023/06/27/Syngenta_logo.png", width=180)
    st.markdown(
        """
        <h3 style='color:#009639;'>Estado del sistema</h3>
        <ul style='margin-left:-1em;'>
            <li>🤖 <b>IA:</b> llama.cpp (Mistral 7B)</li>
            <li>🗂️ Documentos cargados</li>
            <li>🔒 <b>Modo:</b> Solo documentos</li>
        </ul>
        <hr style='border:0.5px solid #009639;'/>
        <h4 style='color:#009639;'>💡 Ejemplos:</h4>
        <ul>
            <li>¿Qué hacer en caso de contacto con los ojos?</li>
            <li>¿Cuál es la dosis recomendada?</li>
            <li>¿Cómo aplicar el producto?</li>
            <li>¿Qué equipos de protección usar?</li>
            <li>¿Cómo almacenar el producto?</li>
        </ul>
        <hr style='border:0.5px solid #009639;'/>
        <div style='color:#555; font-size:0.95em;'>
            <b>Configuración IA:</b><br>
            Modelo: Mistral 7B<br>
            Motor: llama.cpp<br>
            Contexto: 2 fragmentos<br>
            Temperatura: 0.3 (conservadora)<br>
            Umbral mínimo: 0.15<br>
        </div>
        """,
        unsafe_allow_html=True
    )