import streamlit as st
import re
import PyPDF2
import os
import glob
import requests

# --- CONFIG VISUAL ---
st.set_page_config(page_title="Chatbot - Hojas de Seguridad", layout="wide", page_icon="🧑‍🔬")

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
        .env-badge {
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

# --- CONFIG PATHS ---
DOCUMENTS_PATH = "documents"
LLAMA_SERVER_URL = "http://127.0.0.1:8000/completion"

# --- DETECTAR ENTORNO ---
def is_streamlit_cloud():
    """Detecta si está corriendo en Streamlit Cloud"""
    # Verificar variables de entorno específicas de Streamlit Cloud
    cloud_indicators = [
        os.getenv("STREAMLIT_SHARING_MODE"),
        os.getenv("STREAMLIT_SERVER_PORT"),
        os.getenv("STREAMLIT_SERVER_ADDRESS")
    ]
    
    # Si alguna está presente, probablemente es cloud
    if any(cloud_indicators):
        return True
    
    # Verificar si estamos en un entorno tipo cloud (sin llama.cpp local)
    try:
        response = requests.get("http://127.0.0.1:8000", timeout=1)
        return False  # Si llama.cpp responde, estamos en local
    except:
        # Si no responde llama.cpp, probablemente estamos en cloud
        return True

# --- FUNCIONES LLM MISTRAL ---
def generate_ai_response(query, relevant_chunks):
    if not relevant_chunks:
        return "No se encontró información relevante en los documentos disponibles."
    
    context = "\n".join([chunk['content'][:400] for chunk in relevant_chunks[:2]])
    
    # Detectar entorno y usar Mistral correspondiente
    if is_streamlit_cloud():
        return generate_mistral_api_response(query, context)
    else:
        return generate_local_mistral_response(query, context)

def generate_mistral_api_response(query, context):
    """Mistral via API para Streamlit Cloud"""
    try:
        # Verificar si tenemos la API key de forma segura
        try:
            api_key = st.secrets["MISTRAL_API_KEY"]
        except:
            return "⚠️ API Key de Mistral no configurada. Ve a Settings → Secrets en Streamlit Cloud."
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mistral-small-latest",
            "messages": [
                {
                    "role": "system", 
                    "content": "Eres un asistente especializado en hojas de seguridad. SOLO responde basándote en la información proporcionada. Si la información no está disponible, di claramente que no se encuentra en los documentos."
                },
                {
                    "role": "user", 
                    "content": f"""CONTEXTO DE DOCUMENTOS:
{context}

PREGUNTA: {query}

Responde únicamente basándote en el contexto anterior:"""
                }
            ],
            "max_tokens": 150,
            "temperature": 0.3
        }
        
        with st.spinner("🤖 Consultando Mistral API..."):
            response = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
        
        if response.status_code == 200:
            data = response.json()
            text = data['choices'][0]['message']['content'].strip()
            
            # Validar que no esté vacía
            if not text:
                return "No pude generar una respuesta clara basada en los documentos."
            
            return text
            
        elif response.status_code == 401:
            return "❌ Error de autenticación con Mistral API. Verifica tu API key."
        elif response.status_code == 429:
            return "⏱️ Límite de consultas alcanzado. Intenta de nuevo en unos minutos."
        else:
            return f"❌ Error API Mistral: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "⏱️ Timeout: Mistral API está tardando demasiado."
    except requests.exceptions.ConnectionError:
        return "❌ Error de conexión con Mistral API."
    except Exception as e:
        return f"❌ Error inesperado: {str(e)}"

def generate_local_mistral_response(query, context):
    """Mistral local via llama.cpp"""
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
        with st.spinner("🤖 Consultando Mistral local..."):
            response = requests.post(
                LLAMA_SERVER_URL,
                json={
                    "prompt": prompt,
                    "n_predict": 150,
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "top_k": 20,
                    "repeat_penalty": 1.05,
                    "stop": ["</s>", "PREGUNTA:", "CONTEXTO:", "REGLAS:"]
                },
                timeout=60
            )
        
        if response.status_code == 200:
            data = response.json()
            text = data.get("content", "").strip()
            
            # Limpiar la respuesta
            if "RESPUESTA:" in text:
                text = text.split("RESPUESTA:")[-1].strip()
            
            # Validar que no esté vacía
            if not text:
                return "No pude generar una respuesta clara basada en los documentos disponibles."
                
            return text
            
        else:
            return f"❌ Error del servidor llama.cpp: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return "⏱️ Timeout: El servidor llama.cpp está tardando demasiado."
    except requests.exceptions.ConnectionError:
        return "❌ No se puede conectar al servidor llama.cpp local. Verifica que esté corriendo en http://127.0.0.1:8000"
    except Exception as e:
        return f"❌ Error de conexión local: {str(e)}"

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

@st.cache_data
def load_documents():
    all_chunks = []
    if not os.path.exists(DOCUMENTS_PATH):
        os.makedirs(DOCUMENTS_PATH)
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
        
        if final_score >= min_score:
            results.append({
                'content': chunk['content'],
                'source': chunk['source'],
                'score': final_score,
                'exact_matches': exact_matches,
                'keywords': chunk_keywords,
                'debug': f"Exact:{exact_score:.2f} Partial:{partial_score:.2f} Phrase:{phrase_score:.2f}"
            })
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

# --- INTERFAZ ---
# Detectar entorno para mostrar badge
is_cloud = is_streamlit_cloud()
env_badge = "☁️ CLOUD (Mistral API)" if is_cloud else "🏠 LOCAL (llama.cpp)"
badge_class = "env-badge" if is_cloud else "local-badge"

st.markdown(
    f"""
    <div style='text-align:center;'>
        <h1 style='color:#009639; font-size: 2.8rem; margin-bottom:0.2em;'>🧑‍🔬 Chatbot - Hojas de Seguridad</h1>
        <span class='{badge_class}'>{env_badge}</span>
    </div>
    <hr style='border:1px solid #009639; margin-top:1em; margin-bottom:1.5em;'/>
    """,
    unsafe_allow_html=True
)

all_chunks = load_documents()


if all_chunks:
    st.success(f"✅ {len(all_chunks)} fragmentos de documentos cargados")
    
    query = st.text_input(
        "💬 Escribe tu pregunta sobre seguridad, aplicación, dosis, etc.",
        placeholder="Ejemplo: ¿Qué hacer en caso de contacto con los ojos?",
        key="input_pregunta"
    )
    
    if query:
        results = hybrid_search(query, all_chunks, top_k=3, min_score=0.15)
        
        if results:
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
                    "<div style='background:#e3f9ed; padding:1em; border-radius:8px; border-left:4px solid #009639;'>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**🤖 Respuesta:**")
                st.markdown(ai_response)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Mostrar fuentes
                st.markdown("---")
                st.markdown("**📄 Fuentes consultadas:**")
                for source in sources:
                    st.markdown(f"• {source}")
                
                # Mostrar fragmentos relevantes (opcional)
                with st.expander("🔍 Ver fragmentos relevantes"):
                    for i, result in enumerate(results[:2], 1):
                        st.markdown(f"**Fragmento {i}** (Score: {result['score']:.2f}) - *{result['source']}*")
                        st.markdown(f"```\n{result['content'][:300]}...\n```")
                        
        else:
            st.warning("❌ No se encontraron fragmentos relevantes para tu consulta.")
            st.info("💡 **Sugerencias:**\n- Usa términos más específicos\n- Verifica la ortografía\n- Intenta con sinónimos")
    
else:
    st.warning("⚠️ No hay documentos cargados")
    st.info("""
    **Para usar el chatbot:**
    1. Crea una carpeta llamada `documents` en el mismo directorio del script
    2. Coloca tus archivos PDF de hojas de seguridad en esa carpeta
    3. Reinicia la aplicación
    """)

# --- SIDEBAR INFO ---
with st.sidebar:
    st.markdown("### ℹ️ Información")
    st.markdown(f"**Entorno:** {env_badge}")
    st.markdown(f"**Documentos:** {len(all_chunks)} fragmentos")
    
    if all_chunks:
        # Mostrar lista de documentos
        pdf_files = glob.glob(os.path.join(DOCUMENTS_PATH, "*.pdf"))
        st.markdown("**📁 Archivos cargados:**")
        for pdf_file in pdf_files:
            filename = os.path.basename(pdf_file)
            st.markdown(f"• {filename}")
    
    st.markdown("---")
    st.markdown("### 🔧 Configuración")
    
    if is_cloud:
        st.info("**Modo Cloud:** Usando Mistral API")
        st.markdown("Configura `MISTRAL_API_KEY` en Secrets")
    else:
        st.info("**Modo Local:** Usando llama.cpp")
        st.markdown("Servidor: http://127.0.0.1:8000")
    
    st.markdown("---")
    st.markdown("### 💡 Consejos")
    st.markdown("""
    - Haz preguntas específicas
    - Menciona el producto si es posible
    - Usa términos técnicos cuando sea necesario
    - Pregunta sobre: seguridad, aplicación, dosis, primeros auxilios
    """)