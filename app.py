import streamlit as st
import re
import PyPDF2
import os
import glob
import requests

# --- CONFIG VISUAL ---
st.set_page_config(page_title="Chatbot - Hojas de Seguridad", layout="wide", page_icon="🧑‍🔬")

# Fondo personalizado
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
DOCUMENTS_PATH = "documents"
LLAMA_SERVER_URL = "http://127.0.0.1:8000/completion"

# --- FUNCIONES LLM MEJORADAS ---
def generate_ai_response(query, relevant_chunks):
    if not relevant_chunks:
        return "No se encontró información relevante en los documentos disponibles."
    
    # Aumentar el contexto significativamente
    context = "\n\n".join([
        f"DOCUMENTO: {chunk['source']}\nCONTENIDO: {chunk['content'][:800]}" 
        for chunk in relevant_chunks[:3]
    ])
    
    # Prompt más claro y directo
    prompt = f"""Eres un asistente especializado en hojas de seguridad de productos químicos.

INSTRUCCIONES:
- Responde ÚNICAMENTE basándote en la información de los documentos proporcionados
- Si no encuentras la información específica, di: "Esta información no se encuentra en los documentos disponibles"
- Sé específico y cita el documento cuando sea relevante
- Usa un lenguaje claro y profesional

DOCUMENTOS DISPONIBLES:
{context}

PREGUNTA DEL USUARIO: {query}

RESPUESTA:"""

    try:
        response = requests.post(
            LLAMA_SERVER_URL,
            json={
                "prompt": prompt,
                "n_predict": 200,
                "temperature": 0.2,  # Más conservador
                "top_p": 0.9,
                "stop": ["PREGUNTA:", "DOCUMENTOS:", "INSTRUCCIONES:"]
            },
            timeout=60
        )
        
        if response.ok:
            data = response.json()
            text = data.get("content", "").strip()
            
            # Limpiar respuesta
            if "RESPUESTA:" in text:
                text = text.split("RESPUESTA:")[-1].strip()
            
            # Validación básica
            if not text or len(text) < 10:
                return "Esta información no se encuentra claramente especificada en los documentos disponibles."
            
            # Verificar que no sea una respuesta genérica
            generic_phrases = [
                "como asistente de ia",
                "no puedo proporcionar",
                "consulta con un profesional",
                "en general",
                "típicamente",
                "en mi conocimiento",
                "según mi información",
                "generalmente se recomienda",
                "es importante recordar que"
            ]
            
            if any(phrase in text.lower() for phrase in generic_phrases):
                return "Esta información no se encuentra claramente especificada en los documentos disponibles."
                
            return text
            
        else:
            return f"Error del servidor: {response.status_code}"
            
    except Exception as e:
        return f"Error de conexión: {e}"

def validate_response_relevance(response, query_keywords, context_keywords):
    """Validación simplificada"""
    # Solo verificar que no sea una respuesta completamente vacía o genérica
    if len(response.strip()) < 10:
        return False
    
    # Verificar frases que indican respuesta inventada
    invented_phrases = [
        "en mi conocimiento",
        "según mi información",
        "generalmente se recomienda",
        "es importante recordar que",
        "como asistente",
        "no puedo proporcionar"
    ]
    
    return not any(phrase in response.lower() for phrase in invented_phrases)

# --- FUNCIONES PDF y búsqueda MEJORADAS ---
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

def simple_chunking(text, source, chunk_size=600):  # Aumenté el tamaño del chunk
    chunks = []
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) < 50:
        return chunks
    overlap = 100  # Aumenté el overlap
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            space_pos = text.rfind(' ', end - 100, end)
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
        'todo', 'todos', 'toda', 'todas', 'cada', 'algunos', 'algunas', 'otro', 'otra',
        'cuando', 'donde', 'cual', 'cuales', 'porque', 'desde', 'hasta', 'sobre'
    }
    words = [w for w in words if w not in stop_words]
    return words

def hybrid_search(query, all_chunks, top_k=4, min_score=0.1):  # Reducí el umbral mínimo
    if not all_chunks:
        return []
    
    query_keywords = extract_keywords(query)
    results = []
    
    for chunk in all_chunks:
        chunk_text = chunk['content'].lower()
        chunk_keywords = extract_keywords(chunk_text)
        
        # Búsqueda de coincidencias exactas
        exact_matches = len(set(query_keywords) & set(chunk_keywords))
        exact_score = exact_matches / max(len(query_keywords), 1)
        
        # Búsqueda parcial mejorada
        partial_score = 0
        for qword in query_keywords:
            for cword in chunk_keywords:
                if len(qword) > 3 and (qword in cword or cword in qword):
                    partial_score += 0.7
        partial_score = min(partial_score / max(len(query_keywords), 1), 1.0)
        
        # Búsqueda de frases clave
        phrase_score = 0
        query_lower = query.lower()
        for qword in query_keywords:
            if qword in chunk_text:
                phrase_score += 0.3
        
        # Búsqueda de frases completas
        full_phrase_score = 0
        if len(query_lower) > 10:
            query_words = query_lower.split()
            for i in range(len(query_words) - 1):
                phrase = ' '.join(query_words[i:i+2])
                if phrase in chunk_text:
                    full_phrase_score += 0.5
        
        final_score = (
            exact_score * 0.3 +
            partial_score * 0.3 + 
            phrase_score * 0.2 +
            full_phrase_score * 0.2
        )
        
        if final_score >= min_score:
            results.append({
                'content': chunk['content'],
                'source': chunk['source'],
                'score': final_score,
                'exact_matches': exact_matches,
                'keywords': chunk_keywords,
                'debug': f"Exact:{exact_score:.2f} Partial:{partial_score:.2f} Phrase:{phrase_score:.2f} Full:{full_phrase_score:.2f}"
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
    st.success(f"✅ Sistema listo - {len(all_chunks)} fragmentos de documentos cargados")
    
    query = st.text_input(
        "💬 Escribe tu pregunta sobre seguridad, aplicación, dosis, etc.",
        placeholder="Ejemplo: ¿Qué hacer en caso de contacto con los ojos?",
        key="input_pregunta"
    )
    
    if query:
        results = hybrid_search(query, all_chunks, top_k=4, min_score=0.1)
        
        # Debug opcional - descomenta para ver información de búsqueda
        debug_mode = st.checkbox("🔍 Mostrar información de debug", value=False)
        
        if debug_mode and results:
            with st.expander("🔍 Debug - Información de búsqueda"):
                st.write(f"**Palabras clave de la consulta:** {extract_keywords(query)}")
                for i, result in enumerate(results):
                    st.write(f"**Resultado {i+1}** (Score: {result['score']:.3f})")
                    st.write(f"Fuente: {result['source']}")
                    st.write(f"Debug: {result['debug']}")
                    st.write(f"Contenido: {result['content'][:300]}...")
                    st.write("---")
        
        if results:
            with st.spinner("🤖 Analizando documentos..."):
                ai_response = generate_ai_response(query, results)
                
                # Validación adicional simplificada
                query_keywords = extract_keywords(query)
                context_keywords = []
                for result in results:
                    context_keywords.extend(result.get('keywords', []))
                
                # Solo aplicar validación si la respuesta no indica que no se encontró info
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
                    f"<div style='text-align:center; color:#666; font-size:0.9em;'>Confianza: {confidence} (Score: {max_score:.2f}) | Basado en documentos oficiales</div>",
                    unsafe_allow_html=True
                )
                
                # Mostrar fragmentos relevantes si el usuario quiere ver más detalle
                if st.checkbox("📄 Ver fragmentos de documentos utilizados", value=False):
                    st.markdown("### Fragmentos relevantes:")
                    for i, result in enumerate(results[:2]):
                        with st.expander(f"Fragmento {i+1} - {result['source']} (Score: {result['score']:.2f})"):
                            st.write(result['content'])
                
                st.markdown(
                    "<div style='text-align:center; color:#aaa; font-size:0.95em;'>---<br>Recuerda siempre consultar la hoja oficial ante dudas graves.<br>---</div>",
                    unsafe_allow_html=True
                )
        else:
            st.warning("❌ No encontré información relevante para tu consulta en los documentos disponibles.")
            st.info("💡 Intenta reformular tu pregunta o usar términos más específicos.")
else:
    st.error("❌ No se pudieron cargar documentos.")
    st.info(f"Verifica que exista la carpeta '{DOCUMENTS_PATH}' y contenga archivos PDF.")

# --- SIDEBAR ---
# --- SIDEBAR ---
with st.sidebar:
    st.image("https://www.syngenta.com/sites/g/files/zhg576/f/styles/large/public/media/2023/06/27/Syngenta_logo.png", width=180)
    
    # Estado del sistema
    st.markdown(
        f"""
        <h3 style='color:#009639;'>Estado del sistema</h3>
        <ul style='margin-left:-1em;'>
            <li>🤖 <b>IA:</b> llama.cpp (Mistral 7B)</li>
            <li>🗂️ <b>Documentos:</b> {len(all_chunks)} fragmentos</li>
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
            <li>¿Cuáles son los primeros auxilios?</li>
            <li>¿Qué precauciones tomar durante la aplicación?</li>
        </ul>
        <hr style='border:0.5px solid #009639;'/>
        <div style='color:#555; font-size:0.9em;'>
            <b>Configuración IA mejorada:</b><br>
            Modelo: Mistral 7B<br>
            Motor: llama.cpp<br>
            Contexto: 3 fragmentos (800 chars c/u)<br>
            Temperatura: 0.2 (muy conservadora)<br>
            Umbral mínimo: 0.1<br>
            Chunk size: 600 chars<br>
            Overlap: 100 chars<br>
        </div>
        <hr style='border:0.5px solid #009639;'/>
        <div style='color:#666; font-size:0.85em;'>
            <b>Mejoras implementadas:</b><br>
            ✅ Mayor contexto por respuesta<br>
            ✅ Búsqueda más flexible<br>
            ✅ Validación mejorada<br>
            ✅ Debug opcional<br>
            ✅ Fragmentos más grandes<br>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Información adicional
    if st.button("🔄 Recargar documentos"):
        st.rerun()
    
    st.markdown(
        """
        <div style='margin-top:2em; padding:1em; background:#f0f8ff; border-radius:8px; border:1px solid #009639;'>
            <small><b>Nota importante:</b><br>
            Este chatbot analiza únicamente los documentos cargados. 
            Para información crítica de seguridad, siempre consulta 
            las hojas de seguridad oficiales.</small>
        </div>
        """,
        unsafe_allow_html=True
    )