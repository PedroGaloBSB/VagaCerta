import logging
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Set
import re
import io
import spacy
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import pypdf
import docx

# --- CONFIGURAÇÃO DE LOGS ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("vaga_certa.log"),
        logging.StreamHandler()
    ]
)

# --- INICIALIZAÇÃO DA APLICAÇÃO ---
app = FastAPI()

# --- CONFIGURAÇÃO DE CORS ---
origins = ["http://localhost", "http://localhost:8000", "http://127.0.0.1", "http://127.0.0.1:8000", "null"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CARREGAMENTO DOS MODELOS DE IA ---
try:
    nlp = spacy.load("pt_core_news_lg")
    logging.info("Modelo 'pt_core_news_lg' do spaCy carregado com sucesso.")
except OSError:
    logging.warning("Modelo 'pt_core_news_lg' do spaCy não encontrado. A extração de habilidades será menos precisa.")
    nlp = None

try:
    semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    logging.info("Modelo de similaridade semântica carregado com sucesso.")
except Exception as e:
    logging.error(f"Não foi possível carregar o modelo semântico: {e}")
    semantic_model = None

try:
    portuguese_stopwords = set(stopwords.words('portuguese'))
except LookupError:
    logging.info("Pacote 'stopwords' do NLTK não encontrado. Baixando agora...")
    nltk.download('stopwords')
    portuguese_stopwords = set(stopwords.words('portuguese'))

# --- TAXONOMIA ---
TAXONOMIA_GERAL = {
    "tecnologia": {"habilidades_tecnicas": ["python", "java", "javascript", "go", "c#", "react", "angular", "django", "spring", "vue.js", "node.js", "flask", "express", "sql", "nosql", "mongodb", "postgresql", "mysql", "aws", "azure", "google cloud", "docker", "kubernetes", "git", "agile", "scrum", "kanban", "devops", "ci/cd", "analise de dados", "inteligencia artificial", "ia", "machine learning"]},
    "administrativo": {"habilidades_tecnicas": ["pacote office", "excel", "word", "powerpoint", "sistemas erp", "sap", "totvs", "gestao de documentos", "emissao de nota fiscal", "controle de estoque", "contas a pagar", "contas a receber", "fluxo de caixa", "rotinas administrativas"]},
    "vendas": {"habilidades_tecnicas": ["crm", "salesforce", "hubspot", "prospeccao", "negociacao", "gestao de clientes", "relacionamento", "funil de vendas", "metas", "pos-venda", "networking"]},
    "marketing": {"habilidades_tecnicas": ["marketing de conteudo", "inbound marketing", "seo", "google ads", "meta ads", "facebook ads", "instagram ads", "e-mail marketing", "analise de metricas", "google analytics", "social media", "branding", "copywriting", "storytelling"]},
    "financas": {"habilidades_tecnicas": ["planejamento financeiro", "analise de mercado", "gestao de custos", "fluxo de caixa", "contabilidade", "auditoria", "analise de credito", "controle de orcamento", "investimentos"]},
    "saude": {"habilidades_tecnicas": ["primeiros socorros", "prontuario eletronico", "gestao de pacientes", "atendimento de emergencia", "procedimentos clinicos", "administracao de medicamentos", "cuidado ao paciente"]},
    "comportamental": {"habilidades": ["organizacao", "planejamento", "comunicacao", "proatividade", "resolucao de problemas", "atendimento ao cliente", "persuasao", "resiliencia", "empatia", "trabalho em equipe", "networking", "lideranca", "criatividade", "storytelling", "pensamento estrategico", "pensamento analitico", "atencao aos detalhes", "etica", "adaptabilidade", "flexibilidade", "colaboracao"]}
}
todas_habilidades_tecnicas = {hab for area in TAXONOMIA_GERAL.values() for hab in area.get("habilidades_tecnicas", [])}
todas_habilidades_comportamentais = set(TAXONOMIA_GERAL["comportamental"]["habilidades"])

# --- FUNÇÕES AUXILIARES ---
async def extrair_texto_de_arquivo(file: UploadFile) -> str:
    filename = file.filename.lower()
    content = await file.read()
    text = ""
    if filename.endswith('.pdf'):
        try:
            pdf_reader = pypdf.PdfReader(io.BytesIO(content))
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao ler o arquivo PDF: {e}")
    elif filename.endswith('.docx'):
        try:
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao ler o arquivo DOCX: {e}")
    else:
        raise HTTPException(status_code=400, detail="Formato de arquivo não suportado. Use .pdf ou .docx.")
    return text

def normalizar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r'[^\w\s#\+]', '', texto)
    return texto

def extrair_habilidades(texto: str, habilidades_disponiveis: Set[str]) -> Set[str]:
    texto_normalizado = normalizar_texto(texto)
    if not nlp:
        return {hab for hab in habilidades_disponiveis if hab in texto_normalizado}
    doc = nlp(texto_normalizado)
    matcher = spacy.matcher.PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(hab) for hab in habilidades_disponiveis]
    matcher.add("HABILIDADES", patterns)
    matches = matcher(doc)
    return {doc[start:end].text for _, start, end in matches}

def calcular_similaridade_semantica(texto1: str, texto2: str) -> float:
    if not semantic_model: return 0.0
    texto1_limpo = " ".join([palavra for palavra in normalizar_texto(texto1).split() if palavra not in portuguese_stopwords])
    texto2_limpo = " ".join([palavra for palavra in normalizar_texto(texto2).split() if palavra not in portuguese_stopwords])
    embedding1 = semantic_model.encode(texto1_limpo, convert_to_tensor=True)
    embedding2 = semantic_model.encode(texto2_limpo, convert_to_tensor=True)
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2).item()
    return max(0, cosine_score * 100)

# --- LÓGICA DE ANÁLISE (ATUALIZADA) ---
def realizar_analise_completa(curriculo_texto: str, vaga_texto: str) -> Dict:
    habilidades_tecnicas_vaga = extrair_habilidades(vaga_texto, todas_habilidades_tecnicas)
    habilidades_comportamentais_vaga = extrair_habilidades(vaga_texto, todas_habilidades_comportamentais)
    
    habilidades_tecnicas_cv = extrair_habilidades(curriculo_texto, todas_habilidades_tecnicas)
    habilidades_comportamentais_cv = extrair_habilidades(curriculo_texto, todas_habilidades_comportamentais)

    match_tecnicas = habilidades_tecnicas_vaga.intersection(habilidades_tecnicas_cv)
    match_comportamentais = habilidades_comportamentais_vaga.intersection(habilidades_comportamentais_cv)
    
    # === MUDANÇA 1: Correção da fórmula do score técnico ===
    # Se não houver habilidades técnicas na vaga, o score é 0, não 100.
    score_tecnico = (len(match_tecnicas) / len(habilidades_tecnicas_vaga)) * 100 if habilidades_tecnicas_vaga else 0.0
    score_comportamental = (len(match_comportamentais) / len(habilidades_comportamentais_vaga)) * 100 if habilidades_comportamentais_vaga else 0.0

    score_semantico = calcular_similaridade_semantica(curriculo_texto, vaga_texto)

    # === MUDANÇA 2: Nova fórmula do Score Geral com "Multiplicador de Relevância" ===
    # O score semântico agora pondera o resultado dos outros scores.
    # Se a semântica for baixa (ex: 30%), o score geral não poderá ser alto.
    score_base_habilidades = (score_tecnico * 0.7) + (score_comportamental * 0.3) # Média ponderada das habilidades
    
    # O score geral é o score das habilidades, mas filtrado pela relevância semântica.
    score_geral = score_base_habilidades * (score_semantico / 100)

    return {
        "score_geral": round(score_geral, 2),
        "score_tecnico": round(score_tecnico, 2),
        "score_comportamental": round(score_comportamental, 2),
        "score_semantico": round(score_semantico, 2),
        "habilidades_tecnicas_encontradas": sorted(list(match_tecnicas)),
        "habilidades_comportamentais_encontradas": sorted(list(match_comportamentais)),
        "habilidades_tecnicas_faltantes": sorted(list(habilidades_tecnicas_vaga - match_tecnicas))
    }

# --- ENDPOINT DA API ---
@app.post("/api/analise")
async def analise_curriculo(
    vaga: str = Form(...),
    curriculo_file: UploadFile = File(...)
):
    logging.info(f"Recebida nova requisição de análise para o arquivo: {curriculo_file.filename}")
    try:
        curriculo_texto = await extrair_texto_de_arquivo(curriculo_file)
        if not curriculo_texto.strip():
            logging.warning(f"Não foi possível extrair texto do arquivo {curriculo_file.filename}")
            raise HTTPException(status_code=400, detail="Não foi possível extrair texto do arquivo do currículo.")

        resultado = realizar_analise_completa(curriculo_texto, vaga)
        logging.info(f"Análise concluída para {curriculo_file.filename}. Score Geral: {resultado['score_geral']}%")
        
        return {"status": "sucesso", "analise": resultado}

    except Exception as e:
        logging.error(f"Erro inesperado durante a análise de {curriculo_file.filename}: {e}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno inesperado no servidor.")
