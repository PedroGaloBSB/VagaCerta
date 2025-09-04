from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

TAXONOMIA_TECNOLOGIA = {
    "habilidades_tecnicas": [
        "python", "java", "javascript", "go", "c#", "react", "angular", "vue.js",
        "node.js", "spring", "django", "flask", "express", "sql", "nosql",
        "postgresql", "mongodb", "mysql", "aws", "azure", "google cloud", "docker",
        "kubernetes", "git", "agile", "scrum", "kanban", "devops",
        "ci/cd", "machine learning", "ia", "inteligencia artificial",
        "analise de dados"
    ],
    "habilidades_comportamentais": [
        "proatividade", "colaboracao", "resolucao de problemas", "comunicacao",
        "lideranca", "adaptabilidade", "trabalho em equipe", "flexibilidade"
    ],
    "verbos_de_acao": [
        "desenvolveu", "implementou", "otimizou", "automatizou", "projetou",
        "integrou", "migrou", "liderou", "criou", "gerenciou"
    ]
}

class AnaliseRequest(BaseModel):
    curriculo: str
    vaga: str

def calcular_score_ats(curriculo: str, vaga: str) -> Dict[str, float]:
    curriculo_lower = curriculo.lower()
    vaga_lower = vaga.lower()
    
    palavras_curriculo = set(curriculo_lower.split())
    palavras_vaga = set(vaga_lower.split())

    habilidades_vaga_presentes = []
    habilidades_comportamentais_vaga_presentes = []
    
    for hab in TAXONOMIA_TECNOLOGIA["habilidades_tecnicas"]:
        if hab in vaga_lower:
            habilidades_vaga_presentes.append(hab)

    for hab in TAXONOMIA_TECNOLOGIA["habilidades_comportamentais"]:
        if hab in vaga_lower:
            habilidades_comportamentais_vaga_presentes.append(hab)

    match_tecnicas = 0
    for hab in habilidades_vaga_presentes:
        if hab in curriculo_lower:
            match_tecnicas += 1

    match_comportamentais = 0
    for hab in habilidades_comportamentais_vaga_presentes:
        if hab in curriculo_lower:
            match_comportamentais += 1

    total_habilidades_vaga = len(habilidades_vaga_presentes)
    total_comportamentais_vaga = len(habilidades_comportamentais_vaga_presentes)
    
    score_tecnico = (match_tecnicas / total_habilidades_vaga) * 100 if total_habilidades_vaga > 0 else 0
    score_comportamental = (match_comportamentais / total_comportamentais_vaga) * 100 if total_comportamentais_vaga > 0 else 0
    
    score_geral = (score_tecnico + score_comportamental) / 2
    
    return {
        "score_geral": round(score_geral, 2),
        "score_tecnico": round(score_tecnico, 2),
        "score_comportamental": round(score_comportamental, 2),
        "habilidades_tecnicas_encontradas": [hab for hab in habilidades_vaga_presentes if hab in curriculo_lower],
        "habilidades_comportamentais_encontradas": [hab for hab in habilidades_comportamentais_vaga_presentes if hab in curriculo_lower]
    }

@app.post("/api/analise")
async def analise_curriculo(request: AnaliseRequest):
    try:
        resultado = calcular_score_ats(request.curriculo, request.vaga)
        return {"status": "sucesso", "analise": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```
eof

### Como usar o código

Para rodar este backend, você precisará ter o Python instalado. Se ainda não tem, instale a versão mais recente.

1.  **Instale o FastAPI e o Uvicorn:** Abra o terminal e execute os comandos:
    ```bash
    pip install "fastapi[all]"
    ```
    ```bash
    pip install uvicorn
    ```
2.  **Rode a API:** Na mesma pasta onde você salvou o arquivo `app.py`, execute:
    ```bash
    uvicorn app:app --reload
