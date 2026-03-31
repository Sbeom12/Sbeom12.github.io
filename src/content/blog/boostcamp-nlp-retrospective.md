---
title: "부스트캠프 7기 NLP 트랙 최종 회고"
description: "부스트캠프 AI Tech 7기 NLP 트랙에서의 해커톤, 프로젝트 경험, 그리고 RAG 시스템 구축까지의 여정을 정리합니다."
date: 2025-02-24
tags: ["AI", "NLP", "부스트캠프", "RAG", "LLM"]
draft: false
---
## 해커톤: 증권사 QA 챗봇 개발

랩큐에서 진행된 해커톤에서 **증권사 QA 챗봇**을 개발했습니다. 증권 관련 문서를 기반으로 사용자의 질문에 정확하게 답변하는 RAG(Retrieval-Augmented Generation) 시스템을 구축하는 것이 목표였습니다.

### 아키텍처 개요

전체 파이프라인은 다음과 같은 흐름으로 구성했습니다:

```
사용자 질문 → Query Transform (HyDE) → Dense Retrieval → Reranking → Reordering → LLM 응답
```

### 1. LangChain을 활용한 RAG 시스템 구축

LangChain의 핵심 컴포넌트들을 활용하여 RAG 파이프라인을 구성했습니다.

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 문서 로드 및 청킹
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "],
)
documents = text_splitter.split_documents(raw_documents)

# 2. 임베딩 및 벡터 스토어 구축
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 3. 프롬프트 설정
prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 증권 전문 AI 어시스턴트입니다.
주어진 컨텍스트만을 기반으로 정확하게 답변하세요.
컨텍스트에 없는 내용은 '해당 정보를 찾을 수 없습니다'라고 답변하세요.

컨텍스트:
{context}"""),
    ("user", "{question}"),
])

# 4. RAG 체인 구성
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. 실행
response = rag_chain.invoke("삼성전자의 최근 배당 정책은?")
```

### 2. Dense Retriever를 활용한 Reranking

초기 검색 결과의 정확도를 높이기 위해 Cross-Encoder 기반의 Reranker를 도입했습니다. Bi-Encoder(Dense Retriever)로 후보군을 빠르게 추리고, Cross-Encoder로 정밀하게 재순위를 매기는 2단계 전략입니다.

```python
from sentence_transformers import CrossEncoder

# Cross-Encoder 기반 Reranker
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

def rerank_documents(query: str, documents: list, top_k: int = 5):
    """검색된 문서들을 Cross-Encoder로 재순위 매김"""
    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)

    # 점수 기준 정렬
    scored_docs = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [doc for doc, score in scored_docs[:top_k]]

# 사용 예시
initial_docs = retriever.invoke("배당 정책")  # 상위 10개
reranked_docs = rerank_documents("배당 정책", initial_docs, top_k=5)
```

> **왜 Reranking이 필요한가?**.
> Bi-Encoder는 쿼리와 문서를 독립적으로 임베딩하므로 빠르지만, 둘 사이의 세밀한 상호작용을 놓칠 수 있습니다. Cross-Encoder는 쿼리-문서 쌍을 함께 입력받아 더 정확한 관련성 점수를 산출합니다. 다만 속도가 느리므로, 후보군을 먼저 줄인 뒤 적용하는 것이 핵심입니다.

### 3. LLM 전달을 위한 Reordering

Reranking 후에도 LLM에 전달하는 문서의 **순서**가 답변 품질에 영향을 미칩니다. "Lost in the Middle" 현상 — LLM이 컨텍스트의 앞부분과 뒷부분에 더 주목하고 중간 부분을 무시하는 경향 — 을 해결하기 위해 Long Context Reorder를 적용했습니다.

```python
from langchain_community.document_transformers import LongContextReorder

reordering = LongContextReorder()

def reorder_for_llm(documents: list) -> list:
    """Lost in the Middle 문제를 해결하기 위한 문서 재배치

    가장 관련성 높은 문서를 앞과 뒤에 배치하고,
    관련성이 낮은 문서를 중간에 배치합니다.
    """
    reordered = reordering.transform_documents(documents)
    return reordered

# 전체 파이프라인
initial_docs = retriever.invoke(query)          # Step 1: 검색
reranked_docs = rerank_documents(query, initial_docs)  # Step 2: 재순위
final_docs = reorder_for_llm(reranked_docs)     # Step 3: 재배치
```

### 4. HyDE (Hypothetical Document Embeddings) Retriever

RAG의 검색 성능을 더 높이기 위해 HyDE 기법을 도입했습니다. 사용자의 질문을 그대로 임베딩하는 대신, LLM이 먼저 가상의 답변을 생성하고, 그 답변을 임베딩하여 검색하는 방식입니다.

```python
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser

# HyDE: 가상 문서 생성 프롬프트
hyde_prompt = ChatPromptTemplate.from_messages([
    ("system", "주어진 질문에 대해 전문적인 답변을 작성하세요. 실제 문서에 있을 법한 형식으로 작성하세요."),
    ("user", "{question}"),
])

hyde_chain = hyde_prompt | ChatOpenAI(temperature=0) | StrOutputParser()

def hyde_retrieve(question: str, retriever, top_k: int = 10):
    """HyDE 기반 검색

    1. 질문으로부터 가상 답변 생성
    2. 가상 답변을 임베딩하여 유사 문서 검색
    3. 원본 질문 검색 결과와 합산
    """
    # 가상 답변 생성
    hypothetical_answer = hyde_chain.invoke({"question": question})

    # 가상 답변으로 검색
    hyde_docs = retriever.invoke(hypothetical_answer)

    # 원본 질문으로도 검색
    original_docs = retriever.invoke(question)

    # 중복 제거 후 합산
    seen = set()
    combined = []
    for doc in hyde_docs + original_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)

    return combined[:top_k]
```

> **HyDE의 핵심 아이디어:**.
> 사용자 질문은 짧고 키워드 중심인 반면, 실제 문서는 긴 설명문입니다. 임베딩 공간에서 질문과 문서 사이의 거리가 멀 수 있는데, HyDE는 LLM이 생성한 "문서 스타일의 답변"을 중간 다리로 활용하여 이 gap을 줄입니다.

---

## 부스트캠프 최종 회고

### 배운 것들

부스트캠프 AI Tech 7기 NLP 트랙은 AI Engineer로서의 출발선에 설 수 있게 해준 프로그램이었습니다.

**모델 학습 경험:**

- V100 GPU 환경에서 BERT 계열 모델 파인튜닝
- LoRA를 활용한 LLM 효율적 학습
- 데이터 전처리부터 평가까지 전체 ML 파이프라인 구축

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
)

# 베이스 모델에 LoRA 적용
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = get_peft_model(base_model, lora_config)

# 학습 가능 파라미터 확인
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
```

**NLP 핵심 기술:**

- Tokenization (BPE, WordPiece, SentencePiece)
- Transformer 아키텍처 이해 및 구현
- 다양한 NLP 태스크: 분류, NER, QA, 요약
- RAG 시스템 설계 및 구현

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

# BERT 기반 텍스트 분류 파이프라인
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "klue/bert-base",
    num_labels=7,
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=5e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

### 솔직한 감상

부스트캠프는 정말 많은 도움이 되었습니다. 특히, AI Engineer로 출발할 수 있는 달리기 트랙 위로 올려주는 프로그램이었습니다. 당시에 구하기 어려웠던 V100 GPU를 제공받아 LLM을 LoRA로 학습하고, BERT 계열 모델들을 직접 다뤄보는 경험은 값진 것이었습니다. 이제는 역사의 뒤안길로 사리지고 있지만, 여전히 V100은 ML 및 distill 영역 및 소형 llm을 사용하게 되면 매우 파워플한 GPU로 제가 개인적으로 구매하기는 더더욱 어려운 것 같습니다.

하지만 솔직히 말하면, 부스트캠프는 **출발선에 올려준 것**이지 주자로 뛰게 해준 것은 아닙니다. 실제로 경쟁력 있는 AI Engineer가 되려면 다양한 논문을 읽고, 최신 기술을 지속적으로 공부해야 합니다. "석사를 위한 등용문"이라는 표현에도 공감합니다. 실제로 회사에 입사해서 보니, 과거의 제 자신이 얼마나 자신 만만했는지 조금은 후회가 되는 부분도 있는거 같습니다. 돌아보면, 랩큐에서 진행한 Rag에서 Dense Retreival을 사용했는데 왜 다시 Reranking을 진행했을까 등 [Lost in the middle](https://velog.io/@sbeom/Lost-in-Middle-How-Language-Model-Use-Long-Contexts)을 읽어서 진행했지만, 관련해서 효율성(속도 및 정확도에 대한 테스트) 아니면 엔지니어링 적으로 더 최적화할 수 있었던 부분들이 보이긴 하지만, 2024년 10월부터 2월까지 총 6개의 프로젝트(1개의 최종 프로젝트, 5개의 사이드 프로젝트)를 진행하면서 많이 성장하고, 배울 수 있어서 좋았습니다!

---

*이 글은 부스트캠프 AI Tech 7기 NLP 트랙을 수료하며 작성한 회고입니다. 감사합니다.*
