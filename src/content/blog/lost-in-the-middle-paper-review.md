---
title: "[논문 리뷰] Lost in the Middle: How Language Models Use Long Contexts"
description: "LLM이 긴 컨텍스트를 실제로 얼마나 잘 활용하는지 분석한 논문을 리뷰합니다. RAG 시스템 설계에 중요한 시사점을 제공하는 연구입니다."
date: 2025-02-20
tags: ["AI", "NLP", "논문리뷰", "RAG", "LLM"]
draft: false
---
## 개요

> Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang (2023)
> [논문 링크](https://arxiv.org/abs/2307.03172) | [GitHub](https://github.com/nelson-liu/lost-in-the-middle)

최근 LLM들은 점점 더 긴 컨텍스트를 처리할 수 있게 되었습니다. GPT-4는 128K, Claude는 100K 토큰을 지원하죠. 하지만 **길게 입력할 수 있다는 것과 실제로 잘 활용한다는 것은 다른 문제**입니다.

이 논문은 바로 그 질문에 답합니다: LLM이 긴 입력 컨텍스트를 **실제로** 얼마나 잘 활용하는가?

---

## 핵심 발견: U자형 성능 곡선

논문의 가장 중요한 발견은 다음과 같습니다:

- 관련 정보가 입력의 **처음이나 끝**에 위치할 때 성능이 가장 좋음
- 관련 정보가 입력의 **중간**에 위치할 때 성능이 급격히 하락
- 심지어 관련 문서가 **아예 없는 경우보다도** 중간에 있을 때 성능이 낮은 경우도 있음

![U자형 성능 곡선 - 정답 문서의 위치에 따른 정확도 변화](/images/posts/lost-in-the-middle-results.png)

GPT-3.5-Turbo 기준, 정답 문서가 1번째나 20번째에 위치할 때 정확도가 가장 높고, 중간(10번째 전후)에 위치할 때 급격히 하락하는 U자형 곡선을 보입니다. 빨간 점선은 관련 문서가 아예 없는 closed-book 성능인데, 중간 위치의 성능이 이보다도 낮은 경우가 있다는 점이 충격적입니다.

이 현상을 저자들은 **"Lost in the Middle"** 이라고 명명했습니다.

---

## 실험 설계

### 사용 모델

| 모델             | 최대 토큰 | 특징                            |
| ---------------- | --------- | ------------------------------- |
| MPT-30B-Instruct | 8,192     | ALiBi 위치 인코딩, 1T 토큰 학습 |
| LongChat-13B     | 16,384    | 16K 시퀀스로 파인튜닝           |
| GPT-3.5-Turbo    | 4K / 16K  | OpenAI 상용 모델                |
| Claude-1.3       | 8K / 100K | Anthropic 상용 모델             |

### Task 1: Multi-Document QA

여러 개의 문서를 입력으로 제공하고, 그 중 하나의 문서에만 정답이 포함된 상황에서 QA를 수행합니다.

**실험 방법:**

1. 정답이 포함된 문서 1개 + 관련 없는 문서 N개를 입력으로 구성
2. 정답 문서의 **위치를 변경**하며 성능 측정 (처음, 중간, 끝)
3. 전체 문서 수를 변경하며 컨텍스트 길이에 따른 성능 변화 측정

```python
# 실험 구성 예시 (의사 코드)
def build_qa_input(documents, answer_doc, position):
    """정답 문서를 특정 위치에 배치하여 입력 구성"""
    other_docs = [d for d in documents if d != answer_doc]

    if position == "beginning":
        ordered = [answer_doc] + other_docs
    elif position == "middle":
        mid = len(other_docs) // 2
        ordered = other_docs[:mid] + [answer_doc] + other_docs[mid:]
    elif position == "end":
        ordered = other_docs + [answer_doc]

    context = "\n\n".join([
        f"Document [{i+1}]: {doc}" for i, doc in enumerate(ordered)
    ])

    return context
```

### Task 2: Key-Value Retrieval

더 통제된 환경에서의 검증을 위해 합성 데이터 기반 실험도 수행했습니다. N개의 Key-Value 쌍 중 특정 Key에 해당하는 Value를 찾는 단순한 태스크입니다.

이 태스크에서도 동일한 U자형 패턴이 관찰되었으며, 이는 현상이 태스크에 국한된 것이 아닌 **모델의 근본적인 한계**임을 시사합니다.

---

## 실험 결과 분석

### 주요 결과

1. **모든 모델에서 U자형 곡선 관찰**: 오픈소스, 상용 모델 불문
2. **컨텍스트가 길수록 성능 하락 심화**: 문서 수가 많아질수록 중간 위치의 성능이 더 낮아짐
3. **상용 모델도 예외가 아님**: GPT-3.5-Turbo와 Claude-1.3도 동일한 패턴

### 왜 이런 현상이 발생하는가?

저자들은 몇 가지 가능한 원인을 제시합니다:

- **학습 데이터 편향**: 대부분의 학습 데이터에서 중요한 정보가 문서의 앞이나 뒤에 위치
- **위치 인코딩의 한계**: Attention 메커니즘이 먼 위치의 토큰에 대해 약해지는 경향
- **Primacy/Recency 효과**: 인간의 기억과 유사하게, 처음과 마지막에 본 정보를 더 잘 기억

```python
# Attention 패턴 시각화 (개념적 코드)
import numpy as np

def attention_pattern(seq_length, decay_factor=0.1):
    """위치에 따른 Attention 강도 시뮬레이션"""
    positions = np.arange(seq_length)
    mid = seq_length // 2

    # U자형 패턴: 처음과 끝에서 높고 중간에서 낮음
    attention = np.exp(-decay_factor * np.minimum(positions, seq_length - positions - 1))

    # Primacy bias (처음 위치에 추가 가중치)
    attention[:3] *= 1.5

    return attention / attention.sum()
```

---

## RAG 시스템에 대한 시사점

이 논문의 발견은 RAG(Retrieval-Augmented Generation) 시스템 설계에 직접적인 영향을 줍니다.

### 문제: 검색된 문서의 순서

일반적인 RAG 시스템에서는 관련성 점수가 높은 순서대로 문서를 나열합니다. 하지만 이렇게 하면 **가장 관련성 높은 문서가 맨 앞에, 덜 관련된 문서가 뒤에** 배치되어, LLM이 뒤쪽의 보조 정보를 놓칠 수 있습니다.

### 해결: Long Context Reorder

이 논문의 발견에 기반하여, **중요한 문서를 입력의 처음과 끝에 배치**하는 Reordering 전략이 제안되었습니다.

```python
from langchain_community.document_transformers import LongContextReorder

def apply_reordering(documents: list) -> list:
    """Lost in the Middle 현상을 완화하기 위한 문서 재배치

    관련성 순위: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    재배치 결과: [1, 3, 5, 7, 9, 10, 8, 6, 4, 2]

    → 홀수 순위는 앞에서부터, 짝수 순위는 뒤에서부터 배치
    → 가장 중요한 문서들이 처음과 끝에 위치
    """
    reordering = LongContextReorder()
    reordered = reordering.transform_documents(documents)
    return reordered
```

### 실제 RAG 파이프라인에서의 적용

```python
from langchain_community.vectorstores import FAISS
from langchain_community.document_transformers import LongContextReorder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 검색기 설정
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Reordering 적용
reordering = LongContextReorder()

def retrieve_and_reorder(query: str) -> str:
    """검색 → 재배치 → 포맷팅"""
    docs = retriever.invoke(query)
    reordered = reordering.transform_documents(docs)
    return "\n\n".join([
        f"[문서 {i+1}] {doc.page_content}"
        for i, doc in enumerate(reordered)
    ])

# RAG 체인 구성
prompt = ChatPromptTemplate.from_messages([
    ("system", "컨텍스트를 기반으로 질문에 답변하세요.\n\n{context}"),
    ("user", "{question}"),
])

chain = (
    {"context": retrieve_and_reorder, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    | StrOutputParser()
)
```

---

## 추가 고려사항

### Reranking과의 조합

실제 프로덕션 환경에서는 Reordering을 단독으로 사용하기보다, **Reranking 이후에 적용**하는 것이 효과적입니다:

```
검색 (Top-K) → Reranking (Cross-Encoder) → Reordering → LLM
```

1. **검색**: Dense Retriever로 후보 문서 20개 검색
2. **Reranking**: Cross-Encoder로 정확도 높은 상위 10개 선별
3. **Reordering**: 선별된 10개를 U자형 배치로 재배치
4. **LLM**: 재배치된 컨텍스트로 답변 생성

### 한계점

- 논문이 2023년 기준이므로, 최신 모델(GPT-4o, Claude 3.5 이상의 모델)에서는 개선되었을 가능성]
  - 관련하여 특히, 요즘 모델에서는 Context length 매우 길어져서 이제는 이 실험만큼의 효율이 나오기 어렵다고 생각합니다.
  - 위치 인코딩 기법의 발전(RoPE 개선, YaRN 등)으로 완화될 수 있음
- 하지만 근본적인 Attention 메커니즘의 한계는 여전히 존재

---

## 정리

| 항목                | 내용                                                      |
| ------------------- | --------------------------------------------------------- |
| **핵심 발견** | LLM은 긴 컨텍스트의 중간에 위치한 정보를 잘 활용하지 못함 |
| **패턴**      | U자형 성능 곡선 (처음/끝 > 중간)                          |
| **원인**      | 학습 데이터 편향, 위치 인코딩 한계, Primacy/Recency 효과  |
| **실무 적용** | RAG에서 Reordering 전략으로 중요 문서를 입력 경계에 배치  |
| **구현**      | LangChain `LongContextReorder` 활용                     |

이 논문은 "더 긴 컨텍스트 = 더 좋은 성능"이라는 단순한 가정에 의문을 제기합니다. RAG 시스템을 설계할 때 단순히 문서를 많이 넣는 것이 아니라, **어떤 순서로 배치할 것인가**까지 고려해야 한다는 중요한 교훈을 줍니다.

---

*이 글은 [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) 논문을 리뷰한 글입니다.*
