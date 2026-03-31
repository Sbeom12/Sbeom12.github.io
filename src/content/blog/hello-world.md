---
title: "Hello World"
description: "블로그를 시작하며, 앞으로 다룰 주제와 방향에 대해 소개합니다."
date: 2026-03-30
tags: ["blog", "intro"]
draft: false
---

## 블로그를 시작합니다

안녕하세요. 이 블로그에서는 AI, Python, DevOps 등 개발 과정에서 배우고 경험한 것들을 기록하고 공유할 예정입니다.

> 기록하지 않으면 기억하지 못한다.

### 다룰 주제들

주로 다음과 같은 주제를 다룰 계획입니다:

- **AI / LLM** — LangGraph, RAG, 프롬프트 엔지니어링
- **Python** — 실무 팁, 라이브러리 활용법
- **DevOps** — Docker, AWS, CI/CD 파이프라인
- **회고** — 프로젝트 회고 및 기술적 의사결정 기록

### 코드 예시

이 블로그에서는 코드 예시를 적극적으로 활용합니다:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}"),
])

chain = prompt | llm
response = chain.invoke({"input": "Hello!"})
print(response.content)
```

### 마무리

앞으로 꾸준히 글을 올리겠습니다. 피드백은 언제든 환영합니다!

---

*이 블로그는 [Astro](https://astro.build)와 [Tailwind CSS](https://tailwindcss.com)로 만들어졌습니다.*
