"""All LLM prompt templates, ported verbatim from the notebook.

Keeping prompts in one place makes them easy to tune without touching control
flow. Each strategy in strategies.py imports the prompts it needs from here.
"""


def question_generation_prompt(abstract: str) -> str:
    return f"""
    Role: You are an expert Research Librarian specializing in information retrieval.

    Task: Given the provided [ABSTRACT], generate 2 distinct search queries or questions that a researcher would use to find this specific paper in a database.

    Constraints:
    Length: Each question must be 1-2 lines maximum.
    No Title-Dropping: Do not use the exact title of the paper in the question.
    Semantic Variety:
    Question 1: Focus on the Problem/Gap the paper addresses.
    Question 2: Focus on the Methodology/Technical Approach used.
    Natural Language: Write them like real human queries (e.g., "How does [X] affect [Y] in the context of [Z]?").
    Input:
    [ABSTRACT]: {abstract}
    """ + """Output Format:
    {
    "problem_query": "...",
    "method_query": "...",
    }
    """


def answer_prompt(query: str, context: str) -> str:
    """Shared answer-grounding prompt (used by every strategy)."""
    return f"""
You are a research assistant answering a question using retrieved research papers.

User Question:
{query}

Retrieved Papers:
{context}

Instructions:
1. Answer the user question using only the retrieved papers.
2. Do not use outside knowledge.
3. Use only claims directly supported by the title or abstract.
4. Do not include citations, paper titles, paper IDs, references, or source names in the final answer.
5. If the retrieved abstracts do not contain enough evidence, clearly say that the available information is insufficient.
6. Do not invent datasets, results, metrics, methods, or conclusions not present in the retrieved papers.
7. If multiple papers are relevant, synthesize them into one direct answer.
8. Keep the answer concise and focused.

Output format:

[Direct answer only]
"""


def rewrite_prompt(query: str) -> str:
    return f"""
You are rewriting a user question for semantic vector search over academic research papers.

The vector database searches dense embeddings of paper titles and abstracts.
Therefore, do NOT use Boolean operators, parentheses, quotes, field syntax, or search-engine syntax.

User Question:
{query}

Rewrite the question as a concise academic semantic search query.

Rules:
1. Preserve the exact scientific domain of the user question.
2. Keep rare technical terms from the original question unchanged.
3. Add only 2-5 highly relevant academic terms if they are clearly related.
4. Do not add broad generic terms unless they are necessary.
5. Do not use AND, OR, NOT, parentheses, quotation marks, or colon labels.
6. Avoid ambiguous terms that may have unrelated meanings in other fields.
7. Do not replace domain-specific concepts with generic synonyms.
8. Do not answer the question.
9. Return only the rewritten query as one plain text line.

Output:
"""


def rerank_prompt(query: str, rewritten_query: str, context: str, final_top_k: int = 3) -> str:
    return f"""
You are an expert academic paper reranker.

You are given:
1. The original user research question
2. The rewritten semantic search query
3. A set of retrieved research papers from a vector database

Original Question:
{query}

Rewritten Query:
{rewritten_query}

Retrieved Papers:
{context}

Your task is to select the top {final_top_k} papers that are most useful for answering the original user question.

Return ONLY valid JSON in this format:
{{
  "ranked_papers": [
    {{"rank": 1, "paper_id": "...", "relevance_score": 0.0, "reason": "..."}}
  ],
  "selected_paper_ids": ["...", "...", "..."]
}}
"""


def fusion_query_prompt(query: str, num_queries: int = 3) -> str:
    return f"""
You are helping retrieve academic research papers from a vector database using RAG-Fusion.

The vector database searches dense embeddings of paper titles and abstracts.
Therefore, do NOT use Boolean operators, parentheses, quotes, field syntax, or search-engine syntax.

User Question:
{query}

Generate {num_queries} diverse academic semantic search queries for retrieving relevant research papers.

Rules:
1. Preserve the exact scientific domain of the user question.
2. Keep rare technical terms from the original question unchanged.
3. Each query should focus on a different useful angle of the same research question.
4. Add only highly relevant academic terms if clearly related.
5. Do not add broad generic terms unless necessary.
6. Do not use AND, OR, NOT, parentheses, quotation marks, or colon labels.
7. Avoid ambiguous terms that may have unrelated meanings in other fields.
8. Do not answer the question.
9. Return only valid JSON.
10. The JSON must contain a single key called "queries".
11. Return exactly {num_queries} queries.

Output format:
{{
  "queries": [
    "query 1",
    "query 2",
    "query 3"
  ]
}}
"""


def judge_prompt(question: str, answer: str) -> str:
    return f"""
Evaluate the quality of the Answer based on the Question.

First decide whether the Answer actually answers the Question.

Set:
- "is_answer": true if the Answer makes a real attempt to answer the Question.
- "is_answer": false if the Answer says there is not enough context, refuses to answer, is empty, irrelevant, or does not provide an actual answer.

Scoring rules:
- If "is_answer" is false, set all score fields to 0.
- If "is_answer" is true, score each metric from 1 to 5.

Question:
{question}

Answer:
{answer}

Provide your evaluation strictly in the following JSON format.
Do not add markdown, explanation, or extra text.

{{
    "is_answer": true,
    "accuracy_score": 1,
    "completeness_score": 1,
    "faithfulness_score": 1,
    "relevance_score": 1,
    "clarity_score": 1,
    "overall_score": 1
}}
"""
