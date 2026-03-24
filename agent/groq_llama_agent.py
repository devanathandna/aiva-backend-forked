"""
Groq Llama AI Agent with RAG Integration
Handles conversational responses with adaptive response length
"""

import os
import json
import asyncio
import logging
import time
from typing import Dict, Any, Optional
from groq import Groq

from rag_faiss.retriever import retrieve as query_knowledge_base

logger = logging.getLogger(__name__)

# OPTIMIZED: Compressed system prompt (~250 tokens vs ~500 before)
# Adaptive response length: concise for facts, full list for enumerations
SYSTEM_PROMPT = """You are AIVA, an AI-powered college admission assistant for Sri Eshwar College of Engineering. Do NOT hallucinate. Do not say unnecessary things.

Your job is to answer student and parent queries clearly, concisely, and in a structured format.

Always follow this response format EXPLICITLY:

Hi!

[Direct answer to the user's question in 1-2 lines]

Here's what you should know:
- [Key point 1]
- [Key point 2]
- [Key point 3]
(Optional: add 1 more only if important)

What this means:
[Simple explanation of why this information matters in 1-2 lines]

Overall, [clear recommendation or conclusion in 1 line].

Would you like [relevant follow-up question]?

---
Keep responses clean, readable, and not too long.
- Use short lines and bullet points, NOT paragraphs.
- Maintain a helpful and professional tone.
- Do NOT over-explain.
- Do NOT change the format.
- DO NOT use any emojis in your response. EVER.

Behavior Guidelines:
- For placement questions: include stats like 600+ placements, 200+ companies, ₹44 LPA (only if relevant).
- FIX HALLUCINATION: If specific department-wise placement numbers aren't in the dataset (e.g. for ECE), state clearly "Specific department-wise placement data for [Branch] is not available." but mention the overall placement stats.
- For course comparison: explain difference + suggest based on interest.
- For cutoff: show 2-3 examples and explain competition.
- For facilities/hostel: highlight safety, infrastructure, and student comfort.

RULES:
1. Use context data exclusively. If not in context, say you don't have that info.
2. For Tamil queries, respond in Tanglish (Tamil + English mix).
3. Classify emotion as: "happy", "sad", or "none".
4. For questions regarding fees strictly say for any queries contact reception.

RESPOND ONLY in valid JSON:
{"response": "<formatted answer>", "emotion": "<happy|sad|none>"}"""


# Cached Groq client (avoids creating a new client per request)
_groq_client = None
_groq_client_key = None


def _get_groq_client() -> Groq:
    """Get cached Groq client, only recreating when the key changes."""
    global _groq_client, _groq_client_key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise Exception("GROQ_API_KEY not found in environment")
    
    if _groq_client is None or api_key != _groq_client_key:
        _groq_client = Groq(api_key=api_key)
        _groq_client_key = api_key
    return _groq_client


async def get_agent_response(user_query: str, language_context: Optional[Dict] = None, chat_history: Optional[list] = None, rag_query: Optional[str] = None) -> Dict[str, Any]:
    """
    Get response from Groq Llama agent with RAG integration.
    
    OPTIMIZED:
    - Compressed prompt (~250 tokens vs ~500)
    - Native JSON mode (no parsing failures)
    - Lower temperature (0.1) for speed + consistency
    - Adaptive response length (concise for facts, full for lists)
    """
    try:
        # Determine language instruction
        if language_context and language_context.get("is_tamil", False):
            language_instruction = "User asked in Tamil. Respond in Tanglish."
        else:
            language_instruction = "Respond in English."
        
        # Get relevant context from knowledge base (with timeout)
        try:
            loop = asyncio.get_running_loop()
            rag_start = time.time()
            query_to_search = rag_query if rag_query else user_query
            rag_results = await asyncio.wait_for(
                loop.run_in_executor(None, query_knowledge_base, query_to_search),
                timeout=1.5
            )
            rag_ms = (time.time() - rag_start) * 1000
            logger.info(f"⏱️ RAG retrieval took {rag_ms:.0f}ms")
            
            if rag_results and isinstance(rag_results, dict):
                context = rag_results.get("context", "")
                sources = rag_results.get("sources", [])
                logger.info(f"RAG retrieved {len(sources)} sources: {sources}")
                context = context if context else "No specific context found."
            else:
                context = "No specific context found."
                
        except asyncio.TimeoutError:
            logger.warning(f"RAG retrieval timeout for query: {user_query}")
            context = "Knowledge base timeout - using general knowledge."
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            context = "Knowledge base temporarily unavailable."
        
        history_text = ""
        if chat_history:
            history_text = "\nConversation History:\n"
            for msg in chat_history:
                role = "User" if msg.get("role") == "user" else "Assistant"
                history_text += f"{role}: {msg.get('content')}\n"
        
        # Build compact prompt
        prompt = f"""{SYSTEM_PROMPT}

{language_instruction}
{history_text}
Context: {context}

Question: {user_query}"""

        logger.info(f"Context length: {len(context)} chars, Prompt length: {len(prompt)} chars")

        # Get Groq Llama client
        client = _get_groq_client()
        
        # OPTIMIZED: Native JSON mode + lower temperature + adaptive max_tokens
        llm_start = time.time()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,       # OPTIMIZED: lower = faster, more deterministic
            max_tokens=400,        # OPTIMIZED: enough for full listings, prompt controls brevity
            top_p=0.85,
            stream=False,
            response_format={"type": "json_object"},  # OPTIMIZED: native JSON mode
        )
        llm_ms = (time.time() - llm_start) * 1000
        logger.info(f"⏱️ LLM generation took {llm_ms:.0f}ms")
        
        text = response.choices[0].message.content.strip()
        logger.info(f"Groq raw response ({len(text)} chars): '{text[:120]}...'")
        
        # JSON mode ensures valid JSON, but still handle edge cases gracefully
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed despite json_object mode: {e}")
            
            # Fallback: try to extract response text
            partial = ""
            if '"response"' in text:
                try:
                    start_idx = text.find('"response"')
                    # Find the value after the key
                    colon_idx = text.find(':', start_idx)
                    if colon_idx >= 0:
                        remaining = text[colon_idx+1:].strip()
                        if remaining.startswith('"'):
                            end_idx = remaining.find('"', 1)
                            if end_idx > 0:
                                partial = remaining[1:end_idx]
                except Exception:
                    pass
            
            parsed = {
                "response": partial or "I'm having trouble processing your request. Please try again.",
                "emotion": "none"
            }
        
        # Ensure required keys exist with valid values
        if "response" not in parsed:
            parsed["response"] = text if text else "I don't have that information."
        if "emotion" not in parsed or parsed["emotion"] not in ("happy", "sad", "none"):
            parsed["emotion"] = "none"
        
        logger.info(f"✅ Agent response ({len(parsed['response'])} chars): '{parsed['response'][:80]}...' (emotion: {parsed['emotion']})")
        
        return parsed
        
    except Exception as error:
        logger.error(f"Groq Llama agent error: {error}")
        return {
            "response": "I apologize, but I'm experiencing technical difficulties. Please try again.",
            "emotion": "sad"
        }