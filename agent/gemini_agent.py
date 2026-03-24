import json
import logging
import google.generativeai as genai
from config.api_keys import get_gemini_ai_key
from rag_faiss.retriever import retrieve as query_knowledge_base

logger = logging.getLogger(__name__)

# Configure Gemini with rotating API key
def _get_gemini_client():
    """Get Gemini client with rotated API key"""
    api_key = get_gemini_ai_key()
    if not api_key:
        raise Exception("No Gemini AI API key available")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

SYSTEM_PROMPT = """\
You are AIVA, an AI-powered college admission assistant for Sri Eshwar College of Engineering. Do NOT hallucinate. Do not say unnecessary things.

Your job is to answer student and parent queries clearly, concisely, and in a structured format.

Always follow this response format EXPLICITLY:

Hi!

[Direct answer to the user's question in 1-2 lines]

Here's what you should know:
- [Key point 1]
- [Key point 2]
- [Key point 3]
(Optional: add 1 more only if important)

Overall, [clear recommendation or conclusion in 1 line].

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
1. Use ONLY the given context. If the answer is not in the context, say you don't have that information.
2. If the input mixes Tamil and English, reply using both (Tanglish).
3. Classify the response emotion as exactly one of: happy, sad, none.
4. Keep answers professional.

Respond ONLY in valid JSON with this schema:
{
"response":"<formatted answer>",
"emotion":"<happy|sad|none>"
}
"""


async def get_agent_response(user_query: str, language_context: dict = None, chat_history: list = None, rag_query: str = None) -> dict:
    """Retrieve context from ChromaDB, send to Gemini, and return structured response."""
    query_to_search = rag_query if rag_query else user_query
    retrieval = query_knowledge_base(query_to_search)
    context = retrieval["context"]
    sources = ", ".join(retrieval["sources"]) if retrieval["sources"] else "None"

    # Limit context size to 700 tokens for faster processing
    if len(context) > 700:
        context = context[:700] + "..."

    # Determine response language based on is_tamil flag
    language_instruction = ""
    if language_context and language_context.get("is_tamil", False):
        language_instruction = "IMPORTANT: User spoke in Tamil. You MUST respond in Tamil + English mix (Tanglish). Use Tamil words but keep technical terms in English."
    else:
        language_instruction = "User spoke in English. Respond completely in English."
        
    history_text = ""
    if chat_history:
        history_text = "\nConversation History:\n"
        for msg in chat_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_text += f"{role}: {msg.get('content')}\n"

    prompt = f"""{SYSTEM_PROMPT}

LANGUAGE INSTRUCTION: {language_instruction}
{history_text}
Context: {context}

Question: {user_query}
"""

    # Get Gemini model with optimized settings
    model = _get_gemini_client()
    
    # Optimized generation config to prevent incomplete JSON
    generation_config = {
        "temperature": 0.1,  # Lower temperature for more consistent JSON
        "max_output_tokens": 800,  # Increased to prevent truncation
        "top_p": 0.9,
        "candidate_count": 1,
    }
    
    response = model.generate_content(prompt, generation_config=generation_config)
    text = response.text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3].strip()
    if text.startswith("json"):
        text = text[4:].strip()

    try:
        parsed = json.loads(text)
        
        # Validate that both required keys exist
        if "response" not in parsed or "emotion" not in parsed:
            raise json.JSONDecodeError("Missing required keys", text, 0)
            
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON from Gemini (length={len(text)}): {e}")
        logger.warning(f"🤖 RAW GEMINI RESPONSE: '{text}'")  # Show full raw response
        print(f"🤖 GEMINI SAID: '{text}'")  # Print to terminal as requested
        logger.debug(f"Raw response: '{text[:100]}...'")  # Log only first 100 chars
        
        # Enhanced error recovery for incomplete JSON responses
        partial_response = ""
        
        # Try multiple patterns to extract response text
        if '"response"' in text:
            try:
                # Pattern 1: Standard JSON structure
                start_patterns = ['"response": "', '"response":"', ': "']
                for pattern in start_patterns:
                    start_idx = text.find(pattern)
                    if start_idx >= 0:
                        start_idx += len(pattern)
                        
                        # Find the end of the response text
                        remaining = text[start_idx:]
                        
                        # Look for natural end points
                        end_patterns = ['",', '"', '\n', "'}"]
                        min_end = len(remaining)
                        
                        for end_pattern in end_patterns:
                            end_pos = remaining.find(end_pattern)
                            if end_pos >= 0 and end_pos < min_end:
                                min_end = end_pos
                        
                        if min_end < len(remaining):
                            partial_response = remaining[:min_end].strip()
                            break
                
                # Clean up the extracted response
                if partial_response:
                    # Remove common JSON artifacts
                    partial_response = partial_response.replace('\\"', '"')
                    partial_response = partial_response.replace('\\n', ' ')
                    partial_response = partial_response.strip()
                    
            except Exception as extraction_error:
                logger.debug(f"Response extraction failed: {extraction_error}")
        
        # Final fallback
        if not partial_response or len(partial_response.strip()) < 3:
            partial_response = "I apologize, but I'm having trouble processing your request right now."
        
        parsed = {
            "response": partial_response,
            "emotion": "none"
        }
        
        logger.info(f"Recovered response: '{partial_response[:50]}...'")
        
        # Don't fall through to text fallback since we handled it

    # Ensure required keys exist
    if "response" not in parsed:
        parsed["response"] = text
    if "emotion" not in parsed or parsed["emotion"] not in ("happy", "sad", "none"):
        parsed["emotion"] = "none"

    return parsed
