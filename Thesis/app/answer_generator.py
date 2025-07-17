import logging
import torch
import re
from typing import List, Dict, Any, Optional
from search_engine import hybrid_search_and_rerank
from mongo_manager import save_message, get_conversation_history
from question_suggester import suggest_questions
from transformers import StoppingCriteria, StoppingCriteriaList
import uuid

# ========== Constants ==========
MIN_SENTENCE_LENGTH = 50
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1
DEFAULT_CONTEXT_WINDOW = 4096

logger = logging.getLogger(__name__)

# ========== Stopping Criteria ==========
class SentenceEndingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, min_length: int = MIN_SENTENCE_LENGTH):
        self.tokenizer = tokenizer
        self.min_length = min_length
        self.end_tokens = {'.', '!', '?', '。', '！', '？'}

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < self.min_length:
            return False
        last_token = self.tokenizer.decode(input_ids[0][-1])
        return last_token[-1] in self.end_tokens

# ========== Utility Functions ==========
def is_identity_question(query: str) -> bool:
    """Check if the question is asking about chatbot's identity"""
    identity_patterns = [
        r'bạn là ai', r'bạn là gì', r'ai đang nói chuyện', r'tên bạn là gì',
        r'bạn làm gì', r'vai trò của bạn', r'nhiệm vụ của bạn'
    ]
    query = query.lower().strip()
    return any(re.search(pattern, query, re.IGNORECASE) for pattern in identity_patterns)

def get_identity_response() -> str:
    """Return a consistent response about chatbot's identity"""
    return (
        "Tôi là một trợ lý AI được thiết kế để hỗ trợ về sức khỏe tinh thần. "
        "Với vai trò là một bác sĩ tâm lý ảo, tôi có thể lắng nghe, thấu hiểu và đưa ra những lời khuyên "
        "dựa trên kiến thức chuyên môn về sức khỏe tâm lý. Tôi luôn cố gắng đồng cảm và hỗ trợ bạn "
        "một cách tốt nhất có thể. Tuy nhiên, xin lưu ý rằng tôi không thể thay thế hoàn toàn các "
        "chuyên gia sức khỏe tâm thần trong thực tế."
    )

def generate_prompt_for_model(
    model_name: str,
    system_message: str,
    user_message: str,
    history_prompt: str,
    context_texts: List[str]
) -> Any:
    """Generate model-specific prompt based on model name"""
    model_name = model_name.lower()
    if "gemma" in model_name:
        prompt = f"<start_of_turn>user\n{system_message}\n\n{user_message}<end_of_turn>\n<start_of_turn>model\nTrả lời với vai trò là bác sĩ tâm lý:\n"
        return prompt
    elif "qwen" in model_name:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        return messages
    elif "llama" in model_name:
        prompt = "<|begin_of_text|>"
        prompt += f"<|start_header_id|>system<|end_header_id>\n{system_message}<|eot_id|>"
        if history_prompt:
            prompt += f"<|start_header_id|>user<|end_header_id>\n{history_prompt}<|eot_id|>"
        if context_texts:
            context_str = "\n\n".join(f"- {ctx}" for ctx in context_texts)
            prompt += f"<|start_header_id|>user<|end_header_id>\n--- THÔNG TIN THAM KHẢO ---\n{context_str}\n--- Kết thúc Thông tin tham khảo ---<|eot_id|>"
        prompt += f"<|start_header_id|>user<|end_header_id>\n{user_message}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id>\n"
        return prompt
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def clean_response(model_name: str, response: str, tokenizer) -> str:
    """Clean model-specific response"""
    model_name = model_name.lower()
    try:
        if "gemma" in model_name:
            response_parts = response.split("<start_of_turn>model\n")
            if len(response_parts) > 1:
                model_response = response_parts[1].split("<end_of_turn>")[0].strip()
                if "Trả lời với vai trò là bác sĩ tâm lý:" in model_response:
                    model_response = model_response.split("Trả lời với vai trò là bác sĩ tâm lý:", 1)[1].strip()
                return model_response.strip()
            return response.strip()
        elif "qwen" in model_name:
            raw_response = response
            if '<|im_start|>' in raw_response and '<|im_end|>' in raw_response:
                parts = raw_response.split('<|im_start|>')
                for part in parts:
                    if '<|im_end|>' in part:
                        content = part.split('<|im_end|>')[0].strip()
                        content = re.sub(r'^(assistant|solver|response):', '', content, flags=re.IGNORECASE)
                        if content.strip():
                            raw_response = content.strip()
                            break
            patterns_to_remove = [
                r'^(Human|Assistant|Solver|Response):\s*',
                r'Here\'s (my |the )?(response|answer):\s*',
                r'<\|.*?\|>',
            ]
            cleaned_response = raw_response
            for pattern in patterns_to_remove:
                cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.IGNORECASE)
            if not cleaned_response.strip() and raw_response.strip():
                cleaned_response = raw_response
            return cleaned_response.strip()
        elif "llama" in model_name:
            response = response.strip()
            if response.startswith("<|start_header_id|>assistant<|end_header_id>"):
                response = response[len("<|start_header_id|>assistant<|end_header_id>"):].strip()
            if response.endswith("<|eot_id|>"):
                response = response[:-len("<|eot_id|>")].strip()
            patterns_to_remove = [
                r'<\|begin_of_text\|>',
                r'<\|start_header_id\|>.*?<\|end_header_id\|>',
                r'<\|eot_id\|>',
                r'^\s*Assistant:\s*',
                r'^\s*Response:\s*'
            ]
            for pattern in patterns_to_remove:
                response = re.sub(pattern, '', response, flags=re.IGNORECASE).strip()
            return response
        else:
            return response.strip()
    except Exception as e:
        logger.error(f"Error cleaning response for {model_name}: {str(e)}")
        return response.strip()

def ensure_complete_response(text: str) -> str:
    """Ensure the response has complete sentences"""
    if not text:
        return text
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in '.!?' and len(current.strip()) >= 10:
            sentences.append(current.strip())
            current = ""
    if current.strip() and len(current.strip()) >= 10 and any(current.strip().endswith(end) for end in '.!?'):
        sentences.append(current.strip())
    return ' '.join(sentences) if sentences else text

# ========== Main Answer Generation ==========
def generate_answer(
    conversation_id: str,
    query: str,
    llm_model,
    llm_tokenizer,
    collection_embeddings,
    bge_model,
    bge_tokenizer,
    reranker_model,
    reranker_tokenizer,
    bm25=None,
    alpha: float = 0.5,
    k_embed: int = 50,
    k_initial: int = 5,
    k_final: int = 3,
    max_history: int = 5
) -> str:
    """Generate an answer for the given query using the LLM and context."""
    logger.info(f"New Query | Conversation ID: {conversation_id} | Query: {query}")

    # Identity question shortcut
    if is_identity_question(query):
        identity_response = get_identity_response()
        save_message(conversation_id, "user", query)
        save_message(conversation_id, "chatbot", identity_response)
        return identity_response

    # Hybrid Search and Rerank
    logger.info("Performing Hybrid Search (BGE+BM25) & Rerank...")
    search_results = hybrid_search_and_rerank(
        query, collection_embeddings, bge_model, bge_tokenizer, reranker_model, reranker_tokenizer, bm25,
        alpha=alpha, k_embed_retrieval=k_embed, top_k_initial=k_initial, top_k_final=k_final
    )
    context_texts = [res['text'] for res in search_results] if search_results else []
    sources = [res['source'] for res in search_results] if search_results else []
    source_list = list(dict.fromkeys(s for s in sources if s and s != "Không có nguồn"))[:3]
    avg_rerank_score = sum(res['rerank_score'] for res in search_results) / len(search_results) if search_results else 0.0
    logger.info(f"Retrieved {len(context_texts)} context passages. Average rerank score: {avg_rerank_score:.2f}")
    if source_list:
        logger.info(f"Sources found: {source_list}")

    # Conversation History
    logger.info(f"Retrieving conversation history (max {max_history})...")
    history = get_conversation_history(conversation_id, max_history=max_history)
    history_prompt = "\n".join(
        f"{'Người dùng' if msg['role'] == 'user' else 'Trợ lý AI'}: {msg['text']}"
        for msg in history
    ) if history else ""
    if history:
        logger.info(f"History found ({len(history)} turns).")
    else:
        logger.info("No previous history found for this conversation.")

    # Build Prompt
    logger.info("Building prioritized prompt for LLM...")
    system_message = (
        "Bạn là một bác sĩ tâm lý chuyên về lĩnh vực sức khỏe tinh thần. Nhiệm vụ của bạn là trả lời câu hỏi của bệnh nhân bằng Tiếng Việt một cách rõ ràng, đồng cảm và chính xác đồng thời đưa ra những lời khuyên chân thành cho họ. Nếu người hỏi có ý định tự hại bản thân thì hãy ngăn họ lại và đưa ra những lời khuyên."
    )
    prompt_sections = [
        "QUAN TRỌNG: Hãy tuân thủ các quy tắc ưu tiên sau khi trả lời:",
        "1. ƯU TIÊN SỐ 1: Xem xét kỹ Lịch sử hội thoại gần đây. Nếu câu hỏi hiện tại của người dùng có liên quan trực tiếp hoặc có thể được trả lời dựa trên thông tin đã trao đổi trong lịch sử, hãy trả lời dựa CHỦ YẾU vào lịch sử đó.",
        "2. ƯU TIÊN SỐ 2: CHỈ KHI Lịch sử hội thoại không đủ thông tin, không liên quan đến câu hỏi hiện tại, hoặc không tồn tại, thì bạn MỚI được phép sử dụng Thông tin tham khảo được cung cấp dưới đây.",
        "3. Khi sử dụng Thông tin tham khảo, hãy tích hợp nó một cách tự nhiên vào câu trả lời, không chỉ liệt kê. Luôn giữ giọng văn thân thiện và hỗ trợ.",
        "4. Tuyệt đối không được bịa đặt thông tin không có trong lịch sử hoặc thông tin tham khảo.",
        "--- LỊCH SỬ HỘI THOẠI GẦN ĐÂY (Ưu tiên 1) ---",
        history_prompt or "--- Không có Lịch sử hội thoại ---",
        "--- Kết thúc Lịch sử hội thoại ---",
        "--- THÔNG TIN THAM KHẢO (Ưu tiên 2 - Chỉ dùng khi lịch sử không đủ) ---",
        "\n\n".join(f"- {ctx}" for ctx in context_texts) if context_texts else "--- Không tìm thấy Thông tin tham khảo ---",
        "--- Kết thúc Thông tin tham khảo ---",
        "--- CÂU HỎI HIỆN TẠI CỦA NGƯỜI DÙNG ---",
        query,
        "--- CÂU TRẢ LỜI CỦA BẠN (Hãy nhớ quy tắc ưu tiên ở trên) ---"
    ]
    user_message = "\n\n".join(prompt_sections).strip()

    # Generate Answer with LLM
    logger.info("Generating answer with LLM model...")
    llm_response = ""
    try:
        generation_params = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "do_sample": True,
            "repetition_penalty": REPETITION_PENALTY
        }
        model_name = str(llm_model.__class__).lower()
        context_window = getattr(getattr(llm_model, 'config', None), 'max_position_embeddings', DEFAULT_CONTEXT_WINDOW)
        prompt = generate_prompt_for_model(model_name, system_message, user_message, history_prompt, context_texts)

        if "gemma" in model_name:
            model_input = llm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=context_window - generation_params["max_new_tokens"],
                padding=True
            ).to(llm_model.device)
            outputs = llm_model.generate(
                model_input['input_ids'],
                pad_token_id=llm_tokenizer.pad_token_id,
                eos_token_id=llm_tokenizer.eos_token_id,
                **generation_params
            )
            full_response = llm_tokenizer.decode(outputs[0], skip_special_tokens=False)
            llm_response = clean_response("gemma", full_response, llm_tokenizer)

        elif "qwen" in model_name:
            if hasattr(llm_tokenizer, 'apply_chat_template'):
                input_text = llm_tokenizer.apply_chat_template(prompt, tokenize=False)
                model_inputs = llm_tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=context_window - generation_params["max_new_tokens"],
                    add_special_tokens=True
                ).to(llm_model.device)
                outputs = llm_model.generate(
                    model_inputs.input_ids,
                    pad_token_id=llm_tokenizer.pad_token_id,
                    eos_token_id=llm_tokenizer.eos_token_id,
                    **generation_params
                )
                full_response = llm_tokenizer.decode(outputs[0], skip_special_tokens=False)
                input_text_with_special = llm_tokenizer.decode(model_inputs.input_ids[0], skip_special_tokens=False)
                raw_response = full_response[len(input_text_with_special):].strip()
                llm_response = clean_response("qwen", raw_response, llm_tokenizer)
            else:
                raise ValueError("Qwen model requires apply_chat_template method in tokenizer")

        elif "llama" in model_name:
            model_input = llm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=context_window - generation_params["max_new_tokens"],
                add_special_tokens=False
            ).to(llm_model.device)
            stopping_criteria = StoppingCriteriaList([
                SentenceEndingCriteria(llm_tokenizer, min_length=MIN_SENTENCE_LENGTH)
            ])
            outputs = llm_model.generate(
                model_input['input_ids'],
                pad_token_id=llm_tokenizer.pad_token_id,
                eos_token_id=llm_tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                **generation_params
            )
            full_response = llm_tokenizer.decode(outputs[0], skip_special_tokens=False)
            llm_response = clean_response("llama", full_response, llm_tokenizer)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Ensure complete response
        llm_response = ensure_complete_response(llm_response)
        if not llm_response.strip():
            raise ValueError("Empty response generated")
    except Exception as e:
        logger.error(f"Error during LLM generation: {str(e)}")
        llm_response = "Xin lỗi, tôi gặp sự cố khi tạo câu trả lời. Vui lòng thử lại."

    # Add Sources and Suggestions
    final_response = llm_response
    if source_list and "Xin lỗi" not in llm_response and avg_rerank_score > 0.7:
        if "Nguồn tham khảo:" not in final_response:
            final_response += f"\n\nNguồn tham khảo:\n\n {', '.join(source_list)}"
    logger.info("Suggesting related questions...")
    related_questions = suggest_questions(query, history)
    if related_questions:
        if "Bạn có thể quan tâm:" not in final_response:
            suggestions = "\n\nBạn có thể quan tâm:\n"
            for i, q in enumerate(related_questions, 1):
                suggestions += f"\n{i}. {q}"
            final_response += suggestions

    # Save to History and Return
    logger.info("Saving conversation...")
    if llm_response and "Xin lỗi" not in llm_response:
        save_message(conversation_id, "user", query)
        save_message(conversation_id, "chatbot", final_response)
    logger.info(f"Final Response:\n{final_response}")
    logger.info("--- End Query ---")
    return final_response