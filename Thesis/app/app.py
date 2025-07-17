import streamlit as st
import time
import logging
from model_loader import load_models, unload_model
from data_processor import initialize_data
from mongo_manager import connect_to_mongodb, get_conversation_history, collection_history
from configuration import AVAILABLE_MODELS  # Add this import
from answer_generator import generate_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Chatbot Sức Khỏe Tinh Thần", layout="wide")

st.markdown("""
<style>
/* Main theme colors */
:root {
    --primary-color: #6C63FF;
    --secondary-color: #F8F9FA;
    --accent-color: #FFC107;
    --text-color: #212529;
    --light-text: #6c757d;
    --bot-message-color: #EFF6FF;
    --user-message-color: #6C63FF;
    --user-message-text: white;
    --bot-message-text: #333;
    --background-gradient: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
}

/* Main container styling */
.main .block-container {
    padding-top: 2rem;
    /* Giảm padding-bottom vì input bar đã fixed */
    padding-bottom: 2rem;
    max-width: 1200px;
    background: var(--background-gradient);
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
}

/* Header styling */
.title-centered {
    font-family: 'Poppins', sans-serif;
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--primary-color);
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    text-shadow: 0px 2px 4px rgba(0,0,0,0.1);
    background: linear-gradient(45deg, #6C63FF, #8E64FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.subtitle {
    font-family: 'Open Sans', sans-serif;
    font-size: 1rem;
    color: var(--light-text);
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Chat container */

.chat-container {
    /* --- MODIFIED --- */
    /* Đặt chiều cao tối đa, ví dụ 70% chiều cao viewport */
    /* Bạn có thể điều chỉnh giá trị vh này */
    max-height: 70vh;
    overflow-y: auto; /* Bật thanh cuộn dọc khi nội dung vượt quá */
    padding-bottom: 120px; /* !!! QUAN TRỌNG: Thêm khoảng đệm bằng hoặc lớn hơn chiều cao của input bar */
    /* --- END MODIFIED --- */

    /* Scrollbar styling (giữ nguyên) */
    scrollbar-width: thin; /* Firefox */
    scrollbar-color: #CED4DA transparent; /* Firefox */
}


.chat-container::-webkit-scrollbar {
    width: 8px;
}

.chat-container::-webkit-scrollbar-track {
    background: transparent;
}
.chat-container::-webkit-scrollbar-thumb {
    background-color: #CED4DA;
    border-radius: 20px;
    border: 3px solid transparent; /* Optional: creates padding around thumb */
    background-clip: content-box; /* Optional */
}

/* Message styling */
.message-container {
    width: 100%;
    overflow: hidden;
    margin-bottom: 20px;
    display: flex;
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.user-message-container {
    justify-content: flex-end;
}

.bot-message-container {
    justify-content: flex-start;
}

.chat-message {
    padding: 12px 18px;
    border-radius: 18px;
    max-width: 70%;
    word-wrap: break-word;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    position: relative;
    line-height: 1.5;
    font-size: 15px;
}

.user-message {
    background-color: var(--user-message-color);
    color: var(--user-message-text);
    border-bottom-right-radius: 4px;
}

.bot-message {
    background-color: var(--bot-message-color);
    color: var(--bot-message-text);
    border-bottom-left-radius: 4px;
}

.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 10px;
    font-size: 18px;
    flex-shrink: 0;
}

.user-avatar {
    background: linear-gradient(135deg, #7F7FD5, #86A8E7);
    color: white;
}

.bot-avatar {
    background: white;
    border: 2px solid var(--primary-color);
    color: var(--primary-color);
    box-shadow: 0 2px 8px rgba(108, 99, 255, 0.2);
}

/* Input container */
.input-container {
    position: fixed;   /* <-- Cố định vị trí */
    bottom: 0;         /* <-- Đặt ở dưới cùng */
    left: 0;           /* <-- Căn sát lề trái */
    right: 0;          /* <-- Kéo dài hết chiều ngang */
    background-color: white; /* Nền trắng để che nội dung bên dưới */
    padding: 15px;     /* Khoảng đệm bên trong */
    border-top: 1px solid rgba(0,0,0,0.05); /* Đường kẻ phân cách */
    z-index: 1000;      /* Đảm bảo nằm trên các yếu tố khác */
    box-shadow: 0 -5px 20px rgba(0,0,0,0.05); /* Đổ bóng */
    /* Các thuộc tính khác như display: flex, align-items: center giữ nguyên */
}

/* Form styling */
.stForm {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

.stForm > div {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

.custom-input > div > div > input {
    border-radius: 25px !important;
    padding: 12px 20px !important;
    width: 100% !important;
    border: 2px solid #E7E7E7 !important;
    font-size: 16px !important;
    transition: all 0.3s ease !important;
    box-shadow: none !important;
}

.custom-input > div > div > input:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.2) !important;
}

.custom-button button {
    border-radius: 25px !important;
    background: linear-gradient(45deg, var(--primary-color), #8662FF) !important;
    color: white !important;
    padding: 10px 30px !important;
    font-weight: 600 !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(108, 99, 255, 0.3) !important;
    width: 100% !important;
}

.custom-button button:hover {
    box-shadow: 0 8px 20px rgba(108, 99, 255, 0.4) !important;
    transform: translateY(-2px) !important;
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #F8F9FA;
    border-right: 1px solid rgba(0,0,0,0.05);
}

.css-1d391kg button {
    border-radius: 10px !important;
    margin-bottom: 8px !important;
    background-color: white !important;
    color: var(--text-color) !important;
    border: 1px solid #E9ECEF !important;
    transition: all 0.2s ease !important;
    padding: 8px 15px !important;
}

.css-1d391kg button:hover {
    background-color: #F1F3F5 !important;
    border-color: #DEE2E6 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05) !important;
}

/* REMOVED: Suggestion boxes CSS */
/*
.suggestion-grid { ... }
.suggestion-box { ... }
.suggestion-grid::-webkit-scrollbar { ... }
.suggestion-grid::-webkit-scrollbar-track { ... }
.suggestion-grid::-webkit-scrollbar-thumb { ... }
.suggestion-grid::-webkit-scrollbar-thumb:hover { ... }
*/

/* Loading animation */
@keyframes typing {
    0% { width: 0; }
    33% { width: 6px; }
    66% { width: 12px; }
    100% { width: 18px; }
}

.typing-indicator {
    display: flex;
    align-items: center;
    margin: 8px 0;
}

.typing-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #aaa;
    margin-right: 4px;
    animation: blink 1.5s infinite;
}

.typing-dot:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-dot:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes blink {
    0%, 100% { opacity: 0.2; }
    50% { opacity: 1; }
}

/* Disclaimer text */
.disclaimer {
    font-size: 13px;
    color: #888;
    text-align: center;
    margin-top: 6px;
    font-style: italic;
}

/* Sidebar header */
.sidebar-header {
    display: flex;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid #e9ecef;
}

.sidebar-header h2 {
    margin: 0;
    font-size: 1.5rem;
    color: var(--primary-color);
    font-weight: 600;
    display: flex;
    align-items: center;
}

.sidebar-header h2 svg {
    margin-right: 10px;
}

.sidebar-actions {
    margin-top: 20px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

/* Action buttons */
.action-button {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    font-weight: 500;
    padding: 10px !important;
}

.new-chat-btn {
    background: linear-gradient(45deg, #2ECC71, #27AE60) !important;
    color: white !important;
}

.delete-btn {
    background: linear-gradient(45deg, #E74C3C, #C0392B) !important;
    color: white !important;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 40px 20px;
    color: var(--light-text);
}

.empty-state img {
    max-width: 180px;
    margin-bottom: 20px;
    opacity: 0.7;
}

.empty-state h3 {
    font-size: 1.4rem;
    margin-bottom: 10px;
    color: var(--text-color);
}

.empty-state p {
    font-size: 0.9rem;
    max-width: 400px;
    margin: 0 auto;
}

/* Conversation history styles */
.chat-history-card {
    padding: 12px;
    border-radius: 8px;
    background-color: white;
    margin-bottom: 10px;
    border: 1px solid #E2E8F0;
    transition: all 0.2s ease;
}

.chat-history-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-color: var(--primary-color);
}

.chat-history-card.active {
    background-color: #EFF6FF;
    border-color: var(--primary-color);
}

.chat-history-title {
    font-weight: 500;
    color: #2D3748;
    margin-bottom: 4px;
}

.chat-history-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.8em;
    color: #718096;
}

.chat-history-container {
    max-height: 60vh;
    overflow-y: auto;
    padding-right: 10px;
}

.chat-history-container::-webkit-scrollbar {
    width: 6px;
}

.chat-history-container::-webkit-scrollbar-track {
    background: transparent;
}

.chat-history-container::-webkit-scrollbar-thumb {
    background-color: #CBD5E0;
    border-radius: 3px;
}

/* Conversation ID style */
.conv-id {
    font-family: monospace;
    color: #6C63FF;
    font-size: 0.8em;
    background: #EFF6FF;
    padding: 2px 6px;
    border-radius: 4px;
    margin-right: 8px;
}

.chat-history-card {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.chat-history-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.chat-history-preview {
    color: #666;
    font-size: 0.9em;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
</style>

<script>
// Helper function to find an element in all potential Streamlit iframes
function findElementInIframes(selector) {
    const iframes = Array.from(window.top.document.getElementsByTagName('iframe'));
    for (const iframe of iframes) {
        try {
            const doc = iframe.contentWindow.document;
            const element = doc.querySelector(selector);
            if (element) {
                return { element, doc };
            }
        } catch (e) {
            continue;
        }
    }
    return null;
}

// Helper function to trigger events on an input element
function triggerInputEvents(input, value) {
    // Set the value
    input.value = value;
    
    // Use native setter in case of any custom handling
    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    nativeInputValueSetter.call(input, value);
    
    // Trigger a set of events to ensure the change is recognized
    const events = [
        new Event('change', { bubbles: true }),
        new InputEvent('input', { 
            bubbles: true,
            cancelable: true,
            inputType: 'insertText',
            data: value
        }),
        new KeyboardEvent('keydown', {
            bubbles: true,
            cancelable: true,
            key: 'Enter',
            code: 'Enter'
        })
    ];
    
    events.forEach(event => input.dispatchEvent(event));
}

// Main function to insert a question and trigger form submission
function insertQuestion(question) {
    try {
        // Find the input element
        const result = findElementInIframes('input[aria-label="query_input"], input[placeholder="Nhập câu hỏi của bạn ở đây..."]');
        
        if (!result) {
            console.error('Could not find input element');
            return;
        }
        
        const { element: input, doc } = result;
        
        // Focus the input first
        input.focus();
        
        // Set value and trigger events
        triggerInputEvents(input, question);
        
        // Find and click submit button
        const form = input.closest('form');
        if (form) {
            const submitButton = form.querySelector('button[type="submit"]');
            if (submitButton) {
                setTimeout(() => submitButton.click(), 100);
            }
        }
    } catch (e) {
        console.error('Error in insertQuestion:', e);
    }
}

// Initialize event handling for suggested questions
function initializeSuggestedQuestions() {
    // Set up click handler
    document.addEventListener('click', function(e) {
        const target = e.target.closest('.suggested-question');
        if (target) {
            e.preventDefault();
            e.stopPropagation();
            const question = target.getAttribute('data-question') || target.textContent.trim();
            if (question) {
                target.style.backgroundColor = 'var(--primary-color)';
                target.style.color = 'white';
                insertQuestion(question);
                setTimeout(() => {
                    target.style.backgroundColor = '';
                    target.style.color = '';
                }, 200);
            }
        }
    });
    
    // Set up keyboard handler for accessibility
    document.addEventListener('keydown', function(e) {
        const target = e.target.closest('.suggested-question');
        if (target && (e.key === 'Enter' || e.key === ' ')) {
            e.preventDefault();
            const question = target.getAttribute('data-question') || target.textContent.trim();
            if (question) {
                insertQuestion(question);
            }
        }
    });
}

// Run initialization when the DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeSuggestedQuestions);
} else {
    initializeSuggestedQuestions();
}

// Set up mutation observer to handle dynamically added elements
const observer = new MutationObserver((mutations) => {
    mutations.forEach(mutation => {
        mutation.addedNodes.forEach(node => {
            if (node.nodeType === 1) {
                if (node.classList && node.classList.contains('suggested-question')) {
                    setupSuggestedQuestion(node);
                }
                const suggestions = node.getElementsByClassName('suggested-question');
                Array.from(suggestions).forEach(setupSuggestedQuestion);
            }
        });
    });
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});

// Helper to set up individual suggestion elements
function setupSuggestedQuestion(element) {
    if (!element.hasAttribute('data-question')) {
        element.setAttribute('data-question', element.textContent.trim());
    }
    element.setAttribute('tabindex', '0');
    element.setAttribute('role', 'button');
}

// Initialize any existing suggestions
document.querySelectorAll('.suggested-question').forEach(setupSuggestedQuestion);
</script>
""", unsafe_allow_html=True)

# Update the select_conversation function
def select_conversation(conv_id):
    """Handle conversation selection and loading"""
    try:
        # Update current conversation ID
        st.session_state.conversation_id = conv_id
        
        # Get full conversation history
        history = get_conversation_history(conv_id)
        if history:
            # Update messages in session state
            st.session_state.messages = [
                {"role": msg["role"], "text": msg["text"]} 
                for msg in history
            ]
            return True
        else:
            st.warning("Không thể tải lịch sử hội thoại này.")
            return False
    except Exception as e:
        logger.error(f"Error selecting conversation {conv_id}: {str(e)}")
        st.error("Lỗi khi tải lịch sử hội thoại.")
        return False

# REMOVED: Function to handle suggestion click
# def handle_suggestion_click(suggestion_text):
#     st.session_state.input_text = suggestion_text
#     st.session_state.submit_suggestion = True

# Function to handle conversation selection
def select_conversation(conv_id):
    st.session_state.conversation_id = conv_id
    history = get_conversation_history(conv_id, max_history=50)
    st.session_state.messages = [{"role": msg["role"], "text": msg["text"]} for msg in history]

# ========== Session State ==========
def initialize_system(model_name):
    st.write("Bắt đầu khởi tạo hệ thống...")
    system_data = {}
    
    with st.spinner("🔄 Đang tải các mô hình AI..."):
        try:
            models = load_models(model_name)
            if not models:
                st.error("❌ Không thể tải mô hình!")
                return None
            
            system_data["bge"] = models["bge"]
            system_data["reranker"] = models["reranker"]
            system_data["llm"] = models["llm"]
            st.success("✅ Đã tải xong các mô hình AI!")
        except Exception as e:
            st.error(f"❌ Lỗi khi tải mô hình: {str(e)}")
            logger.error(f"Model loading failed: {str(e)}")
            return None

    with st.spinner("🔄 Đang xử lý dữ liệu..."):
        try:
            documents_data, bm25, collection_embeddings = initialize_data(system_data["bge"][0], system_data["bge"][1])
            if not collection_embeddings:
                st.error("❌ Không thể xử lý dữ liệu!")
                return None
            system_data["bm25"] = bm25
            system_data["collection_embeddings"] = collection_embeddings
            st.success("✅ Đã xử lý xong dữ liệu!")
        except Exception as e:
            st.error(f"❌ Lỗi khi xử lý dữ liệu: {str(e)}")
            return None

    with st.spinner("🔄 Đang kết nối MongoDB..."):
        try:
            connect_to_mongodb()
            st.success("✅ Đã kết nối MongoDB!")
        except Exception as e:
            st.error(f"❌ Lỗi khi kết nối MongoDB: {str(e)}")
            return None

    return system_data

# ========== Session State ==========
# Initialize selected_model with the first model from AVAILABLE_MODELS
if "selected_model" not in st.session_state:
    st.session_state.selected_model = list(AVAILABLE_MODELS.keys())[0]  # This will be "Qwen 2.5B"

if "system" not in st.session_state or st.session_state.system is None:
    initial_system = initialize_system(AVAILABLE_MODELS[st.session_state.selected_model])
    if initial_system is None:
        st.error("Failed to initialize system. Please refresh the page.")
        st.stop()
    st.session_state.system = initial_system

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = f"chat_{int(time.time())}"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_ids" not in st.session_state:
    st.session_state.conversation_ids = []
if "input_text" not in st.session_state: # Kept for potential future use or direct input value
    st.session_state.input_text = ""
if "is_typing" not in st.session_state:
    st.session_state.is_typing = False

# ========== Tiêu đề ==========
st.markdown('<div class="title-centered">Chatbot Sức Khỏe Tinh Thần</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hỗ trợ bạn với các vấn đề về sức khỏe tinh thần một cách đồng cảm và chuyên nghiệp.</div>', unsafe_allow_html=True)

# ========== REMOVED: Gợi ý câu hỏi ==========
# if not st.session_state.messages:
#     suggestions = [ ... ]
#     st.markdown('<div class="suggestion-grid">', unsafe_allow_html=True)
#     for i, sugg in enumerate(suggestions):
#         st.markdown(f''' ... ''', unsafe_allow_html=True)
#         if st.button("", key=f"sugg_btn_{i}", type="primary", use_container_width=True):
#             handle_suggestion_click(sugg['question'])
#     st.markdown('</div>', unsafe_allow_html=True)

# ========== Chatbox ==========
st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)
chat_placeholder = st.empty()
with chat_placeholder.container():
    if not st.session_state.messages:
        # Display initial bot message only if no history exists
        st.markdown('''
        <div class="message-container bot-message-container">
            <div class="avatar bot-avatar">🤖</div>
            <div class="chat-message bot-message">
                Chào bạn! Tôi là chatbot sức khỏe tinh thần. Bạn khỏe không? Hãy hỏi tôi bất cứ điều gì về sức khỏe tinh thần, tôi sẽ cố gắng hỗ trợ bạn một cách tốt nhất!
            </div>
        </div>
        ''', unsafe_allow_html=True)
        # REMOVED: Displaying suggestions here, now just the welcome message
    else:
        # Display existing messages
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'''
                <div class="message-container user-message-container">
                    <div class="chat-message user-message">{msg["text"]}</div>
                    <div class="avatar user-avatar">👤</div>
                </div>
                ''', unsafe_allow_html=True)
            else: # chatbot
                st.markdown(f'''
                <div class="message-container bot-message-container">
                    <div class="avatar bot-avatar">🤖</div>
                    <div class="chat-message bot-message">{msg["text"]}</div>
                </div>
                ''', unsafe_allow_html=True)

    # Display typing indicator if the bot is processing
    if st.session_state.is_typing:
        st.markdown('''
        <div class="message-container bot-message-container">
            <div class="avatar bot-avatar">🤖</div>
            <div class="chat-message bot-message">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# Script để cuộn xuống dưới cùng
st.markdown("""
<script>
    function scrollToBottom() {
        const chatContainer = document.getElementById('chat-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }

    // Run on page load and after any changes
    document.addEventListener('DOMContentLoaded', scrollToBottom);
    setTimeout(scrollToBottom, 500); // Ensure it runs after initial render

    // Add mutation observer to monitor changes in the chat container
    const observer = new MutationObserver(scrollToBottom);
    setTimeout(() => {
        const chatContainer = document.getElementById('chat-container');
        if (chatContainer) {
            observer.observe(chatContainer, { childList: true, subtree: true });
        }
    }, 1000); // Delay observer setup to ensure container exists
</script>
""", unsafe_allow_html=True)

# ========== Nhập tin nhắn ==========
# Use st.container() to group the input elements
input_container = st.container()
with input_container:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            # Use CSS class for custom styling
            st.markdown('<div class="custom-input">', unsafe_allow_html=True)
            query = st.text_input(
                "query_input",
                value="",
                label_visibility="collapsed",
                placeholder="Nhập câu hỏi của bạn ở đây...",
                key="query_input"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
             # Use CSS class for custom styling
            st.markdown('<div class="custom-button">', unsafe_allow_html=True)
            submitted = st.form_submit_button("Gửi")
            st.markdown('</div>', unsafe_allow_html=True)

        # Disclaimer text below the input field
        st.markdown('<div class="disclaimer">🤖 Chatbot có thể mắc lỗi. Hãy kiểm tra các thông tin quan trọng.</div>', unsafe_allow_html=True)

        # REMOVED: Handle suggestion submission logic
        # if st.session_state.submit_suggestion:
        #     query = st.session_state.input_text
        #     st.session_state.submit_suggestion = False
        #     submitted = True

        if submitted and query.strip():
            st.session_state.messages.append({"role": "user", "text": query})
            st.session_state.input_text = "" # Clear input text state if needed
            st.session_state.is_typing = True
            st.rerun() # Rerun to show user message and typing indicator

# Processing after rerun to show the typing indicator
if st.session_state.is_typing and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    try:
        system = st.session_state.system
        if system is None or "llm" not in system:
            raise ValueError("System not properly initialized")
            
        user_query = st.session_state.messages[-1]["text"]
        llm_model, llm_tokenizer = system["llm"]
        
        answer = generate_answer(
            st.session_state.conversation_id, user_query,
            llm_model, llm_tokenizer,
            system["collection_embeddings"], system["bge"][0], system["bge"][1],
            system["reranker"][0], system["reranker"][1], system["bm25"],
            alpha=0.5, k_embed=50, k_initial=20, k_final=3, max_history=5
        )

        st.session_state.messages.append({"role": "chatbot", "text": answer})
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}", exc_info=True)
        st.session_state.messages.append({
            "role": "chatbot", 
            "text": "Xin lỗi, tôi gặp lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại sau."
        })
    finally:
        st.session_state.is_typing = False
        st.rerun()

# ========== Sidebar: Lịch sử hội thoại ==========
with st.sidebar:
    st.markdown('<div class="sidebar-header"><h2>📜 Lịch Sử Hội Thoại</h2></div>', unsafe_allow_html=True)

    if collection_history is not None:
        try:
            all_conversations = collection_history.find({}, {"conversation_id": 1, "messages": 1})
            sorted_conversations = sorted(
                all_conversations,
                key=lambda x: int(x["conversation_id"].replace("chat_", "")) if x["conversation_id"].startswith("chat_") else 0,
                reverse=True
            )
            
            if sorted_conversations:
                st.caption(f"Tổng số cuộc hội thoại: {len(sorted_conversations)}")
                st.markdown("---")

                st.markdown('<div class="chat-history-container">', unsafe_allow_html=True)
                
                for conv in sorted_conversations:
                    conv_id = conv["conversation_id"]
                    messages = conv.get("messages", [])
                    
                    if messages:
                        first_user_msg = next((msg["text"] for msg in messages if msg["role"] == "user"), "Không có tin nhắn")
                        msg_count = len(messages)
                        timestamp = int(conv_id.replace("chat_", ""))
                        formatted_time = time.strftime("%d/%m/%Y %H:%M", time.localtime(timestamp))
                        
                        is_active = conv_id == st.session_state.conversation_id
                        st.markdown(f"""
                            <div class="chat-history-card{'active' if is_active else ''}" 
                                 id="conv_{conv_id}">
                                <div class="chat-history-header">
                                    <span class="conv-id">{conv_id}</span>
                                    <span>💬 {msg_count}</span>
                                </div>
                                <div class="chat-history-preview">
                                    {first_user_msg[:50] + '...' if len(first_user_msg) > 50 else first_user_msg}
                                </div>
                                <div class="chat-history-meta">
                                    <span>🕒 {formatted_time}</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button("", key=f"btn_{conv_id}", help=f"Xem lịch sử chat {conv_id}", use_container_width=True):
                            # Load full conversation history when clicked
                            st.session_state.conversation_id = conv_id
                            st.session_state.messages = [
                                {"role": msg["role"], "text": msg["text"]}
                                for msg in messages
                            ]
                            st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.markdown('''
                <div class="empty-state">
                    <img src="https://img.icons8.com/clouds/100/000000/chat.png" alt="No conversations">
                    <h3>Chưa có cuộc hội thoại</h3>
                    <p>Bắt đầu trò chuyện với chatbot để lưu lịch sử hội thoại của bạn.</p>
                </div>
                ''', unsafe_allow_html=True)
                
        except Exception as e:
            logger.error(f"Error loading conversation history: {str(e)}")
            st.error("Không thể tải lịch sử hội thoại.")

    st.markdown('<div class="sidebar-actions">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 Tạo mới", key="new_chat", use_container_width=True, help="Tạo cuộc trò chuyện mới", type="primary"):
            # Save current conversation ID before creating new one
            current_conv_id = st.session_state.conversation_id
            if current_conv_id not in st.session_state.conversation_ids and len(st.session_state.messages) > 0:
                st.session_state.conversation_ids.insert(0, current_conv_id) # Add to top

            # Create new conversation
            st.session_state.messages = []
            st.session_state.conversation_id = f"chat_{int(time.time())}"
            st.success("Đã tạo cuộc hội thoại mới!")
            time.sleep(1) # Short delay for user to see success
            st.success("Đã tạo cuộc hội thoại mới!")
            time.sleep(1) # Short delay for user to see success message
            st.rerun()

    with col2:
        if st.button("🗑️ Xóa tất cả", key="delete_all", use_container_width=True, help="Xóa toàn bộ lịch sử hội thoại"):
            if collection_history is not None:
                try:
                    delete_result = collection_history.delete_many({})
                    logger.info(f"Deleted {delete_result.deleted_count} conversations.")
                    st.session_state.conversation_ids = []
                    st.session_state.messages = []
                    # Reset to a new conversation ID after deleting all
                    st.session_state.conversation_id = f"chat_{int(time.time())}"
                    st.success("Đã xóa toàn bộ lịch sử hội thoại!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                     logger.error(f"Error deleting conversations: {str(e)}")
                     st.error("Không thể xóa lịch sử hội thoại.")
            else:
                 st.warning("Không thể kết nối đến cơ sở dữ liệu để xóa.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Additional sidebar content
    st.markdown("""
    <div style="margin-top: 40px; padding: 15px; background-color: #e9ecef; border-radius: 10px; border-left: 4px solid #6C63FF;">
        <h4 style="margin-top: 0; color: #212529;">💡 Mẹo sử dụng</h4>
        <ul style="margin-bottom: 0; padding-left: 20px; font-size: 14px; color: #495057;">
            <li>Hỏi về các phương pháp đối phó với lo âu</li>
            <li>Tìm hiểu về kỹ thuật thư giãn</li>
            <li>Các dấu hiệu của trầm cảm và stress</li>
            <li>Phương pháp cải thiện giấc ngủ</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # ========== Model selection ==========
    # st.markdown('<div class="sidebar-header"><h2>⚙️ Cài đặt</h2></div>', unsafe_allow_html=True)
    
    # Model selection with safe index lookup
    current_index = 0  # Default to first model
    if st.session_state.selected_model in AVAILABLE_MODELS.keys():
        current_index = list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model)
    
    selected_model_name = st.selectbox(
        "Chọn mô hình",
        options=list(AVAILABLE_MODELS.keys()),
        index=current_index,
        key="model_selector"
    )
    
    # Handle model switching
    if selected_model_name != st.session_state.selected_model:
        with st.spinner("Đang chuyển đổi mô hình..."):
            try:
                # Unload current LLM model if it exists
                if st.session_state.system is not None and "llm" in st.session_state.system:
                    current_model, _ = st.session_state.system["llm"]
                    unload_model(current_model)
                
                # Update selected model
                st.session_state.selected_model = selected_model_name
                
                # Reload models
                new_system = initialize_system(AVAILABLE_MODELS[selected_model_name])
                if new_system is not None:
                    st.session_state.system = new_system
                    st.success(f"Đã chuyển sang mô hình {selected_model_name}!")
                else:
                    st.error("Lỗi khi chuyển đổi mô hình!")
                    if st.session_state.system is None:
                        st.stop()
            except Exception as e:
                st.error(f"Lỗi khi chuyển đổi mô hình: {str(e)}")
                logger.error(f"Error switching models: {str(e)}")
                if st.session_state.system is None:
                    st.stop()
            st.rerun()