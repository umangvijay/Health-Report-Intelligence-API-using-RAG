"""
ADVANCED UI COMPONENTS
======================
Modern UI/UX with animations for AI Doctor

Features:
- Animated login/signup pages
- Glassmorphism design
- Smooth transitions
- Loading animations
- Medical-themed styling
- Responsive design
"""

# CSS Styles with animations
ADVANCED_CSS = """
<style>
/* ============ ROOT VARIABLES ============ */
:root {
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --secondary: #10b981;
    --accent: #8b5cf6;
    --danger: #ef4444;
    --warning: #f59e0b;
    --dark: #1e293b;
    --light: #f8fafc;
    --glass: rgba(255, 255, 255, 0.1);
}

/* ============ ANIMATIONS ============ */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-50px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes slideInRight {
    from { opacity: 0; transform: translateX(50px); }
    to { opacity: 1; transform: translateX(0); }
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

@keyframes heartbeat {
    0%, 100% { transform: scale(1); }
    25% { transform: scale(1.1); }
    50% { transform: scale(1); }
    75% { transform: scale(1.1); }
}

@keyframes typing {
    from { width: 0; }
    to { width: 100%; }
}

/* ============ GLOBAL STYLES ============ */
.stApp {
    background: linear-gradient(-45deg, #0f172a, #1e293b, #0f172a, #1e3a5f);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
}

/* ============ LOGIN/SIGNUP PAGE ============ */
.auth-container {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    animation: fadeIn 0.8s ease-out;
}

.auth-card {
    background: rgba(30, 41, 59, 0.8);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 3rem;
    max-width: 450px;
    width: 100%;
    box-shadow: 
        0 25px 50px -12px rgba(0, 0, 0, 0.5),
        0 0 0 1px rgba(255, 255, 255, 0.05);
    animation: slideInLeft 0.6s ease-out;
}

.auth-card:hover {
    transform: translateY(-5px);
    box-shadow: 
        0 30px 60px -12px rgba(0, 0, 0, 0.6),
        0 0 40px rgba(37, 99, 235, 0.1);
    transition: all 0.3s ease;
}

.auth-logo {
    text-align: center;
    margin-bottom: 2rem;
    animation: float 3s ease-in-out infinite;
}

.auth-logo svg {
    width: 80px;
    height: 80px;
    fill: var(--primary);
}

.auth-title {
    font-size: 2rem;
    font-weight: 700;
    text-align: center;
    color: white;
    margin-bottom: 0.5rem;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: fadeIn 0.8s ease-out 0.2s backwards;
}

.auth-subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 2rem;
    animation: fadeIn 0.8s ease-out 0.4s backwards;
}

/* ============ FORM INPUTS ============ */
.stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 2px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    color: white !important;
    padding: 1rem 1.25rem !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.2) !important;
    background: rgba(255, 255, 255, 0.08) !important;
}

.stTextInput > div > div > input::placeholder {
    color: #64748b !important;
}

/* ============ BUTTONS ============ */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.875rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(37, 99, 235, 0.4) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* Secondary button */
.secondary-btn > button {
    background: transparent !important;
    border: 2px solid rgba(255, 255, 255, 0.2) !important;
    box-shadow: none !important;
}

.secondary-btn > button:hover {
    background: rgba(255, 255, 255, 0.05) !important;
    border-color: var(--primary) !important;
}

/* ============ CARDS ============ */
.glass-card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    animation: fadeIn 0.5s ease-out;
    transition: all 0.3s ease;
}

.glass-card:hover {
    background: rgba(30, 41, 59, 0.8);
    border-color: rgba(255, 255, 255, 0.2);
    transform: translateY(-3px);
}

/* ============ CHAT INTERFACE ============ */
.chat-container {
    background: rgba(15, 23, 42, 0.8);
    border-radius: 20px;
    padding: 1.5rem;
    height: 500px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.chat-message {
    display: flex;
    margin-bottom: 1rem;
    animation: slideInLeft 0.3s ease-out;
}

.chat-message.user {
    flex-direction: row-reverse;
    animation: slideInRight 0.3s ease-out;
}

.chat-bubble {
    max-width: 70%;
    padding: 1rem 1.25rem;
    border-radius: 18px;
    font-size: 0.95rem;
    line-height: 1.5;
}

.chat-bubble.ai {
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.2), rgba(139, 92, 246, 0.2));
    border: 1px solid rgba(37, 99, 235, 0.3);
    color: #e2e8f0;
}

.chat-bubble.user {
    background: linear-gradient(135deg, var(--primary), var(--accent));
    color: white;
}

/* ============ LOADING ANIMATIONS ============ */
.loading-container {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
}

.loading-spinner {
    width: 50px;
    height: 50px;
    border: 4px solid rgba(255, 255, 255, 0.1);
    border-top-color: var(--primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

.loading-dots {
    display: flex;
    gap: 8px;
}

.loading-dot {
    width: 12px;
    height: 12px;
    background: var(--primary);
    border-radius: 50%;
    animation: pulse 1.5s ease-in-out infinite;
}

.loading-dot:nth-child(2) { animation-delay: 0.2s; }
.loading-dot:nth-child(3) { animation-delay: 0.4s; }

/* ============ PROGRESS BARS ============ */
.progress-container {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    height: 8px;
    overflow: hidden;
    margin: 1rem 0;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    border-radius: 10px;
    transition: width 0.5s ease;
}

/* Shimmer effect */
.shimmer {
    background: linear-gradient(90deg, 
        rgba(255,255,255,0) 0%,
        rgba(255,255,255,0.1) 50%,
        rgba(255,255,255,0) 100%);
    background-size: 200% 100%;
    animation: shimmer 2s infinite;
}

/* ============ MEDICAL ICONS ============ */
.medical-icon {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: heartbeat 2s ease-in-out infinite;
}

/* ============ FEATURE CARDS ============ */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.feature-card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    animation: fadeIn 0.5s ease-out;
}

.feature-card:hover {
    transform: translateY(-8px);
    border-color: var(--primary);
    box-shadow: 0 20px 40px rgba(37, 99, 235, 0.2);
}

.feature-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

.feature-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: white;
    margin-bottom: 0.5rem;
}

.feature-desc {
    color: #94a3b8;
    font-size: 0.9rem;
}

/* ============ STATS DISPLAY ============ */
.stats-container {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 1.5rem 0;
}

.stat-card {
    flex: 1;
    min-width: 150px;
    background: rgba(37, 99, 235, 0.1);
    border: 1px solid rgba(37, 99, 235, 0.3);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    animation: fadeIn 0.5s ease-out;
}

.stat-value {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stat-label {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 0.25rem;
}

/* ============ ALERTS ============ */
.alert {
    padding: 1rem 1.25rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    animation: slideInLeft 0.3s ease-out;
}

.alert-success {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
}

.alert-warning {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fbbf24;
}

.alert-error {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: #f87171;
}

.alert-info {
    background: rgba(37, 99, 235, 0.1);
    border: 1px solid rgba(37, 99, 235, 0.3);
    color: #60a5fa;
}

/* ============ SIDEBAR ============ */
.css-1d391kg, [data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95) !important;
    backdrop-filter: blur(20px) !important;
}

.sidebar-header {
    padding: 1.5rem;
    text-align: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.sidebar-nav-item {
    padding: 0.75rem 1rem;
    margin: 0.25rem 0.5rem;
    border-radius: 8px;
    color: #94a3b8;
    cursor: pointer;
    transition: all 0.2s ease;
}

.sidebar-nav-item:hover {
    background: rgba(37, 99, 235, 0.1);
    color: white;
}

.sidebar-nav-item.active {
    background: var(--primary);
    color: white;
}

/* ============ MEDICAL REPORT CARD ============ */
.report-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.9));
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    animation: fadeIn 0.5s ease-out;
}

.report-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.report-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: white;
}

.report-badge {
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

.badge-normal {
    background: rgba(16, 185, 129, 0.2);
    color: #34d399;
}

.badge-warning {
    background: rgba(245, 158, 11, 0.2);
    color: #fbbf24;
}

.badge-critical {
    background: rgba(239, 68, 68, 0.2);
    color: #f87171;
}

/* ============ RESPONSIVE ============ */
@media (max-width: 768px) {
    .auth-card {
        padding: 2rem;
        margin: 1rem;
    }
    
    .feature-grid {
        grid-template-columns: 1fr;
    }
    
    .stats-container {
        flex-direction: column;
    }
}

/* ============ SCROLLBAR ============ */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.3);
}

/* ============ TRANSITIONS ============ */
* {
    transition: background-color 0.2s ease, 
                border-color 0.2s ease,
                color 0.2s ease;
}
</style>
"""

# HTML Components
def get_animated_logo():
    """Return animated medical logo SVG"""
    return """
    <div class="auth-logo">
        <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#3b82f6"/>
                    <stop offset="100%" style="stop-color:#8b5cf6"/>
                </linearGradient>
            </defs>
            <circle cx="50" cy="50" r="45" fill="none" stroke="url(#logoGrad)" stroke-width="3"/>
            <path d="M50 25 L50 75 M25 50 L75 50" stroke="url(#logoGrad)" stroke-width="8" stroke-linecap="round"/>
            <circle cx="50" cy="50" r="8" fill="url(#logoGrad)">
                <animate attributeName="r" values="8;12;8" dur="2s" repeatCount="indefinite"/>
            </circle>
        </svg>
    </div>
    """

def get_loading_animation():
    """Return loading animation HTML"""
    return """
    <div class="loading-container">
        <div class="loading-dots">
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
            <div class="loading-dot"></div>
        </div>
    </div>
    """

def get_spinner():
    """Return spinner animation HTML"""
    return """
    <div class="loading-container">
        <div class="loading-spinner"></div>
    </div>
    """

def get_feature_cards():
    """Return feature cards HTML"""
    features = [
        ("🔬", "AI Diagnosis", "Advanced ensemble of 12+ medical AI models"),
        ("💊", "Drug Lookup", "50+ medicines with interactions & dosage"),
        ("📊", "Blood Analysis", "Automated parameter interpretation"),
        ("🩻", "Image Analysis", "X-ray, CT, MRI classification"),
        ("🧠", "Smart Learning", "RLHF continuous improvement"),
        ("🔒", "Privacy First", "HIPAA-compliant, local processing"),
    ]
    
    cards_html = '<div class="feature-grid">'
    for icon, title, desc in features:
        cards_html += f'''
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        '''
    cards_html += '</div>'
    return cards_html

def get_stats_display(models_loaded: int, accuracy: float, queries: int):
    """Return stats display HTML"""
    return f"""
    <div class="stats-container">
        <div class="stat-card">
            <div class="stat-value">{models_loaded}</div>
            <div class="stat-label">Models Active</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{accuracy:.1f}%</div>
            <div class="stat-label">Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{queries:,}</div>
            <div class="stat-label">Queries Today</div>
        </div>
    </div>
    """

def get_alert(message: str, alert_type: str = "info"):
    """Return alert HTML"""
    icons = {
        "success": "✓",
        "warning": "⚠",
        "error": "✕",
        "info": "ℹ"
    }
    return f"""
    <div class="alert alert-{alert_type}">
        <span>{icons.get(alert_type, 'ℹ')}</span>
        <span>{message}</span>
    </div>
    """

def get_chat_bubble(message: str, is_user: bool = False):
    """Return chat bubble HTML"""
    bubble_class = "user" if is_user else "ai"
    return f"""
    <div class="chat-message {bubble_class}">
        <div class="chat-bubble {bubble_class}">
            {message}
        </div>
    </div>
    """

def get_progress_bar(value: int, max_value: int = 100):
    """Return progress bar HTML"""
    percentage = min(100, max(0, (value / max_value) * 100))
    return f"""
    <div class="progress-container">
        <div class="progress-bar" style="width: {percentage}%"></div>
    </div>
    """

def get_report_card(title: str, status: str, content: str):
    """Return medical report card HTML"""
    badge_class = {
        "normal": "badge-normal",
        "warning": "badge-warning", 
        "critical": "badge-critical"
    }.get(status.lower(), "badge-normal")
    
    return f"""
    <div class="report-card">
        <div class="report-header">
            <div class="report-title">{title}</div>
            <div class="report-badge {badge_class}">{status.upper()}</div>
        </div>
        <div class="report-content">
            {content}
        </div>
    </div>
    """


# Streamlit helper functions
def inject_css():
    """Inject CSS into Streamlit app"""
    import streamlit as st
    st.markdown(ADVANCED_CSS, unsafe_allow_html=True)

def show_login_page():
    """Show animated login page"""
    import streamlit as st
    
    inject_css()
    
    st.markdown(get_animated_logo(), unsafe_allow_html=True)
    st.markdown('<h1 class="auth-title">AI Doctor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="auth-subtitle">Your Intelligent Medical Assistant</p>', unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        col1, col2 = st.columns(2)
        with col1:
            submit = st.form_submit_button("Sign In", use_container_width=True)
        with col2:
            st.markdown('<div class="secondary-btn">', unsafe_allow_html=True)
            signup = st.form_submit_button("Create Account", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    return username, password, submit, signup

def show_loading(message: str = "Processing..."):
    """Show loading animation"""
    import streamlit as st
    
    with st.spinner(message):
        st.markdown(get_loading_animation(), unsafe_allow_html=True)

def show_success(message: str):
    """Show success alert"""
    import streamlit as st
    st.markdown(get_alert(message, "success"), unsafe_allow_html=True)

def show_error(message: str):
    """Show error alert"""
    import streamlit as st
    st.markdown(get_alert(message, "error"), unsafe_allow_html=True)

def show_warning(message: str):
    """Show warning alert"""
    import streamlit as st
    st.markdown(get_alert(message, "warning"), unsafe_allow_html=True)


# Export
__all__ = [
    "ADVANCED_CSS",
    "inject_css",
    "get_animated_logo",
    "get_loading_animation",
    "get_spinner",
    "get_feature_cards",
    "get_stats_display",
    "get_alert",
    "get_chat_bubble",
    "get_progress_bar",
    "get_report_card",
    "show_login_page",
    "show_loading",
    "show_success",
    "show_error",
    "show_warning",
]
