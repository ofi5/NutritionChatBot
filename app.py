import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import nest_asyncio
import asyncio
import os
from dotenv import load_dotenv

# The Google Generative AI gRPC client expects an event loop in the current
# thread, but Streamlit's ScriptRunner thread has none ("There is no current
# event loop"). Streamlit's server runs on uvloop, which nest_asyncio cannot
# patch ("Can't patch loop of type uvloop.Loop"), so give this thread its own
# standard SelectorEventLoop before applying nest_asyncio.
try:
    _loop = asyncio.get_event_loop()
    _needs_loop = "uvloop" in type(_loop).__module__
except RuntimeError:
    _needs_loop = True
if _needs_loop:
    asyncio.set_event_loop(asyncio.SelectorEventLoop())

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load API keys from a local .env file (for local development)
load_dotenv()


def get_api_key(name):
    """Read a key from the environment (.env) first, then Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:
        return None


# Get API keys from .env (local) or Streamlit secrets (cloud)
groq_api_key = get_api_key("GROQ_API_KEY")
google_api_key = get_api_key("GOOGLE_API_KEY")

LLM_MODEL = "llama-3.3-70b-versatile"
DATA_PATH = "./data/nutritiondata.pdf"

# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Nutribot", page_icon="🥗", layout="wide")

st.markdown(
    """
    <style>
    /* Tighten the top padding and widen the usable area so text doesn't wrap */
    .block-container {
        padding-top: 1.2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 1700px;
    }
    /* Trim the internal padding of the bordered section panels */
    div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] {
        gap: 0.6rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        padding: 0.4rem 0.2rem;
    }

    /* --- Nav bar (soft, light green) --- */
    .nb-navbar {
        background: linear-gradient(120deg, #dcfce7 0%, #ecfdf5 55%, #f0fdfa 100%);
        border: 1px solid rgba(34,197,94,0.25);
        border-radius: 18px;
        padding: 22px 30px;
        color: #14532d;
        box-shadow: 0 8px 24px rgba(34,197,94,0.12);
        margin-bottom: 8px;
    }
    .nb-brand { display: flex; align-items: center; gap: 14px; }
    .nb-logo { font-size: 40px; line-height: 1; }
    .nb-title { font-size: 30px; font-weight: 800; letter-spacing: -0.5px; margin: 0; color: #166534; }
    .nb-tag { font-size: 15px; opacity: 0.9; margin: 2px 0 0 0; font-weight: 400; color: #15803d; }
    .nb-pills { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
    .nb-pill {
        background: rgba(255,255,255,0.7);
        border: 1px solid rgba(34,197,94,0.35);
        color: #15803d;
        padding: 5px 13px; border-radius: 999px;
        font-size: 13px; font-weight: 600;
    }

    /* --- Cards --- */
    .nb-card {
        background: rgba(220,252,231,0.5);
        border: 1px solid rgba(34,197,94,0.2);
        border-radius: 12px; padding: 10px 12px; margin-bottom: 8px;
    }
    .nb-card h4 { margin: 0 0 3px 0; font-size: 15px; color: #166534; }
    .nb-card p  { margin: 0; font-size: 13px; opacity: 0.85; line-height: 1.35; }

    .nb-section-title { font-size: 22px; font-weight: 800; margin: 4px 0 2px 0; color: #166534; }
    .nb-section-sub   { font-size: 14px; opacity: 0.7; margin-bottom: 14px; }

    /* Make sample-question buttons look like soft chips */
    div[data-testid="stButton"] > button {
        border-radius: 12px;
        border: 1px solid rgba(34,197,94,0.3);
        background: rgba(240,253,244,0.6);
        font-weight: 500;
        text-align: left;
        padding: 0.3rem 0.5rem;
        line-height: 1.3;
        white-space: normal;
        word-break: normal;
        overflow-wrap: break-word;
    }
    /* Keep the button label from breaking words apart letter-by-letter */
    div[data-testid="stButton"] > button p {
        word-break: normal;
        overflow-wrap: break-word;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: #4ade80;
        background: #f0fdf4;
        color: #15803d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Nav bar
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="nb-navbar">
        <div class="nb-brand">
            <div class="nb-logo">🥗</div>
            <div>
                <p class="nb-title">Nutribot</p>
                <p class="nb-tag">Your AI nutrition companion — ask about food, calories &amp; healthy habits, and build a personalized meal plan.</p>
            </div>
        </div>
        <div class="nb-pills">
            <span class="nb-pill">🧭 Learn the basics</span>
            <span class="nb-pill">💬 Ask anything</span>
            <span class="nb-pill">🍽️ Get a meal plan</span>
            <span class="nb-pill">📄 Grounded in nutrition data</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Key check
# ---------------------------------------------------------------------------
missing = [n for n, v in (("GROQ_API_KEY", groq_api_key), ("GOOGLE_API_KEY", google_api_key)) if not v]
if missing:
    st.error(
        "Missing API key(s): **" + ", ".join(missing) + "**. "
        "Add them to a `.env` file in the project root, then rerun.\n\n"
        "```\nGROQ_API_KEY=your_groq_key\nGOOGLE_API_KEY=your_google_key\n```"
    )
    st.stop()


# ---------------------------------------------------------------------------
# RAG core (cached so the index builds once)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="🔎 Reading the nutrition knowledge base…")
def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        api_key=google_api_key,
        task_type="retrieval_document",
    )
    docs = PyPDFLoader(DATA_PATH).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model=LLM_MODEL, api_key=groq_api_key)


QA_PROMPT = ChatPromptTemplate.from_template(
    """
You are Nutribot, a friendly and knowledgeable nutrition assistant.
Answer the user's question using the nutrition context below as your primary source.
If the context doesn't fully cover it, you may add general, well-established nutrition
guidance, but never invent specific numbers. Keep answers clear, practical and encouraging.

<context>
{context}
</context>

Question: {input}
"""
)


def answer_question(question):
    """Run the retrieval chain and return (answer, source_documents)."""
    vectors = get_vectorstore()
    document_chain = create_stuff_documents_chain(get_llm(), QA_PROMPT)
    retriever = vectors.as_retriever()
    chain = create_retrieval_chain(retriever, document_chain)
    response = chain.invoke({"input": question})
    return response["answer"], response.get("context", [])


MEAL_PLAN_PROMPT = ChatPromptTemplate.from_template(
    """
You are Nutribot, a nutrition and meal-planning assistant.
Using the nutrition reference context below plus sound dietary science, create a
personalized one-day meal plan for the person described.

<context>
{context}
</context>

Person's details:
{profile}

Produce a well-formatted Markdown response with:
1. A short summary line with an estimated daily calorie target and why.
2. A table of meals — **Breakfast, Lunch, Dinner, and 1–2 Snacks** — each with
   the food items and an approximate calorie count.
3. The approximate total calories for the day.
4. 3–4 concise, personalized tips aligned with their goal and preferences.

Respect all dietary preferences and avoid every listed allergen. Be encouraging.
"""
)


def generate_meal_plan(profile_text):
    vectors = get_vectorstore()
    document_chain = create_stuff_documents_chain(get_llm(), MEAL_PLAN_PROMPT)
    retriever = vectors.as_retriever()
    chain = create_retrieval_chain(retriever, document_chain)
    # 'input' feeds the retriever; the profile is injected into the prompt too.
    response = chain.invoke(
        {"input": profile_text, "profile": profile_text}
    )
    return response["answer"]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

SAMPLE_QUESTIONS = [
    "How many calories are in a banana?",
    "What are good sources of plant-based protein?",
    "How much water should I drink each day?",
    "What should I eat before a workout?",
    "Which foods are high in fiber?",
    "How can I reduce sugar in my daily diet?",
]

# Process any pending question queued from the sample buttons (runs on every
# rerun regardless of the active tab, so the answer shows up in the chat).
if st.session_state.pending_question:
    q = st.session_state.pending_question
    st.session_state.pending_question = None
    st.session_state.messages.append({"role": "user", "content": q})
    try:
        with st.spinner("Nutribot is thinking…"):
            ans, _ = answer_question(q)
    except Exception as e:
        ans = f"⚠️ Sorry, I hit an error: {e}"
    st.session_state.messages.append({"role": "assistant", "content": ans})

# ---------------------------------------------------------------------------
# Three vertical sections, side by side
# ---------------------------------------------------------------------------
col_basics, col_chat, col_plan = st.columns([1, 1.35, 1.25], gap="large")

# --- Section 1 (left): Basics & sample questions ---------------------------
with col_basics:
    with st.container(border=True):
        st.markdown('<div class="nb-section-title">🧭 Basics</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="nb-section-sub">What Nutribot can do — tap a question to try it.</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="nb-card"><h4>💬 Ask questions</h4><p>Foods, calories, nutrients and '
            "healthy habits, grounded in nutrition data.</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="nb-card"><h4>🍽️ Plan meals</h4><p>Share a few details and get a '
            "personalized one-day meal plan.</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="nb-card"><h4>📄 Trustworthy</h4><p>Answers are retrieved from a '
            "nutrition knowledge base, not made up.</p></div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="nb-section-title" style="font-size:17px;">Sample questions</div>', unsafe_allow_html=True)
        for i, q in enumerate(SAMPLE_QUESTIONS):
            if st.button(f"💡 {q}", key=f"sample_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.toast("Answering it in the chat →", icon="✅")
                st.rerun()

# --- Section 2 (center): Conversation --------------------------------------
with col_chat:
    with st.container(border=True):
        head_l, head_r = st.columns([4, 1])
        with head_l:
            st.markdown('<div class="nb-section-title">💬 Ask Nutribot</div>', unsafe_allow_html=True)
        with head_r:
            if st.button("🧹 Clear", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        st.markdown(
            '<div class="nb-section-sub">Ask about nutrition, food habits and calories.</div>',
            unsafe_allow_html=True,
        )

        # Scrollable conversation area
        chat_area = st.container(height=420)
        with chat_area:
            if not st.session_state.messages:
                st.info("👋 Hi! I'm Nutribot. Ask me anything about food, calories or healthy eating.")
            for msg in st.session_state.messages:
                avatar = "🧑" if msg["role"] == "user" else "🥗"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

        # Input stays inside the center column; clears itself after sending
        with st.form("chat_form", clear_on_submit=True):
            in_l, in_r = st.columns([5, 1])
            with in_l:
                user_question = st.text_input(
                    "Your question", placeholder="Ask a nutrition question…",
                    label_visibility="collapsed",
                )
            with in_r:
                sent = st.form_submit_button("Send", use_container_width=True)
        if sent and user_question:
            st.session_state.pending_question = user_question
            st.rerun()

# --- Section 3 (right): Meal planner ---------------------------------------
with col_plan:
    with st.container(border=True):
        st.markdown('<div class="nb-section-title">🍽️ Meal Planner</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="nb-section-sub">Tell Nutribot about yourself for a one-day plan.</div>',
            unsafe_allow_html=True,
        )

        with st.form("meal_plan_form"):
            name = st.text_input("Name (optional)", placeholder="e.g. Alex")
            fc1, fc2 = st.columns(2)
            with fc1:
                age = st.number_input("Age", min_value=1, max_value=120, value=30)
                weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.5)
            with fc2:
                gender = st.selectbox("Gender", ["Female", "Male", "Other"])
                height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)

            activity = st.selectbox(
                "Activity level",
                ["Sedentary", "Lightly active", "Moderately active", "Very active", "Athlete"],
            )
            goal = st.selectbox("Goal", ["Lose weight", "Maintain weight", "Gain muscle"])
            diet = st.selectbox(
                "Dietary preference",
                ["No preference", "Vegetarian", "Vegan", "Pescatarian", "Keto", "Halal"],
            )
            allergies = st.text_input("Allergies / foods to avoid", placeholder="e.g. peanuts, shellfish")
            notes = st.text_area("Anything else?", placeholder="e.g. I skip breakfast", height=70)

            submitted = st.form_submit_button("🍽️  Generate my meal plan", use_container_width=True)

        if submitted:
            bmi = weight / ((height / 100) ** 2)
            profile = (
                f"- Name: {name or 'N/A'}\n"
                f"- Age: {age}\n"
                f"- Gender: {gender}\n"
                f"- Weight: {weight} kg\n"
                f"- Height: {height} cm\n"
                f"- BMI: {bmi:.1f}\n"
                f"- Activity level: {activity}\n"
                f"- Goal: {goal}\n"
                f"- Dietary preference: {diet}\n"
                f"- Allergies / avoid: {allergies or 'none'}\n"
                f"- Notes: {notes or 'none'}\n"
            )
            try:
                with st.spinner("Building your personalized meal plan…"):
                    plan = generate_meal_plan(profile)
                st.success(f"Here's your plan{' , ' + name if name else ''}! (BMI: {bmi:.1f})")
                st.markdown(plan)
            except Exception as e:
                st.error(f"Error generating meal plan: {e}")
