import streamlit as st
import pandas as pd
import base64
import joblib
import numpy as np
import json
from thefuzz import fuzz, process
import io
import html
from code import fetch_jobs
import google.generativeai as genai
import random
import re
import plotly.graph_objects as go
import plotly.graph_objects as go
from secrets_helper import load_secrets

# --- Page and State Setup ---
st.set_page_config(
    page_title="SkillSpread",
    page_icon="🚀",
    layout="wide"
    # REMOVED: theme="light" - this was causing the error
)


# --- CSS is embedded directly ---
def load_css():
    css_styles = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    /* Global Font & Smooth Scroll */
    body, .stApp {
        font-family: 'Poppins', sans-serif !important;
        scroll-behavior: smooth;
    }

    /* Main Container Polish */
    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Headings - Strong Hierarchy */
    h1 { font-size: 2.8rem !important; font-weight: 700 !important; margin-bottom: 1rem; }
    h2 { font-size: 2.2rem !important; font-weight: 600 !important; }
    h3 { font-size: 1.7rem !important; font-weight: 600 !important; color: #E0E0E0 !important; }
    h4 { font-size: 1.4rem !important; font-weight: 600 !important; }

    /* Sidebar - Modern Look */
    div[data-testid="stSidebar"] {
        background: #0f1b2a;
        border-right: 1px solid #2a3b55;
        box-shadow: 4px 0 20px rgba(0,0,0,0.3);
    }
    .css-1d391kg { padding: 1.5rem 1rem; } /* Sidebar header spacing */
    .sidebar .sidebar-content { padding-top: 1rem; }

    /* Input Form - Floating Labels & Better Look */
    div[data-testid="stTextInput"] > div > div > input,
    div[data-testid="stTextArea"] > div > div > textarea {
        background-color: #1e2a38 !important;
        border: 1.5px solid #2a7fff !important;
        border-radius: 12px !important;
        padding: 14px !important;
        transition: all 0.3s ease;
    }
    div[data-testid="stTextInput"] > div > div > input:focus,
    div[data-testid="stTextArea"] > div > div > textarea:focus {
        box-shadow: 0 0 0 3px rgba(42, 127, 255, 0.3) !important;
        transform: translateY(-2px);
    }

    /* Selectbox - Modern Dropdown */
    div[data-baseweb="select"] > div {
        background-color: #1e2a38 !important;
        border: 1.5px solid #2a7fff !important;
        border-radius: 12px !important;
        color: #E0E0E0 !important;
    }

    /* Primary Button - Glow Effect */
    .stButton > button {
        background: linear-gradient(135deg, #2a7fff, #1a6de6) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 6px 20px rgba(42,127,255,0.4) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 30px rgba(42,127,255,0.6) !important;
    }

    /* Skill Chips - Elevated */
    .skill-chip {
        padding: 8px 16px !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        transition: transform 0.2s ease;
    }
    .skill-chip:hover {
        transform: translateY(-2px);
    }

    /* Job Cards - Premium Feel */
    .job-card-custom {
        background: linear-gradient(145deg, #1e2a38, #16222f) !important;
        border: 1.5px solid #2a7fff !important;
        border-radius: 20px !important;
        padding: 24px !important;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(42,127,255,0.15) !important;
        transition: all 0.4s ease;
    }
    .job-card-custom:hover {
        transform: translateY(-10px) !important;
        box-shadow: 0 20px 40px rgba(42,127,255,0.3) !important;
        border-color: #5a9fff !important;
    }

    /* Apply Button - Hero Level */
    .btn-apply-now {
        background: linear-gradient(135deg, #2a7fff, #1a6de6) !important;
        padding: 16px 40px !important;
        border-radius: 16px !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        box-shadow: 0 8px 25px rgba(42,127,255,0.5) !important;
        transition: all 0.3s ease !important;
    }
    .btn-apply-now:hover {
        transform: translateY(-6px) !important;
        box-shadow: 0 16px 40px rgba(42,127,255,0.7) !important;
    }

    /* Donut Chart - Gradient Ring */
    .js-plotly-plot .plotly .main-svg {
        filter: drop-shadow(0 0 20px rgba(42,127,255,0.3));
    }

    /* Chat Messages - Card Style */
    [data-testid="stChatMessage"] {
        border-radius: 16px !important;
        padding: 16px !important;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        border: 1px solid #2a7fff33;
    }

    /* Fixed Chat Input at Bottom */
    footer { visibility: hidden; }
    div[data-testid="stChatInput"] {
        position: fixed;
        bottom: 0;
        left: 300px;
        right: 0;
        background: #0f1b2a;
        padding: 1rem;
        border-top: 1px solid #2a7fff;
        z-index: 9999;
        box-shadow: 0 -5px 20px rgba(0,0,0,0.4);
    }

    /* Responsive */
    @media (max-width: 768px) {
        div[data-testid="stChatInput"] { left: 0; }
        .main .block-container { padding: 1rem; }
    }
    </style>
    """
    st.markdown("""
        <div style="text-align: center; padding: 3rem 0 2rem;">
            <h1 style="font-size: 3.5rem; background: linear-gradient(90deg, #2a7fff, #00d4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                SkillSpread Pro
            </h1>
            <p style="font-size: 1.3rem; color: #B0B0B0; margin-top: 1rem;">
                Get hired faster with AI-powered skill matching
            </p>
        </div>
    """, unsafe_allow_html=True)


# --- Helper, GenAI, and Skill Processing Functions ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None


def create_prompt(skills, job_description):
    # Modified prompt for conciseness
    return f"""As an expert career coach, based on these skills: {skills} and this job description: {job_description}, write 3-5 powerful, highly concise resume bullet points starting with action verbs. Focus on matching the user's skills to the job requirements without inventing skills. Format them as a bulleted list."""


def generate_response(prompt):
    try:
        # FIX: Access key via the "api" section, not a non-existent "gemini" section.
        genai.configure(api_key=st.secrets["api"]["gemini_api_key"])
        model = genai.GenerativeModel('models/gemini-pro-latest')

        # CRITICAL FIX: Use streaming to reduce perceived latency
        response_stream = model.generate_content(prompt, stream=True)

        # Return the stream object
        return response_stream

    except Exception as e:
        # This catch is good, but the key error makes the traceback misleading
        st.error(f"An error occurred with the AI model: {e}")
        return None


# Helper function to display skills as styled chips
def display_skill_chips(skills, skill_type):
    if not skills:
        return

    type_map = {
        "success": "Skills you have:",
        "error": "Core skills to learn:",
        "info": "Optional skills to explore:"
    }

    st.markdown(f"{type_map.get(skill_type, 'Skills:')}")

    chips_html = "".join([f'<span class="skill-chip skill-chip-{skill_type}">{skill}</span>' for skill in skills])
    st.markdown(f'<div class="skill-chip-container">{chips_html}</div>', unsafe_allow_html=True)


def fuzzy_match(skill, skill_list, threshold=65):
    if not skill or not skill_list:
        return None
    match, score = process.extractOne(skill, skill_list, scorer=fuzz.token_set_ratio)
    return match if score >= threshold else None


def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[/\-_+]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


enhanced_skill_normalizer = {
    "powerbi": "power bi", "power-bi": "power bi", "ml": "machine learning", "ai": "artificial intelligence",
    "sql": "sql", "py": "python", "js": "javascript", "nodejs": "node.js", "reactjs": "react",
    "angularjs": "angular", "css3": "css", "html5": "html", "aws": "amazon web services", "gcp": "google cloud platform"
}


def enhanced_normalize_skill(skill):
    clean_skill = skill.lower().strip()
    return enhanced_skill_normalizer.get(clean_skill, clean_skill)


# Initialize session state variables
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
if "generated_points" not in st.session_state:
    st.session_state["generated_points"] = None
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {"role": "assistant",
         "content": "Hi! I'm your AI Career Mentor 🤖. Ask me about learning roadmaps, skill gaps, or project ideas!"}
    ]


def display_jobs(jobs, user_skills):
    if not user_skills:
        st.info("No skills provided to match against job descriptions.")
        return

    for job in jobs:
        # --- Skill matching ---
        raw_text = (job.get("title", "") + " " + job.get("description", ""))
        text_to_search = clean_text(raw_text)
        matched_skills_in_job = list(dict.fromkeys([
            original_skill for original_skill in user_skills
            if enhanced_normalize_skill(original_skill) in text_to_search
        ]))

        if matched_skills_in_job:
            skills_html = "".join(
                f'<span class="skill-tag-match">{skill}</span>' for skill in matched_skills_in_job
            )
        else:
            skills_html = '<span class="skill-tag-neutral">No direct skill matches found</span>'

        # --- Basic job info (escaped for safety, BUT NOT THE BUTTON) ---
        company   = html.escape(job.get("company", {}).get("display_name", "Unknown Company"))
        location  = html.escape(job.get("location", {}).get("display_name", "Remote"))
        title     = html.escape(job.get("title", "No Title"))
        description = html.escape(job.get("description", "No description available")[:240] + "...")

        # --- URL handling ---
        adzuna_url = job.get("redirect_url", "")
        direct_url = job.get("adref", None)

        # Safe fallback URL
        safe_url = adzuna_url if adzuna_url.startswith("http") else f"https://{adzuna_url}" if adzuna_url else "#"

        # Decide final URL and button text
        final_url   = safe_url
        button_text = "Apply Now"

        if direct_url and isinstance(direct_url, str) and len(direct_url) > 10:
            candidate = direct_url if direct_url.startswith("http") else f"https://{direct_url}"
            lower = candidate.lower()
            if any(dom in lower for dom in ["linkedin", "naukri", "indeed", "greenhouse", "lever", "workday"]):
                final_url = candidate
                if "linkedin" in lower:
                    button_text = "Apply Now on LinkedIn"
                elif "naukri" in lower:
                    button_text = "Apply Now on Naukri"
                else:
                    button_text = "Apply Now (Direct Link)"

        # --- Build the fallback link (only shown when we are using the direct link) ---
        fallback_html = ""
        if final_url != safe_url and safe_url != "#":
            fallback_html = f'''
            <div style="text-align: center; margin-top: 8px; font-size: 0.8em; opacity: 0.7;">
                <a href="{safe_url}" target="_blank" rel="noopener" style="color: #88c0ff; text-decoration: underline;">
                    Not working? Try safe link
                </a>
            </div>
            '''

        # --- CRITICAL FIX: Don't escape the button HTML ---
        # Create the button HTML separately without escaping
        button_html = f'''
        <div style="margin: 25px 0; text-align: center;">
            <a href="{final_url}" target="_blank" rel="noopener noreferrer" class="btn-apply-now">
                {button_text}
            </a>
        </div>
        '''

        # --- Final card HTML (button is NOT escaped) ---
        job_card = f"""
        <div class="job-card-custom">
            <h4 class="job-title">{title}</h4>
            <p class="job-company"><strong>{company}</strong> • 📍 {location}</p>
            <p class="job-description">{description}</p>
            <div class="job-skills"><strong>Matching Skills:</strong> {skills_html}</div>

            {button_html}

            {fallback_html}
        </div>
        <br>
        """

        st.markdown(job_card, unsafe_allow_html=True)


# --- Data & Model Loading ---
@st.cache_data
def load_data():
    data = pd.read_csv("skillspread_dataset.csv")
    job_roles_dataset = pd.read_csv("realistic_unique_job_roles_dataset.csv")
    all_skills = set(job_roles_dataset["Skills"])
    roles_from_data = sorted(data['job_role'].str.title().unique())
    return data, all_skills, roles_from_data


@st.cache_resource
def load_models():
    model = joblib.load("job_role_predictor.pkl")
    vectorizer = joblib.load("skill_vectorizer.pkl")
    return model, vectorizer, model.classes_


load_css()

data, all_skills, roles_from_data = load_data()
model, vectorizer, job_roles = load_models()

# --- UI: Input Form (Main Page) ---
if not st.session_state.get("analysis_done", False):
    try:
        # Get the base64 string for the image
        img_base64 = get_base64_of_bin_file("image.png")

        # Inject custom HTML and CSS to center the image.
        st.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 25px;">
            <img src="data:image/png;base64,{img_base64}" width="300"
            style="display: block; margin: 0 auto;"/>
            </div>
    """,
            unsafe_allow_html=True
        )

    except FileNotFoundError:
        st.error("Error: 'image.png' not found. Please make sure it's in the project folder.")
    except Exception as e:
        # Handle case where base64 conversion fails (e.g., file not found)
        st.error(f"Error displaying logo: {e}")

    with st.container():
        st.markdown("👋 Welcome! Let's get started by analyzing your skills against your desired job role.")
        with st.form("input_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Your Name", placeholder="e.g., Alex Doe")
                skills = st.text_area('Your Skills',
                                      placeholder='Enter skills separated by a comma: Python, SQL, Power BI...')
            with col2:
                job_role = st.selectbox("Your Target Job Role", options=['Select a Role'] + roles_from_data)

            analyze_button = st.form_submit_button('Analyze My Skills', use_container_width=True, type="primary")

            if analyze_button:
                if job_role == "Select a Role" or not skills.strip():
                    st.error("Please select a job role and enter at least one skill.")
                else:
                    st.session_state.analysis_done = True
                    st.session_state.name = name
                    st.session_state.job_role = job_role
                    st.session_state.original_skills = [s.strip() for s in skills.split(",")]
                    user_skills_cleaned = [enhanced_normalize_skill(s.strip()) for s in skills.split(",")]
                    st.session_state.user_skills_cleaned = user_skills_cleaned
                    index = data[data["job_role"] == job_role.lower()].index[0]
                    st.session_state.index = index
                    core_skills = [enhanced_normalize_skill(s.strip()) for s in data.iat[index, 1].split(",")]
                    st.session_state.core_skills = core_skills
                    optional_skills = [enhanced_normalize_skill(s.strip()) for s in data.iat[index, 2].split(",")]
                    st.session_state.optional_skills = optional_skills
                    st.session_state.matching_core_skills = list(
                        set(m for s in user_skills_cleaned if (m := fuzzy_match(s, core_skills))))
                    st.session_state.missing_core_skills = [s for s in core_skills if
                                                            s not in st.session_state.matching_core_skills]
                    st.session_state.matching_optional_skills = list(
                        set(m for s in user_skills_cleaned if (m := fuzzy_match(s, optional_skills))))
                    st.session_state.missing_optional_skills = [s for s in optional_skills if
                                                                s not in st.session_state.matching_optional_skills]
                    st.session_state.all_matched_skills = list(
                        set(st.session_state.matching_core_skills + st.session_state.matching_optional_skills))
                    st.session_state.valid_user_skills = [s for s in user_skills_cleaned if
                                                          fuzzy_match(s, all_skills, threshold=85)]
                    st.session_state.generated_points = None
                    st.rerun()

# --- Main App Logic (After Analysis) ---
if st.session_state.get("analysis_done", False):

    # --- Sidebar Controls ---
    img_base64_sidebar = get_base64_of_bin_file("logo.png")
    if img_base64_sidebar:
        st.sidebar.image(f"data:image/png;base64,{img_base64_sidebar}", width=150)

    st.sidebar.header(f"Welcome, {st.session_state.name or 'User'}!")

    # Define the list of options clearly
    NAV_OPTIONS = ["📊 Analysis Report", "🔍 Find Jobs", "📝 Resume Points Builder", "🤖 AI Career Chat"]
    DEFAULT_PAGE = NAV_OPTIONS[0]  # Use the first element as the default

    current_page = st.session_state.get("page", DEFAULT_PAGE)

    # If the stored page is somehow invalid, reset it to the default
    if current_page not in NAV_OPTIONS:
        current_page = DEFAULT_PAGE

    selected_page = st.sidebar.radio(
        "Navigation",
        NAV_OPTIONS,
        index=NAV_OPTIONS.index(current_page)
    )
    st.session_state["page"] = selected_page

    st.sidebar.divider()
    if st.sidebar.button("⬅ Start New Analysis"):
        st.session_state.clear()
        st.rerun()
    st.sidebar.divider()
    st.sidebar.info(f"*Target Role:*\n\n{st.session_state.job_role}")

    # --- Page 1: Analysis Report ---
    if st.session_state.get("page") == "📊 Analysis Report":
        st.markdown(f"# Your Personalized Report for *{st.session_state.job_role}*")
        st.markdown("---")

        st.subheader("🚀 Your Skill Profile")

        # FIX: Define the divisor variables safely and explicitly to avoid NameError/ZeroDivisionError.
        total_core_skills = len(st.session_state.core_skills)
        actual_core_divisor = total_core_skills if total_core_skills > 0 else 1

        total_optional_skills = len(st.session_state.optional_skills)
        actual_optional_divisor = total_optional_skills if total_optional_skills > 0 else 1

        # Calculate weights using the safe divisors
        core_weight = (len(st.session_state.matching_core_skills) / actual_core_divisor) * 80
        optional_weight = (len(st.session_state.matching_optional_skills) / actual_optional_divisor) * 20

        per = int(core_weight + optional_weight)
        per = min(per, 100)

        col1, col2 = st.columns([1, 2])
        with col1:
            fig = go.Figure(go.Pie(
                values=[per, 100 - per], hole=0.6, marker_colors=['#00C49A', '#E0E0E0'],
                textinfo='none', hoverinfo='none', direction='clockwise', sort=False
            ))
            fig.update_layout(
                showlegend=False, height=200, width=200, margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                annotations=[dict(text=f'<b>{per}%</b>', x=0.5, y=0.5, font_size=22, showarrow=False,
                                  font=dict(color='#E0E0E0'))]
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric(label="Core Skills Matched",
                      value=f"{len(st.session_state.matching_core_skills)} / {len(st.session_state.core_skills)}")
            st.metric(label="Optional Skills Matched",
                      value=f"{len(st.session_state.matching_optional_skills)} / {len(st.session_state.optional_skills)}")

        st.markdown("---")
        st.subheader("🔮 Top 3 Predicted Roles Based on Your Skills")
        if st.session_state.valid_user_skills:
            valid_skills_text = ", ".join(list(set(st.session_state.valid_user_skills)))
            X_input = vectorizer.transform([valid_skills_text])
            probas = model.predict_proba(X_input)[0]
            top3_idx = np.argsort(probas)[::-1][:3]
            for rank, idx in enumerate(top3_idx, start=1):
                role = job_roles[idx]
                st.write(f"{rank}. {role}")
        else:
            st.warning("⚠ Enter more skills for a role prediction.")

        st.markdown("---")
        st.subheader("🧠 Core Skills Analysis")

        display_skill_chips(st.session_state.matching_core_skills, "success")
        display_skill_chips(st.session_state.missing_core_skills, "error")

        if not st.session_state.matching_core_skills and not st.session_state.missing_core_skills:
            st.warning("No core skills listed for this role.")
        elif not st.session_state.missing_core_skills:
            st.success("🎯 You have all the core skills for this role. Great job!", icon="🎉")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.missing_core_skills:
            with st.expander("📚 Learning Resources for Missing Core Skills", expanded=True):
                resources_raw = data.iat[st.session_state.index, 3]
                try:
                    decoded_once = json.loads(resources_raw)
                    resources_dict = json.loads(decoded_once) if isinstance(decoded_once, str) else decoded_once
                except Exception:
                    resources_dict = {}

                core_table_data = [{"Skill": s.title(), "Free Resources": resources_dict.get(s, {}).get("free", "N/A"),
                                    "Paid Courses": resources_dict.get(s, {}).get("paid", "N/A")} for s in
                                   st.session_state.missing_core_skills]
                if core_table_data:
                    st.table(core_table_data)

        st.markdown("---")
        st.subheader("✨ Optional Skills Analysis")

        display_skill_chips(st.session_state.matching_optional_skills, "success")
        display_skill_chips(st.session_state.missing_optional_skills, "info")

        if not st.session_state.matching_optional_skills and not st.session_state.missing_optional_skills:
            st.info("No optional skills listed for this role.")
        elif not st.session_state.missing_optional_skills:
            st.info("You've matched all listed optional skills!", icon="⭐")

    # --- Page 2: Find Jobs ---
    elif st.session_state.get("page") == "🔍 Find Jobs":
        st.markdown("# 🔍 Find Relevant Jobs")
        inner_tab1, inner_tab2 = st.tabs(["🔮 By Predicted Role", "🛠 By Your Skills"])
        with inner_tab1:
            if st.session_state.valid_user_skills:
                skills_text = ", ".join(list(set(st.session_state.valid_user_skills)))
                skills_vector = vectorizer.transform([skills_text])
                probabilities = model.predict_proba(skills_vector)
                top_role_index = np.argmax(probabilities[0])
                top_role = job_roles[top_role_index]

                st.info(f"Showing jobs for your top predicted role: *{top_role}*")
                with st.spinner(f"Fetching jobs for {top_role}..."):
                    # Original call without experience argument:
                    jobs = fetch_jobs(top_role)
                    display_jobs(jobs[:5], st.session_state.get("original_skills", []))
            else:
                st.caption("Enter valid skills for a role prediction.")
        with inner_tab2:
            if st.session_state.user_skills_cleaned:
                all_skills = st.session_state.original_skills
                query_skills = " ".join(all_skills)
                # --- END FIX ---

                st.info(f"Showing jobs for all user-entered skills: *{query_skills}*")
                with st.spinner(f"Fetching jobs for your skills..."):
                    jobs = fetch_jobs(query_skills)
                    display_jobs(jobs[:5], st.session_state.get("original_skills", []))
            else:
                st.caption("Enter skills to see relevant jobs.")
    # --- Page 3: Resume Builder ---
    elif st.session_state.get("page") == "📝 Resume Points Builder":
        with st.spinner("Preparing AI Builder interface..."):
            st.markdown("# 📝 AI Resume Builder")
            st.info("Paste a job description below to get tailored resume bullet points based on your skills.",
                    icon="📄")
        job_desc = st.text_area("Paste Job Description Here", height=200, label_visibility="collapsed")
        if st.button("Generate Resume Points", use_container_width=True):
            if not job_desc.strip():
                st.warning("Please paste a job description.")
            else:
                with st.spinner("✨ Your AI career coach is writing..."):
                    user_skills_str = ", ".join(st.session_state.all_matched_skills)
                    prompt = create_prompt(user_skills_str, job_desc)

                    # Store generated points temporarily
                    st.session_state.generated_points = generate_response(prompt)

        if st.session_state.generated_points:
            st.markdown("#### ✅ Your AI-Generated Resume Points")

            # Streaming the response for speed (if generate_response returns a stream)
            full_response = ""
            assistant_response_box = st.empty()

            # Assuming st.session_state.generated_points holds the stream object
            if hasattr(st.session_state.generated_points, '_iter_'):
                for chunk in st.session_state.generated_points:
                    if chunk.text:  # <--- CRITICAL FIX ADDED HERE
                        full_response += chunk.text
                        assistant_response_box.markdown(full_response)
            else:
                # Fallback if st.session_state.generated_points is just text
                st.markdown(st.session_state.generated_points, unsafe_allow_html=True)


    # --- Page 4: AI Career Chat ---
    elif st.session_state.get("page") == "🤖 AI Career Chat":
        st.markdown("# 🤖 AI Career Chat")

        for msg in st.session_state.chat_messages:
            st.chat_message(msg["role"]).write(msg["content"])

# --- FINAL CHAT INPUT AND RESPONSE PROCESSING ---

# This block executes if analysis is done AND we are on the Chat page.
if st.session_state.get("analysis_done", False) and st.session_state.get("page") == "🤖 AI Career Chat":

    # 1. Check if an AI response is pending from a previous run
    if st.session_state.chat_messages and st.session_state.chat_messages[-1]["role"] == "user":

        # Get the user prompt (which is the last message)
        current_prompt = st.session_state.chat_messages[-1]["content"]

        # Temporary placeholder logic to show the "Thinking" state
        with st.empty():
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Define context for the AI model (optimized for conciseness)
                    missing_core = ", ".join(st.session_state.get("missing_core_skills", []))
                    job_role = st.session_state.get("job_role", "Not specified")
                    context = f"""You are an AI Career Mentor for the role of {job_role}. The user's missing core skills are: {missing_core}. Please answer their question: {current_prompt}"""

                    response_stream = generate_response(context)  # Get the streaming object

                    full_response = ""
                    assistant_response_box = st.empty()  # Placeholder for streaming text

                    for chunk in response_stream:
                        if chunk.text:
                            full_response += chunk.text
                            assistant_response_box.markdown(full_response)

        # Append the final response to history
        st.session_state.chat_messages.append({"role": "assistant", "content": full_response})

        # Force a clean render to show the completed message in the main history
        st.rerun()

    # 2. Define the st.chat_input ONCE at the end of the script to anchor it to the bottom.
    if prompt := st.chat_input("Ask for a learning roadmap...", key="final_chat_input"):
        # Append user prompt to history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # Trigger the response processing logic above on the next rerun
        st.rerun()