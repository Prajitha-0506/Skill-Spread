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

    /* --- Import Google Font --- */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    /* --- Sidebar Styling (Only shows after analysis) --- */
    div[data-testid="stSidebar"] {
        background: var(--secondary-background-color); /* FIXED: Uses dark theme variable */
        border-right: 1px solid #e0e0e0;
    }

    /* --- Heading Fix (Make sure this rule is strong) --- */
    h1, h2, h3, h4, h5, h6 {
        color: #E0E0E0 !important; 
    }

    /* ADD/ENSURE THIS: Target h3 specifically in the sidebar */
    div[data-testid="stSidebar"] h3 {
        font-size: 1.5rem !important; /* Force the size on the header */
        font-weight: 600 !important;
    }


    /* --- Widget Styling (FIXED for DARK text) --- */

    /* Labels for all widgets */
    label[data-testid="stWidgetLabel"],
    div[data-testid="stSelectbox"] label {
        color: #2a7fff !important; /* Vibrant Blue label (Kept as accent) */
        font-weight: 500 !important;
    }

    /* --- This is the fix for DARK text in all inputs --- */

    /* Text inside "Your Name" box */
    div[data-testid="stTextInput"] input {
        color: #E0E0E0 !important; /* FIXED for Dark Theme */
    }
    /* Text inside "Your Skills" box */
    div[data-testid="stTextArea"] textarea {
        color: #E0E0E0 !important; /* FIXED for Dark Theme */
    }

    /* Text inside "Select a Role" box */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        color: #E0E0E0 !important; /* FIXED for Dark Theme */
    }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div {
        color: #E0E0E0 !important; /* FIXED for Dark Theme */
    }
    /* Fix for the "Select a Role" placeholder */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div[class*="placeholder"] {
         color: #B0B0B0 !important; /* Adjusted for dark background */
    }
    /* --- End of Widget Styling Fix --- */

    /* --- Fix for Text in Tabs --- */
    button[data-baseweb="tab"] div[data-testid="stMarkdownContainer"] p {
        color: #E0E0E0 !important; 
    }
    /* --- End of Tab Fix --- */


    /* --- Skill Chip Styling --- */
    .skill-chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }
    .skill-chip {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 16px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .skill-chip-success {
        background-color: #e6f7f0;
        color: #0d683f;
        border: 1px solid #b7e4cf;
    }
    .skill-chip-error {
        background-color: #fdecea;
        color: #a91e2c;
        border: 1px solid #f8c9c7;
    }
    .skill-chip-info {
        background-color: #fff3e0;
        color: #e65100;
        border: 1px solid #ffe0b2;
    }

    /* --- Job Card Styling --- */
    .job-card-custom {
        background: #1e2a38;
        border: 1px solid #444;
        border-radius: 16px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
    }
    .job-card-custom:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(42,127,255,0.25);
        border-color: #2a7fff;
    }
    .job-title {
        margin: 0 0 5px 0;
        color: #E0E0E0; /* FIXED for Dark Theme */
    }
    .job-company {
        margin: 3px 0;
        color: #B0B0B0; /* Adjusted for dark background */
        font-size: 1rem;
    }
    .job-description {
        margin: 10px 0;
        color: #B0B0B0; /* Adjusted for dark background */
        font-size: 0.95rem;
    }
    .job-skills {
        margin: 10px 0;
    }
    .skill-tag-match {
        background-color: #e7f3ff;
        color: #0063f7;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 2px;
    }
    .skill-tag-neutral {
        background-color: #f0f0f0;
        color: #555;
        padding: 4px 10px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 2px;
    }

    /* BIG APPLY NOW BUTTON - HIGH VISIBILITY */
    .btn-apply-now {
        display: inline-block !important;
        padding: 14px 36px !important;
        background: linear-gradient(135deg, #2a7fff, #1a6de6) !important;
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        text-decoration: none !important;
        border-radius: 12px !important;
        box-shadow: 0 6px 20px rgba(42,127,255,0.4) !important;
        transition: all 0.3s ease !important;
    }
    .btn-apply-now:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 30px rgba(42,127,255,0.6) !important;
    }
    .btn-apply-now:active {
        transform: translateY(-1px) !important;
    }


    /* --- Chat Styling --- */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 12px;
        background-color: var(--secondary-background-color) !important; /* FIXED: Uses dark theme variable */
        border: 1px solid #e0e0e0;
    }

    /* --- DEFINITIVE FIXED CHAT INPUT POSITION --- */
    /* Targets the footer element where Streamlit places st.chat_input */
    footer {
        position: fixed !important;
        bottom: 0 !important;
        left: 0;
        /* Calculate width: 100% of viewport minus the sidebar width (300px is Streamlit's default) */
        width: calc(100% - 300px) !important; 
        z-index: 9999; 
        background-color: var(--secondary-background-color); 
        padding-bottom: 10px;
    }

    /* Adjust padding to account for the sidebar offset */
    section[data-testid="stSidebar"] + div > footer {
        left: 300px; /* Aligns the footer to the main content area */
        width: calc(100% - 300px) !important;
    }

    /* Target the chat input itself for internal padding/styling */
    div[data-testid="stChatInput"] {
        background-color: var(--background-color); 
        padding: 10px 1rem 10px 1rem;
        box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.2);
    }
    /* --- End DEFINITIVE FIXED CHAT INPUT POSITION --- */

    /* --- ULTIMATE FLICKER FIX --- */
    /* This targets the main content body and forces it to hide immediately 
       when Streamlit adds the 'stApp-loading' class (during a page transition). */
    .stApp.stApp-loading > div[data-testid="stAppViewContainer"] > .main {
        opacity: 0 !important;
        visibility: hidden !important;
    }

    /* Ensure the main page area always uses your theme background color */
    div[data-testid="stAppViewContainer"] {
        background-color: var(--background-color) !important;
    }
    /* --- End ULTIMATE FLICKER FIX --- */

    /* Final successful fix concept: High specificity and deep nesting */
    div[data-testid="stSidebar"] div[data-testid*="stBlock"] > div > div > div > p {
        font-size: 1.25rem !important;
        /* Force Poppins for text, but allow System Emojis for the icons */
        font-family: 'Poppins', "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji", sans-serif !important;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    """
    st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)


# --- Helper, GenAI, and Skill Processing Functions ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None


def create_prompt(skills, job_description):
    return f"""
    You are an expert Executive Resume Writer. 
    USER SKILLS: {skills}
    JOB DESCRIPTION: {job_description}

    TASK: Generate 3-5 high-impact resume bullet points.

    META-PROMPTING INSTRUCTIONS:
    1. First, identify the 3 most critical keywords or requirements from the Job Description.
    2. Map the user's skills to these specific requirements.
    3. For each bullet, use the Google XYZ formula: 'Accomplished [X] as measured by [Y], by doing [Z]'.
    4. Start each bullet with a powerful action verb (e.g., 'Spearheaded', 'Optimized', 'Architected').
    5. Be concise and do not invent experience.

    OUTPUT: Provide only the bullet points in a clean markdown list.
    """


def generate_response(prompt):
    try:
        # Use the 2026 stable model ID
        AI_MODEL_NAME = 'gemini-2.5-flash'

        genai.configure(api_key=st.secrets["api"]["gemini_api_key"])

        # Rename this variable to 'career_ai' to avoid clashing with your .pkl 'model'
        career_ai = genai.GenerativeModel(AI_MODEL_NAME)

        # Start generating content
        response_stream = career_ai.generate_content(prompt, stream=True)
        return response_stream

    except Exception as e:
        st.error(f"AI Connection Error: {e}. Please verify your API key and model name.")
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
    data = pd.read_excel("skillspread_dataset.xlsx")
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

        # User input for location
        job_loc = st.text_input("📍 Preferred Location (Leave blank for all locations)", value="")

        # Dynamic location text helper
        location_text = f" in **{job_loc.strip()}**" if job_loc.strip() else ""

        inner_tab1, inner_tab2, inner_tab3 = st.tabs(["🎯 My Target Role", "🔮 Predicted Role", "🛠 By My Skills"])

        with inner_tab1:
            target_role = st.session_state.get("job_role")
            if target_role and target_role != 'Select a Role':
                st.info(f"Showing jobs for: **{target_role}**{location_text}")
                with st.spinner(f"Searching for {target_role}..."):
                    jobs = fetch_jobs(target_role, location=job_loc)
                    display_jobs(jobs[:5], st.session_state.get("original_skills", []))
            else:
                st.warning("Please select a target role on the home page first.")

        with inner_tab2:
            if st.session_state.valid_user_skills:
                # 1. Prepare skills for the ML Model
                skills_text = ", ".join(list(set(st.session_state.valid_user_skills)))
                skills_vector = vectorizer.transform([skills_text])

                # 2. Get Top Predicted Role from your RandomForest model
                probabilities = model.predict_proba(skills_vector)
                top_role_index = np.argmax(probabilities[0])
                top_role = job_roles[top_role_index]

                st.info(f"Top predicted role based on your profile: **{top_role}**{location_text}")

                with st.spinner(f"Fetching jobs for {top_role}..."):
                    jobs = fetch_jobs(top_role, location=job_loc)
                    display_jobs(jobs[:5], st.session_state.get("original_skills", []))
            else:
                st.warning("Please enter more valid skills on the home page for a prediction.")

        with inner_tab3:
            all_user_skills = st.session_state.get("original_skills", [])
            if all_user_skills:
                # Broader search: use top 3 skills only to avoid empty results
                search_query = ", ".join(all_user_skills[:3])

                st.info(f"Showing jobs matching skills: **{search_query}**{location_text}")

                with st.spinner("Fetching matches..."):
                    jobs = fetch_jobs(search_query, location=job_loc)

                    # Fallback if query is too specific
                    if not jobs and len(all_user_skills) > 1:
                        jobs = fetch_jobs(all_user_skills[0], location=job_loc)

                    display_jobs(jobs[:5], all_user_skills)
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

                    response_stream = generate_response(prompt)

                    # 💡 CORRECT FIX: Concatenate the stream immediately into a string
                    full_response_text = ""
                    for chunk in response_stream:
                        if chunk.candidates:
                            # Safely access text from the first candidate's content parts
                            # Note: This is equivalent to chunk.text, but we rely on st.write_stream
                            # for better error handling/display in the chat, so let's stick to simple iteration here.
                            try:
                                full_response_text += chunk.text
                            except ValueError:
                                # Catch the specific ValueError and skip the chunk
                                pass

                                # Alternative and safer: Use the original Streamlit fix concept but robustly:
                    # full_response_text = "".join(chunk.text for chunk in response_stream if chunk.text)
                    # The issue is stream consumption. The best fix is to use the `try...except` block above.

                    # Store the final text
                    st.session_state.generated_points_text = full_response_text

                    # CRITICAL: SET A FLAG to indicate generation is complete and ready to display
                    st.session_state.generated_points_display = True

                    # Change the display condition to the new, corrected flag
        if st.session_state.get("generated_points_display", False):
            st.markdown("#### ✅ Your AI-Generated Resume Points")
            # Display the stored text
            st.markdown(st.session_state.generated_points_text, unsafe_allow_html=True)



    # --- Page 4: AI Career Chat ---
    elif st.session_state.get("page") == "🤖 AI Career Chat":
        st.markdown("# 🤖 AI Career Chat")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # Display history
        for msg in st.session_state.chat_messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # Check for new user input to process
        if st.session_state.chat_messages and st.session_state.chat_messages[-1]["role"] == "user":

            # Map history to Gemini format (user -> user, assistant -> model)
            history_context = []
            for m in st.session_state.chat_messages[:-1]:
                role = "model" if m["role"] == "assistant" else "user"
                history_context.append({"role": role, "parts": [m["content"]]})

            user_prompt = st.session_state.chat_messages[-1]["content"]
            missing_core = ", ".join(st.session_state.get("missing_core_skills", []))
            job_role = st.session_state.get("job_role", "Not specified")

            system_instruction = f"Context: You are a career mentor for {job_role}. User lacks: {missing_core}."

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""

                try:
                    genai.configure(api_key=st.secrets["api"]["gemini_api_key"])
                    # Using the 2026 standard model ID
                    career_ai = genai.GenerativeModel('gemini-2.5-flash')

                    # Correctly starting the chat session with history
                    chat_session = career_ai.start_chat(history=history_context)

                    response_stream = chat_session.send_message(
                        f"{system_instruction}\n\nUser: {user_prompt}",
                        stream=True
                    )

                    for chunk in response_stream:
                        if chunk.text:
                            full_response += chunk.text
                            placeholder.markdown(full_response + "▌")

                    placeholder.markdown(full_response)

                except Exception as e:
                    placeholder.error(f"Chat Error: {e}. Try changing model to 'gemini-2.0-flash'.")
                    full_response = "I'm having trouble connecting to my brain. Try again in a second!"

                st.session_state.chat_messages.append({"role": "assistant", "content": full_response})

        if prompt := st.chat_input("Ask about roadmaps or interview tips..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            st.rerun()