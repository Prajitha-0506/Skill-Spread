import streamlit as st
import requests


def fetch_jobs(skill, min_experience_years=0, location="India"):
    """
    Fetches job listings from the Adzuna API, filtering by skill and minimum experience.
    The min_experience_years parameter uses a salary benchmark proxy.
    """
    app_id = st.secrets["api"]["adzuna_app_id"]
    app_key = st.secrets["api"]["adzuna_app_key"]

    if not app_id or not app_key:
        # Check for missing credentials first
        st.error("API Key Missing: Adzuna App ID or App Key not found in Streamlit secrets.")
        return []
    if not skill.strip():
        return []

    # --- TIERED SALARY BENCHMARK (in INR) ---
    # Using realistic ANNUAL minimum salaries as proxy for experience level.
    if min_experience_years >= 5:
        annual_salary_min = 1000000  # 10.0 LPA for Senior/Expert roles
    elif min_experience_years >= 3:
        annual_salary_min = 600000  # 6.0 LPA for Mid-level roles (3-5 YOE)
    elif min_experience_years >= 1:
        annual_salary_min = 400000  # 4.0 LPA for Junior roles (1-3 YOE)
    else:
        annual_salary_min = 250000  # 2.5 LPA for Entry-Level/Fresher (0 YOE)

    # 2. Construct API URL and Parameters
    url = f"https://api.adzuna.com/v1/api/jobs/in/search/1"
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": 10,
        "what": skill,
        "where": location,
        "content-type": "application/json",

        # USE THE REALISTIC BENCHMARK
        "salary_min": annual_salary_min
    }

    # Clean up params to remove any potentially passed None values
    params = {k: v for k, v in params.items() if v is not None}

    # 3. API Request with Error Handling
    try:
        response = requests.get(url, params=params, timeout=15)  # Added timeout for stability

        if response.status_code == 200:
            return response.json().get("results", [])

        elif response.status_code in [403, 429]:
            st.error(
                f"🚨 API QUOTA/KEY ERROR: Adzuna returned status {response.status_code} (Forbidden/Rate Limit Exceeded).")
            return []

        else:
            st.error(f"Failed to fetch job listings. Adzuna returned status {response.status_code}.")
            return []

    except requests.exceptions.RequestException as e:
        st.error(f"Network/Timeout Error: Could not connect to Adzuna API. Details: {e}")
        return []