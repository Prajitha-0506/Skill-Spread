import streamlit as st
import requests


def fetch_jobs(skill, location="India"):
    """
    Fetches job listings from the Adzuna API using role/skill and location.
    Salary benchmarks and experience filters have been removed for a cleaner search.
    """
    # Accessing secrets from Streamlit Cloud dashboard
    app_id = st.secrets["api"]["adzuna_app_id"]
    app_key = st.secrets["api"]["adzuna_app_key"]

    if not app_id or not app_key:
        st.error("API Key Missing: Adzuna App ID or App Key not found in Streamlit secrets.")
        return []

    if not skill.strip():
        return []

    # 1. Construct API URL (using 'in' for India, you can change 'in' to 'us', 'gb', etc. for global)
    url = f"https://api.adzuna.com/v1/api/jobs/in/search/1"

    # 2. Parameters focused only on What (Skill/Role) and Where (Location)
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": 10,
        "what": skill,
        "where": location,
        "content-type": "application/json"
    }

    if location.strip():
        params["where"] = location

    # 3. API Request with Error Handling
    try:
        response = requests.get(url, params=params, timeout=15)

        if response.status_code == 200:
            return response.json().get("results", [])

        elif response.status_code in [403, 429]:
            st.error(f"🚨 API QUOTA/KEY ERROR: Status {response.status_code}.")
            return []

        else:
            st.error(f"Failed to fetch job listings. Status: {response.status_code}.")
            return []

    except requests.exceptions.RequestException as e:
        st.error(f"Network Error: Could not connect to Adzuna API. Details: {e}")
        return []