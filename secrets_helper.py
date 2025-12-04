import os
import json
import toml

def load_secrets():
    """
    Loads secrets from the STREAMLIT_SECRETS_TOML environment variable
    if the standard secrets file is not found (common in cloud environments).
    """
    # 1. Try Streamlit's default method first
    try:
        import streamlit as st
        # Attempt to access a secret to force loading
        _ = st.secrets["api"]["adzuna_app_id"]
        return # Success, Streamlit found the file/variables
    except Exception:
        pass # StreamlitSecretNotFoundError or KeyError occurred

    # 2. If default failed, load from environment variable
    secrets_content = os.environ.get("STREAMLIT_SECRETS_TOML")
    if secrets_content:
        # Streamlit doesn't expose a method to load strings, so we simulate the file
        
        # Create the .streamlit directory if it doesn't exist
        os.makedirs(".streamlit", exist_ok=True)
        
        # Write the content to the expected file path
        with open(".streamlit/secrets.toml", "w") as f:
            f.write(secrets_content)
        
        # Rerun to force Streamlit to load the new file
        # This fix typically works without needing a manual st.rerun on first load
        # But we ensure the file is created for st.secrets to find it.

# Execute the helper function when imported
load_secrets()