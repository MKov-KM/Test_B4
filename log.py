import subprocess
import streamlit as st

def authenticate(username, password):
    # You can customize this function to perform the authentication logic
    valid_username = "admin"
    valid_password = "password"

    if username == valid_username and password == valid_password:
        return True
    else:
        return False

def main():
    st.title("Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate(username, password):
            st.success("Authentication successful!")
            st.write("Redirecting to Home Page...")
            subprocess.Popen(["streamlit", "run", "home.py"])  # Open home.py in a separate process
            st.stop()  # Stop the current script execution
        else:
            st.error("Invalid username or password")

if __name__ == "__main__":
    main()
