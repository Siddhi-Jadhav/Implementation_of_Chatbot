import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("/Users/Lenovo/OneDrive/Documents/AICTE internship/intent.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
# Train NLP Model
vectorizer = TfidfVectorizer()
clf = LogisticRegression(max_iter=1000)

tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

X = vectorizer.fit_transform(patterns)
y = tags
clf.fit(X, y)

def chatbot_response(user_input):
    X_input = vectorizer.transform([user_input])
    tag = clf.predict(X_input)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that. Can you rephrase?"

# Initialize session state for appointments
if 'appointments' not in st.session_state:
    st.session_state.appointments = []

def book_appointment(date, time):
    confirmation_msg = f"Appointment confirmed for {date} at {time}."
    st.session_state.appointments.append({"date": date, "time": time, "confirmation": confirmation_msg})
    return confirmation_msg

def reschedule_appointment(old_date, old_time, new_date, new_time):
    for app in st.session_state.appointments:
        if app['date'] == old_date and app['time'] == old_time:
            app['date'] = new_date
            app['time'] = new_time
            return f"Your appointment has been rescheduled to {new_date} at {new_time}."
    return "No matching appointment found to reschedule."

def cancel_appointment(date, time):
    for app in st.session_state.appointments:
        if app['date'] == date and app['time'] == time:
            st.session_state.appointments.remove(app)
            return f"Your appointment on {date} at {time} has been canceled."
    return "No matching appointment found to cancel."

def main():
    st.title("📅 Appointment Scheduling Chatbot")
    menu = ["Home", "Appointment History", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome! How can I assist you with your appointments today?")
        user_input = st.text_input("You:")

        if user_input:
            response = chatbot_response(user_input)
            st.text_area("Chatbot:", value=response, height=100)

            if "Please provide the date and time" in response:
                date = st.date_input("Select Date")
                time = st.time_input("Select Time")
                if st.button("Confirm Appointment"):
                    confirmation = book_appointment(str(date), str(time))
                    st.success(confirmation)
            
            elif "Please provide the new date and time" in response:
                old_date = st.date_input("Select Current Appointment Date")
                old_time = st.time_input("Select Current Appointment Time")
                new_date = st.date_input("Select New Date")
                new_time = st.time_input("Select New Time")
                if st.button("Reschedule Appointment"):
                    reschedule_msg = reschedule_appointment(str(old_date), str(old_time), str(new_date), str(new_time))
                    st.success(reschedule_msg)
            
            elif "Please provide the date and time of the appointment you want to cancel." in response:
                date = st.date_input("Select Appointment Date to Cancel")
                time = st.time_input("Select Appointment Time to Cancel")
                if st.button("Cancel Appointment"):
                    cancel_msg = cancel_appointment(str(date), str(time))
                    st.warning(cancel_msg)
    
    elif choice == "Appointment History":
        st.header("📜 Your Scheduled Appointments")
        if st.session_state.appointments:
            for app in st.session_state.appointments:
                st.write(f"📅 Date: {app['date']} | ⏰ Time: {app['time']}")
                st.write(f"✅ Confirmation: {app['confirmation']}")
                st.markdown("---")
        else:
            st.write("No appointments scheduled yet.")

        # Conversation History Menu
    elif choice == "Conversation History":
        # Display the conversation history in a collapsible expander
        st.header("Conversation History")
        # with st.beta_expander("Click to see Conversation History"):
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")
    
    elif choice == "About":
        st.write("## 🤖 Appointment Scheduling Chatbot")

        st.write("""
        This chatbot is designed to assist users in scheduling, rescheduling, and managing appointments efficiently. 
        It utilizes Natural Language Processing (NLP) to understand user queries and respond accordingly. 
        The chatbot is built using **Logistic Regression** for intent classification and **Streamlit** for an interactive web-based interface.
        """)

        st.subheader("📌 Project Overview:")

        st.write("""
        The project consists of two main components:
        1. **NLP and Machine Learning**: The chatbot is trained using labeled intent data and Logistic Regression to classify user queries into predefined categories.
        2. **Streamlit Chatbot Interface**: A user-friendly web interface built using Streamlit to allow users to interact with the chatbot, book appointments, and manage scheduling requests.
        """)

        st.subheader("📊 Dataset:")

        st.write("""
        The dataset used for this chatbot includes labeled appointment-related intents such as:
        - **Intent Categories**: Booking, Rescheduling, Cancellation, Checking Availability, Setting Reminders, and more.
        - **Entities Extracted**: Date, Time, Appointment Type, Location, and User Preferences.
        """)

        st.subheader("🖥️ Chatbot Features:")

        st.write("""
        - 📅 **Book Appointments**: Users can schedule appointments based on available slots.
        - 🔄 **Reschedule or Cancel**: Users can modify or cancel their bookings.
        - 🔍 **Check Availability**: Displays open time slots for booking.
        - 🔔 **Set Reminders**: Notifies users before their scheduled appointment.
        - 📍 **Appointment Location**: Provides details about where the appointment is scheduled.
        - 💬 **Interactive Conversation**: Uses NLP to understand natural language queries.
        """)

        st.subheader("🚀 Future Enhancements:")

        st.write("""
        - Implementing advanced NLP techniques such as Transformer-based models (BERT, GPT).
        - Adding voice-enabled appointment booking.
        - Integration with external calendars (Google Calendar, Outlook).
        - Expanding support for multiple languages.
        """)

        st.subheader("🎯 Conclusion:")

        st.write("""
        This project demonstrates how **AI-driven chatbots** can simplify appointment scheduling by automating the booking process.
        It provides a seamless user experience, reduces manual effort, and enhances efficiency.
        Future improvements can make it even smarter and more adaptable to various use cases.
        """)
if __name__ == '__main__':
    main()