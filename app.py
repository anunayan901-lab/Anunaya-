import os
import streamlit as st
from datetime import datetime, date, time as dtime
from typing import List

# LangChain & docs
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(page_title="Campus Assistant", layout="wide", page_icon="🎓")
st.title("🎓 AI-powered Campus Assistant (Hackathon MVP)")

# -------------------------
# Sample static data (replace with campus JSON or DB)
# -------------------------
timetable = {
    "Monday": [
        {"time": "09:00", "course": "Data Structures", "room": "301"},
        {"time": "11:00", "course": "Physics", "room": "102"},
    ],
    "Tuesday": [
        {"time": "10:00", "course": "DBMS", "room": "201"},
        {"time": "14:00", "course": "Calculus", "room": "103"},
    ],
    "Wednesday": [
        {"time": "09:00", "course": "Machine Learning", "room": "301"},
        {"time": "11:00", "course": "Engineering Graphics", "room": "301"},
    ],
    "Thursday": [
        {"time": "09:00", "course": "Mathematics", "room": "301"},
        {"time": "10:00", "course": "Data Structures", "room": "301"},
    ],
    "Friday": [
        {"time": "09:00", "course": "Python", "room": "301"},
        {"time": "11:00", "course": "DBMS", "room": "301"},

    ]
}

faculty = {
    "Prof. Sharma": {"email": "sharma@college.edu", "phone": "+91-90000-00001", "office": "Block A, 2nd floor"},
    "Prof. Mehta": {"email": "mehta@college.edu", "phone": "+91-90000-00002", "office": "Block B, 1st floor"},
}

locations = {
    "Library": {"desc": "Central library, Block A, 2nd floor", "lat": 28.6139, "lon": 77.2090},
    "Auditorium": {"desc": "Main auditorium, Block C, ground floor", "lat": 28.6145, "lon": 77.2080},
    "Cafeteria": {"desc": "Student cafeteria near north gate", "lat": 28.6140, "lon": 77.2100},
}

# -------------------------
# Session state
# -------------------------
if "events" not in st.session_state:
    st.session_state.events = [
        {"title": "TechFest", "date": date(2025, 10, 5), "time": dtime(17, 0), "loc": "Auditorium"},
        {"title": "Hackathon", "date": date(2025, 10, 10), "time": dtime(9, 30), "loc": "Block C"},
    ]
if "history" not in st.session_state:
    st.session_state.history = []  # (user, bot)
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Campus Data")
if st.sidebar.button("Show sample timetable JSON"):
    st.sidebar.json(timetable)

st.sidebar.markdown("### Add an event (for demo)")
with st.sidebar.form("add_event", clear_on_submit=True):
    etitle = st.text_input("Event title")
    edate = st.date_input("Date", value=date.today())
    etime = st.time_input("Time", value=dtime(17, 0))
    eloc = st.selectbox("Location", list(locations.keys()))
    submit_event = st.form_submit_button("Add event")
    if submit_event and etitle:
        st.session_state.events.append({"title": etitle, "date": edate, "time": etime, "loc": eloc})
        st.sidebar.success(f"Added {etitle} on {edate} at {etime}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Upload course PDFs (for doubt solving)**")
uploaded_files = st.sidebar.file_uploader("PDFs (multiple allowed)", type="pdf", accept_multiple_files=True)
# Build index button
if st.sidebar.button("Build search index from PDFs"):
    if not uploaded_files:
        st.sidebar.warning("Upload at least one PDF first.")
    else:
        with st.spinner("Processing PDFs & building index..."):
            docs = []
            for f in uploaded_files:
                loader = PyPDFLoader(f)
                docs += loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            split_docs = splitter.split_documents(docs)

            embeddings = OpenAIEmbeddings()
            db = FAISS.from_documents(split_docs, embeddings)
            retriever = db.as_retriever(search_kwargs={"k": 4})

            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
            st.session_state.vectorstore = db
            st.session_state.qa_chain = qa
            st.sidebar.success("Index built! You can now ask questions from your notes.")

# -------------------------
# Helper functions
# -------------------------
def format_timetable_day(day_name: str):
    items = timetable.get(day_name, [])
    if not items:
        return f"No classes on {day_name}."
    lines = [f"{it['time']} — {it['course']} (Room {it['room']})" for it in items]
    return "\n".join(lines)

def find_faculty(name_query: str):
    name_query = name_query.lower()
    for name, info in faculty.items():
        if name_query in name.lower() or any(name_query in v.lower() for v in info.values()):
            return name, info
    return None, None

def find_location(q: str):
    q = q.lower()
    for name, info in locations.items():
        if q in name.lower() or q in info["desc"].lower():
            return name, info
    return None, None

def next_class_for_today():
    today = datetime.today().strftime("%A")
    items = timetable.get(today, [])
    if not items:
        return f"No classes today ({today})."
    now = datetime.now().strftime("%H:%M")
    for it in items:
        if it["time"] >= now:
            return f"Next class today: {it['time']} — {it['course']} (Room {it['room']})"
    return f"No more classes today. ({today})"

def ask_pdf_chain(question: str):
    qa = st.session_state.qa_chain
    if not qa:
        return "No PDF index available. Upload PDFs and press 'Build search index from PDFs' in the sidebar."
    res = qa({"question": question, "chat_history": []})
    return res.get("answer") or "I couldn't find an answer in the documents."

def llm_fallback(question: str):
    if not os.getenv("OPENAI_API_KEY"):
        return "No OpenAI API key found. Set OPENAI_API_KEY to allow the model to answer."
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
    resp = llm.generate([{"role": "user", "content": question}])
    try:
        return resp.generations[0][0].text
    except Exception:
        return "LLM fallback failed."

# -------------------------
# Main UI - Chat area
# -------------------------
st.markdown("### Chat with campus assistant")
col1, col2 = st.columns([3,1])

with col1:
    user_input = st.chat_input("Ask about timetable, exams, faculty, notes, or campus navigation...")
    if user_input:
        qlower = user_input.lower()
        bot_reply = None

        # --- 1. Check if user typed a day directly ---
        days = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
        for day in days:
            if day in qlower:
                bot_reply = format_timetable_day(day.capitalize())
                break

        # --- 2. Next class queries ---
        if not bot_reply and any(k in qlower for k in ["next class", "next lecture", "my next class"]):
            bot_reply = next_class_for_today()

        # --- 3. Full timetable queries ---
        elif not bot_reply and ("timetable" in qlower or "class schedule" in qlower or "when is my" in qlower):
            bot_reply = f"Today: {next_class_for_today()}\n\nFull timetable sample:\n" + \
                       "\n".join([f"{d}: {format_timetable_day(d)}" for d in timetable if timetable[d]])

        # --- 4. Faculty queries ---
        elif not bot_reply and any(k in qlower for k in ["prof", "professor", "faculty", "email", "contact"]):
            name_found, info = find_faculty(qlower)
            if name_found:
                bot_reply = f"{name_found}: email {info['email']}, phone {info['phone']}, office {info['office']}"
            else:
                bot_reply = "I couldn't find that faculty member. Try full name or check spelling."

        # --- 5. Location queries ---
        elif not bot_reply and any(k in qlower for k in ["where is", "how to get to", "location", "find"]):
            name_found, info = find_location(qlower)
            if name_found:
                gmaps = f"https://www.google.com/maps/search/?api=1&query={info['lat']},{info['lon']}"
                bot_reply = f"{name_found}: {info['desc']}\nMap: {gmaps}"
            else:
                bot_reply = "I couldn't find that location. Try 'Library', 'Auditorium', or 'Cafeteria'."

        # --- 6. Event queries ---
        elif not bot_reply and any(k in qlower for k in ["event", "remind", "upcoming", "events"]):
            evs = sorted(st.session_state.events, key=lambda e: (e["date"], e["time"]))
            lines = [f"{e['title']} — {e['date'].isoformat()} {e['time'].strftime('%H:%M')} @ {e['loc']}" for e in evs]
            bot_reply = "Upcoming events:\n" + "\n".join(lines)

        # --- 7. Academic/doubt solving queries ---
        elif not bot_reply and any(k in qlower for k in ["explain", "solve", "what is", "how to", "derive", "prove", "why"]):
            if st.session_state.qa_chain:
                bot_reply = ask_pdf_chain(user_input)
            else:
                bot_reply = llm_fallback(user_input)

        # --- 8. Default fallback ---
        elif not bot_reply:
            if st.session_state.qa_chain:
                bot_reply = ask_pdf_chain(user_input)
            else:
                bot_reply = llm_fallback(user_input)

        # store & display
        st.session_state.history.append(("You: " + user_input, "Assistant: " + bot_reply))

# display chat history
with col1:
    for u, r in st.session_state.history[-10:]:
        st.markdown(f"**{u}**")
        st.markdown(r)

with col2:
    st.markdown("### Quick actions")
    if st.button("Show today's next class"):
        st.info(next_class_for_today())
    if st.button("Show faculty list"):
        for name, info in faculty.items():
            st.write(f"- **{name}** — {info['email']}, {info['phone']}")
    st.markdown("---")
    st.write("**Events (next 5)**")
    for e in sorted(st.session_state.events, key=lambda x: (x["date"], x["time"]))[:5]:
        st.write(f"- {e['title']} — {e['date'].isoformat()} {e['time'].strftime('%H:%M')} @ {e['loc']}")





            
