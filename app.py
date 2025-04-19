# app.py - Main Flask Application
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import firebase_admin
from firebase_admin import credentials, firestore
import uuid
from datetime import datetime
import json
import os
import requests
import chromadb
from chromadb.utils import embedding_functions
import chromadb.db.base
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "mental_health_monitor_secret_key")

# Initialize Firebase
firebase_cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
cred = credentials.Certificate(firebase_cred_path)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize ChromaDB for conversation memory
chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
chroma_client = chromadb.PersistentClient(chroma_path)

# Together AI API settings
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_API_URL = os.getenv("TOGETHER_API_URL", "https://api.together.xyz/v1/completions")
TOGETHER_EMBEDDING_URL = "https://api.together.xyz/v1/embeddings"
MODEL_NAME = os.getenv("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "togethercomputer/m2-bert-80M-8k-retrieval")

# Create custom embedding function using Together AI
# Change your TogetherAIEmbeddings class to use 'input' instead of 'texts'
class TogetherAIEmbeddings(embedding_functions.EmbeddingFunction):
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
    
    def __call__(self, input):  # Changed from 'texts' to 'input'
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "input": input  # Changed from 'texts' to 'input'
        }
        
        response = requests.post(TOGETHER_EMBEDDING_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return [embedding_data["embedding"] for embedding_data in result["data"]]
        else:
            # Return empty embeddings on error
            print(f"Error from Together API Embeddings: {response.text}")
            # Return zero vectors with dimension 768 (standard for embedding models)
            return [[0.0] * 768 for _ in range(len(input))]
# Initialize Together AI embedding function
together_embeddings = TogetherAIEmbeddings(TOGETHER_API_KEY, EMBEDDING_MODEL)

# Create or get collection for storing conversation history
# Uncomment this block to use ChromaDB
try:
    conversation_collection = chroma_client.create_collection(
        name="conversations", 
        embedding_function=together_embeddings,
        metadata={"hnsw:space": "cosine"}
    )
except (ValueError, chromadb.db.base.UniqueConstraintError):
    # Collection already exists
    conversation_collection = chroma_client.get_collection(
        name="conversations", 
        embedding_function=together_embeddings
    )

# Function to analyze sentiment using Together AI
def analyze_sentiment(text):
    prompt = f"""
    [INST] Analyze the sentiment of the following text and rate it on a scale of 0 to 10 where:
    0-3.4: negative
    3.5-6.9: neutral
    7-10: positive
    
    Please return the result in JSON format with these fields:
    - score: the numerical score (0-10, with one decimal place)
    - sentiment: the category (negative, neutral, or positive)
    - raw_score: the raw sentiment probability between 0 and 1
    
    Text to analyze: "{text}" [/INST]
    """
    
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.2,
        "top_p": 0.7,
        "stop": ["</s>", "[/INST]"],
        "frequency_penalty": 0,
        "presence_penalty": 0
    }
    
    response = requests.post(TOGETHER_API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        try:
            # Extract JSON from the model's response
            json_str = result["choices"][0]["text"].strip()
            # Find JSON pattern in the text
            import re
            json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if json_match:
                sentiment_data = json.loads(json_match.group(0))
                return sentiment_data
            else:
                # Fallback if JSON extraction fails
                return {
                    "score": 5.0,
                    "sentiment": "neutral",
                    "raw_score": 0.5
                }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing sentiment response: {e}")
            return {
                "score": 5.0,
                "sentiment": "neutral",
                "raw_score": 0.5
            }
    else:
        print(f"Error from Together API: {response.text}")
        return {
            "score": 5.0,
            "sentiment": "neutral",
            "raw_score": 0.5
        }

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        user_type = request.form.get('user_type')
        
        if user_type == 'student':
            # Check if student exists
            students_ref = db.collection('students')
            query = students_ref.where('email', '==', email).limit(1).get()
            
            if len(query) > 0:
                student = query[0].to_dict()
                session['user'] = student
                session['user_type'] = 'student'
                return redirect(url_for('student_dashboard'))
            else:
                return render_template('login.html', error="Student not found")
                
        elif user_type == 'faculty':
            # Check if faculty exists
            faculty_ref = db.collection('faculty')
            query = faculty_ref.where('email', '==', email).limit(1).get()
            
            if len(query) > 0:
                faculty = query[0].to_dict()
                session['user'] = faculty
                session['user_type'] = 'faculty'
                return redirect(url_for('faculty_dashboard'))
            else:
                return render_template('login.html', error="Faculty not found")
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        age = request.form.get('age')
        gender = request.form.get('gender')
        class_section = request.form.get('class_section')
        proctor = request.form.get('proctor')
        user_type = request.form.get('user_type')
        
        if user_type == 'student':
            # Create student record
            student_id = str(uuid.uuid4())
            student_data = {
                'student_id': student_id,
                'name': name,
                'email': email,
                'age': age,
                'gender': gender,
                'class_section': class_section,
                'proctor': proctor,
                'created_at': datetime.now()
            }
            
            db.collection('students').document(student_id).set(student_data)
            session['user'] = student_data
            session['user_type'] = 'student'
            return redirect(url_for('student_dashboard'))
        
        elif user_type == 'faculty':
            # Create faculty record
            faculty_id = str(uuid.uuid4())
            faculty_data = {
                'faculty_id': faculty_id,
                'name': name,
                'email': email,
                'created_at': datetime.now()
            }
            
            db.collection('faculty').document(faculty_id).set(faculty_data)
            session['user'] = faculty_data
            session['user_type'] = 'faculty'
            return redirect(url_for('faculty_dashboard'))
    
    return render_template('register.html')

@app.route('/student/dashboard')
def student_dashboard():
    if 'user' not in session or session['user_type'] != 'student':
        return redirect(url_for('login'))
    
    return render_template('student_dashboard.html', student=session['user'])

@app.route('/faculty/dashboard')
def faculty_dashboard():
    if 'user' not in session or session['user_type'] != 'faculty':
        return redirect(url_for('login'))
    
    # Get all students for monitoring
    students = []
    students_ref = db.collection('students').get()
    
    for student_doc in students_ref:
        student_data = student_doc.to_dict()
        
        # Get latest sentiment info
        # FIXED: Create index for this query in Firebase console
        # See error message for link to create the index
        chat_ref = db.collection('chat_history')\
            .where('student_id', '==', student_data['student_id'])\
            .order_by('timestamp', direction=firestore.Query.DESCENDING)\
            .limit(1)\
            .get()
        
        latest_sentiment = "No data"
        latest_score = 0
        if len(chat_ref) > 0:
            latest_sentiment = chat_ref[0].to_dict().get('sentiment', 'No data')
            latest_score = chat_ref[0].to_dict().get('sentiment_score', 0)
        
        student_data['latest_sentiment'] = latest_sentiment
        student_data['latest_score'] = latest_score
        students.append(student_data)
    
    return render_template('faculty_dashboard.html', faculty=session['user'], students=students)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/api/chat', methods=['POST'])
def chat():
    if 'user' not in session or session['user_type'] != 'student':
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    user_message = data.get('message', '')
    student = session['user']
    student_id = student['student_id']
    
    # Analyze sentiment
    sentiment_result = analyze_sentiment(user_message)
    
    # Generate chatbot response based on sentiment and context using memory
    chatbot_response = generate_response_with_memory(user_message, sentiment_result, student)
    
    # Store chat in database
    chat_data = {
        'student_id': student_id,
        'student_message': user_message,
        'chatbot_response': chatbot_response,
        'sentiment': sentiment_result['sentiment'],
        'sentiment_score': sentiment_result['score'],
        'raw_score': sentiment_result.get('raw_score', 0.5),
        'timestamp': datetime.now()
    }
    
    db.collection('chat_history').add(chat_data)
    
    # Store in ChromaDB for contextual memory
    # Ensure conversation_collection is defined before this call
    conversation_collection.add(
        documents=[f"User: {user_message}\nBot: {chatbot_response}"],
        metadatas=[{
            "student_id": student_id,
            "timestamp": datetime.now().isoformat(),
            "sentiment": sentiment_result['sentiment'],
            "sentiment_score": sentiment_result['score']
        }],
        ids=[f"{student_id}-{datetime.now().timestamp()}"]
    )
    
    return jsonify({
        "response": chatbot_response,
        "sentiment": sentiment_result['sentiment'],
        "score": sentiment_result['score']
    })

@app.route('/api/student/history')
def get_student_history():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    student_id = session['user']['student_id'] if session['user_type'] == 'student' else request.args.get('student_id')
    
    if not student_id:
        return jsonify({"error": "Student ID required"}), 400
    
    try:
        # Get chat history for student
        # FIXED: You need to create an index for this query
        # Follow the link in the error message to create it
        chat_ref = db.collection('chat_history')\
            .where('student_id', '==', student_id)\
            .order_by('timestamp')\
            .get()
        
        history = []
        for doc in chat_ref:
            chat_data = doc.to_dict()
            # Convert timestamp to string for JSON serialization
            chat_data['timestamp'] = chat_data['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            history.append(chat_data)
        
        return jsonify(history)

    except Exception as e:
        print(f"Error getting student history: {e}")
        return jsonify({"error": "Failed to retrieve history. Please ensure the index has been created."}), 500

@app.route('/api/faculty/student_data')
def get_student_data():
    if 'user' not in session or session['user_type'] != 'faculty':
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get sentiment distribution for all students
    students_ref = db.collection('students').get()
    
    result = []
    for student_doc in students_ref:
        student_data = student_doc.to_dict()
        student_id = student_data['student_id']
        
        # Get chat history for sentiment distribution
        chat_ref = db.collection('chat_history')\
            .where('student_id', '==', student_id)\
            .get()
        
        positive = 0
        neutral = 0
        negative = 0
        
        for chat in chat_ref:
            sentiment = chat.to_dict().get('sentiment')
            if sentiment == 'positive':
                positive += 1
            elif sentiment == 'neutral':
                neutral += 1
            elif sentiment == 'negative':
                negative += 1
        
        total = positive + neutral + negative
        
        if total > 0:
            distribution = {
                'positive': round(positive / total * 100, 1),
                'neutral': round(neutral / total * 100, 1),
                'negative': round(negative / total * 100, 1)
            }
        else:
            distribution = {
                'positive': 0,
                'neutral': 0,
                'negative': 0
            }
        
        result.append({
            'student_id': student_id,
            'name': student_data['name'],
            'sentiment_distribution': distribution,
            'total_interactions': total
        })
    
    return jsonify(result)

def generate_response_with_memory(user_message, sentiment_result, student):
    """Generate an appropriate response using Together AI and conversation memory from ChromaDB"""
    
    student_id = student['student_id']
    name = student['name'].split()[0]  # Get first name
    
    try:
        # Retrieve relevant past conversations from ChromaDB
        results = conversation_collection.query(
            query_texts=[user_message],
            where={"student_id": student_id},
            n_results=5
        )
        
        # Format conversation history
        conversation_history = ""
        if results['documents'][0]:
            conversation_history = "\n".join(results['documents'][0])
        
        # Get recent sentiments from Firebase for context
        recent_chats = db.collection('chat_history')\
            .where('student_id', '==', student_id)\
            .order_by('timestamp', direction=firestore.Query.DESCENDING)\
            .limit(5)\
            .get()
        
        # Track sentiment trends
        sentiment_trend = []
        for chat in recent_chats:
            sentiment_trend.append(chat.to_dict().get('sentiment'))
        
        # Check if there's a concerning pattern (multiple negative sentiments)
        concern_level = "normal"
        if len(sentiment_trend) >= 3 and sentiment_trend.count("negative") >= 2:
            concern_level = "concerned"
        
        # Build the prompt for the LLM
        prompt = f"""
        [INST] You are a supportive mental health chatbot for students. Your name is MindfulMentor. 
        Your role is to provide empathetic responses and support for the student you're talking to.
        
        Student Profile:
        - Name: {student['name']} (use their first name: {name})
        - Age: {student['age']}
        - Gender: {student['gender']}
        
        Current Sentiment Analysis:
        - Score: {sentiment_result['score']}/10
        - Category: {sentiment_result['sentiment']}
        
        Conversation History:
        {conversation_history}
        
        Recent Sentiment Trend: {', '.join(sentiment_trend[:3]) if sentiment_trend else 'No history'}
        Concern Level: {concern_level}
        
        The student just said: "{user_message}"
        
        Please respond in an empathetic, supportive way that:
        1. Acknowledges their current emotion
        2. Provides appropriate support or guidance
        3. Asks a thoughtful follow-up question if appropriate
        4. Keeps your response concise (2-4 sentences)
        5. Avoids being overly clinical or using psychological jargon
        6. Never suggests you are an AI or language model
        
        If they express serious mental health concerns, encourage seeking professional help from school counselors.
        [/INST]
        """
        
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "stop": ["</s>", "[/INST]", "Student:"]
        }
        
        response = requests.post(TOGETHER_API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["text"].strip()
        else:
            # If API call fails, use simplified prompt for fallback
            return generate_fallback_response(user_message, sentiment_result, name)
    except Exception as e:
        print(f"Error generating response: {e}")
        return generate_fallback_response(user_message, sentiment_result, name)

def generate_fallback_response(user_message, sentiment_result, name):
    """Generate a fallback response using LLM with a simpler prompt if main call fails"""
    try:
        # Simplified prompt for fallback
        simple_prompt = f"""
        [INST] You are a supportive mental health chatbot. Respond to this student message:
        
        Student {name} said: "{user_message}"
        
        Their current sentiment is: {sentiment_result['sentiment']}
        
        Reply with a brief supportive message (2-3 sentences):
        [/INST]
        """
        
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": MODEL_NAME,
            "prompt": simple_prompt,
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5
        }
        
        response = requests.post(TOGETHER_API_URL, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["text"].strip()
        else:
            print(f"Fallback response failed with status {response.status_code}")
            # Ultimate fallback if even the simplified prompt fails
            message_type = sentiment_result['sentiment']
            if message_type == 'negative':
                return f"I understand this is difficult, {name}. Would you like to share more about what you're experiencing?"
            elif message_type == 'neutral':
                return f"Thanks for sharing, {name}. How else have you been feeling lately?"
            else:
                return f"That's wonderful to hear, {name}! What else has been going well for you?"
    except Exception as e:
        print(f"Error in fallback response: {e}")
        return f"I'm here to listen, {name}. Would you like to tell me more about how you're feeling?"

if __name__ == '__main__':
    app.run(debug=True)