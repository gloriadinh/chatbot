from flask import jsonify
from flask import Flask, request, render_template_string, session, url_for, redirect
from flask_session import Session  # Make sure to install this with `pip install Flask-Session`
from chatbot import IntentClassifier, ProcessingData, RouteFinder, ChatBot

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Initialize components of your ChatBot
classifier = IntentClassifier("./intents.json")
data_processor = ProcessingData("./airlines_reviews.csv")
route_finder = RouteFinder(data_processor, ["Route", "Airline"])
chat_bot = ChatBot(classifier, route_finder, data_processor, "./analyzed_sentiment_result.csv")

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ChatBot Interface</title>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined&display=swap">
    <style>
        body, html {
            height: 100%;
            margin: 0;
            padding: 0;
            background-color: #f4f4f4; /* Light grey background */
            display: flex;
            justify-content: center;
            align-items: center;
            font-family: Arial, sans-serif;
            font-size: 1rem; /* Relative font size */
        }
        #chat-box {
            width: 35vw; /* Increased width */
            max-width: 600px; /* Increased maximum width */
            height: 75vh; /* Full height */
            max-height: 800px; /* Maximum height */
            display: flex;
            flex-direction: column;
            background-color: white;
            overflow: hidden;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
            border-radius: 10px;
        }
        #message-container {
            flex-grow: 1;
            overflow-y: auto;
            padding: 1rem; /* Relative padding */
            display: flex;
            flex-direction: column;
        }
        .message {
            padding: 0.625rem 0.9375rem; /* Relative padding */
            border-radius: 1.25rem;
            margin: 0.25rem 0;
            max-width: 80%; /* Responsive max-width */
            word-break: break-word;
            white-space: pre-line; /* Maintains formatting for any included line breaks */
        }
        .user-message {
            background-color: #3182ce; /* Blue background for user messages */
            color: white;
            align-self: flex-end;
            border: 1px solid #3182ce; /* Keep the border for user messages */
        }
        .bot-message {
            background-color: #d0d7dc; /* Light grey background for bot messages */
            color: black;
            align-self: flex-start;
            border: 1px solid #d0d7dc;
        }
        .chat-form {
            display: flex;
            padding: 0.625rem; /* Relative padding */
            background: #fff;
            border-top: 1px solid #ccc;
            position: relative;
        }
        .chat-form input[type="text"] {
            flex-grow: 1;
            padding: 0.5rem; /* Relative padding */
            border: 1px solid #ccc;
            border-radius: 0.3125rem; /* Relative border-radius */
            margin-right: 1px; /* Relative margin */
        }
        .send-button {
            position: absolute;
            right: 1.0rem;
            top: 54%;
            transform: translateY(-50%);
            background: none;
            border: none;
            cursor: pointer;
            color: #1f9ffb;
            font-size: 1.5rem; /* Adjust icon size */
        }
        .end-chat-button {
            padding: 0.625rem 0.5rem; /* Relative padding */
            background-color: white;
            color: #1f9ffb;
            border: none;
            cursor: pointer;
            text-align: center;
            width: 100%;
            border-top: 1px solid #ccc;
        }
        .border-box {
            border-bottom: 1px solid lightgrey; /* Light grey bottom border */
            font-size: 1.5625rem; /* Relative font size */
            text-align: center; /* Center the text */
            padding: 0.625rem 0; /* Relative padding */
            font-weight: normal; /* Remove bold */
        }
    </style>
</head>
<body>
    <div id="chat-box">
        <div class="border-box">
            <p>Airline supporter</p>
        </div>
        <div id="message-container"></div>
        <form class="chat-form" onsubmit="sendMessage(event)">
            <input type="text" name="message" placeholder="Type your message..." required>
            <button class="send-button" type="submit">
                <span class="material-symbols-outlined">send</span>
            </button>
        </form>
        <button class="end-chat-button" onclick="endChat()">End Chat</button>
    </div>
    <script>
        function sendMessage(event) {
            event.preventDefault();
            const input = document.querySelector('input[name="message"]');
            const message = input.value.trim();
            if (!message) return;
            addMessage(message, 'user');
            fetch('/message', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: message})
            })
            .then(response => response.json())
            .then(data => {
                addMessage(data.message, 'bot');
            })
            .catch(error => console.error('Error:', error));
            input.value = '';
        }

        function addMessage(text, sender) {
            const container = document.getElementById('message-container');
            const messageDiv = document.createElement('div');
            messageDiv.textContent = text;
            messageDiv.className = 'message ' + (sender === 'user' ? 'user-message' : 'bot-message');
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight; // Ensure the newest messages are visible
        }

        function endChat() {
            const container = document.getElementById('message-container');
            container.innerHTML = ''; // Clear all chat messages
        }
    </script>
</body>
</html>



"""

def handle_user_message(user_input):
    chat_bot.set_user_message(user_input)
    if chat_bot.tag == "airline":
        response = chat_bot.airline_tag_response()
    elif chat_bot.tag == "recommended_airlines":
        response = chat_bot.recommended_airlines_responses(user_input, chat_bot.conversation_state)
    else:
        response = chat_bot.other_tags_responses()

    if not response:
        response = "Sorry, I couldn't understand that. Can you rephrase?"

    print(response)

    return response


from flask import jsonify

@app.route('/message', methods=['POST'])
def message():
    try:
        user_input = request.get_json().get('message')  # Correctly get the message from JSON
        if not user_input:
            raise ValueError("Empty message received")

        response = handle_user_message(user_input)
        return jsonify({'message': response})

    except Exception as e:
        print(f"Error processing message: {e}")
        return jsonify({'message': 'Error processing your message'}), 500

@app.route('/', methods=['GET', 'POST'])
def chat():
    print("Accessing chat endpoint")
    if 'messages' not in session:
        print("Initializing messages in session")
        session['messages'] = []

    if request.method == 'POST':
        user_message = request.form['message']
        print(f"Received message: {user_message}")
        session['messages'].append((user_message, 'user'))

        chat_bot.set_user_message(user_message)
        response = "No response generated"  # Default response
        if chat_bot.tag == "airline":
            response = chat_bot.airline_tag_response()
        elif chat_bot.tag == "recommended_airlines":
            response = chat_bot.recommended_airlines_responses(user_message, chat_bot.conversation_state)
        else:
            response = chat_bot.other_tags_responses()
        print(f"Generated response: {response}")

        session['messages'].append((response, 'bot'))
        session.modified = True

    print("Rendering page")
    return render_template_string(HTML, messages=session['messages'])

@app.route('/end_chat', methods=['POST'])
def end_chat():
    # Clear the session
    session.pop('messages', None)
    return redirect(url_for('chat'))

if __name__ == "__main__":
    app.run(debug=True, port=5002)
