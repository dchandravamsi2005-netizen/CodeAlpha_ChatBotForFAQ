from flask import Flask, render_template, request, jsonify
from faq_data import faqs
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

questions = list(faqs.keys())

vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(questions)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():

    user_input = request.json['message']

    if user_input.lower() in ['hi', 'hello', 'hey']:
        return jsonify({'reply': 'Hello! How can I help you?'})

    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vector, vectors)

    index = similarity.argmax()

    score = similarity[0][index]

    if score > 0.2:
        answer = faqs[questions[index]]
    else:
        answer = "Sorry, I couldn't understand your question."

    return jsonify({'reply': answer})


if __name__ == "__main__":
    app.run(debug=True)