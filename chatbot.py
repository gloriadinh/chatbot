import spacy
from spacy.pipeline import EntityRuler
from spacy.language import Language
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


class IntentClassifier:
    def __init__(self, intents_path):
        self.intents_path = intents_path
        intents_data = self.load_model()
        self.model = self.train_model(intents_data)
        self.nlp = spacy.load("en_core_web_sm")
        self.add_pipeline()
    
    def load_model(self):  # this method is used to load intents
        try:
            with open(self.intents_path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Error decoding JSON from provided intents file.")
            return []
        except FileNotFoundError:
            print("Intents file not found.")
            return []

    def train_model(self, intents_data):  # train model
        tags = []
        patterns = []
        for intent in intents_data["intents"]:
            for pattern in intent["patterns"]:
                patterns.append(pattern)
                tags.append(intent['tag'])
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(random_state=42))  # using Logistic Regression to classify patterns and tags
        ])

        parameters = {
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'tfidf__max_features': [None, 500, 1000],
            'clf__C': [0.1, 1, 10, 100]
        }  # identify the parameters to be tuned then use grid search to find the best model

        grid_search = GridSearchCV(pipeline, parameters, cv=6, n_jobs=2)
        grid_search.fit(patterns, tags)
        return grid_search.best_estimator_
    
    def predict_intent(self, user_message):
        vectorized_message = self.model['tfidf'].transform([user_message])
        predicted_tag = self.model['clf'].predict(vectorized_message)[0]
        return predicted_tag
    
    @spacy.Language.factory("custom_entity_ruler")
    def create_custom_entity_ruler(nlp, name):
        ruler = EntityRuler(nlp)
        patterns = [{"label": "GPE", "pattern": "Adelaide"}]
        ruler.add_patterns(patterns)
        return ruler

    def add_pipeline(self):
        self.nlp.add_pipe("custom_entity_ruler", before="ner")


class ProcessingData:
    def __init__(self, air_line_file_path):
        self.air_line_file_path = air_line_file_path
    
    def handle_data(self, removed_columns):
        self.df = pd.read_csv(self.air_line_file_path)
        self.df = self.df.drop(removed_columns, errors='ignore')
        return self.df


class RouteFinder:
    def __init__(self, data_loader, removed_columns):
        self.data_loader = data_loader
        self.nlp = spacy.load("en_core_web_sm")
        self.df = self.data_loader.handle_data(removed_columns)

    def find_matched_route(self, message):
        airport_codes = re.findall(r'\b[A-Z]{3}\b', message)  # using regex to find the route that is performed by airport codes
        if airport_codes:
            return " to ".join(airport_codes)
        doc = self.nlp(message)
        locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]  # otherwise, using NLP to recognize GPE 
        return " to ".join(locations)

    def find_matched_airline(self, route):
        if not route:
            return "Route is not determined.", []
        matched_airlines = set()
        for _, row in self.df.iterrows():
            if all(location.lower() in row["Route"].lower() for location in route.split()):
                matched_airlines.add(row["Airline"])

        return route, list(matched_airlines)


class ChatBot:
    def __init__(self, classifier, route_finder, data_processor, sentiment_analyse_path):
        self.classifier = classifier
        self.route_finder = route_finder
        self.data_processor = data_processor
        self.sentiment_analyse_path = sentiment_analyse_path
        self.conversation_state = {
            "route": None,
            "airlines": None,
            "last_tag": None
        }
        self.response_generated = False
        self.chat_response = ""

        self.intents_data = classifier.load_model()
        self.tag = None
        self.route = None
    
    def set_user_message(self, user_message):
        self.user_message = user_message
        self.tag = self.classifier.predict_intent(user_message)
        self.route = self.route_finder.find_matched_route(user_message)
        _, self.matched_airlines = self.route_finder.find_matched_airline(self.route)
        self.route_format_requests = [
            "Can you specify your request?",
            "I didn't catch that. Could you restate your route?",
            "I need more details please.",
            "Could you tell me where you're leaving?",
            "Hmm, could you clarify your route?",
            "Please provide me more information.",
            "Help me out here. Where are you traveling?",
            "I need a bit more info to help."
        ]
        self.response_templates = [
            "You have multiple options: {}. Which would you prefer?",
            "You can choose among these airlines for your route: {}.",
            "For your journey, consider these airlines: {}.",
            "Several airlines are available for your route, including {}."
        ]
        self.recommended_airlines_response_templates = [
            "Considering all factors, {best_airline} would be a great choice for your trip!",
            "Based on our analysis, {best_airline} is the top recommendation for your route.",
            "You might enjoy traveling with {best_airline}, as it's highly rated for this route.",
            "For a pleasant journey, I'd recommend going with {best_airline} for your route."
        ]

    def airline_tag_response(self):
        if self.tag == "airline":
            self.route = self.route_finder.find_matched_route(self.user_message)  # Find the route based on user message
            _, self.matched_airlines = self.route_finder.find_matched_airline(self.route)  # Find airlines for the route

            if self.matched_airlines:
                self.conversation_state.update({
                    "route": self.route,
                    "airlines": self.matched_airlines,
                    "last_tag": self.tag
                })
                
                if len(self.matched_airlines) > 1:
                    self.airlines_list = ", ".join(self.matched_airlines[:-1]) + ", or " + self.matched_airlines[-1]
                    self.chat_response = random.choice(self.response_templates).format(self.airlines_list)
                else:
                    self.chat_response = f"The only airline available that i can provide for your route is {self.matched_airlines[0]}."
                return self.chat_response
            else:
                self.chat_response = random.choice(self.route_format_requests)
                return self.chat_response
        return "I don't have enough information to respond to that right now."

    def recommended_airlines_responses(self, user_message, conversation_state):
        self.route = conversation_state["route"]
        self.matched_airlines = conversation_state["airlines"]

        if not self.route or not self.matched_airlines:
            self.route = self.route_finder.find_matched_route(user_message)
            _, self.matched_airlines = self.route_finder.find_matched_airline(self.route)
            conversation_state["route"] = self.route
            conversation_state["airlines"] = self.matched_airlines

        if self.matched_airlines:
            data = pd.read_csv(self.sentiment_analyse_path)
            matched_route = data[data['Route'].str.contains(self.route, case=False) & data['Airline'].isin(self.matched_airlines)]
            if not matched_route.empty:
                best_airline = matched_route.loc[matched_route['sentiment_score'].idxmax()]['Airline']
                self.chat_response = random.choice(self.recommended_airlines_response_templates).format(best_airline=best_airline)
                return self.chat_response
            else:
                return "No clear best airline found based on the data."
        else:
            return "No airlines found for this route."

    def other_tags_responses(self):
        for intent in self.intents_data["intents"]:
            if intent['tag'] == self.tag and intent['tag'] not in ["airline", "recommended_airlines"]:
                if 'responses' in intent:
                    return random.choice(intent['responses'])
                else:
                    return "I don't have enough information to respond to that right now."
        return "I'm not sure how to help with that."

    def history_of_convo(self, histr_path, text):
        with open(histr_path, "a") as file:
            file.write(text + "\n")


def test():
    classifier = IntentClassifier("./intents.json")
    data_processor = ProcessingData("./airlines_reviews.csv")
    processed_df = data_processor.handle_data(["Name", "Title", "Review Date", "Verified", "Type of Traveller", "Month Flown"])
    route_finder = RouteFinder(data_processor, ["Route", "Airline"])
    intend_dt = classifier.load_model()
    chat_bot = ChatBot(classifier, route_finder, data_processor, "./analyzed_sentiment_result.csv")

    while True:
        try:
            user_message = input()
            if user_message.lower() == 'exit':
                break
            
            if not route_finder.find_matched_route(user_message):
                predicted_intent = classifier.predict_intent(user_message)
            else:
                predicted_intent = classifier.predict_intent(user_message)
            
            chat_bot.set_user_message(user_message)

            if chat_bot.tag == "airline":
                response = chat_bot.airline_tag_response()
            elif chat_bot.tag == "recommended_airlines":
                conver_state = chat_bot.conversation_state
                response = chat_bot.recommended_airlines_responses(user_message, conver_state)
            else:
                response = chat_bot.other_tags_responses()
            
            print(response)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            continue

if __name__ == "__main__":
    test()
