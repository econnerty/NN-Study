# Erik Connerty
# 6/24/2023
# USC - AI institute
from transformers import Conversation, pipeline
chatbot = pipeline("conversational", model = "TrainedModels/GPT2_TrainedModels")

# Start a conversation
conversation = Conversation()

while True:
    # User input
    user_input = input("You: ")

    # Append the user input to the conversation
    conversation.add_user_input(user_input)

    # Generate a response
    chatbot_response = chatbot(conversation)

    # Print the chatbot's response
    print("Chatbot:", chatbot_response.generated_responses[-1])
