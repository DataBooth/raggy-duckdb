import os

from dotenv import load_dotenv
from together import Together

load_dotenv()


client = Together()
models = client.models.list()

print("Available models:")
for model in models:  # models is a list of ModelObject
    # Print id and display_name if available, else just id
    display_name = getattr(model, "display_name", None)
    if display_name:
        print(f"- {model.id}: {display_name}")
    else:
        print(f"- {model.id}")

    # If you want to debug further, print all keys of the model object:
    # print(model.__dict__)  # or vars(model)
