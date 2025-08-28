import ollama
import instructor
from pydantic import BaseModel, Field
from typing import Optional

# --- 1. Define Your Pydantic Schema ---
# This is the structure we want the LLM to return.
# Pydantic models are used for data validation and settings management.
# The Field class is used to provide extra information about the model's fields,
# such as descriptions, default values, etc.
class UserDetails(BaseModel):
    name: str = Field(description="The user's full name, extracted from the text.")
    age: int = Field(description="The user's age in years, extracted from the text.")
    city: Optional[str] = Field(description="The city where the user lives, if mentioned.")

# --- 2. Create a client using the correct provider ---
# Instead of patching, we use `instructor.from_provider` which is designed
# to handle clients like Ollama. We specify the model directly here.
# `mode=instructor.Mode.JSON` tells instructor to use Ollama's JSON mode
# for more reliable structured outputs.
client = instructor.from_provider(
    "ollama/llama3",
    mode=instructor.Mode.JSON,
)

# --- 3. Make the API Call with `response_model` ---
def extract_user_details(text: str) -> UserDetails:
    """
    Extracts user details from a string of text using Ollama and Instructor.

    Args:
        text (str): The text to extract information from.

    Returns:
        UserDetails: A Pydantic object containing the extracted user information.
    """
    print("--- Attempting to extract user details ---")
    print(f"Input text: '{text}'")

    try:
        # The `from_provider` method gives us an OpenAI-compatible client.
        # So we use the `chat.completions.create` method.
        # - `response_model`: We pass our Pydantic class `UserDetails`. Instructor will
        #   ensure the LLM's output conforms to this schema.
        # - `max_retries`: If the LLM returns invalid data (e.g., age as a string),
        #   Instructor will automatically detect the Pydantic validation error,
        #   send it back to the LLM, and ask it to fix its own output, up to 3 times.
        user = client.chat.completions.create(
            messages=[{'role': 'user', 'content': f"Extract the user details from this text: {text}"}],
            response_model=UserDetails,
            max_retries=3,
        )
        return user
    except Exception as e:
        print(f"An error occurred: {e}")
        print("This could be because the Ollama server is not running,")
        print("or the specified model ('llama3') is not available.")
        return None

# --- 4. Run the Demonstration ---
if __name__ == "__main__":
    # Note: For this to work, you must have the Ollama application running
    # and have the 'llama3' model downloaded (e.g., by running `ollama run llama3`).

    input_text = "My name is John Doe, I am 30 years old and I live in New York City."

    extracted_user = extract_user_details(input_text)

    if extracted_user:
        print("\n--- Successfully extracted and validated data ---")
        print(f"Name: {extracted_user.name}")
        print(f"Age: {extracted_user.age}")
        print(f"City: {extracted_user.city}")

        # You can also see the entire object as a dictionary
        print("\n--- Pydantic object as dictionary ---")
        print(extracted_user.model_dump_json(indent=2))

        # The type of the returned object is UserDetails, not a dict
        print(f"\nType of result: {type(extracted_user)}")
    else:
        print("\n--- Failed to extract user details ---")
