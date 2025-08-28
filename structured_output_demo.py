import json

# --- Method 1: Schema in Prompt ---
# This is the most basic method. You simply tell the model what you want in the prompt.

def get_user_details_from_prompt(text: str):
    """
    This function demonstrates asking for structured data by describing the
    desired JSON schema directly within the prompt.

    Pros:
    - Simple to implement, no special libraries needed.
    - Works with any LLM that can follow instructions.

    Cons:
    - Prone to errors. The LLM might not perfectly adhere to the format.
    - Requires manual JSON parsing and validation, which can be messy.
    - The model might add extra text (like "Here is the JSON you requested:")
      before or after the JSON object, causing parsing to fail.
    """
    prompt = f"""
    Extract the user's name, age, and city from the following text.
    Return the information as a valid JSON object with the keys "name", "age", and "location".

    Input text: "{text}"

    JSON output:
    """

    # --- SIMULATED LLM CALL ---
    # In a real application, you would make an API call to an LLM here.
    # We will simulate the LLM's response for this demo.
    # Notice that even in a good case, it's just a string that we have to parse.
    print(f"--- Method 1: Prompt Sent to LLM ---\n{prompt}")
    simulated_llm_response = '{\n  "name": "Alice",\n  "age": 34,\n  "location": "New York"\n}'
    print(f"--- Method 1: Raw LLM Response ---\n{simulated_llm_response}\n")

    try:
        # We have to manually parse the string into a Python dictionary.
        structured_data = json.loads(simulated_llm_response)
        return structured_data
    except json.JSONDecodeError:
        print("--- Method 1: FAILED to parse JSON. The LLM response was not valid JSON.")
        return None

# --- DEMONSTRATION ---
user_text = "I'm Alice, and I'm 34 years old. I live in New York."
user_data = get_user_details_from_prompt(user_text)

if user_data:
    print(f"--- Method 1: Successfully Parsed Data ---\nName: {user_data.get('name')}\nAge: {user_data.get('age')}\nLocation: {user_data.get('location')}\n")

print("=" * 40 + "\n")


# --- Method 2: Pydantic for Structured Output ---
# This is the modern, recommended approach. It uses the Pydantic library
# to define a data schema. Many LLM client libraries (like OpenAI's, Anthropic's,
# Instructor, etc.) have a special parameter (e.g., `response_model`) that
# you can pass your Pydantic class to.

from pydantic import BaseModel, Field
from typing import Optional

# First, we define the structure of our desired output using a Pydantic model.
# This class serves as the single source of truth for our data structure.
class UserDetails(BaseModel):
    """A model to hold details about a user."""
    name: str = Field(description="The user's full name.")
    age: int = Field(description="The user's age in years.")
    location: Optional[str] = Field(description="The city where the user lives.", default=None)

def get_user_details_with_pydantic(text: str):
    """
    This function demonstrates using a Pydantic model to get structured output.

    Pros:
    - Highly reliable and robust.
    - Automatic Parsing & Validation: The LLM client library handles converting the
      LLM's string response into a Pydantic object. If the data doesn't match the
      schema (e.g., age is not an integer), Pydantic raises a validation error.
    - Self-Correcting: Some libraries use these validation errors to make another
      call to the LLM, telling it what it did wrong so it can fix its own output.
    - Developer Experience: You get a clean, typed object back, not a string or a
      plain dictionary. This means you get autocomplete and type-checking in your editor.

    Cons:
    - Requires an LLM client library that explicitly supports this feature.
      (Most modern ones do!)
    """
    prompt = f"Extract the user's details from the following text: '{text}'"

    # --- SIMULATED LLM CALL WITH `response_model` ---
    # In a real application, the `client.chat.completions.create` call would
    # handle everything. It converts our Pydantic class into a schema the LLM
    # understands, makes the API call, and parses the response back into an
    # instance of our `UserDetails` class.
    print(f"--- Method 2: Prompt Sent to LLM ---\n{prompt}")
    print(f"--- Method 2: Pydantic `response_model` provided ---\n{UserDetails.__name__}\n")

    # The library returns a ready-to-use Pydantic object, not a raw string.
    # No manual `json.loads()` or validation is needed from our side.
    simulated_llm_response_object = UserDetails(name="Alice", age=34, location="New York")

    return simulated_llm_response_object

# --- DEMONSTRATION ---
user_text = "I'm Alice, and I'm 34 years old. I live in New York."
user_data_pydantic = get_user_details_with_pydantic(user_text)

if user_data_pydantic:
    # We can access attributes directly, with type safety.
    print(f"--- Method 2: Successfully Received Pydantic Object ---\nName: {user_data_pydantic.name}\nAge: {user_data_pydantic.age}\nLocation: {user_data_pydantic.location}\n")

    # What if the LLM provided bad data? Pydantic would catch it before we even get the object.
    # For example, if the LLM returned `{"name": "Bob", "age": "twenty"}`.
    # A good library would see Pydantic's validation error for `age` and could
    # even ask the LLM to fix it automatically. This is the power of this approach.
