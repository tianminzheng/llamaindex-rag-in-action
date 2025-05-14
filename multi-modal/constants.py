from pathlib import Path

# Directories
ROOT_DIR = Path(__file__).parent.parent.absolute()
# this is just for demonstration, it should be changed to a real blob storage URL.
STORAGE_DIR = Path(ROOT_DIR, "mocked_storage")

# Prompts
DATA_EXTRACT_STR = """\
    can you summarize what is in the image\
    and return the answer with json format \
"""

# Mock response for testing other functionalities not including LLM response generation
EXAMPLE_RESPONSE = {
    "restaurant": "Not Specified",
    "food": "8 Wings or Chicken Poppers",
    "discount": "Black Friday Offer",
    "price": "$8.73",
    "rating": "Not Available",
    "review": "Not Available",
}
