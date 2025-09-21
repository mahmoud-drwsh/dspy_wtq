# hello_dspy_lmstudio.py
import os
import dspy

# Enable DSPy verbose logging
dspy.settings.configure(verbose=True)

# Configure LM Studio (OpenAI-compatible)
lm = dspy.LM(
    "openai/deepseek/deepseek-r1-0528-qwen3-8b",
    api_base="http://10.1.11.218:1234/v1",
    api_key="local",
    model_type="chat",
    temperature=1
)
dspy.configure(lm=lm)

print("=== Basic LM Call ===")
print(lm("say hello to the world!")[0])

print("\n=== ReAct Demo with Weather and Search Tools ===")

# Define tools as functions (from tools.md examples)
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In a real implementation, this would call a weather API
    weather_info = f"The weather in {city} is sunny and 75°F"
    print(f"🔧 Tool called: get_weather(city='{city}') -> '{weather_info}'")
    return weather_info

def search_web(query: str) -> str:
    """Search the web for information."""
    # In a real implementation, this would call a search API
    search_results = f"Search results for '{query}': [relevant information about {query}...]"
    print(f"🔧 Tool called: search_web(query='{query}') -> '{search_results}'")
    return search_results

# Create a ReAct agent (from tools.md)
react_agent = dspy.ReAct(
    signature="question -> answer",
    tools=[get_weather, search_web],
    max_iters=5
)

# Test ReAct with weather questions
print("Question: What's the weather like in Tokyo?")
result1 = react_agent(question="What's the weather like in Tokyo?")
print(f"Answer: {result1.answer}")
print("Tool calls made:", result1.trajectory)

print("\nQuestion: Search for information about DSPy")
result2 = react_agent(question="Search for information about DSPy")
print(f"Answer: {result2.answer}")
print("Tool calls made:", result2.trajectory)

print("\n=== Demo Complete ===")
