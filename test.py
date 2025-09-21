import dspy
from utils.wtq import load_wtq_test_questions_with_tables
from utils.table_utils import format_table_token_efficient

# 1) Pick a model provider & configure
lm = dspy.LM("ollama_chat/phi4-mini:3.8b", max_tokens=1024, context_length=16384, cache=False)
dspy.configure(lm=lm)

# Enable usage tracking and disable cache
dspy.settings.configure(track_usage=True)
dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

# 2) Load a simple WTQ test example
examples = load_wtq_test_questions_with_tables(limit=10)
example = examples[4]  # Try a different example
table = example["table"]
question = example["question"]
answers = example["answers"]

# 3) Format the table using token-efficient method (disable row filtering for debugging)
table_data = format_table_token_efficient(table, question=None, max_rows=1000)

print(f"WTQ Question: {question}")
print(f"Expected Answers: {answers}")

print(f"\nTable Data (token-efficient format):")
print(table_data)
print(f"Character count: {len(table_data)}")

# 4) Create a simple ChainOfThought module with compact input
agent = dspy.ChainOfThought("table_data, question -> answer")

# 5) Test with WTQ question
print(f"\nAsking: {question}")
result = agent(table_data=table_data, question=question)
print(f"Answer: {result.answer}")
print(f"Reasoning: {result.reasoning}")

# 6) Debug: Show the reasoning process
print("\nDebug - LLM History:")
dspy.inspect_history()

# 7) Token usage statistics
print("\nToken Usage Statistics:")
usage_stats = result.get_lm_usage()
if usage_stats:
    for model_name, stats in usage_stats.items():
        print(f"Model: {model_name}")
        print(f"Input tokens: {stats.get('prompt_tokens', 'N/A')}")
        print(f"Output tokens: {stats.get('completion_tokens', 'N/A')}")
        print(f"Total tokens: {stats.get('total_tokens', 'N/A')}")
        print(f"API calls: {stats.get('calls', 'N/A')}")
else:
    print("No usage statistics available")
