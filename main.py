import dspy
from utils import load_wtq_test_questions_with_tables, configure_dspy_lm_studio, print_token_usage
from utils.table_utils import format_table_token_efficient


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_column_values(table_data, column_name, condition_value):
    """
    Count rows where a specific column has a specific value.
    
    Args:
        table_data: Delimited table data (header row + data rows)
        column_name: Name of the column to check
        condition_value: Value to count (e.g., "1", "yes", etc.)
    
    Returns:
        int: Count of rows matching the condition
    """
    lines = table_data.strip().split('\n')
    if len(lines) < 2:
        return 0
    
    # Parse header to find column index
    headers = [h.strip() for h in lines[0].split('|')]
    try:
        column_index = headers.index(column_name)
    except ValueError:
        print(f"Column '{column_name}' not found in headers: {headers}")
        return 0
    
    # Count matching rows
    count = 0
    for line in lines[1:]:  # Skip header
        values = [v.strip() for v in line.split('|')]
        if column_index < len(values) and values[column_index] == str(condition_value):
            count += 1
    
    print(f"🔢 Counted {count} rows where '{column_name}' = '{condition_value}'")
    return count


# ============================================================================
# GLOBAL STATE
# ============================================================================

# Global variable to track the current table
current_table_data = None


# ============================================================================
# TABLE TOOLS FOR REACT AGENT
# ============================================================================

def set_current_table(table_data: str) -> str:
    """Set the current table data for analysis."""
    global current_table_data
    current_table_data = table_data
    return f"Table loaded with {len(table_data.split(chr(10)))} lines"

def get_table_headers() -> str:
    """Get the column headers from the current table."""
    global current_table_data
    if current_table_data is None:
        return "Error: No table data available. This should not happen."
    
    lines = current_table_data.strip().split('\n')
    if len(lines) < 1:
        return "Table has no headers"
    
    headers = [h.strip() for h in lines[0].split('|')]
    return f"Table columns: {headers}"

def count_column_values_tool(column_name: str, condition_value: str) -> str:
    """
    Count rows where a specific column has a specific value.
    
    Args:
        column_name: Name of the column to check
        condition_value: Value to count (e.g., "1", "yes", etc.)
    
    Returns:
        str: Description of the count result
    """
    global current_table_data
    if current_table_data is None:
        return "Error: No table data available. This should not happen."
    
    count = count_column_values(current_table_data, column_name, condition_value)
    return f"Found {count} rows where column '{column_name}' equals '{condition_value}'"

def get_sample_rows(num_rows: int = 3) -> str:
    """
    Get the first N rows of the table data to see the actual format of values.
    
    Args:
        num_rows: Number of data rows to show (default: 3)
    
    Returns:
        str: First N rows of the table data
    """
    global current_table_data
    if current_table_data is None:
        return "Error: No table data available. This should not happen."
    
    lines = current_table_data.strip().split('\n')
    if len(lines) < 2:
        return "Table has no data rows"
    
    # Include header + first num_rows data rows
    sample_lines = lines[:min(num_rows + 1, len(lines))]
    return f"First {num_rows} rows of table data:\n" + "\n".join(sample_lines)


# ============================================================================
# AGENT CREATION
# ============================================================================

def create_table_question_answerer():
    """Create a ReAct agent for answering table questions with table analysis capabilities."""
    return dspy.ReAct(
        signature="question -> answer",
        tools=[get_table_headers, get_sample_rows, count_column_values_tool],
        max_iters=5
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function for the WTQ table question answering system."""
    
    # 1) Configure DSPy with LM Studio
    print("🔧 Configuring DSPy with LM Studio...")
    lm = configure_dspy_lm_studio(
        model_name="openai/qwen/qwen3-4b-2507",
        api_base="http://10.1.11.218:1234/v1",
        api_key="local",
        max_tokens=8192  # Increase for React module responses
    )

    # 1.5) Smoke test to verify DSPy config is working
    print("🧪 Running smoke test...")
    try:
        smoke_response = lm("Say 'Hello from LM Studio!' in exactly 3 words.")
        print(f"✅ Smoke test passed: {smoke_response}")
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        exit(1)

    # 2) Load a simple WTQ test example
    print("📊 Loading WTQ test example...")
    examples = load_wtq_test_questions_with_tables(limit=10)
    example = examples[4]  # Try a different example
    table = example["table"]
    question = example["question"]
    answers = example["answers"]

    # 3) Format the table using token-efficient method (disable row filtering for debugging)
    print("📋 Formatting table data...")
    table_data = format_table_token_efficient(table, question=None, max_rows=1000)

    print(f"WTQ Question: {question}")
    print(f"Expected Answers: {answers}")

    print(f"\nTable Data (token-efficient format):")
    print(table_data)
    print(f"Character count: {len(table_data)}")

    # 4) Set the current table data and test the header tool
    print(f"\n=== Setting Current Table ===")
    set_current_table(table_data)
    print(f"Table loaded successfully")

    # Test the header tool directly
    print(f"\n=== Testing Header Tool ===")
    header_result = get_table_headers()
    print(f"Headers: {header_result}")

    # 5) Create a React agent with table tools
    print("🤖 Creating ReAct agent...")
    agent = create_table_question_answerer()

    # 6) Test with WTQ question
    print(f"\nAsking: {question}")
    result = agent(question=question)
    print(f"Answer: {result.answer}")
    print(f"Tool calls made: {result.trajectory}")

    # 7) Debug: Show the reasoning process
    print("\nDebug - LLM History:")
    dspy.inspect_history()

    # 8) Token usage statistics
    print_token_usage(result)


if __name__ == "__main__":
    main()
