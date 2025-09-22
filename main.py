import dspy
import datetime
import json
import os
from utils import (
    load_wtq_test_questions_with_tables, 
    configure_dspy_lm_studio,
    format_table_token_efficient,
    run_evaluation_loop,
    print_evaluation_summary,
    save_run_results,
    save_reasoning_analysis,
    print_token_usage
)


# ============================================================================
# CUSTOM SIGNATURES
# ============================================================================

class TableQuestionAnswering(dspy.Signature):
    """Answer questions about tabular data using the provided table context and available tools.
    
    Answer Format Instructions:
    - Give a concise, direct answer (e.g., "Italy", "17", "January 26, 1995")
    - If the answer is not found in the table, respond with "I don't know"
    - Do not provide explanations, reasoning, or verbose descriptions
    - Do not say "The question cannot be answered" - use "I don't know" instead
    """
    
    question: str = dspy.InputField(desc="The question to answer about the table data")
    table_headers: str = dspy.InputField(desc="List of column headers in the table")
    total_columns: int = dspy.InputField(desc="Total number of columns in the table")
    total_rows: int = dspy.InputField(desc="Total number of data rows in the table")
    sample_rows: str = dspy.InputField(desc="First few rows of the table data to understand the data format and values")
    answer: str = dspy.OutputField(desc="Concise answer to the question. Use 'I don't know' if answer not found in table.")


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
    print(f"🔧 set_current_table called with {len(table_data.split(chr(10)))} lines")
    return f"Table loaded with {len(table_data.split(chr(10)))} lines"

def get_table_headers() -> str:
    """Get the column headers from the current table."""
    global current_table_data
    print("🔧 get_table_headers called")
    if current_table_data is None:
        return "Error: No table data available. This should not happen."
    
    lines = current_table_data.strip().split('\n')
    if len(lines) < 1:
        return "Table has no headers"
    
    headers = [h.strip() for h in lines[0].split('|')]
    return f"Table columns: {headers}"

def get_table_headers_list() -> list:
    """Get list of table headers."""
    global current_table_data
    print("🔧 get_table_headers_list called")
    if current_table_data is None:
        return []
    
    lines = current_table_data.strip().split('\n')
    if len(lines) < 1:
        return []
    
    return [h.strip() for h in lines[0].split('|')]

def get_table_row_count() -> int:
    """Get total number of data rows."""
    global current_table_data
    print("🔧 get_table_row_count called")
    if current_table_data is None:
        return 0
    
    lines = current_table_data.strip().split('\n')
    return max(0, len(lines) - 1)  # Subtract header

def get_sample_rows(num_rows: int = 3) -> str:
    """
    Get the first N rows of the table data to see the actual format of values.
    
    Args:
        num_rows: Number of data rows to show (default: 3)
    
    Returns:
        str: First N rows of the table data
    """
    global current_table_data
    print(f"🔧 get_sample_rows called: num_rows={num_rows}")
    if current_table_data is None:
        return "Error: No table data available. This should not happen."
    
    lines = current_table_data.strip().split('\n')
    if len(lines) < 2:
        return "Table has no data rows"
    
    # Include header + first num_rows data rows
    sample_lines = lines[:min(num_rows + 1, len(lines))]
    return f"First {num_rows} rows of table data:\n" + "\n".join(sample_lines)

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
    print(f"🔧 count_column_values_tool called: column='{column_name}', value='{condition_value}'")
    if current_table_data is None:
        return "Error: No table data available. This should not happen."
    
    try:
        lines = current_table_data.strip().split('\n')
        if len(lines) < 2:
            return "Table has no data rows"
        
        # Parse header to find column index
        headers = [h.strip() for h in lines[0].split('|')]
        try:
            column_index = headers.index(column_name)
        except ValueError:
            return f"Column '{column_name}' not found in headers: {headers}"
        
        # Count matching rows
        count = 0
        for line in lines[1:]:  # Skip header
            values = [v.strip() for v in line.split('|')]
            if column_index < len(values) and values[column_index] == str(condition_value):
                count += 1
        
        print(f"🔢 Counted {count} rows where '{column_name}' = '{condition_value}'")
        return f"Found {count} rows where column '{column_name}' equals '{condition_value}'"
    
    except Exception as e:
        return f"Error: {str(e)}"

def count_column_contains_tool(column_name: str, search_value: str) -> str:
    """
    Count rows where a specific column contains a search value (partial match).
    
    Args:
        column_name: Name of the column to check
        search_value: Value to search for within cells (e.g., "ITA", "USA", etc.)
    
    Returns:
        str: Description of the count result
    """
    global current_table_data
    print(f"🔧 count_column_contains_tool called: column='{column_name}', search='{search_value}'")
    if current_table_data is None:
        return "Error: No table data available. This should not happen."
    
    try:
        lines = current_table_data.strip().split('\n')
        if len(lines) < 2:
            return "Table has no data rows"
        
        # Parse header to find column index
        headers = [h.strip() for h in lines[0].split('|')]
        try:
            column_index = headers.index(column_name)
        except ValueError:
            return f"Column '{column_name}' not found in headers: {headers}"
        
        # Count matching rows (case-insensitive partial match)
        count = 0
        search_lower = search_value.lower()
        for line in lines[1:]:  # Skip header
            values = [v.strip() for v in line.split('|')]
            if column_index < len(values) and search_lower in values[column_index].lower():
                count += 1
        
        print(f"🔍 Counted {count} rows where '{column_name}' contains '{search_value}'")
        return f"Found {count} rows where column '{column_name}' contains '{search_value}'"
    
    except Exception as e:
        return f"Error: {str(e)}"

def get_row_by_condition_tool(column_name: str, condition_value: str, match_type: str = "exact") -> str:
    """
    Get the entire row where a specific column meets a condition.
    
    Args:
        column_name: Name of the column to check
        condition_value: Value to match against
        match_type: "exact" for exact match, "contains" for partial match
    
    Returns:
        str: The complete row data or error message
    """
    global current_table_data
    print(f"🔧 get_row_by_condition_tool called: column='{column_name}', value='{condition_value}', match_type='{match_type}'")
    if current_table_data is None:
        return "Error: No table data available. This should not happen."
    
    try:
        lines = current_table_data.strip().split('\n')
        if len(lines) < 2:
            return "Table has no data rows"
        
        # Parse header to find column index
        headers = [h.strip() for h in lines[0].split('|')]
        try:
            column_index = headers.index(column_name)
        except ValueError:
            return f"Column '{column_name}' not found in headers: {headers}"
        
        # Find matching row
        for line in lines[1:]:  # Skip header
            values = [v.strip() for v in line.split('|')]
            if column_index < len(values):
                cell_value = values[column_index]
                
                # Check match condition
                if match_type == "exact" and cell_value == condition_value:
                    # Format the row with headers
                    row_data = {}
                    for i, header in enumerate(headers):
                        row_data[header] = values[i] if i < len(values) else ""
                    
                    print(f"📋 Found row where '{column_name}' = '{condition_value}'")
                    return f"Row data: {row_data}"
                
                elif match_type == "contains" and condition_value.lower() in cell_value.lower():
                    # Format the row with headers
                    row_data = {}
                    for i, header in enumerate(headers):
                        row_data[header] = values[i] if i < len(values) else ""
                    
                    print(f"📋 Found row where '{column_name}' contains '{condition_value}'")
                    return f"Row data: {row_data}"
        
        return f"No row found where '{column_name}' {match_type} '{condition_value}'"
    
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# AGENT CREATION
# ============================================================================

def create_table_question_answerer():
    """Create a ReAct agent for answering table questions with simple table analysis capabilities."""
    return dspy.ReAct(
        signature=TableQuestionAnswering,
        tools=[
            count_column_values_tool,
            count_column_contains_tool,
            get_row_by_condition_tool
        ],
        max_iters=5
    )


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Configuration file {config_path} not found")
        raise
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in configuration file: {e}")
        raise
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        raise


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function for the WTQ table question answering system."""
    
    # 1) Load configuration from JSON file
    print("🔧 Loading configuration...")
    config = load_config()
    
    # 2) Configure DSPy with LM Studio
    print("🔧 Configuring DSPy with LM Studio...")
    
    lm = configure_dspy_lm_studio(
        model=config["model"],
        api_base=config["api_base"],
        api_key=config["api_key"],
        model_type=config["model_type"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        track_usage=config["track_usage"],
        disk_cache=config["disk_cache"],
        memory_cache=config["memory_cache"]
    )

    # 3) Smoke test to verify DSPy config is working
    print("🧪 Running smoke test...")
    try:
        smoke_response = lm("Say 'Hello from LM Studio!' in exactly 3 words.")
        print(f"✅ Smoke test passed: {smoke_response}")
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        exit(1)

    # 4) Load WTQ test examples
    print("📊 Loading WTQ test examples...")
    examples = load_wtq_test_questions_with_tables(limit=config["test_questions_limit"])
    print(f"Loaded {len(examples)} examples")

    # 5) Create a React agent with table tools
    print("🤖 Creating ReAct agent...")
    agent = create_table_question_answerer()

    # 6) Run evaluation loop
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results = run_evaluation_loop(agent, examples, config, run_timestamp)

    # 7) Print summary and save results
    accuracy, correct_count = print_evaluation_summary(results, examples)
    
    # 8) Save final results for analysis
    print(f"\n{'='*80}")
    print("💾 SAVING FINAL RESULTS")
    print(f"{'='*80}")
    saved_file, _ = save_run_results(config, results, accuracy, len(examples), correct_count, 
                                   is_incremental=False, run_timestamp=run_timestamp)
    
    # 9) Save detailed reasoning analysis
    print(f"\n{'='*80}")
    print("🧠 SAVING REASONING ANALYSIS")
    print(f"{'='*80}")
    reasoning_file = save_reasoning_analysis(results, run_timestamp)
    
    # 10) Token usage statistics (from last result)
    if results:
        print(f"\nToken Usage (last question):")
        # Note: We'd need to pass the last result object to print_token_usage
        # This would require modifying run_evaluation_loop to return the last result
        print("Token usage statistics not available in current implementation")


if __name__ == "__main__":
    main()