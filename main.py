import dspy
import json
import datetime
import os
from utils import load_wtq_test_questions_with_tables, print_token_usage
from utils.table_utils import format_table_token_efficient


# ============================================================================
# SIMPLIFIED EVALUATION
# ============================================================================

def normalize_answer(text):
    """Normalize text for comparison - simplified version of evaluator.py normalize function."""
    if not text:
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower().strip()
    
    # Remove common punctuation and formatting
    import re
    # Remove commas from numbers (100,000 -> 100000)
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    # Remove trailing periods
    text = text.rstrip('.')
    # Normalize dashes (en-dash, em-dash to regular dash)
    text = re.sub(r'[–—−]', '-', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def is_answer_correct(predicted, expected_answers):
    """Check if predicted answer matches any expected answer using simplified evaluation."""
    if not predicted or not expected_answers:
        return False
    
    # Normalize predicted answer
    pred_norm = normalize_answer(predicted)
    
    # Handle "I don't know" responses
    if "i don't know" in pred_norm or "don't know" in pred_norm:
        # Check if any expected answer is also "I don't know" or similar
        for expected in expected_answers:
            exp_norm = normalize_answer(expected)
            if "i don't know" in exp_norm or "don't know" in exp_norm or "unknown" in exp_norm:
                return True
        # If expected answers are not "I don't know", then "I don't know" is incorrect
        return False
    
    # Check against each expected answer
    for expected in expected_answers:
        exp_norm = normalize_answer(expected)
        
        # Direct match
        if pred_norm == exp_norm:
            return True
        
        # Try to parse as numbers and compare
        try:
            pred_num = float(pred_norm.replace(',', ''))
            exp_num = float(exp_norm.replace(',', ''))
            if abs(pred_num - exp_num) < 1e-6:
                return True
        except (ValueError, TypeError):
            pass
    
    return False

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
# RESULTS SAVING
# ============================================================================

def save_run_results(config, results, accuracy, total_questions, correct_count, is_incremental=False, run_timestamp=None):
    """Save run results and configuration to a JSON file for analysis."""
    
    # Create results directory if it doesn't exist
    results_dir = "run_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Use provided timestamp or generate new one
    if run_timestamp is None:
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if is_incremental:
        # For incremental saves, overwrite the same file
        filename = f"{results_dir}/run_{run_timestamp}_incremental.json"
    else:
        # For final save, use the main filename
        filename = f"{results_dir}/run_{run_timestamp}_final.json"
    
    # Prepare the complete run data
    run_data = {
        "timestamp": run_timestamp,
        "datetime": datetime.datetime.now().isoformat(),
        "config": config,
        "summary": {
            "total_questions": total_questions,
            "correct_count": correct_count,
            "accuracy": accuracy,
            "accuracy_percentage": f"{accuracy:.1f}%",
            "is_incremental": is_incremental
        },
        "detailed_results": results
    }
    
    # Save to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(run_data, f, indent=2, ensure_ascii=False)
    
    if not is_incremental:
        print(f"📁 Results saved to: {filename}")
    return filename, run_timestamp

def save_reasoning_analysis(results, run_timestamp=None):
    """Save detailed reasoning trajectory and tool selection analysis to a separate JSON file."""
    
    # Create results directory if it doesn't exist
    results_dir = "run_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Use provided timestamp or generate new one
    if run_timestamp is None:
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"{results_dir}/reasoning_analysis_{run_timestamp}.json"
    
    # Extract all reasoning trajectories and tool selections
    all_reasoning = []
    all_tool_selections = []
    tool_usage_stats = {}
    
    for i, result in enumerate(results, 1):
        question_id = i
        
        # Process reasoning trajectory
        for reasoning_step in result.get('reasoning_trajectory', []):
            all_reasoning.append({
                'question_id': question_id,
                'question': result['question'],
                'step': reasoning_step['step'],
                'reasoning': reasoning_step['reasoning'],
                'correct': result['correct']
            })
        
        # Process tool selections
        for tool_step in result.get('tool_selections', []):
            all_tool_selections.append({
                'question_id': question_id,
                'question': result['question'],
                'step': tool_step['step'],
                'tool_name': tool_step['tool_name'],
                'tool_input': tool_step['tool_input'],
                'tool_output': tool_step['tool_output'],
                'correct': result['correct']
            })
            
            # Track tool usage statistics
            tool_name = tool_step['tool_name']
            if tool_name not in tool_usage_stats:
                tool_usage_stats[tool_name] = {
                    'total_uses': 0,
                    'correct_questions': 0,
                    'incorrect_questions': 0,
                    'questions_used': set()
                }
            
            tool_usage_stats[tool_name]['total_uses'] += 1
            tool_usage_stats[tool_name]['questions_used'].add(question_id)
            if result['correct']:
                tool_usage_stats[tool_name]['correct_questions'] += 1
            else:
                tool_usage_stats[tool_name]['incorrect_questions'] += 1
    
    # Convert sets to counts for JSON serialization
    for tool_name, stats in tool_usage_stats.items():
        stats['unique_questions'] = len(stats['questions_used'])
        del stats['questions_used']  # Remove set for JSON serialization
    
    # Prepare the analysis data
    analysis_data = {
        "timestamp": run_timestamp,
        "datetime": datetime.datetime.now().isoformat(),
        "summary": {
            "total_questions": len(results),
            "total_reasoning_steps": len(all_reasoning),
            "total_tool_calls": len(all_tool_selections),
            "unique_tools_used": len(tool_usage_stats)
        },
        "tool_usage_statistics": tool_usage_stats,
        "reasoning_trajectory": all_reasoning,
        "tool_selections": all_tool_selections
    }
    
    # Save to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    print(f"🧠 Reasoning analysis saved to: {filename}")
    return filename

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function for the WTQ table question answering system."""
    
    # 1) Configure DSPy with LM Studio
    print("🔧 Configuring DSPy with LM Studio...")
    
    # Configuration parameters
    config = {
        "model": "openai/qwen/qwen3-4b-2507",
        "api_base": "http://10.1.11.218:1234/v1",
        "api_key": "local",
        "model_type": "chat",
        "temperature": 0.0,
        "max_tokens": 16384,
        "track_usage": True,
        "disk_cache": False,
        "memory_cache": False,
        "max_iters": 5,
        "tools": ["count_column_values_tool", "count_column_contains_tool", "get_row_by_condition_tool"],
        "signature_fields": ["question", "table_headers", "total_columns", "total_rows", "sample_rows", "answer"],
        "evaluation_method": "simplified_normalization",
        "sample_rows_count": 5,
        "max_table_size": 10000,
        "max_table_rows": 500,
        "test_questions_limit": 500
    }
    
    lm = dspy.LM(
        config["model"],
        api_base=config["api_base"],
        api_key=config["api_key"],
        model_type=config["model_type"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )
    dspy.configure(lm=lm)

    # Configure usage tracking and cache
    dspy.settings.configure(track_usage=config["track_usage"])
    dspy.configure_cache(enable_disk_cache=config["disk_cache"], enable_memory_cache=config["memory_cache"])

    print(f"✅ DSPy configured with LM Studio model: {config['model']} at {config['api_base']}")

    # 1.5) Smoke test to verify DSPy config is working
    print("🧪 Running smoke test...")
    try:
        smoke_response = lm("Say 'Hello from LM Studio!' in exactly 3 words.")
        print(f"✅ Smoke test passed: {smoke_response}")
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        exit(1)

    # 2) Load WTQ test examples
    print("📊 Loading WTQ test examples...")
    examples = load_wtq_test_questions_with_tables(limit=config["test_questions_limit"])
    print(f"Loaded {len(examples)} examples")

    # 3) Create a React agent with table tools
    print("🤖 Creating ReAct agent...")
    agent = create_table_question_answerer()

    # 4) Test with multiple questions
    print(f"\n{'='*80}")
    print(f"🧪 TESTING {config['test_questions_limit']} WTQ QUESTIONS")
    print(f"{'='*80}")
    
    # Generate a single timestamp for this entire run
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🕐 Run timestamp: {run_timestamp}")
    
    results = []
    correct_count = 0
    
    for i, example in enumerate(examples, 1):
        print(f"\n--- Question {i}/{len(examples)} ---")
        
        table = example["table"]
        question = example["question"]
        expected_answers = example["answers"]
        
        # Format the table using token-efficient method with size limit
        table_data = format_table_token_efficient(table, question=None, max_rows=500)  # Reduced from 1000
        
        print(f"Question: {question}")
        print(f"Expected Answers: {expected_answers}")
        print(f"Table size: {len(table_data)} characters")
        
        # Skip questions with very large tables to avoid errors
        if len(table_data) > 10000:  # 10KB limit
            print("⚠️  Skipping question due to large table size")
            results.append({
                'question': question,
                'expected': expected_answers,
                'predicted': "SKIPPED (table too large)",
                'correct': False,
                'tool_calls': 0
            })
            continue
        
        # Set the current table data
        set_current_table(table_data)
        
        # Get table information in structured format
        headers = get_table_headers_list()
        total_columns = len(headers)
        total_rows = get_table_row_count()
        sample_rows = get_sample_rows(5)  # Get first 5 rows for context
        
        # Ask the question with error handling
        print("🤖 Agent reasoning...")
        try:
            result = agent(
                question=question, 
                table_headers=str(headers), 
                total_columns=total_columns, 
                total_rows=total_rows, 
                sample_rows=sample_rows
            )
            predicted_answer = result.answer.strip()
        except Exception as e:
            print(f"❌ Error processing question: {str(e)[:100]}...")
            predicted_answer = "ERROR"
            result = type('Result', (), {'answer': 'ERROR', 'trajectory': []})()
        
        # Check if answer is correct using simplified evaluation
        is_correct = is_answer_correct(predicted_answer, expected_answers)
        if is_correct:
            correct_count += 1
            status = "✅ CORRECT"
        else:
            status = "❌ INCORRECT"
        
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Status: {status}")
        print(f"Tool calls: {len(result.trajectory) if hasattr(result, 'trajectory') else 0}")
        
        # Extract reasoning trajectory and tool selection data
        reasoning_trajectory = []
        tool_selections = []
        
        if hasattr(result, 'trajectory') and result.trajectory:
            trajectory = result.trajectory
            
            # Handle dictionary-based trajectory (DSPy ReAct format)
            if isinstance(trajectory, dict):
                # Find all step numbers by looking for thought_* keys
                step_numbers = set()
                for key in trajectory.keys():
                    if key.startswith('thought_'):
                        step_num = key.split('_')[1]
                        step_numbers.add(step_num)
                
                # Process each step
                for step_num in sorted(step_numbers, key=int):
                    thought_key = f'thought_{step_num}'
                    tool_name_key = f'tool_name_{step_num}'
                    tool_args_key = f'tool_args_{step_num}'
                    observation_key = f'observation_{step_num}'
                    
                    # Extract reasoning/thought
                    if thought_key in trajectory and trajectory[thought_key]:
                        reasoning_trajectory.append({
                            'step': int(step_num) + 1,
                            'reasoning': trajectory[thought_key],
                            'type': 'reasoning'
                        })
                    
                    # Extract tool call
                    if tool_name_key in trajectory and trajectory[tool_name_key]:
                        tool_selections.append({
                            'step': int(step_num) + 1,
                            'tool_name': trajectory[tool_name_key],
                            'tool_input': trajectory.get(tool_args_key, {}),
                            'tool_output': trajectory.get(observation_key, ''),
                            'type': 'tool_selection'
                        })
            
            # Handle list-based trajectory (alternative format)
            elif isinstance(trajectory, list):
                for i, step in enumerate(trajectory):
                    # Try different possible attribute names for reasoning/thoughts
                    reasoning_text = None
                    if hasattr(step, 'thought') and step.thought:
                        reasoning_text = step.thought
                    elif hasattr(step, 'reasoning') and step.reasoning:
                        reasoning_text = step.reasoning
                    elif hasattr(step, 'rationale') and step.rationale:
                        reasoning_text = step.rationale
                    elif hasattr(step, 'reason') and step.reason:
                        reasoning_text = step.reason
                    
                    if reasoning_text:
                        reasoning_trajectory.append({
                            'step': i + 1,
                            'reasoning': reasoning_text,
                            'type': 'reasoning'
                        })
                    
                    # Try different possible attribute names for tool calls
                    tool_name = None
                    tool_input = None
                    tool_output = None
                    
                    if hasattr(step, 'action') and step.action:
                        tool_name = step.action
                        tool_input = getattr(step, 'action_input', '')
                        tool_output = getattr(step, 'observation', '')
                    elif hasattr(step, 'tool') and step.tool:
                        tool_name = step.tool
                        tool_input = getattr(step, 'tool_input', '')
                        tool_output = getattr(step, 'tool_output', '')
                    elif hasattr(step, 'function') and step.function:
                        tool_name = step.function
                        tool_input = getattr(step, 'function_input', '')
                        tool_output = getattr(step, 'function_output', '')
                    
                    if tool_name:
                        tool_selections.append({
                            'step': i + 1,
                            'tool_name': tool_name,
                            'tool_input': tool_input,
                            'tool_output': tool_output,
                            'type': 'tool_selection'
                        })
        
        # Store result for summary
        result_data = {
            'question': question,
            'expected': expected_answers,
            'predicted': predicted_answer,
            'correct': is_correct,
            'tool_calls': len(result.trajectory) if hasattr(result, 'trajectory') else 0,
            'reasoning_trajectory': reasoning_trajectory,
            'tool_selections': tool_selections
        }
        results.append(result_data)
        
        # Save incremental results after each question
        current_accuracy = (correct_count / i) * 100
        try:
            save_run_results(config, results, current_accuracy, i, correct_count, is_incremental=True, run_timestamp=run_timestamp)
        except Exception as e:
            print(f"⚠️  Warning: Could not save incremental results: {e}")
        
        # Clear history for next question (if method exists)
        if hasattr(dspy, 'clear_history'):
            dspy.clear_history()
        else:
            # Alternative: reset the LM history if available
            if hasattr(lm, 'history'):
                lm.history = []

    # 5) Summary statistics
    print(f"\n{'='*80}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total Questions: {len(examples)}")
    print(f"Correct Answers: {correct_count}")
    accuracy = correct_count/len(examples)*100
    print(f"Accuracy: {accuracy:.1f}%")
    
    print(f"\nDetailed Results:")
    for i, result in enumerate(results, 1):
        status = "✅" if result['correct'] else "❌"
        print(f"{i:2d}. {status} Expected: {result['expected']} | Got: {result['predicted']} | Tools: {result['tool_calls']}")
    
    # 6) Save final results for analysis
    print(f"\n{'='*80}")
    print("💾 SAVING FINAL RESULTS")
    print(f"{'='*80}")
    saved_file, _ = save_run_results(config, results, accuracy, len(examples), correct_count, is_incremental=False, run_timestamp=run_timestamp)
    
    # 6.5) Save detailed reasoning analysis
    print(f"\n{'='*80}")
    print("🧠 SAVING REASONING ANALYSIS")
    print(f"{'='*80}")
    reasoning_file = save_reasoning_analysis(results, run_timestamp)
    
    # 7) Token usage statistics (from last result)
    if results:
        print(f"\nToken Usage (last question):")
        print_token_usage(result)


if __name__ == "__main__":
    main()
