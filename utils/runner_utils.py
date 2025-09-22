"""
Main execution orchestration utilities for WTQ evaluation runs.
"""

import datetime
from typing import Dict, List, Any, Optional

from .eval_utils import is_answer_correct
from .results_utils import save_run_results, save_reasoning_analysis
from .table_utils import format_table_token_efficient


def run_evaluation_loop(agent, examples: List[Dict], config: Dict, 
                       run_timestamp: Optional[str] = None) -> List[Dict]:
    """
    Run the main evaluation loop for WTQ questions.
    
    Args:
        agent: The DSPy ReAct agent to use for evaluation
        examples: List of WTQ examples with questions and tables
        config: Configuration dictionary
        run_timestamp: Optional timestamp for this run
    
    Returns:
        List of result dictionaries with evaluation outcomes
    """
    if run_timestamp is None:
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*80}")
    print(f"🧪 TESTING {config['test_questions_limit']} WTQ QUESTIONS")
    print(f"{'='*80}")
    print(f"🕐 Run timestamp: {run_timestamp}")
    
    results = []
    correct_count = 0
    
    for i, example in enumerate(examples, 1):
        print(f"\n--- Question {i}/{len(examples)} ---")
        
        table = example["table"]
        question = example["question"]
        expected_answers = example["answers"]
        
        # Format the table using token-efficient method with size limit
        table_data = format_table_token_efficient(table, question=None, max_rows=500)
        
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
                'tool_calls': 0,
                'reasoning_trajectory': [],
                'tool_selections': []
            })
            continue
        
        # Set the current table data
        from main import set_current_table, get_table_headers_list, get_table_row_count, get_sample_rows
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
        reasoning_trajectory, tool_selections = extract_trajectory_data(result)
        
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
            save_run_results(config, results, current_accuracy, i, correct_count, 
                           is_incremental=True, run_timestamp=run_timestamp)
        except Exception as e:
            print(f"⚠️  Warning: Could not save incremental results: {e}")
        
        # Clear history for next question (if method exists)
        import dspy
        if hasattr(dspy, 'clear_history'):
            dspy.clear_history()
        else:
            # Alternative: reset the LM history if available
            if hasattr(agent, 'lm') and hasattr(agent.lm, 'history'):
                agent.lm.history = []
    
    return results


def extract_trajectory_data(result) -> tuple:
    """
    Extract reasoning trajectory and tool selection data from a DSPy result.
    
    Args:
        result: DSPy result object with trajectory information
    
    Returns:
        Tuple of (reasoning_trajectory, tool_selections) lists
    """
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
    
    return reasoning_trajectory, tool_selections


def print_evaluation_summary(results: List[Dict], examples: List[Dict]):
    """Print a summary of the evaluation results."""
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = correct_count / len(examples) * 100 if examples else 0
    
    print(f"\n{'='*80}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total Questions: {len(examples)}")
    print(f"Correct Answers: {correct_count}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    print(f"\nDetailed Results:")
    for i, result in enumerate(results, 1):
        status = "✅" if result['correct'] else "❌"
        print(f"{i:2d}. {status} Expected: {result['expected']} | Got: {result['predicted']} | Tools: {result['tool_calls']}")
    
    return accuracy, correct_count
