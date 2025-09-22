"""
Results saving and analysis utilities for WTQ evaluation runs.
"""

import datetime
import json
import os
from typing import Dict, List, Any, Optional


def save_run_results(config: Dict, results: List[Dict], accuracy: float, 
                    total_questions: int, correct_count: int, 
                    is_incremental: bool = False, run_timestamp: Optional[str] = None) -> tuple:
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


def save_reasoning_analysis(results: List[Dict], run_timestamp: Optional[str] = None) -> str:
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
