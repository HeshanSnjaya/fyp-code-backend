from radon.complexity import cc_visit
from radon.metrics import mi_visit
from typing import Dict
import numpy as np
from .schemas import QualityMetrics, QualityScore

def calculate_code_metrics(sources: Dict[str, str], ast_features: Dict) -> QualityMetrics:
    """Calculate code quality metrics using radon"""
    
    total_complexity = 0
    complexity_count = 0
    total_lines = 0
    comment_lines = 0
    function_lengths = []
    
    for filepath, code in sources.items():
        lines = code.split("\n")
        total_lines += len(lines)
        
        # Count comments
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                comment_lines += 1
        
        # Cyclomatic complexity
        try:
            cc_results = cc_visit(code)
            for result in cc_results:
                total_complexity += result.complexity
                complexity_count += 1
                # Function length (lines)
                if hasattr(result, 'endline') and hasattr(result, 'lineno'):
                    function_lengths.append(result.endline - result.lineno)
        except:
            pass
    
    avg_complexity = total_complexity / max(complexity_count, 1)
    comment_ratio = comment_lines / max(total_lines, 1)
    avg_func_length = np.mean(function_lengths) if function_lengths else 0
    
    return QualityMetrics(
        cyclomatic_complexity=round(avg_complexity, 2),
        lines_of_code=total_lines,
        comment_ratio=round(comment_ratio, 3),
        avg_function_length=round(avg_func_length, 1),
        functions_count=ast_features.get("functions_count", 0)
    )

def calculate_quality_score(metrics: QualityMetrics) -> QualityScore:
    """Calculate overall quality score (0-10) using rule-based scoring"""
    
    score = 10.0
    
    # Complexity penalty
    if metrics.cyclomatic_complexity > 10:
        score -= min(2.0, (metrics.cyclomatic_complexity - 10) * 0.2)
    
    # Function length penalty
    if metrics.avg_function_length > 50:
        score -= min(1.5, (metrics.avg_function_length - 50) * 0.02)
    
    # Comment ratio bonus/penalty
    if metrics.comment_ratio < 0.05:
        score -= 1.0
    elif metrics.comment_ratio > 0.15:
        score += 0.5
    
    # Function decomposition bonus
    if metrics.functions_count > 5:
        score += 0.5
    elif metrics.functions_count < 2:
        score -= 1.0
    
    score = max(0.0, min(10.0, score))
    
    # Assign grade
    if score >= 9:
        grade = "A"
    elif score >= 7.5:
        grade = "B"
    elif score >= 6:
        grade = "C"
    elif score >= 4:
        grade = "D"
    else:
        grade = "F"
    
    return QualityScore(
        score=round(score, 1),
        grade=grade,
        metrics=metrics
    )
