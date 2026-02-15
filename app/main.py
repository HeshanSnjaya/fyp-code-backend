from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
import traceback

from .schemas import (
    AnalyzeRequest, AnalyzeResponse, EvaluateAnswersRequest, FinalResult,
    DetectedPattern, AnswerScore
)
from .github_fetch import clone_repository, collect_python_files, read_source_files, cleanup_old_repositories
from .ast_analyzer import analyze_ast_patterns
from .ml_classifier import classifier
from .quality_scorer import calculate_code_metrics, calculate_quality_score
from .question_generator import generate_questions
from .answer_evaluator import evaluator

app = FastAPI(
    title="Code Evaluation Platform",
    description="AI-powered code evaluation and interview system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML models on startup
@app.on_event("startup")
async def load_models():
    print("🚀 Loading ML models...")
    classifier.load_model()
    evaluator.load_model()
    print("✓ Models loaded successfully")

@app.get("/")
async def root():
    return {
        "message": "Code Evaluation Platform API",
        "status": "running",
        "endpoints": ["/analyze", "/evaluate", "/cleanup"]
    }

@app.post("/cleanup")
async def cleanup_cache():
    """
    Manually cleanup cached repositories.
    Removes old repositories from _repos folder.
    """
    try:
        print("\n🧹 Manual cleanup requested...")
        cleanup_old_repositories(max_age_hours=0)  # Remove all
        return {
            "status": "success",
            "message": "Repository cache cleaned successfully"
        }
    except Exception as e:
        print(f"❌ Cleanup error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_repository(request: AnalyzeRequest):
    """
    STEP 1-8: Analyze GitHub repository and generate questions
    Always clones fresh repository to get latest changes.
    """
    try:
        print(f"\n{'='*60}")
        print(f"📥 Starting analysis: {request.repo_url}")
        print(f"{'='*60}")
        
        # STEP 2: Clone repository and extract Python files (always fresh)
        local_path = clone_repository(request.repo_url)
        py_files = collect_python_files(local_path)
        
        if not py_files:
            raise HTTPException(status_code=400, detail="No Python files found in repository")
        
        sources = read_source_files(py_files)
        print(f"✓ Found {len(sources)} Python files")
        
        # STEP 3-4: AST analysis and rule-based pattern detection
        ast_features = analyze_ast_patterns(sources)
        print(f"✓ AST analysis complete")
        
        # STEP 5: ML-based algorithm classification
        combined_code = "\n\n".join(sources.values())[:5000]  # Limit to first 5000 chars
        algo_label, algo_confidence = classifier.predict(combined_code)
        print(f"✓ Algorithm detected: {algo_label} ({algo_confidence:.2f})")
        
        # STEP 6: Hybrid pattern detection (combine AST + ML)
        patterns = _create_pattern_list(ast_features, algo_label)
        
        # STEP 7: Code quality evaluation
        metrics = calculate_code_metrics(sources, ast_features)
        quality = calculate_quality_score(metrics)
        print(f"✓ Quality score: {quality.score}/10 ({quality.grade})")
        
        # STEP 8: Generate interview questions
        algorithm_pred = type('obj', (object,), {
            'label': algo_label,
            'confidence': algo_confidence
        })()
        questions = generate_questions(patterns, algorithm_pred, quality)
        print(f"✓ Generated {len(questions)} questions")
        
        return AnalyzeResponse(
            repo_url=request.repo_url,
            status="success",
            files_analyzed=len(sources),
            patterns=patterns,
            algorithm={
                "label": algo_label,
                "confidence": algo_confidence,
                "detected_by": "Hybrid"
            },
            quality=quality,
            questions=questions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/evaluate", response_model=FinalResult)
async def evaluate_answers(request: EvaluateAnswersRequest):
    """
    STEP 9-13: Evaluate student answers and generate final score
    """
    try:
        print(f"\n📝 Evaluating answers for {request.repo_url}")
        
        # STEP 10: Evaluate each answer using SBERT
        answer_scores = []
        answer_dict = {a.question_id: a.answer_text for a in request.answers}
        
        for question in request.questions:
            student_answer = answer_dict.get(question.id, "")
            if not student_answer.strip():
                # No answer provided
                score = AnswerScore(
                    question_id=question.id,
                    question_text=question.question,
                    similarity=0.0,
                    marks_obtained=0.0,
                    max_marks=question.max_marks,
                    feedback="No answer provided."
                )
            else:
                score = evaluator.evaluate_answer(question, student_answer)
            
            answer_scores.append(score)
        
        # STEP 11: Calculate final score
        total_answer_marks = sum(s.marks_obtained for s in answer_scores)
        max_answer_marks = sum(s.max_marks for s in answer_scores)
        answer_percentage = (total_answer_marks / max_answer_marks * 100) if max_answer_marks > 0 else 0
        
        # Weighted final score
        code_quality_score = request.quality_score
        algorithm_score = request.algorithm_confidence * 10  # Scale to 0-10
        answer_score = answer_percentage / 10  # Scale to 0-10
        
        final_score = (
            0.4 * code_quality_score +
            0.3 * algorithm_score +
            0.3 * answer_score
        )
        
        # Assign grade
        if final_score >= 9:
            grade = "A"
        elif final_score >= 7.5:
            grade = "B"
        elif final_score >= 6:
            grade = "C"
        elif final_score >= 4:
            grade = "D"
        else:
            grade = "F"
        
        # STEP 12: Generate feedback
        feedback, strengths, improvements = _generate_feedback(
            code_quality_score, algorithm_score, answer_score, answer_scores
        )
        
        print(f"✓ Final score: {final_score:.1f}/10 ({grade})")
        
        return FinalResult(
            answer_scores=answer_scores,
            component_scores={
                "code_quality": round(code_quality_score, 2),
                "algorithm_correctness": round(algorithm_score, 2),
                "answer_evaluation": round(answer_score, 2)
            },
            final_score=round(final_score, 2),
            grade=grade,
            feedback=feedback,
            strengths=strengths,
            improvements=improvements
        )
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

def _create_pattern_list(ast_features: Dict, algo_label: str) -> List[DetectedPattern]:
    """Create pattern detection list from AST features with improved categorization"""
    
    patterns = []
    
    # Algorithm categorization mappings
    sorting_algos = ["Sorting", "merge_sort", "quick_sort", "bubble_sort", "heap_sort", "insertion_sort"]
    searching_algos = ["Searching", "binary_search", "linear_search"]
    
    # Algorithm patterns (5)
    is_sorting = any(algo in algo_label for algo in sorting_algos)
    patterns.append(DetectedPattern(
        name="Sorting",
        present=is_sorting,
        confidence=0.85 if is_sorting else 0.0,
        evidence={"source": "ML Classifier", "algorithm": algo_label} if is_sorting else None
    ))
    
    is_searching = any(algo in algo_label for algo in searching_algos)
    patterns.append(DetectedPattern(
        name="Searching",
        present=is_searching,
        confidence=0.85 if is_searching else 0.0,
        evidence={"source": "ML Classifier", "algorithm": algo_label} if is_searching else None
    ))
    
    patterns.append(DetectedPattern(
        name="Recursion",
        present=ast_features["has_recursion"],
        confidence=1.0,
        evidence={"source": "AST Analysis"} if ast_features["has_recursion"] else None
    ))
    
    is_dp = "Dynamic Programming" in algo_label or "dp" in algo_label.lower()
    patterns.append(DetectedPattern(
        name="Dynamic Programming",
        present=is_dp,
        confidence=0.85 if is_dp else 0.0,
        evidence={"source": "ML Classifier"} if is_dp else None
    ))
    
    is_greedy = "Greedy" in algo_label
    patterns.append(DetectedPattern(
        name="Greedy Algorithms",
        present=is_greedy,
        confidence=0.85 if is_greedy else 0.0,
        evidence={"source": "ML Classifier"} if is_greedy else None
    ))
    
    # Data structure patterns (3)
    patterns.append(DetectedPattern(
        name="Arrays / Lists",
        present=ast_features["has_list"],
        confidence=1.0,
        evidence={"source": "AST Analysis"} if ast_features["has_list"] else None
    ))
    
    patterns.append(DetectedPattern(
        name="Stack / Queue",
        present=ast_features["has_stack_queue"],
        confidence=1.0,
        evidence={"source": "AST Analysis"} if ast_features["has_stack_queue"] else None
    ))
    
    patterns.append(DetectedPattern(
        name="HashMap / Dictionary",
        present=ast_features["has_dict"],
        confidence=1.0,
        evidence={"source": "AST Analysis"} if ast_features["has_dict"] else None
    ))
    
    # Code quality patterns (4)
    patterns.append(DetectedPattern(
        name="Nested loops",
        present=ast_features["max_nested_loops"] >= 2,
        confidence=1.0,
        evidence={"depth": ast_features["max_nested_loops"]} if ast_features["max_nested_loops"] >= 2 else None
    ))
    
    patterns.append(DetectedPattern(
        name="Object-Oriented Programming (classes)",
        present=ast_features["has_classes"],
        confidence=1.0,
        evidence={"source": "AST Analysis"} if ast_features["has_classes"] else None
    ))
    
    patterns.append(DetectedPattern(
        name="Exception handling",
        present=ast_features["has_exceptions"],
        confidence=1.0,
        evidence={"source": "AST Analysis"} if ast_features["has_exceptions"] else None
    ))
    
    patterns.append(DetectedPattern(
        name="Function decomposition",
        present=ast_features["function_decomposition"],
        confidence=1.0,
        evidence={"functions": ast_features["functions_count"]} if ast_features["function_decomposition"] else None
    ))
    
    return patterns

def _generate_feedback(
    quality: float,
    algorithm: float,
    answer: float,
    answer_scores: List[AnswerScore]
) -> tuple[List[str], List[str], List[str]]:
    """Generate personalized feedback, strengths, and improvements"""
    
    feedback = []
    strengths = []
    improvements = []
    
    # Quality feedback
    if quality >= 8:
        strengths.append("Excellent code quality with good structure")
    elif quality >= 6:
        feedback.append("Code quality is acceptable but has room for improvement")
    else:
        improvements.append("Focus on improving code quality: reduce complexity, add comments")
    
    # Algorithm feedback
    if algorithm >= 7:
        strengths.append("Algorithm choice and implementation are solid")
    else:
        improvements.append("Review algorithm fundamentals and implementation patterns")
    
    # Answer feedback
    if answer >= 7:
        strengths.append("Strong theoretical understanding demonstrated in answers")
    elif answer >= 5:
        feedback.append("Decent understanding but answers need more depth")
    else:
        improvements.append("Study core concepts more thoroughly before attempting questions")
    
    # Specific answer feedback
    low_score_questions = [s for s in answer_scores if s.marks_obtained < s.max_marks * 0.5]
    if low_score_questions:
        improvements.append(f"Review topics: {', '.join(set(s.question_text.split()[0:3])[0] for s in low_score_questions[:2])}")
    
    return feedback, strengths, improvements

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
