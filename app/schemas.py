from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Optional

class AnalyzeRequest(BaseModel):
    repo_url: str = Field(..., examples=["https://github.com/username/repo"])

class DetectedPattern(BaseModel):
    name: str
    present: bool
    confidence: float = 1.0
    evidence: Optional[Dict] = None

class AlgorithmPrediction(BaseModel):
    label: str
    confidence: float
    detected_by: str  # "ML" or "Rule-based" or "Hybrid"

class QualityMetrics(BaseModel):
    cyclomatic_complexity: float
    lines_of_code: int
    comment_ratio: float
    avg_function_length: float
    functions_count: int

class QualityScore(BaseModel):
    score: float  # 0-10
    grade: str  # A-F
    metrics: QualityMetrics

class InterviewQuestion(BaseModel):
    id: str
    pattern: str
    question: str
    difficulty: str  # "easy", "medium", "hard"
    expected_keywords: List[str]
    max_marks: float = 1.0

class AnalyzeResponse(BaseModel):
    repo_url: str
    status: str
    files_analyzed: int
    patterns: List[DetectedPattern]
    algorithm: AlgorithmPrediction
    quality: QualityScore
    questions: List[InterviewQuestion]

class AnswerItem(BaseModel):
    question_id: str
    answer_text: str

class EvaluateAnswersRequest(BaseModel):
    repo_url: str
    algorithm_confidence: float
    quality_score: float
    questions: List[InterviewQuestion]
    answers: List[AnswerItem]

class AnswerScore(BaseModel):
    question_id: str
    question_text: str
    similarity: float
    marks_obtained: float
    max_marks: float
    feedback: str

class FinalResult(BaseModel):
    answer_scores: List[AnswerScore]
    component_scores: Dict[str, float]
    final_score: float
    grade: str
    feedback: List[str]
    strengths: List[str]
    improvements: List[str]
