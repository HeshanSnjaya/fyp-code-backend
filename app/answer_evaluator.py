from sentence_transformers import SentenceTransformer, util
from typing import List, Dict
from .schemas import InterviewQuestion, AnswerItem, AnswerScore
import torch

class AnswerEvaluator:
    """SBERT-based answer evaluator"""
    
    def __init__(self):
        self.model = None
        self.loaded = False
        
    def load_model(self):
        """Load Sentence-BERT model"""
        if self.loaded:
            return
        
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.loaded = True
            print("✓ SBERT model loaded")
        except Exception as e:
            print(f"⚠ Could not load SBERT model: {e}")
    
    def evaluate_answer(
        self,
        question: InterviewQuestion,
        student_answer: str
    ) -> AnswerScore:
        """Evaluate a single answer using semantic similarity"""
        
        if not self.loaded or not self.model:
            return self._fallback_evaluate(question, student_answer)
        
        try:
            # Create expected answer from keywords
            expected_text = " ".join(question.expected_keywords)
            
            # Compute embeddings
            emb1 = self.model.encode(student_answer, convert_to_tensor=True)
            emb2 = self.model.encode(expected_text, convert_to_tensor=True)
            
            # Cosine similarity
            similarity = util.cos_sim(emb1, emb2).item()
            similarity = max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
            
            # Apply scoring rules
            if similarity >= 0.75:
                marks = question.max_marks
                feedback = "Excellent answer! Shows strong understanding."
            elif similarity >= 0.5:
                marks = question.max_marks * 0.6
                feedback = "Good answer, but could be more detailed."
            else:
                marks = question.max_marks * 0.2
                feedback = "Answer needs improvement. Review key concepts."
            
            return AnswerScore(
                question_id=question.id,
                question_text=question.question,
                similarity=round(similarity, 3),
                marks_obtained=round(marks, 2),
                max_marks=question.max_marks,
                feedback=feedback
            )
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return self._fallback_evaluate(question, student_answer)
    
    def _fallback_evaluate(
        self,
        question: InterviewQuestion,
        student_answer: str
    ) -> AnswerScore:
        """Simple keyword-based fallback evaluation"""
        
        answer_lower = student_answer.lower()
        keywords_found = sum(1 for kw in question.expected_keywords if kw.lower() in answer_lower)
        total_keywords = len(question.expected_keywords)
        
        similarity = keywords_found / max(total_keywords, 1)
        
        if similarity >= 0.6:
            marks = question.max_marks
            feedback = "Good answer with relevant keywords."
        elif similarity >= 0.3:
            marks = question.max_marks * 0.6
            feedback = "Partial answer. Include more details."
        else:
            marks = question.max_marks * 0.2
            feedback = "Answer lacks key concepts."
        
        return AnswerScore(
            question_id=question.id,
            question_text=question.question,
            similarity=round(similarity, 3),
            marks_obtained=round(marks, 2),
            max_marks=question.max_marks,
            feedback=feedback
        )

# Global evaluator instance
evaluator = AnswerEvaluator()
