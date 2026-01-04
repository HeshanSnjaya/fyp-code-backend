from typing import List, Dict
from .schemas import InterviewQuestion, DetectedPattern, AlgorithmPrediction, QualityScore
import random

# Question templates organized by pattern
QUESTION_TEMPLATES = {
    "Sorting": [
        {
            "question": "Explain the time complexity of your sorting algorithm in best, average, and worst cases.",
            "difficulty": "medium",
            "keywords": ["time complexity", "O(n log n)", "O(n^2)", "best case", "worst case"]
        },
        {
            "question": "What is the space complexity of your sorting implementation and why?",
            "difficulty": "medium",
            "keywords": ["space complexity", "in-place", "auxiliary space", "O(n)", "O(1)"]
        },
        {
            "question": "Is your sorting algorithm stable? Explain what stability means and why it matters.",
            "difficulty": "hard",
            "keywords": ["stable", "unstable", "equal elements", "relative order"]
        }
    ],
    "Searching": [
        {
            "question": "Explain how your search algorithm works step-by-step.",
            "difficulty": "easy",
            "keywords": ["binary search", "linear search", "divide", "comparison", "target"]
        },
        {
            "question": "What is the time complexity of your search algorithm and what conditions must be met?",
            "difficulty": "medium",
            "keywords": ["O(log n)", "O(n)", "sorted", "prerequisite", "time complexity"]
        }
    ],
    "Recursion": [
        {
            "question": "What is the base case in your recursive function and why is it necessary?",
            "difficulty": "easy",
            "keywords": ["base case", "termination", "prevents infinite", "recursion stops"]
        },
        {
            "question": "Explain the call stack behavior of your recursive function with an example.",
            "difficulty": "hard",
            "keywords": ["call stack", "stack frames", "unwinding", "depth", "memory"]
        },
        {
            "question": "What is the time and space complexity of your recursive solution?",
            "difficulty": "medium",
            "keywords": ["time complexity", "space complexity", "stack depth", "recursive calls"]
        }
    ],
    "Dynamic Programming": [
        {
            "question": "Identify the overlapping subproblems in your solution and explain how memoization helps.",
            "difficulty": "hard",
            "keywords": ["overlapping subproblems", "memoization", "cache", "recomputation", "optimization"]
        },
        {
            "question": "What is the optimal substructure property in your DP solution?",
            "difficulty": "hard",
            "keywords": ["optimal substructure", "subproblem", "optimal solution", "build up"]
        }
    ],
    "Greedy Algorithms": [
        {
            "question": "Explain the greedy choice property and why it works for this problem.",
            "difficulty": "hard",
            "keywords": ["greedy choice", "locally optimal", "globally optimal", "proof"]
        }
    ],
    "Nested loops": [
        {
            "question": "What is the time complexity introduced by your nested loops?",
            "difficulty": "medium",
            "keywords": ["O(n^2)", "O(n^3)", "nested", "quadratic", "cubic"]
        },
        {
            "question": "Can you reduce the nested loop complexity? Suggest an optimization approach.",
            "difficulty": "hard",
            "keywords": ["optimization", "reduce complexity", "hash table", "preprocessing"]
        }
    ],
    "HashMap / Dictionary": [
        {
            "question": "Why did you choose to use a dictionary/hashmap in your solution?",
            "difficulty": "medium",
            "keywords": ["O(1) lookup", "constant time", "key-value", "faster access"]
        }
    ],
    "Object-Oriented Programming (classes)": [
        {
            "question": "Explain the purpose of the classes you defined and their relationships.",
            "difficulty": "medium",
            "keywords": ["encapsulation", "abstraction", "inheritance", "class design"]
        }
    ],
    "Exception handling": [
        {
            "question": "What exceptions are you handling and why is it important?",
            "difficulty": "easy",
            "keywords": ["try-except", "error handling", "robustness", "edge cases"]
        }
    ],
    "Function decomposition": [
        {
            "question": "Explain how you broke down the problem into functions and why.",
            "difficulty": "medium",
            "keywords": ["modularity", "single responsibility", "reusability", "maintainability"]
        }
    ]
}

def generate_questions(
    patterns: List[DetectedPattern],
    algorithm: AlgorithmPrediction,
    quality: QualityScore
) -> List[InterviewQuestion]:
    """Generate 3-5 personalized interview questions based on detected patterns"""
    
    questions = []
    question_id = 1
    
    # Always include algorithm question if detected
    if algorithm.label in QUESTION_TEMPLATES:
        algo_questions = QUESTION_TEMPLATES[algorithm.label]
        selected = random.choice(algo_questions)
        questions.append(InterviewQuestion(
            id=f"q{question_id}",
            pattern=algorithm.label,
            question=selected["question"],
            difficulty=selected["difficulty"],
            expected_keywords=selected["keywords"],
            max_marks=1.0
        ))
        question_id += 1
    
    # Add questions for detected patterns
    for pattern in patterns:
        if pattern.present and pattern.name in QUESTION_TEMPLATES:
            if len(questions) >= 5:
                break
            
            pattern_questions = QUESTION_TEMPLATES[pattern.name]
            selected = random.choice(pattern_questions)
            
            questions.append(InterviewQuestion(
                id=f"q{question_id}",
                pattern=pattern.name,
                question=selected["question"],
                difficulty=selected["difficulty"],
                expected_keywords=selected["keywords"],
                max_marks=1.0
            ))
            question_id += 1
    
    # Ensure minimum 3 questions
    if len(questions) < 3:
        # Add general questions
        general = [
            {
                "pattern": "General",
                "question": "What is the overall time complexity of your solution?",
                "difficulty": "medium",
                "keywords": ["time complexity", "Big O", "analysis"]
            },
            {
                "pattern": "General",
                "question": "What test cases would you write to verify your implementation?",
                "difficulty": "easy",
                "keywords": ["test cases", "edge cases", "validation", "corner cases"]
            }
        ]
        
        for q in general:
            if len(questions) >= 3:
                break
            questions.append(InterviewQuestion(
                id=f"q{question_id}",
                pattern=q["pattern"],
                question=q["question"],
                difficulty=q["difficulty"],
                expected_keywords=q["keywords"],
                max_marks=1.0
            ))
            question_id += 1
    
    return questions[:5]  # Max 5 questions
