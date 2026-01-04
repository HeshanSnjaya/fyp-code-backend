import ast
from typing import Dict, List, Set, Tuple
from collections import defaultdict

class ASTPatternDetector(ast.NodeVisitor):
    """AST visitor to detect patterns in Python code"""
    
    def __init__(self):
        self.functions = []
        self.classes = []
        self.loops = 0
        self.nested_loop_depth = 0
        self.max_nested_depth = 0
        self.recursion_calls = defaultdict(set)
        self.current_function = None
        self.has_try_except = False
        self.dict_usage = False
        self.list_usage = False
        self.stack_queue_patterns = False
        self.loop_depth = 0
        
    def visit_FunctionDef(self, node):
        prev_function = self.current_function
        self.current_function = node.name
        self.functions.append(node.name)
        self.generic_visit(node)
        self.current_function = prev_function
        
    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        self.generic_visit(node)
        
    def visit_For(self, node):
        self.loops += 1
        self.loop_depth += 1
        self.max_nested_depth = max(self.max_nested_depth, self.loop_depth)
        self.generic_visit(node)
        self.loop_depth -= 1
        
    def visit_While(self, node):
        self.loops += 1
        self.loop_depth += 1
        self.max_nested_depth = max(self.max_nested_depth, self.loop_depth)
        self.generic_visit(node)
        self.loop_depth -= 1
        
    def visit_Call(self, node):
        if self.current_function and isinstance(node.func, ast.Name):
            if node.func.id == self.current_function:
                self.recursion_calls[self.current_function].add(self.current_function)
        self.generic_visit(node)
        
    def visit_Try(self, node):
        self.has_try_except = True
        self.generic_visit(node)
        
    def visit_Dict(self, node):
        self.dict_usage = True
        self.generic_visit(node)
        
    def visit_List(self, node):
        self.list_usage = True
        self.generic_visit(node)
        
    def visit_Attribute(self, node):
        # Detect stack/queue operations
        if isinstance(node.attr, str):
            if node.attr in {"append", "pop", "popleft"}:
                self.stack_queue_patterns = True
        self.generic_visit(node)

def analyze_ast_patterns(sources: Dict[str, str]) -> Dict:
    """Analyze code using AST and return pattern detection results"""
    
    all_detectors = []
    total_functions = 0
    total_loops = 0
    max_nested = 0
    has_recursion = False
    has_classes = False
    has_exceptions = False
    has_dicts = False
    has_lists = False
    has_stack_queue = False
    
    for filepath, code in sources.items():
        try:
            tree = ast.parse(code)
            detector = ASTPatternDetector()
            detector.visit(tree)
            all_detectors.append(detector)
            
            total_functions += len(detector.functions)
            total_loops += detector.loops
            max_nested = max(max_nested, detector.max_nested_depth)
            
            if detector.recursion_calls:
                has_recursion = True
            if detector.classes:
                has_classes = True
            if detector.has_try_except:
                has_exceptions = True
            if detector.dict_usage:
                has_dicts = True
            if detector.list_usage:
                has_lists = True
            if detector.stack_queue_patterns:
                has_stack_queue = True
                
        except SyntaxError:
            continue
    
    return {
        "functions_count": total_functions,
        "loops_count": total_loops,
        "max_nested_loops": max_nested,
        "has_recursion": has_recursion,
        "has_classes": has_classes,
        "has_exceptions": has_exceptions,
        "has_dict": has_dicts,
        "has_list": has_lists,
        "has_stack_queue": has_stack_queue,
        "function_decomposition": total_functions > 3
    }
