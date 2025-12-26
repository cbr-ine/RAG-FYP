"""
Reasoning Module with LLM Integration for RAG System
Enhanced version that uses LLM for intelligent question decomposition
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import re
import os
from abc import ABC, abstractmethod


class QuestionType(Enum):
    """Types of questions for classification"""
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    AGGREGATION = "aggregation"
    COMPOSITIONAL = "compositional"


class DependencyType(Enum):
    """Types of dependencies between sub-questions"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


@dataclass
class SubQuestion:
    """Represents a sub-question in the decomposition"""
    id: str
    question: str
    question_type: QuestionType
    dependencies: List[str]
    dependency_type: DependencyType
    priority: int
    context_required: bool = False
    reasoning: Optional[str] = None  # Why this sub-question is needed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "question_type": self.question_type.value,
            "dependencies": self.dependencies,
            "dependency_type": self.dependency_type.value,
            "priority": self.priority,
            "context_required": self.context_required,
            "reasoning": self.reasoning
        }


@dataclass
class ReasoningPlan:
    """Represents the complete reasoning plan for a complex question"""
    original_question: str
    sub_questions: List[SubQuestion]
    execution_order: List[List[str]]
    reasoning_strategy: str
    confidence_score: Optional[float] = None  # LLM's confidence in this plan
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_question": self.original_question,
            "sub_questions": [sq.to_dict() for sq in self.sub_questions],
            "execution_order": self.execution_order,
            "reasoning_strategy": self.reasoning_strategy,
            "confidence_score": self.confidence_score
        }


# ============================================================================
# LLM Client Abstraction Layer
# ============================================================================

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Generate text from the LLM"""
        pass
    
    @abstractmethod
    def generate_json(self, prompt: str, temperature: float = 0.3) -> Dict[str, Any]:
        """Generate structured JSON output from the LLM"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client wrapper"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        try:
            import openai
            self.openai = openai
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.model = model
            
            if not self.api_key:
                raise ValueError("OpenAI API key not provided")
            
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def generate_json(self, prompt: str, temperature: float = 0.3) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client wrapper"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        try:
            import anthropic
            self.anthropic = anthropic
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            self.model = model
            
            if not self.api_key:
                raise ValueError("Anthropic API key not provided")
            
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    
    def generate_json(self, prompt: str, temperature: float = 0.3) -> Dict[str, Any]:
        prompt_with_json = f"{prompt}\n\nProvide your response as valid JSON only, with no additional text."
        response = self.generate(prompt_with_json, temperature=temperature)
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(response)


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing without API calls"""
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        return "Mock response: This is a simulated LLM response."
    
    def generate_json(self, prompt: str, temperature: float = 0.3) -> Dict[str, Any]:
        return {
            "question_type": "comparative",
            "sub_questions": [
                {"id": "sq_0", "question": "Mock sub-question 1", "priority": 0},
                {"id": "sq_1", "question": "Mock sub-question 2", "priority": 1}
            ],
            "reasoning_strategy": "mock_strategy"
        }


# ============================================================================
# Enhanced Reasoning Module with LLM Integration
# ============================================================================

class ReasoningModule:
    """
    Enhanced reasoning module with LLM integration for intelligent question decomposition.
    Falls back to rule-based approach if LLM is not available.
    """
    
    def __init__(self, llm_client: Optional[BaseLLMClient] = None, use_llm: bool = True):
        """
        Initialize the reasoning module.
        
        Args:
            llm_client: LLM client instance (OpenAI, Anthropic, etc.)
            use_llm: Whether to use LLM for decomposition (if False, uses rule-based)
        """
        self.llm_client = llm_client
        self.use_llm = use_llm and llm_client is not None
        
        # Rule-based decomposition strategies (fallback)
        self.rule_based_strategies = {
            QuestionType.COMPARATIVE: self._decompose_comparative_rule,
            QuestionType.CAUSAL: self._decompose_causal_rule,
            QuestionType.TEMPORAL: self._decompose_temporal_rule,
            QuestionType.AGGREGATION: self._decompose_aggregation_rule,
            QuestionType.COMPOSITIONAL: self._decompose_compositional_rule,
        }
    
    def decompose_question(self, question: str, use_llm: Optional[bool] = None) -> ReasoningPlan:
        """
        Main interface: Decompose a complex question into sub-questions.
        
        Args:
            question: The original complex question
            use_llm: Override the default LLM usage setting
            
        Returns:
            ReasoningPlan containing sub-questions and execution order
        """
        should_use_llm = use_llm if use_llm is not None else self.use_llm
        
        if should_use_llm:
            try:
                return self._decompose_with_llm(question)
            except Exception as e:
                print(f"LLM decomposition failed: {e}. Falling back to rule-based.")
                return self._decompose_rule_based(question)
        else:
            return self._decompose_rule_based(question)
    
    # ========================================================================
    # LLM-based Decomposition
    # ========================================================================
    
    def _decompose_with_llm(self, question: str) -> ReasoningPlan:
        """Use LLM to decompose the question intelligently"""
        
        # Step 1: Classify question type with LLM
        question_type = self._classify_with_llm(question)
        
        # Step 2: Generate decomposition with LLM
        decomposition_prompt = self._build_decomposition_prompt(question, question_type)
        llm_response = self.llm_client.generate_json(decomposition_prompt, temperature=0.3)
        
        # Step 3: Parse LLM response into SubQuestion objects
        sub_questions = self._parse_llm_decomposition(llm_response)
        
        # Step 4: Determine execution order
        execution_order = self._determine_execution_order(sub_questions)
        
        # Step 5: Create reasoning plan
        reasoning_plan = ReasoningPlan(
            original_question=question,
            sub_questions=sub_questions,
            execution_order=execution_order,
            reasoning_strategy=f"llm_{question_type.value}",
            confidence_score=llm_response.get("confidence", None)
        )
        
        return reasoning_plan
    
    def _classify_with_llm(self, question: str) -> QuestionType:
        """Use LLM to classify the question type"""
        
        classification_prompt = f"""Analyze the following question and classify its type.

Question: "{question}"

Available question types:
- FACTUAL: Simple factual questions that can be answered directly
- COMPARATIVE: Questions comparing two or more entities
- CAUSAL: Questions about causes, reasons, or why something happens
- TEMPORAL: Questions about timelines, history, or chronological order
- AGGREGATION: Questions requiring counting, listing, or aggregating information
- COMPOSITIONAL: Questions with multiple independent parts

Respond with a JSON object in this format:
{{
    "question_type": "<one of the types above>",
    "confidence": <0.0 to 1.0>,
    "reasoning": "<brief explanation>"
}}"""

        response = self.llm_client.generate_json(classification_prompt)
        
        try:
            qtype_str = response["question_type"].upper()
            return QuestionType[qtype_str]
        except (KeyError, ValueError):
            # Fallback to rule-based classification
            return self._classify_question_rule(question)
    
    def _build_decomposition_prompt(self, question: str, question_type: QuestionType) -> str:
        """Build the prompt for LLM-based decomposition"""
        
        prompt = f"""You are an expert at breaking down complex questions into simpler sub-questions for a Retrieval-Augmented Generation (RAG) system.

Original Question: "{question}"
Question Type: {question_type.value}

Your task: Decompose this question into logical sub-questions that:
1. Are simpler and more focused than the original
2. Can be answered by retrieving specific information
3. Build upon each other logically
4. Together provide all information needed to answer the original question

For each sub-question, specify:
- The question text
- Dependencies (which other sub-questions must be answered first)
- Priority (execution order, lower numbers execute first)
- Whether it requires context from previous answers

Provide your response as JSON in this exact format:
{{
    "sub_questions": [
        {{
            "id": "sq_0",
            "question": "<sub-question text>",
            "dependencies": [],
            "priority": 0,
            "context_required": false,
            "reasoning": "<why this sub-question is needed>"
        }}
    ],
    "reasoning_strategy": "{question_type.value}",
    "confidence": 0.95
}}

Guidelines:
- Create 2-5 sub-questions (not too many)
- Dependencies are IDs of other sub-questions (e.g., ["sq_0", "sq_1"])
- Priority should reflect logical order (0 = first, higher = later)
- Set context_required=true if the sub-question needs answers from dependencies
"""
        
        return prompt
    
    def _parse_llm_decomposition(self, llm_response: Dict[str, Any]) -> List[SubQuestion]:
        """Parse LLM JSON response into SubQuestion objects"""
        
        sub_questions = []
        
        for sq_data in llm_response.get("sub_questions", []):
            # Determine question type (default to FACTUAL for sub-questions)
            qtype_str = sq_data.get("question_type", "FACTUAL").upper()
            try:
                qtype = QuestionType[qtype_str]
            except (KeyError, ValueError):
                qtype = QuestionType.FACTUAL
            
            # Determine dependency type based on dependencies
            dependencies = sq_data.get("dependencies", [])
            if len(dependencies) == 0:
                dep_type = DependencyType.PARALLEL
            elif sq_data.get("context_required", False):
                dep_type = DependencyType.SEQUENTIAL
            else:
                dep_type = DependencyType.CONDITIONAL
            
            sub_question = SubQuestion(
                id=sq_data.get("id", f"sq_{len(sub_questions)}"),
                question=sq_data["question"],
                question_type=qtype,
                dependencies=dependencies,
                dependency_type=dep_type,
                priority=sq_data.get("priority", len(sub_questions)),
                context_required=sq_data.get("context_required", False),
                reasoning=sq_data.get("reasoning", None)
            )
            
            sub_questions.append(sub_question)
        
        return sub_questions
    
    # ========================================================================
    # Rule-based Decomposition (Fallback)
    # ========================================================================
    
    def _decompose_rule_based(self, question: str) -> ReasoningPlan:
        """Rule-based decomposition (original implementation)"""
        
        question_type = self._classify_question_rule(question)
        
        if question_type in self.rule_based_strategies:
            sub_questions = self.rule_based_strategies[question_type](question)
        else:
            sub_questions = [SubQuestion(
                id="sq_0",
                question=question,
                question_type=QuestionType.FACTUAL,
                dependencies=[],
                dependency_type=DependencyType.PARALLEL,
                priority=0
            )]
        
        execution_order = self._determine_execution_order(sub_questions)
        
        reasoning_plan = ReasoningPlan(
            original_question=question,
            sub_questions=sub_questions,
            execution_order=execution_order,
            reasoning_strategy=f"rule_{question_type.value}"
        )
        
        return reasoning_plan
    
    def _classify_question_rule(self, question: str) -> QuestionType:
        """Rule-based question classification"""
        question_lower = question.lower()
        
        comparative_keywords = ['compare', 'difference between', 'versus', 'vs', 
                               'better than', 'worse than', 'similar', 'contrast']
        if any(kw in question_lower for kw in comparative_keywords):
            return QuestionType.COMPARATIVE
        
        causal_keywords = ['why', 'reason', 'cause', 'because', 'lead to', 
                          'result in', 'how does', 'what makes']
        if any(kw in question_lower for kw in causal_keywords):
            return QuestionType.CAUSAL
        
        temporal_keywords = ['when', 'timeline', 'history', 'evolution', 
                           'over time', 'chronological', 'sequence']
        if any(kw in question_lower for kw in temporal_keywords):
            return QuestionType.TEMPORAL
        
        aggregation_keywords = ['how many', 'total', 'sum', 'average', 
                               'count', 'all', 'list all']
        if any(kw in question_lower for kw in aggregation_keywords):
            return QuestionType.AGGREGATION
        
        if ' and ' in question_lower and '?' not in question_lower[:-1]:
            return QuestionType.COMPOSITIONAL
        
        return QuestionType.FACTUAL
    
    # Rule-based decomposition methods (same as before)
    def _decompose_comparative_rule(self, question: str) -> List[SubQuestion]:
        entities = self._extract_comparison_entities(question)
        sub_questions = []
        
        for i, entity in enumerate(entities):
            sub_questions.append(SubQuestion(
                id=f"sq_{i}",
                question=f"What are the characteristics/properties of {entity}?",
                question_type=QuestionType.FACTUAL,
                dependencies=[],
                dependency_type=DependencyType.PARALLEL,
                priority=0,
                context_required=False
            ))
        
        sub_questions.append(SubQuestion(
            id=f"sq_{len(entities)}",
            question=f"Compare the characteristics of {' and '.join(entities)}",
            question_type=QuestionType.FACTUAL,
            dependencies=[f"sq_{i}" for i in range(len(entities))],
            dependency_type=DependencyType.SEQUENTIAL,
            priority=1,
            context_required=True
        ))
        
        return sub_questions
    
    def _decompose_causal_rule(self, question: str) -> List[SubQuestion]:
        phenomenon = self._extract_causal_phenomenon(question)
        return [
            SubQuestion(
                id="sq_0",
                question=f"What is {phenomenon}?",
                question_type=QuestionType.FACTUAL,
                dependencies=[],
                dependency_type=DependencyType.PARALLEL,
                priority=0
            ),
            SubQuestion(
                id="sq_1",
                question=f"What are the direct causes of {phenomenon}?",
                question_type=QuestionType.FACTUAL,
                dependencies=["sq_0"],
                dependency_type=DependencyType.SEQUENTIAL,
                priority=1
            ),
            SubQuestion(
                id="sq_2",
                question=f"What are the underlying factors contributing to {phenomenon}?",
                question_type=QuestionType.FACTUAL,
                dependencies=["sq_1"],
                dependency_type=DependencyType.SEQUENTIAL,
                priority=2
            )
        ]
    
    def _decompose_temporal_rule(self, question: str) -> List[SubQuestion]:
        topic = self._extract_temporal_topic(question)
        return [
            SubQuestion(
                id="sq_0",
                question=f"What is the origin/earliest known information about {topic}?",
                question_type=QuestionType.FACTUAL,
                dependencies=[],
                dependency_type=DependencyType.PARALLEL,
                priority=0
            ),
            SubQuestion(
                id="sq_1",
                question=f"What are the key milestones in the development of {topic}?",
                question_type=QuestionType.FACTUAL,
                dependencies=["sq_0"],
                dependency_type=DependencyType.SEQUENTIAL,
                priority=1
            ),
            SubQuestion(
                id="sq_2",
                question=f"What is the current state of {topic}?",
                question_type=QuestionType.FACTUAL,
                dependencies=["sq_1"],
                dependency_type=DependencyType.SEQUENTIAL,
                priority=2
            )
        ]
    
    def _decompose_aggregation_rule(self, question: str) -> List[SubQuestion]:
        category = self._extract_aggregation_category(question)
        return [
            SubQuestion(
                id="sq_0",
                question=f"What are the criteria for {category}?",
                question_type=QuestionType.FACTUAL,
                dependencies=[],
                dependency_type=DependencyType.PARALLEL,
                priority=0
            ),
            SubQuestion(
                id="sq_1",
                question=f"What are all instances/examples of {category}?",
                question_type=QuestionType.FACTUAL,
                dependencies=["sq_0"],
                dependency_type=DependencyType.SEQUENTIAL,
                priority=1
            ),
            SubQuestion(
                id="sq_2",
                question=f"Aggregate and count all instances of {category}",
                question_type=QuestionType.FACTUAL,
                dependencies=["sq_1"],
                dependency_type=DependencyType.SEQUENTIAL,
                priority=2,
                context_required=True
            )
        ]
    
    def _decompose_compositional_rule(self, question: str) -> List[SubQuestion]:
        parts = [part.strip() for part in question.split(' and ')]
        sub_questions = []
        
        for i, part in enumerate(parts):
            if not part.endswith('?'):
                part += '?'
            
            sub_questions.append(SubQuestion(
                id=f"sq_{i}",
                question=part,
                question_type=QuestionType.FACTUAL,
                dependencies=[],
                dependency_type=DependencyType.PARALLEL,
                priority=0
            ))
        
        return sub_questions
    
    # ========================================================================
    # Shared Utilities
    # ========================================================================
    
    def _determine_execution_order(self, sub_questions: List[SubQuestion]) -> List[List[str]]:
        """Determine execution order via topological sort"""
        dependency_graph = {sq.id: sq.dependencies for sq in sub_questions}
        execution_order = []
        remaining = set(sq.id for sq in sub_questions)
        
        while remaining:
            current_batch = []
            for sq_id in remaining:
                deps = dependency_graph[sq_id]
                if all(dep not in remaining for dep in deps):
                    current_batch.append(sq_id)
            
            if not current_batch:
                current_batch = [min(remaining, key=lambda x: next(
                    sq.priority for sq in sub_questions if sq.id == x
                ))]
            
            execution_order.append(current_batch)
            remaining -= set(current_batch)
        
        return execution_order
    
    # Entity extraction helpers
    def _extract_comparison_entities(self, question: str) -> List[str]:
        patterns = [
            r'compare\s+(.+?)\s+(?:and|with|vs|versus)\s+(.+?)[\?\.]',
            r'(.+?)\s+(?:vs|versus)\s+(.+?)[\?\.]',
            r'difference between\s+(.+?)\s+and\s+(.+?)[\?\.]',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question.lower())
            if match:
                return [match.group(1).strip(), match.group(2).strip()]
        
        return ["entity A", "entity B"]
    
    def _extract_causal_phenomenon(self, question: str) -> str:
        question_clean = re.sub(r'^(why|what causes?|how does|what makes)\s+', '', question.lower())
        question_clean = re.sub(r'\?$', '', question_clean).strip()
        return question_clean
    
    def _extract_temporal_topic(self, question: str) -> str:
        question_clean = re.sub(r'^(when|what is the timeline of|history of)\s+', '', question.lower())
        question_clean = re.sub(r'\?$', '', question_clean).strip()
        return question_clean
    
    def _extract_aggregation_category(self, question: str) -> str:
        question_clean = re.sub(r'^(how many|list all|count)\s+', '', question.lower())
        question_clean = re.sub(r'\?$', '', question_clean).strip()
        return question_clean
    
    def visualize_reasoning_plan(self, reasoning_plan: ReasoningPlan) -> str:
        """Create a text visualization of the reasoning plan"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"REASONING PLAN FOR: {reasoning_plan.original_question}")
        lines.append("=" * 80)
        lines.append(f"\nStrategy: {reasoning_plan.reasoning_strategy}")
        if reasoning_plan.confidence_score:
            lines.append(f"Confidence: {reasoning_plan.confidence_score:.2f}")
        lines.append(f"Total Sub-Questions: {len(reasoning_plan.sub_questions)}")
        lines.append(f"Execution Batches: {len(reasoning_plan.execution_order)}")
        
        lines.append("\n" + "-" * 80)
        lines.append("EXECUTION ORDER:")
        lines.append("-" * 80)
        
        for batch_idx, batch in enumerate(reasoning_plan.execution_order):
            lines.append(f"\nBatch {batch_idx + 1} (Parallel Execution):")
            for sq_id in batch:
                sq = next(sq for sq in reasoning_plan.sub_questions if sq.id == sq_id)
                lines.append(f"  [{sq.id}] {sq.question}")
                if sq.reasoning:
                    lines.append(f"       Reasoning: {sq.reasoning}")
                if sq.dependencies:
                    lines.append(f"       Dependencies: {', '.join(sq.dependencies)}")
        
        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


# ============================================================================
# Public Interface Functions
# ============================================================================

def create_reasoning_module(
    llm_type: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    use_llm: bool = True
) -> ReasoningModule:
    """
    Factory function to create a ReasoningModule with LLM integration.
    
    Args:
        llm_type: Type of LLM to use ("openai", "anthropic", "mock", or "none")
        api_key: API key for the LLM service
        model: Model name to use
        use_llm: Whether to enable LLM-based decomposition
        
    Returns:
        ReasoningModule instance
        
    Example:
        # With OpenAI
        module = create_reasoning_module("openai", api_key="sk-...")
        
        # With Anthropic Claude
        module = create_reasoning_module("anthropic", api_key="sk-ant-...")
        
        # Without LLM (rule-based only)
        module = create_reasoning_module("none")
    """
    llm_client = None
    
    if llm_type.lower() == "openai":
        default_model = model or "gpt-4"
        llm_client = OpenAIClient(api_key=api_key, model=default_model)
    elif llm_type.lower() == "anthropic":
        default_model = model or "claude-3-5-sonnet-20241022"
        llm_client = AnthropicClient(api_key=api_key, model=default_model)
    elif llm_type.lower() == "mock":
        llm_client = MockLLMClient()
    elif llm_type.lower() == "none":
        llm_client = None
        use_llm = False
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")
    
    return ReasoningModule(llm_client=llm_client, use_llm=use_llm)


def decompose_complex_question(
    question: str,
    reasoning_module: Optional[ReasoningModule] = None,
    use_llm: bool = True
) -> ReasoningPlan:
    """
    High-level interface to decompose a complex question.
    
    Args:
        question: The complex question to decompose
        reasoning_module: Optional existing ReasoningModule instance
        use_llm: Whether to use LLM for this decomposition
        
    Returns:
        ReasoningPlan with sub-questions and execution order
    """
    if reasoning_module is None:
        reasoning_module = ReasoningModule(use_llm=False)
    
    return reasoning_module.decompose_question(question, use_llm=use_llm)


def get_next_questions_to_execute(
    reasoning_plan: ReasoningPlan,
    completed_ids: List[str]
) -> List[SubQuestion]:
    """
    Get the next batch of questions that can be executed.
    
    Args:
        reasoning_plan: The reasoning plan
        completed_ids: List of completed sub-question IDs
        
    Returns:
        List of SubQuestion objects ready to execute
    """
    completed_set = set(completed_ids)
    
    for batch in reasoning_plan.execution_order:
        batch_questions = [sq for sq in reasoning_plan.sub_questions if sq.id in batch]
        
        if not all(sq.id in completed_set for sq in batch_questions):
            executable = []
            for sq in batch_questions:
                if sq.id not in completed_set:
                    if all(dep in completed_set for dep in sq.dependencies):
                        executable.append(sq)
            
            if executable:
                return executable
    
    return []


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    print("REASONING MODULE WITH LLM INTEGRATION - TEST SUITE")
    print("=" * 80)
    
    # Test 1: Rule-based decomposition (no LLM)
    print("\n\n>>> TEST 1: Rule-based Decomposition (No LLM)")
    print("-" * 80)
    
    reasoning_module = create_reasoning_module("none")
    question = "Compare the economic impact of renewable energy versus fossil fuels"
    
    plan = reasoning_module.decompose_question(question)
    print(reasoning_module.visualize_reasoning_plan(plan))
    
    # Test 2: Mock LLM (for testing without API calls)
    print("\n\n>>> TEST 2: Mock LLM Decomposition")
    print("-" * 80)
    
    mock_module = create_reasoning_module("mock")
    question = "Why did the Roman Empire fall and what were the long-term consequences?"
    
    try:
        plan = mock_module.decompose_question(question)
        print(mock_module.visualize_reasoning_plan(plan))
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Real LLM (uncomment and add your API key to test)
    """
    print("\n\n>>> TEST 3: OpenAI GPT-4 Decomposition")
    print("-" * 80)
    
    openai_module = create_reasoning_module(
        "openai",
        api_key="your-api-key-here",  # Replace with your key
        model="gpt-4"
    )
    
    question = "How has artificial intelligence evolved over the past decade and what are its current applications?"
    plan = openai_module.decompose_question(question)
    print(openai_module.visualize_reasoning_plan(plan))
    print("\nJSON Output:")
    print(json.dumps(plan.to_dict(), indent=2))
    """
    
    # Test 4: Execution simulation
    print("\n\n>>> TEST 4: Execution Simulation")
    print("-" * 80)
    
    question = "What is machine learning and what are its main applications?"
    plan = reasoning_module.decompose_question(question)
    
    print(f"Question: {question}\n")
    completed = []
    step = 1
    
    while True:
        next_questions = get_next_questions_to_execute(plan, completed)
        if not next_questions:
            break
        
        print(f"Step {step}:")
        for sq in next_questions:
            print(f"  ✓ Executing: [{sq.id}] {sq.question}")
            completed.append(sq.id)
        print()
        step += 1
    
    print("All sub-questions completed!")