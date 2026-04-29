# C6 Sequential Workflow - Professional Documentation

## Executive Summary

This repository contains five comprehensive Jupyter notebooks demonstrating sequential workflow implementation using LangGraph. Each example progressively builds upon fundamental concepts to create sophisticated, multi-step computational pipelines with advanced state management, context accumulation, and conditional processing patterns.

---

## 📋 Table of Contents
1. [Installation Verification](#installation-verification)
2. [BMI Calculation Workflow](#bmi-calculation-workflow)
3. [Simple LLM Q&A System](#simple-llm-qa-system)
4. [Prompt Chaining Pipeline](#prompt-chaining-pipeline)
5. [Generic Multi-Stage Evaluation](#generic-multi-stage-evaluation)
6. [Comparative Analysis](#comparative-analysis)
7. [Setup Instructions](#setup-instructions)

---

## Installation Verification

### **Notebook:** `0_test_installation.ipynb`

#### **Purpose**
Validates the LangGraph installation and confirms all required dependencies are correctly installed in the Python environment.

#### **Implementation**
```python
from langgraph.graph import StateGraph
```

This single import statement serves as a diagnostic to verify proper installation and accessibility of the LangGraph library.

#### **Workflow Architecture**
```
START → END
```
No processing occurs; this notebook functions purely as a diagnostic tool.

#### **Success Criteria**
- No ImportError exceptions
- Successful module resolution
- Clean execution without warnings

#### **Use Cases**
- Environment setup validation
- Dependency verification in CI/CD pipelines
- PBMI Calculation Workflow

### **Notebook:** `1_bmi_workflow.ipynb`

#### **Purpose**
Demonstrates a pure computational workflow that calculates Body Mass Index (BMI) and assigns health classification categories. This represents the simplest workflow architecture with two sequential processing nodes.

#### **State Schema**
```python
class BMIState(TypedDict):
    weight_kg: float      # Body weight in kilograms
    height_m: float       # Height in meters
    bmi: float            # Calculated BMI value
    category: str         # Health classification category
```

#### **Pipeline Architecture**

**Stage 1: BMI Calculation Node**
- **Input**: Weight and height values from state
- **Processing Algorithm**:
  1. Extract weight_kg and height_m from state
  2. Apply formula: `BMI = weight / (height²)`
  3. Round result to two decimal places
  4. Store in state object
- **Output**: Updated state with computed BMI value

**Stage 2: Health Classification Node**
- **Input**: BMI value from previous node
- **Classification Logic**:
  1. Evaluate BMI against WHO standards
  2. Assign classification:
     - BMI < 18.5 → "Underweight"
     - 18.5 ≤ BMI < 25 → "Normal"
     - 25 ≤ BMI < 30 → "Overweight"
     - BMI ≥ 30 → "Obese"
- **Output**: Updated state with health category

#### **Execution Flow**
```
START → calculate_bmi → label_bmi → END
```

#### **Sample Input**
```python
initial_state = {'weight_kg': 80, 'height_m': 1.73}
```

#### **Sample Output**
```python
{
    'weight_kg': 80,
    'height_m': 1.73,
    'bmi': 26.73,
    'category': 'Overweight'
}
```

#### **Technical Patterns**
- **State Immutability**: Data flows through nodes as complete objects
- **Stateless Computation**: Pure functions with no external dependencies
- **Deterministic Processing**: Identical inputs always produce identical outputs
- **Type Safety**: TypedDict ensures field consistency

#### **Real-World Applications**
- Healthcare and wellness platforms
- Fitness tracking applications
- Population health monitoring
- Medical assessment systems
- ISimple LLM Q&A System

### **Notebook:** `2_simple_llm_workflow.ipynb`

#### **Purpose**
Implements a foundational question-answering system integrating large language models (LLMs) from HuggingFace. This workflow introduces machine learning components while maintaining architectural simplicity.

#### **Prerequisites**
- Active HuggingFace API authentication token
- `langchain-huggingface` library installation
- Environment configuration via `.env` file
- Internet connectivity for API communication

#### **State Schema**
```python
class LLMState(TypedDict):
    question: str    # User-provided query
    answer: str      # LLM-generated response
```

#### **Model Configuration**
```python
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational"
)
model = ChatHuggingFace(llm=llm)
```

#### **Processing Pipeline**

**Single Processing Node: `llm_qa`**
- **Input**: User question extracted from state
- **Processing Steps**:
  1. Retrieve question from state dictionary
  2. Construct prompt template: `"Answer the following question: {question}"`
  3. Invoke language model with formatted prompt
  4. Extract text content from model response
  5. Update state with generated answer
- **Output**: Modified state containing both question and answer

#### **Execution Flow**
```
START → llm_qa → END
```

#### **Sample Input**
```python
initial_state = {'question': "What is the capital of India?"}
```

#### **Sample Output**
```python
{
    'question': 'What is the capital of India?',
    'answer': 'The capital of India is New Delhi.'
}
```

#### **Technical Characteristics**
- **External API Integration**: Remote language model invocation
- **Asynchronous Processing**: Non-blocking API calls
- **Prompt Engineering**: Structured input formatting
- **Response Parsing**: Extraction of relevant content

#### **Real-World Applications**
- Interactive chatbot systems
- Automated FAQ response generation
- Customer support automation
- Knowledge base query systems
- Virtual assistant implementations

#### **Performance Considerations**
- **Average Latency**: 1-5 seconds (depends on model and network)
- *Prompt Chaining Pipeline

### **Notebook:** `3_prompt_chaining.ipynb`

#### **Purpose**
Demonstrates advanced workflow design through prompt chaining—a technique where the output from one language model instance becomes the input for the next. This pattern enables production of complex, higher-quality content through structured, multi-stage processing.

#### **State Schema**
```python
class BlogState(TypedDict):
    title: str      # Blog article title
    outline: str    # Generated structural outline
    content: str    # Complete blog post content
```

#### **Model Configuration**
```python
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational"
)
model = ChatHuggingFace(llm=llm)
```

#### **Processing Pipeline**

**Stage 1: Outline Generation (`create_outline`)**
- **Input**: Blog article title
- **Processing Logic**:
  1. Extract topic from state
  2. Formulate prompt: `"Generate a detailed outline for a blog on: {title}"`
  3. Invoke language model
  4. Store structured outline in state
- **Output**: State enriched with blog outline
- **Purpose**: Establish structural framework for content generation

**Stage 2: Content Generation (`create_blog`)**
- **Input**: Title and outline from previous stage
- **Processing Logic**:
  1. Retrieve both title and outline from state
  2. Construct comprehensive prompt: `"Write a detailed blog on {title} using this outline: {outline}"`
  3. Invoke language model with full context
  4. Store generated content in state
- **Output**: Fully populated state with complete blog post
- **Purpose**: Generate detailed, well-structured content

#### **Execution Flow**
```
START → create_outline → create_blog → END
```

#### **Sample Input**
```python
initial_state = {
    'title': "The Impact of Artificial Intelligence on Modern Job Markets"
}
```

#### **Sample Output**
```python
{
    'title': 'The Impact of Artificial Intelligence on Modern Job Markets',
    'outline': '1. Introduction\n2. Current State of AI\n3. Job Market Implications...',
    'content': '[Complete blog article with introduction, body, and conclusion]'
}
```

#### **Architectural Patterns**

**Prompt Chaining**: Sequential prompts where each stage builds upon previous outputs
- Stage 1 provides structural framework
- Stage 2 fills framework with detailed content
- Each stage produces higher-quality output through increased context

**Context Accumulation**: Full history preserved across nodes
- Enables consistent, coherent outputs
- Reduces hallucination probability
- Improves factual accuracy

#### **Real-World Applications**
- Automated blog and article generation
- Professional report creation
- Newsletter composition
- Academic paper structuring
- MGeneric Multi-Stage Evaluation

### **Notebook:** `ht_evaluate.ipynb`

#### **Purpose**
Provides a reusable template for multi-stage sequential workflows. This generic architecture demonstrates how to structure complex, scalable data processing pipelines adaptable to diverse use cases beyond simple computation or single-model tasks.

#### **State Schema**
```python
class WorkflowState(TypedDict):
    input_text: str      # Original data input
    outline: str         # First-stage processing output
    blob: str            # Second-stage expanded representation
    evaluation: str      # Final evaluation result
```

#### **Processing Pipeline**

**Stage 1: Structuring (`outline` node)**
- **Input**: Raw input text
- **Processing**: Transforms unstructured input into organized outline format
- **Output**: State enriched with structured outline
- **Purpose**: Provides preliminary organization and framework

**Stage 2: Expansion (`blob` node)**
- **Input**: Outline from Stage 1
- **Processing**: Elaborates and expands outline structure
- **Output**: Extended blob representation in state
- **Purpose**: Develops outline into comprehensive intermediate form

**Stage 3: Evaluation (`evaluate` node)**
- **Input**: Blob from Stage 2
- **Processing**: Analyzes and evaluates expanded content
- **Output**: Final evaluation results in state
- **Purpose**: Produces actionable insights and findings

#### **Execution Flow**
```
START → outline → blob → evaluate → END
```

#### **Sample Input**
```python
initial_state = {
    'input_text': 'This is a test document for processing'
}
```

#### **Sample Output**
```python
{
    'input_text': 'This is a test document for processing',
    'outline': 'Outline: Key points identified...',
    'blob': 'Expanded form: Detailed breakdown...',
    'evaluation': 'Assessment: Quality analysis and findings...'
}
```

#### **Architectural Properties**

**Linear Sequential Processing**:
- Each stage processes previous stage output
- No branching or conditional routing
- Deterministic execution path

**Data Preservation**:
- All intermediate results maintained in state
- Complete history available for analysis
- Enables debugging and auditing

**Design Flexibility**:
- Simple extension with additional stages
- State dictionary easily expanded
- Accommodates new processing logic

#### **Real-World Applications**
- ETL (Extract-Transform-Load) pipelines
- Document analysis and classification
- Quality assurance and validation systems
- Business process automation
- Data enrichment workflows
- MComparative Analysis

### Feature Comparison Matrix

| Characteristic | Installation | BMI | Simple LLM | Prompt Chain | Evaluation |
|---|---|---|---|---|---|
| **Processing Nodes** | 0 | 2 | 1 | 2 | 3 |
| **ML/AI Component** | ❌ | ❌ | ✅ | ✅ | ❌ |
| **Complexity Level** | Diagnostic | Basic | Intermediate | Advanced | Intermediate |
| **Primary Use Case** | Verification | Computation | Q&A | Content | Generic |
| **State Fields** | 0 | 4 | 2 | 3 | 4 |
| **Latency (ms)** | <100 | <10 | 1000-5000 | 2000-10000 | <100 |
| **External Dependencies** | None | None | API | API | None |

---

## Standard Workflow Architecture Pattern
Step 1: Import Required Dependencies
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# Step 2: Define State Structure with Type Hints
class WorkflowState(TypedDict):
    input_data: str
    intermediate_result: str
    final_output: str

# Step 3: Implement Processing Functions
def process_stage_one(state: WorkflowState) -> WorkflowState:
    """First processing stage"""
    # Extract input
    data = state['input_data']
    
    # Apply processing logic
    result = execute_processing(data)
    
    # Update state
    state['intermediate_result'] = result
    return state

def process_stage_two(state: WorkflowState) -> WorkflowState:
    """Second processing stage"""
    # Use previous output
    intermediate = state['intermediate_result']
    
    # Apply additional processing
    output = execute_final_processing(intermediate)
    
    # Update state
    state['final_output'] = output
    return state

# Step 4: Construct Graph Architecture
graph = StateGraph(WorkflowState)

# Register processing nodes
graph.add_node('stage_1', process_stage_one)
graph.add_node('stage_2', process_stage_two)

# Define execution sequence
graph.add_edge(START, 'stage_1')
graph.add_edge('stage_1', 'stage_2')
graph.add_edge('stage_2', END)

# Step 5: Compile Graph
workflow = graph.compile()

# Step 6: Execute Workflow
initial_state = {'input_data': 'your_data_here'}
final_state = workflow.invoke(initial_state)
print(final_state)
```

---

## Recommended Learning Progression

Follow this structured learning path for optimal understanding:

1. **[Installation Verification](#installation-verification)** 
   - Environment setup validation
   - Dependency verification
   - Pre-requisite confirmation

2. **[BMI Calculation Workflow](#bmi-calculation-workflow)** 
   - Basic state management
   - Multi-node sequential processing
   - Pure computational workflows

3. **[Simple LLM Q&A System](#simple-llm-qa-system)** 
   - Language model integration
   - External API interaction
   - Single-stage ML processing

4. **[Prompt Chaining Pipeline](#prompt-chaining-pipeline)** 
   - Multi-stage LLM workflows
   - Context accumulation across stages
   - Advanced prompt engineering

5. **[Generic Multi-Stage Evaluation](#generic-multi-stage-evaluation)** 
   - Scalable pipeline architecture
   - Extensible workflow templates
   - Production-ready patterns

---

## Setup and Installation

### System Requirements
- Python 3.10 or higher
- pip package manager
- Internet connectivity (for LLM workflows)

### Dependency Installation

```bash
pip install langgraph langchain langchain-huggingface python-dotenv jupyter
```

### API Configuration for LLM Workflows

Create `.env` file in project root:
```env
HUGGINGFACEHUB_API_TOKEN=your_token_from_huggingface
```

Obtain token: https://huggingface.co/settings/tokens

### Validation

```bash
jupyter notebook 0_test_installation.ipynb
```

---

## Core Architectural Principles

### State Management
- **Immutability**: State passed as complete objects between nodes
- **Field Consistency**: TypedDict ensures type safety across workflow
- **History Preservation**: Intermediate results retained in state

### Node Functions
- **Function Signature**: `def node(state: StateType) -> StateType:`
- **Pure Functions**: Deterministic, no external side effects
- **State Transformation**: Nodes return modified state objects

### Graph Construction
- **Nodes**: Represent processing stages
- **Edges**: Define execution flow and dependencies
- **Pseudo-Nodes**: START and END mark entry/exit

### Execution Model
- **Sequential Flow**: Nodes execute sequentially
- **State Propagation**: Modified state flows to next node
- **Pre-Execution Compilation**: Graph compiled before running

---

## Advanced Workflow Patterns

### Conditional Routing
```python
graph.add_conditional_edges(
    'decision_node',
    condition_function,
    {True: 'path_a', False: 'path_b'}
)
```

### Parallel Execution
```python
graph.add_edge(START, ['parallel_node_1', 'parallel_node_2'])
graph.add_edge('parallel_node_1', 'merge_node')
graph.add_edge('parallel_node_2', 'merge_node')
```

### Iterative Processing
```python
graph.add_edge('process_node', 'check_node')
graph.add_conditional_edges(
    'check_node',
    should_continue,
    {True: 'process_node', False: END}
)
```

### Error Handling
```python
try:
    result = workflow.invoke(initial_state)
except Exception as error:
    handle_workflow_error(error)
```

---

## API Reference

### Execution Methods

```python
# Single synchronous execution
result = workflow.invoke(initial_state)

# Stream results incrementally
for chunk in workflow.stream(initial_state):
    process_output(chunk)

# Generate graph visualization
graph_visualization = workflow.get_graph().draw_mermaid_png()
```

### Common Operations

```python
# Create sequential edge
graph.add_edge('source', 'destination')

# Add conditional branching
graph.add_conditional_edges('node', condition)

# Register processing node
graph.add_node('node_name', processing_function)

# Parallel execution entry
graph.add_edge(START, ['node_a', 'node_b'])
```

---

## Troubleshooting Guide

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| ImportError for langchain | Incomplete installation | `pip install -U langchain langchain-huggingface` |
| API authentication failure | Invalid/missing token | Verify HUGGINGFACEHUB_API_TOKEN in .env |
| Kernel crash in notebook | Memory exhaustion | Restart kernel, run cells sequentially |
| API timeout errors | Network issues | Check connectivity, increase timeout values |
| Type mismatch errors | Incompatible state fields | Verify TypedDict definitions match usage |

---

## Project Metadata

- **Framework**: LangGraph 0.0.x+
- **Language**: Python 3.10+
- **Documentation Version**: 2.0 (Professional Edition)
- **Last Updated**: April 2026
- **Author**: LangGraph Educational Team

---

## License and Usage

Educational and instructional material. Free for learning and educational purposes.

---

## Additional Resources

### Official Documentation
- [LangGraph Documentation](https://github.com/langchain-ai/langgraph)
- [LangChain Documentation](https://python.langchain.com/)
- [HuggingFace API](https://huggingface.co/docs/api-inference)

### Learning Materials
- Sequential workflow patterns
- State management best practices
- LLM integration techniques

---

## Support

For issues or clarification:
1. Review the troubleshooting section above
2. Consult official LangGraph documentation
3. Check HuggingFace API documentation
4. Review example notebooks in this directorying:** Nodes এবং edges define করে workflow logic build করা হয়
5. **Compilation:** Graph compile করার পর `invoke()` দিয়ে execute করা হয়

---

## 🎓 Advanced Topics (পরবর্তী learning)

- **Conditional Edges:** Different paths based on conditions
- **Parallel Nodes:** একসাথে multiple nodes run করা
- **Loop/Cycles:** Workflow-এ loop structure
- **Error Handling:** Node failures handle করা
- **Streaming:** Results stream করে receive করা

---

## 📞 Quick Reference

### Workflow Execution
```python
# Single execution
result = workflow.invoke(initial_state)

# Stream results
for chunk in workflow.stream(initial_state):
    print(chunk)

# Get graph visualization
workflow.get_graph().draw_mermaid_png()
```

### Common Patterns
```python
# Sequential
add_edge('node1', 'node2')

# Branching (conditional)
add_conditional_edges('node1', condition_func)

# Parallel
add_edge(START, ['node1', 'node2'])
```

---

**Created for:** LangGraph Sequential Workflow Tutorial
**Version:** 1.0
