# X (Twitter) Post Generator - Iterative AI Workflow

## Overview

The **X Post Generator** is an intelligent, multi-agent AI system that automatically generates, evaluates, and iteratively improves tweet content. Using advanced language models and a state-driven workflow engine (LangGraph), the system creates viral-worthy tweets that meet strict quality and humor standards.

## Features

- **Intelligent Tweet Generation** - Creates original, witty tweets with observational humor and cultural relevance
- **Automated Quality Evaluation** - Assesses tweets using multiple criteria including originality, humor, punchiness, and virality potential
- **Iterative Optimization** - Continuously refines tweets based on structured feedback until approval
- **Multi-Model Architecture** - Leverages specialized language models for different workflow stages:
  - Generator LLM (temperature: 0.7) - Creative tweet composition
  - Evaluator LLM (temperature: 0.2) - Critical assessment
  - Optimizer LLM (temperature: 0.5) - Strategic refinement

## Technical Stack

- **LangGraph** - State management and workflow orchestration
- **LangChain** - LLM integration and message handling
- **Hugging Face Models** - Qwen2.5-7B-Instruct for generation and evaluation
- **Python 3.8+** - Core programming language
- **Pydantic** - Data validation and schema definition

## Architecture

### Workflow Pipeline

```
[Generate Tweet] → [Evaluate Tweet] → [Route Decision]
                                            ↓
                                    ┌──────┴──────┐
                                    ↓             ↓
                              [Approved]    [Optimize]
                                    ↑             ↓
                                    └─────────────┘
```

### Key Components

#### 1. **State Management** (`TweetState`)
Maintains the complete workflow context:
- `topic` - Subject of the tweet
- `tweet` - Current tweet content
- `evaluation` - Approval status (approved/needs_improvement)
- `feedback` - Evaluation feedback
- `iteration` - Current iteration count
- `max_iteration` - Maximum allowed iterations
- `tweet_history` - All generated tweet versions
- `feedback_history` - All feedback iterations

#### 2. **Generator Node**
Creates original, humorous tweets following specific rules:
- Max 280 characters
- No question-answer format
- Observational humor, irony, and cultural references
- Simple, everyday English

#### 3. **Evaluator Node**
Critiques tweets on five dimensions:
1. **Originality** - Freshness and uniqueness
2. **Humor** - Genuine laugh factor
3. **Punchiness** - Short, sharp, scroll-stopping appeal
4. **Virality Potential** - Retweet/share-worthiness
5. **Format Compliance** - Technical tweet requirements

Auto-rejection criteria:
- Question-answer format
- Exceeds 280 characters
- Traditional setup-punchline joke structure
- Generic or deflating conclusions

#### 4. **Optimizer Node**
Refines tweets based on evaluator feedback while maintaining:
- Adherence to original topic
- Quality humor standards
- Character limit compliance
- Viral appeal

#### 5. **Router**
Conditional logic determining workflow path:
- **Approved path**: Workflow concludes with final tweet
- **Needs Improvement path**: Routes to optimizer for refinement
- **Max Iterations**: Outputs current best tweet regardless of approval

## Installation

### Prerequisites
- Python 3.8 or higher
- Hugging Face API token

### Setup

1. **Clone or download the repository**
   ```bash
   cd C9_Iterative_Workflows
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install langgraph langchain-core langchain-huggingface huggingface-hub python-dotenv pydantic
   ```

4. **Configure environment variables**
   Create a `.env` file in the project directory:
   ```
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
   ```

## Usage

### Basic Example

```python
from notebook_name import workflow

# Define initial state
initial_state = {
    "topic": "your_tweet_topic",
    "iteration": 1,
    "max_iteration": 5
}

# Execute workflow
result = workflow.invoke(initial_state)

# Access results
final_tweet = result['tweet']
tweet_history = result['tweet_history']
feedback_history = result['feedback_history']
```

### Understanding Output

The workflow returns a complete state dictionary containing:
- **`tweet`** - Final approved tweet
- **`evaluation`** - Approval status
- **`feedback`** - Final feedback from evaluator
- **`tweet_history`** - All iterations of the tweet
- **`feedback_history`** - Feedback for each iteration
- **`iteration`** - Number of iterations performed

## Configuration

### Model Parameters

Adjust LLM behavior by modifying temperature values:

```python
# More creative/random
temperature=0.7  # Generator
temperature=0.5  # Optimizer

# More deterministic
temperature=0.2  # Evaluator
```

### Iteration Control

Control the refinement process:
```python
max_iteration = 5  # Maximum refinement cycles
```

## Performance Considerations

- **API Calls**: Each workflow invocation makes 1-3 API calls per iteration (generation, evaluation, optimization)
- **Latency**: Typical execution time: 30-90 seconds depending on max iterations
- **Cost**: Depends on Hugging Face endpoint pricing and model selection

## Troubleshooting

### Common Issues

**Issue**: "HUGGINGFACEHUB_API_TOKEN not found"
- **Solution**: Verify `.env` file exists and contains valid token

**Issue**: Tweets consistently fail evaluation
- **Solution**: Increase `max_iteration` value or adjust evaluator feedback prompts

**Issue**: Low humor/originality scores
- **Solution**: Adjust generator LLM temperature (increase for more creativity)

## Future Enhancements

- [ ] Support for multiple social media platforms (LinkedIn, Instagram, TikTok)
- [ ] Custom evaluation criteria per user
- [ ] Batch tweet generation for campaigns
- [ ] A/B testing framework for tweet variants
- [ ] User feedback integration for continuous improvement
- [ ] Advanced sentiment and engagement prediction

## API Reference

### Nodes

#### `generate_tweet(state: TweetState) → dict`
Generates initial tweet based on topic.

**Returns**: Dictionary with `tweet` and `tweet_history`

#### `evaluate_tweet(state: TweetState) → dict`
Evaluates tweet quality and provides structured feedback.

**Returns**: Dictionary with `evaluation`, `feedback`, and `feedback_history`

#### `optimize_tweet(state: TweetState) → dict`
Refines tweet based on feedback and increments iteration counter.

**Returns**: Dictionary with refined `tweet`, updated `iteration`, and `tweet_history`

#### `route_evaluation(state: TweetState) → str`
Determines workflow path based on evaluation result.

**Returns**: "approved" or "needs_improvement"

## License

This project is part of the AgenticAI Tutorial series.

## Support

For issues, questions, or contributions, please refer to the main tutorial documentation.

---

**Version**: 1.0  
**Last Updated**: June 2026  
**Framework**: LangGraph  
**Model**: Qwen2.5-7B-Instruct
