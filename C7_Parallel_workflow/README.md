# C7 Parallel Workflow

This folder contains two LangGraph notebook examples that demonstrate parallel workflow design:

1. `4_batsman_workflow.ipynb` - a simple cricket analytics workflow.
2. `5_UPSC_essay_workflow.ipynb` - an essay evaluation workflow using a Hugging Face chat model.

Both notebooks are useful as learning examples for building state-based graphs with parallel branches, node aggregation, and final summary generation.

## What This Folder Demonstrates

- How to define a typed state for LangGraph.
- How to register multiple nodes that run in parallel from the `START` node.
- How to merge node outputs into a final node.
- How to combine deterministic computation with LLM-based evaluation.
- How to debug common notebook issues such as schema mismatches and model output shape mismatches.

## Notebook 1: Batting Analysis Workflow

### Purpose
This notebook calculates basic batting metrics for a cricket player and creates a short summary.

### Workflow Structure

Input state:
- `runs`
- `balls`
- `fours`
- `sixes`

Computed outputs:
- `sr` - strike rate
- `bpb` - balls per boundary
- `boundary_percent` - percentage of runs scored through boundaries
- `summary` - text summary of the three metrics

### Flow

1. `calculate_sr` computes strike rate.
2. `calculate_bpb` computes balls per boundary.
3. `calculate_boundary_percent` computes the boundary contribution.
4. `summary` formats the final result.
5. `workflow.invoke(initial_state)` runs the graph and returns the summary.

### Code Review Notes

What works well:
- The graph is clean and easy to follow.
- Each metric is isolated in its own node.
- The final summary node reads clearly as the final aggregation step.

What needs improvement:
- `calculate_bpb` can raise a division-by-zero error if `fours + sixes` is `0`.
- `calculate_boundary_percent` can fail if `runs` is `0`.
- The notebook assumes valid cricket statistics without input validation.
- The summary node depends on prior keys being present; if the graph is modified later, state consistency must be preserved.

### Recommended Hardening

- Validate input values before running the graph.
- Guard divisions with fallback logic.
- Add a small test cell for edge cases such as zero boundaries or zero runs.

## Notebook 2: UPSC Essay Evaluation Workflow

### Purpose
This notebook evaluates an essay across three dimensions:

- language quality
- depth of analysis
- clarity of thought

It then combines those reviews into a final overall feedback with an average score.

### Workflow Structure

The notebook uses:
- `HuggingFaceEndpoint` as the LLM backend.
- `ChatHuggingFace` as the chat wrapper.
- `EvaluationSchema` as the structured output schema.
- `UPSCState` as the LangGraph state.

### Flow

1. The essay text is stored in `essay`.
2. `evaluate_language` generates a language review and score.
3. `evaluate_analysis` generates an analysis review and score.
4. `evaluate_thought` generates a clarity review and score.
5. `final_evaluation` summarizes the three reviews and computes the average score.
6. `workflow.invoke(initial_state)` executes the full evaluation pipeline.

### Current Implementation Notes

The notebook is now aligned to the current model behavior:
- `with_structured_output(..., method="json_schema")` is used because the model path does not support Pydantic structured output through function calling.
- The structured output is accessed as a dictionary, for example `output["feedback"]` and `output["score"]`.
- `UPSCState` includes `essay` and uses `individual_scores` consistently across all evaluator nodes and the final reducer.

### Code Review Notes

What works well:
- The workflow is logically split into parallel evaluation branches.
- Structured output makes the scoring step more reliable than free-form parsing.
- The final node produces an interpretable aggregate result.

What needs improvement:
- The notebook is dependent on an external Hugging Face endpoint, so execution speed and reliability depend on network and model availability.
- The graph definition contains a duplicate `graph = StateGraph(UPSCState)` line, which should be cleaned up.
- The `EvaluationSchema` field description contains a minor typo: `Detailed feedbackfor the essay`.
- The `UPSCState` model could be tightened further by separating input fields from output fields more clearly.

### Common Failure Modes and Fixes

- If structured output raises a `NotImplementedError`, use `method="json_schema"` instead of relying on default function calling.
- If a node raises `KeyError: 'essay'`, ensure the input state includes the essay text before invoking the workflow.
- If structured output returns a dictionary, access fields with square brackets rather than attribute access.
- If the workflow appears to hang, the cause is usually the remote LLM call rather than the graph itself.

### Recommended Hardening

- Add explicit timeout handling around LLM calls.
- Add retry logic for temporary API failures.
- Normalize the state schema so input-only and output-only keys are easier to distinguish.
- Add a final cell that prints both the detailed feedback and the numeric score.

## Overall Assessment

This folder is a solid learning sequence for LangGraph parallel workflows:

- The first notebook shows deterministic parallel computation.
- The second notebook shows a practical LLM-based parallel evaluation pipeline.

Together, they demonstrate how LangGraph can be used for both rule-based computation and structured LLM orchestration.

## Execution Status

Based on the latest notebook state:
- `4_batsman_workflow.ipynb` executes successfully.
- `5_UPSC_essay_workflow.ipynb` is structurally fixed and its key cells run successfully, though the final workflow still depends on a remote model call and may be slower than local execution.

## Suggested Next Steps

1. Add input validation to the cricket workflow.
2. Add a compact result display cell to the essay workflow.
3. Add notebook-level comments or markdown cells explaining the graph structure for future readers.
