# Conditional Workflows

This folder demonstrates how to build **conditional workflows** with LangGraph and Python. The notebooks show how a graph can route execution based on computed state so the application can take different paths without hard-coding a single linear sequence.

## What is a Conditional Workflow?

A conditional workflow is a graph-based pipeline where the next step depends on a decision made from the current state. Instead of running the same code every time, the workflow evaluates an intermediate result and then chooses the next node dynamically.

In LangGraph, this is typically done with:

- `StateGraph` to define the workflow
- `START` and `END` to mark entry and exit points
- nodes that transform or enrich the shared state
- conditional routing logic that selects the next path

## Files in This Folder

This folder contains two notebook-based examples:

1. `6_quadratic_equation_workflow.ipynb`
2. `7_review_reply_workflow.ipynb`

Each notebook demonstrates a different use case, but both follow the same core idea: compute a piece of state, inspect it, and route execution based on the result.

---

## 1. Quadratic Equation Workflow

**Notebook:** `6_quadratic_equation_workflow.ipynb`

### Goal

This workflow solves a quadratic equation of the form:

$$ax^2 + bx + c = 0$$

The graph first prints the equation, then computes the discriminant, and finally routes to the correct result node.

### Core State

The notebook uses a `TypedDict` called `QuadState` with these fields:

- `a`, `b`, `c`: coefficients of the equation
- `equation`: formatted equation string
- `discriminant`: computed value of $b^2 - 4ac$
- `result`: final human-readable output

### Workflow Steps

1. `show_equation`
   - Formats the quadratic equation as a string.
   - Stores it in the state.

2. `calculate_discriminant`
   - Computes the discriminant using:

     $$D = b^2 - 4ac$$

3. `check_condition`
   - Routes execution based on the discriminant:
   - If `D > 0`, go to `real_roots`
   - If `D == 0`, go to `repeated_root`
   - If `D < 0`, go to `no_real_root`

4. Result nodes
   - `real_roots`: computes two real roots
   - `repeated_root`: computes one repeated root
   - `no_real_root`: returns that no real roots exist

### How the Code Works

The notebook builds a `StateGraph`, registers each function as a node, and connects the nodes with edges. The important part is the conditional routing function `check_condition`, which returns the name of the next node instead of always following the same edge.

That means the graph can react to the computed discriminant and choose the right branch automatically.

### Example Input

The notebook tests the graph with:

- `a = 1`
- `b = -3`
- `c = 2`

This produces a quadratic equation with two real roots.

---

## 2. Review Reply Workflow

**Notebook:** `7_review_reply_workflow.ipynb`

### Goal

This workflow reads a user review, detects its sentiment, and then routes to either a positive reply flow or a negative support flow.

### Core State

The notebook uses a `TypedDict` called `ReviewState` with these fields:

- `review`: the user review text
- `sentiment`: positive or negative classification
- `diagnosis`: structured issue details for negative reviews
- `response`: final generated reply

### Workflow Steps

1. `find_sentiment`
   - Analyzes the review text.
   - Produces `positive` or `negative`.

2. `check_sentiment`
   - Routes the graph based on the sentiment.
   - If the sentiment is positive, it goes to `positive_response`.
   - If the sentiment is negative, it goes to `run_diagnosis`.

3. `positive_response`
   - Generates a warm thank-you response.
   - Asks the user to leave feedback on the website.

4. `run_diagnosis`
   - Extracts issue details from a negative review.
   - Produces a diagnosis with:
     - `issue_type`
     - `tone`
     - `urgency`

5. `negative_response`
   - Uses the diagnosis to write an empathetic support reply.

### How the Code Works

The workflow first creates a shared state from the review text. The sentiment node classifies the review, then `graph.add_conditional_edges()` sends the flow down the right branch.

This design keeps the logic modular:

- one node for classification
- one node for positive replies
- one node for diagnosis
- one node for support response

That makes the workflow easy to extend later, for example by adding a spam branch, a refund branch, or a language translation branch.

### Important Implementation Note

This notebook uses Hugging Face models through `HuggingFaceEndpoint` and `ChatHuggingFace`. If you want structured outputs, make sure the model setup supports the output format you are asking for. In some cases, you may need to parse JSON manually instead of relying on `with_structured_output()`.

---

## Typical Execution Order

For both notebooks, run the cells in order:

1. Import dependencies
2. Define the state
3. Define workflow functions
4. Build the graph
5. Compile the workflow
6. Prepare sample input
7. Invoke the workflow

Running cells out of order can cause `NameError` issues because later cells depend on variables created earlier.

## Common Errors and Fixes

### `NameError`

Usually means a previous cell was not run.

Fix:

- Restart the kernel
- Run all cells from top to bottom

### Graph node error

Usually means an edge points to a function instead of a registered node name.

Fix:

- Add the function with `graph.add_node('node_name', function)`
- Use the same node name in `graph.add_edge()` or `graph.add_conditional_edges()`

### Hugging Face model error

Can happen if the API key is missing or the selected model does not support the requested structured output.

Fix:

- Set `HUGGINGFACEHUB_API_TOKEN` in your environment or `.env`
- Restart the kernel
- Use prompt-based JSON output parsing if structured output is not supported

## Learning Outcome

After completing both notebooks, you should understand how to:

- design branching workflows
- pass and update shared state between nodes
- use conditional routing in LangGraph
- map a real problem into a graph structure
- adapt workflow logic for both mathematical and NLP use cases

## Recommended Next Step

If you want to continue this series, consider adding a third notebook for:

- multi-branch routing
- fallback handling
- tool calling
- validation before routing

---

## Quick Summary

- `6_quadratic_equation_workflow.ipynb` demonstrates branching logic for mathematical solving.
- `7_review_reply_workflow.ipynb` demonstrates branching logic for sentiment-based review replies.
- Both notebooks show the same LangGraph pattern: compute state, decide route, produce final output.
