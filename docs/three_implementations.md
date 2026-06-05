# Three implementations

The project keeps three versions because the teaching goal is to show
how the same problem changes under different orchestration styles.

| Dimension | Static Python | LangChain | LangGraph |
|---|---|---|---|
| Mental model | Plain sequence | Runnable composition | State machine |
| Best for | First read and debugging | Bounded chains and tool-like steps | Explicit state and branch points |
| State handling | Local variables | Payload passed through Runnable | `GraphState` updates per node |
| Error surface | Python traceback | Runnable invocation boundary | Node boundary |
| Trace output | `PipelineTrace` | `PipelineTrace` | `PipelineTrace` |
| Teaching value | Shows the core idea without framework noise | Shows how to wrap the same contract | Shows why graph edges matter |

The old repo used a broad ReAct-style LangChain agent. The overhaul uses
a bounded `RunnableLambda` because the purpose is framework comparison,
not open-ended tool wandering.

The LangGraph version uses six nodes:

1. `preprocess`
2. `search_fetch`
3. `summarize`
4. `bias_detect`
5. `critique`
6. `reconcile`

All three implementations share `core/`. If a test passes in one
implementation and fails in another, the framework wrapper is the first
place to inspect.
