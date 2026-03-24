# /add-node

Add a new node to an existing LangGraph graph in AutoResearch AI.

## Instructions

The user will describe what the new node should do. Follow these steps:

1. **Identify the graph** — ask if it belongs to `build_graph()` (research) or `build_creative_graph()` (creative), unless obvious from the description.

2. **Add state fields** — add any new fields to the relevant TypedDict (`ResearchState` or `CreativeState`) in `agent.py`.

3. **Write the node function** in `agent.py`:
   - Takes `state: ResearchState` or `state: CreativeState`
   - Returns a partial state dict (only the fields it changes)
   - Appends to `status_log` with a human-readable message
   - Sets `current_node` to the node name
   - If it calls the LLM: use `get_langfuse_prompt(name, fallback, **vars)` from `chains.py`

4. **Wire it into the graph**:
   - `graph.add_node("node_name", node_function)`
   - Add `graph.add_edge(...)` before and after

5. **If it calls the LLM**:
   - Add an inline fallback constant `_NODENAME_FALLBACK` in `agent.py`
   - Remind the user to create the matching prompt in Langfuse UI

6. **If it scrapes**:
   - Add the extractor function to `tools.py`
   - Import it in `agent.py`

7. **Update app.py** if the new node should appear in the progress tracker UI.

## Example usage
`/add-node — Add a sentiment analysis node to the research graph that scores brand perception from Reddit results`
