# langchain-energy-middleware

## Installation & running
Note this currently needs a beefy GPU. In the future should make an option to use external models.

- Configure & activate a venv
- `pip install -r requirements`
- Download ollama: https://ollama.com/download
- `ollama pull qwen3.5:4b`
- `ollama pull qwen3.5:2b`
- In a separate terminal: `ollama serve` (might already be running)
- `ollama ps` to see if model is running, otherwise start with `ollama run qwen3.5`
- `python3 main.py` to run a hardcoded test query
- `streamlit run app.py` to open the interactive chat UI in your browser

## TODO
- Right now all data points are added together, but I think it would be better to associate the sub-prompts with the main prompt that called them. - Aadesh
- Find more realistic power estimates. Maybe a nicer API to get metrics from the state/model. - Ruy
- Feed metrics back into the agent. Maybe do this automatically every N steps through middleware or just give it a tool to check itself. Or maybe in the context? - Samuel
- Make a way to interactively demo the agent, either cli or simple web interface. Maybe look at what previous year's project did. - Jan
- More complicated example agent, so it actually has a meaningful choice between sub-agents - Andrea
- At the end: create pip package. 