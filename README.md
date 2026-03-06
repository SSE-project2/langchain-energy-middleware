# langchain-energy-middleware

## Installation & running
Note this currently needs a beefy GPU. In the future should make an option to use external models.

- Configure & activate a venv
- `pip install -r requirements`
- Download ollama: https://ollama.com/download
- `ollama pull qwen3.5:9b`
- In a separate terminal: `ollama serve` (might already be running)
- `ollama ps` to see if model is running, otherwise start with `ollama start qwen3.5`
- `python3 main.py`