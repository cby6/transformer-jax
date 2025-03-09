venv:
		uv venv -p 3.12

tidy:
		ruff format transformer-jax/
		ruff check --select I --fix transformer-jax/

lint:
		mypy --strict transformer-jax/