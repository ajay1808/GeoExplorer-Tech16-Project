FROM python:3.12-slim

# Streamlit writes its config and cache under $HOME.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/app

RUN useradd --create-home --uid 10001 app
WORKDIR /app

# Dependencies first, so a source edit does not invalidate the install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY app.py ./
COPY .streamlit ./.streamlit
RUN pip install --no-cache-dir --no-deps -e .

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/healthz')"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
