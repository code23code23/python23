FROM python:3.11-slim

WORKDIR /app

COPY . .

# Install package; swap standard polars for the LTS CPU-compatible build
RUN pip install --no-cache-dir -e . && \
    pip uninstall -y polars && \
    pip install --no-cache-dir polars-lts-cpu

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "qis_engine_ui.main:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
