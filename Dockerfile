FROM python:slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1


WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


COPY . .

RUN pip install --no-cache-dir -e .


RUN python pipeline/training_pipeline.py


EXPOSE 8000 8080

CMD ["sh", "-c", "fastapi run api.py & streamlit run appplication.py --server.port 8080"]
