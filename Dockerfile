FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install -r requirements.txt

COPY . .

ENTRYPOINT ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8080", "--config", "settings.py" , "app:app", "-n"]