FROM python:3.10-slim

WORKDIR /

COPY /api ./api
COPY /models ./models

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--port", "8000"]
