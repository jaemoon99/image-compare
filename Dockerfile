FROM python:3.12

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 18000

CMD uvicorn --host=0.0.0.0 --port 18000 main:app