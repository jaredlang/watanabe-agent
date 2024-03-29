FROM python:3.9 

EXPOSE 8000

WORKDIR /app

COPY requirements.txt ./requirements.txt 

RUN pip install --no-cache-dir -r requirements.txt 

COPY . . 

CMD [ "uvicorn", "app2:api", "--port=8000", "--host=0.0.0.0"]
