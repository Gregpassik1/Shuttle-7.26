FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8080

CMD ["streamlit", "run", "shuttle_app_fixed.py", "--server.port=8080", "--server.address=0.0.0.0"]
