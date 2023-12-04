FROM python:3.11-slim as builder
#ENV PORT 8080
#ENV APP_MODULE app.app:app
#ENV LOG_LEVEL debug
#ENV WEB_CONCURRENCY 2
ENV HOME /
RUN apt-get -y update && apt-get -y install build-essential wget
COPY ./app ./code/app
COPY ./requirements.txt /code/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /code/requirements.txt
RUN python3 -m spacy download en_core_web_trf

FROM builder as python_done
WORKDIR llms
RUN ["wget", "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf?download=true"]
RUN ["mv", "mistral-7b-instruct-v0.1.Q4_K_M.gguf?download=true", "mistral-7b-instruct-v0.1.Q4_K_M.gguf"]

FROM python_done as llm_done
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
EXPOSE 8050
WORKDIR /code/app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0","--port", "8050"]
