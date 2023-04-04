# base image for nltk and flask app
FROM python:3.7.3

# set working directory

WORKDIR /app
COPY . /app
EXPOSE 8000
COPY . .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('stopwords')"
RUN python -c "import nltk; nltk.download('punkt')"
RUN python -c "import nltk; nltk.download('omw-1.4')"
RUN python -c "import nltk; nltk.download('wordnet')"
RUN cd /app
# run using flask dev server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "30", "app:app"]


