FROM python:3.7

WORKDIR /app
RUN pip install "poetry>=0.12"
RUN poetry config virtualenvs.create false

ADD pyproject.toml poetry.lock /app/
RUN poetry install

ADD src/ /app/src/
ADD scripts/ /app/scripts/
ADD notebooks/ /app/notebooks/
# ADD *.py *.yml /app/

ENTRYPOINT []