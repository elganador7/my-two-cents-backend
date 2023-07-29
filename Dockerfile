ARG APP_HOME=/app

FROM python:3.9-slim-buster as build
RUN apt-get clean && apt-get update --allow-unauthenticated
RUN apt-get -y install default-libmysqlclient-dev build-essential

ARG APP_HOME
WORKDIR $APP_HOME

ARG DEPLOY_ENVIRONMENT
ENV ENVIRONMENT=${DEPLOY_ENVIRONMENT}

ARG POETRY_LIBRARY_VERSION \
POETRY_BUILD_VERSION

ARG POETRY_HTTP_BASIC_GECKOROBOTICS_PYPI_USERNAME \
POETRY_HTTP_BASIC_GECKOROBOTICS_PYPI_PASSWORD

RUN pip install poetry==$POETRY_LIBRARY_VERSION

COPY pyproject.toml poetry.lock ./

RUN poetry version $POETRY_BUILD_VERSION && \
poetry config virtualenvs.create false && \
poetry install -v --no-interaction --no-ansi

ARG APP_HOME
COPY prediction_engine $APP_HOME/prediction_engine
COPY ./tests $APP_HOME/tests

ENTRYPOINT ["uvicorn", "prediction_engine.web:app", "--host", "0.0.0.0", "--port", "8080"]

FROM build as test

ENTRYPOINT pytest tests


# The format stage runs formatting tools on the
# mounted volume containing source code
FROM build as format

ARG APP_HOME
WORKDIR $APP_HOME

RUN pip install flake8 && \
  pip install black && \
  pip install isort

ENTRYPOINT isort prediction_engine; black prediction_engine; flake8 prediction_engine