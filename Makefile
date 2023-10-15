# Variables
RELEASE_TAG := $(shell git describe --tags)
BUILD_VERSION := $(RELEASE_TAG:v%=%)
APPLICATION := prediction-engine
DOCKER_TAG := ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${GCP_PROJECT_NAME}/${APPLICATION}
CHART_NAME := $$(cat .deploy/cloud-run/chart.yaml | grep name | cut -d: -f2 | xargs)
CONTAINER_NAME := ${APPLICATION}-container
POETRY_LIBRARY_VERSION=1.4.0

# Docker build args
BUILD_VERSION_ARG=--build-arg POETRY_BUILD_VERSION=$(shell echo ${BUILD_VERSION} | sed 's@-@+@') #needed for PEP440
POETRY_LIBRARY_VERSION_ARG=--build-arg POETRY_LIBRARY_VERSION=$(POETRY_LIBRARY_VERSION)
DEPLOY_ENVIRONMENT_ARG=--build-arg DEPLOY_ENVIRONMENT=$(ENVIRONMENT)
BUILD_ARGS=$(BUILD_VERSION_ARG) $(POETRY_LIBRARY_VERSION_ARG) $(DEPLOY_ENVIRONMENT_ARG)


# Sets up docker using gcloud
.PHONY: setup-gcloud
setup-gcloud:
	@gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev

# Docker build, and outputs the image name and image tag to GITHUB_OUTPUT if it exists for CI
.PHONY: build
build:
	@docker build --platform amd64 $(BUILD_ARGS) --target build -t ${DOCKER_TAG}:${RELEASE_TAG} .
	@echo "IMAGE=${DOCKER_TAG}:${RELEASE_TAG}" >> $${GITHUB_OUTPUT:-/dev/null}

# Runs the image using docker-run
.PHONY: run-docker
run-docker:
	@docker run -p 8080:8080 ${DOCKER_TAG}:${RELEASE_TAG}

# builds the test target and runs it
.PHONY: test
test:
	@docker build --target test $(BUILD_ARGS) -t $(DOCKER_TAG):test .
	@docker run --rm \
		--name $(CONTAINER_NAME)-test \
		$(DOCKER_TAG):test

# Pushes the image to the remote repository
.PHONY: push
push: setup-gcloud
	@docker push ${DOCKER_TAG}:${RELEASE_TAG}

# Create rendered yaml from the helm chart to deploy
.PHONY: .deploy/rendered.yaml
.deploy/rendered.yaml:
	@${HELM_COMMAND} > .deploy/rendered.yaml
	@sed -i '' 's@${DOCKER_TAG}:latest@${DOCKER_TAG}:${RELEASE_TAG}@g' .deploy/rendered.yaml 

# outputs the rendered template
.PHONY: helm-template
helm-template: .deploy/rendered.yaml
	cat .deploy/rendered.yaml

# uses the rendered template to deploy to cloud run
.PHONY: cloud-run-deploy 
cloud-run-deploy: setup-gcloud .deploy/rendered.yaml
	@gcloud run services replace .deploy/rendered.yaml --project ${GCP_PROJECT} --region ${GCP_REGION}

# builds the format target and runs it
.PHONY: format
format:
	@docker build --target format $(BUILD_ARGS) -t $(DOCKER_TAG):format .
	@docker run --rm --name $(CONTAINER_NAME)-format -v $(shell pwd):/app $(DOCKER_TAG):format


# runs the format commands locally instead of in the container
.PHONY: format-local
format-local:
	# Run what CI/CD will run in the docker container
	isort prediction_engine tests; black prediction_engine tests; flake8 prediction_engine tests
