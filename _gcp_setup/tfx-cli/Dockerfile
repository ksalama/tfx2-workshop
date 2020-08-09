FROM gcr.io/deeplearning-platform-release/base-cpu:m42
RUN apt-get update -y && apt-get -y install kubectl
RUN curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/latest/skaffold-linux-amd64 \
&& chmod +x skaffold \
&& mv skaffold /usr/local/bin
COPY requirements.txt .
RUN python -m pip install -U -r requirements.txt

ENTRYPOINT ["tfx"]