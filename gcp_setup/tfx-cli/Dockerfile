FROM gcr.io/deeplearning-platform-release/base-cpu:m42
COPY requirements.txt .
RUN python -m pip install -U -r requirements.txt

ENTRYPOINT ["tfx"]