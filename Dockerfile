FROM tensorflow/tfx:0.21.4
RUN mkdir modules
COPY modules modules
COPY raw_schema raw_schema