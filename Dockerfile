FROM tensorflow/tfx:0.22.1
RUN mkdir modules
COPY modules/train.py modules/
COPY modules/transform.py modules/
COPY modules/custom_components.py modules/
COPY modules/helper.py modules/
COPY modules/sql_utils.py modules/
COPY modules/__init__.py modules/
RUN mkdir raw_schema
COPY raw_schema/schema.pbtxt raw_schema/