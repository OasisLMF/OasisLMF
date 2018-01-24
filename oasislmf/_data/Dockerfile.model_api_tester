FROM coreoasis/oasislmf-cli:%CLI_VERSION%

RUN mkdir /home/oasisapi_client/%MODEL_ID%
RUN mkdir /home/oasisapi_client/%MODEL_ID%/data
COPY %LOCAL_MODEL_DATA_PATH%/ /home/oasisapi_client/%MODEL_ID%/data/

CMD ["test", "test-model", "-a", "%MODEL_ID%/data/analysis_settings.json", "-i", "%MODEL_ID%/data/input", "-o", "%{MODEL_ID}%/data/output", "%OASIS_API_SERVER_URL%"]
