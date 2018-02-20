FROM coreoasis/ktools:R_0_0_0_178

RUN useradd -ms /bin/bash oasisapi-client

RUN apt-get update && \
    apt-get install libsqlite3-dev unixodbc unixodbc-dev -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /var/log/oasis
RUN chown oasisapi-client /var/log/oasis
RUN touch /var/log/oasis/oasisapi_client.log
RUN chmod -R 744 /var/log/oasis

RUN mkdir /home/oasisapi_client
RUN chown oasisapi-client /home/oasisapi_client

COPY . /tmp/oasislmf/
RUN pip install /tmp/oasislmf/

USER oasisapi-client
WORKDIR /home/oasisapi_client

ENTRYPOINT ["oasislmf-cli"]
