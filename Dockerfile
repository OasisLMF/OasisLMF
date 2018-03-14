FROM coreoasis/ktools:R_0_0_0_178

RUN useradd -ms /bin/bash oasisapi-client && \
    apt update && \
    apt upgrade -y && \
    apt install libsqlite3-dev unixodbc unixodbc-dev -y --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /var/log/oasis && \
    chown oasisapi-client /var/log/oasis && \
    touch /var/log/oasis/oasisapi_client.log && \
    chmod -R 744 /var/log/oasis && \
    mkdir /home/oasisapi_client && \
    chown oasisapi-client /home/oasisapi_client

COPY . /tmp/oasislmf/

RUN pip install /tmp/oasislmf/

USER oasisapi-client
WORKDIR /home/oasisapi_client

ENTRYPOINT ["oasislmf-cli"]
