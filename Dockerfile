FROM python:3.8
WORKDIR /home/run

# Copy Notebook files 
COPY ["./fm_testing_tool/requirements.txt", \
      "./fm_testing_tool/data/*", \
      "./fm_testing_tool/*.ipynb", \ 
      "./fm_testing_tool/*.py", \ 
      "./fm_testing_tool/postBuild", \ 
      "./"]
COPY ./fm_testing_tool/set_jupyter_password.py /usr/local/bin/
COPY ./fm_testing_tool/startup.sh /usr/local/bin/

# Build Notebook
# RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip3 install -r requirements.txt
RUN ./postBuild
RUN rm requirements.txt
RUN mkdir -p /root/.jupyter/
RUN touch /root/.jupyter/jupyter_notebook_config.json

# Install Develop Oasislmf 
ARG oasis_ver=feature/692-fm-testing-tool
RUN pip3 install git+git://github.com/OasisLMF/OasisLMF.git@$oasis_ver#egg=oasislmf

# Copy test cases 
COPY ./validation/examples /home/run/examples

# RUN
EXPOSE 8888
ENTRYPOINT startup.sh
