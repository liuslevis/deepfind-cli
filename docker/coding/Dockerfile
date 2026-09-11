ARG BASE_IMAGE=python@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7
FROM ${BASE_IMAGE}

RUN python -m venv /opt/venv-template \
    && /opt/venv-template/bin/pip install --no-cache-dir \
        matplotlib==3.9.2 \
        numpy==2.1.3 \
        openpyxl==3.1.5 \
        pandas==2.2.3 \
        pypdf==6.9.2 \
        python-docx==1.2.0 \
        scipy==1.14.1 \
        seaborn==0.13.2 \
    && chmod -R a-w /opt/venv-template \
    && mkdir -p /opt/deepfind /workspace \
    && chown -R 65532:65532 /workspace

COPY --chown=0:0 deepfind/coding_runner.py /opt/deepfind/runner.py
COPY --chown=0:0 deepfind/workspace_terminal_bridge.py /opt/deepfind/terminal_bridge.py

USER 65532:65532
WORKDIR /workspace
ENTRYPOINT ["python", "/opt/deepfind/runner.py"]
