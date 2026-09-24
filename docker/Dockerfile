FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=8 \
    OPENBLAS_NUM_THREADS=8 \
    MKL_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8

WORKDIR /workspace
COPY requirements.txt /tmp/clinicaltensorsepsis-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/clinicaltensorsepsis-requirements.txt \
    && python -m pip check

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
