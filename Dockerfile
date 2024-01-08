# Use the official micromamba image as the base
FROM mambaorg/micromamba:1.4.2

# Activate the conda environment during the build
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Must use pre-existing directory /home/$MAMBA_USER because the default behaviour of WORKDIR
# is to create a directory as root instead of USER
WORKDIR /home/$MAMBA_USER

# Copy the file containing the dependencies and create the micromamba environment from it
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml environment.yml
COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt requirements.txt
RUN micromamba install --yes --name base --file environment.yml && \
    micromamba clean --all --yes

# Add executables from conda environment to PATH
ENV PATH=/opt/conda/bin/:$PATH

# Install AWS CLI to sync between PVC and S3 bucket.
RUN python -m pip --no-cache-dir install awscli

# Copy the application files to the container's root directory
COPY --chown=$MAMBA_USER:$MAMBA_USER . .

# Install the application in editable mode using pip
RUN /opt/conda/bin/pip install -e .
