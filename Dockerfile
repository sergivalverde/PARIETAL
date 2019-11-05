FROM sergivalverde/pytorch:1.2.0_cuda10.1_medical
MAINTAINER Sergi Valverde <svalverde@eia.udg.edu>


# --------------------------------------------------
# packages to install. See readme for a list of
# included packages
# --------------------------------------------------

# all packages are included in the base container (see above)

USER docker
RUN mkdir $HOME/data
ADD base.py $HOME/src/base.py
ADD dataset.py $HOME/src/dataset.py
ADD model.py $HOME/src/model.py
COPY models $HOME/src/models/
ADD model_utils.py $HOME/src/model_utils.py
COPY mri_utils $HOME/src/mri_utils
ADD run_skull.py $HOME/src/run_skull.py

WORKDIR $HOME/data
