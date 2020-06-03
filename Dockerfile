FROM sergivalverde/pytorch:1.2.0_cuda10.1_medical
MAINTAINER Sergi Valverde <svalverde@eia.udg.edu>

# --------------------------------------------------
# packages to install. See readme for a list of
# included packages
# --------------------------------------------------

# all packages are included in the base container (see above)

USER docker
RUN mkdir $HOME/data
RUN mkdir $HOME/data/input
RUN mkdir $HOME/data/output
COPY _utils $HOME/src/_utils
COPY config $HOME/src/config/
COPY models $HOME/src/models/
ADD model.py $HOME/src/model.py
ADD __init__.py $HOME/src/__init__.py
ADD brain_extraction.py $HOME/src/brain_extraction.py
ADD parietal $HOME/src/parietal

WORKDIR $HOME/data
