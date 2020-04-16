FROM python:3.8

RUN pip install torch torchvision

RUN pip install streamlit

RUN pip install spacy
RUN pip install -U spacy-lookups-data
RUN python -m spacy download en_core_web_lg
RUN python -m spacy download es_core_news_md

RUN curl -fsSL https://starship.rs/install.sh > starship.sh
RUN bash starship.sh --yes
RUN echo 'eval "$(starship init bash)"' >> root/.bashrc
RUN rm starship.sh

RUN pip install -U black
RUN pip install -U mypy
RUN pip install -U pytorch-crf