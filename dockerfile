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
RUN pip install -U transformers
RUN pip install -U tqdm boto3 requests regex sentencepiece sacremoses

RUN pip install nltk streamlit
RUN python -c "import nltk; nltk.download('stopwords')"
RUN python -c "import nltk; nltk.download('wordnet')"
RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
RUN python -c "import nltk; nltk.download('sentiwordnet')"
RUN python -c "import nltk; nltk.download('omw')"

RUN pip install -U networkx