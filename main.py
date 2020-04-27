import pandas as pd
import streamlit as st
from kdtools.conceptnet import ConceptNet, ConceptNetEncoder

tab = st.sidebar.selectbox("Tab", ["hello", "cnet", "cnet_encoder"])

if tab == "hello":
    "# Hello"

    "Choose a tag in the sidebar!!!"

elif tab == "cnet":

    "# ConceptNet"

    cnet = ConceptNet()

    "## Head "
    st.table(cnet().sample(5))

    st.show(cnet.unknown_rels)
    st.show(cnet.SYMMETRIC_RELS)
    st.show(cnet.ASYMMETRIC_RELS)

    possible_languages = cnet.possible_languages()
    possible_pos = cnet.possible_pos()

    st.sidebar.subheader("Relation")

    rel = st.sidebar.selectbox("Label", [None] + cnet.relations())

    st.sidebar.subheader("Source")

    source = st.sidebar.text_input("Text", key="Source Text")
    source = source if source.strip() else None

    source_language = st.sidebar.multiselect(
        "Language", possible_languages, key="Source Language", default=["es"]
    )
    source_language = source_language if source_language else None

    source_pos = st.sidebar.multiselect("Pos", possible_pos, key="Source Pos")
    source_pos = source_pos if source_pos else None

    st.sidebar.subheader("Head")

    head = st.sidebar.text_input("Text", key="Head Text", value="asthma")
    head = head if head.strip() else None

    head_language = st.sidebar.multiselect(
        "Language", possible_languages, key="Head Language", default=["en"]
    )
    head_language = head_language if head_language else None

    head_pos = st.sidebar.multiselect("Pos", possible_pos, key="Head Pos")
    head_pos = head_pos if head_pos else None

    selected = cnet(
        rel=rel,
        source=source,
        head=head,
        source_language=source_language,
        head_language=head_language,
        source_pos=source_pos,
        head_pos=head_pos,
    )
    try:
        st.table(selected.sample(5))
    except ValueError:
        st.table(selected.head())

elif tab == "cnet_encoder":

    "# ConceptNet Encoder"

    encoder = ConceptNetEncoder.from_entities(["asma", "asthma"])

    st.show(encoder._entities)
    st.show(encoder._relations)
    st.show(len(encoder))
    st.show(encoder("asma", "asthma"))
    st.show(encoder("asma", "gato"))
