import re
import streamlit as st
from streamlit_observable import observable

from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset, Lemma
import networkx as nx


import os
if not os.path.exists(os.path.expanduser("~/nltk_data/corpora/wordnet")):
    import nltk
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download("wordnet")

def inspect(obj):
    a,b = st.beta_columns(2)
    a.write(obj)
    b.write(dir(obj))

st.set_page_config(layout='wide')

@st.cache()
def all_lemma_names(lang):
    return tuple(wn.all_lemma_names(pos='n', lang=lang))

@st.cache()
def all_lemmas(keyword, lang='eng'):
    if keyword.strip() == "":
        return set()
    r = {x for x in all_lemma_names(lang=lang) if keyword in x}
    if keyword in r:
        r.remove(keyword)
        return [keyword] + list(r)
    else:
        return list(r)


def pick(my_st, lang='eng'):
    with my_st:
        keyword = st.text_input(f"input a {lang} word").strip()
        key_sets = all_lemmas(keyword, lang=lang)
        return st.selectbox(f"pick a {lang} word", key_sets)

def lemma_list(word, lang='eng'):
    if word is None or word.strip() == "":
        return []
    return list(wn.lemmas(word, pos=wn.NOUN, lang=lang))

def synset_data(synset, lang):
    dict = {}
    for prop_name in dir(synset):
        if not prop_name.startswith("_"):
            prop = getattr(synset, prop_name)
            if callable(prop):
                try:
                    dict[prop_name] = prop()
                    dict[f"{prop_name}_{lang}"] = prop(lang=lang)
                except TypeError:
                    pass
            # elif isinstance(prop, s)
    return dict

@st.cache(allow_output_mutation=True)
def get_synset_graph():
    synsets = wn.all_synsets(pos=wn.NOUN)
    G = nx.DiGraph()
    for synset in synsets:
        synset: Synset
        for hyper in synset.hypernyms():
            G.add_edge(hyper, synset)
    return G

@st.cache(allow_output_mutation=True)
def get_synset_tree():
    synsets = wn.all_synsets(pos=wn.NOUN)
    G = nx.DiGraph()
    for synset in synsets:
        synset: Synset
        for hyper, _ in zip(synset.hypernyms(), [0]):
            G.add_edge(hyper, synset)
    return G

def is_element(synset):
    return get_synset_graph().out_degree(synset) > 0

lang = st.selectbox("pick a language", wn.langs(), wn.langs().index('cmn'))

def full_name(synset: Synset):
    return f"{synset.name()}.{','.join(synset.lemma_names(lang=lang))}"

def expand_tree(graph: nx.DiGraph, source: Synset):
    root = {"children": [], "name": full_name(source), "synset_key": source.name()}
    # st.write(root)
    for edge in graph.out_edges(source):
        _, child = edge
        root["children"].append(expand_tree(graph, child))
    return root

def set_reachable_nodes():
    graph = get_synset_graph()
    root = {}
    for node in graph.nodes():
        if node not in root:
            node_reachables, _ = reachable(graph, node)
            root.update(node_reachables)
    nx.set_node_attributes(graph, root, "size")
    
from collections import defaultdict

def reachable(graph: nx.DiGraph, source: Synset):
    result_dict = defaultdict(int)
    self_count = 1
    for _, child in graph.out_edges(source):
        child_dict, child_sum = reachable(graph, child)
        result_dict.update(child_dict)
        self_count += child_sum

    result_dict[source] = self_count
    return result_dict, self_count



graph_limit = st.number_input('graph maxnodes limit', min_value=5, value=200)

eng_st, lang_st = st.beta_columns(2)
eng_word, lang_word = pick(eng_st), pick(lang_st, lang=lang)
lemma = st.selectbox("pick a lemma", lemma_list(eng_word) + lemma_list(lang_word, lang))
# st.write(synset.definition())
# syn = st.selectbox("pick a synset", synsets, index=0, format_func=lambda syn: f"{syn.name()}[{','.join(syn.lemma_names(lang))}]: {syn.definition()}")
synset = lemma.synset()

set_reachable_nodes()

def show_synset(synset):
    st.header(f"{'ELEMENT' if is_element(synset) else 'ITEM'} {full_name(synset)}:{synset.definition()}")
    #st.sidebar.write(synset_data(synset, lang))
    synset_root_paths = synset_data(synset, lang)["hypernym_paths"]
    for path, my_st in zip(synset_root_paths, st.beta_columns(len(synset_root_paths))):
        with my_st:
            st.text('\n'.join(f"{full_name(x)}" for x in path))

    synset_graph = get_synset_graph()
    expected_size = synset_graph.nodes(data=True)[synset]["size"]
    st.header(f'{full_name(synset)} has total {expected_size} (items/elements)')
    if expected_size > graph_limit:
        st.header(f"trim to {graph_limit}.")
    
    bar = st.progress(0)
    sub_synsets = set([synset])
    for (_, sub_item), _ in zip(nx.bfs_edges(synset_graph, synset), range(graph_limit - 1)):
        if sub_item not in sub_synsets:
            sub_synsets.add(sub_item)
            bar.progress(len(sub_synsets)/expected_size)
    synset_subtree = expand_tree(synset_graph.subgraph(sub_synsets), synset)
    item_sum, element_sum = 0,0
    for sub_synset in sub_synsets:
        if is_element(sub_synset):
            element_sum += 1
        else:
            item_sum += 1
    st.header(f"there are {item_sum} items and {element_sum} elements in {full_name(synset)}. ")
    observer_item_tree_view = observable(f"Tree view for {full_name(synset)}", 
        notebook="@ledenel-observable/radial-dendrogram",
        targets=["viewof chart"],
        observe=["selected"],
        redefine={
            'data': synset_subtree
        }
    )
    if st.checkbox(f"enable selected details for {full_name(synset)}", value=True):
        selected = observer_item_tree_view.get("selected")
        selected_synset = wn.synset(selected["synset_key"])
        show_synset(selected_synset)
    # else:
    #     st.header(f"(> preset graph limit {graph_limit}.)")

show_synset(synset)
