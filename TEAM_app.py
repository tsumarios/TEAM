#!/usr/bin/env python3

"""
TEAM Web Application

Launch: streamlit run team_app.py
"""

import streamlit as st
import pandas as pd
from embracing_utils import (
    filter_dataframe_by_threshold,
    find_synset_relations,
    synset_relations,
    Synset,
)


# Initialise session state variables
if "embraced_threats" not in st.session_state:
    st.session_state.embraced_threats = []
if "processed_indices" not in st.session_state:
    st.session_state.processed_indices = set()
if "s1" not in st.session_state:
    st.session_state.s1 = None
if "s2" not in st.session_state:
    st.session_state.s2 = None


# Streamlit configuration
st.set_page_config(
    page_title="TEAM App",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Report a bug": "https://github.com/tsumarios/TEAM/issues",
        "About": """
        **TEAM** (Threat Embracing by Automated Methods) is a tool developed by [Mario Raciti](https://www.linkedin.com/in/marioraciti/) under the supervision of [Prof. Giampaolo Bella](https://www.linkedin.com/in/giampaolo-bella-a905315a).
        The purpose of TEAM is to automate threat embracing aiding the threat elicitation process within the [SPADA methodology for threat modelling](https://github.com/tsumarios/Threat-Modelling-Research/tree/main/SPADA).
        The tool is open-source and can be found on [GitHub](https://github.com/tsumarios/TEAM).
        """,
    },
)

# App title and sidebar setup
st.title("TEAM - Threat Embracing by Automated Methods")
st.sidebar.header("🚀 Setup")

with st.expander("📖 __Expand/Collapse Guide__"):
    st.markdown(
        """
        ### Welcome to TEAM!
        
        This application helps automate threat embracing within the [SPADA methodology for threat modelling](https://github.com/tsumarios/Threat-Modelling-Research/tree/main/SPADA) by leveraging semantic analysis to identify and group related threats. Here’s a quick guide to get started.

        #### 0 📋 Prerequisites

        Before diving into the core of threat embracing, please make sure you have the input files prepared as follows.
        
        #### 0.1 📝 Prepare the Input Threats CSV

        The Input Threats CSV must have the following columns:
        
        __(ID,Threat,Source of documentation)__

        Where:

        - ID: Identifies the threat
        - Threat: Describes the threat
        - Source of documentation: Specifies the document source which the threat was retrieved from.

        Example:

        _(t_19,Manipulation of information,ENISA)_

        #### 0.2 🧮 Obtain the Semantic Similarity Scores CSV

        The Semantic Similarity Scores CSV can be generated by the TEAM `compute_tuple_similarity_score` script. Open you favourite Terminal and type as follows:
            
        ```sh
        python3 compute_tuple_similarity_scores.py --k <k> --in_path ./data/input_threats.csv [--previous_k_scores_path ./data/previous_k_semantic_similarity_scores.csv] --out_path ./data/output_similarity_scores.csv
        ```

        Where:

        - --k <k>: Specifies the tuple cardinality for similarity scoring.
        - --in_path: Path to the input threats CSV file.
        - --previous_k_scores_path: Path to the precomputed Semantic Similarity Scores CSV related to k-1 (only needed if k > 2).
        - --out_path: Path to the output file for the computed similarity scores.

        You can choose to compute similarity scores either using the TEAM `compute_tuple_similarity_score` script or adopting different methods. In the latter case, just make sure the Input Threats CSV and Semantic Similarity Scores CSV follow the same structure as expected by this app.
        
        The Semantic Similarity Scores CSV must have the following columns:
        
        - _(sentence1,sentence2,score)_ for k=2;
        - _(max,mean,min,scores,sentence1,sentence2,...,sentencek)_ for k>2.

        #### 1. 🚀 Setup

        You can provide the input files and set up several options from the left sidebar.

        ##### 1.1 📄 Upload Input Files

        The following input files are required:

        - **Threat List**: Upload a CSV file containing the list of threats.
        - **Semantic Similarity Scores**: Upload a CSV file with pairwise similarity scores for each threat tuple.

        ##### 1.2 ⚙️ Configuring Threat Embracing Settings

        You can fine-tune the following options:

        - **Tuple Cardinality (k)**: Choose the number of threats per tuple for similarity calculation. For k=2, the "score" method will be used; for higher cardinalities, you can choose an aggregation method.
        - **Aggregation Method**: If `k > 2`, select a method (`mean`, `max`, `min`) to aggregate similarity scores within each tuple.
        - **Threshold**: Set the similarity score threshold for filtering tuples. Only tuples meeting or exceeding this threshold will be displayed as embraceable candidates.
        - **Custom Threat ID Prefix**: Choose a custom prefix for threat IDs (e.g., “t”, “x”) for easier tracking. The app generates unique IDs with this prefix automatically.


        #### 2. 🔍 Reviewing Embraceable Candidates

        - Candidates that meet the threshold are listed, allowing you to inspect each tuple. Use the `Select Tuple Index to Embrace` option to choose a specific candidate tuple for embracing.

        ##### 2.1 🕸️ Semantic Relations

        - View detailed __semantic relationships__ between the selected threats, including _part-of and type-of relations, synonyms, and hierarchy terms_ (i.e., hypernyms, hyponyms, etc.) from the WordNet synset.

        #### 3. 🪄 Threat Embracing

        Embrace the candidates that look _suspicious_, following the embracing operation micro-steps as described below.

        ##### 3.1 ✅ Selection

        - **Conductor**: Select one threat in the tuple to act as the "conductor" for embracing similar threats.
        - **Orchestra**: The remaining threats in the tuple form the "orchestra" that the conductor embraces.

        ##### 3.2 ✍️ (Optional) Rename

        - Optionally rename the new, embraced threat, or leave it as the conductor's label.

        ##### 3.3 ❌ (Optional) Discard 

        - Optionally discard the conductor or orchestra members (also partially) from the threat list.

        #### 4. 💾 Exporting Data

        - **Embraced Threats**: Save the embraced threats to a CSV for documentation and further analysis.
        - **Current List of Threats**: Export the updated list of threats to track the changes made during embracing (and re-iterate the process for further refinement, following TEAM 3 for example!).

        ---
        
        ### ℹ️ Additional Information

        - Use the threshold and aggregation settings to fine-tune the sensitivity of threat embracing as you prefer!
        - Documentation in the form of tables will be generated incrementally as you embrace each candidate tuple.

        *Enjoy TEAM!*
        """
    )


# File uploads for threats and semantic similarity scores
uploaded_threats_file = st.sidebar.file_uploader(
    "Upload Input Threats CSV", type="csv", accept_multiple_files=False
)
uploaded_ss_scores_file = st.sidebar.file_uploader(
    "Upload Semantic Similarity Scores CSV", type="csv", accept_multiple_files=False
)

# Load CSV files only once if not already in session_state
if uploaded_threats_file and uploaded_ss_scores_file:
    if "threats_df" not in st.session_state:
        st.session_state.threats_df = pd.read_csv(uploaded_threats_file)
    if "ss_scores_df" not in st.session_state:
        st.session_state.ss_scores_df = pd.read_csv(uploaded_ss_scores_file)
    st.info(
        f"Successfully loaded {len(st.session_state.threats_df)} input threats and {len(st.session_state.ss_scores_df)} semantic similarity tuple scores."
    )
else:
    st.warning(
        "Please upload both the Input Threats and Semantic Similarity Scores CSV files."
    )
    st.stop()

st.header("Input List of Threats", divider=True)
st.dataframe(st.session_state.threats_df)


# Filter candidates based on threshold and aggregation method
def filter_candidates(df, threshold, aggregation_method="mean"):
    try:
        return filter_dataframe_by_threshold(df, aggregation_method, threshold)
    except KeyError as e:
        st.warning(
            f"{str(e)} column not found (perhaps k>2 ?). Please adjust the value of k accordingly."
        )
        st.stop()
        return None


# Function to get semantic relations for selected candidate
def get_semantic_relations():
    # Initialise a list to store relations for each sentence pair
    relations = []

    # Iterate over each consecutive sentence pair
    for i in range(k - 1):  # From the first sentence to the second-to-last
        s1, s2 = st.session_state.sentences[i], st.session_state.sentences[i + 1]

        # Find the relations between the pair of sentences
        is_partof, is_typeof = find_synset_relations(s1, s2)

        # Append relations for the current sentence pair
        relations.append((s1, s2, is_partof, is_typeof))

        # Display the relations for each sentence pair
        st.markdown(f"Comparing: *{s1}* and *{s2}*")
        st.markdown(f"Is part of: *{is_partof}*  \nIs type of: *{is_typeof}*")

    # Subheader for detailed semantic relations
    st.write("__Detailed Semantic Relations__")

    # Create a DataFrame to store the synset information for all sentences
    synset_dfs = []

    for i in range(k):
        sentence = st.session_state.sentences[i]

        # Fetch the synset relations for the sentence
        synset_dict = synset_relations(sentence)

        # Convert the dictionary into a DataFrame and add to the list
        synset_df = pd.DataFrame.from_dict(synset_dict["terms"])
        synset_dfs.append(synset_df)

    # Concatenate all synset DataFrames into a single DataFrame
    focus_df = pd.concat(synset_dfs, ignore_index=True)

    if not focus_df.empty:
        # Convert Synset objects to strings and handle individual and lists of Synsets
        for col in focus_df.columns:
            focus_df[col] = focus_df[col].apply(
                lambda x: (
                    [str(syn) for syn in x]
                    if isinstance(x, list)
                    else (str(x) if isinstance(x, Synset) else x)
                )
            )

        # Limit the number of items displayed
        focus_df["synonyms"] = focus_df["synonyms"].apply(
            lambda x: x[:3] if isinstance(x, list) else x
        )
        focus_df["hypernyms [L1]"] = focus_df["hypernyms [L1]"].apply(
            lambda x: x[:1] if isinstance(x, list) else x
        )
        focus_df["hyponyms [L1]"] = focus_df["hyponyms [L1]"].apply(
            lambda x: x[:1] if isinstance(x, list) else x
        )
        focus_df["meronyms"] = focus_df["meronyms"].apply(
            lambda x: x[:1] if isinstance(x, list) else x
        )
        focus_df["holonyms"] = focus_df["holonyms"].apply(
            lambda x: x[:1] if isinstance(x, list) else x
        )

        # Display the final DataFrame with all the semantic relations
        st.dataframe(focus_df)
    else:
        st.write("No additional semantic terms available.")

    return relations


# Selection of aggregation method based on k
st.sidebar.header("⚙️ Threat Embracing Settings")
k = st.sidebar.slider(
    "Select Cardinality of Tuples (k)",
    min_value=2,
    max_value=10,
    value=2,
    step=1,
)

# Decide aggregation method
if k == 2:
    aggregation_method = "score"
else:
    aggregation_method = st.sidebar.selectbox(
        "Select Aggregation Method for k > 2:", ["mean", "max", "min"]
    )

# Threshold slider for filtering candidate tuples
threshold = st.sidebar.slider(
    "Select Semantic Similarity Threshold",
    min_value=-1.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
)

# Text input for ID prefix
id_prefix = st.sidebar.text_input(
    "Specify the prefix for the new IDs (e.g., 't', 'p')", value="t"
)

# Ensure the prefix is only a single letter and defaults to "t" if invalid input
if len(id_prefix) != 1 or not id_prefix.isalpha():
    st.warning("Please provide a single alphabet letter as a prefix.")
    id_prefix = "t"  # Default to "t" if input is invalid

st.subheader("Embraceable Candidates", divider=True)
# Apply the filtering with selected threshold and aggregation method
embraceable_candidates = filter_candidates(
    st.session_state.ss_scores_df, threshold, aggregation_method
)
st.write(
    f"Candidates above threshold {threshold} with '{aggregation_method}' aggregation: {len(embraceable_candidates)}"
)
st.dataframe(embraceable_candidates)


# UI for Threat Embracing
st.header("Threat Embracing", divider=True)

# User input to select candidate tuple index
try:
    index_to_embrace = st.number_input(
        "Select Tuple Index to Embrace",
        step=1,
        value=embraceable_candidates.index[0],
    )
except IndexError:
    st.error("Please lower the Semantic Similarity Threshold.")
    st.stop()

# Check if the entered index is valid
if index_to_embrace not in embraceable_candidates.index:
    st.error("Invalid index. Please select a valid index from the list.")
    st.stop()

# Extract selected tuple data

tuple_data = embraceable_candidates.loc[
    embraceable_candidates.index == index_to_embrace
].iloc[0]


# Extract sentences based on k
sentences = [tuple_data[f"sentence{i+1}"] for i in range(k)]
st.session_state.sentences = sentences

# Display selected sentences
selected_sentences = ", ".join([f"**{sentence}**" for sentence in sentences])
st.markdown(f"Selected threats: {selected_sentences}")

with st.expander("__Expand/Collapse Semantic Relations__"):
    with st.spinner("Retrieving semantic relations..."):
        get_semantic_relations()

st.subheader("Select the Conductor and its Orchestra", divider=True)

# Select the conductor
conductor = st.radio(
    "Select the _conductor_ threat:",
    options=st.session_state.sentences,
)
# Select the orchestra based on the conductor
orchestra = [
    sentence for sentence in st.session_state.sentences if sentence != conductor
]

# Rename the new threat (optional), defaulting to the conductor label if no input is given
new_threat = st.text_input("(Optional) Rename threat", value=conductor)

# Discard conductor and/or orchestra member(s)
discard_conductor = (
    st.checkbox(f"(Optional) Discard the conductor: {conductor}")
    if new_threat != conductor
    else False
)
discard_orchestra = st.multiselect(
    f"(Optional) Discard _orchestra member(s)_:",
    options=orchestra,
    default=[],
)

if st.button("Embrace Threat"):
    if index_to_embrace in st.session_state.processed_indices:
        st.warning("This candidate has already been processed.")
    else:
        st.session_state.embraced_threats.append(
            (
                new_threat,
                conductor,
                orchestra,
            )
        )

        st.session_state.processed_indices.add(index_to_embrace)
        st.success("Threat embraced successfully!")

        # Add new threat to the current list
        if (new_threat not in st.session_state.threats_df["Threat"].values) or (
            new_threat != conductor and new_threat != orchestra
        ):
            # Extract the last used number for the specified prefix
            last_t_id_value = (
                st.session_state.threats_df["ID"]
                .str.extract(rf"{id_prefix}_(\d+)")
                .astype(float)
                .max()[0]
            )

            # Generate the next incremental ID
            next_t_id_value = (
                f"{id_prefix}_{int(last_t_id_value) + 1}"
                if not pd.isna(last_t_id_value)
                else f"{id_prefix}_1"
            )

            # Create the new row
            new_row = pd.DataFrame(
                {
                    "ID": [next_t_id_value],
                    "Threat": [new_threat],
                    "Source of documentation": ["TEAM"],
                }
            )

            # Append the new row to the threats_df
            st.session_state.threats_df = pd.concat(
                [st.session_state.threats_df, new_row], ignore_index=True
            ).reset_index(drop=True)

        if discard_conductor:
            st.session_state.threats_df = st.session_state.threats_df[
                ~st.session_state.threats_df["Threat"].isin([conductor])
            ]
            st.info(
                f"Occurrences of the discarded threat _'{conductor}'_ removed from current list of threats."
            )

        if discard_orchestra:
            st.session_state.threats_df = st.session_state.threats_df[
                ~st.session_state.threats_df["Threat"].isin(discard_orchestra)
            ]
            st.info(
                f"Occurrences of the discarded threat(s) _'{', '.join(discard_orchestra)}'_ removed from current list of threats."
            )


# Display embraced threats
st.header("Embraced Threats", divider=True)
embraced_df = pd.DataFrame(
    st.session_state.embraced_threats,
    columns=["New Threat", "Conductor", "Orchestra"],
)
st.dataframe(embraced_df)

# Export embraced threats to CSV
st.subheader("Export Embraced Threats")
export_path_embraced = st.text_input(
    "Insert export path", "./output_embraced_threats.csv"
)
if st.button("Export to CSV", key="export_embraced"):
    embraced_df.to_csv(export_path_embraced, index=False)
    st.success(f"Output embraced threats exported to {export_path_embraced}")
    st.write("Download the exported CSV file:", export_path_embraced)

# Display current list of threats
st.header("Current List of Threats", divider=True)
st.dataframe(st.session_state.threats_df)

# Export embraced threats to CSV
st.subheader("Export Current List of Threats")
export_path_current = st.text_input(
    "Insert export path", "./current_list_of_threats.csv"
)
if st.button("Export to CSV", key="export_current"):
    st.session_state.threats_df.to_csv(export_path_current, index=False)
    st.success(f"Current list of threats exported to {export_path_current}")
    st.write("Download the exported CSV file:", export_path_current)
