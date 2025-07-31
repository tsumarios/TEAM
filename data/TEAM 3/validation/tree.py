import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ast
import re
from pathlib import Path
from collections import defaultdict
import textwrap

# This script requires networkx, matplotlib, and pydot to be installed.
# You may also need to install Graphviz on your system.
# pip install networkx matplotlib pydot


def parse_team3_threats(base_dir):
    """
    Parses threat data from a directory of CSV files.
    NOTE: This function assumes the CSV files exist in the specified 'base_dir'.
    """
    round_name_to_index = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
    }
    base = Path(base_dir)
    if not base.exists():
        print(
            f"❌ Error: Base directory '{base_dir}' not found. Please check the path."
        )
        return {}, {}, {}

    manual_renames = {
        0: [
            ("TH04", "Remote spying", "Unauthorized surveillance"),
            (
                "TH09",
                "Retrieval of recycled or discarded media",
                "Unauthorized retrieval of disposed media",
            ),
        ],
        2: [("TP01", "Fire", "Fire-related incidents on site")],
        4: [
            (
                "TH02",
                "Social engineering",
                "Social engineering and human-related attacks",
            ),
            ("TH05", "Eavesdropping", "Eavesdropping on private communications"),
        ],
        6: [
            (
                "TP06",
                "Dust, corrosion, freezing",
                "Physical conditions, such as dust, corrosion, freezing",
            ),
            (
                "TH15",
                "Replay attack, man-in-the-middle attack",
                "Interception and re-use of communications",
            ),
        ],
        7: [
            (
                "TC04",
                "Denial of actions",
                "Service compromise due to denial of actions",
            ),
            (
                "TN06",
                "Pandemic/epidemic phenomenon",
                "Widespread natural health phenomenon",
            ),
        ],
    }
    manual_discards = {7: [("TH14", "Unauthorized use of web-based exploits")]}

    operations_data = defaultdict(
        lambda: {"renames": [], "embraces": [], "discards": []}
    )
    threat_name_map = {}

    try:
        iso_file = base / "ISO27005_threats.csv"
        iso = pd.read_csv(iso_file)[["ID", "Threat"]]
        for tid, tname in iso.itertuples(index=False):
            threat_name_map[tid] = tname
        operations_data[0]["renames"] = manual_renames.get(0, [])
        for tid, _, new_name in manual_renames.get(0, []):
            threat_name_map[tid] = new_name

    except FileNotFoundError:
        print(
            f"⚠️ Warning: Could not find '{iso_file}'. Initial threat list will be empty."
        )

    name_to_id_map = {}
    for folder in sorted(
        base.iterdir(), key=lambda d: round_name_to_index.get(d.name, 0)
    ):
        r = round_name_to_index.get(folder.name)
        if not r:
            continue

        try:
            inp_path = next(folder.glob("input_threats_*.csv"))
            inp = pd.read_csv(inp_path)
            name_to_id_map[r] = dict(zip(inp["Threat"], inp["ID"]))
            for tid, tname in zip(inp["ID"], inp["Threat"]):
                threat_name_map[tid] = tname

            out_path = next(folder.glob("output_threats_*.csv"))
            out = pd.read_csv(out_path)
            output_name2ids = {
                tname: list(group["ID"]) for tname, group in out.groupby("Threat")
            }

            emb_path = next(folder.glob("embraced_threats_*.csv"))
            emb_df = pd.read_csv(emb_path)
            embraces = []
            for _, row in emb_df.iterrows():
                new_label, conductor_name, raw_orch = (
                    row["New Threat"],
                    row["Conductor"],
                    row["Orchestra"],
                )

                try:
                    orch = (
                        ast.literal_eval(raw_orch)
                        if isinstance(raw_orch, str) and raw_orch.startswith("[")
                        else [raw_orch]
                    )
                except (ValueError, SyntaxError):
                    orch = [raw_orch]

                old_cid = name_to_id_map[r].get(conductor_name)
                member_ids = [
                    name_to_id_map[r].get(x) for x in orch if x in name_to_id_map[r]
                ]
                if not old_cid or not member_ids:
                    continue

                new_id = output_name2ids.get(new_label, [old_cid])[0]
                embraces.append((new_id, old_cid, member_ids))
                threat_name_map[new_id] = new_label

            operations_data[r].update(
                {
                    "renames": manual_renames.get(r, []),
                    "embraces": embraces,
                    "discards": manual_discards.get(r, []),
                }
            )
            for tid, _, new_name in manual_renames.get(r, []):
                threat_name_map[tid] = new_name

        except (FileNotFoundError, StopIteration) as e:
            print(f"⚠️ Warning: Could not process files in '{folder.name}'. Error: {e}")

    return operations_data, threat_name_map


# --- DERIVATION GRAPH ---
def calculate_dynamic_sizes(n_nodes, n_edges, rankdir="LR"):
    """
    Calculates dynamic figure, node, and font sizes based on graph complexity.
    """
    # --- Figure Size Calculation ---
    if rankdir == "LR":
        # Scale width more for LR layout
        width = 15 + (n_nodes * 0.4) + (n_edges * 0.1)
        height = 10 + (n_nodes * 0.2) + (n_edges * 0.05)
    else:  # 'TB' layout
        # Scale height more for TB layout
        width = 15 + (n_nodes * 0.2) + (n_edges * 0.1)
        height = 10 + (n_nodes * 0.4) + (n_edges * 0.05)

    # Apply reasonable caps to the size
    figsize = (max(15, min(width, 50)), max(10, min(height, 40)))

    # --- Node and Font Size Calculation ---
    if n_nodes > 40:
        node_size = 4000
        font_size = 18
        label_pos = 0.5
    elif n_nodes > 20:
        node_size = 5000
        font_size = 16
        label_pos = 0.6
    else:
        node_size = 6000
        font_size = 14
        label_pos = 0.6

    return figsize, node_size, font_size, label_pos


def build_backtrace_graph(operations_data):
    """Builds the complete derivation graph with detailed labels."""
    G = nx.DiGraph()

    for r, ops in sorted(operations_data.items()):
        # Add a self-loop for renames with the CORRECT, detailed label
        for tid, _, new_name in ops["renames"]:
            # --- THIS IS THE FIX FOR RENAME LABELS ---
            label = f"R_{r}-rename({tid}, [SPADA])"
            G.add_edge(tid, tid, type="rename", op=label)

        # Add edges for embrace operations with the CORRECT, detailed label
        for new_id, conductor_id, member_ids in ops["embraces"]:
            # --- THIS IS THE FIX FOR EMBRACE LABELS ---
            member_ids = [m for m in member_ids if m != conductor_id]
            label = f"R_{r}-embrace({conductor_id}, [{', '.join(member_ids)}], [SPADA])"

            # Conductor has a solid edge with the full operation as a label
            G.add_edge(conductor_id, new_id, type="conductor", op=label)

            # Orchestra members have dashed edges with no label
            for member_id in member_ids:
                if member_id != conductor_id:
                    G.add_edge(member_id, new_id, type="orchestra")
    return G


def trace_back(final_id, G):
    """Traces all ancestors of a final threat to create a subgraph."""
    if final_id not in G:
        return nx.DiGraph()
    nodes_to_keep = set(nx.ancestors(G, final_id))
    nodes_to_keep.add(final_id)
    return G.subgraph(nodes_to_keep).copy()


def plot_derivation_tree(G_sub, final_threat_id, output_path):
    """Visualizes the derivation subgraph with improved edge label readability."""
    if not G_sub.nodes:
        print(f"Cannot plot empty graph for {final_threat_id}.")
        return

    figsize, node_size, font_size, label_pos = calculate_dynamic_sizes(
        G_sub.number_of_nodes(),
        G_sub.number_of_edges(),
        rankdir="LR",  # Keep the left-to-right layout
    )

    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=figsize)
    G_sub.graph["graph"] = {"rankdir": "LR"}

    pos = nx.nx_pydot.graphviz_layout(G_sub, prog="dot")

    # Node labels now include the threat name, wrapped for readability
    node_labels = {n: f"{n}" for n in G_sub.nodes()}

    # Separate edges by type for custom styling
    conductor_edges = [
        (u, v) for u, v, d in G_sub.edges(data=True) if d.get("type") == "conductor"
    ]
    orchestra_edges = [
        (u, v) for u, v, d in G_sub.edges(data=True) if d.get("type") == "orchestra"
    ]
    rename_loops = [
        (u, v) for u, v, d in G_sub.edges(data=True) if d.get("type") == "rename"
    ]

    # Get the detailed labels generated in the build function
    edge_labels = {
        (u, v): d.get("op", "")
        for u, v, d in G_sub.edges(data=True)
        if d.get("type") in ["conductor", "rename"]
    }

    # Draw nodes and their labels
    nx.draw_networkx_nodes(
        G_sub, pos, node_color="skyblue", node_shape="o", node_size=node_size, alpha=0.9
    )
    nx.draw_networkx_labels(
        G_sub, pos, labels=node_labels, font_size=font_size, font_weight="bold"
    )

    # Draw the different types of edges
    nx.draw_networkx_edges(
        G_sub,
        pos,
        edgelist=conductor_edges,
        edge_color="black",
        width=1.5,
        arrows=True,
        arrowsize=20,
    )
    nx.draw_networkx_edges(
        G_sub,
        pos,
        edgelist=orchestra_edges,
        edge_color="gray",
        width=1.0,
        style="dashed",
        arrows=True,
        arrowsize=20,
    )
    nx.draw_networkx_edges(
        G_sub,
        pos,
        edgelist=rename_loops,
        edge_color="lightblue",
        width=1.5,
        arrows=False,
        connectionstyle="arc3,rad=0.5",
    )

    # Draw edge labels with improved positioning and background
    nx.draw_networkx_edge_labels(
        G_sub,
        pos,
        edge_labels=edge_labels,
        label_pos=label_pos,
        font_size=14,
        font_color="black",
        bbox=dict(
            facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round, pad=0.3"
        ),
        rotate=False,
    )

    plt.title(
        f"Derivation tree for threat: {final_threat_id}", fontsize=24, fontweight="bold"
    )
    plt.box(False)
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"✅ Saved derivation graph to {output_path}")


# --- MAIN ---
if __name__ == "__main__":
    base_dir = "."

    operations_data, threat_name_map = parse_team3_threats(base_dir)

    if not operations_data:
        print("\n❌ Critical Error: No operations data loaded. Check 'base_dir' path.")
    else:
        G_full = build_backtrace_graph(operations_data)
        final_threats = ["t_30", "t_35", "t_49", "t_52", "t_64", "t_68", "t_70"]

        print("\n--- Generating Derivation Graphs ---")
        for threat_id in final_threats:
            if threat_id not in G_full:
                print(
                    f"❌ Error: Final threat ID '{threat_id}' not found in the graph. Skipping."
                )
                continue

            G_sub = trace_back(threat_id, G_full)
            output_pdf = f"{threat_id}_derivation_tree.pdf"
            plot_derivation_tree(G_sub, threat_id, output_pdf)
