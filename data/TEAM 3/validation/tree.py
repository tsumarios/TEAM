import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ast
from pathlib import Path
from collections import defaultdict


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

    manual_renames = {
        0: [
            (
                "TH04",
                "Remote spying",
                "Unauthorized surveillance",
                ["TH16", "TH17", "TH18"],
            ),
            (
                "TH09",
                "Retrieval of recycled or discarded media",
                "Unauthorized retrieval of disposed media",
                ["TH16", "TH17", "TH18"],
            ),
        ],
        2: [("TP01", "Fire", "Fire-related incidents on site", ["t_20"])],
        4: [
            (
                "TH02",
                "Social engineering",
                "Social engineering and human-related attacks",
                [],
            ),
            ("TH05", "Eavesdropping", "Eavesdropping on private communications", []),
        ],
        6: [
            (
                "TP06",
                "Dust, corrosion, freezing",
                "Physical conditions, such as dust, corrosion, freezing",
                ["t_63"],
            ),
            (
                "TH15",
                "Replay attack, man-in-the-middle attack",
                "Interception and re-use of communications",
                ["TH14"],
            ),
        ],
        7: [
            (
                "TC04",
                "Denial of actions",
                "Service compromise due to denial of actions",
                ["t_52"],
            ),
            (
                "TN06",
                "Pandemic/epidemic phenomenon",
                "Widespread natural health phenomenon",
                ["t_30"],
            ),
        ],
    }
    manual_discards = {7: [("TH14", "Unauthorized use of web-based exploits")]}

    threat_name_map = {}
    operations_data = defaultdict(
        lambda: {"renames": [], "embraces": [], "discards": []}
    )
    tms_data = {}

    # Initial TMS₀ from ISO
    iso_path = base / "ISO27005_threats.csv"
    iso_df = pd.read_csv(iso_path)[["ID", "Threat"]]
    tms_data[0] = sorted(zip(iso_df["ID"], iso_df["Threat"]), key=lambda x: x[0])
    for tid, tname in tms_data[0]:
        threat_name_map[tid] = tname

    # Add rename names from R0
    for tid, _, new_name, _ in manual_renames.get(0, []):
        threat_name_map[tid] = new_name
    operations_data[0]["renames"] = manual_renames.get(0, [])

    # Process rounds
    name_to_id_map = {}
    for folder in sorted(
        base.iterdir(), key=lambda d: round_name_to_index.get(d.name, 0)
    ):
        r = round_name_to_index.get(folder.name)
        if not r:
            continue

        inp_path = next(folder.glob("input_threats_*.csv"))
        inp_df = pd.read_csv(inp_path)
        name_to_id_map[r] = dict(zip(inp_df["Threat"], inp_df["ID"]))
        for tid, tname in zip(inp_df["ID"], inp_df["Threat"]):
            threat_name_map[tid] = tname
        tms_data[r] = list(zip(inp_df["ID"], inp_df["Threat"]))

        out_path = next(folder.glob("output_threats_*.csv"))
        out_df = pd.read_csv(out_path)
        output_name2ids = {
            tname: list(group["ID"]) for tname, group in out_df.groupby("Threat")
        }

        emb_path = next(folder.glob("embraced_threats_*.csv"))
        emb_df = pd.read_csv(emb_path)
        embraces = []
        for _, row in emb_df.iterrows():
            new_label = row["New Threat"]
            conductor_name = row["Conductor"]
            raw_orch = row["Orchestra"]
            try:
                orch = (
                    ast.literal_eval(raw_orch)
                    if isinstance(raw_orch, str) and raw_orch.startswith("[")
                    else [raw_orch]
                )
            except Exception:
                orch = [raw_orch]
            cid = name_to_id_map[r].get(conductor_name)
            member_ids = [
                name_to_id_map[r].get(x) for x in orch if x in name_to_id_map[r]
            ]
            if not cid or not member_ids:
                continue
            new_id = output_name2ids.get(new_label, [cid])[0]
            threat_name_map[new_id] = new_label
            embraces.append((new_id, cid, member_ids))

        operations_data[r]["embraces"] = embraces
        operations_data[r]["renames"] = manual_renames.get(r, [])
        operations_data[r]["discards"] = manual_discards.get(r, [])

        # Post-operations → rename/discard target names
        for tid, _, new_name, _ in manual_renames.get(r, []):
            threat_name_map[tid] = new_name

    return tms_data, operations_data, threat_name_map


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
        # Add a self-loop for renames
        for tid, _, new_name, member_ids in ops["renames"]:
            member_ids = [m for m in member_ids] or []
            label = (
                f"R_{r}-rename({tid}, [{', '.join(member_ids)}], [SPADA])"
                if member_ids
                else f"R_{r}-rename({tid}, [SPADA])"
            )
            G.add_edge(tid, tid, type="rename", op=label)

        # Add edges for embrace operations
        for new_id, conductor_id, member_ids in ops["embraces"]:
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

    _, operations_data, threat_name_map = parse_team3_threats(base_dir)
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
