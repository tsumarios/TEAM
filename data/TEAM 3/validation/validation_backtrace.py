import os
import pandas as pd
import graphviz
from collections import defaultdict


# Configuration
def get_rounds(base_dir):
    order = {
        "ISO": 0,
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "sixth": 6,
        "seventh": 7,
        "eighth": 8,
        "ninth": 9,
        "tenth": 10,
    }
    rounds = [
        name
        for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name)) and not name.startswith(".")
    ]
    rounds.sort(key=lambda name: order.get(name.lower(), 1000))
    return rounds


BASE_DIR = "."
INITIAL_FILE = os.path.join(BASE_DIR, "ISO27005_threats.csv")
ROUNDS = get_rounds(BASE_DIR)


# Utilities
def find_file_with_prefix(path, prefix):
    return next(
        (
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.startswith(prefix) and f.endswith(".csv")
        ),
        None,
    )


def load_csv_safe(path):
    return pd.read_csv(path) if path and os.path.exists(path) else pd.DataFrame()


lineage = defaultdict(list)
rename_count = 1
embrace_count = 1
initial_df = load_csv_safe(INITIAL_FILE)
input_first = load_csv_safe(
    find_file_with_prefix(os.path.join(BASE_DIR, "first"), "input_threats")
)

if not initial_df.empty and not input_first.empty:
    for _, row in initial_df.iterrows():
        iso_id, iso_threat = row["ID"], row["Threat"]
        match = input_first[input_first["ID"] == iso_id]
        if not match.empty:
            new_threat = match.iloc[0]["Threat"]
            if new_threat != iso_threat:
                lineage[new_threat].append(
                    (iso_threat, "rename", "ISO", rename_count, [])
                )
                rename_count += 1

for i in range(1, len(ROUNDS)):
    prev_round, curr_round = ROUNDS[i - 1], ROUNDS[i]
    out_prev = load_csv_safe(
        find_file_with_prefix(os.path.join(BASE_DIR, prev_round), "output_threats")
    )
    in_curr = load_csv_safe(
        find_file_with_prefix(os.path.join(BASE_DIR, curr_round), "input_threats")
    )
    embrace_df = load_csv_safe(
        find_file_with_prefix(os.path.join(BASE_DIR, curr_round), "embraced_threats")
    )

    if not out_prev.empty and not in_curr.empty:
        for _, row in out_prev.iterrows():
            prev_id, prev_threat = row["ID"], row["Threat"]
            match = in_curr[in_curr["ID"] == prev_id]
            if not match.empty:
                new_threat = match.iloc[0]["Threat"]
                if new_threat != prev_threat:
                    lineage[new_threat].append(
                        (prev_threat, "rename", curr_round, rename_count, [])
                    )
                    rename_count += 1

    if not embrace_df.empty:
        for _, row in embrace_df.iterrows():
            child = row["New Threat"]
            conductor = row["Conductor"]
            orchestra_list = [
                o.strip() for o in str(row.get("Orchestra", "")).split("+") if o.strip()
            ]
            lineage[child].append(
                (
                    conductor,
                    "embrace-conductor",
                    curr_round,
                    embrace_count,
                    orchestra_list,
                )
            )
            for o in orchestra_list:
                lineage[child].append(
                    (o, "embrace-orchestra", curr_round, embrace_count, orchestra_list)
                )
            embrace_count += 1


def build_tree(threat, visited, out_edges):
    if threat in visited:
        return
    visited.add(threat)
    entries = lineage.get(threat, [])
    if not entries and threat in set(initial_df["Threat"]):
        return

    for parent, op, rnd, count, orchestra in entries:
        if op == "rename":
            out_edges.append(
                (parent, threat, f'rename("{parent}", SPADA)', "solid", rnd)
            )
            build_tree(parent, visited, out_edges)

    groups = defaultdict(list)
    for parent, op, rnd, count, orch in entries:
        if op.startswith("embrace"):
            groups[count].append((parent, op, orch, rnd))

    for cnt, members in groups.items():
        cond = next((m for m in members if m[1] == "embrace-conductor"), members[0])
        cond_parent, _, orchestra_list, rnd = cond
        list_str = ", ".join(f'"{o}"' for o in orchestra_list)
        label = f'embrace("{cond_parent}",\n[{list_str}], SPADA)'
        out_edges.append((cond_parent, threat, label, "solid", rnd))
        for m in orchestra_list:
            out_edges.append((m, threat, "", "dashed", rnd))
        build_tree(cond_parent, visited, out_edges)
        for m in orchestra_list:
            if m != cond_parent:
                build_tree(m, visited, out_edges)


def compute_node_rounds(all_edges):
    node_rounds = {}
    order = {name: i for i, name in enumerate(["ISO"] + ROUNDS)}
    for parent, child, _, _, rnd in all_edges:
        for node in (parent, child):
            prev = node_rounds.get(node)
            curr = rnd
            if prev is None or order.get(curr, 1000) < order.get(prev, 1000):
                node_rounds[node] = curr
    return node_rounds


def visualize_tree(threat_name, output_file="threat_backtrace_tree"):
    visited, all_edges = set(), []
    build_tree(threat_name, visited, all_edges)
    node_round_map = compute_node_rounds(all_edges)

    dot = graphviz.Digraph(format="pdf")
    dot.attr(rankdir="LR", splines="true")
    dot.attr(
        "node", shape="box", style="rounded,filled", fillcolor="white", fontsize="20"
    )
    dot.attr("edge", fontsize="16")

    round_order = ["ISO"] + ROUNDS
    present = [r for r in round_order if r in node_round_map.values()]

    colors = [
        "#f2f2f2",
        "#e6f7ff",
        "#fff0e6",
        "#f9f0ff",
        "#f6ffed",
        "#fffbe6",
        "#f0f5ff",
        "#ffe6f0",
        "#e8f5e9",
        "#fffde7",
        "#ede7f6",
        "#fbe9e7",
    ]

    for rnd in present:
        bg_color = colors[present.index(rnd) % len(colors)]
        cluster = f"cluster_{rnd}"
        with dot.subgraph(name=cluster) as c:
            c.attr(
                style="filled",
                color=bg_color,
                label=f"<<B>Round: {rnd}</B>>",
                labelloc="t",
                labeljust="l",
                fontsize="24",
            )
            for node, node_rnd in node_round_map.items():
                if node_rnd == rnd:
                    c.node(node)

    for parent, child, label, style, _ in all_edges:
        dot.edge(parent, child, label=label, style=style)

    dot.render(output_file, cleanup=True)
    print(f"✅ Graph written to {output_file}.pdf")


if __name__ == "__main__":
    last_round = ROUNDS[-1]
    threats = load_csv_safe(
        find_file_with_prefix(os.path.join(BASE_DIR, last_round), "output_threats")
    )
    for threat in threats["Threat"].unique():
        print(f"Visualizing threat: {threat}")
        visualize_tree(threat, output_file=f"tree_{threat}")
