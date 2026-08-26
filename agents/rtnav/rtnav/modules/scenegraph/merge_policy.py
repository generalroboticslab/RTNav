"""The fixed same-label merge rule for scene-graph nodes."""


def same_label_pairs(pairs, nodes):
    return [(a, b) for a, b in pairs if nodes[a].label == nodes[b].label]
