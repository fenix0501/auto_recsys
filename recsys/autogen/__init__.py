"""Automatic feature generators.

v1_schema   V1: schema-driven aggregates with fixed windows
v2_grammar  V2: V1 grammar + dynamics/affinity/segment families, windows searched on validation
v3_formulas V3: typed formula synthesis (seeds, mutations, attribute lifts), subset chosen on validation
v4_graph    V4: V2 + graph families (co-visitation, recent co-visitation, SVD, filtered popularity)
"""
