# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Read Dataset
df = pd.read_csv(r"C:\Users\princ\OneDrive\Desktop\BGDA 511\sample_bankmortgage_data.csv")

# Visualization 1: Gender vs Car
sns.countplot(x="sex", hue="car", data=df)
plt.title("Equity Plan: Gender vs Car")
plt.show()
# Gender vs Pep
sns.countplot(x="sex", hue="pep", data=df)
plt.title("Equity Plan: Gender vs P.E.P")
plt.show()

# Visualization 2: Correlations between Region and Mortgage Liability
corr = pd.crosstab(df.region, df.mortgage)
sns.heatmap(corr, annot=True, cmap="YlGnBu")
plt.title("Correlations between Region and Mortgage Liability")
plt.show()

# Visualization 3: Create Node and Edge Graphs
G_20_30 = nx.Graph()
G_40_50 = nx.Graph()
G_55_65 = nx.Graph()

# Add Nodes to Graphs Based on Age, Mortgage, and Children
for index, row in df.iterrows():
    age = row["age"]
    mortgage = row["mortgage"]
    children = row["children"]
    if 20 <= age <= 30:
        G_20_30.add_node((age, mortgage, children))
    elif 40 <= age <= 50:
        G_40_50.add_node((age, mortgage, children))
    elif 55 <= age <= 65:
        G_55_65.add_node((age, mortgage, children))

# Add Edges to Graphs Based on Shared Attributes
for node1 in G_20_30.nodes():
    for node2 in G_20_30.nodes():
        if node1 != node2:
            if len(set(node1) & set(node2)) == 2:
                # Calculate the strength of the relationship
                strength = 1 / (df[(df["age"] == node1[0]) & (df["mortgage"] == node1[1]) & (df["children"] == node1[2])]["age"].count())
                G_20_30.add_edge(node1, node2, weight=strength)

for node1 in G_40_50.nodes():
    for node2 in G_40_50.nodes():
        if node1 != node2:
            if len(set(node1) & set(node2)) == 2:
                # Calculate the strength of the relationship
                strength = 1 / (df[(df["age"] == node1[0]) & (df["mortgage"] == node1[1]) & (df["children"] == node1[2])]["age"].count())
                G_40_50.add_edge(node1, node2, weight=strength)

for node1 in G_55_65.nodes():
    for node2 in G_55_65.nodes():
        if node1 != node2:
            if len(set(node1) & set(node2)) == 2:
                # Calculate the strength of the relationship
                strength = 1 / (df[(df["age"] == node1[0]) & (df["mortgage"] == node1[1]) & (df["children"] == node1[2])]["age"].count())
                G_55_65.add_edge(node1, node2, weight=strength)

# Visualize Graphs
# Visualize First Graph
pos = nx.spring_layout(G_20_30)
weights = [G_20_30[u][v]['weight'] for u, v in G_20_30.edges()]
nx.draw_networkx_nodes(G_20_30, pos, node_size=500)
nx.draw_networkx_edges(G_20_30, pos, edgelist=G_20_30.edges(), width=weights, edge_color='r')
nx.draw_networkx_labels(G_20_30, pos, font_size=10, font_family="sans-serif")
plt.title("Relationship Between Age Category (20-30), Mortgage & Children Graph")
plt.axis("off")
plt.show()

# Visualize Second Graph
pos = nx.spring_layout(G_40_50)
weights = [G_40_50[u][v]['weight'] for u, v in G_40_50.edges()]
nx.draw_networkx_nodes(G_40_50, pos, node_size=500)
nx.draw_networkx_edges(G_40_50, pos, edgelist=G_40_50.edges(), width=weights, edge_color='r')
nx.draw_networkx_labels(G_40_50, pos, font_size=10, font_family="sans-serif")
plt.title("Relationship Between Age Category (40-50), Mortgage & Children Graph")
plt.axis("off")
plt.show()

# Visualize Third Graph
pos = nx.spring_layout(G_55_65)
weights = [G_55_65[u][v]['weight'] for u, v in G_55_65.edges()]
nx.draw_networkx_nodes(G_55_65, pos, node_size=500)
nx.draw_networkx_edges(G_55_65, pos, edgelist=G_55_65.edges(), width=weights, edge_color='r')
nx.draw_networkx_labels(G_55_65, pos, font_size=10, font_family="sans-serif")
plt.title("Relationship Between Age Category (55-65), Mortgage & Children Graph")
plt.axis("off")
plt.show()
