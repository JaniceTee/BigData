import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# read the dataset
df = pd.read_csv(r"C:\Users\princ\OneDrive\Desktop\BGDA 511\sample_bankmortgage_data.csv")

# Gender vs Equity Plan
sns.countplot(x="sex", hue="save_act", data=df)
plt.title("Gender vs Savings Account")
plt.show()

# Correlation between Region and Mortgage Liability
corr = pd.crosstab(df.region, df.mortgage)
sns.heatmap(corr, annot=True, cmap="PuBuGn")
plt.title("Region and Mortgage Correlation")
plt.show()

# Create the graphs
G_25_35 = nx.Graph()
G_35_45 = nx.Graph()

# Add nodes to the graphs based on age, mortgage, and children
for index, row in df.iterrows():
    age = row["age"]
    mortgage = row["mortgage"]
    children = row["children"]
    if 25 <= age <= 35:
        G_25_35.add_node((age, mortgage, children))
    elif 35 <= age <= 45:
        G_35_45.add_node((age, mortgage, children))

# Add edges to the graphs based on shared attributes
for node1 in G_25_35.nodes():
    for node2 in G_25_35.nodes():
        if node1 != node2:
            if len(set(node1) & set(node2)) == 2:
                G_25_35.add_edge(node1, node2)

for node1 in G_35_45.nodes():
    for node2 in G_35_45.nodes():
        if node1 != node2:
            if len(set(node1) & set(node2)) == 2:
                G_35_45.add_edge(node1, node2)

# Visualize the graphs
# Visualize the first graph
pos = nx.spring_layout(G_25_35)
nx.draw_networkx_nodes(G_25_35, pos, node_size=500)
nx.draw_networkx_edges(G_25_35, pos, edgelist=G_25_35.edges(), width=1)
nx.draw_networkx_labels(G_25_35, pos, font_size=12, font_family="sans-serif")
plt.title("Age (25-35) Graph")
plt.axis("off")
plt.show()

# Visualize the second graph
pos = nx.spring_layout(G_35_45)
nx.draw_networkx_nodes(G_35_45, pos, node_size=500)
nx.draw_networkx_edges(G_35_45, pos, edgelist=G_35_45.edges(), width=1)
nx.draw_networkx_labels(G_35_45, pos, font_size=12, font_family="sans-serif")
plt.title("Age (35-45) Graph")
plt.axis("off")
plt.show()


