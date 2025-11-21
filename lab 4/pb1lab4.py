from pgmpy.models import MarkovNetwork
import matplotlib.pyplot as plt
import networkx as nx

# definim Markov Random Field-ul
mn = MarkovNetwork()

# noduri
mn.add_nodes_from(['A1', 'A2', 'A3', 'A4', 'A5'])

# muchii conform enuntului
mn.add_edges_from([
    ('A1', 'A2'),
    ('A1', 'A3'),
    ('A2', 'A4'),
    ('A2', 'A5'),
    ('A3', 'A4'),
    ('A4', 'A5'),
])

# vizualizare (optional, într-un notebook/script)
nx.draw(mn, with_labels=True)
plt.show()

# cliques (clicile maxime)
print(mn.get_cliques())
