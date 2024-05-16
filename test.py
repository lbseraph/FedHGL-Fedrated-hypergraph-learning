def extract_subgraph(G, V):
    # 创建一个映射，将V中的节点映射到新的编号
    old_to_new = {node: i for i, node in enumerate(V)}
    print(old_to_new)
    newG = []  # 新图的边集
    
    # 遍历原图中的每条边
    for u, v in G:
        # 如果这两个节点都在V中，则添加到新图中
        if u in old_to_new and v in old_to_new:
            newG.append((old_to_new[u], old_to_new[v]))
    
    return newG

# 示例使用
G = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 4), (0, 2), (5,4)]
V = [1, 2, 4, 5]
newG = extract_subgraph(G, V)
print(newG)
