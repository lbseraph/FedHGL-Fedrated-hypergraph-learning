# Updating the function to handle `split_idx` as a list of lists where `split_idx[i]` represents all nodes of client `i`.

def find_unsave_nodes(neighbors, current_client_idx, split_idx, edge_list):

    current_client_nodes = set(split_idx[current_client_idx])
    unsave_nodes = []

    for neighbor in neighbors:
        is_safe = True
        for edge in edge_list:
            if any(node in current_client_nodes for node in edge) and neighbor in edge:
                # Check if there are no other nodes in the edge from the same client as `neighbor`
                neighbor_client = next(
                    (client_nodes for client_nodes in split_idx if neighbor in client_nodes), None
                )
                if neighbor_client and all(
                    node == neighbor or node not in neighbor_client for node in edge
                ):
                    is_safe = False
                    break

        if not is_safe:
            unsave_nodes.append(neighbor)

    return unsave_nodes

# The function has been adjusted to work with `split_idx` as a list of lists.
# If you need any further testing or examples to illustrate its use, please provide sample data.

a=find_unsave_nodes([2,5,6,8,9,11],2,[[2,5,6,0,4],[13,8,9,11],[1,3,7,10,12]],[[2,3,7],[5,6,1],[8,10,1],[9,6,12],[11,8,3]])
print(a)