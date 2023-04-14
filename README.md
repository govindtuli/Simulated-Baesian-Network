# Simulated-Baesian-Network
Simulating a Baesian Network to improve upon University of Regina's Paper "A Web-based Intelligent Tutoring System for Computer Programming"C.J. Butz, S. Hua, R.B. Maguire


This is a simulation of a user interacting with a learning management system based on a Bayesian Network. The Bayesian Network models the user's knowledge of various programming concepts. The network structure is defined using the edges list, which represents the dependencies between programming concepts.

The plot_3d_scatter function generates a 3D scatter plot using Plotly, showing the relationship between the weight of a parent node, the probability of a correct answer, and the final probability for a given node. The function takes two arguments: child_node and parent_node. To generate the 3D scatter plot, the function first calls generate_plot_data to create a 2D array of final probabilities for various combinations of weights and probabilities. Then, the function creates the scatter plot using Plotly Express, with the weights on the x-axis, probabilities on the y-axis, and final probabilities on the z-axis. The plot is colored based on the z-axis values.

The generate_plot_data function creates a 2D array of final probabilities for a given child node and parent node, based on various combinations of weights and probabilities. The function iterates through the weights and probabilities, updating the weight and probability_known attributes of the parent and child nodes, respectively. The function then calls update_cpd to update the conditional probability distribution (CPD) of the nodes based on the new values. The final probabilities are stored in a 2D array, which is returned by the function.

The simulate_user_interaction function demonstrates how the user interacts with the system. It first checks if the user can access the content of the given node by calling the can_access_node function. If the user can access the content, the function prints the node and its content, then updates the CPD of the node based on the probability of the user's answer being correct. If the user cannot access the content, the function prints a message stating that the node cannot be accessed due to unsatisfied parent requirements.

The code also includes a print_final_probabilities function, which prints the final probabilities for each node in the network. This can be used to inspect the probabilities after the user has interacted with the system.

The update_cpd, update_child_cpd, and update_parent_cpd functions are used to update the CPDs of the nodes based on the user's interaction with the system. These functions consider the node's weight and the probability of a correct answer to update the probability_known attribute of the node.
