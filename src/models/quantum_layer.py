"""
Quantum Layer for the Hybrid Model.
This defines a small quantum circuit that acts as a neural network layer.
"""

import pennylane as qml
import torch

# Number of qubits in our quantum circuit (small = fast)
N_QUBITS = 4

# Create a quantum "device" — this is the simulator (runs on your computer, FREE)
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    """
    The actual quantum circuit.
    - inputs: data coming from the classical part (4 numbers)
    - weights: the learnable parameters (trained like normal weights)
    """
    # Step 1: Encode classical numbers into quantum states (angle encoding)
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))

    # Step 2: Apply trainable quantum operations (the "learning" part)
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))

    # Step 3: Measure each qubit and return the results
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


def build_quantum_layer(n_layers: int = 2):
    """
    Wraps the quantum circuit into a PyTorch layer so it plugs into our model.
    """
    weight_shapes = {"weights": (n_layers, N_QUBITS)}
    quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    return quantum_layer


# Quick test: run this file directly to check it works
if __name__ == "__main__":
    layer = build_quantum_layer()
    sample_input = torch.tensor([0.1, 0.2, 0.3, 0.4])
    output = layer(sample_input)
    print("✅ Quantum layer works!")
    print("Input :", sample_input.tolist())
    print("Output:", output.tolist())