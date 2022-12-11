# MLP
Implementation of a Multi-layer perceptron in python only using the numpy library. 

## Example for solving the XOR-Problem

```python
inputs = np.array([[0.0, 0.0,], [0.0,1.0],[1.0,0.0],[1.0,1.0]])
outputs = np.array([[-1.0],[1.0],[1.0],[-1.0]])
mlp = create_network([1,4,20,1], tanh)
mlp.train(inputs, outputs, learning_rate = 0.1, epochs = 500 )
```

## Example for Auto-Encoder

```python
inputs, outputs = np.eye(8), np.eye(8)
mlp = create_network([8,3,8], sgn)
mlp.train(inputs, outputs, learning_rate = 0.1, epochs = 4000 )
```
