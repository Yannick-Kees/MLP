# MLP
Build MLP from scratch only using python.numpy

```python
inputs = np.array([[0.0, 0.0,], [0.0,1.0],[1.0,0.0],[1.0,1.0]])
outputs = np.array([[-1.0],[1.0],[1.0],[-1.0]])

mlp = create_network([1,4,20,1], tanh)
mlp.train(inputs, outputs, learning_rate = 0.1, epochs = 500 )
```
