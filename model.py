from tinygrad import Tensor, nn
import copy

class AgentModel:
    def __init__(self):
        # CNN
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(6401, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 actions: up, down, left, right

    def __call__(self,  x):
        x_spatial, x_hunger = x
        # x: input grid of shape (1, 10, 10)
        # hunger: scalar hunger value
        x = self.conv1(x_spatial).relu()
        x = self.conv2(x).relu()
        # Flatten the output
        x = x.flatten(start_dim=1)
        # Reshape hunger to match batch dimension for concatenation
        batch_size = x.shape[0]
        x_hunger = x_hunger.reshape(batch_size, 1)
        # Concatenate hunger value
        x = Tensor.cat(x, x_hunger, dim=1)
        x = self.fc1(x).relu().dropout(0.5)  # Add dropout for regularization
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = AgentModel()
    print(model)
    # Example input
    x_spatial = Tensor.randn(1, 1, 10, 10)  # Batch size 1, 1 channel, 10x10 grid
    x_hunger = Tensor.randn(1, 1)  # Batch size 1, scalar hunger value
    x = x_spatial, x_hunger
    output = model(x)
    print(output.shape)  # Should be (1, 4) for the action logits
    print(output.numpy())  # Print the output logits

    target_model = copy.deepcopy(model)
    print(target_model)
    print("Target model created as a deep copy of the original model.")