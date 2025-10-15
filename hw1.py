# #Задание 2
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class ConvNetPadding0(nn.Module):
#     def __init__(self):
#         super(ConvNetPadding0, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # padding=1
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)  # padding=0
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # padding=1
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         return x
#
# class ConvNetPadding1(nn.Module):
#     def __init__(self):
#         super(ConvNetPadding1, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # padding=1
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # padding=1
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # padding=1
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         return x
#
# # Создаем входные данные (64x64)
# input_image = torch.randn(1, 1, 64, 64)  # 1 изображение, 1 канал, размер 64x64
#
# model_padding0 = ConvNetPadding0()
# model_padding1 = ConvNetPadding1()
#
# output_padding0 = model_padding0(input_image)
# output_padding1 = model_padding1(input_image)
#
# print("Размер после слоя с padding=0:", output_padding0.shape)
# print("Размер после слоя с padding=1:", output_padding1.shape)
#
# mean_activation_padding0 = output_padding0.mean().item()
# mean_activation_padding1 = output_padding1.mean().item()
#
# print("Среднее значение активаций для padding=0:", mean_activation_padding0)
# print("Среднее значение активаций для padding=1:", mean_activation_padding1)

# Задание 5
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class DeepConvNet(nn.Module):
    def __init__(self):
        super(DeepConvNet, self).__init__()

        # 6 слоев с чередующимися Conv2d и ReLU
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        # Инициализация весов для каждого слоя
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv4.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv5.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv6.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        print("После conv1:", x.shape)  # Печатаем размерность после каждого слоя
        x = F.relu(self.conv2(x))
        print("После conv2:", x.shape)
        x = F.relu(self.conv3(x))
        print("После conv3:", x.shape)
        x = F.relu(self.conv4(x))
        print("После conv4:", x.shape)
        x = F.relu(self.conv5(x))
        print("После conv5:", x.shape)
        x = self.conv6(x)
        print("После conv6:", x.shape)
        return x


# Создаем модель
model = DeepConvNet()

# Создаем случайный вход (например, 3 канала, 64x64)
input_image = torch.randn(1, 3, 64, 64, requires_grad=True)

# Применяем модель
output = model(input_image)

# Печатаем размер выходного тензора
print("Размер выходного тензора:", output.shape)

# Рассчитываем потерю (пока используем sum как пример)
loss = output.sum()
print("Потеря:", loss.item())

# Выполняем обратное распространение (backward)
loss.backward()

# Печатаем градиенты для первого слоя
print("Градиенты для conv1:")
print(model.conv1.weight.grad)

# Визуализируем градиенты для каждого слоя
grads = [param.grad for param in model.parameters()]
layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6']
for i, grad in enumerate(grads):
    plt.figure()
    plt.title(f"Градиенты для {layer_names[i]}")
    plt.hist(grad.detach().numpy().flatten(), bins=100)
    plt.xlabel("Градиент")
    plt.ylabel("Частота")
    plt.show()
